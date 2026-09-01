"""HI-VIS — run the YOLO vs LLM/VLM comparison live, in the browser.

Reuses scripts/compare_models.run_comparison_steps() — the same generator
main() (the CLI) drives — so results stream into this page one (image,
model) pair at a time instead of only appearing after a full CLI/notebook
run finishes. detections.csv/presence.csv are checkpointed to runs/llm/
every 10 pairs, same files main() writes, so a browser refresh never loses
more than 10 pairs of progress.

Pausing: Streamlit can't let a button interrupt a running script and then
carry on in the same session — clicking ANY widget (the "Pause run" button
below, or Streamlit's own native Stop control) aborts the whole script the
same way, with no hook left to run cleanup code after. So a paused run is
really a *stopped* run: whatever's already checkpointed to disk stays
there, un-suffixed (no `_COMPLETE`/manifest yet). What makes it a real
pause instead of losing everything is the PAUSED RUNS section below —
picking one up calls run_comparison_steps() again with `skip_pairs` set to
every (file, model) pair the checkpoint already covers, so resuming never
re-spends an API call on work that's already done, and only continues the
run's original run_dir instead of starting a new one.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
from compare_models import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_YOLO_WEIGHTS,
    LLM_RUNS_ROOT,
    build_adapter,
    build_run_name,
    image_path_for,
    run_comparison_steps,
    sample_test_images,
)
from gemini_prompt_comparison import DESCRIPTIVE_PROMPT  # noqa: E402
from model_adapters import ADAPTERS, DEFAULT_PROMPT_TEMPLATE  # noqa: E402

import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("LIVE MODEL COMPARISON", "YOLO vs LLM/VLM — runs in this tab"), unsafe_allow_html=True)

MUTED = "#71736D"
GREEN = "#1B7A3D"   # matches this app's existing compliant=green convention
RED = "#B02A20"     # matches this app's existing non-compliant=red convention

st.caption(
    "Same sampling + adapters as `scripts/compare_models.py`, run live in this tab — results "
    "stream in below as each image finishes. For an unattended overnight batch that survives "
    "closing this tab, use the CLI instead."
)


# ---------------------------------------------------------------------------
# run history — finished runs, and paused ones that can be resumed
# ---------------------------------------------------------------------------

def list_past_runs():
    """Every *finished* run on disk under runs/llm/ — newest first. Not
    cached: it's a handful of small JSON reads, and this page itself adds a
    new run every time it's used, so a stale cache would hide the run you
    just made."""
    rows = []
    if not LLM_RUNS_ROOT.exists():
        return rows
    for d in sorted(LLM_RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        manifest_path = d / "run_manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            rows.append({
                "run": d.name,
                "created_at": m.get("created_at", "")[:19].replace("T", " "),
                "status": m.get("status", "—"),
                "images": m.get("n_images_sampled"),
                "models": ", ".join(m.get("models", [])),
                "parse failures": sum(m.get("parse_failures", {}).values()),
            })
        elif (d / "prompt_comparison.csv").exists():
            df = pd.read_csv(d / "prompt_comparison.csv")
            rows.append({
                "run": d.name, "created_at": "", "status": "qualitative (unscored)",
                "images": df["file"].nunique(), "models": "gemini/claude (descriptive)", "parse failures": 0,
            })
    return rows


def list_paused_runs():
    """A run this page started (run_config.json exists) but that never
    finished (no run_manifest.json) — i.e. paused, or the tab got closed."""
    rows = []
    if not LLM_RUNS_ROOT.exists():
        return rows
    for d in sorted(LLM_RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        config_path = d / "run_config.json"
        if not (d.is_dir() and config_path.exists() and not (d / "run_manifest.json").exists()):
            continue
        cfg = json.loads(config_path.read_text())
        if cfg.get("kind", "presence") != "presence":
            continue  # a checklist_compare.py run paused mid-way — that page lists its own
        done_pairs = 0
        presence_path = d / "presence.csv"
        if presence_path.exists():
            done_pairs = len(pd.read_csv(presence_path)[["file", "model"]].drop_duplicates())
        rows.append({
            "run_dir": d, "run_name": d.name, "models": cfg["model_names"],
            "n_images": len(cfg["sampled_files"]), "done_pairs": done_pairs,
            "total_pairs": len(cfg["sampled_files"]) * len(cfg["model_names"]),
        })
    return rows


past_runs = list_past_runs()
with st.expander(f"PREVIOUS RUNS ({len(past_runs)})", expanded=False):
    if not past_runs:
        st.caption("No finished runs yet.")
    else:
        st.dataframe(pd.DataFrame(past_runs), hide_index=True, width="stretch")
        st.caption("Full scored breakdown of any of these: the **LLM vs YOLO** page.")

paused_runs = list_paused_runs()
resume_clicked = None
if paused_runs:
    st.markdown(f'<div class="hv-h1" style="font-size:15px;margin:14px 0 6px">PAUSED RUNS ({len(paused_runs)})</div>',
                unsafe_allow_html=True)
    for p in paused_runs:
        rc1, rc2 = st.columns([4, 1])
        rc1.markdown(
            f'<span class="hv-mono" style="font-size:12px">{p["run_name"]}</span> — '
            f'{p["done_pairs"]}/{p["total_pairs"]} pairs done ({", ".join(p["models"])})',
            unsafe_allow_html=True,
        )
        if rc2.button("Resume", key=f"resume_{p['run_name']}"):
            resume_clicked = p
    st.divider()


# ---------------------------------------------------------------------------
# shared rendering — used both for the live feed and for replaying a
# resumed run's already-checkpointed results before it continues
# ---------------------------------------------------------------------------

def chip(label, color):
    return (
        f'<span class="hv-mono" style="display:inline-block;font-size:10px;padding:2px 7px;'
        f'margin:2px 3px 2px 0;background:{color};color:#FFFFFF;white-space:nowrap">{label}</span>'
    )


def render_card(feed, model_names, file_stem, buf):
    """One row per finished image: thumbnail + every model's presence chips
    (green = positive class detected, red = no-* class detected) and, if
    collected, its free-text descriptive response — rendered once as soon as
    every model in this run has finished that image, not once per model."""
    with feed:
        cols = st.columns([1, 3])
        img_path = image_path_for(file_stem)
        if img_path:
            cols[0].image(str(img_path), width="stretch")
        with cols[1]:
            st.markdown(f'<div class="hv-mono" style="font-size:11px;color:{MUTED}">{file_stem}</div>',
                        unsafe_allow_html=True)
            for name in model_names:
                mb = buf.get(name)
                if mb is None:
                    continue
                if mb["parse_error"]:
                    chips = f'<span style="color:{MUTED};font-size:11px">parse error</span>'
                else:
                    chips = "".join(chip(c, GREEN) for c in sorted(mb["positive"]))
                    chips += "".join(chip(c, RED) for c in sorted(mb["negative"]))
                    chips = chips or f'<span style="color:{MUTED};font-size:11px">nothing detected</span>'
                st.markdown(f'<div style="margin-bottom:4px"><b>{name}</b> {chips}</div>', unsafe_allow_html=True)
                if mb["descriptive"]:
                    st.caption(mb["descriptive"])
        st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)


def buf_from_rows(model_names, presence_df, descriptive_df, file_stem):
    """Rebuild render_card()'s per-model buf dict from already-checkpointed
    rows — used to replay a resumed run's history before it continues."""
    buf = {}
    sub = presence_df[presence_df["file"] == file_stem]
    for name in model_names:
        model_rows = sub[sub["model"] == name]
        if model_rows.empty:
            continue
        present = set(model_rows.loc[model_rows["present"] == True, "class_name"])  # noqa: E712
        descriptive = None
        if descriptive_df is not None:
            hit = descriptive_df[(descriptive_df["file"] == file_stem) & (descriptive_df["model"] == name)]
            if not hit.empty:
                descriptive = hit.iloc[0]["response_text"]
        buf[name] = {
            "positive": {c for c in present if not c.startswith("no-")},
            "negative": {c for c in present if c.startswith("no-")},
            "parse_error": bool(model_rows["parse_error"].iloc[0]),
            "descriptive": descriptive,
        }
    return buf


# ---------------------------------------------------------------------------
# the live loop itself — shared by a fresh run and a resumed one
# ---------------------------------------------------------------------------

def run_live(run_dir, run_name, model_names, args, sampled_files, adapters,
             presence_rows, detection_rows, descriptive_rows, parse_failures, skip_pairs):
    detections_path = run_dir / "detections.csv"
    presence_path = run_dir / "presence.csv"
    descriptive_path = run_dir / "descriptive_responses.csv"

    progress_bar = st.progress(0.0)
    status_line = st.empty()
    st.button(
        "⏸ Pause run", key="live_pause_btn",
        help="Stops now — the checkpoint already on disk is safe. Pick it back up from PAUSED RUNS above.",
    )
    st.markdown('<div class="hv-h1" style="font-size:15px;margin:16px 0 6px">RESULTS (live)</div>',
                unsafe_allow_html=True)
    feed = st.container()

    # Replay whatever this run already has on disk (a resume) before the
    # live loop below picks up with the remaining pairs. A file with every
    # model already done gets its final card now; a file caught mid-way
    # (paused between two of its models) seeds file_buf/current_file
    # instead, so the live loop below merges its remaining models into the
    # same card rather than rendering it twice.
    file_buf = {}
    current_file = None
    if skip_pairs:
        covered = {}
        for f, m in skip_pairs:
            covered.setdefault(f, set()).add(m)
        presence_df = pd.DataFrame(presence_rows)
        descriptive_df = pd.DataFrame(descriptive_rows) if descriptive_rows else None
        for file_stem in sampled_files:
            if file_stem not in covered:
                continue
            buf = buf_from_rows(model_names, presence_df, descriptive_df, file_stem)
            if covered[file_stem] == set(model_names):
                render_card(feed, model_names, file_stem, buf)
            else:
                current_file, file_buf = file_stem, buf

    t_start = time.perf_counter()

    for step in run_comparison_steps(adapters, sampled_files, image_path_for=image_path_for,
                                      descriptive_prompt=DESCRIPTIVE_PROMPT, skip_pairs=skip_pairs):
        if step["skipped"] or step["resumed"]:
            if not step["skipped"]:
                progress_bar.progress(step["done"] / step["total"])
            continue

        if step["file"] != current_file:
            if current_file is not None:
                render_card(feed, model_names, current_file, file_buf)
            current_file, file_buf = step["file"], {}

        presence_rows.extend(step["presence_rows"])
        detection_rows.extend(step["detection_rows"])
        if step["descriptive_row"]:
            descriptive_rows.append(step["descriptive_row"])
        if step["parse_error"]:
            parse_failures[step["model"]] += 1

        present = {r["class_name"] for r in step["presence_rows"] if r["present"]}
        file_buf[step["model"]] = {
            "positive": {c for c in present if not c.startswith("no-")},
            "negative": {c for c in present if c.startswith("no-")},
            "parse_error": step["parse_error"],
            "descriptive": step["descriptive_row"]["response_text"] if step["descriptive_row"] else None,
        }

        done, total = step["done"], step["total"]
        progress_bar.progress(done / total)
        per_model = " · ".join(f"{n}:{c}/{len(sampled_files)}" for n, c in step["done_per_model"].items())
        status_line.markdown(f"**{done}/{total}** pairs · {time.perf_counter() - t_start:.0f}s elapsed — {per_model}")

        if done % 10 == 0 or done == total:  # checkpoint: survive a paused/killed tab, not just a clean finish
            pd.DataFrame(detection_rows).to_csv(detections_path, index=False)
            pd.DataFrame(presence_rows).to_csv(presence_path, index=False)
            if descriptive_rows:
                pd.DataFrame(descriptive_rows).to_csv(descriptive_path, index=False)

    if current_file is not None:
        render_card(feed, model_names, current_file, file_buf)

    manifest = {
        "run_name": f"{run_name}_COMPLETE",
        "status": "COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": DATASET_NAME,
        "n_images_requested": len(sampled_files),
        "n_images_sampled": len(sampled_files),
        "seed": args.seed,
        "models": model_names,
        "queryable_classes": {n: adapters[n].queryable_classes for n in model_names},
        "supports_grounding": {n: adapters[n].supports_grounding for n in model_names},
        "prompt_template": args.prompt_template,
        "config": vars(args),
        "parse_failures": parse_failures,
        "sampled_files": sampled_files,
        "descriptive_prompt": DESCRIPTIVE_PROMPT if descriptive_rows else None,
    }
    final_dir = LLM_RUNS_ROOT / f"{run_name}_COMPLETE"
    run_dir.rename(final_dir)
    pd.DataFrame(detection_rows).to_csv(final_dir / "detections.csv", index=False)
    pd.DataFrame(presence_rows).to_csv(final_dir / "presence.csv", index=False)
    if descriptive_rows:
        pd.DataFrame(descriptive_rows).to_csv(final_dir / "descriptive_responses.csv", index=False)
    (final_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    st.success(f"Done in {time.perf_counter() - t_start:.0f}s — saved to `runs/llm/{final_dir.name}/`. "
               "See it scored on the **LLM vs YOLO** page.")
    if any(parse_failures.values()):
        st.warning(f"Parse failures (excluded from presence scoring): {parse_failures}")


def build_args(model_names):
    return SimpleNamespace(
        yolo_weights=str(DEFAULT_YOLO_WEIGHTS),
        ollama_model=ADAPTERS["ollama"]["default_model"],
        qwen3_vl_model=ADAPTERS["qwen3-vl"]["default_model"],
        gemma4_model=ADAPTERS["gemma4"]["default_model"],
        minicpm_v_model=ADAPTERS["minicpm-v"]["default_model"],
        ollama_url="http://localhost:11434",
        claude_model=ADAPTERS["claude"]["default_model"],
        gemini_model=ADAPTERS["gemini"]["default_model"],
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        seed=None,  # filled in by the caller — kept here only so both run paths share one config shape
    )


def load_adapters(model_names, args):
    adapters = {}
    load_status = st.empty()
    for name in model_names:
        load_status.markdown(f"Loading `{name}`...")
        adapters[name] = build_adapter(name, args)
    load_status.markdown("Loaded: " + ", ".join(f"`{n}`" for n in adapters))
    return adapters


# ---------------------------------------------------------------------------
# resume path
# ---------------------------------------------------------------------------

if resume_clicked is not None:
    run_dir = resume_clicked["run_dir"]
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model_names = cfg["model_names"]
    sampled_files = cfg["sampled_files"]

    args = build_args(model_names)
    args.seed = cfg["seed"]
    adapters = load_adapters(model_names, args)

    presence_rows = pd.read_csv(run_dir / "presence.csv").to_dict("records") if (run_dir / "presence.csv").exists() else []
    detection_rows = pd.read_csv(run_dir / "detections.csv").to_dict("records") if (run_dir / "detections.csv").exists() else []
    descriptive_path = run_dir / "descriptive_responses.csv"
    descriptive_rows = pd.read_csv(descriptive_path).to_dict("records") if descriptive_path.exists() else []

    presence_df = pd.DataFrame(presence_rows)
    skip_pairs = set(zip(presence_df.get("file", []), presence_df.get("model", [])))
    parse_failures = {n: 0 for n in model_names}
    if not presence_df.empty:
        failed = presence_df[presence_df["parse_error"] == True]  # noqa: E712
        for name, group in failed.groupby("model"):
            parse_failures[name] = group["file"].nunique()

    st.write(f"Resuming **{run_dir.name}** — {len(skip_pairs)}/{len(sampled_files) * len(model_names)} pairs already done.")
    run_live(run_dir, run_dir.name, model_names, args, sampled_files, adapters,
              presence_rows, detection_rows, descriptive_rows, parse_failures, skip_pairs)
    st.stop()

# ---------------------------------------------------------------------------
# fresh-run config form
# ---------------------------------------------------------------------------

with st.form("live_run_config"):
    c1, c2 = st.columns(2)
    n_images = c1.slider("Images to sample", 1, 100, 20)
    seed = c2.number_input("Seed", value=42, step=1)
    model_names = st.multiselect(
        "Models",
        list(ADAPTERS),
        default=["yolo", "ollama"],
        help="ollama/qwen3-vl/gemma4/minicpm-v need Ollama running locally; claude/gemini call a paid API.",
    )
    cloud_in_selection = [m for m in model_names if ADAPTERS[m]["is_cloud"]]
    # key= pins this checkbox's identity — without it, the auto-derived key
    # includes the label text above, which embeds cloud_in_selection. Change
    # which models are selected and the label (hence the "identity") changes
    # too, so Streamlit treats it as a brand-new widget and resets it to
    # value=False right when the form is submitted — the exact "I checked
    # it, hit Run, and it unchecked itself" bug this was.
    include_cloud = st.checkbox(
        f"Allow cloud models ({', '.join(cloud_in_selection) or 'claude/gemini'}) — calls a paid API",
        value=False, key="live_include_cloud",  # page-specific: session_state is shared across all pages
    )
    submitted = st.form_submit_button("Run comparison", type="primary")

if not submitted:
    st.info("Pick models above and hit **Run comparison** — results stream in below as each image finishes.")
    st.stop()
if not model_names:
    st.error("Pick at least one model.")
    st.stop()
if cloud_in_selection and not include_cloud:
    st.error(f"{cloud_in_selection} need “Allow cloud models” checked — that's a deliberate cost guard.")
    st.stop()

args = build_args(model_names)
args.seed = seed

sampled_files = sample_test_images(n_images, seed)
st.write(f"Sampled **{len(sampled_files)}** test images.")

adapters = load_adapters(model_names, args)

run_name = build_run_name(n_images, seed, model_names)
run_dir = LLM_RUNS_ROOT / run_name
run_dir.mkdir(parents=True, exist_ok=True)
# Written before the loop starts, not after — this is what a paused run
# needs to be resumable at all (see list_paused_runs() above).
(run_dir / "run_config.json").write_text(json.dumps(
    {"kind": "presence", "model_names": model_names, "sampled_files": sampled_files, "seed": seed, "n_images": n_images},
    indent=2,
))

run_live(run_dir, run_name, model_names, args, sampled_files, adapters,
          [], [], [], {n: 0 for n in model_names}, set())
