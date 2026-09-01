"""HI-VIS — LLM/VLM vs. our trained YOLO detector, presence/absence only
("person agnostic" — no per-person split, no boxes; see the Person
Checklist page for the per-person comparison). One page, two halves:

1. RUN A NEW COMPARISON — same generator scripts/compare_models.py's CLI
   uses (run_comparison_steps()), streamed live into this tab. Checkpoints
   to runs/llm/ every 10 pairs; PAUSED RUNS below lists anything that got
   interrupted (Pause button, or Streamlit's own Stop) so it can be picked
   up again without re-spending API calls on work already done. Finishing
   a run here clears the caches below so it shows up immediately, no
   reload needed.
2. REVIEW RESULTS — two slides. ALL RUNS is built live from every
   `runs/llm/*/run_manifest.json` on disk, plus any `prompt_comparison.csv`
   runs that have no manifest (the qualitative, unscored ones). LATEST RUN
   scores the most recent scored run (by manifest `created_at`) live from
   its `presence.csv` against `data/merged/labels_long.csv` + `data.yaml`,
   falling back to the baked-in numbers from `reports/llm_vs_yolo_comparison.md`
   if the ground truth files aren't present on this checkout (they're
   git-ignored) — same pattern `pages/demo.py` uses for missing weights:
   degrade with an explanation, never fake a result.

Color: sequential black->white for the accuracy heatmap (magnitude, one
hue); categorical black-vs-muted-grey for "ours vs. every VLM" in the bar
chart, with red reserved for negative-class bars specifically (matches
this app's existing non-compliant=red convention); green for a live run's
positive-class chips (matches the compliant=green convention) — never a
rainbow, never color-only identity (every bar/cell/chip is also directly
labeled).
"""

import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
from compare_models import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_YOLO_WEIGHTS,
    MERGED_ROOT,
    LLM_RUNS_ROOT as RUNS_LLM_ROOT,
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
st.markdown(vh.header_html("LLM vs YOLO (PERSON AGNOSTIC)", "presence/absence per class — runs/llm/"),
            unsafe_allow_html=True)

INK = "#141414"
MUTED = "#71736D"
FAINT = "#C4C6C0"
NEGATIVE_RED = "#B02A20"
POSITIVE_GREEN = "#1B7A3D"  # matches this app's existing compliant=green convention
CHART_FONT = "IBM Plex Sans, sans-serif"
alt.themes.enable("none")

MODEL_LABEL = {
    "yolo": "YOLO26 (ours)", "gemini": "Gemini 3.6 Flash", "qwen3-vl": "Qwen3-VL", "gemma4": "Gemma 4",
    "minicpm-v": "MiniCPM-V", "ollama": "LLaVA (Ollama)", "florence2": "Florence-2", "yoloe": "YOLO-E",
}
CLASS_ORDER = ["person", "helmet", "gloves", "boots", "vest", "no-helmet", "no-gloves", "no-boots", "no-vest"]
POSITIVE_CLASSES = {"person", "helmet", "gloves", "boots", "vest"}

# Fallback macro-level metrics — reports/llm_vs_yolo_comparison.md, used only
# if data/merged/labels_long.csv (git-ignored) isn't present on this
# checkout to score the latest run's presence.csv live.
# (tp, fp, fn, tn) per class.
_FALLBACK_RAW = {
    "yolo":      {"person": (49, 0, 0, 51), "helmet": (47, 1, 2, 50), "gloves": (31, 0, 2, 67), "boots": (32, 0, 3, 65), "vest": (29, 2, 3, 66),
                  "no-helmet": (31, 2, 4, 63), "no-gloves": (17, 1, 1, 81), "no-boots": (16, 0, 0, 84), "no-vest": (22, 1, 2, 75)},
    "gemini":    {"person": (48, 51, 1, 0), "helmet": (45, 2, 4, 49), "gloves": (27, 10, 6, 57), "boots": (25, 10, 10, 55), "vest": (28, 0, 4, 68),
                  "no-helmet": (33, 33, 2, 32), "no-gloves": (13, 58, 5, 24), "no-boots": (12, 29, 4, 55), "no-vest": (20, 59, 4, 17)},
}


def _base_config(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(labelFont=CHART_FONT, titleFont=CHART_FONT, labelColor=MUTED, titleColor=INK,
                         grid=False, domainColor=FAINT, tickColor=FAINT, labelFontSize=11.5, titleFontSize=11.5)
        .configure_legend(labelFont=CHART_FONT, titleFont=CHART_FONT, labelColor=INK, titleColor=INK,
                           labelFontSize=11.5, titleFontSize=11.5, orient="top", symbolType="square")
        .configure_text(font=CHART_FONT)
    )


def model_chip(name):
    label = MODEL_LABEL.get(name, name)
    bg = INK if name == "yolo" else "#FFFFFF"
    fg = "#FFFFFF" if name == "yolo" else INK
    border = INK if name == "yolo" else "#C4C6C0"
    return (f'<span class="hv-mono" style="display:inline-block;font-size:10.5px;padding:3px 8px;'
            f'background:{bg};color:{fg};border:1px solid {border};margin:2px 4px 2px 0;white-space:nowrap">{label}</span>')


def stat_tile(label, value, note, bg=INK, fg="#FFFFFF", border=None):
    border_css = f"border:1px solid {border};" if border else ""
    return f"""
    <div style="background:{bg};color:{fg};{border_css}padding:16px 20px 14px">
      <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:{'#9B9D97' if bg == INK else MUTED}">{label}</div>
      <div class="hv-h1" style="font-size:44px;line-height:1;color:{fg}">{value}</div>
      <div style="font-size:12px;color:{'#9B9D97' if bg == INK else MUTED}">{note}</div>
    </div>"""


# ===========================================================================
# SECTION 1 — run a new comparison, live, in this tab
# ===========================================================================

st.markdown('<div class="hv-h1" style="font-size:20px;margin-bottom:2px">① RUN A NEW COMPARISON</div>',
            unsafe_allow_html=True)
st.caption(
    "Same sampling + adapters as `scripts/compare_models.py`, run live in this tab — results stream in "
    "below as each image finishes, and the REVIEW RESULTS section further down picks up the finished run "
    "automatically. For an unattended overnight batch that survives closing this tab, use the CLI instead."
)


def list_paused_runs():
    """A run this page started (run_config.json exists) but that never
    finished (no run_manifest.json) — i.e. paused, or the tab got closed."""
    rows = []
    if not RUNS_LLM_ROOT.exists():
        return rows
    for d in sorted(RUNS_LLM_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
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


paused_runs = list_paused_runs()
resume_clicked = None
if paused_runs:
    st.markdown(f'<div class="hv-h1" style="font-size:15px;margin:14px 0 6px">PAUSED RUNS ({len(paused_runs)})</div>',
                unsafe_allow_html=True)
    for p in paused_runs:
        rc1, rc2 = st.columns([4, 1])
        rc1.markdown(
            f'<span class="hv-mono" style="font-size:12px">{p["run_name"]}</span> — '
            f'<b>{p["done_pairs"]}/{p["total_pairs"]}</b> pairs done ({", ".join(model_chip(m) for m in p["models"])})',
            unsafe_allow_html=True,
        )
        if rc2.button("▶ Resume", key=f"resume_{p['run_name']}"):
            resume_clicked = p
    st.divider()


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
                    chips = "".join(chip(c, POSITIVE_GREEN) for c in sorted(mb["positive"]))
                    chips += "".join(chip(c, NEGATIVE_RED) for c in sorted(mb["negative"]))
                    chips = chips or f'<span style="color:{MUTED};font-size:11px">nothing detected</span>'
                st.markdown(f'<div style="margin-bottom:4px;display:flex;align-items:center;gap:6px">'
                            f'{model_chip(name)}{chips}</div>', unsafe_allow_html=True)
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
    final_dir = RUNS_LLM_ROOT / f"{run_name}_COMPLETE"
    run_dir.rename(final_dir)
    pd.DataFrame(detection_rows).to_csv(final_dir / "detections.csv", index=False)
    pd.DataFrame(presence_rows).to_csv(final_dir / "presence.csv", index=False)
    if descriptive_rows:
        pd.DataFrame(descriptive_rows).to_csv(final_dir / "descriptive_responses.csv", index=False)
    (final_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    st.success(f"Done in {time.perf_counter() - t_start:.0f}s — saved to `runs/llm/{final_dir.name}/`. "
               "REVIEW RESULTS below now includes it.")
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


ran_this_load = False  # a fresh run or a resume just finished — caches below need clearing

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
    ran_this_load = True
else:
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
        # key= pins this checkbox's identity — without it, the auto-derived
        # key includes the label text above, which embeds cloud_in_selection.
        # Change which models are selected and the label (hence the
        # "identity") changes too, so Streamlit treats it as a brand-new
        # widget and resets it to value=False right when the form submits.
        include_cloud = st.checkbox(
            f"Allow cloud models ({', '.join(cloud_in_selection) or 'claude/gemini'}) — calls a paid API",
            value=False, key="live_include_cloud",  # page-specific: session_state is shared across all pages
        )
        submitted = st.form_submit_button("Run comparison", type="primary")

    if not submitted:
        st.info("Pick models above and hit **Run comparison** — results stream in below as each image finishes.")
    elif not model_names:
        st.error("Pick at least one model.")
    elif cloud_in_selection and not include_cloud:
        st.error(f"{cloud_in_selection} need “Allow cloud models” checked — that's a deliberate cost guard.")
    else:
        args = build_args(model_names)
        args.seed = seed

        sampled_files = sample_test_images(n_images, seed)
        st.write(f"Sampled **{len(sampled_files)}** test images.")

        adapters = load_adapters(model_names, args)

        run_name = build_run_name(n_images, seed, model_names)
        run_dir = RUNS_LLM_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        # Written before the loop starts, not after — this is what a paused
        # run needs to be resumable at all (see list_paused_runs() above).
        (run_dir / "run_config.json").write_text(json.dumps(
            {"kind": "presence", "model_names": model_names, "sampled_files": sampled_files,
             "seed": seed, "n_images": n_images},
            indent=2,
        ))
        run_live(run_dir, run_name, model_names, args, sampled_files, adapters,
                  [], [], [], {n: 0 for n in model_names}, set())
        ran_this_load = True

st.markdown("<hr style='margin:26px 0 18px'/>", unsafe_allow_html=True)

# ===========================================================================
# SECTION 2 — review results (every run, then the latest one in depth)
# ===========================================================================

st.markdown('<div class="hv-h1" style="font-size:20px;margin-bottom:10px">② REVIEW RESULTS</div>',
            unsafe_allow_html=True)


@st.cache_data
def load_all_runs():
    scored_runs = []
    for f in sorted(glob.glob(str(RUNS_LLM_ROOT / "*" / "run_manifest.json"))):
        m = json.loads(Path(f).read_text())
        scored_runs.append({
            "run_name": m["run_name"], "kind": "scored", "created_at": pd.Timestamp(m["created_at"]),
            "n_images": m["n_images_sampled"], "models": m["models"],
            "parse_failures": sum(m.get("parse_failures", {}).values()),
        })
    scored_dirs = {Path(f).parent.name for f in glob.glob(str(RUNS_LLM_ROOT / "*" / "run_manifest.json"))}
    qualitative_runs = []
    for f in sorted(glob.glob(str(RUNS_LLM_ROOT / "*" / "prompt_comparison.csv"))):
        run_dir = Path(f).parent
        if run_dir.name in scored_dirs:
            continue
        df = pd.read_csv(f)
        # run_name embeds a UTC "YYYYMMDD_HHMMSS" prefix — parse it back out
        # for a real timestamp rather than falling back to file mtime, which
        # only reflects when this checkout last touched the file.
        ts = pd.to_datetime(run_dir.name[:15], format="%Y%m%d_%H%M%S", utc=True, errors="coerce")
        qualitative_runs.append({
            "run_name": run_dir.name, "kind": "qualitative (prompt-style, unscored)",
            "created_at": ts if pd.notna(ts) else pd.Timestamp(run_dir.stat().st_mtime, unit="s", tz="UTC"),
            "n_images": df["file"].nunique(), "models": ["gemini"], "parse_failures": 0,
        })
    runs = pd.DataFrame(scored_runs + qualitative_runs)
    return runs.sort_values("created_at").reset_index(drop=True)


@st.cache_data
def score_latest_run():
    all_runs = load_all_runs()
    scored_only = all_runs[all_runs["kind"] == "scored"]
    if scored_only.empty:
        return None, None
    latest_run_name = scored_only.iloc[-1]["run_name"]
    run_dir = RUNS_LLM_ROOT / latest_run_name
    presence_path = run_dir / "presence.csv"
    labels_path = MERGED_ROOT / "labels_long.csv"
    data_yaml_path = MERGED_ROOT / "data.yaml"
    if not (presence_path.exists() and labels_path.exists() and data_yaml_path.exists()):
        return None, latest_run_name

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    presence = pd.read_csv(presence_path)
    labels_long = pd.read_csv(labels_path)
    class_names = yaml.safe_load(data_yaml_path.read_text())["names"]
    id_to_name = {i: n for i, n in enumerate(class_names)}
    labels_long = labels_long.copy()
    labels_long["class_name"] = labels_long["class_id"].map(id_to_name)
    sampled_files = manifest["sampled_files"]
    gt = labels_long[labels_long["file"].isin(sampled_files)]
    gt_by_file = gt.groupby("file")["class_name"].apply(set).to_dict()

    rows = []
    for model in manifest["models"]:
        sub = presence[(presence["model"] == model) & (~presence["parse_error"])]
        sub_by_key = {(r.file, r.class_name): bool(r.present) for r in sub.itertuples()}
        for cls in sorted(sub["class_name"].unique()):
            tp = fp = fn = tn = 0
            for f in sampled_files:
                if (f, cls) not in sub_by_key:
                    continue
                pred = sub_by_key[(f, cls)]
                actual = cls in gt_by_file.get(f, set())
                if pred and actual:
                    tp += 1
                elif pred and not actual:
                    fp += 1
                elif not pred and actual:
                    fn += 1
                else:
                    tn += 1
            rows.append({"model": model, "class_name": cls, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows), latest_run_name


if ran_this_load:
    load_all_runs.clear()
    score_latest_run.clear()

all_runs = load_all_runs()


def metrics_from_raw(raw_dict):
    rows = []
    for model, classes in raw_dict.items():
        for cls, (tp, fp, fn, tn) in classes.items():
            rows.append({"model": model, "class_name": cls, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


def add_prf1(df):
    df = df.copy()
    total = df["tp"] + df["tn"] + df["fp"] + df["fn"]
    # Accuracy and recall are our main metrics. Precision/F1 stay as
    # secondary diagnostic columns (e.g. explaining why accuracy is high but
    # recall is low), not the headline numbers.
    # float("nan"), not pd.NA — replace() on an int64 column upcasts it to
    # object dtype for pd.NA (mixed int/NA), and .astype(float) below then
    # calls Python's float() per element, which chokes on pd.NA ("float()
    # argument must be a string or a real number, not 'NAType'"). A real
    # NaN upcasts the column to float64 instead, which astype(float) (and
    # normal division) handles natively.
    df["accuracy"] = (df["tp"] + df["tn"]) / total.replace(0, float("nan"))
    df["recall"] = df["tp"] / (df["tp"] + df["fn"]).replace(0, float("nan"))
    df["precision"] = df["tp"] / (df["tp"] + df["fp"]).replace(0, float("nan"))
    for col in ("accuracy", "recall", "precision"):
        df[col] = df[col].astype(float)
    denom = df["precision"] + df["recall"]
    df["f1"] = (2 * df["precision"] * df["recall"] / denom).where(denom > 0)
    df["model_label"] = df["model"].map(lambda m: MODEL_LABEL.get(m, m))
    df["group"] = df["class_name"].map(lambda c: "Positive (present)" if c in POSITIVE_CLASSES else "Negative (absent)")
    return df


live_raw, latest_run_name = score_latest_run()
using_live_data = live_raw is not None
metrics = add_prf1(live_raw if using_live_data else metrics_from_raw(_FALLBACK_RAW))
if latest_run_name is None:
    latest_run_name = "runs/llm/20260831_merged_n100_seed42_yolo-gemini"

model_order_present = list(dict.fromkeys(metrics.sort_values("model")["model"]))
MODEL_ORDER = sorted(model_order_present, key=lambda m: (m != "yolo", m != "gemini", m))
MODEL_LABEL_ORDER = [MODEL_LABEL.get(m, m) for m in MODEL_ORDER]

# Macro (mean-of-classes) accuracy and recall — our two main metrics — split
# by positive/negative class group, plus an overall (all-9-class) mean of
# each. Precision/F1 aren't tabled at this macro level; they're still on the
# per-class heatmap below for anyone who wants the detail.
def macro_table(metric):
    t = (
        metrics.groupby(["model", "model_label", "group"])[metric].mean().reset_index()
        .pivot(index=["model", "model_label"], columns="group", values=metric).reset_index()
    )
    t["overall"] = metrics.groupby("model")[metric].mean().reindex(t["model"]).values
    return t.set_index("model").loc[MODEL_ORDER].reset_index()


macro = macro_table("accuracy")
macro_recall = macro_table("recall")

# ---------------------------------------------------------------------------
# slide switcher
# ---------------------------------------------------------------------------

st.markdown("""
<style>
div[role="radiogroup"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

slide = st.radio(
    "Slide", ["① ALL RUNS", "② LATEST RUN"], horizontal=True, label_visibility="collapsed", key="llm_slide",
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ===========================================================================
# SLIDE 1 — every run, at a glance
# ===========================================================================

if slide == "① ALL RUNS":
    n_runs = len(all_runs)
    all_models_ever = sorted({m for models in all_runs["models"] for m in models} - {"yolo"})
    total_images_scored = int(all_runs.loc[all_runs["kind"] == "scored", "n_images"].sum())
    date_span = f'{all_runs["created_at"].min():%b %d} – {all_runs["created_at"].max():%b %d}'

    st.markdown(
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:22px">'
        + stat_tile("COMPARISON RUNS", n_runs, f"{date_span}, 2026")
        + stat_tile("VLMS TRIED AGAINST YOLO", len(all_models_ever),
                    ", ".join(MODEL_LABEL.get(m, m) for m in all_models_ever), bg="#FFFFFF", fg=INK, border=FAINT)
        + stat_tile("IMAGES SCORED (SCORED RUNS)", total_images_scored,
                    f'across {(all_runs["kind"] == "scored").sum()} scored runs', bg="#FFFFFF", fg=INK, border=FAINT)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hv-h1" style="font-size:22px;margin-bottom:2px">EVERY RUN, IN ORDER</div>', unsafe_allow_html=True)
    st.caption("Bar length = images sampled. The lineup grew (and dropped experiments) run over run — see the model chips below each bar.")

    timeline_df = all_runs.copy()
    timeline_df["row_label"] = timeline_df.apply(
        lambda r: f'{r["created_at"]:%b %d, %H:%M} UTC — {"scored" if r["kind"] == "scored" else "prompt-style"}', axis=1)
    timeline_df["row_order"] = range(len(timeline_df))

    timeline_df["models_str"] = timeline_df["models"].apply(lambda ms: ", ".join(MODEL_LABEL.get(m, m) for m in ms))
    row_h = 42
    chart_h = row_h * len(timeline_df)

    tl_bar = (
        alt.Chart(timeline_df)
        .mark_bar(size=row_h - 14, cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
        .encode(
            y=alt.Y("row_label:N", title=None, sort=alt.SortField("row_order"),
                    scale=alt.Scale(paddingInner=0.3, paddingOuter=0.2)),
            x=alt.X("n_images:Q", title="images sampled"),
            color=alt.Color("kind:N", title=None,
                             scale=alt.Scale(domain=["scored", "qualitative (prompt-style, unscored)"], range=[INK, FAINT])),
            tooltip=[alt.Tooltip("run_name:N", title="Run"), alt.Tooltip("n_images:Q", title="Images"),
                     alt.Tooltip("models_str:N", title="Models")],
        )
        .properties(height=chart_h)
    )
    tl_text = (
        alt.Chart(timeline_df)
        .mark_text(align="left", dx=6, font=CHART_FONT, fontSize=11, color=INK)
        .encode(y=alt.Y("row_label:N", sort=alt.SortField("row_order"), scale=alt.Scale(paddingInner=0.3, paddingOuter=0.2)),
                x="n_images:Q", text="n_images:Q")
        .properties(height=chart_h)
    )
    st.altair_chart(_base_config(tl_bar + tl_text), width="stretch")

    for _, r in timeline_df.sort_values("row_order", ascending=False).iterrows():
        chips = "".join(model_chip(m) for m in r["models"])
        note = f' · {r["parse_failures"]} parse failures' if r["parse_failures"] else ""
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:8px 12px;margin-bottom:6px;
             display:flex;align-items:center;gap:14px;flex-wrap:wrap">
          <span class="hv-mono" style="font-size:10.5px;color:#71736D;white-space:nowrap">{r["created_at"]:%b %d %H:%M}</span>
          <span style="font-size:11px;color:#4A4B47;white-space:nowrap">{r["n_images"]} img · {r["kind"]}{note}</span>
          <span>{chips}</span>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Florence-2 and YOLO-E (Aug 27) were tried as alternative grounding-capable baselines alongside YOLO, then "
        "dropped from later runs in favor of the settled 5-VLM lineup (ollama/qwen3-vl/gemma4/minicpm-v) plus, "
        "most recently, Gemini. Source: every run_manifest.json + prompt_comparison.csv under runs/llm/."
    )

# ===========================================================================
# SLIDE 2 — the latest run, in depth
# ===========================================================================

else:
    if not using_live_data:
        st.markdown(
            '<div style="background:#FFFFFF;border:1px dashed #9B9D97;padding:10px 16px;margin-bottom:16px;'
            'font-size:12px;color:#71736D">Showing the last known-good numbers from '
            '<code>reports/llm_vs_yolo_comparison.md</code> — <code>data/merged/labels_long.csv</code> isn\'t on '
            'this checkout (it\'s git-ignored) so the latest run couldn\'t be scored live.</div>',
            unsafe_allow_html=True,
        )

    yolo_overall = macro.loc[macro["model"] == "yolo", "overall"].iloc[0] if "yolo" in macro["model"].values else None
    best_vlm_row = macro[macro["model"] != "yolo"].assign(_o=lambda d: d["overall"]).sort_values("_o", ascending=False)
    best_vlm = best_vlm_row.iloc[0] if not best_vlm_row.empty else None

    yolo_recall = macro_recall.loc[macro_recall["model"] == "yolo", "overall"].iloc[0] if "yolo" in macro_recall["model"].values else None
    best_vlm_recall = macro_recall.loc[macro_recall["model"] == best_vlm["model"], "overall"].iloc[0] if best_vlm is not None else None

    tiles = ['<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:22px">']
    if yolo_overall is not None:
        tiles.append(stat_tile("YOLO26 (OURS) — ACCURACY / RECALL", f"{yolo_overall:.2f} / {yolo_recall:.2f}",
                                "macro-averaged across all 9 classes"))
    if best_vlm is not None:
        tiles.append(stat_tile(f'BEST VLM ({best_vlm["model_label"].upper()}) — ACCURACY / RECALL',
                                f'{best_vlm["overall"]:.2f} / {best_vlm_recall:.2f}',
                                "strongest non-YOLO model in this run", bg="#FFFFFF", fg=INK, border=FAINT))
    if yolo_recall is not None and best_vlm_recall is not None:
        yolo_neg_recall = macro_recall.loc[macro_recall["model"] == "yolo", "Negative (absent)"].iloc[0]
        vlm_neg_recall = macro_recall.loc[macro_recall["model"] == best_vlm["model"], "Negative (absent)"].iloc[0]
        gap_x = yolo_neg_recall / vlm_neg_recall if vlm_neg_recall else float("nan")
        tiles.append(stat_tile("RECALL GAP ON ABSENCE DETECTION", f"{gap_x:.1f}×",
                                f"YOLO's negative-class recall ({yolo_neg_recall:.2f}) vs. best VLM's ({vlm_neg_recall:.2f})",
                                bg="#EFE600", fg=INK))
    tiles.append("</div>")
    st.markdown("".join(tiles), unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:12.5px;color:#4A4B47;margin:-8px 0 20px">Latest scored run: '
        f'<code>{latest_run_name}</code> — {"scored live from presence.csv + ground truth" if using_live_data else "baked-in fallback"}. '
        f'Full write-up: <code>reports/llm_vs_yolo_comparison.md</code>.</div>',
        unsafe_allow_html=True,
    )

    def macro_bar(table, metric_label):
        """One row of grouped bars (positive vs. negative class group) per
        model, for a single macro metric — used twice below (accuracy, then
        recall), so the shape lives here once."""
        df = table.melt(id_vars=["model", "model_label"], value_vars=["Positive (present)", "Negative (absent)"],
                         var_name="group", value_name="value").dropna(subset=["value"])
        bar = (
            alt.Chart()
            .mark_bar(size=16, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("group:N", title=None, axis=None),
                y=alt.Y("value:Q", title=f"Macro {metric_label}", scale=alt.Scale(domain=[0, 1.02])),
                color=alt.Color("group:N", title=None,
                                 scale=alt.Scale(domain=["Positive (present)", "Negative (absent)"], range=[INK, NEGATIVE_RED])),
                tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("group:N", title="Class group"),
                         alt.Tooltip("value:Q", title=f"Macro {metric_label}", format=".3f")],
            )
        )
        text = (
            alt.Chart()
            .mark_text(dy=-6, font=CHART_FONT, fontSize=10.5, color=INK)
            .encode(x=alt.X("group:N", axis=None), y=alt.Y("value:Q"), text=alt.Text("value:Q", format=".2f"))
        )
        return (
            alt.layer(bar, text, data=df)
            .properties(width=70, height=220)
            .facet(column=alt.Column("model_label:N", title=None, sort=MODEL_LABEL_ORDER,
                                      header=alt.Header(labelFont=CHART_FONT, labelFontSize=12.5, labelColor=INK, labelOrient="bottom")))
        )

    st.markdown('<div class="hv-h1" style="font-size:22px;margin-bottom:2px">MACRO ACCURACY BY MODEL</div>', unsafe_allow_html=True)
    st.caption("Negative classes (no-helmet, no-boots, …) are where every VLM falls apart — YOLO barely notices the difference.")
    st.altair_chart(_base_config(macro_bar(macro, "accuracy")), width="stretch")

    st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 2px">MACRO RECALL BY MODEL</div>', unsafe_allow_html=True)
    st.caption("Recall = of the cases actually present, how many the model caught — the number that matters most for a safety product.")
    st.altair_chart(_base_config(macro_bar(macro_recall, "recall")), width="stretch")

    st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 2px">PER-CLASS ACCURACY — EVERY MODEL, EVERY CLASS</div>',
                unsafe_allow_html=True)
    st.caption("Darker = higher accuracy. Reads left→right as PPE-present classes, then the four absence classes.")

    heat_df = metrics.copy()
    heat_df["accuracy_display"] = heat_df["accuracy"].apply(lambda v: "—" if pd.isna(v) else f"{v:.2f}")

    cells = (
        alt.Chart(heat_df)
        .mark_rect(stroke="#E4E5E2", strokeWidth=2)
        .encode(
            x=alt.X("class_name:N", title=None, sort=CLASS_ORDER, axis=alt.Axis(labelAngle=-40, labelFontSize=11)),
            y=alt.Y("model_label:N", title=None, sort=MODEL_LABEL_ORDER),
            color=alt.Color("accuracy:Q", title="Accuracy", scale=alt.Scale(scheme="greys", domain=[0, 1]),
                             legend=alt.Legend(orient="right", gradientLength=140)),
            tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("class_name:N", title="Class"),
                     alt.Tooltip("accuracy:Q", title="Accuracy", format=".3f"), alt.Tooltip("recall:Q", title="Recall", format=".3f"),
                     alt.Tooltip("tp:Q", title="tp"), alt.Tooltip("fp:Q", title="fp"),
                     alt.Tooltip("fn:Q", title="fn"), alt.Tooltip("tn:Q", title="tn")],
        )
        .properties(height=max(120, 40 * len(MODEL_ORDER)))
    )
    labels = (
        alt.Chart(heat_df)
        .mark_text(font=CHART_FONT, fontSize=11)
        .encode(
            x=alt.X("class_name:N", sort=CLASS_ORDER), y=alt.Y("model_label:N", sort=MODEL_LABEL_ORDER),
            text="accuracy_display:N",
            color=alt.condition(alt.datum.accuracy > 0.55, alt.value("#FFFFFF"), alt.value(INK)),
        )
        .properties(height=max(120, 40 * len(MODEL_ORDER)))
    )
    st.altair_chart(_base_config((cells + labels)), width="stretch")

    if "person" in metrics["class_name"].values:
        st.markdown('<div class="hv-h1" style="font-size:22px;margin:26px 0 8px">WHY "PERSON" LOOKS WORSE THAN IT IS</div>',
                    unsafe_allow_html=True)
        gt_col1, gt_col2 = st.columns([3, 2])
        with gt_col1:
            person_row = metrics[(metrics["class_name"] == "person") & (metrics["model"] != "yolo")].iloc[0] \
                if (metrics["class_name"] == "person").any() and (metrics["model"] != "yolo").any() else None
            if person_row is not None:
                n_total = int(person_row["tp"] + person_row["fp"])
                gt_df = pd.DataFrame([
                    {"label": "Has a person box", "count": int(person_row["tp"])},
                    {"label": "PPE boxed, but no person box", "count": int(person_row["fp"])},
                ])
                gt_bar = (
                    alt.Chart(gt_df).mark_bar(size=44, cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
                    .encode(
                        y=alt.Y("label:N", title=None, sort=["Has a person box", "PPE boxed, but no person box"]),
                        x=alt.X("count:Q", title=f"of {n_total} sampled test images", scale=alt.Scale(domain=[0, n_total])),
                        color=alt.Color("label:N", scale=alt.Scale(domain=["Has a person box", "PPE boxed, but no person box"],
                                                                     range=[INK, FAINT]), legend=None),
                        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("count:Q", title="images")],
                    ).properties(height=110)
                )
                gt_text = (
                    alt.Chart(gt_df).mark_text(align="left", dx=6, font=CHART_FONT, fontSize=13, fontWeight="bold", color=INK)
                    .encode(y=alt.Y("label:N", sort=["Has a person box", "PPE boxed, but no person box"]), x="count:Q", text="count:Q")
                    .properties(height=110)
                )
                st.altair_chart(_base_config(gt_bar + gt_text), width="stretch")
        with gt_col2:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px dashed #9B9D97;padding:16px;height:100%;box-sizing:border-box">
              <div style="font-size:13px;line-height:1.5;color:#141414">
              Every non-YOLO model in this run reports the same false-positive count on <code>person</code> — that's
              not a coincidence; it's a labeling gap in the test set. Every VLM correctly says <i>"yes, there's a
              person"</i> and gets marked wrong by an incomplete ground-truth box. See the <b>LLM vs YOLO (person
              detection)</b> page for a per-person comparison that corrects for this gap.
              </div>
              <div style="font-size:11.5px;color:#71736D;margin-top:10px">YOLO scores near-perfectly here only because it was
              <i>trained</i> on this same gapped label set — consistency with the labels, not a stronger read on the photo.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:30px 0 18px'/>", unsafe_allow_html=True)
    st.markdown('<div class="hv-h1" style="font-size:18px;margin-bottom:2px;color:#71736D">EXTRA — GEMINI: DESCRIPTIVE VS. STRUCTURED PROMPT</div>',
                unsafe_allow_html=True)
    st.caption(
        "Unscored, from a separate qualitative run — there's no ground truth for prose. Same images, two prompts: "
        "a free-text \"site record\" description, and the strict-JSON prompt everything above is scored on."
    )
    candidates = sorted(glob.glob(str(RUNS_LLM_ROOT / "*" / "prompt_comparison.csv")))
    if not candidates:
        st.markdown(
            '<div style="background:#FFFFFF;border:1px solid #C4C6C0;padding:20px;color:#71736D;font-size:13px">'
            'No <code>prompt_comparison.csv</code> found under <code>runs/llm/</code> yet — run '
            '<code>scripts/gemini_prompt_comparison.py</code> to populate this section.</div>',
            unsafe_allow_html=True,
        )
    else:
        prompt_df = pd.read_csv(candidates[-1])
        pivot = prompt_df.pivot(index="file", columns="prompt_name", values="response_text")
        sample_files = pivot.index[:3]
        cols = st.columns(len(sample_files))
        for col, file in zip(cols, sample_files):
            with col:
                img_path = image_path_for(file)
                if img_path is not None:
                    thumb_b64 = vh.b64_image(Image.open(img_path).convert("RGB"), max_dim=360)
                    st.markdown(f'<img src="data:image/jpeg;base64,{thumb_b64}" '
                                f'style="width:100%;aspect-ratio:4/3;object-fit:cover;display:block;border:1px solid #C4C6C0"/>',
                                unsafe_allow_html=True)
                descriptive = pivot.loc[file].get("1_new_descriptive", "—")
                structured = pivot.loc[file].get("2_existing_structured", "—")
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #C4C6C0;border-top:none;padding:12px 14px 14px">
                  <div style="font-size:12.5px;line-height:1.45;color:#141414">{descriptive}</div>
                  <div class="hv-mono" style="font-size:10px;color:#71736D;background:#F0F1EC;padding:6px 8px;margin-top:10px;
                       white-space:pre-wrap;word-break:break-word">{structured}</div>
                </div>
                """, unsafe_allow_html=True)

    st.caption("Source data: every run_manifest.json / presence.csv / prompt_comparison.csv under runs/llm/.")
