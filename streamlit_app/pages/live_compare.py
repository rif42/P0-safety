"""HI-VIS — run the YOLO vs LLM/VLM comparison live, in the browser.

Reuses scripts/compare_models.run_comparison_steps() — the same generator
main() (the CLI) drives — so results stream into this page one (image,
model) pair at a time instead of only appearing after a full CLI/notebook
run finishes. detections.csv/presence.csv are checkpointed to runs/llm/
every 10 pairs, same files main() writes, so a browser refresh never loses
more than 10 pairs of progress.

One honest caveat: Streamlit's own Stop control (the square icon that
replaces the run icon while a script is executing) kills the Python
thread outright — there's no hook to catch it and finalize the run the
way Ctrl+C is caught on the CLI (see compare_models.main()). A mid-run
Stop leaves the last checkpoint sitting under the plain, un-suffixed run
folder instead of a tidy `_COMPLETE` one with a manifest. Letting the run
finish is the only way to get both.
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


def list_past_runs():
    """Every run already on disk under runs/llm/ — newest first. Not
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


past_runs = list_past_runs()
with st.expander(f"PREVIOUS RUNS ({len(past_runs)})", expanded=False):
    if not past_runs:
        st.caption("No runs yet — run one below.")
    else:
        st.dataframe(pd.DataFrame(past_runs), hide_index=True, width="stretch")
        st.caption("Full scored breakdown of any of these: the **LLM vs YOLO** page.")

with st.form("live_run_config"):
    c1, c2 = st.columns(2)
    n_images = c1.slider("Images to sample", 5, 100, 20)
    seed = c2.number_input("Seed", value=42, step=1)
    model_names = st.multiselect(
        "Models",
        list(ADAPTERS),
        default=["yolo", "ollama"],
        help="ollama/qwen3-vl/gemma4/minicpm-v need Ollama running locally; claude/gemini call a paid API.",
    )
    cloud_in_selection = [m for m in model_names if ADAPTERS[m]["is_cloud"]]
    include_cloud = st.checkbox(
        f"Allow cloud models ({', '.join(cloud_in_selection) or 'claude/gemini'}) — calls a paid API",
        value=False,
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

args = SimpleNamespace(
    yolo_weights=str(DEFAULT_YOLO_WEIGHTS),
    ollama_model=ADAPTERS["ollama"]["default_model"],
    qwen3_vl_model=ADAPTERS["qwen3-vl"]["default_model"],
    gemma4_model=ADAPTERS["gemma4"]["default_model"],
    minicpm_v_model=ADAPTERS["minicpm-v"]["default_model"],
    ollama_url="http://localhost:11434",
    claude_model=ADAPTERS["claude"]["default_model"],
    gemini_model=ADAPTERS["gemini"]["default_model"],
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
)

sampled_files = sample_test_images(n_images, seed)
st.write(f"Sampled **{len(sampled_files)}** test images.")

adapters = {}
load_status = st.empty()
for name in model_names:
    load_status.markdown(f"Loading `{name}`...")
    adapters[name] = build_adapter(name, args)
load_status.markdown("Loaded: " + ", ".join(f"`{n}`" for n in adapters))

run_name = build_run_name(n_images, seed, model_names)
run_dir = LLM_RUNS_ROOT / run_name
run_dir.mkdir(parents=True, exist_ok=True)
detections_path = run_dir / "detections.csv"
presence_path = run_dir / "presence.csv"
descriptive_path = run_dir / "descriptive_responses.csv"

detection_rows, presence_rows, descriptive_rows = [], [], []
parse_failures = {n: 0 for n in model_names}

progress_bar = st.progress(0.0)
status_line = st.empty()
st.markdown('<div class="hv-h1" style="font-size:15px;margin:16px 0 6px">RESULTS (live)</div>', unsafe_allow_html=True)
feed = st.container()


def chip(label, color):
    return (
        f'<span class="hv-mono" style="display:inline-block;font-size:10px;padding:2px 7px;'
        f'margin:2px 3px 2px 0;background:{color};color:#FFFFFF;white-space:nowrap">{label}</span>'
    )


def render_card(file_stem, buf):
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


file_buf = {}
current_file = None
t_start = time.perf_counter()

for step in run_comparison_steps(adapters, sampled_files, image_path_for=image_path_for, descriptive_prompt=DESCRIPTIVE_PROMPT):
    if step["skipped"]:
        continue

    if step["file"] != current_file:
        if current_file is not None:
            render_card(current_file, file_buf)
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

    if done % 10 == 0 or done == total:  # checkpoint: survive a killed tab, not just a clean finish
        pd.DataFrame(detection_rows).to_csv(detections_path, index=False)
        pd.DataFrame(presence_rows).to_csv(presence_path, index=False)
        if descriptive_rows:
            pd.DataFrame(descriptive_rows).to_csv(descriptive_path, index=False)

if current_file is not None:
    render_card(current_file, file_buf)

manifest = {
    "run_name": f"{run_name}_COMPLETE",
    "status": "COMPLETE",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dataset_name": DATASET_NAME,
    "n_images_requested": n_images,
    "n_images_sampled": len(sampled_files),
    "seed": seed,
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
