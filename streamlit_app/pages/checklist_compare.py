"""HI-VIS — per-person PPE checklist comparison: run + analyze in one page.

Different question than the other comparison pages: instead of "does class
X appear anywhere in this image" (the flat presence prompt every other
page uses), this asks every chat-style model "how many people do you see,
and for each, is helmet/vest/gloves/boots present?" — scripts/
compare_models.run_checklist_steps() drives it, model_adapters.
parse_person_checklist_json() parses it. YOLO isn't in the model list here
— it has no describe()-style free-text interface, and this prompt shape
isn't something a box detector answers.

Ground truth: our labels undercount people — a known gap (PPE item boxes
exist in ~half the test images with no matching Person box at all). So
"ground truth" here isn't just the label counts: for every count series
(person, and each item's present/absent count), the *effective* ground
truth is max(label count, median of the models' own counts) — if the
model consensus claims more instances than the labels have, the labels
are very likely the ones that are wrong, not every model at once. Every
image where that happens is flagged, and the ANALYSIS section below
breaks down which images/sources it happens to most, so the gap is
visible rather than silently baked into the metrics.
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
from compare_models import (  # noqa: E402
    DATASET_NAME,
    LLM_RUNS_ROOT,
    MERGED_ROOT,
    build_adapter,
    build_run_name,
    image_path_for,
    run_checklist_steps,
    sample_test_images,
)
from model_adapters import ADAPTERS, CHECKLIST_ITEMS, DEFAULT_PROMPT_TEMPLATE  # noqa: E402

import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("PERSON CHECKLIST COMPARISON", "how many people, and what are they wearing?"),
            unsafe_allow_html=True)

MUTED = "#71736D"
CHAT_MODELS = [m for m in ADAPTERS if m != "yolo"]  # yolo has no describe() — this prompt doesn't apply to it
SERIES = ["person"] + CHECKLIST_ITEMS + [f"no-{item}" for item in CHECKLIST_ITEMS]

st.caption(
    "Every model gets one prompt: count the people, then check helmet/vest/gloves/boots for "
    "each. Runs live in this tab (same checkpoint-to-disk pattern as Live Comparison), then "
    "scores itself immediately below — see the module docstring for why \"ground truth\" here "
    "is enhanced by model consensus, not just the raw labels."
)


# ---------------------------------------------------------------------------
# ground truth + scoring helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_gt_counts(sampled_files):
    """dict[file][series] -> ground-truth box count, straight from labels —
    no enhancement here, that happens per-image against the live model
    counts below."""
    labels_path = MERGED_ROOT / "labels_long.csv"
    data_yaml_path = MERGED_ROOT / "data.yaml"
    if not (labels_path.exists() and data_yaml_path.exists()):
        return None
    labels_long = pd.read_csv(labels_path)
    class_names = yaml.safe_load(data_yaml_path.read_text())["names"]
    id_to_name = {i: n for i, n in enumerate(class_names)}
    sub = labels_long[labels_long["file"].isin(sampled_files)].copy()
    sub["class_name"] = sub["class_id"].map(id_to_name)
    counts = sub.groupby(["file", "class_name"]).size()
    return {f: {s: int(counts.get((f, s), 0)) for s in SERIES} for f in sampled_files}


def model_counts_for_people(people):
    """people: list[{item: bool}] or None (parse failure) -> dict[series]->
    int, or None if the model's answer couldn't be read at all — kept
    distinct from a genuine "0 people" answer, which scores normally."""
    if people is None:
        return None
    n = len(people)
    counts = {"person": n}
    for item in CHECKLIST_ITEMS:
        present = sum(1 for p in people if p.get(item))
        counts[item] = present
        counts[f"no-{item}"] = n - present
    return counts


def enhance_gt(gt_count, model_counts_by_model):
    """max(label count, median of the models' counts) — the "enhanced"
    ground truth this page scores against — plus the consensus value and
    whether it exceeded the raw label (the flag)."""
    votes = sorted(c for c in model_counts_by_model.values() if c is not None)
    if not votes:
        return gt_count, None, False
    consensus = statistics.median_low(votes)
    return max(gt_count, consensus), consensus, consensus > gt_count


# ---------------------------------------------------------------------------
# config form
# ---------------------------------------------------------------------------

with st.form("checklist_run_config"):
    c1, c2 = st.columns(2)
    n_images = c1.slider("Images to sample", 5, 100, 20)
    seed = c2.number_input("Seed", value=42, step=1)
    model_names = st.multiselect(
        "Models",
        CHAT_MODELS,
        default=["ollama", "qwen3-vl"],
        help="ollama/qwen3-vl/gemma4/minicpm-v need Ollama running locally; claude/gemini call a paid API. "
             "YOLO isn't offered here — it can't answer a free-text checklist prompt.",
    )
    cloud_in_selection = [m for m in model_names if ADAPTERS[m]["is_cloud"]]
    include_cloud = st.checkbox(
        f"Allow cloud models ({', '.join(cloud_in_selection) or 'claude/gemini'}) — calls a paid API",
        value=False,
    )
    submitted = st.form_submit_button("Run checklist comparison", type="primary")

if not submitted:
    st.info("Pick models above and hit **Run checklist comparison** — results stream in below, "
            "then score themselves once the run finishes.")
    st.stop()
if not model_names:
    st.error("Pick at least one model.")
    st.stop()
if cloud_in_selection and not include_cloud:
    st.error(f"{cloud_in_selection} need “Allow cloud models” checked — that's a deliberate cost guard.")
    st.stop()

sampled_files = sample_test_images(n_images, seed)
st.write(f"Sampled **{len(sampled_files)}** test images.")

# build_adapter()/each adapter's __init__ still wants a prompt_template= —
# unused here since this page only ever calls .describe(), never .predict().
args = SimpleNamespace(ollama_url="http://localhost:11434", prompt_template=DEFAULT_PROMPT_TEMPLATE)
for name in model_names:
    setattr(args, f"{name.replace('-', '_')}_model", ADAPTERS[name]["default_model"])

adapters = {}
load_status = st.empty()
for name in model_names:
    load_status.markdown(f"Loading `{name}`...")
    adapters[name] = build_adapter(name, args)
load_status.markdown("Loaded: " + ", ".join(f"`{n}`" for n in adapters))

run_name = build_run_name(n_images, seed, model_names)
run_dir = LLM_RUNS_ROOT / run_name
run_dir.mkdir(parents=True, exist_ok=True)
counts_path = run_dir / "person_counts.csv"
items_path = run_dir / "person_items.csv"

# ---------------------------------------------------------------------------
# live run
# ---------------------------------------------------------------------------

progress_bar = st.progress(0.0)
status_line = st.empty()
st.markdown('<div class="hv-h1" style="font-size:15px;margin:16px 0 6px">RESULTS (live)</div>',
            unsafe_allow_html=True)
feed = st.container()

count_rows = []       # file, model, person_count (None on parse failure)
item_rows = []         # file, model, person_idx, helmet, vest, gloves, boots
people_by_pair = {}    # (file, model) -> people list or None, for scoring below
file_buf = {}
current_file = None
t_start = time.perf_counter()


def render_file_card(file_stem, buf):
    with feed:
        cols = st.columns([1, 3])
        img_path = image_path_for(file_stem)
        if img_path:
            cols[0].image(str(img_path), width="stretch")
        with cols[1]:
            st.markdown(f'<div class="hv-mono" style="font-size:11px;color:{MUTED}">{file_stem}</div>',
                        unsafe_allow_html=True)
            for name in model_names:
                counts = buf.get(name)
                if counts is None:
                    st.markdown(f'<div><b>{name}</b>: <span style="color:{MUTED}">parse error</span></div>',
                                unsafe_allow_html=True)
                    continue
                items_line = " · ".join(f"{item} {counts[item]}/{counts['person']}" for item in CHECKLIST_ITEMS)
                st.markdown(f"<div><b>{name}</b>: {counts['person']} people — {items_line}</div>",
                            unsafe_allow_html=True)
        st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)


for step in run_checklist_steps(adapters, sampled_files, image_path_for=image_path_for):
    if step["skipped"]:
        continue

    if step["file"] != current_file:
        if current_file is not None:
            render_file_card(current_file, file_buf)
        current_file, file_buf = step["file"], {}

    people = step["people"]
    people_by_pair[(step["file"], step["model"])] = people
    count_rows.append({"file": step["file"], "model": step["model"],
                        "person_count": len(people) if people is not None else None})
    if people:
        for idx, p in enumerate(people):
            item_rows.append({"file": step["file"], "model": step["model"], "person_idx": idx, **p})

    counts = model_counts_for_people(people)
    file_buf[step["model"]] = counts

    done, total = step["done"], step["total"]
    progress_bar.progress(done / total)
    per_model = " · ".join(f"{n}:{c}/{len(sampled_files)}" for n, c in step["done_per_model"].items())
    status_line.markdown(f"**{done}/{total}** pairs · {time.perf_counter() - t_start:.0f}s elapsed — {per_model}")

    if done % 10 == 0 or done == total:  # checkpoint: survive a killed tab
        pd.DataFrame(count_rows).to_csv(counts_path, index=False)
        pd.DataFrame(item_rows).to_csv(items_path, index=False)

if current_file is not None:
    render_file_card(current_file, file_buf)

manifest = {
    "run_name": f"{run_name}_COMPLETE",
    "status": "COMPLETE",
    "kind": "checklist",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dataset_name": DATASET_NAME,
    "n_images_sampled": len(sampled_files),
    "seed": seed,
    "models": model_names,
    "checklist_items": CHECKLIST_ITEMS,
    "sampled_files": sampled_files,
}
final_dir = LLM_RUNS_ROOT / f"{run_name}_COMPLETE"
run_dir.rename(final_dir)
pd.DataFrame(count_rows).to_csv(final_dir / "person_counts.csv", index=False)
pd.DataFrame(item_rows).to_csv(final_dir / "person_items.csv", index=False)
(final_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
st.success(f"Done in {time.perf_counter() - t_start:.0f}s — saved to `runs/llm/{final_dir.name}/`.")

# ---------------------------------------------------------------------------
# analysis — live, right below, no separate page/reload needed
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:20px;margin:28px 0 6px">ANALYSIS</div>', unsafe_allow_html=True)

gt_counts = load_gt_counts(tuple(sampled_files))
if gt_counts is None:
    st.warning("data/merged/labels_long.csv or data.yaml not found on this checkout — can't score against "
               "ground truth (they're git-ignored). The run above is still saved.")
    st.stop()

per_image_rows = []       # file, series, gt, effective_gt, consensus, flagged, {model: count}
for f in sampled_files:
    model_counts = {m: model_counts_for_people(people_by_pair.get((f, m))) for m in model_names}
    for series in SERIES:
        by_model = {m: (mc[series] if mc is not None else None) for m, mc in model_counts.items()}
        gt = gt_counts[f][series]
        effective_gt, consensus, flagged = enhance_gt(gt, by_model)
        per_image_rows.append({
            "file": f, "series": series, "gt": gt, "effective_gt": effective_gt,
            "consensus": consensus, "flagged": flagged, **{f"m_{m}": by_model[m] for m in model_names},
        })
per_image = pd.DataFrame(per_image_rows)

# --- headline: how often the labels undercount, and where ------------------
person_rows = per_image[per_image["series"] == "person"]
n_flagged = int(person_rows["flagged"].sum())
t1, t2, t3 = st.columns(3)
t1.metric("Images scored", len(sampled_files))
t2.metric("Person-count flagged", n_flagged,
          help="Images where the model consensus claims MORE people than our labels have a Person box for.")
t3.metric("Flagged rate", f"{n_flagged / len(sampled_files):.0%}" if sampled_files else "—")

if n_flagged:
    st.subheader("Flagged images — label undercounts people vs. model consensus")
    flagged_view = person_rows[person_rows["flagged"]][["file", "gt", "consensus", "effective_gt"]].rename(
        columns={"gt": "label person count", "consensus": "model consensus", "effective_gt": "effective (used below)"}
    )
    st.dataframe(flagged_view, hide_index=True, width="stretch")

    st.subheader("Which sources it's prone to")
    st.caption("Filename prefix before \"__\" identifies the original dataset a test image came from.")
    person_rows = person_rows.copy()
    person_rows["source"] = person_rows["file"].str.split("__").str[0]
    by_source = person_rows.groupby("source").agg(images=("file", "count"), flagged=("flagged", "sum"))
    by_source["flag_rate"] = (by_source["flagged"] / by_source["images"]).round(2)
    st.dataframe(by_source.sort_values("flag_rate", ascending=False), width="stretch")
else:
    st.caption("No images where model consensus exceeded the label's person count in this run.")

st.divider()

# --- per-series accuracy against the ENHANCED ground truth ------------------
st.subheader("Count accuracy per model (scored against the enhanced ground truth above)")
st.caption(
    "Exact-match rate: fraction of images where the model's count equals the effective ground "
    "truth exactly. Mean abs. error: average |model count − effective ground truth|, lower is "
    "better. Only images the model actually answered (no parse failure) count toward either."
)
macro_rows = []
for series in SERIES:
    sub = per_image[per_image["series"] == series]
    for m in model_names:
        col = sub[f"m_{m}"].dropna()
        if col.empty:
            continue
        err = (col - sub.loc[col.index, "effective_gt"]).abs()
        macro_rows.append({
            "series": series, "model": m, "n": len(col),
            "exact_match_rate": round((err == 0).mean(), 2),
            "mean_abs_error": round(err.mean(), 2),
        })
macro = pd.DataFrame(macro_rows)

exact_pivot = macro.pivot(index="series", columns="model", values="exact_match_rate").reindex(SERIES)
st.caption("Exact-match rate by class (rows) and model (columns)")
st.bar_chart(exact_pivot, stack=False)  # grouped, not stacked — these are independent 0-1 rates, not parts of a whole

error_pivot = macro.pivot(index="series", columns="model", values="mean_abs_error").reindex(SERIES)
st.caption("Mean absolute count error by class (rows) and model (columns) — lower is better")
st.bar_chart(error_pivot, stack=False)

with st.expander("Full numbers"):
    st.dataframe(macro, hide_index=True, width="stretch")
