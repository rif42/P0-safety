"""HI-VIS — per-person PPE checklist comparison: run + analyze in one page.

Different question than the other comparison pages: instead of "does class
X appear anywhere in this image" (the flat presence prompt every other
page uses), this asks every chat-style model "how many people do you see,
and for each, is helmet/vest/gloves/boots present?" — scripts/
compare_models.run_checklist_steps() drives it. YOLO participates too, via
its own real boxes instead of a prompt (model_adapters.
people_from_detections() — person box + contained item box = present), so
it runs through the identical scoring pipeline as every chat model rather
than being left out because it has no describe().

Ground truth: our labels undercount people — a known gap (PPE item boxes
exist in ~half the test images with no matching Person box at all). So
"ground truth" here isn't just the label counts: for every count series
(person, and each item's present/absent count), the *effective* ground
truth is max(label count, median of the models' own counts) — if the
model consensus claims more instances than the labels have, the labels
are very likely the ones that are wrong, not every model at once. Every
image where that happens is flagged, and the ANALYSIS section below
breaks down which images/sources it happens to most, so the gap is
visible rather than silently baked into the metrics. The live feed also
draws the dataset's own ground-truth boxes on each thumbnail, so the gap
(or lack of one) is visible at a glance too, not just in the numbers.

Pause/resume: same story as live_compare.py — a "Pause" click aborts the
running script the same way Streamlit's native Stop does (no cleanup hook
either way), so what makes a paused run resumable is the checkpoint
already on disk (person_counts.csv/person_items.csv, written every 10
pairs) plus run_config.json (written before the loop starts) recording
enough to rebuild sampling/models later. PAUSED RUNS below lists any run
this page started that never got a run_manifest.json.
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
from PIL import Image, ImageDraw

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
from compare_models import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_YOLO_WEIGHTS,
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
GT_BOX_COLOR = "#1B7A3D"
SERIES = ["person"] + CHECKLIST_ITEMS + [f"no-{item}" for item in CHECKLIST_ITEMS]
CLASS_NAMES = yaml.safe_load((MERGED_ROOT / "data.yaml").read_text())["names"]

st.caption(
    "Every model gets one prompt: count the people, then check helmet/vest/gloves/boots for "
    "each — YOLO answers the same question from its own boxes, not a prompt. Runs live in this "
    "tab, then scores itself immediately below — see the module docstring for why \"ground "
    "truth\" here is enhanced by model consensus, not just the raw labels."
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
    if not labels_path.exists():
        return None
    labels_long = pd.read_csv(labels_path)
    id_to_name = {i: n for i, n in enumerate(CLASS_NAMES)}
    sub = labels_long[labels_long["file"].isin(sampled_files)].copy()
    sub["class_name"] = sub["class_id"].map(id_to_name)
    counts = sub.groupby(["file", "class_name"]).size()
    return {f: {s: int(counts.get((f, s), 0)) for s in SERIES} for f in sampled_files}


def load_gt_boxes(file_stem):
    """[(class_name, (x1,y1,x2,y2) normalized)] straight from the
    YOLO-format label .txt for this test image."""
    label_path = MERGED_ROOT / "test" / "labels" / f"{file_stem}.txt"
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cid, cx, cy, bw, bh = line.split()
        cid, cx, cy, bw, bh = int(cid), float(cx), float(cy), float(bw), float(bh)
        boxes.append((CLASS_NAMES[cid], (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return boxes


def draw_gt_overlay(image_path, boxes):
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    draw = ImageDraw.Draw(img)
    line_w = max(2, min(iw, ih) // 250)
    for class_name, (x1, y1, x2, y2) in boxes:
        px = (x1 * iw, y1 * ih, x2 * iw, y2 * ih)
        draw.rectangle(px, outline=GT_BOX_COLOR, width=line_w)
        draw.text((px[0] + 2, px[1] + 2), class_name, fill=GT_BOX_COLOR)
    return img


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
# run history — finished runs, and paused ones that can be resumed. Filtered
# to kind=="checklist" so this page's runs and live_compare.py's don't show
# up in each other's lists — both write run_config.json/run_manifest.json.
# ---------------------------------------------------------------------------

def list_past_checklist_runs():
    rows = []
    if not LLM_RUNS_ROOT.exists():
        return rows
    for d in sorted(LLM_RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        manifest_path = d / "run_manifest.json"
        if not (d.is_dir() and manifest_path.exists()):
            continue
        m = json.loads(manifest_path.read_text())
        if m.get("kind") != "checklist":
            continue
        rows.append({
            "run": d.name, "created_at": m.get("created_at", "")[:19].replace("T", " "),
            "images": m.get("n_images_sampled"), "models": ", ".join(m.get("models", [])),
        })
    return rows


def list_paused_checklist_runs():
    rows = []
    if not LLM_RUNS_ROOT.exists():
        return rows
    for d in sorted(LLM_RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        config_path = d / "run_config.json"
        if not (d.is_dir() and config_path.exists() and not (d / "run_manifest.json").exists()):
            continue
        cfg = json.loads(config_path.read_text())
        if cfg.get("kind") != "checklist":
            continue
        done_pairs = 0
        counts_path = d / "person_counts.csv"
        if counts_path.exists():
            done_pairs = len(pd.read_csv(counts_path)[["file", "model"]].drop_duplicates())
        rows.append({
            "run_dir": d, "run_name": d.name, "models": cfg["model_names"], "seed": cfg.get("seed"),
            "n_images": len(cfg["sampled_files"]), "done_pairs": done_pairs,
            "total_pairs": len(cfg["sampled_files"]) * len(cfg["model_names"]),
        })
    return rows


def load_existing_checklist_rows(run_dir):
    """Reconstruct count_rows/item_rows/people_by_pair from a paused run's
    checkpoint — people_by_pair also doubles as the resume's skip_pairs
    (every (file, model) key in it was already asked, parse failure or not;
    a failure isn't retried automatically here, same as everywhere else in
    this codebase)."""
    counts_path = run_dir / "person_counts.csv"
    items_path = run_dir / "person_items.csv"
    count_rows = pd.read_csv(counts_path).to_dict("records") if counts_path.exists() else []
    items_df = pd.read_csv(items_path) if items_path.exists() else pd.DataFrame()
    item_rows = items_df.to_dict("records")

    people_by_pair = {}
    for row in count_rows:
        key = (row["file"], row["model"])
        pc = row["person_count"]
        if pd.isna(pc):
            people_by_pair[key] = None
            continue
        if int(pc) == 0 or items_df.empty:
            people_by_pair[key] = []
            continue
        sub = items_df[(items_df["file"] == row["file"]) & (items_df["model"] == row["model"])].sort_values("person_idx")
        people_by_pair[key] = [{item: bool(r[item]) for item in CHECKLIST_ITEMS} for _, r in sub.iterrows()]
    return count_rows, item_rows, people_by_pair


# ---------------------------------------------------------------------------
# shared rendering
# ---------------------------------------------------------------------------

def render_file_card(feed, model_names, file_stem, buf):
    with feed:
        cols = st.columns([1, 3])
        img_path = image_path_for(file_stem)
        if img_path:
            gt_boxes = load_gt_boxes(file_stem)
            cols[0].image(draw_gt_overlay(img_path, gt_boxes) if gt_boxes else str(img_path), width="stretch")
            cols[0].caption(f"ground truth: {len(gt_boxes)} box(es)" if gt_boxes else "no ground-truth boxes for this image")
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


def build_args(model_names):
    args = SimpleNamespace(
        ollama_url="http://localhost:11434",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,  # unused by describe()/predict()-via-boxes, adapters still want it
        yolo_weights=str(DEFAULT_YOLO_WEIGHTS),
    )
    for name in model_names:
        if name != "yolo":
            setattr(args, f"{name.replace('-', '_')}_model", ADAPTERS[name]["default_model"])
    return args


def load_adapters(model_names, args):
    adapters = {}
    load_status = st.empty()
    for name in model_names:
        load_status.markdown(f"Loading `{name}`...")
        adapters[name] = build_adapter(name, args)
    load_status.markdown("Loaded: " + ", ".join(f"`{n}`" for n in adapters))
    return adapters


def run_checklist_live(run_dir, run_name, model_names, sampled_files, adapters, seed,
                        count_rows, item_rows, people_by_pair, skip_pairs):
    counts_path = run_dir / "person_counts.csv"
    items_path = run_dir / "person_items.csv"

    progress_bar = st.progress(0.0)
    status_line = st.empty()
    st.button(
        "⏸ Pause run", key="checklist_pause_btn",
        help="Stops now — the checkpoint already on disk is safe. Pick it back up from PAUSED RUNS above.",
    )
    st.markdown('<div class="hv-h1" style="font-size:15px;margin:16px 0 6px">RESULTS (live)</div>',
                unsafe_allow_html=True)
    feed = st.container()

    # Replay whatever this run already has on disk (a resume) before the
    # live loop continues — same file-boundary logic as live_compare.py: a
    # fully-covered file gets its final card now, a file caught mid-way
    # seeds file_buf/current_file so the loop below merges into it.
    file_buf = {}
    current_file = None
    if skip_pairs:
        covered = {}
        for f, m in skip_pairs:
            covered.setdefault(f, set()).add(m)
        for file_stem in sampled_files:
            if file_stem not in covered:
                continue
            buf = {m: model_counts_for_people(people_by_pair.get((file_stem, m))) for m in covered[file_stem]}
            if covered[file_stem] == set(model_names):
                render_file_card(feed, model_names, file_stem, buf)
            else:
                current_file, file_buf = file_stem, buf

    t_start = time.perf_counter()
    for step in run_checklist_steps(adapters, sampled_files, image_path_for=image_path_for, skip_pairs=skip_pairs):
        if step["skipped"]:
            continue
        if step["resumed"]:
            progress_bar.progress(step["done"] / step["total"])
            continue

        if step["file"] != current_file:
            if current_file is not None:
                render_file_card(feed, model_names, current_file, file_buf)
            current_file, file_buf = step["file"], {}

        people = step["people"]
        people_by_pair[(step["file"], step["model"])] = people
        count_rows.append({"file": step["file"], "model": step["model"],
                            "person_count": len(people) if people is not None else None})
        if people:
            for idx, p in enumerate(people):
                item_rows.append({"file": step["file"], "model": step["model"], "person_idx": idx, **p})

        file_buf[step["model"]] = model_counts_for_people(people)

        done, total = step["done"], step["total"]
        progress_bar.progress(done / total)
        per_model = " · ".join(f"{n}:{c}/{len(sampled_files)}" for n, c in step["done_per_model"].items())
        status_line.markdown(f"**{done}/{total}** pairs · {time.perf_counter() - t_start:.0f}s elapsed — {per_model}")

        if done % 10 == 0 or done == total:  # checkpoint: survive a paused/killed tab
            pd.DataFrame(count_rows).to_csv(counts_path, index=False)
            pd.DataFrame(item_rows).to_csv(items_path, index=False)

    if current_file is not None:
        render_file_card(feed, model_names, current_file, file_buf)

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
# history + resume UI
# ---------------------------------------------------------------------------

past_runs = list_past_checklist_runs()
with st.expander(f"PREVIOUS RUNS ({len(past_runs)})", expanded=False):
    if not past_runs:
        st.caption("No finished checklist runs yet.")
    else:
        st.dataframe(pd.DataFrame(past_runs), hide_index=True, width="stretch")

paused_runs = list_paused_checklist_runs()
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
# resume path, or fresh-run config form
# ---------------------------------------------------------------------------

if resume_clicked is not None:
    run_dir = resume_clicked["run_dir"]
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model_names = cfg["model_names"]
    sampled_files = cfg["sampled_files"]
    seed = cfg.get("seed")

    args = build_args(model_names)
    adapters = load_adapters(model_names, args)
    count_rows, item_rows, people_by_pair = load_existing_checklist_rows(run_dir)
    skip_pairs = set(people_by_pair.keys())

    st.write(f"Resuming **{run_dir.name}** — {len(skip_pairs)}/{len(sampled_files) * len(model_names)} pairs already done.")
    run_checklist_live(run_dir, run_dir.name, model_names, sampled_files, adapters, seed,
                        count_rows, item_rows, people_by_pair, skip_pairs)
else:
    with st.form("checklist_run_config"):
        c1, c2 = st.columns(2)
        n_images = c1.slider("Images to sample", 1, 100, 20)
        seed = c2.number_input("Seed", value=42, step=1)
        model_names = st.multiselect(
            "Models",
            list(ADAPTERS),
            default=["yolo", "ollama"],
            help="YOLO answers from its own boxes, no prompt. ollama/qwen3-vl/gemma4/minicpm-v need Ollama "
                 "running locally; claude/gemini call a paid API.",
        )
        cloud_in_selection = [m for m in model_names if ADAPTERS[m]["is_cloud"]]
        # key= pins this checkbox's identity — without it the auto-derived
        # key includes the label text below, which embeds cloud_in_selection.
        # Changing which models are selected then changes the label (hence
        # the widget's "identity"), so Streamlit treats it as a brand-new
        # checkbox and resets it to value=False right as the form submits.
        include_cloud = st.checkbox(
            f"Allow cloud models ({', '.join(cloud_in_selection) or 'claude/gemini'}) — calls a paid API",
            value=False, key="checklist_include_cloud",  # page-specific: session_state is shared across all pages
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

    args = build_args(model_names)
    adapters = load_adapters(model_names, args)

    run_name = build_run_name(n_images, seed, model_names)
    run_dir = LLM_RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Written before the loop starts, not after — this is what a paused run
    # needs to be resumable at all (see list_paused_checklist_runs() above).
    (run_dir / "run_config.json").write_text(json.dumps(
        {"kind": "checklist", "model_names": model_names, "sampled_files": sampled_files,
         "seed": seed, "n_images": n_images},
        indent=2,
    ))
    count_rows, item_rows, people_by_pair = [], [], {}
    run_checklist_live(run_dir, run_name, model_names, sampled_files, adapters, seed,
                        count_rows, item_rows, people_by_pair, set())

# ---------------------------------------------------------------------------
# analysis — live, right below, common to both the resume and fresh paths
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:20px;margin:28px 0 6px">ANALYSIS</div>', unsafe_allow_html=True)

gt_counts = load_gt_counts(tuple(sampled_files))
if gt_counts is None:
    st.warning("data/merged/labels_long.csv not found on this checkout — can't score against ground truth "
               "(it's git-ignored). The run above is still saved.")
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

if macro.empty:
    st.caption("No answered pairs to score yet.")
else:
    exact_pivot = macro.pivot(index="series", columns="model", values="exact_match_rate").reindex(SERIES)
    st.caption("Exact-match rate by class (rows) and model (columns)")
    st.bar_chart(exact_pivot, stack=False)  # grouped, not stacked — independent 0-1 rates, not parts of a whole

    error_pivot = macro.pivot(index="series", columns="model", values="mean_abs_error").reindex(SERIES)
    st.caption("Mean absolute count error by class (rows) and model (columns) — lower is better")
    st.bar_chart(error_pivot, stack=False)

    with st.expander("Full numbers"):
        st.dataframe(macro, hide_index=True, width="stretch")
