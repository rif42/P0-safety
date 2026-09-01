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

Pause/resume: same story as llm_comparison.py — a "Pause" click aborts the
running script the same way Streamlit's native Stop does (no cleanup hook
either way), so what makes a paused run resumable is the checkpoint
already on disk (person_counts.csv/person_items.csv, written every 10
pairs) plus run_config.json (written before the loop starts) recording
enough to rebuild sampling/models later. PAUSED RUNS below lists any run
this page started that never got a run_manifest.json.
"""

import html
import json
import shutil
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
from gemini_prompt_comparison import DESCRIPTIVE_PROMPT  # noqa: E402
from model_adapters import ADAPTERS, CHECKLIST_ITEMS, DEFAULT_PROMPT_TEMPLATE  # noqa: E402

import view_helpers as vh

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("LLM vs YOLO (PERSON DETECTION)", "how many people, and what are they wearing?"),
            unsafe_allow_html=True)

INK = "#141414"
MUTED = "#71736D"
FAINT = "#C4C6C0"
GT_BOX_COLOR = "#1B7A3D"
POSITIVE_GREEN = "#1B7A3D"  # matches this app's existing compliant=green convention
NEGATIVE_RED = "#B02A20"    # matches this app's existing non-compliant=red convention
SERIES = ["person"] + CHECKLIST_ITEMS + [f"no-{item}" for item in CHECKLIST_ITEMS]
CLASS_NAMES = yaml.safe_load((MERGED_ROOT / "data.yaml").read_text())["names"]


def delete_run(run_dir):
    shutil.rmtree(run_dir, ignore_errors=True)


def stat_tile(label, value, note, bg=INK, fg="#FFFFFF", border=None):
    border_css = f"border:1px solid {border};" if border else ""
    return f"""
    <div style="background:{bg};color:{fg};{border_css}padding:16px 20px 14px">
      <div class="hv-mono" style="font-size:11px;letter-spacing:1.5px;color:{'#9B9D97' if bg == INK else MUTED}">{label}</div>
      <div class="hv-h1" style="font-size:44px;line-height:1;color:{fg}">{value}</div>
      <div style="font-size:12px;color:{'#9B9D97' if bg == INK else MUTED}">{note}</div>
    </div>"""


def model_chip(name):
    bg = INK if name == "yolo" else "#FFFFFF"
    fg = "#FFFFFF" if name == "yolo" else INK
    border = INK if name == "yolo" else FAINT
    return (f'<span class="hv-mono" style="display:inline-block;font-size:10.5px;padding:3px 8px;'
            f'background:{bg};color:{fg};border:1px solid {border};margin:2px 4px 2px 0;white-space:nowrap">{name}</span>')

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
# to kind=="checklist" so this page's runs and llm_comparison.py's don't show
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
            "run_dir": d, "run": d.name, "created_at": m.get("created_at", "")[:19].replace("T", " "),
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
    """Reconstruct count_rows/item_rows/text_rows/descriptive_rows/
    people_by_pair/text_by_pair/descriptive_by_pair/latency_by_pair from a
    paused run's checkpoint — people_by_pair also doubles as the resume's
    skip_pairs (every (file, model) key in it was already asked, parse
    failure or not; a failure isn't retried automatically here, same as
    everywhere else in this codebase)."""
    counts_path = run_dir / "person_counts.csv"
    items_path = run_dir / "person_items.csv"
    text_path = run_dir / "raw_responses.csv"
    descriptive_path = run_dir / "descriptive_responses.csv"
    count_rows = pd.read_csv(counts_path).to_dict("records") if counts_path.exists() else []
    items_df = pd.read_csv(items_path) if items_path.exists() else pd.DataFrame()
    item_rows = items_df.to_dict("records")
    text_df = pd.read_csv(text_path) if text_path.exists() else pd.DataFrame()
    text_rows = text_df.to_dict("records")
    text_by_pair = {(r["file"], r["model"]): r["response_text"] for r in text_rows}
    descriptive_df = pd.read_csv(descriptive_path) if descriptive_path.exists() else pd.DataFrame()
    descriptive_rows = descriptive_df.to_dict("records")
    descriptive_by_pair = {(r["file"], r["model"]): r["response_text"] for r in descriptive_rows}

    people_by_pair = {}
    latency_by_pair = {}
    for row in count_rows:
        key = (row["file"], row["model"])
        latency_by_pair[key] = row.get("latency") if not pd.isna(row.get("latency")) else None
        pc = row["person_count"]
        if pd.isna(pc):
            people_by_pair[key] = None
            continue
        if int(pc) == 0 or items_df.empty:
            people_by_pair[key] = []
            continue
        sub = items_df[(items_df["file"] == row["file"]) & (items_df["model"] == row["model"])].sort_values("person_idx")
        people_by_pair[key] = [{item: bool(r[item]) for item in CHECKLIST_ITEMS} for _, r in sub.iterrows()]
    return (count_rows, item_rows, text_rows, descriptive_rows,
            people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair)


# ---------------------------------------------------------------------------
# shared rendering
# ---------------------------------------------------------------------------

# SERIES is [person, helmet, vest, gloves, boots, no-helmet, no-vest, no-gloves, no-boots] —
# the first N_POSITIVE are "present" counts, the rest are "absent" counts. Used to chunk
# the wide column list into two labeled groups (Gestalt grouping: related columns read as
# one unit, not 9 flat items) rather than adding a 10th visual meaning on top of the
# green/red accuracy color already carried by each cell.
N_POSITIVE = 1 + len(CHECKLIST_ITEMS)
_DIVIDER = "border-left:2px solid #9B9D97;"


def _cell_style(base, series=None):
    divider = _DIVIDER if series == SERIES[N_POSITIVE] else ""
    return base.format(divider=divider)


def _num_cell(value, series=None):
    return _cell_style('<td style="padding:3px 8px;text-align:right;border-bottom:1px solid #E4E5E2;{divider}">'
                        + f"{value}</td>", series)


def _plain_cell(value, series=None):
    """Centered, uncolored — for the ground-truth row, which isn't being
    judged against anything so it never gets the green/red diff treatment."""
    return _cell_style('<td style="padding:3px 8px;text-align:center;border-bottom:1px solid #E4E5E2;{divider}">'
                        + f"{value}</td>", series)


def _grouped_head(extra_cols=()):
    """Two-row header: a top row labeling the PRESENT vs. ABSENT column
    groups (so the 9 class columns read as two chunks, not nine unrelated
    ones — the single biggest lever for scanning a wide table quickly),
    then the actual column names. `extra_cols` (total, latency, ...) get a
    blank top cell — they aren't part of either group."""
    n_negative = len(SERIES) - N_POSITIVE
    group_row = (
        "<tr>"
        '<th style="border:none"></th>'
        f'<th colspan="{N_POSITIVE}" style="padding:1px 4px;text-align:center;font-size:9.5px;'
        f'letter-spacing:.5px;color:{MUTED};font-weight:600">PRESENT</th>'
        f'<th colspan="{n_negative}" style="padding:1px 4px;text-align:center;font-size:9.5px;'
        f'letter-spacing:.5px;color:{MUTED};font-weight:600;{_DIVIDER}">ABSENT</th>'
        + "".join('<th style="border:none"></th>' for _ in extra_cols)
        + "</tr>"
    )
    name_row = (
        '<tr style="border-bottom:1px solid #9B9D97">'
        '<th style="padding:3px 8px;text-align:left">model</th>'
        + "".join(f'<th style="padding:3px 6px;text-align:center;{_DIVIDER if s == SERIES[N_POSITIVE] else ""}">'
                  f"{s}</th>" for s in SERIES)
        + "".join(f'<th style="padding:3px 8px;text-align:center">{c}</th>' for c in extra_cols)
        + "</tr>"
    )
    return group_row + name_row


def _hex_to_rgb(hexstr):
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i:i + 2], 16) for i in (0, 2, 4))


_GREEN_RGB = _hex_to_rgb(POSITIVE_GREEN)
_RED_RGB = _hex_to_rgb(NEGATIVE_RED)


def _diff_bg(count, reference):
    """Green when count matches the reference (ground truth) exactly,
    sliding to red the further off it is — a real gradient, not a fixed
    palette, so "how wrong" is visible at a glance instead of just
    "wrong/not wrong". None (either side missing — parse failure, or no
    ground truth for this class) means "can't judge," not "bad": no color.

    Error is relative to the reference (abs(diff) / max(reference, 1)), not
    a fixed count — a single image's reference is small (0-5ish), so being
    off by 1 there is a big, dramatic miss; an aggregate table's reference
    is a sum across every image (dozens+), so the same absolute diff of 1
    is trivial. One formula reads correctly at both scales without the
    caller needing to know which table it's in."""
    if count is None or reference is None:
        return None
    t = min(abs(count - reference) / max(reference, 1), 1.0)
    r, g, b = (round(_GREEN_RGB[i] + (_RED_RGB[i] - _GREEN_RGB[i]) * t) for i in range(3))
    return f"rgb({r},{g},{b})"


def _count_cell(count, reference, series=None):
    """The actual number, not dots — a cluster of 5+ dots reads as "a
    blob," a number reads instantly. Colored by how close it is to the
    ground truth (see _diff_bg()) so accuracy is visible without reading
    two cells and doing the subtraction yourself."""
    if count is None:
        return _cell_style('<td style="padding:3px 8px;text-align:center;border-bottom:1px solid #E4E5E2;'
                            f'color:{MUTED};' + '{divider}">—</td>', series)
    bg = _diff_bg(count, reference)
    if bg is None:
        return _cell_style('<td style="padding:3px 8px;text-align:center;border-bottom:1px solid #E4E5E2;{divider}">'
                            + f"{count}</td>", series)
    return _cell_style(
        '<td style="padding:3px 8px;text-align:center;border-bottom:1px solid #E4E5E2;{divider}'
        f'background:{bg};color:#FFFFFF;font-weight:600">{count}</td>', series,
    )


def _fmt_latency(seconds):
    if seconds is None:
        return "—"
    return f"{seconds:.1f}s" if seconds >= 1 else f"{seconds * 1000:.0f}ms"


def _latency_cell(seconds, max_seconds):
    """A filled bar, not a bare number — latency reads as a length (a
    pre-attentive visual cue) instead of requiring you to compare digits
    across rows. Scaled to the slowest model in *this* card, so the bar is
    meaningful relative to what's actually being compared right now."""
    if seconds is None:
        return '<td style="padding:3px 8px;border-bottom:1px solid #E4E5E2"><span style="color:' + MUTED + '">—</span></td>'
    pct = min(seconds / max_seconds, 1.0) * 100 if max_seconds else 0
    return (
        '<td style="padding:3px 8px;border-bottom:1px solid #E4E5E2">'
        f'<div style="position:relative;background:#E4E5E2;height:14px;width:64px">'
        f'<div style="position:absolute;inset:0;width:{pct:.0f}%;background:{INK}"></div>'
        f'<span style="position:relative;font-size:9.5px;color:{"#FFFFFF" if pct > 45 else INK};'
        f'padding-left:4px;line-height:14px;white-space:nowrap">{_fmt_latency(seconds)}</span>'
        "</div></td>"
    )


def _response_html(raw_text, counts, descriptive_text, show_json, show_descriptive):
    """Rendered OUTSIDE the table (see render_file_card) so toggling a
    response on never reflows the counts table above it — a wide <pre>
    block inside a <td> would otherwise force that whole column wider for
    every row. Nothing renders at all unless its toggle is on."""
    parts = []
    if show_json and raw_text:  # YOLO has none — it answers from boxes, not text
        parsed_ok = counts is not None
        label = "json" if parsed_ok else "⚠ unparsed"
        parts.append(f'<div style="margin:2px 0"><span style="color:{MUTED};font-size:10px">{label}:</span> '
                      f'<code style="font-size:10px">{html.escape(raw_text)}</code></div>')
    if show_descriptive and descriptive_text:
        parts.append(f'<div style="margin:2px 0;font-size:10.5px">{html.escape(descriptive_text)}</div>')
    return "".join(parts)


def render_file_card(feed, model_names, file_stem, buf, gt_counts, show_json, show_descriptive):
    """Real HTML <table>, one row per model plus a final "ground truth"
    row — not a text caption under the thumbnail — columns chunked into
    PRESENT/ABSENT groups (see _grouped_head()), each a number colored
    green -> red by how far it is from ground truth (_count_cell()), a
    total (diffed the same way), and latency as a filled bar
    (_latency_cell()). Text responses render below the table, not in a
    cell (see _response_html()), so toggling them never reflows the
    columns above."""
    with feed:
        cols = st.columns([1, 3])
        img_path = image_path_for(file_stem)
        if img_path:
            gt_boxes = load_gt_boxes(file_stem)
            cols[0].image(draw_gt_overlay(img_path, gt_boxes) if gt_boxes else str(img_path), width="stretch")
        with cols[1]:
            st.markdown(f'<div class="hv-mono" style="font-size:11px;color:{MUTED};margin-bottom:4px">{file_stem}</div>',
                        unsafe_allow_html=True)

            gt = (gt_counts or {}).get(file_stem, {s: 0 for s in SERIES})
            gt_total = sum(gt.values())
            max_latency = max((buf[n]["latency"] for n in model_names if buf.get(n) and buf[n].get("latency")),
                               default=None)

            head = _grouped_head(["total", "latency"])
            body_rows = []
            response_blocks = []
            for name in model_names:
                entry = buf.get(name)
                counts = entry["counts"] if entry else None
                raw_text = entry.get("raw_text") if entry else None
                descriptive_text = entry.get("descriptive_text") if entry else None
                latency = entry.get("latency") if entry else None
                cells = "".join(_count_cell(counts[s] if counts else None, gt[s], series=s) for s in SERIES)
                total = sum(counts.values()) if counts else None
                body_rows.append(
                    f'<tr><td style="padding:3px 8px;border-bottom:1px solid #E4E5E2">{model_chip(name)}</td>'
                    f'{cells}{_count_cell(total, gt_total)}{_latency_cell(latency, max_latency)}</tr>'
                )
                resp = _response_html(raw_text, counts, descriptive_text, show_json, show_descriptive)
                if resp:
                    response_blocks.append(f'<div style="margin-top:2px"><b>{name}</b>{resp}</div>')

            gt_cells = "".join(_plain_cell(gt[s], series=s) for s in SERIES)
            body_rows.append(
                f'<tr style="background:#F0F1EC;font-weight:600">'
                f'<td style="padding:3px 8px">ground truth</td>{gt_cells}'
                f'{_plain_cell(gt_total)}{_plain_cell("—")}</tr>'
            )

            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;font-size:11.5px">'
                f'<thead>{head}</thead><tbody>{"".join(body_rows)}</tbody></table>',
                unsafe_allow_html=True,
            )
            if response_blocks:
                st.markdown(
                    f'<div style="margin-top:6px;padding:8px 10px;background:#FFFFFF;border:1px solid {FAINT};'
                    f'font-size:11px;line-height:1.5">' + "".join(response_blocks) + "</div>",
                    unsafe_allow_html=True,
                )
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


def run_checklist_live(run_dir, run_name, model_names, sampled_files, adapters, seed, gt_counts,
                        count_rows, item_rows, text_rows, descriptive_rows,
                        people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair, skip_pairs,
                        show_json, show_descriptive):
    counts_path = run_dir / "person_counts.csv"
    items_path = run_dir / "person_items.csv"
    text_path = run_dir / "raw_responses.csv"
    descriptive_path = run_dir / "descriptive_responses.csv"

    progress_bar = st.progress(0.0)
    status_line = st.empty()
    st.button(
        "⏸ Pause run", key="checklist_pause_btn",
        help="Stops now — the checkpoint already on disk is safe. Pick it back up from PAUSED RUNS above.",
    )
    st.markdown('<div class="hv-h1" style="font-size:15px;margin:16px 0 6px">RESULTS (live)</div>',
                unsafe_allow_html=True)
    # show_json/show_descriptive come in as params, not widgets declared here —
    # the toggles live at module top-level (see the block above the run/resume
    # branch below) since any widget click reruns the whole script and would
    # otherwise wipe this function's output before it ever runs again.
    feed = st.container()

    # Replay whatever this run already has on disk (a resume) before the
    # live loop continues — same file-boundary logic as llm_comparison.py: a
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
            buf = {m: {"counts": model_counts_for_people(people_by_pair.get((file_stem, m))),
                       "raw_text": text_by_pair.get((file_stem, m)),
                       "descriptive_text": descriptive_by_pair.get((file_stem, m)),
                       "latency": latency_by_pair.get((file_stem, m))}
                   for m in covered[file_stem]}
            if covered[file_stem] == set(model_names):
                render_file_card(feed, model_names, file_stem, buf, gt_counts, show_json, show_descriptive)
            else:
                current_file, file_buf = file_stem, buf

    t_start = time.perf_counter()
    for step in run_checklist_steps(adapters, sampled_files, image_path_for=image_path_for,
                                     skip_pairs=skip_pairs, descriptive_prompt=DESCRIPTIVE_PROMPT):
        if step["skipped"]:
            continue
        if step["resumed"]:
            progress_bar.progress(step["done"] / step["total"])
            continue

        if step["file"] != current_file:
            if current_file is not None:
                render_file_card(feed, model_names, current_file, file_buf, gt_counts, show_json, show_descriptive)
            current_file, file_buf = step["file"], {}

        people = step["people"]
        key = (step["file"], step["model"])
        people_by_pair[key] = people
        latency_by_pair[key] = step["latency"]
        count_rows.append({"file": step["file"], "model": step["model"],
                            "person_count": len(people) if people is not None else None,
                            "latency": step["latency"]})
        if people:
            for idx, p in enumerate(people):
                item_rows.append({"file": step["file"], "model": step["model"], "person_idx": idx, **p})
        if step["raw_text"]:
            text_by_pair[key] = step["raw_text"]
            text_rows.append({"file": step["file"], "model": step["model"], "response_text": step["raw_text"]})
        if step["descriptive_text"]:
            descriptive_by_pair[key] = step["descriptive_text"]
            descriptive_rows.append({"file": step["file"], "model": step["model"], "response_text": step["descriptive_text"]})

        file_buf[step["model"]] = {
            "counts": model_counts_for_people(people), "raw_text": step["raw_text"],
            "descriptive_text": step["descriptive_text"], "latency": step["latency"],
        }

        done, total = step["done"], step["total"]
        progress_bar.progress(done / total)
        per_model = " · ".join(f"{n}:{c}/{len(sampled_files)}" for n, c in step["done_per_model"].items())
        status_line.markdown(f"**{done}/{total}** pairs · {time.perf_counter() - t_start:.0f}s elapsed — {per_model}")

        if done % 10 == 0 or done == total:  # checkpoint: survive a paused/killed tab
            pd.DataFrame(count_rows).to_csv(counts_path, index=False)
            pd.DataFrame(item_rows).to_csv(items_path, index=False)
            if text_rows:
                pd.DataFrame(text_rows).to_csv(text_path, index=False)
            if descriptive_rows:
                pd.DataFrame(descriptive_rows).to_csv(descriptive_path, index=False)

    if current_file is not None:
        render_file_card(feed, model_names, current_file, file_buf, gt_counts, show_json, show_descriptive)

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
    if text_rows:
        pd.DataFrame(text_rows).to_csv(final_dir / "raw_responses.csv", index=False)
    if descriptive_rows:
        pd.DataFrame(descriptive_rows).to_csv(final_dir / "descriptive_responses.csv", index=False)
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
        for p in past_runs:
            pc1, pc2 = st.columns([5, 1])
            pc1.markdown(
                f'<span class="hv-mono" style="font-size:12px">{p["run"]}</span> — '
                f'{p["created_at"]} · {p["images"]} images · {p["models"]}',
                unsafe_allow_html=True,
            )
            if pc2.button("🗑 Delete", key=f"delete_past_{p['run']}"):
                delete_run(p["run_dir"])
                st.rerun()

paused_runs = list_paused_checklist_runs()
resume_clicked = None
if paused_runs:
    st.markdown(f'<div class="hv-h1" style="font-size:15px;margin:14px 0 6px">PAUSED RUNS ({len(paused_runs)})</div>',
                unsafe_allow_html=True)
    for p in paused_runs:
        rc1, rc2, rc3 = st.columns([4, 1, 1])
        rc1.markdown(
            f'<span class="hv-mono" style="font-size:12px">{p["run_name"]}</span> — '
            f'{p["done_pairs"]}/{p["total_pairs"]} pairs done ({", ".join(p["models"])})',
            unsafe_allow_html=True,
        )
        if rc2.button("▶ Resume", key=f"resume_{p['run_name']}"):
            resume_clicked = p
        if rc3.button("🗑 Delete", key=f"delete_paused_{p['run_name']}"):
            delete_run(p["run_dir"])
            st.rerun()
    st.divider()

# ---------------------------------------------------------------------------
# response toggles — top-level, not inside run_checklist_live(): ANY widget
# click reruns this whole script from scratch, and if these lived inside
# the live-run function they (and everything they control) would vanish
# the instant you touched one, since neither "submitted" nor "resume"
# would be true on that rerun. Living here instead means flipping a toggle
# after a run finishes redraws the SAME completed run from session_state
# (see the "redraw" branch below) instead of losing it.
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:15px;margin:14px 0 2px">RESPONSES</div>', unsafe_allow_html=True)
st.caption("Hidden by default — flip either on to see it in every card below, no per-row clicking.")
toggle_col1, toggle_col2 = st.columns(2)
show_json = toggle_col1.toggle("Show JSON responses", value=False, key="checklist_show_json")
show_descriptive = toggle_col2.toggle("Show descriptions", value=False, key="checklist_show_descriptive")

# ---------------------------------------------------------------------------
# resume path, fresh-run config form, or redraw the last completed run
# (only reached when this rerun is neither of those — e.g. a toggle click)
# ---------------------------------------------------------------------------

ran_this_load = False

if resume_clicked is not None:
    run_dir = resume_clicked["run_dir"]
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model_names = cfg["model_names"]
    sampled_files = cfg["sampled_files"]
    seed = cfg.get("seed")
    gt_counts = load_gt_counts(tuple(sampled_files))

    args = build_args(model_names)
    adapters = load_adapters(model_names, args)
    (count_rows, item_rows, text_rows, descriptive_rows,
     people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair) = load_existing_checklist_rows(run_dir)
    skip_pairs = set(people_by_pair.keys())

    st.write(f"Resuming **{run_dir.name}** — {len(skip_pairs)}/{len(sampled_files) * len(model_names)} pairs already done.")
    run_checklist_live(run_dir, run_dir.name, model_names, sampled_files, adapters, seed, gt_counts,
                        count_rows, item_rows, text_rows, descriptive_rows,
                        people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair, skip_pairs,
                        show_json, show_descriptive)
    ran_this_load = True
else:
    with st.form("checklist_run_config"):
        c1, c2 = st.columns(2)
        n_images = c1.slider("Images to sample", 1, 100, 20)
        seed = c2.number_input("Seed", value=7, step=1)  # != 42, so a run here isn't just re-sampling the same images
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
    elif not model_names:
        st.error("Pick at least one model.")
    elif cloud_in_selection and not include_cloud:
        st.error(f"{cloud_in_selection} need “Allow cloud models” checked — that's a deliberate cost guard.")
    else:
        sampled_files = sample_test_images(n_images, seed)
        st.write(f"Sampled **{len(sampled_files)}** test images.")
        gt_counts = load_gt_counts(tuple(sampled_files))

        args = build_args(model_names)
        adapters = load_adapters(model_names, args)

        run_name = build_run_name(n_images, seed, model_names)
        run_dir = LLM_RUNS_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        # Written before the loop starts, not after — this is what a paused
        # run needs to be resumable at all (see list_paused_checklist_runs()).
        (run_dir / "run_config.json").write_text(json.dumps(
            {"kind": "checklist", "model_names": model_names, "sampled_files": sampled_files,
             "seed": seed, "n_images": n_images},
            indent=2,
        ))
        count_rows, item_rows, text_rows, descriptive_rows = [], [], [], []
        people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair = {}, {}, {}, {}
        run_checklist_live(run_dir, run_name, model_names, sampled_files, adapters, seed, gt_counts,
                            count_rows, item_rows, text_rows, descriptive_rows,
                            people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair, set(),
                            show_json, show_descriptive)
        ran_this_load = True

if ran_this_load:
    # Stashed so a later rerun that ISN'T a new submission/resume — a toggle
    # flip above is the main case — can redraw this exact run instead of
    # losing it (see the "redraw" branch below).
    st.session_state["checklist_last_run"] = {
        "model_names": model_names, "sampled_files": sampled_files, "gt_counts": gt_counts,
        "people_by_pair": people_by_pair, "text_by_pair": text_by_pair,
        "descriptive_by_pair": descriptive_by_pair, "latency_by_pair": latency_by_pair,
    }
elif "checklist_last_run" in st.session_state:
    saved = st.session_state["checklist_last_run"]
    model_names, sampled_files, gt_counts = saved["model_names"], saved["sampled_files"], saved["gt_counts"]
    people_by_pair, text_by_pair = saved["people_by_pair"], saved["text_by_pair"]
    descriptive_by_pair, latency_by_pair = saved["descriptive_by_pair"], saved["latency_by_pair"]
    st.markdown('<div class="hv-h1" style="font-size:15px;margin:16px 0 6px">RESULTS (last completed run)</div>',
                unsafe_allow_html=True)
    feed = st.container()
    for f in sampled_files:
        buf = {m: {"counts": model_counts_for_people(people_by_pair.get((f, m))),
                   "raw_text": text_by_pair.get((f, m)),
                   "descriptive_text": descriptive_by_pair.get((f, m)),
                   "latency": latency_by_pair.get((f, m))}
               for m in model_names}
        render_file_card(feed, model_names, f, buf, gt_counts, show_json, show_descriptive)
else:
    st.stop()

# ---------------------------------------------------------------------------
# analysis — live, right below, common to both the resume and fresh paths
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:20px;margin:28px 0 6px">ANALYSIS</div>', unsafe_allow_html=True)

# gt_counts was already computed above (before the live loop, so the per-image
# tables could use it too) — reused here, not recomputed.
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
flag_rate = n_flagged / len(sampled_files) if sampled_files else 0.0
st.markdown(
    '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:22px">'
    + stat_tile("IMAGES SCORED", len(sampled_files), "test images in this run")
    + stat_tile("PERSON-COUNT FLAGGED", n_flagged,
                "images where model consensus > label's person count", bg="#FFFFFF", fg=INK, border=FAINT)
    + stat_tile("FLAGGED RATE", f"{flag_rate:.0%}",
                "how often the label likely undercounts people",
                bg="#EFE600" if n_flagged else "#FFFFFF", fg=INK, border=None if n_flagged else FAINT)
    + "</div>",
    unsafe_allow_html=True,
)

if n_flagged:
    st.markdown('<div class="hv-h1" style="font-size:18px;margin-bottom:2px">FLAGGED IMAGES</div>', unsafe_allow_html=True)
    st.caption("Label undercounts people vs. model consensus — the label count is likely wrong, not every model at once.")
    flagged_view = person_rows[person_rows["flagged"]][["file", "gt", "consensus", "effective_gt"]].rename(
        columns={"gt": "label person count", "consensus": "model consensus", "effective_gt": "effective (used below)"}
    )
    st.dataframe(
        flagged_view, hide_index=True, width="stretch",
        column_config={
            "label person count": st.column_config.NumberColumn(),
            "model consensus": st.column_config.NumberColumn(),
            "effective (used below)": st.column_config.NumberColumn(help="max(label, consensus) — what's actually scored against"),
        },
    )

    st.markdown('<div class="hv-h1" style="font-size:18px;margin:18px 0 2px">WHICH SOURCES IT\'S PRONE TO</div>',
                unsafe_allow_html=True)
    st.caption("Filename prefix before \"__\" identifies the original dataset a test image came from.")
    person_rows = person_rows.copy()
    person_rows["source"] = person_rows["file"].str.split("__").str[0]
    by_source = (
        person_rows.groupby("source").agg(images=("file", "count"), flagged=("flagged", "sum"))
        .assign(flag_rate=lambda d: d["flagged"] / d["images"])
        .sort_values("flag_rate", ascending=False)
    )
    st.dataframe(
        by_source, width="stretch",
        column_config={"flag_rate": st.column_config.ProgressColumn("flag rate", format="%.0f%%", min_value=0, max_value=1)},
    )
else:
    st.caption("No images where model consensus exceeded the label's person count in this run.")

st.markdown("<hr style='margin:24px 0 18px'/>", unsafe_allow_html=True)

# --- results & ground truth, one table: rows = model (+ ground truth), columns = class --------
model_order = sorted(model_names, key=lambda m: (m != "yolo", m))
st.markdown('<div class="hv-h1" style="font-size:20px;margin-bottom:2px">RESULTS vs. GROUND TRUTH</div>', unsafe_allow_html=True)
st.caption(
    "Total count per class, summed across every scored image — each model's own answer, plus the "
    "\"ground truth (effective)\" row (max of the label and the model consensus — see above). Green = "
    "matches ground truth exactly, sliding to red the further off it is."
)
gt_totals = {series: int(per_image.loc[per_image["series"] == series, "effective_gt"].sum()) for series in SERIES}
count_body = []
for m in model_order:
    row_totals = {s: int(per_image.loc[per_image["series"] == s, f"m_{m}"].dropna().sum()) for s in SERIES}
    cells = "".join(_count_cell(row_totals[s], gt_totals[s], series=s) for s in SERIES)
    count_body.append(f'<tr><td style="padding:3px 8px;border-bottom:1px solid #E4E5E2">{model_chip(m)}</td>{cells}</tr>')
count_body.append(
    '<tr style="background:#F0F1EC;font-weight:600"><td style="padding:3px 8px">ground truth (effective)</td>'
    + "".join(_plain_cell(gt_totals[s], series=s) for s in SERIES) + "</tr>"
)
st.markdown(
    f'<table style="width:100%;border-collapse:collapse;font-size:11.5px">'
    f'<thead>{_grouped_head()}</thead><tbody>{"".join(count_body)}</tbody></table>',
    unsafe_allow_html=True,
)

# --- per-model, per-class accuracy against the ENHANCED ground truth --------------------------
st.markdown('<div class="hv-h1" style="font-size:20px;margin:22px 0 2px">COUNT ACCURACY PER MODEL</div>', unsafe_allow_html=True)
st.caption(
    "Scored per image against the enhanced ground truth above, then averaged. Exact-match rate: how "
    "often the model's count is exactly right — green is better. Mean abs. error: average "
    "|model count − ground truth| — green is better here too (a small error). Only images "
    "the model actually answered (no parse failure) count toward either."
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
    exact_table = macro.pivot(index="model", columns="series", values="exact_match_rate").reindex(index=model_order, columns=SERIES)
    error_table = macro.pivot(index="model", columns="series", values="mean_abs_error").reindex(index=model_order, columns=SERIES)

    st.markdown('<div class="hv-h1" style="font-size:15px;margin:16px 0 2px">EXACT-MATCH RATE (green = better)</div>',
                unsafe_allow_html=True)
    st.dataframe(exact_table.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format("{:.2f}"), width="stretch")

    st.markdown('<div class="hv-h1" style="font-size:15px;margin:20px 0 2px">MEAN ABSOLUTE COUNT ERROR (green = better)</div>',
                unsafe_allow_html=True)
    st.dataframe(error_table.style.background_gradient(cmap="RdYlGn_r", vmin=0).format("{:.2f}"), width="stretch")

    with st.expander("Full numbers (long form, with sample sizes)"):
        st.dataframe(macro, hide_index=True, width="stretch")
