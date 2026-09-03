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

import base64
import html
import io
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
import streamlit.components.v1 as components
import yaml
from PIL import Image

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
from model_adapters import ADAPTERS, CHECKLIST_ITEMS, DEFAULT_PROMPT_TEMPLATE, Detection, YOLOAdapter, _norm_class_name  # noqa: E402

import detector as det  # noqa: E402 — streamlit_app/ is already on sys.path (this page lives in pages/)
import view_helpers as vh

# "Our previous model weights" — every trained run detector.py already curates
# as comparable (a working Person class + at least one PPE slot, real
# vocabulary, not a throwaway smoke test). ALTEC_WEIGHTS is deliberately
# excluded: no Person class at all, so it can't drive this page's per-person
# pipeline (see detector.ALTEC_NO_PERSON_NOTE). Reused rather than re-scanning
# runs/detect/ ourselves — detector.py is already the one place that decides
# which runs are worth showing vs. experimental noise.
YOLO_WEIGHT_CHOICES = [
    ("yolo-v8-pretrained-100e", det.V8_LABEL, det.V8_WEIGHTS),
    ("yolo-css-100e", det.V26_LABEL, det.V26_WEIGHTS),
    ("yolo-merged-100e", det.MERGED_LABEL, det.MERGED_WEIGHTS),
    ("yolo-merged-m-150e", det.MERGED_M_LABEL, det.MERGED_M_WEIGHTS),
    ("yolo-mergedpeople-150e", det.MERGEDPEOPLE_LABEL, det.MERGEDPEOPLE_WEIGHTS),
    ("yolo-supervisorv1-300e", det.SUPERVISOR_V1_LABEL, det.SUPERVISOR_V1_WEIGHTS),
    ("yolo-supervisorv4-300e", det.SUPERVISOR_V4_LABEL, det.SUPERVISOR_V4_WEIGHTS),
]
YOLO_WEIGHTS_BY_KEY = {key: path for key, _label, path in YOLO_WEIGHT_CHOICES}
YOLO_LABEL_BY_KEY = {key: label for key, label, _path in YOLO_WEIGHT_CHOICES}
# Short display name for table chips — the run's own folder name (e.g.
# "yolo26s_supervisorv4_300e"), already familiar from run history elsewhere
# in this app, rather than the full detector.py label or the raw multiselect key.
YOLO_SHORT_BY_KEY = {key: path.parent.parent.name for key, _label, path in YOLO_WEIGHT_CHOICES}

# Grouped, in the requested order, for the Models multiselect below: our
# previous YOLO weight runs, then locally-hosted LLMs, then API-based ones.
LOCAL_LLM_NAMES = [n for n, spec in ADAPTERS.items() if n != "yolo" and not spec["is_cloud"]]
CLOUD_LLM_NAMES = [n for n, spec in ADAPTERS.items() if spec["is_cloud"]]
MODEL_OPTIONS = [key for key, _label, _path in YOLO_WEIGHT_CHOICES] + LOCAL_LLM_NAMES + CLOUD_LLM_NAMES


def _is_yolo_name(name):
    """True for any YOLO weight choice — the new per-run keys, or the bare
    "yolo" a run started before this feature existed still has recorded in
    its run_config.json/manifest (kept resumable/reopenable, just not
    offered as a fresh choice — see MODEL_OPTIONS above)."""
    return name == "yolo" or name in YOLO_WEIGHTS_BY_KEY


def _yolo_weights_for(name):
    return DEFAULT_YOLO_WEIGHTS if name == "yolo" else YOLO_WEIGHTS_BY_KEY[name]


def _display_name(name):
    return "yolo" if name == "yolo" else YOLO_SHORT_BY_KEY.get(name, name)


def _group_rank(name):
    """0/1/2 — YOLO weight runs, then local LLMs, then API LLMs, matching
    MODEL_OPTIONS' order. Used to sort model_names for display wherever
    the order otherwise reflects raw selection order (e.g. the progress
    indicator) instead of this grouping."""
    if _is_yolo_name(name):
        return 0
    return 2 if ADAPTERS.get(name, {}).get("is_cloud") else 1

st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True)
st.markdown(vh.header_html("LLM vs YOLO (PERSON DETECTION)", "how many people, and what are they wearing?"),
            unsafe_allow_html=True)

INK = "#141414"
MUTED = "#71736D"
FAINT = "#C4C6C0"
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
    is_yolo = _is_yolo_name(name)
    bg = INK if is_yolo else "#FFFFFF"
    fg = "#FFFFFF" if is_yolo else INK
    border = INK if is_yolo else FAINT
    return (f'<span class="hv-mono" style="display:inline-block;font-size:10.5px;padding:3px 8px;'
            f'background:{bg};color:{fg};border:1px solid {border};margin:2px 4px 2px 0;'
            f'white-space:nowrap">{_display_name(name)}</span>')

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
    """dict[file][series] -> ground-truth box count, straight from each
    image's own label .txt (via load_gt_boxes(), which also normalizes raw
    class names) — no enhancement here, that happens per-image against the
    live model counts below. Deliberately not read from a separately-built
    labels_long.csv: that's a derived, gitignored artifact that can (and
    did) drift out of sync with the raw dataset after a re-export swapped
    in a different class vocabulary — reading the labels directly means
    there's nothing to regenerate or go stale."""
    if not (MERGED_ROOT / "test" / "labels").exists():
        return None
    counts_by_file = {}
    for f in sampled_files:
        counts = {s: 0 for s in SERIES}
        for class_name, _bbox in load_gt_boxes(f):
            if class_name in counts:
                counts[class_name] += 1
        counts_by_file[f] = counts
    return counts_by_file


def load_gt_boxes(file_stem):
    """[(class_name, (x1,y1,x2,y2) normalized)] straight from the label .txt
    for this test image — a mix of plain YOLO bbox and YOLO segmentation
    (variable-length polygon) lines in this dataset, the latter bounded to
    its own axis-aligned box. class_name normalized via
    model_adapters._norm_class_name() (hardhat -> helmet, "safety vest" ->
    vest, ...) so it lines up with SERIES regardless of which raw dataset
    export data/merged/data.yaml currently points at."""
    label_path = MERGED_ROOT / "test" / "labels" / f"{file_stem}.txt"
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cid, coords = int(parts[0]), [float(v) for v in parts[1:]]
        if len(coords) == 4:  # plain YOLO bbox: cx, cy, w, h
            cx, cy, bw, bh = coords
            bbox = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        else:  # YOLO segmentation polygon (variable-length x,y pairs) — bound it
            xs, ys = coords[0::2], coords[1::2]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        boxes.append((_norm_class_name(CLASS_NAMES[cid]), bbox))
    return boxes


def model_counts_for_people(people):
    """people: list[{"bbox":..., "items": [Detection, ...]}] or None (parse
    failure) -> dict[series] -> int, or None if the model's answer couldn't
    be read at all — kept distinct from a genuine "0 people" answer, which
    scores normally."""
    if people is None:
        return None
    n = len(people)
    counts = {"person": n}
    for item in CHECKLIST_ITEMS:
        present = sum(1 for p in people if any(d.class_name == item for d in p["items"]))
        counts[item] = present
        counts[f"no-{item}"] = n - present
    return counts


def _buf_for_file(file_stem, model_names, people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair):
    """The render_*_row() "buf" shape for one file, rebuilt on demand from
    the (file, model)-keyed dicts every render path already carries —
    shared by the resume-replay/redraw/Open-run call sites (previously
    each built this dict inline) and by the image-gallery dialog, which
    needs to be able to look up ANY sampled file, not just the one whose
    thumbnail was clicked."""
    return {m: {"counts": model_counts_for_people(people_by_pair.get((file_stem, m))),
                "people": people_by_pair.get((file_stem, m)),
                "raw_text": text_by_pair.get((file_stem, m)),
                "descriptive_text": descriptive_by_pair.get((file_stem, m)),
                "latency": latency_by_pair.get((file_stem, m))}
            for m in model_names}


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

    # person_items.csv is one row per (file, model, person_idx, slot) — new
    # schema, carrying class/confidence/bbox per slot instead of one boolean
    # column per item. A run checkpointed under the OLD schema (no "slot"
    # column) still opens: reconstructed with bare presence, no
    # confidence/bbox, same as a chat model that never mentioned that slot.
    old_schema = not items_df.empty and "slot" not in items_df.columns

    def _people_for(file_stem, model, n):
        if old_schema:
            sub = items_df[(items_df["file"] == file_stem) & (items_df["model"] == model)].sort_values("person_idx")
            return [{"bbox": None, "items": [Detection(item if r[item] else f"no-{item}", bool(r[item]), None, None)
                                              for item in CHECKLIST_ITEMS]}
                    for _, r in sub.iterrows()]
        sub = items_df[(items_df["file"] == file_stem) & (items_df["model"] == model)]
        people = []
        for idx in range(n):
            grp = sub[sub["person_idx"] == idx]
            pbbox = grp.iloc[0]["person_bbox"] if not grp.empty else None
            pbbox = json.loads(pbbox) if isinstance(pbbox, str) and pbbox else None
            items = []
            for item in CHECKLIST_ITEMS:
                row = grp[grp["slot"] == item]
                if row.empty:
                    items.append(Detection(f"no-{item}", False, None, None))
                    continue
                r = row.iloc[0]
                bbox = json.loads(r["bbox"]) if isinstance(r["bbox"], str) and r["bbox"] else None
                conf = r["confidence"]
                conf = float(conf) if conf not in ("", None) and not pd.isna(conf) else None
                items.append(Detection(r["class"], not str(r["class"]).startswith("no-"), conf,
                                        tuple(bbox) if bbox else None))
            people.append({"bbox": tuple(pbbox) if pbbox else None, "items": items})
        return people

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
        people_by_pair[key] = _people_for(row["file"], row["model"], int(pc))
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


# One-time CSS for the inline table thumbnails: a small image that stays
# inside the row (no forced height blowing up every row — the old fixed-
# height iframe's whole problem), plus a bigger copy of the SAME image,
# absolutely positioned and hidden until :hover — pure CSS, no JS, no
# scroll/drag handlers to fight with. Injected once, module-level, like
# vh.HV_STYLE_CSS below.
_THUMB_CSS = """
<style>
.hv-thumb-wrap { position:relative; display:inline-block; line-height:0; }
.hv-thumb-wrap img.hv-thumb { height:44px; width:auto; display:block; border:1px solid #C4C6C0; }
.hv-thumb-wrap .hv-thumb-big {
  display:none; position:absolute; z-index:200; left:0; top:0;
  width:480px; max-width:min(70vw, 480px); border:2px solid #141414; background:#111;
  box-shadow:0 8px 24px rgba(0,0,0,.35);
}
.hv-thumb-wrap:hover .hv-thumb-big { display:block; }
</style>
"""
st.markdown(_THUMB_CSS, unsafe_allow_html=True)


def _annotated_image_uri(img_path, boxes, max_side=900):
    """The source image with `boxes` (a flat list of {"bbox","color","label",
    "dashed"?}) drawn as SVG shapes over it, returned as a data: URI ready
    for a plain <img src=...> — SVG (not a rasterized copy) keeps the boxes
    crisp whether it's shown as a 44px thumbnail or blown up in the hover
    preview/gallery dialog, from the exact same URI.
    ponytail: rebuilds + re-embeds the JPEG on every render (no cache) — a
    tab isn't hammering this in a loop, so it's not worth memoizing yet."""
    img = Image.open(img_path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()

    font_size = max(w, h) * 0.024
    shapes = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        bx, by, bw, bh = x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h
        dash = ' stroke-dasharray="5 4"' if b.get("dashed") else ""
        shapes.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                      f'fill="none" stroke="{b["color"]}" stroke-width="2"{dash}/>')
        if b.get("label"):
            shapes.append(f'<text x="{bx:.1f}" y="{max(by - 4, font_size):.1f}" fill="{b["color"]}" '
                          f'font-size="{font_size:.0f}" font-family="monospace">{html.escape(b["label"])}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
           f'<image href="data:image/jpeg;base64,{b64}" width="{w}" height="{h}"/>{"".join(shapes)}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _thumb_html(uri):
    return f'<div class="hv-thumb-wrap"><img class="hv-thumb" src="{uri}"><img class="hv-thumb-big" src="{uri}"></div>'


def _gt_boxes_for_zoom(gt_boxes):
    """load_gt_boxes()'s [(class_name, bbox)] -> _zoom_image_html()'s box
    shape: green for a present slot, red for its "no-X" negative, neutral
    for "person" (not a compliance call). Anything outside SERIES (e.g.
    mask/no-mask on an 11-class export — not a tracked slot here) is
    dropped rather than mis-colored."""
    boxes = []
    for class_name, bbox in gt_boxes:
        if class_name not in SERIES:
            continue
        color = MUTED if class_name == "person" else (NEGATIVE_RED if class_name.startswith("no-") else POSITIVE_GREEN)
        boxes.append({"bbox": bbox, "color": color, "label": class_name})
    return boxes


def _person_boxes_for_zoom(people):
    """A model's people (see model_adapters.parse_person_checklist_json())
    -> _zoom_image_html()'s box shape: each person's own box dashed/neutral,
    each item green (present) or red (absent), labeled "class conf%"."""
    boxes = []
    for p in people or []:
        if p["bbox"]:
            boxes.append({"bbox": p["bbox"], "color": MUTED, "label": None, "dashed": True})
        for d in p["items"]:
            if d.bbox:
                label = f"{d.class_name} {d.confidence:.0%}" if d.confidence is not None else d.class_name
                boxes.append({"bbox": d.bbox, "color": POSITIVE_GREEN if d.present else NEGATIVE_RED, "label": label})
    return boxes


# Shared between the header-only table and every single-row table below —
# each row is its own <table> (an interactive per-row image can't live
# inside one <td> of a single shared <table>), so this is what keeps their
# columns lined up despite that.
_TABLE_COLGROUP = (
    "<colgroup>"
    '<col style="width:15%">'
    + "".join(f'<col style="width:{60 / len(SERIES):.2f}%">' for _ in SERIES)
    + '<col style="width:10%"><col style="width:15%">'
    + "</colgroup>"
)


def _row_table(row_tr):
    return (f'<table style="width:100%;border-collapse:collapse;font-size:11.5px">{_TABLE_COLGROUP}'
            f'<tbody>{row_tr}</tbody></table>')


def _card_head(container, file_stem):
    with container:
        st.markdown(f'<div class="hv-mono" style="font-size:11px;color:{MUTED};margin:10px 0 2px">{file_stem}</div>',
                    unsafe_allow_html=True)
        img_col, tbl_col = st.columns([1, 3])
        with img_col:
            st.caption("hover to enlarge · 🔍 to browse")
        with tbl_col:
            st.markdown(f'<table style="width:100%;border-collapse:collapse;font-size:11.5px">{_TABLE_COLGROUP}'
                        f'<thead>{_grouped_head(["total", "latency"])}</thead></table>', unsafe_allow_html=True)


def render_gt_row(container, file_stem, gt, gt_total):
    img_path = image_path_for(file_stem)
    gt_row = (f'<tr style="background:#F0F1EC;font-weight:600"><td style="padding:3px 8px">ground truth</td>'
              f'{"".join(_plain_cell(gt[s], series=s) for s in SERIES)}'
              f'{_plain_cell(gt_total)}{_plain_cell("—")}</tr>')
    with container:
        img_col, tbl_col = st.columns([1, 3])
        with img_col:
            if img_path:
                st.markdown(_thumb_html(_annotated_image_uri(img_path, _gt_boxes_for_zoom(load_gt_boxes(file_stem)))),
                             unsafe_allow_html=True)
                if st.button("🔍", key=f"gallery_{file_stem}_gt", help="Browse all images"):
                    st.session_state["checklist_gallery_click"] = (file_stem, "__gt__")
        with tbl_col:
            st.markdown(_row_table(gt_row), unsafe_allow_html=True)


def render_model_row(container, file_stem, name, entry, gt, gt_total, max_latency, show_json, show_descriptive):
    """Draws exactly ONE model's row (image + table) into `container` —
    entry=None (nothing back yet) renders a pending "—" row. Used both by
    render_file_card() below (a one-shot full render, `container` a plain
    st.container()) and by the live loop's per-row st.empty() placeholders
    — the latter is why this had to split out of render_file_card() at
    all: redrawing the WHOLE card (every row's image) on every single
    model's completion was O(n_models^2) total image re-transmission for
    one file, the likely cause of results visibly "blinking and
    disappearing" under load with many models selected. Redrawing just the
    one row that actually changed costs one image re-embed, not N."""
    img_path = image_path_for(file_stem)
    counts = entry["counts"] if entry else None
    people = entry.get("people") if entry else None
    raw_text = entry.get("raw_text") if entry else None
    descriptive_text = entry.get("descriptive_text") if entry else None
    latency = entry.get("latency") if entry else None
    cells = "".join(_count_cell(counts[s] if counts else None, gt[s], series=s) for s in SERIES)
    total = sum(counts.values()) if counts else None
    row_tr = (f'<tr><td style="padding:3px 8px;border-bottom:1px solid #E4E5E2">{model_chip(name)}</td>'
              f'{cells}{_count_cell(total, gt_total)}{_latency_cell(latency, max_latency)}</tr>')
    with container:
        img_col, tbl_col = st.columns([1, 3])
        with img_col:
            if img_path:
                st.markdown(_thumb_html(_annotated_image_uri(img_path, _person_boxes_for_zoom(people))),
                             unsafe_allow_html=True)
                if entry and st.button("🔍", key=f"gallery_{file_stem}_{name}", help="Browse all images"):
                    st.session_state["checklist_gallery_click"] = (file_stem, name)
        with tbl_col:
            st.markdown(_row_table(row_tr), unsafe_allow_html=True)
    return _response_html(raw_text, counts, descriptive_text, show_json, show_descriptive)


def render_file_card(container, model_names, file_stem, buf, gt_counts, show_json, show_descriptive):
    """One-shot full render — ground truth row FIRST, then each model,
    built from render_gt_row()/render_model_row() above. Used wherever a
    file's data is already complete and only needs drawing once (resume
    replay of an already-covered file, redraw from session_state, an
    opened past run) — for the LIVE, still-in-progress case see
    init_live_card()/update_live_row() instead, which update one row at a
    time rather than re-rendering everything on every model's result."""
    gt = (gt_counts or {}).get(file_stem, {s: 0 for s in SERIES})
    gt_total = sum(gt.values())
    max_latency = max((buf[n]["latency"] for n in model_names if buf.get(n) and buf[n].get("latency")),
                       default=None)

    _card_head(container, file_stem)
    with container:
        gt_container = st.container()
        row_containers = {name: st.container() for name in model_names}
        response_area = st.container()

    render_gt_row(gt_container, file_stem, gt, gt_total)
    response_blocks = []
    for name in model_names:
        resp = render_model_row(row_containers[name], file_stem, name, buf.get(name), gt, gt_total,
                                 max_latency, show_json, show_descriptive)
        if resp:
            response_blocks.append(f'<div style="margin-top:2px"><b>{name}</b>{resp}</div>')
    if response_blocks:
        with response_area:
            st.markdown(
                f'<div style="margin-top:6px;padding:8px 10px;background:#FFFFFF;border:1px solid {FAINT};'
                f'font-size:11px;line-height:1.5">' + "".join(response_blocks) + "</div>",
                unsafe_allow_html=True,
            )
    with container:
        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)


@st.dialog("Browse images", width="large")
def _show_gallery(model_names, sampled_files, gt_counts, people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair):
    """The image browser a thumbnail's 🔍 button opens — a real, native
    st.dialog (Streamlit >= 1.31) rather than a hand-rolled modal, since a
    true full-page overlay isn't something a components.html() iframe can
    do (it's clipped to its own box). Left/right buttons walk through
    every sampled file in the run, keeping whichever annotation view was
    clicked (ground truth, or one model) so you're comparing the same
    thing across images; the file's full result table is shown right
    below the image for context."""
    file_stem, model = st.session_state["checklist_gallery_click"]
    idx = sampled_files.index(file_stem) if file_stem in sampled_files else 0

    nav_l, nav_mid, nav_r, nav_close = st.columns([1, 5, 1, 1])
    if nav_l.button("←", key="gallery_prev", disabled=idx == 0):
        idx -= 1
    if nav_r.button("→", key="gallery_next", disabled=idx >= len(sampled_files) - 1):
        idx += 1
    if nav_close.button("✕", key="gallery_close"):
        del st.session_state["checklist_gallery_click"]
        st.rerun()
    file_stem = sampled_files[idx]
    st.session_state["checklist_gallery_click"] = (file_stem, model)
    label = "ground truth" if model == "__gt__" else _display_name(model)
    nav_mid.markdown(
        f'<div class="hv-mono" style="text-align:center;font-size:12px">{html.escape(file_stem)} — '
        f'{html.escape(label)} ({idx + 1}/{len(sampled_files)})</div>', unsafe_allow_html=True)

    img_path = image_path_for(file_stem)
    if img_path:
        boxes = (_gt_boxes_for_zoom(load_gt_boxes(file_stem)) if model == "__gt__"
                 else _person_boxes_for_zoom(people_by_pair.get((file_stem, model))))
        uri = _annotated_image_uri(img_path, boxes, max_side=1600)
        st.markdown(f'<img src="{uri}" style="display:block;width:100%;max-height:65vh;'
                    f'object-fit:contain;background:#111;margin-bottom:10px">', unsafe_allow_html=True)

    gt = (gt_counts or {}).get(file_stem, {s: 0 for s in SERIES})
    gt_total = sum(gt.values())
    buf = _buf_for_file(file_stem, model_names, people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair)
    max_latency = max((buf[n]["latency"] for n in model_names if buf.get(n) and buf[n].get("latency")), default=None)
    body_rows = [f'<tr style="background:#F0F1EC;font-weight:600"><td style="padding:3px 8px">ground truth</td>'
                 f'{"".join(_plain_cell(gt[s], series=s) for s in SERIES)}{_plain_cell(gt_total)}{_plain_cell("—")}</tr>']
    for m in sorted(model_names, key=lambda n: (_group_rank(n), n)):
        entry = buf.get(m)
        counts = entry["counts"] if entry else None
        total = sum(counts.values()) if counts else None
        cells = "".join(_count_cell(counts[s] if counts else None, gt[s], series=s) for s in SERIES)
        latency = entry.get("latency") if entry else None
        body_rows.append(f'<tr><td style="padding:3px 8px;border-bottom:1px solid #E4E5E2">{model_chip(m)}</td>'
                          f'{cells}{_count_cell(total, gt_total)}{_latency_cell(latency, max_latency)}</tr>')
    st.markdown(f'<table style="width:100%;border-collapse:collapse;font-size:11.5px">{_TABLE_COLGROUP}'
                f'<thead>{_grouped_head(["total", "latency"])}</thead><tbody>{"".join(body_rows)}</tbody></table>',
                unsafe_allow_html=True)

    # Best-effort arrow-key navigation: Streamlit has no keyboard-shortcut
    # API, so this reaches into the parent document (components.html runs
    # in its own iframe) and clicks the real Prev/Next buttons above by
    # their visible glyph. If a future Streamlit DOM change ever breaks
    # this, the visible buttons still work — only the shortcut is lost.
    components.html("""
    <script>
    (function() {
      const doc = window.parent.document;
      function findByText(t) { return Array.from(doc.querySelectorAll('button')).find(b => b.innerText.trim() === t); }
      doc.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowLeft') { const b = findByText('←'); if (b && !b.disabled) b.click(); }
        if (e.key === 'ArrowRight') { const b = findByText('→'); if (b && !b.disabled) b.click(); }
      });
    })();
    </script>
    """, height=0)


# Fixed reference for the LIVE per-row latency bar, since a row drawn on its
# own can't know the eventual slowest model the way render_file_card()'s
# one-shot max_latency does — accepted trade-off for not re-touching every
# already-drawn row every time a new completion changes the true max.
LIVE_LATENCY_CEILING = 60.0


def init_live_card(feed, model_names, file_stem, gt_counts):
    """First-touch setup for a file in the live loop: draws the title,
    shared header, and the ground-truth row (known upfront, drawn once),
    plus one empty per-model placeholder pre-filled with a pending row so
    the card looks complete immediately. Returns the state
    update_live_row() needs to fill in each row independently as that
    model's result arrives."""
    gt = (gt_counts or {}).get(file_stem, {s: 0 for s in SERIES})
    gt_total = sum(gt.values())
    _card_head(feed, file_stem)
    with feed:
        gt_container = st.container()
        row_placeholders = {name: st.empty() for name in model_names}
        response_ph = st.empty()
        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
    render_gt_row(gt_container, file_stem, gt, gt_total)
    for name in model_names:
        render_model_row(row_placeholders[name], file_stem, name, None, gt, gt_total,
                          LIVE_LATENCY_CEILING, False, False)
    return {"row_placeholders": row_placeholders, "response_ph": response_ph,
            "gt": gt, "gt_total": gt_total, "responses": {}}


def update_live_row(scaffold, file_stem, name, entry, show_json, show_descriptive):
    resp = render_model_row(scaffold["row_placeholders"][name], file_stem, name, entry,
                             scaffold["gt"], scaffold["gt_total"], LIVE_LATENCY_CEILING,
                             show_json, show_descriptive)
    scaffold["responses"][name] = resp
    blocks = [f'<div style="margin-top:2px"><b>{n}</b>{r}</div>' for n, r in scaffold["responses"].items() if r]
    with scaffold["response_ph"]:
        if blocks:
            st.markdown(
                f'<div style="margin-top:6px;padding:8px 10px;background:#FFFFFF;border:1px solid {FAINT};'
                f'font-size:11px;line-height:1.5">' + "".join(blocks) + "</div>",
                unsafe_allow_html=True,
            )


def _progress_html(model_names, done_per_model, total_per_model, in_flight_by_model):
    """One visual row per model — a small bar plus an icon (✅ done, ⏳
    actively running with elapsed time, ⌛ queued) — instead of one long
    wrapped line of "name:n/N" pairs, which stops being readable past a
    couple of models. `in_flight_by_model`: name -> (file, secs, n_pending)
    from the generator's ~1s "tick" events (see run_checklist_steps());
    empty right after a plain completion event, when the next tick (at
    most ~1s later) will have it again."""
    rows = []
    for name in sorted(model_names, key=lambda n: (_group_rank(n), n)):
        done_n = done_per_model.get(name, 0)
        flight = in_flight_by_model.get(name)
        if done_n >= total_per_model:
            icon, detail, bar_color = "✅", "done", POSITIVE_GREEN
        elif flight:
            f, secs, n_pending = flight
            extra = f" (+{n_pending - 1} more queued)" if n_pending > 1 else ""
            icon, detail, bar_color = "⏳", f"running on {f} · {secs:.0f}s{extra}", INK
        else:
            icon, detail, bar_color = "⌛", "queued", MUTED
        pct = (done_n / total_per_model * 100) if total_per_model else 0
        rows.append(
            '<div style="display:flex;align-items:center;gap:8px;font-size:11.5px;padding:2px 0">'
            f'<span style="width:18px">{icon}</span>'
            f'<span class="hv-mono" style="width:180px;flex-shrink:0">{html.escape(_display_name(name))}</span>'
            f'<div style="background:#E4E5E2;height:8px;width:110px;position:relative;flex-shrink:0">'
            f'<div style="position:absolute;inset:0;width:{pct:.0f}%;background:{bar_color}"></div></div>'
            f'<span style="color:{MUTED};font-size:10.5px;white-space:nowrap">{done_n}/{total_per_model} — '
            f'{html.escape(detail)}</span></div>'
        )
    return "".join(rows)


def _in_flight_by_model(in_flight):
    by_model = {}
    for f, n, secs in in_flight:
        cur = by_model.get(n)
        if cur is None:
            by_model[n] = [f, secs, 1]
        else:
            cur[2] += 1
            if secs > cur[1]:
                cur[0], cur[1] = f, secs
    return {n: tuple(v) for n, v in by_model.items()}


def build_args(model_names):
    args = SimpleNamespace(
        ollama_url="http://localhost:11434",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,  # unused by describe()/predict()-via-boxes, adapters still want it
    )
    for name in model_names:
        if not _is_yolo_name(name):
            setattr(args, f"{name.replace('-', '_')}_model", ADAPTERS[name]["default_model"])
    return args


def load_adapters(model_names, args):
    adapters = {}
    load_status = st.empty()
    for name in model_names:
        load_status.markdown(f"Loading `{name}`...")
        # Multiple YOLO weight choices can be selected at once, each its own
        # YOLOAdapter — bypasses compare_models.build_adapter()'s single
        # shared args.yolo_weights, which only ever points at one run.
        adapters[name] = YOLOAdapter(_yolo_weights_for(name)) if _is_yolo_name(name) else build_adapter(name, args)
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
    # live loop continues — a fully-covered file gets its final card now
    # (render_file_card — done, never changes again), a file caught
    # mid-way gets a live scaffold (init_live_card/update_live_row) so the
    # loop below merges into it. Keyed by file (not one "current file"
    # slot): run_checklist_steps() now fires every (file, model) pair at
    # once, so several files can be genuinely in progress at the same
    # time — each gets its own scaffold, updated independently as its
    # models finish, in whatever order they actually complete.
    file_bufs = {}
    file_scaffolds = {}
    if skip_pairs:
        covered = {}
        for f, m in skip_pairs:
            covered.setdefault(f, set()).add(m)
        for file_stem in sampled_files:
            if file_stem not in covered:
                continue
            buf = _buf_for_file(file_stem, model_names, people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair)
            if covered[file_stem] == set(model_names):
                render_file_card(feed, model_names, file_stem, buf, gt_counts, show_json, show_descriptive)
            else:
                file_bufs[file_stem] = buf
                scaffold = init_live_card(feed, model_names, file_stem, gt_counts)
                for name in covered[file_stem]:
                    update_live_row(scaffold, file_stem, name, buf[name], show_json, show_descriptive)
                file_scaffolds[file_stem] = scaffold

    t_start = time.perf_counter()
    for step in run_checklist_steps(adapters, sampled_files, image_path_for=image_path_for,
                                     skip_pairs=skip_pairs, descriptive_prompt=DESCRIPTIVE_PROMPT):
        if step.get("tick"):
            # Nothing finished in the last ~1s — still tell the user what's
            # actually in flight and for how long, instead of going quiet
            # (several Ollama-backed models share one lock and run one at a
            # time — see model_adapters.OllamaAdapter — so a multi-minute
            # gap between completions is normal, not a hang).
            status_line.markdown(
                f"**{step['done']}/{step['total']}** pairs · {time.perf_counter() - t_start:.0f}s elapsed<br>"
                + _progress_html(model_names, step["done_per_model"], len(sampled_files),
                                  _in_flight_by_model(step["in_flight"])),
                unsafe_allow_html=True,
            )
            continue
        if step["skipped"]:
            continue
        if step["resumed"]:
            progress_bar.progress(step["done"] / step["total"])
            continue

        file_stem = step["file"]
        if file_stem not in file_bufs:
            file_bufs[file_stem] = {}
            file_scaffolds[file_stem] = init_live_card(feed, model_names, file_stem, gt_counts)

        people = step["people"]
        key = (step["file"], step["model"])
        people_by_pair[key] = people
        latency_by_pair[key] = step["latency"]
        count_rows.append({"file": step["file"], "model": step["model"],
                            "person_count": len(people) if people is not None else None,
                            "latency": step["latency"]})
        if people:
            for idx, p in enumerate(people):
                pbbox = p["bbox"]
                for d in p["items"]:
                    item_rows.append({
                        "file": step["file"], "model": step["model"], "person_idx": idx,
                        "person_bbox": json.dumps(list(pbbox)) if pbbox else "",
                        "slot": d.class_name[3:] if d.class_name.startswith("no-") else d.class_name,
                        "class": d.class_name,
                        "confidence": d.confidence if d.confidence is not None else "",
                        "bbox": json.dumps(list(d.bbox)) if d.bbox else "",
                    })
        if step["raw_text"]:
            text_by_pair[key] = step["raw_text"]
            text_rows.append({"file": step["file"], "model": step["model"], "response_text": step["raw_text"]})
        if step["descriptive_text"]:
            descriptive_by_pair[key] = step["descriptive_text"]
            descriptive_rows.append({"file": step["file"], "model": step["model"], "response_text": step["descriptive_text"]})

        entry = {
            "counts": model_counts_for_people(people), "people": people, "raw_text": step["raw_text"],
            "descriptive_text": step["descriptive_text"], "latency": step["latency"],
        }
        file_bufs[file_stem][step["model"]] = entry
        # Redraw ONLY this one model's row, not the whole card — every model
        # runs in its own thread and yields the instant it finishes (see
        # run_checklist_steps()), so this is the point where that reaches
        # the screen without re-sending every OTHER already-drawn row's
        # image too (see render_model_row()'s docstring for why that matters).
        update_live_row(file_scaffolds[file_stem], file_stem, step["model"], entry, show_json, show_descriptive)

        done, total = step["done"], step["total"]
        progress_bar.progress(done / total)
        status_line.markdown(
            f"**{done}/{total}** pairs · {time.perf_counter() - t_start:.0f}s elapsed<br>"
            + _progress_html(model_names, step["done_per_model"], len(sampled_files), {}),
            unsafe_allow_html=True,
        )

        if done % 10 == 0 or done == total:  # checkpoint: survive a paused/killed tab
            pd.DataFrame(count_rows).to_csv(counts_path, index=False)
            pd.DataFrame(item_rows).to_csv(items_path, index=False)
            if text_rows:
                pd.DataFrame(text_rows).to_csv(text_path, index=False)
            if descriptive_rows:
                pd.DataFrame(descriptive_rows).to_csv(descriptive_path, index=False)

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
            pc1, pc2, pc3 = st.columns([5, 1, 1])
            pc1.markdown(
                f'<span class="hv-mono" style="font-size:12px">{p["run"]}</span> — '
                f'{p["created_at"]} · {p["images"]} images · {p["models"]}',
                unsafe_allow_html=True,
            )
            if pc2.button("Open", key=f"open_past_{p['run']}"):
                manifest = json.loads((p["run_dir"] / "run_manifest.json").read_text())
                model_names, sampled_files = manifest["models"], manifest["sampled_files"]
                (_, _, _, _, people_by_pair, text_by_pair, descriptive_by_pair,
                 latency_by_pair) = load_existing_checklist_rows(p["run_dir"])
                st.session_state["checklist_last_run"] = {
                    "model_names": model_names, "sampled_files": sampled_files,
                    "gt_counts": load_gt_counts(tuple(sampled_files)),
                    "people_by_pair": people_by_pair, "text_by_pair": text_by_pair,
                    "descriptive_by_pair": descriptive_by_pair, "latency_by_pair": latency_by_pair,
                }
                st.rerun()
            if pc3.button("🗑 Delete", key=f"delete_past_{p['run']}"):
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
# resume path or fresh-run config form — decides WHAT to run but doesn't call
# run_checklist_live() yet, so the response toggles below can render between
# "configure the run" and "see the results" instead of above both.
# ---------------------------------------------------------------------------

should_run = False

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
    should_run = True
else:
    with st.form("checklist_run_config"):
        c1, c2 = st.columns(2)
        n_images = c1.slider("Images to sample", 1, 100, 20)
        seed = c2.number_input("Seed", value=7, step=1)  # != 42, so a run here isn't just re-sampling the same images
        model_names = st.multiselect(
            "Models",
            MODEL_OPTIONS,
            default=["yolo-supervisorv4-300e", "ollama"],
            format_func=lambda k: YOLO_LABEL_BY_KEY.get(k, k),
            help="Grouped in order: our previous YOLO training runs (own boxes, no prompt — pick several "
                 "to compare weight versions directly), then locally-hosted LLMs (need Ollama running), "
                 "then API-based LLMs (calls a paid API). Local LLMs share one Ollama server and run one "
                 "at a time, not in parallel — picking several roughly sums their individual times.",
        )
        cloud_in_selection = [m for m in model_names if ADAPTERS.get(m, {}).get("is_cloud")]
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
        should_run = True

# ---------------------------------------------------------------------------
# response toggles — placed here (after the config/resume controls above,
# before any results below), not inside run_checklist_live(): ANY widget
# click reruns this whole script from scratch, and if these lived inside
# the live-run function they (and everything they control) would vanish
# the instant you touched one, since neither "should_run" nor a resume
# would be true on that rerun. Living here instead means flipping a toggle
# after a run finishes redraws the SAME completed run from session_state
# (see the "redraw" branch below) instead of losing it.
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:15px;margin:14px 0 2px">RESPONSES</div>', unsafe_allow_html=True)
st.caption("Hidden by default — flip either on to see it in every card below, no per-row clicking.")
toggle_col1, toggle_col2 = st.columns(2)
show_json = toggle_col1.toggle("Show JSON responses", value=False, key="checklist_show_json")
show_descriptive = toggle_col2.toggle("Show descriptions", value=False, key="checklist_show_descriptive")

ran_this_load = False
if should_run:
    if resume_clicked is not None:
        run_checklist_live(run_dir, run_dir.name, model_names, sampled_files, adapters, seed, gt_counts,
                            count_rows, item_rows, text_rows, descriptive_rows,
                            people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair, skip_pairs,
                            show_json, show_descriptive)
    else:
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
        buf = _buf_for_file(f, model_names, people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair)
        render_file_card(feed, model_names, f, buf, gt_counts, show_json, show_descriptive)
else:
    st.stop()

if "checklist_gallery_click" in st.session_state:
    _show_gallery(model_names, sampled_files, gt_counts, people_by_pair, text_by_pair, descriptive_by_pair, latency_by_pair)

# ---------------------------------------------------------------------------
# analysis — live, right below, common to both the resume and fresh paths
# ---------------------------------------------------------------------------

st.markdown('<div class="hv-h1" style="font-size:20px;margin:28px 0 6px">ANALYSIS</div>', unsafe_allow_html=True)

# gt_counts was already computed above (before the live loop, so the per-image
# tables could use it too) — reused here, not recomputed.
if gt_counts is None:
    st.warning("data/merged/test/labels/ not found on this checkout — can't score against ground truth "
               "(the dataset is git-ignored). The run above is still saved.")
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
model_order = sorted(model_names, key=lambda m: (not _is_yolo_name(m), m))
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
