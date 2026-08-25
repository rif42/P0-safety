"""HI-VIS — YOLO-based PPE detector: model loading, inference, and the
person/item matching + compliance-assessment logic shared by the demo page.

TO SWAP IN A DIFFERENT TRAINED RUN: change DEFAULT_WEIGHTS below (or set the
HIVIS_MODEL_PATH environment variable — no code edit needed for that path).
Everything else keeps working as long as the new weights were trained on a
dataset with classes named "Person", "Hardhat", "NO-Hardhat", "Safety Vest",
"NO-Safety Vest" (extra classes, e.g. Mask/Safety Cone/machinery/vehicle, are
simply ignored). If the new model also detects boots, add a "boots" /
"no-boots" mapping to _RAW_TO_KEY and TRACKED_CLASSES.
"""

import os
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = _APP_DIR.parent  # streamlit_app/.. == repo root

# ---------------------------------------------------------------------------
# Swap the model here. DEFAULT_WEIGHTS is the trained run this demo ships
# with; HIVIS_MODEL_PATH (and HIVIS_MODEL_LABEL) let you point at a different
# run without touching code, e.g.:
#   HIVIS_MODEL_PATH=runs/scratch/yolov8n_scratch/weights/best.pt streamlit run app.py
#
# V8_*/V26_* are named so pages that need BOTH runs at once (e.g. the model
# comparison page) can load them explicitly instead of only ever getting
# whichever one DEFAULT_WEIGHTS/HIVIS_MODEL_PATH currently points at.
V8_WEIGHTS = REPO_ROOT / "runs" / "pretrained_100e" / "weights" / "best.pt"
V8_LABEL = "YOLOv8n · pretrained_100e"
V26_WEIGHTS = REPO_ROOT / "runs" / "pretrained_v26" / "weights" / "best.pt"
V26_LABEL = "YOLO26s · pretrained_v26"  # different architecture from v8 (C3k2/C2PSA blocks, not C2f) — verified from the checkpoint itself

DEFAULT_WEIGHTS = V26_WEIGHTS
DEFAULT_LABEL = V26_LABEL

_weights_env = os.environ.get("HIVIS_MODEL_PATH")
WEIGHTS_PATH = (REPO_ROOT / _weights_env) if _weights_env else DEFAULT_WEIGHTS
MODEL_LABEL = os.environ.get("HIVIS_MODEL_LABEL", DEFAULT_LABEL if not _weights_env else _weights_env)

# Inference is run once per image at this low confidence floor; the UI's
# confidence-threshold slider then filters the cached raw detections in pure
# python, so moving the slider is instant and never re-runs the model. This
# must be <= the slider's minimum (0.10) or low-threshold results would be
# missing detections that were never kept at inference time.
BASE_CONF = 0.10
NMS_IOU = 0.5  # ultralytics' own per-class NMS at prediction time
DEDUPE_IOU = 0.6  # extra same-class merge to clean up near-duplicate boxes

# ---------------------------------------------------------------------------

CLASS_META = {
    "person":    {"label": "Person",         "color": "#8A8D90"},
    "hardhat":   {"label": "Hardhat",        "color": "#3D9BE9"},
    "nohardhat": {"label": "NO-Hardhat",     "color": "#FF4D42"},
    "vest":      {"label": "Safety Vest",    "color": "#2FBE6B"},
    "novest":    {"label": "NO-Safety Vest", "color": "#FF8A00"},
    "boots":     {"label": "Boots",          "color": "#B07CFF"},
}

# Boots has no entry here on purpose: this dataset/model has no boots class,
# so it's never detected — every person's boots status comes back "not
# visible" (see assess()), same as any other item the model can't see.
_RAW_TO_KEY = {
    "person": "person",
    # pretrained_100e's vocabulary:
    "hardhat": "hardhat",
    "no-hardhat": "nohardhat",
    "safety vest": "vest",
    "no-safety vest": "novest",
    # pretrained_v26's vocabulary — same four PPE states, different words.
    # Add any future run's own class names here rather than retraining to
    # match; this is the one place that needs to know about it.
    "helmet": "hardhat",
    "no-helmet": "nohardhat",
    "vest": "vest",
    "no-vest": "novest",
}
TRACKED_CLASSES = ["person", "hardhat", "nohardhat", "vest", "novest"]
UNTRACKED_NOTE = "Boots: not a class in this model's training data — always reported as not visible."


def _norm(name):
    return name.strip().lower()


_model_cache = {}


def load_model(weights_path=None):
    """Load (and cache) a YOLO model. Safe to call repeatedly — cheap after
    the first call for a given path. Returns None if the weights file is
    missing so callers can show a friendly message instead of crashing."""
    path = Path(weights_path) if weights_path else WEIGHTS_PATH
    key = str(path)
    if key in _model_cache:
        return _model_cache[key]
    if not path.exists():
        _model_cache[key] = None
        return None
    from ultralytics import YOLO
    model = YOLO(str(path))
    _model_cache[key] = model
    return model


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _containment(item_box, container_box):
    """Fraction of item_box's area that lies inside container_box. More
    forgiving than IoU for matching a small item (a hardhat) against a much
    bigger container (the person's full-body box)."""
    ix1, iy1, ix2, iy2 = item_box
    cx1, cy1, cx2, cy2 = container_box
    ox1, oy1 = max(ix1, cx1), max(iy1, cy1)
    ox2, oy2 = min(ix2, cx2), min(iy2, cy2)
    ow, oh = max(0.0, ox2 - ox1), max(0.0, oy2 - oy1)
    inter = ow * oh
    item_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / item_area if item_area > 0 else 0.0


def _dedupe(boxes, iou_thresh=DEDUPE_IOU):
    """Merge near-duplicate same-class boxes (common at a very low confidence
    floor), keeping the highest-confidence box of each cluster."""
    by_key = {}
    for b in boxes:
        by_key.setdefault(b["key"], []).append(b)
    out = []
    for key, group in by_key.items():
        group = sorted(group, key=lambda b: -b["conf"])
        kept = []
        for b in group:
            if any(_iou(b["box"], k["box"]) > iou_thresh for k in kept):
                continue
            kept.append(b)
        out.extend(kept)
    return out


def detect_raw(model, image):
    """Run inference once; return every kept detection above BASE_CONF as a
    plain dict with a normalized (0-1) xyxy box, so results are resolution-
    independent and easy to draw back onto any size of the same image."""
    results = model.predict(image, conf=BASE_CONF, iou=NMS_IOU, verbose=False)[0]
    w, h = image.size
    names = results.names
    boxes = []
    for b in results.boxes:
        cls_name = _norm(names[int(b.cls[0])])
        key = _RAW_TO_KEY.get(cls_name)
        if key is None:
            continue  # ignore Mask / NO-Mask / Safety Cone / machinery / vehicle
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        boxes.append({
            "key": key,
            "conf": float(b.conf[0]),
            "box": (x1 / w, y1 / h, x2 / w, y2 / h),
        })
    return _dedupe(boxes)


def assess(raw_boxes, threshold, required=("hardhat", "vest")):
    """Group raw detections into per-person PPE findings at a given
    confidence threshold — pure python, safe to call on every slider move.

    `required` controls which item(s) count toward the compliance verdict —
    e.g. pass ("hardhat",) to check hardhats only. Status is still computed
    for every tracked slot regardless, so switching `required` at the UI
    layer needs no re-inference, just a re-call of this function.

    Rule set: whichever of hardhat / hi-vis vest are in `required` (boots is
    always advisory and, on this model, untracked). A person is
    non-compliant when a required item has a NO-Hardhat or NO-Safety Vest
    box matched to them at or above the threshold.
    """
    persons_raw = [b for b in raw_boxes if b["key"] == "person"]
    items_raw = [b for b in raw_boxes if b["key"] != "person"]

    persons = [{"box": p["box"], "conf": p["conf"], "above": p["conf"] >= threshold, "items": {}}
               for p in persons_raw]

    for it in items_raw:
        best, best_score = None, 0.0
        for p in persons:
            score = _containment(it["box"], p["box"])
            if score > best_score:
                best, best_score = p, score
        if best is None or best_score < 0.3:
            continue  # doesn't sit inside any detected person — leave unassigned
        slot = "hardhat" if it["key"] in ("hardhat", "nohardhat") else "vest"
        current = best["items"].get(slot)
        if current is None or it["conf"] > current["conf"]:
            best["items"][slot] = {"key": it["key"], "conf": it["conf"], "box": it["box"]}

    out = []
    for p in persons:
        if not p["above"]:
            continue
        status = {}
        for slot in ("hardhat", "vest"):
            it = p["items"].get(slot)
            if it is not None and it["conf"] >= threshold:
                is_negative = it["key"] in ("nohardhat", "novest")
                status[slot] = {"state": "missing" if is_negative else "present",
                                 "conf": it["conf"], "box": it["box"], "class_key": it["key"]}
            else:
                status[slot] = {"state": "notvisible", "conf": None, "box": None, "class_key": None}
        status["boots"] = {"state": "notvisible", "conf": None, "box": None, "class_key": None}
        non_compliant = any(status[slot]["state"] == "missing" for slot in required)
        out.append({"box": p["box"], "conf": p["conf"], "status": status,
                     "verdict": "non" if non_compliant else "ok"})

    if not out:
        recoverable = len(persons_raw) > 0
        best_person_conf = max((p["conf"] for p in persons_raw), default=None)
        return {"persons": [], "verdict": "none", "recoverable": recoverable,
                "best_person_conf": best_person_conf}
    verdict = "non" if any(x["verdict"] == "non" for x in out) else "ok"
    return {"persons": out, "verdict": verdict, "recoverable": False}
