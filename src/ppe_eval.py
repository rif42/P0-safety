"""
ppe_eval.py
===========
Operating-point evaluation for the PPE detector.

Why not just read mAP
---------------------
mAP integrates precision over every recall level at every threshold. It is the
right number for comparing architectures and the wrong number for deciding
whether to deploy a safety-reporting system. What a safety report needs is:

  "At the confidence threshold we will actually run in production, what fraction
   of genuine no-helmet instances do we catch, and how often do we accuse a
   compliant worker?"

That is per-class recall and precision at one chosen threshold. This module runs
inference once at a low floor, then sweeps thresholds cheaply over the cached
detections, so you can choose the operating point deliberately rather than
inheriting Ultralytics' default of 0.25.

The asymmetry to keep in mind: a missed violation is a safety risk that shows up
in an accident report, while a false violation is a credibility risk that shows
up in someone's inbox. They are not equally costly, and they are not equally
costly for every class either. Choose per class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def load_gt(images_dir: Path, labels_dir: Path) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    """Read YOLO labels into {image_stem: [(class_id, xyxy_normalised), ...]}."""
    gt: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for img in sorted(Path(images_dir).iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lp = Path(labels_dir) / f"{img.stem}.txt"
        items: List[Tuple[int, np.ndarray]] = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                cx, cy, w, h = (float(v) for v in p[1:5])
                items.append((c, np.array([cx - w / 2, cy - h / 2,
                                           cx + w / 2, cy + h / 2])))
        gt[img.stem] = items
    return gt


def collect_predictions(model, images_dir: Path, conf_floor: float = 0.05,
                        imgsz: int = 640, end2end: bool = True,
                        batch: int = 16, device: int | str = 0
                        ) -> Dict[str, List[Tuple[int, float, np.ndarray]]]:
    """One inference pass at a low confidence floor.

    Boxes come back normalised so they can be compared with YOLO labels directly.
    `end2end=False` switches YOLO26 to its one-to-many head with NMS, which is
    the mode to use when you want a conventional score distribution to sweep.
    """
    files = [p for p in sorted(Path(images_dir).iterdir())
             if p.suffix.lower() in IMG_EXT]
    out: Dict[str, List[Tuple[int, float, np.ndarray]]] = {}
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        kw = dict(conf=conf_floor, imgsz=imgsz, device=device, verbose=False)
        try:
            results = model.predict([str(c) for c in chunk], end2end=end2end, **kw)
        except TypeError:
            # end2end is a YOLO26 argument; earlier architectures reject it
            results = model.predict([str(c) for c in chunk], **kw)
        for path, r in zip(chunk, results):
            dets: List[Tuple[int, float, np.ndarray]] = []
            if r.boxes is not None and len(r.boxes):
                h, w = r.orig_shape
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for k in range(len(cls)):
                    b = xyxy[k] / np.array([w, h, w, h], dtype=float)
                    dets.append((int(cls[k]), float(conf[k]), b))
            out[path.stem] = dets
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


@dataclass
class ClassMetrics:
    cls_id: int
    name: str
    threshold: float
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float("nan")

    @property
    def support(self) -> int:
        return self.tp + self.fn


def evaluate_at(preds: Dict[str, List[Tuple[int, float, np.ndarray]]],
                gt: Dict[str, List[Tuple[int, np.ndarray]]],
                class_names: Sequence[str], threshold: float,
                iou_thr: float = 0.5) -> List[ClassMetrics]:
    """Greedy highest-confidence-first matching within each class, per image."""
    tp = np.zeros(len(class_names), dtype=int)
    fp = np.zeros(len(class_names), dtype=int)
    fn = np.zeros(len(class_names), dtype=int)

    for stem, g in gt.items():
        p = [d for d in preds.get(stem, []) if d[1] >= threshold]
        for c in range(len(class_names)):
            gb = np.array([b for cc, b in g if cc == c]).reshape(-1, 4)
            pd = sorted([d for d in p if d[0] == c], key=lambda x: -x[1])
            pb = np.array([d[2] for d in pd]).reshape(-1, 4)
            if len(pb) == 0:
                fn[c] += len(gb)
                continue
            if len(gb) == 0:
                fp[c] += len(pb)
                continue
            M = _iou_matrix(pb, gb)
            used = set()
            for i in range(len(pb)):
                j = int(np.argmax(M[i]))
                if M[i, j] >= iou_thr and j not in used:
                    used.add(j)
                    tp[c] += 1
                else:
                    fp[c] += 1
            fn[c] += len(gb) - len(used)

    return [ClassMetrics(c, class_names[c], threshold, int(tp[c]), int(fp[c]), int(fn[c]))
            for c in range(len(class_names))]


def sweep(preds, gt, class_names: Sequence[str],
          thresholds: Optional[Sequence[float]] = None,
          iou_thr: float = 0.5):
    """Return a tidy DataFrame of per-class precision/recall across thresholds."""
    import pandas as pd
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        for m in evaluate_at(preds, gt, class_names, float(t), iou_thr):
            rows.append({"class": m.name, "cls_id": m.cls_id, "threshold": float(t),
                         "tp": m.tp, "fp": m.fp, "fn": m.fn, "support": m.support,
                         "precision": m.precision, "recall": m.recall, "f1": m.f1})
    return pd.DataFrame(rows)


def pick_operating_point(df, cls_name: str, min_precision: float = 0.60,
                         objective: str = "recall") -> Optional[dict]:
    """Highest-recall threshold for a class that still clears a precision floor.

    This is the inverse of the usual habit of maximising F1. For violation
    classes the question is 'how many can we catch without crying wolf too
    often', so precision is a constraint and recall is the objective.
    Returns None when no threshold clears the floor - which is itself a finding,
    and means the class is not fit to report on yet.
    """
    sub = df[(df["class"] == cls_name) & (df["precision"] >= min_precision)]
    sub = sub.dropna(subset=["precision", "recall"])
    if sub.empty:
        return None
    best = sub.sort_values(objective, ascending=False).iloc[0]
    return {"class": cls_name, "threshold": float(best["threshold"]),
            "precision": float(best["precision"]), "recall": float(best["recall"]),
            "f1": float(best["f1"]), "support": int(best["support"])}


def box_size_report(gt: Dict[str, List[Tuple[int, np.ndarray]]],
                    class_names: Sequence[str]):
    """Distribution of object sizes as a fraction of image area, per class.

    Drives the imgsz decision. Anything whose median box is under roughly 0.1 per
    cent of frame area is a small object: at 640 px it occupies a handful of
    pixels and no amount of extra model capacity will recover it. Raising imgsz is
    the lever that works. Boots are usually the smallest thing in the taxonomy.
    """
    import pandas as pd
    rows = []
    for stem, items in gt.items():
        for c, b in items:
            w, h = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
            rows.append({"class": class_names[c] if c < len(class_names) else str(c),
                         "area_frac": w * h,
                         "px_at_640": np.sqrt(max(w * h, 0)) * 640})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("class").agg(
        n=("area_frac", "size"),
        median_area_pct=("area_frac", lambda s: round(100 * float(np.median(s)), 4)),
        p10_area_pct=("area_frac", lambda s: round(100 * float(np.percentile(s, 10)), 4)),
        median_px_at_640=("px_at_640", lambda s: round(float(np.median(s)), 1)),
    ).reset_index()
    agg["small_object"] = agg["median_area_pct"] < 0.1
    return agg.sort_values("median_area_pct")


__all__ = ["load_gt", "collect_predictions", "evaluate_at", "sweep",
           "pick_operating_point", "box_size_report", "ClassMetrics"]
