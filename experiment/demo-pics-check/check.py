"""Run detection on demo-pics with four YOLO models and build 2x2 compare grid.

Models:
  - yolo26m_merged_150ev2: runs/detect/yolo26m_merged_150ev2/weights/best.pt
  - yolov8n_scratch:       runs/scratch/yolov8n_scratch/weights/best.pt
  - yolo26m_css_300e:      runs/detect/yolo26m_css_300e/weights/best.pt
  - yolo26m_merged_150e:   runs/detect/yolo26m_merged_150e/weights/best.pt

Output: experiment/demo-pics-check/output/
  - output/<model>/{typical,challenging}/*_pred.jpg  (annotated + model label)
  - output/<model>/{typical,challenging}/*_pred.json (per-image labels: [{key, conf, box}])
  - output/<model>/results.json + summary.csv
  - output/compare/{typical,challenging}/*_grid2x2.jpg (2x2, model name per quadrant)
"""
from __future__ import annotations

import json
import shutil
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]  # E:/work/P0-safety
SRC = ROOT / "demo-pics"
OUT = Path(__file__).resolve().parent / "output"

MODELS = {
    "yolo26m_merged_150ev2": ROOT / "runs/detect/yolo26m_merged_150ev2/weights/best.pt",
    "yolov8n_scratch": ROOT / "runs/scratch/yolov8n_scratch/weights/best.pt",
    "yolo26m_css_300e": ROOT / "runs/detect/yolo26m_css_300e/weights/best.pt",
    "yolo26m_merged_150e": ROOT / "runs/detect/yolo26m_merged_150e/weights/best.pt",
}

CONF = 0.35
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 2x2 tile size (WxH). 960x640 -> grid 1920x1280
TILE_W, TILE_H = 960, 640


def find_images(src: Path) -> list[Path]:
    return sorted(
        p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def add_model_label(img: np.ndarray, label: str) -> np.ndarray:
    """Draw model name banner at top of image (BGR)."""
    h, w = img.shape[:2]
    banner_h = max(36, int(h * 0.065))
    # estimate font scale from banner height
    font_scale = max(0.7, banner_h / 40 * 0.9)
    thickness = max(2, int(banner_h / 18))
    out = img.copy()
    cv2.rectangle(out, (0, 0), (w, banner_h), (30, 30, 30), -1)
    # fit text inside width
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if tw > w - 20:
        font_scale *= (w - 20) / tw
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(10, (w - tw) // 2)
    ty = (banner_h + th) // 2
    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def fit_tile(img: np.ndarray, tw: int = TILE_W, th: int = TILE_H) -> np.ndarray:
    """Resize preserving aspect and pad to tw x th with gray."""
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    inter = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=inter)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def annotate_and_save(result, out_path: Path, model_name: str):
    """Save result.plot() image with model name label; handle webp/PIL compat."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()  # BGR numpy array
    labeled = add_model_label(plotted, model_name)
    ok = cv2.imwrite(str(out_path), labeled)
    if not ok:
        from PIL import Image

        rgb = cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(out_path)


def run_model(model_name: str, weight: Path, images: list[Path]):
    print(f"\n[{model_name}] loading {weight}")
    model = YOLO(str(weight))
    print(f"  names: {model.names}")

    model_out = OUT / model_name
    if model_out.exists():
        shutil.rmtree(model_out)

    results_all: list[dict] = []
    rows: list[dict] = []

    for img in images:
        rel = img.relative_to(SRC)
        save_path = model_out / rel.parent / f"{img.stem}_pred.jpg"
        results = model.predict(source=str(img), conf=CONF, verbose=False)
        r = results[0]

        annotate_and_save(r, save_path, model_name)

        # per-image labels in pasted format: [{"key", "conf", "box":[x1,y1,x2,y2] normalized 0-1}]
        h, w = r.orig_shape  # (h, w)
        labels: list[dict] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(cls)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                labels.append(
                    {
                        "key": model.names[int(cls[i])],
                        "conf": float(conf[i]),
                        "box": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)],
                    }
                )

        # sidecar per-image json (same array as pasted example)
        json_path = save_path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(labels, jf, indent=2, ensure_ascii=False)

        results_all.append({"image": str(rel).replace("\\", "/"), "detections": labels})
        rows.append({"image": str(rel).replace("\\", "/"), "count": len(labels)})
        print(f"  {rel}: {len(labels)} dets")

    with open(model_out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)

    with open(model_out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "count"])
        w.writeheader()
        w.writerows(rows)

    total = sum(r["count"] for r in rows)
    print(f"[{model_name}] done: {len(images)} images, {total} total detections")
    return results_all


def make_compare_2x2(images: list[Path]):
    """Build 2x2 grid (row-major in MODELS order) with per-quadrant model label.

    Tiles are already labeled via annotate_and_save, so we just fit + grid.
    Order: [yolo26m_merged_150ev2, yolov8n_scratch] on top row,
           [yolo26m_css_300e, yolo26m_merged_150e] on bottom row.
    """
    compare_root = OUT / "compare"
    if compare_root.exists():
        shutil.rmtree(compare_root)
    model_names = list(MODELS.keys())
    assert len(model_names) == 4, "expected 4 models for 2x2 grid"

    for img in images:
        rel = img.relative_to(SRC)
        tiles: list[np.ndarray] = []
        for m in model_names:
            p = OUT / m / rel.parent / f"{img.stem}_pred.jpg"
            im = cv2.imread(str(p))
            if im is None:
                # missing -> placeholder gray with label
                im = np.full((TILE_H, TILE_W, 3), 80, dtype=np.uint8)
                im = add_model_label(im, f"{m} (missing)")
            tiles.append(fit_tile(im))

        # 2x2
        top = np.hstack([tiles[0], tiles[1]])
        bottom = np.hstack([tiles[2], tiles[3]])
        grid = np.vstack([top, bottom])

        out = compare_root / rel.parent / f"{img.stem}_grid2x2.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), grid)
    print(f"[compare] 2x2 grid saved to {compare_root} ({len(images)} images, order={model_names})")


def main():
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")
    for n, w in MODELS.items():
        if not w.exists():
            raise SystemExit(f"Weight not found [{n}]: {w}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    images = find_images(SRC)
    if not images:
        raise SystemExit(f"No images found in {SRC}")
    print(f"Found {len(images)} images in {SRC}")
    for p in images:
        print(f"  {p.relative_to(SRC)}")

    for name, weight in MODELS.items():
        run_model(name, weight, images)

    make_compare_2x2(images)
    print(f"\nAll outputs in: {OUT}")
    print(f"Models order in grid (TL/TR/BL/BR): {list(MODELS.keys())}")


if __name__ == "__main__":
    main()
