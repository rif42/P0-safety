#!/usr/bin/env python
"""
Evaluate yolo26m_merged_150ev2 against demo-pics/ with GT from data/merged/.

GT resolution:
  demo-pics/ is a sampled subset of data/merged/ (filenames with .rf.<hash>),
  but some files are stored under their original_filename (e.g. snehilsanyal
  bare names) while others use merged_filename (anuragraj03__..., ketakichalke-...).
  This script uses data/merged/merge_manifest.csv as the canonical index:
    - merged_filename  -> data/merged/<split>/labels/<merged_filename>.txt
    - original_filename -> same label path via the merged_filename column
  Files with no manifest entry (external images like Building-Site-CCTV etc.)
  are reported as unmatched and excluded from detection metrics.

Metrics (at IoU=0.5, conf=0.25 by default):
  - per-class and micro-averaged TP/FP/FN via greedy IoU matching per class
  - precision = TP/(TP+FP), recall = TP/(TP+FN), F1, accuracy = TP/(TP+FP+FN)
  - challenging vs typical breakdown (demo-pics/challenging, demo-pics/typical)

Usage:
  python experiment/falcon/eval/eval_yolo_demo_pics.py \
    --weights runs/detect/yolo26m_merged_150ev2/weights/best.pt \
    --source demo-pics \
    --data data/merged \
    --manifest data/merged/merge_manifest.csv \
    --iou 0.5 --conf 0.25 --out outputs/eval_demo_pics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="YOLO demo-pics vs data/merged GT evaluation")
    p.add_argument("--weights", type=str, default="runs/detect/yolo26m_merged_150ev2/weights/best.pt")
    p.add_argument("--source", type=str, default="demo-pics")
    p.add_argument("--data", type=str, default="data/merged")
    p.add_argument("--manifest", type=str, default="data/merged/merge_manifest.csv")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    p.add_argument("--out", type=str, default="outputs/eval_demo_pics.json")
    p.add_argument("--csv-out", type=str, default=None, help="Optional CSV path for per-class table")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str, default=None, help="Ultralytics device, e.g. '0' or 'cpu'")
    return p.parse_args()


def load_manifest(manifest_path: Path):
    by_merged = {}
    by_original = {}
    if not manifest_path.exists():
        return by_merged, by_original
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            by_merged[row["merged_filename"]] = row
            by_original[row["original_filename"]] = row
    return by_merged, by_original


def resolve_gt(demo_name: str, by_merged, by_original, data_root: Path):
    """Return (label_path, split, merged_filename) or (None, None, None)."""
    row = None
    if demo_name in by_merged:
        row = by_merged[demo_name]
    elif demo_name in by_original:
        row = by_original[demo_name]
    if row is None:
        return None, None, None
    merged_name = row["merged_filename"]
    split = row["split"]
    label_path = data_root / split / "labels" / (Path(merged_name).stem + ".txt")
    return label_path, split, merged_name


def parse_yolo_txt(label_path: Path):
    """Return list of (cls:int, cx, cy, w, h) normalized."""
    if not label_path or not label_path.exists():
        return []
    out = []
    for line in open(label_path):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        out.append((cls, cx, cy, w, h))
    return out


def xywhn_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def box_iou(a, b):
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def evaluate_one(gt_boxes_by_cls, pred_boxes_by_cls, iou_thr: float):
    """Greedy matching per class. Returns dict cls->(tp,fp,fn)."""
    result = {}
    all_classes = set(gt_boxes_by_cls.keys()) | set(pred_boxes_by_cls.keys())
    for cls in all_classes:
        gts = gt_boxes_by_cls.get(cls, [])
        preds = pred_boxes_by_cls.get(cls, [])
        # preds sorted by conf desc already
        n_gt = len(gts)
        n_pred = len(preds)
        if n_gt == 0 and n_pred == 0:
            result[cls] = (0, 0, 0)
            continue
        if n_gt == 0:
            result[cls] = (0, n_pred, 0)
            continue
        if n_pred == 0:
            result[cls] = (0, 0, n_gt)
            continue
        # IoU matrix
        matched_gt = [False] * n_gt
        tp = 0
        for pb in preds:
            best_iou = 0
            best_j = -1
            for j, gb in enumerate(gts):
                if matched_gt[j]:
                    continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                matched_gt[best_j] = True
                tp += 1
        fp = n_pred - tp
        fn = n_gt - tp
        result[cls] = (tp, fp, fn)
    return result


def main():
    args = parse_args()
    weights = Path(args.weights)
    source = Path(args.source)
    data_root = Path(args.data)
    manifest_path = Path(args.manifest)
    out_path = Path(args.out)

    if not weights.exists():
        print(f"[error] weights not found: {weights}")
        raise SystemExit(2)
    if not source.exists():
        print(f"[error] source not found: {source}")
        raise SystemExit(2)

    # Load manifest
    by_merged, by_original = load_manifest(manifest_path)
    print(f"manifest: {len(by_merged)} merged, {len(by_original)} original entries from {manifest_path}")

    # Load model
    from ultralytics import YOLO

    print(f"loading {weights} ...")
    model = YOLO(str(weights))
    names = model.names  # dict id->name
    # Ensure ordered list 0..nc-1
    nc = len(names)
    if isinstance(names, dict):
        ordered = [names[i] for i in range(nc)]
    else:
        ordered = list(names)
    print(f"model nc={nc} names={ordered}")
    # also load data.yaml names if available
    data_yaml = data_root / "data.yaml"
    if data_yaml.exists():
        try:
            import yaml

            d = yaml.safe_load(open(data_yaml))
            print(f"data.yaml nc={d.get('nc')} names={d.get('names')}")
        except Exception as e:
            print(f"warn: could not read {data_yaml}: {e}")

    # Collect demo images
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []
    for split_name in ["challenging", "typical"]:
        split_dir = source / split_name
        if not split_dir.exists():
            # if source is flat, just glob
            continue
        for p in sorted(split_dir.iterdir()):
            if p.suffix.lower() in exts:
                images.append((p, split_name))
    if not images:
        # flat
        for p in sorted(source.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                # infer bucket by parent name
                bucket = p.parent.name if p.parent.name in ("challenging", "typical") else "unknown"
                images.append((p, bucket))
    print(f"found {len(images)} images in {source} (challenging+typical)")

    # Accumulators
    total_per_cls = defaultdict(lambda: [0, 0, 0])  # tp,fp,fn
    bucket_per_cls = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    n_matched = 0
    n_unmatched = 0
    unmatched_list = []
    per_image_rows = []

    for img_path, bucket in images:
        demo_name = img_path.name
        label_path, split, merged_name = resolve_gt(demo_name, by_merged, by_original, data_root)

        if label_path is None:
            n_unmatched += 1
            unmatched_list.append(str(img_path))
            if args.verbose:
                print(f"[skip no GT] {img_path} (no manifest entry)")
            continue
        if not label_path.exists():
            # manifest says it should exist but file missing
            n_unmatched += 1
            unmatched_list.append(str(img_path) + f" -> missing {label_path}")
            if args.verbose:
                print(f"[skip missing label] {img_path} -> {label_path}")
            continue

        # Load GT boxes
        gt_raw = parse_yolo_txt(label_path)
        # Need image size to denormalize
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            # fallback to merged image size if demo-pics was resized
            merged_img = data_root / split / "images" / merged_name
            if merged_img.exists():
                with Image.open(merged_img) as im:
                    img_w, img_h = im.size
            else:
                print(f"[warn] cannot open {img_path} nor {merged_img}, skipping")
                n_unmatched += 1
                unmatched_list.append(str(img_path))
                continue

        gt_by_cls = defaultdict(list)
        for cls, cx, cy, w, h in gt_raw:
            xyxy = xywhn_to_xyxy(cx, cy, w, h, img_w, img_h)
            gt_by_cls[cls].append(xyxy)

        # Inference
        # Ultralytics predict handles webp etc.
        kwargs = dict(conf=args.conf, iou=0.7, verbose=False)
        if args.device:
            kwargs["device"] = args.device
        results = model.predict(str(img_path), **kwargs)
        r = results[0]
        # r.boxes: xyxy, cls, conf
        pred_by_cls = defaultdict(list)
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy()
            # sort by conf desc globally, but per-class order also desc
            order = np.argsort(-conf_arr)
            for idx in order:
                c = int(cls_arr[idx])
                box = xyxy[idx].tolist()
                pred_by_cls[c].append(box)

        # Match
        per_cls = evaluate_one(gt_by_cls, pred_by_cls, args.iou)
        for c, (tp, fp, fn) in per_cls.items():
            total_per_cls[c][0] += tp
            total_per_cls[c][1] += fp
            total_per_cls[c][2] += fn
            bucket_per_cls[bucket][c][0] += tp
            bucket_per_cls[bucket][c][1] += fp
            bucket_per_cls[bucket][c][2] += fn

        n_matched += 1
        if args.verbose:
            gt_n = sum(len(v) for v in gt_by_cls.values())
            pred_n = sum(len(v) for v in pred_by_cls.values())
            tp_sum = sum(v[0] for v in per_cls.values())
            print(f"[{bucket}] {demo_name}: GT={gt_n} pred={pred_n} TP={tp_sum} split={split}")

        per_image_rows.append(
            {
                "image": str(img_path.as_posix()),
                "demo_name": demo_name,
                "bucket": bucket,
                "split": split,
                "merged_name": merged_name,
                "gt_count": sum(len(v) for v in gt_by_cls.values()),
                "pred_count": sum(len(v) for v in pred_by_cls.values()),
                "tp": sum(v[0] for v in per_cls.values()),
                "fp": sum(v[1] for v in per_cls.values()),
                "fn": sum(v[2] for v in per_cls.values()),
            }
        )

    # Compute metrics
    def metrics_from(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # Jaccard / detection accuracy
        return prec, rec, f1, acc

    # Overall
    total_tp = sum(v[0] for v in total_per_cls.values())
    total_fp = sum(v[1] for v in total_per_cls.values())
    total_fn = sum(v[2] for v in total_per_cls.values())
    overall_prec, overall_rec, overall_f1, overall_acc = metrics_from(total_tp, total_fp, total_fn)

    per_class = {}
    for c in range(nc):
        tp, fp, fn = total_per_cls.get(c, [0, 0, 0])
        prec, rec, f1, acc = metrics_from(tp, fp, fn)
        per_class[str(c)] = {
            "name": ordered[c] if c < len(ordered) else str(c),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "support": int(tp + fn),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
        }

    buckets = {}
    for bucket, d in bucket_per_cls.items():
        btp = sum(v[0] for v in d.values())
        bfp = sum(v[1] for v in d.values())
        bfn = sum(v[2] for v in d.values())
        prec, rec, f1, acc = metrics_from(btp, bfp, bfn)
        buckets[bucket] = {"tp": int(btp), "fp": int(bfp), "fn": int(bfn), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

    result = {
        "weights": str(weights),
        "source": str(source),
        "data": str(data_root),
        "manifest": str(manifest_path),
        "iou_threshold": args.iou,
        "conf_threshold": args.conf,
        "model_names": ordered,
        "n_demo_images": len(images),
        "n_matched_with_gt": n_matched,
        "n_unmatched_no_gt": n_unmatched,
        "unmatched": unmatched_list,
        "overall": {
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "precision": overall_prec,
            "recall": overall_rec,
            "f1": overall_f1,
            "accuracy": overall_acc,
        },
        "per_class": per_class,
        "by_bucket": buckets,
        "per_image": per_image_rows,
    }

    # Print table
    print("\n" + "=" * 78)
    print(f"YOLO demo-pics eval  IoU={args.iou} conf={args.conf}")
    print(f"  {n_matched}/{len(images)} images matched to GT in {data_root}  ({n_unmatched} unmatched/external)")
    print(f"  Overall: P={overall_prec:.3f} R={overall_rec:.3f} F1={overall_f1:.3f} Acc(J)={overall_acc:.3f}  (TP={total_tp} FP={total_fp} FN={total_fn})")
    for bucket, m in buckets.items():
        print(f"  [{bucket:12s}] P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Acc={m['accuracy']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    print("-" * 78)
    header = f"{'cls':>3}  {'name':<12} {'sup':>5} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}"
    print(header)
    print("-" * 78)
    for c in range(nc):
        r = per_class[str(c)]
        print(
            f"{c:>3}  {r['name']:<12} {r['support']:>5} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}  {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['accuracy']:>6.3f}"
        )
    print("=" * 78)
    if n_unmatched:
        print(f"\nUnmatched (no GT, excluded from metrics): {n_unmatched}")
        for u in unmatched_list[:20]:
            print(f"  - {u}")
        if len(unmatched_list) > 20:
            print(f"  ... and {len(unmatched_list)-20} more")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved JSON -> {out_path}")

    csv_out = Path(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")
    # per-class CSV
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "support", "tp", "fp", "fn", "precision", "recall", "f1", "accuracy"])
        for c in range(nc):
            r = per_class[str(c)]
            w.writerow([c, r["name"], r["support"], r["tp"], r["fp"], r["fn"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['accuracy']:.4f}"])
        w.writerow([])
        w.writerow(["overall", "", total_tp + total_fn, total_tp, total_fp, total_fn, f"{overall_prec:.4f}", f"{overall_rec:.4f}", f"{overall_f1:.4f}", f"{overall_acc:.4f}"])
    print(f"Saved CSV  -> {csv_out}")


if __name__ == "__main__":
    main()
