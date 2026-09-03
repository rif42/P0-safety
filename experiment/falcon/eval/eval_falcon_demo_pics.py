#!/usr/bin/env python
"""
Evaluate Falcon Perception against demo-pics/ with GT from data/merged/.

Mirrors experiment/falcon/eval/eval_yolo_demo_pics.py but runs Falcon
instead of YOLO, converting its per-query segmentation masks (COCO RLE)
to detection bboxes so the same IoU-based precision/recall/F1/accuracy
metrics apply.

Falcon path (reuses existing code, no new deps):
  load_and_prepare_model + PagedInferenceEngine + build_prompt_for_task
  -> Sequence(text=prompt, image=PIL, task="segmentation")
  -> engine.generate -> seq.output_aux.masks_rle / bboxes_raw

Mask -> bbox:
  primary: RLE -> pycocotools.mask.decode or mask_util.toBbox -> [x,y,w,h] -> [x1,y1,x2,y2]
           (resize_rle to original resolution first, same as eval/pbench.py)
  fallback: pair_bbox_entries(bboxes_raw) when do_segmentation=False (detection-only)
  paired bbox preferred when idx < len(paired); else RLE tight bbox (see server/app.py _build_masks).

Class -> query:
  9 merged classes from data/merged/data.yaml (person/helmet/gloves/boots/vest/no-helmet/no-gloves/no-boots/no-vest)
  Each is issued as its own text query; override via --queries "person,helmet,..."

Usage:
  python experiment/falcon/eval/eval_falcon_demo_pics.py --hf-local-dir /path/to/export
  python experiment/falcon/eval/eval_falcon_demo_pics.py --hf-model-id tiiuae/Falcon-Perception --device cuda --dtype bfloat16
  python experiment/falcon/eval/eval_falcon_demo_pics.py --verbose --iou 0.5 --out outputs/eval_falcon_demo_pics.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ---- GT helpers (same as eval_yolo_demo_pics.py) ----

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
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + ab - inter + 1e-9)


def evaluate_one(gt_boxes_by_cls, pred_boxes_by_cls, iou_thr: float):
    result = {}
    all_classes = set(gt_boxes_by_cls.keys()) | set(pred_boxes_by_cls.keys())
    for cls in all_classes:
        gts = gt_boxes_by_cls.get(cls, [])
        preds = pred_boxes_by_cls.get(cls, [])
        n_gt = len(gts); n_pred = len(preds)
        if n_gt == 0 and n_pred == 0:
            result[cls] = (0,0,0); continue
        if n_gt == 0:
            result[cls] = (0,n_pred,0); continue
        if n_pred == 0:
            result[cls] = (0,0,n_gt); continue
        matched_gt = [False]*n_gt; tp=0
        for pb in preds:
            best_iou=0; best_j=-1
            for j, gb in enumerate(gts):
                if matched_gt[j]: continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou=iou; best_j=j
            if best_j>=0 and best_iou >= iou_thr:
                matched_gt[best_j]=True; tp+=1
        fp=n_pred-tp; fn=n_gt-tp
        result[cls]=(tp,fp,fn)
    return result


# ---- Falcon mask -> bbox ----

def rle_to_xyxy(rle: dict):
    """Tight bbox from RLE via pycocotools.mask.toBbox / decode."""
    try:
        import pycocotools.mask as mask_util
        # ensure bytes counts
        r = rle
        if isinstance(r.get("counts"), str):
            r = {**r, "counts": r["counts"].encode("utf-8")}
        bbox = mask_util.toBbox(r).tolist()  # [x,y,w,h]
        x, y, w, h = bbox
        return [float(x), float(y), float(x+w), float(y+h)]
    except Exception:
        # fallback: decode then tight bounds
        try:
            from falcon_perception.visualization_utils import decode_coco_rle
            m = decode_coco_rle(rle)
            if m is None or m.size==0:
                return None
            rows = np.any(m, axis=1); cols = np.any(m, axis=0)
            if not rows.any() or not cols.any():
                return None
            rmin, rmax = np.where(rows)[0][[0,-1]]
            cmin, cmax = np.where(cols)[0][[0,-1]]
            return [float(cmin), float(rmin), float(cmax+1), float(rmax+1)]
        except Exception:
            return None


def normalized_bbox_to_xyxy(b: dict, img_w: int, img_h: int):
    cx = b.get("x", 0.5); cy = b.get("y", 0.5); bh = b.get("h", 0.0); bw = b.get("w", 0.0)
    x1 = (cx - bw/2)*img_w; y1 = (cy - bh/2)*img_h
    x2 = (cx + bw/2)*img_w; y2 = (cy + bh/2)*img_h
    return [x1,y1,x2,y2]


def parse_args():
    p = argparse.ArgumentParser(description="Falcon demo-pics vs data/merged GT evaluation (mask->bbox)")
    p.add_argument("--hf-model-id", type=str, default=None, help="HF model id (default tiiuae/Falcon-Perception)")
    p.add_argument("--hf-local-dir", type=str, default=None, help="Local export dir with model.safetensors + tokenizer.json")
    p.add_argument("--hf-revision", type=str, default="main")
    p.add_argument("--source", type=str, default="demo-pics")
    p.add_argument("--data", type=str, default="data/merged")
    p.add_argument("--manifest", type=str, default="data/merged/merge_manifest.csv")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", type=str, default="outputs/eval_falcon_demo_pics.json")
    p.add_argument("--csv-out", type=str, default=None)
    p.add_argument("--queries", type=str, default=None, help="Comma-separated class queries (order = class id). Default: data.yaml names")
    p.add_argument("--max-dimension", type=int, default=1024)
    p.add_argument("--min-dimension", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float32","float16"])
    p.add_argument("--task", type=str, default="segmentation", choices=["segmentation","detection"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Limit images (0=all)")
    return p.parse_args()


def main():
    args = parse_args()
    source = Path(args.source); data_root = Path(args.data)
    manifest_path = Path(args.manifest); out_path = Path(args.out)

    # Resolve source
    if not source.exists():
        print(f"[error] source not found: {source}"); raise SystemExit(2)

    by_merged, by_original = load_manifest(manifest_path)
    print(f"manifest: {len(by_merged)} merged, {len(by_original)} original from {manifest_path}")

    # Load class names (same order as YOLO: data.yaml nc=9)
    ordered = ["person","helmet","gloves","boots","vest","no-helmet","no-gloves","no-boots","no-vest"]
    nc = 9
    data_yaml = data_root / "data.yaml"
    if data_yaml.exists():
        try:
            import yaml
            d = yaml.safe_load(open(data_yaml))
            if d.get("names"):
                ordered = list(d["names"])
                nc = int(d.get("nc", len(ordered)))
                print(f"data.yaml nc={nc} names={ordered}")
        except Exception as e:
            print(f"[warn] data.yaml read failed: {e}")

    if args.queries:
        qs = [q.strip() for q in args.queries.split(",") if q.strip()]
        if len(qs) != nc:
            print(f"[warn] --queries has {len(qs)} entries, expected {nc}; using provided order anyway")
        ordered_queries = qs + ordered[len(qs):] if len(qs) < nc else qs[:nc]
    else:
        ordered_queries = ordered

    print(f"Falcon queries ({len(ordered_queries)}): {ordered_queries}")
    print(f"model: hf_local_dir={args.hf_local_dir} hf_model_id={args.hf_model_id or 'tiiuae/Falcon-Perception'} task={args.task}")

    # Lazy load Falcon (only when needed, so --help works without torch)
    import torch
    from falcon_perception import build_prompt_for_task, load_and_prepare_model, setup_torch_config
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams, Sequence, engine_config_for_gpu

    setup_torch_config()

    # Import metrics for RLE resizing (same as pbench)
    import sys
    sys.path.insert(0, str(Path("experiment/falcon/eval").resolve()))
    import metrics as falcon_metrics  # noqa

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=args.hf_model_id,
        hf_revision=args.hf_revision,
        hf_local_dir=args.hf_local_dir,
        device=args.device,
        dtype=args.dtype,
        compile=True,
    )
    if args.task == "segmentation" and not getattr(model_args, "do_segmentation", True):
        print("[warn] model do_segmentation=False, forcing task=detection")
        args.task = "detection"

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    cfg = engine_config_for_gpu(max_image_size=args.max_dimension, dtype=model.dtype)
    engine = PagedInferenceEngine(model, tokenizer, image_processor, max_seq_length=model_args.max_seq_len, **cfg)
    print(f"Engine config: {cfg}")

    # Collect demo images
    exts = {".jpg",".jpeg",".png",".webp",".bmp"}
    images = []
    for split_name in ["challenging","typical"]:
        d = source / split_name
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in exts:
                    images.append((p, split_name))
    if not images:
        for p in sorted(source.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                bucket = p.parent.name if p.parent.name in ("challenging","typical") else "unknown"
                images.append((p, bucket))
    if args.limit and args.limit > 0:
        images = images[:args.limit]
    print(f"found {len(images)} images in {source}")

    # For visualization_utils reuse
    try:
        from falcon_perception.visualization_utils import pair_bbox_entries
        has_pair = True
    except Exception:
        has_pair = False
        def pair_bbox_entries(raw):
            out=[]; cur={}
            for e in raw or []:
                if not isinstance(e, dict): continue
                cur.update(e)
                if all(k in cur for k in ("x","y","h","w")):
                    out.append(dict(cur)); cur={}
            return out

    total_per_cls = defaultdict(lambda: [0,0,0])
    bucket_per_cls = defaultdict(lambda: defaultdict(lambda: [0,0,0]))
    n_matched=0; n_unmatched=0; unmatched_list=[]; per_image_rows=[]

    sampling_params_template = dict(
        max_new_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id] + ([tokenizer.end_of_query_token_id] if hasattr(tokenizer, "end_of_query_token_id") else []),
        segmentation_threshold=0.3,
        hr_upsample_ratio=8,
    )

    for img_path, bucket in images:
        demo_name = img_path.name
        label_path, split, merged_name = resolve_gt(demo_name, by_merged, by_original, data_root)
        if label_path is None:
            n_unmatched+=1; unmatched_list.append(str(img_path))
            if args.verbose: print(f"[skip no GT] {img_path}")
            continue
        if not label_path.exists():
            n_unmatched+=1; unmatched_list.append(str(img_path)+f" -> missing {label_path}")
            if args.verbose: print(f"[skip missing label] {img_path} -> {label_path}")
            continue

        # GT
        gt_raw = parse_yolo_txt(label_path)
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            merged_img = data_root / split / "images" / merged_name
            if merged_img.exists():
                with Image.open(merged_img) as im:
                    img_w, img_h = im.size
            else:
                print(f"[warn] cannot open {img_path} nor {merged_img}, skipping")
                n_unmatched+=1; unmatched_list.append(str(img_path)); continue
        gt_by_cls = defaultdict(list)
        for cls,cx,cy,w,h in gt_raw:
            gt_by_cls[cls].append(xywhn_to_xyxy(cx,cy,w,h,img_w,img_h))

        # ---- Falcon inference: one Sequence per query (per class) ----
        # Build sequences batched by query so engine can batch them
        seqs = []
        seq_to_cls = []
        for cls_id, q in enumerate(ordered_queries):
            prompt = build_prompt_for_task(q, args.task)
            pil = Image.open(img_path).convert("RGB")
            # Do NOT force-resize here; let Sequence + engine handle min/max_dimension
            seq = Sequence(text=prompt, image=pil, min_image_size=args.min_dimension, max_image_size=args.max_dimension, request_idx=cls_id, task=args.task)
            seqs.append(seq)
            seq_to_cls.append(cls_id)

        # Need sampling params per sequence (same)
        sp = SamplingParams(
            max_new_tokens=sampling_params_template["max_new_tokens"],
            stop_token_ids=sampling_params_template["stop_token_ids"],
            hr_upsample_ratio=sampling_params_template["hr_upsample_ratio"],
        )

        with torch.inference_mode():
            engine.generate(seqs, sampling_params=sp, use_tqdm=False)

        pred_by_cls = defaultdict(list)
        for seq, cls_id in zip(seqs, seq_to_cls):
            # output_aux
            aux = getattr(seq, "output_aux", None)
            if aux is None:
                continue
            # need original size for RLE resize (same as pbench: back to original)
            orig_w, orig_h = img_w, img_h
            # Resize RLEs to original before bbox
            rles = []
            for rle in getattr(aux, "masks_rle", []) or []:
                if not isinstance(rle, dict) or "counts" not in rle or "size" not in rle:
                    continue
                # metrics.resize_rle expects target_h, target_w
                try:
                    rle_resized = falcon_metrics.resize_rle(rle, orig_h, orig_w)
                except Exception:
                    rle_resized = rle
                rles.append(rle_resized)
            # optional NMS on RLEs (same threshold as pbench)
            if len(rles) > 1:
                try:
                    rles = falcon_metrics.nms(rles, falcon_metrics.NMS_THRESHOLD)
                except Exception:
                    pass
            paired = pair_bbox_entries(getattr(aux, "bboxes_raw", []) or [])
            # produce one bbox per RLE (preferred paired when idx < len(paired), else tight bbox)
            # If detection task (no masks), use paired bboxes directly
            if not rles and paired:
                for b in paired:
                    xyxy = normalized_bbox_to_xyxy(b, img_w, img_h)
                    pred_by_cls[cls_id].append(xyxy)
            else:
                for idx, rle in enumerate(rles):
                    if idx < len(paired):
                        xyxy = normalized_bbox_to_xyxy(paired[idx], img_w, img_h)
                    else:
                        xyxy = rle_to_xyxy(rle)
                        if xyxy is None:
                            continue
                    # Already at original res; clamp
                    xyxy = [max(0, min(float(xyxy[0]), img_w)), max(0, min(float(xyxy[1]), img_h)),
                            max(0, min(float(xyxy[2]), img_w)), max(0, min(float(xyxy[3]), img_h))]
                    if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                        pred_by_cls[cls_id].append(xyxy)

        per_cls = evaluate_one(gt_by_cls, pred_by_cls, args.iou)
        for c,(tp,fp,fn) in per_cls.items():
            total_per_cls[c][0]+=tp; total_per_cls[c][1]+=fp; total_per_cls[c][2]+=fn
            bucket_per_cls[bucket][c][0]+=tp; bucket_per_cls[bucket][c][1]+=fp; bucket_per_cls[bucket][c][2]+=fn
        n_matched+=1
        if args.verbose:
            gt_n = sum(len(v) for v in gt_by_cls.values()); pred_n = sum(len(v) for v in pred_by_cls.values())
            tp_sum = sum(v[0] for v in per_cls.values())
            print(f"[{bucket}] {demo_name}: GT={gt_n} pred={pred_n} TP={tp_sum} split={split}")

        per_image_rows.append({
            "image": str(img_path.as_posix()), "demo_name": demo_name, "bucket": bucket, "split": split,
            "merged_name": merged_name,
            "gt_count": sum(len(v) for v in gt_by_cls.values()),
            "pred_count": sum(len(v) for v in pred_by_cls.values()),
            "tp": sum(v[0] for v in per_cls.values()),
            "fp": sum(v[1] for v in per_cls.values()),
            "fn": sum(v[2] for v in per_cls.values()),
        })

    def metrics_from(tp,fp,fn):
        prec = tp/(tp+fp) if tp+fp>0 else 0.0
        rec = tp/(tp+fn) if tp+fn>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
        acc = tp/(tp+fp+fn) if tp+fp+fn>0 else 0.0
        return prec,rec,f1,acc

    total_tp = sum(v[0] for v in total_per_cls.values()); total_fp = sum(v[1] for v in total_per_cls.values()); total_fn = sum(v[2] for v in total_per_cls.values())
    overall_prec, overall_rec, overall_f1, overall_acc = metrics_from(total_tp,total_fp,total_fn)

    per_class={}
    model_names = ordered
    for c in range(nc):
        tp,fp,fn = total_per_cls.get(c,[0,0,0])
        prec,rec,f1,acc = metrics_from(tp,fp,fn)
        per_class[str(c)]={"name": ordered[c] if c < len(ordered) else str(c), "tp":int(tp),"fp":int(fp),"fn":int(fn),"support":int(tp+fn),"precision":prec,"recall":rec,"f1":f1,"accuracy":acc}
    buckets={}
    for bucket,d in bucket_per_cls.items():
        btp=sum(v[0] for v in d.values()); bfp=sum(v[1] for v in d.values()); bfn=sum(v[2] for v in d.values())
        prec,rec,f1,acc = metrics_from(btp,bfp,bfn)
        buckets[bucket]={"tp":int(btp),"fp":int(bfp),"fn":int(bfn),"precision":prec,"recall":rec,"f1":f1,"accuracy":acc}

    result={
        "weights": args.hf_local_dir or args.hf_model_id or "tiiuae/Falcon-Perception",
        "task": args.task,
        "source": str(source), "data": str(data_root), "manifest": str(manifest_path),
        "iou_threshold": args.iou, "model_names": model_names, "queries": ordered_queries,
        "n_demo_images": len(images), "n_matched_with_gt": n_matched, "n_unmatched_no_gt": n_unmatched,
        "unmatched": unmatched_list,
        "overall": {"tp":int(total_tp),"fp":int(total_fp),"fn":int(total_fn),"precision":overall_prec,"recall":overall_rec,"f1":overall_f1,"accuracy":overall_acc},
        "per_class": per_class, "by_bucket": buckets, "per_image": per_image_rows,
    }

    print("\n"+"="*78)
    print(f"Falcon demo-pics eval  task={args.task} IoU={args.iou}  {n_matched}/{len(images)} matched ({n_unmatched} unmatched)")
    print(f"  Overall: P={overall_prec:.3f} R={overall_rec:.3f} F1={overall_f1:.3f} Acc(J)={overall_acc:.3f} (TP={total_tp} FP={total_fp} FN={total_fn})")
    for bucket,m in buckets.items():
        print(f"  [{bucket:12s}] P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Acc={m['accuracy']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    print("-"*78)
    print(f"{'cls':>3}  {'name':<12} {'sup':>5} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
    print("-"*78)
    for c in range(nc):
        r=per_class[str(c)]
        print(f"{c:>3}  {r['name']:<12} {r['support']:>5} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}  {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['accuracy']:>6.3f}")
    print("="*78)
    if n_unmatched:
        print(f"\nUnmatched (no GT, excluded): {n_unmatched}")
        for u in unmatched_list[:20]: print(f"  - {u}")
        if len(unmatched_list)>20: print(f"  ... and {len(unmatched_list)-20} more")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"w") as f: json.dump(result,f,indent=2)
    print(f"\nSaved JSON -> {out_path}")
    csv_out = Path(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")
    with open(csv_out,"w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["class_id","name","support","tp","fp","fn","precision","recall","f1","accuracy"])
        for c in range(nc):
            r=per_class[str(c)]
            w.writerow([c,r["name"],r["support"],r["tp"],r["fp"],r["fn"],f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1']:.4f}",f"{r['accuracy']:.4f}"])
        w.writerow([])
        w.writerow(["overall","",total_tp+total_fn,total_tp,total_fp,total_fn,f"{overall_prec:.4f}",f"{overall_rec:.4f}",f"{overall_f1:.4f}",f"{overall_acc:.4f}"])
    print(f"Saved CSV  -> {csv_out}")


if __name__ == "__main__":
    main()
