"""Eval A vs B on valid/test: image-level violation detection for no-helmet.

A: pretrained 10-class (best.pt) — any NO-Hardhat box -> violation
B: Hardhat-only + pose head anchor + center-in-box — inconclusive kept separate

Ground truth: label-direct image-level (any NO-Hardhat label -> positive)

Improvements over baseline (to make B standalone competitive):
- Configurable CONF_THR / KPT_CONF_THR / HEAD_PAD / imgsz via CLI
- yolo26m-pose.pt support (larger pose model reduces pose-miss inconclusive)
- Seg-Person fallback is now *decidable*: geometric head ROI (top 15% of box)
  instead of auto-inconclusive. Only "no person at all" stays inconclusive.
- Pose head-miss also uses geometric fallback instead of abstaining.
- Optional IoU fallback for helmet matching (helps tiny/distant helmets).
- Per-image CSV/JSON dump with reason tags for failure attribution.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# experiment/algo-pose is 2 levels below repo root
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "css-data"
CLASS_NAMES = ['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']

# defaults — overridable via CLI
HEAD_PAD = 2.0
CONF_THR = 0.40
KPT_CONF_THR = 0.30
HEAD_IDX = [0,1,2,3,4]


def head_roi_for_person(kpt_xy, kpt_conf, person_xyxy, kpt_conf_thr=KPT_CONF_THR, head_pad=HEAD_PAD):
    vis = kpt_conf[HEAD_IDX] > kpt_conf_thr
    if vis.sum() >= 1:
        pts = kpt_xy[HEAD_IDX][vis]
        cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
        eye_ok = (kpt_conf[1] > kpt_conf_thr and kpt_conf[2] > kpt_conf_thr)
        if eye_ok:
            eye_dist = float(np.linalg.norm(kpt_xy[1] - kpt_xy[2]))
            half = max(eye_dist * 0.90, 8)
        else:
            half = max(float(np.ptp(pts[:,0])), float(np.ptp(pts[:,1]))) * 0.75 + 8 if len(pts) > 1 else 12
        bw = float(person_xyxy[2]-person_xyxy[0])
        half = max(half, bw * 0.12)
        half *= head_pad
        return [cx-half, cy-half, cx+half, cy+half], (cx,cy), True, False
    x1,y1,x2,y2 = map(float, person_xyxy)
    h = y2 - y1
    cx, cy = (x1+x2)/2, y1 + h*0.075
    half = max((x2-x1)*0.18, h*0.12) * head_pad
    return [cx-half, cy-half, cx+half, cy+half], (cx,cy), False, True


def is_helmet_on_head(head_xyxy, helmet_boxes, iou_thr=None):
    """Center-in-box primary; optionally also accept IoU > iou_thr."""
    hx1, hy1, hx2, hy2 = head_xyxy
    hw, hh = max(1, hx2-hx1), max(1, hy2-hy1)
    head_area = hw * hh
    for x1,y1,x2,y2,conf in helmet_boxes:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
            return True, (cx,cy), conf
        if iou_thr is not None:
            # IoU fallback for small/offset helmets
            ix1, iy1 = max(hx1,x1), max(hy1,y1)
            ix2, iy2 = min(hx2,x2), min(hy2,y2)
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw*ih
            helm_area = max(1,(x2-x1)*(y2-y1))
            union = head_area + helm_area - inter
            iou = inter/union if union>0 else 0
            if iou >= iou_thr:
                return True, (cx,cy), conf
    return False, None, None


def parse_valid_gt_has_no(valid_label_path):
    if not valid_label_path.exists():
        return False
    for line in valid_label_path.read_text().splitlines():
        if not line.strip():
            continue
        cid = int(line.split()[0])
        if CLASS_NAMES[cid] == "NO-Hardhat":
            return True
    return False


def predict_A(seg_model, img_path, conf_thr=0.25, imgsz=640):
    res = seg_model(str(img_path), verbose=False, imgsz=imgsz)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return False, 0
    for i in range(len(res.boxes)):
        cid = int(res.boxes.cls[i].item())
        name = seg_model.names[cid]
        conf = float(res.boxes.conf[i].item())
        if name == "NO-Hardhat" and conf >= conf_thr:
            return True, conf
    return False, 0


def predict_B(seg_model, pose_model, img_path, conf_thr=CONF_THR, kpt_conf_thr=KPT_CONF_THR, head_pad=HEAD_PAD, imgsz=640, iou_thr=0.05, use_seg_fallback=True):
    """Returns (verdict, n_persons, n_inconc, helmet_boxes, debug).

    verdict: True=violation, False=compliant, None=inconclusive
    debug: dict with reason tags for attribution
    """
    seg_res = seg_model(str(img_path), verbose=False, imgsz=imgsz)[0]
    pose_res = pose_model(str(img_path), verbose=False, imgsz=imgsz)[0]
    helmet_boxes = []
    if seg_res.boxes is not None:
        for i in range(len(seg_res.boxes)):
            cid = int(seg_res.boxes.cls[i].item())
            name = seg_model.names[cid]
            conf = float(seg_res.boxes.conf[i].item())
            if name == "Hardhat" and conf >= conf_thr:
                helmet_boxes.append([*seg_res.boxes.xyxy[i].tolist(), conf])

    # collect seg Person boxes for fallback
    seg_persons = []
    if seg_res.boxes is not None:
        for i in range(len(seg_res.boxes)):
            cid = int(seg_res.boxes.cls[i].item())
            if seg_model.names[cid] == "Person":
                # use seg conf threshold ~0.25 to avoid noise
                conf = float(seg_res.boxes.conf[i].item())
                if conf >= 0.25:
                    seg_persons.append(seg_res.boxes.xyxy[i].tolist())

    # Merge pose persons + seg persons (union by IoU) to avoid missing persons when pose fails on small/distant
    # Keep pose persons as primary; add seg persons that don't overlap pose persons (IoU<0.5)
    pose_persons_xyxy = None
    persons_xyxy = None
    kpts = None
    kconf = None
    has_pose_persons = pose_res.boxes is not None and len(pose_res.boxes) > 0 and pose_res.keypoints is not None and len(pose_res.keypoints) > 0
    debug = {"has_pose_persons": bool(has_pose_persons), "n_seg_persons": len(seg_persons), "n_helmet_boxes": len(helmet_boxes)}

    if has_pose_persons:
        pose_persons_xyxy = pose_res.boxes.xyxy.cpu().numpy()
        pose_kpts = pose_res.keypoints.xy.cpu().numpy()
        pose_kconf = pose_res.keypoints.conf.cpu().numpy() if pose_res.keypoints.conf is not None else np.ones((len(pose_kpts),17))
        if use_seg_fallback and seg_persons:
            # Add seg persons that don't overlap any pose person (IoU<0.5)
            seg_arr = np.array(seg_persons)
            extra = []
            for sb in seg_arr:
                ious = []
                for pb in pose_persons_xyxy:
                    ix1=max(float(sb[0]),float(pb[0])); iy1=max(float(sb[1]),float(pb[1]))
                    ix2=min(float(sb[2]),float(pb[2])); iy2=min(float(sb[3]),float(pb[3]))
                    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih
                    a=(sb[2]-sb[0])*(sb[3]-sb[1]); b=(pb[2]-pb[0])*(pb[3]-pb[1])
                    iou=inter/max(1,a+b-inter)
                    ious.append(iou)
                if not ious or max(ious) < 0.5:
                    extra.append(sb)
            if extra:
                extra = np.array(extra)
                persons_xyxy = np.vstack([pose_persons_xyxy, extra])
                kpts = np.vstack([pose_kpts, np.zeros((len(extra),17,2))])
                kconf = np.vstack([pose_kconf, np.zeros((len(extra),17))])
                debug["n_extra_seg_persons"] = len(extra)
            else:
                persons_xyxy = pose_persons_xyxy; kpts = pose_kpts; kconf = pose_kconf
        else:
            persons_xyxy = pose_persons_xyxy; kpts = pose_kpts; kconf = pose_kconf
    else:
        if use_seg_fallback and seg_persons:
            persons_xyxy = np.array(seg_persons)
            kpts = np.zeros((len(persons_xyxy), 17, 2))
            kconf = np.zeros((len(persons_xyxy), 17))
            debug["fallback"] = "seg_person_geometric"
        else:
            debug["reason"] = "no_person_detected" if not seg_persons else "no_pose_no_fallback"
            return False, 0, 0, helmet_boxes, debug

    inconc = 0
    violation = False
    per_person_reasons = []
    for i in range(len(persons_xyxy)):
        head_xyxy, head_ctr, head_det, is_fallback = head_roi_for_person(kpts[i], kconf[i], persons_xyxy[i], kpt_conf_thr=kpt_conf_thr, head_pad=head_pad)
        # If head_det is False we now use geometric fallback ROI instead of abstaining
        # Only count as inconclusive if we have no box at all (should not happen)
        worn, _, conf = is_helmet_on_head(head_xyxy, helmet_boxes, iou_thr=iou_thr)
        if worn:
            per_person_reasons.append("helmet_found" if head_det else "helmet_found_fallback_head")
        else:
            # no helmet matched — determine why
            if len(helmet_boxes) == 0:
                per_person_reasons.append("hardhat_miss_no_boxes" if head_det else "hardhat_miss_no_boxes_fallback_head")
            else:
                per_person_reasons.append("hardhat_miss_roi_miss" if head_det else "hardhat_miss_roi_miss_fallback_head")
            violation = True

    debug["per_person_reasons"] = per_person_reasons
    # With fallback, we no longer abstain just because head kpts missing
    # Only inconclusive if no persons at all (handled above)
    # Keep a small inconclusive path for backwards compat: if we disabled fallback, count head misses
    if not use_seg_fallback:
        # legacy: count head_det failures as inconc
        inconc = sum(1 for r in per_person_reasons if "fallback" in r or "no_head" in r)
        if inconc == len(persons_xyxy) and inconc > 0:
            debug["reason"] = "all_no_head_kpt"
            return None, len(persons_xyxy), inconc, helmet_boxes, debug

    # Determine image-level reason tag
    if violation:
        # image is violation if any person lacks helmet
        if len(helmet_boxes) == 0:
            debug["reason"] = "hardhat_miss"
        else:
            debug["reason"] = "roi_miss_or_missing_helmet"
    else:
        debug["reason"] = "all_compliant"

    return violation, len(persons_xyxy), inconc, helmet_boxes, debug


def compute_metrics(y_true, y_pred, inconc_mask=None):
    """y_true/pred bool lists. inconc_mask: True where B abstains."""
    if inconc_mask is not None:
        filt_true = [t for t, m in zip(y_true, inconc_mask) if not m]
        filt_pred = [p for p, m in zip(y_pred, inconc_mask) if not m]
    else:
        filt_true, filt_pred = y_true, y_pred
    tp = sum(1 for t,p in zip(filt_true,filt_pred) if t and p)
    fp = sum(1 for t,p in zip(filt_true,filt_pred) if not t and p)
    tn = sum(1 for t,p in zip(filt_true,filt_pred) if not t and not p)
    fn = sum(1 for t,p in zip(filt_true,filt_pred) if t and not p)
    prec = tp / (tp+fp) if (tp+fp)>0 else 0
    rec = tp / (tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return {"tp":tp,"fp":fp,"tn":tn,"fn":fn,"precision":prec,"recall":rec,"f1":f1, "n":len(filt_true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(DATA_DIR.parent / "results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt"))
    ap.add_argument("--pose", default="yolo26s-pose.pt")
    ap.add_argument("--split", default="valid", choices=["valid","test"])
    ap.add_argument("--conf-a", type=float, default=0.25, help="NO-Hardhat conf for A")
    ap.add_argument("--conf-thr", type=float, default=CONF_THR, help="Hardhat conf for B")
    ap.add_argument("--kpt-conf-thr", type=float, default=KPT_CONF_THR)
    ap.add_argument("--head-pad", type=float, default=HEAD_PAD)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou-thr", type=float, default=0.05, help="optional IoU fallback threshold, e.g. 0.05 (None to disable)")
    ap.add_argument("--no-seg-fallback", action="store_true", help="disable seg-Person geometric fallback (legacy abstention)")
    ap.add_argument("--dump-csv", type=str, default=None, help="path to dump per-image CSV")
    args = ap.parse_args()
    img_dir = DATA_DIR / args.split / "images"
    lbl_dir = DATA_DIR / args.split / "labels"
    seg_model = YOLO(args.seg)
    pose_model = YOLO(args.pose)
    imgs = sorted(img_dir.glob("*"))
    print(f"split {args.split}: {len(imgs)} images")
    print(f"A seg: {args.seg}  B pose: {args.pose}  HEAD_PAD={args.head_pad} CONF_THR={args.conf_thr} KPT_THR={args.kpt_conf_thr} conf_a={args.conf_a} imgsz={args.imgsz} iou_thr={args.iou_thr} seg_fallback={not args.no_seg_fallback}")
    rows=[]
    y_true=[]
    y_pred_a=[]
    y_pred_b=[]
    b_inconc_mask=[]
    debug_rows=[]
    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        gt_viol = parse_valid_gt_has_no(lbl)
        pred_a, _ = predict_A(seg_model, img, conf_thr=args.conf_a, imgsz=args.imgsz)
        pred_b, n_pers, n_inc, helmet_boxes, dbg = predict_B(seg_model, pose_model, img, conf_thr=args.conf_thr, kpt_conf_thr=args.kpt_conf_thr, head_pad=args.head_pad, imgsz=args.imgsz, iou_thr=args.iou_thr, use_seg_fallback=not args.no_seg_fallback)
        is_inc = pred_b is None
        pred_b_for_filter = False if is_inc else pred_b
        y_true.append(gt_viol)
        y_pred_a.append(pred_a)
        y_pred_b.append(pred_b_for_filter)
        b_inconc_mask.append(is_inc)
        rows.append({"image":img.name, "gt_violation":gt_viol, "pred_A":pred_a, "pred_B":pred_b, "b_inconc":is_inc, "n_persons":n_pers, "n_helmet_boxes": len(helmet_boxes), "reason": dbg.get("reason",""), "per_person": ";".join(dbg.get("per_person_reasons",[]))})
        debug_rows.append({"image":img.name, "gt":gt_viol, "pred_B":pred_b, "inconc":is_inc, "n_persons":n_pers, "n_helmet":len(helmet_boxes), "debug":dbg})

    m_a = compute_metrics(y_true, y_pred_a)
    m_b_filtered = compute_metrics(y_true, y_pred_b, inconc_mask=b_inconc_mask)
    y_pred_b_as_viol = [True if m else p for p,m in zip(y_pred_b, b_inconc_mask)]
    m_b_as_viol = compute_metrics(y_true, y_pred_b_as_viol)
    b_inconc_rate = sum(b_inconc_mask)/len(b_inconc_mask) if b_inconc_mask else 0

    print("\n=== Image-level violation (NO-Hardhat present) ===")
    print(f"A (NO-Hardhat class): tp={m_a['tp']} fp={m_a['fp']} tn={m_a['tn']} fn={m_a['fn']}  prec={m_a['precision']:.3f} rec={m_a['recall']:.3f} f1={m_a['f1']:.3f} n={m_a['n']}")
    print(f"B filtered (exclude inconclusive {sum(b_inconc_mask)}/{len(b_inconc_mask)}={b_inconc_rate:.1%}): tp={m_b_filtered['tp']} fp={m_b_filtered['fp']} tn={m_b_filtered['tn']} fn={m_b_filtered['fn']}  prec={m_b_filtered['precision']:.3f} rec={m_b_filtered['recall']:.3f} f1={m_b_filtered['f1']:.3f} n={m_b_filtered['n']}")
    print(f"B inconc-as-violation: tp={m_b_as_viol['tp']} fp={m_b_as_viol['fp']} tn={m_b_as_viol['tn']} fn={m_b_as_viol['fn']}  prec={m_b_as_viol['precision']:.3f} rec={m_b_as_viol['recall']:.3f} f1={m_b_as_viol['f1']:.3f}")

    print("\n--- Disagree GT=viol but B missed (fn) ---")
    for r, t, pb, inc in zip(rows, y_true, y_pred_b, b_inconc_mask):
        gt_viol = t
        b_pred = pb if not inc else None
        if gt_viol and not inc and b_pred == False:
            print(f"  {r['image']} gt=viol B=compliant persons={r['n_persons']} reason={r['reason']}")
        if gt_viol and inc:
            print(f"  {r['image']} gt=viol B=inconclusive persons={r['n_persons']} reason={r['reason']}")
    print("--- Disagree GT=clean but B flags violation (fp) ---")
    for r, t, pb, inc in zip(rows, y_true, y_pred_b, b_inconc_mask):
        if not t and not inc and pb==True:
            print(f"  {r['image']} gt=clean B=violation persons={r['n_persons']} reason={r['reason']}")

    # reason histogram
    from collections import Counter
    hist = Counter(r["reason"] for r in rows)
    print("\n--- B reason histogram ---")
    for k,v in hist.most_common():
        print(f"  {k}: {v}")
    hist_inc = Counter(r["reason"] for r in rows if r["b_inconc"])
    if hist_inc:
        print(" inconclusive breakdown:")
        for k,v in hist_inc.most_common():
            print(f"  {k}: {v}")

    out = {
        "split": args.split,
        "n": len(imgs),
        "metrics": {"A": m_a, "B_filtered": m_b_filtered, "B_as_violation": m_b_as_viol, "B_inconc_rate": b_inconc_rate, "B_inconc_count": sum(b_inconc_mask)},
        "params": {"HEAD_PAD": args.head_pad, "CONF_THR": args.conf_thr, "KPT_CONF_THR": args.kpt_conf_thr, "conf_a": args.conf_a, "imgsz": args.imgsz, "iou_thr": args.iou_thr, "pose": args.pose, "seg_fallback": not args.no_seg_fallback},
        "reason_hist": dict(hist),
    }
    # outputs live next to this script (experiment/algo-pose/)
    out_dir = Path(__file__).resolve().parent
    (out_dir / "eval_helmet_results.json").write_text(json.dumps(out, indent=2))
    (out_dir / f"eval_helmet_results_{args.split}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_dir / 'eval_helmet_results.json'} and {out_dir / f'eval_helmet_results_{args.split}.json'}")

    # per-image dump
    dump_path = args.dump_csv or str(out_dir / f"eval_helmet_debug_{args.split}.csv")
    with open(dump_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image","gt_violation","pred_A","pred_B","b_inconc","n_persons","n_helmet_boxes","reason","per_person"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {dump_path}")
    # also json debug
    (out_dir / f"eval_helmet_debug_{args.split}.json").write_text(json.dumps(debug_rows, indent=2))

    md = f"| method | prec | rec | f1 | tp | fp | tn | fn | n | note |\n|---|---|---|---|---|---|---|---|---|---|\n| A NO-Hardhat | {m_a['precision']:.3f} | {m_a['recall']:.3f} | {m_a['f1']:.3f} | {m_a['tp']} | {m_a['fp']} | {m_a['tn']} | {m_a['fn']} | {m_a['n']} | conf_a={args.conf_a} |\n| B filtered | {m_b_filtered['precision']:.3f} | {m_b_filtered['recall']:.3f} | {m_b_filtered['f1']:.3f} | {m_b_filtered['tp']} | {m_b_filtered['fp']} | {m_b_filtered['tn']} | {m_b_filtered['fn']} | {m_b_filtered['n']} | excl {sum(b_inconc_mask)} inconc |\n| B inconc=viol | {m_b_as_viol['precision']:.3f} | {m_b_as_viol['recall']:.3f} | {m_b_as_viol['f1']:.3f} | {m_b_as_viol['tp']} | {m_b_as_viol['fp']} | {m_b_as_viol['tn']} | {m_b_as_viol['fn']} | {len(imgs)} | safety-first |\nB inconclusive rate: {b_inconc_rate:.1%} ({sum(b_inconc_mask)}/{len(imgs)})\n"
    print("\n" + md)

if __name__ == "__main__":
    main()
