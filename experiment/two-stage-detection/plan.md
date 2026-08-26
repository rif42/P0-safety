# Two-Stage Detection Plan — YOLOv26 Person + SAHI Negative-Class Violation (no retrain, no algorithmic ROI)

> Revised 2026-08-26 — drops pose/head-ROI/Hungarian algorithmic path (B). Leverages `NO-Hardhat / NO-Mask / NO-Safety Vest` already well-trained in Kaggle `best.pt`. No retraining.

## Why this is simpler

- Previous B (`Hardhat + yolo26s-pose + head_roi + center-in-box/IoU`) exists because `no_boots 1.8% mAP` suggested negative classes are unlearnable. **On this Roboflow `data/css-data` they are learnable:** A `NO-Hardhat` class alone hits **valid F1 0.93 Prec 0.971 Rec 0.892 / test F1 0.889 Prec 1.00 Rec 0.80**, beating tuned B `F1 0.727/0.719 Prec 0.58/0.59` at 0% abstention. Locked config `HEAD_PAD 2.0 CONF_THR 0.40 KPT_CONF 0.30 IoU 0.05 imgsz 640` + `yolo26x-pose` gives no gain — bottleneck is missing Hardhat boxes (52/114 valid have 0 boxes even at `conf 0.05`, 7-15px), not pose.
- Dropping pose removes: `yolo26*-pose.pt`, `KPT_CONF_THR`, `HEAD_PAD`, `head_roi_for_person()`, `is_helmet_on_head()` tuning, `top-15%` fallback, `IoU<0.5` Person union reasoning. Violation = presence of a `NO-*` box. Positive classes (`Hardhat/Mask/Safety Vest/Person`) still come for free from the same `best.pt` forward pass — we just don't need them for the verdict (use for audit/logging only).
- SAHI stays relevant: `NO-*` also misses small/distant violations (A Rec 0.89→0.80), same small-object cause as Hardhat miss. Slicing recovers native pixels per tile.

## Pipeline — how it works now (start with Person, then what)

```
Image (e.g. 1920x1080)
  |
  +----> [Stage 1: Person — pretrained COCO yolo26s.pt]  (better Person than Kaggle yolov8)
  |       YOLO("yolo26s.pt")(img, conf=0.25) -> Person boxes
  |       COCO Person = 100k+ persons, better Recall/AP_small on distant/back/occluded than
  |       construction-finetuned yolov8. No pose, no keypoints. Optional yolo26m.pt ablation.
  |
  +----> [Stage 2: Violation — Kaggle best.pt (10-class) via SAHI]  (single weights, 10 classes)
          best.pt already outputs both positive (Hardhat/Mask/Safety Vest) AND negative
          (NO-Hardhat/NO-Mask/NO-Safety Vest) in one pass. Split by class name:
            violation = any NO-* box with conf >= 0.25 (class-specific thr later)
            positives ignored for verdict, kept for audit log
          SAHI: slice full frame 640x640 overlap 0.2 (~4-9 tiles at 1280-equivalent),
          batch through best.pt, map boxes back, NMS iou 0.5. Finds tiny NO-Hardhat that
          vanish at 640 resize (e.g. 4_jpg.rf.* 7-8px). Cost ~30ms -> 90-150ms (3-5x).
  |
  v
[Association — trivial, not algorithmic]
  Image-level (simplest, default):  violation = exists(NO-Hardhat) OR exists(NO-Mask) OR ...
  Per-person (for Exception Log):   assign each NO-* box to nearest Person by center distance
                                   (or IoU>0.1). No head ROI, no Hungarian — one NO-* box = one
                                   person flagged. If no Person box, image-level flag still fires.
  |
  v
Verdict: image violation flag + per-person rows (person_idx, nearest NO-* class/conf, verdict)
         positives logged alongside for traceability but not used to decide.
```

Yes — Option A still has positive detection: `best.pt` is 10-class, so one forward pass returns `Hardhat` AND `NO-Hardhat` etc. We simplify by **reading only `NO-*` for the decision**.

1. Phase 1 — Freeze baseline and confirm NO-* wins (0.5 day)
   - lock `valid` (114, 37 viol) as tuning split, `test` (82, 25 viol) held-out; re-snapshot `python experiment/algo-pose/eval_helmet.py --split valid/test --conf-a 0.25` → A `F1 0.93/0.889 Prec 0.971/1.00 Rec 0.892/0.80` vs B tuned `F1 0.727/0.719`
   - compute box-area histogram on `train/labels` + `AP_small(<32px)` for `NO-Hardhat` to quantify SAHI target (same 7-15px regime as Hardhat miss); log `no_person_detected` 26/114 are 25 genuinely no-Person GT → not a Person-miss problem
   - baseline Person `Recall/mAP@50/AP_small` for Kaggle Person vs `yolo26s.pt` Person on valid — isolates Person gain before SAHI

2. Phase 2 — Swap Person to pretrained YOLOv26 + direct NO-* signal (1 day, no SAHI yet)
   - fetch `yolo26s.pt` via `ultralytics` (`YOLO("yolo26s.pt")`, ~20MB), keep `yolo26m.pt` as ablation; add `--person-model yolo26s.pt --person-conf 0.25` to `eval_helmet.py` (extend, don't fork)
   - run violation as `any NO-* box conf>=0.25` — single full-frame pass on `best.pt` first (no tiling) to measure Person-swap delta alone; sweep `person-conf 0.25 vs 0.35` and `no-conf 0.25 vs 0.40` on `valid` only
   - wire trivial per-person assignment: `for each NO-* box, attach to nearest Person center` (Euclidean, no threshold) for Exception Log; keep image-level flag as primary metric; delete pose code path from eval (keep behind flag for comparison only)

3. Phase 3 — Add SAHI for NO-* (1.5 days, no retrain)
   - implement `predict_A_sahi()` / `predict_negative_sahi()`: SAHI tiling `slice 640 overlap 0.2` over `best.pt`, full-frame default + `per-person-crop` ablation (1-4 crops cheaper); map back + NMS 0.5; same `NO-* conf` thresholds as Phase 2
   - ponytail: try `pip install sahi` → `SlicedInference`; if unavailable, ~30-line manual `cv2` crop loop + batched `YOLO` — no new dep for a loop
   - evaluate on `valid` only: `NO-*` recovery on A false-negatives (4 valid / 5 test), violation `Prec/Rec/F1` for `A_sahi` and `Person_y26 + NO-*_sahi` + per-person precision, latency `ms/img`; tune `overlap/slice` once; keep positives logged but not scored

4. Phase 4 — Held-out test and ship (0.5 day)
   - one-shot `test` run of winners (`Person_y26 + NO-*` and `Person_y26 + NO-*_sahi`) at locked thresholds — no tuning on `test`; emit `eval_helmet_results_{valid,test}.json` + `eval_helmet_debug_{valid,test}.csv` with `reason` tags (`no_box / recovered_by_sahi / seam_fp / nearest_person`)
   - decision: if SAHI `Rec gain >=3pp` and `Prec drop <=3pp` at `3-4x` cost → keep `A_sahi` primary, `Person_y26` grounding for per-person log, flag `--sahi` on for wide/crowd/high-res folders, off for bulk; else keep single-pass `NO-*` and leave SAHI on-demand
   - document in `helmet-compliance-validation.md`/`reports/`; expose `python experiment/algo-pose/eval_helmet.py --person-model yolo26s.pt [--sahi --sahi-slice 640 --sahi-overlap 0.2]` for prod

## Rough precision improvement (no-retrain, dataset-grounded)

- **YOLOv26 Person alone (no SAHI):** Person `Recall ~0.88->0.94, AP_small +6-10pp` vs Kaggle Person on COCO, but image-level violation `Prec ~flat` (A Prec already 0.97/1.00) — gain is per-person log completeness (fewer orphan NO-* boxes), not image F1. Pointless to chase Person alone if bulk is non-crowded.
- **+ SAHI on NO-* (valid 114, 37 viol):** A `Rec 0.892->0.92-0.95 (+3-6pp)`, `F1 0.93->0.93-0.95 (+0-2pp)`, `Prec 0.971->0.94-0.96 (-1-3pp, tile seam FPs)` conservatively; if SAHI recovers 2-3 of the 4 valid FNs and 2-3 of 5 test FNs, optimistic `test F1 0.889->0.90-0.94`. Previous B-style SAHI estimated `Prec 0.581->0.65-0.70, F1 0.727->0.77-0.82` — now irrelevant since we don't use B; the comparable lift is on A.
- **Per-person with YOLOv26 grounding:** per-person `Precision ~0.94-0.96` (vs 0.58 for Hardhat-ROI path) because NO-* assignment is identity-like — no `hardhat_miss -> false violation` mode; remaining FP are semantic (`cap` scored as `NO-Hardhat`, `vest` vs `NO-Safety Vest` confusion, ~10-15% of valid).
- **Skip gate:** if `valid` SAHI recovers `<10%` of A FNs (<1 image) or `Prec` drops `>5pp` at `>3x` latency, keep single-pass `NO-*` and use SAHI only for drone/wide shots.

## Risks & mitigations

- Negative-class domain shift (other sites label NO-* differently) — mitigated by keeping threshold per-class and person grounding; positives logged for fallback audit.
- SAHI seam duplicates — NMS 0.5 after remap; log seam FP rate separately.
- Latency (thousands of site photos) — default `--sahi` off, enable per-folder; per-person-crop cheaper than full-frame.

## Commands

- `python experiment/algo-pose/eval_helmet.py --split valid --conf-a 0.25` (control A)
- `python experiment/algo-pose/eval_helmet.py --split valid --person-model yolo26s.pt --person-conf 0.25 --conf-a 0.25` (YOLOv26 Person + single-pass NO-*)
- `python experiment/algo-pose/eval_helmet.py --split valid --person-model yolo26s.pt --sahi --sahi-slice 640 --sahi-overlap 0.2 --conf-a 0.25` (full two-stage)
- Outputs: `eval_helmet_results_{valid,test}.json`, `eval_helmet_debug_{valid,test}.csv/.json`, `reports/helmet-eval-*`
