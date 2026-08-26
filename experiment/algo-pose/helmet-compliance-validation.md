# Helmet Compliance Validation — Valid vs Test (image-level NO-Hardhat)

Generated: 2026-08-25. Updated: 2026-08-26 (finetuning). Code: `eval_helmet.py` (A = pretrained 10-class `NO-Hardhat`, B = Hardhat-only + pose head-anchor). Primary: **valid** (114). Test (82) held out. Ground truth: any `NO-Hardhat` label → violation (`data/css-data/{valid,test}/labels`).

## Baseline (2026-08-25) — `HEAD_PAD=1.4 CONF_THR=0.40 KPT_CONF_THR=0.50` + `yolo26s-pose.pt`

### Valid (114 images, 37 with NO-Hardhat GT)

| Method | F1 | Prec | Rec | TP | FP | TN | FN | N | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A NO-Hardhat** (conf_a=0.25) | **0.930** | 0.971 | 0.892 | 33 | 1 | 76 | 4 | 114 | pretrained 10-class |
| B filtered (excl. inconc) | 0.791 | 0.654 | 1.000 | 34 | 18 | 11 | 0 | 63 | excl 51 inconc (44.7%) |
| B inconc=viol | 0.529 | 0.359 | 1.000 | 37 | 66 | 11 | 0 | 114 | safety-first |

### Test — held out (82 images)

| Method | F1 | Prec | Rec | TP | FP | TN | FN | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A NO-Hardhat** | **0.889** | 1.000 | 0.800 | 20 | 0 | 57 | 5 | 82 |
| B filtered | 0.702 | 0.541 | 1.000 | 20 | 17 | 4 | 0 | 41 |
| B inconc=viol | 0.485 | 0.321 | 1.000 | 25 | 53 | 4 | 0 | 82 |

Artifacts (baseline): `eval_helmet_results_valid.json`, `eval_helmet_results_test.json`. Neat report: `reports/helmet-eval-valid-2026-08-25.md`.

---

## Finetuning (2026-08-26) — making B decidable

### What was changed in `eval_helmet.py`

- **Abstract `eval_helmet.py` knobs** — `HEAD_PAD`, `CONF_THR`, `KPT_CONF_THR`, `imgsz`, `iou_thr`, `pose`, `use_seg_fallback` are now CLI flags (e.g. `--conf-thr 0.40 --kpt-conf-thr 0.30 --head-pad 2.0 --iou-thr 0.05 --imgsz 640`). Defaults updated to the locked config below.
- **`no_person_detected` no longer abstains** — `predict_B` returns `False` (compliant) instead of `None` (inconclusive) when no `Person` is detected. 25/26 inconclusive in valid at 640 are `Safety Cone`/`machinery` images with `Person=False` GT; abstaining was noise. This alone cuts valid abstention 26/114 → 0.
- **Seg-Person fallback is decidable** — when pose finds nothing but `seg Person conf≥0.25` exists, build a geometric head ROI (`cx=(x1+x2)/2, cy=y1+h*0.075, half=max(w*0.18,h*0.12)*HEAD_PAD`) and run `is_helmet_on_head` instead of returning inconclusive. Pose-miss cases become decidable.
- **Pose+seg union** — keep all `yolo26*-pose` persons and append seg persons with `IoU<0.5` to any pose person (adds distant/small persons pose misses). Avoids dropping persons when pose fails.
- **IoU fallback for helmet matching** — `is_helmet_on_head` now: center-in-box OR `IoU(head_roi, helmet_box) ≥ iou_thr` (helps when a tiny helmet center sits just outside the head box). Default `iou_thr=0.05`.
- **Per-image CSV/JSON + reason tags** — `--dump-csv` / `eval_helmet_debug_{valid,test}.csv/.json` with `reason` (`hardhat_miss` / `roi_miss_or_missing_helmet` / `all_compliant` / `no_person_detected`) and `per_person` for attribution.

### Sweep on `valid` only (test held out, no tuning on test)

Grid over `valid` (114) with fallback ON:

| Knob | Finding |
|---|---|
| `CONF_THR 0.40 → 0.10` | **No gain** — 52/114 `hardhat_miss` images have 0 `Hardhat` boxes even at 0.05 conf (e.g. `4_jpg.rf.*` with 3 tiny 7–8 px GT helmets). Lowering the threshold does not create boxes. |
| `KPT_CONF_THR 0.50 → 0.30` | `0.714 → 0.727` — recovers 1 viol FN by using weak head kpts instead of geometric fallback. |
| `HEAD_PAD 1.4 → 2.0 → 2.5` | `0.66 → 0.68` at `2.0`; `2.5` no further gain. |
| `IoU 0.05` | `0.68 → 0.714–0.727` — fixes offset tiny helmets where center just misses the head box. |
| `imgsz 640 → 960 → 1280` | Finds some 7–15 px helmets at 960 (`4_jpg` found at 960 conf 0.30) but **hurts overall**: `A prec 0.971→0.917`, `B 0.727→0.707` at locked config. Locked `640`. |

**Locked config (new defaults):** `HEAD_PAD=2.0 CONF_THR=0.40 KPT_CONF_THR=0.30 iou_thr=0.05 imgsz=640 seg_fallback=True pose=yolo26s-pose.pt`

### Tuned results — locked config, 0% abstention

Valid and test re-run with no tuning on test:

| Split | Method | F1 | Prec | Rec | TP | FP | TN | FN | N | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| valid 114 (37 viol) | A NO-Hardhat | **0.930** | 0.971 | 0.892 | 33 | 1 | 76 | 4 | 114 | — |
| valid 114 | **B tuned (s, locked)** | 0.727 | 0.581 | 0.973 | 36 | 26 | 51 | 1 | 114 | **0%** |
| test 82 (25 viol) | A NO-Hardhat | **0.889** | 1.000 | 0.800 | 20 | 0 | 57 | 5 | 82 | — |
| test 82 | **B tuned (s, locked)** | 0.719 | 0.590 | 0.920 | 23 | 16 | 41 | 2 | 82 | **0%** |

B reason hist (tuned, valid): `hardhat_miss 52, all_compliant 26, no_person_detected 26, roi_miss_or_missing_helmet 10`. Test: `hardhat_miss 31, no_person_detected 22, all_compliant 21, roi_miss 8`.

Key deltas vs baseline B: abstention **44.7% → 0%** (valid) / **50% → 0%** (test); F1 on valid drops 0.79→0.73 because the former 51 inconclusive hid 25 clean TNs that are now correctly counted as TN, but the 26 FP from detector miss are now visible. B now catches **36/37** violations on valid (1 FN left: `681_jpg.rf.*` which is also `seg Person=0` — genuinely no person detected, counted as compliant for image-level task; treat as review-queue if strict safety-first is needed).

### Pose-model ablation — what if we use `yolo26x-pose`?

`yolo26x-pose.pt` (121 MB, auto-downloaded from `ultralytics/assets v8.4.0`) tested at the locked config alongside `yolo26s-pose.pt` (24 MB) and `yolo26m-pose.pt` (47 MB).

| Split | Pose | F1 | Prec | Rec | TP | FP | TN | FN | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| valid | **yolo26s-pose** | **0.727** | 0.581 | 0.973 | 36 | 26 | 51 | 1 | **kept — best valid** |
| valid | yolo26m-pose | 0.722 | 0.583 | 0.946 | 35 | 25 | 52 | 2 | 1 extra FN vs s |
| valid | yolo26x-pose | 0.714 | 0.574 | 0.946 | 35 | 26 | 51 | 2 | 1 extra FN vs s (`ppe_0355` becomes FN, `681` still FN) |
| test | yolo26s-pose | **0.719** | 0.590 | 0.920 | 23 | 16 | 41 | 2 |  |
| test | yolo26x-pose | 0.719 | 0.590 | 0.920 | 23 | 16 | 41 | 2 | tie with s |

**No material gain from `yolo26x-pose` here.** The remaining errors are **detector-limited** (`Hardhat` miss → false violation, `_png`/`construction` images with no box at any threshold), not pose-limited. `yolo26x` costs ~5× weights and ~2× inference vs `s` for — at best — parity and at worst —1 TP on valid. Keep `yolo26s-pose.pt` as default; switch via `--pose yolo26x-pose.pt` if you want to reproduce the ablation.

```
python eval_helmet.py --split valid --pose yolo26x-pose.pt
python eval_helmet.py --split test  --pose yolo26x-pose.pt
```

## Updated verdict (2026-08-26)

**A still wins (valid 0.93 vs tuned B 0.73; test 0.89 vs 0.72).** Finetuning makes B **fully decidable (0% abstention)** and +0.02 on test vs baseline filtered, but the ~0.20 F1 gap remains because the 10-class `Hardhat` detector misses helmets — especially tiny/distant ones (7–15 px boxes at 640). No threshold/IoU/HEAD_PAD/pose-size knob closes that; `CONF_THR 0.25` as previously hypothesized does not help since the boxes simply do not exist.

**Recommendation (unchanged):** keep **A as primary**. Use **B as verifier / human-review queue** — tuned B catches 3 of A's 4 FN on valid (`autox3_mp4-78` now decidable) while keeping 0% abstention; optionally route `no_person_detected (26 valid / 22 test)` to review if safety-first demands.

**Next step to make B truly competitive (deferred):** retrain a **7-class Hardhat-only detector** from `data/css-data/data.yaml` dropping `NO-Hardhat`/`NO-Mask`/`NO-Safety Vest`, or train at `imgsz 960` / with small-object/tiling augmentation. Pose side is not the bottleneck.

Artifacts (tuned): `eval_helmet_results_valid.json`, `eval_helmet_results_test.json`, `eval_helmet_debug_valid.csv/.json`, `eval_helmet_debug_test.csv/.json`. Re-run: `python eval_helmet.py --split valid --conf-a 0.25` / `--split test` (tuned defaults baked in).

## How to read B

- Use `no_person_detected` / low-person-count images as a **review list**, not auto-violation (tuned `no_person_detected` is now correctly TN for the image-level `NO-Hardhat` label, but a stricter deployment may still queue them).
- `hardhat_miss` = detector miss → false violation; `roi_miss_or_missing_helmet` = helmet exists but center missed expanded head box (IoU now recovers some).
