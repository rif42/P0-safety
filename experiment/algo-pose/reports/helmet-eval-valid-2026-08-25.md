# Helmet Detection — Valid Split Eval

**Question:** Is the pretrained `NO-Hardhat` class (A) or Hardhat + pose + algorithm without negative classes (B) better at detecting no-helmet?

**Run:** `python scripts/eval_helmet.py --split valid --conf-a 0.25`
**Models:** `data/results_yolov8n_100e/.../best.pt` (10-class) + `yolo26s-pose.pt` · **Thresholds:** `HEAD_PAD=1.4` `CONF_THR=0.40` `KPT_CONF_THR=0.50`
**Ground truth:** image-level violation = any `NO-Hardhat` label. **Valid:** 114 images (37 violation / 77 clean).

## Result

| Method | F1 | Precision | Recall | TP | FP | TN | FN | N |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A — NO-Hardhat class** | **0.930** | **0.971** | 0.892 | 33 | **1** | 76 | 4 | 114 | |
| B — Hardhat+pose (excl. inconclusive) | 0.791 | 0.654 | **1.000** | 34 | 18 | 11 | 0 | 63 | 51 inconclusive excluded |
| B — inconclusive = violation | 0.529 | 0.359 | 1.000 | 37 | 66 | 11 | 0 | 114 | safety-first scoring |

B abstention: **44.7%** (51/114) — body detected but head pose missing.

## Verdict

**A wins (F1 0.93 vs 0.79).** The negative class is well-trained on this dataset. B trades precision for perfect recall — 18 false violations because the Hardhat detector misses the helmet, so a compliant person is called violation.

For reference, held-out `test` (82 images): A F1 0.89 vs B 0.70 filtered, same pattern.

## How to read B

- Use `inconclusive` as a **human-review queue**, not auto-violation (F1 drops to 0.53 if you do).
- Path to make B competitive: lower `CONF_THR` to 0.25, add seg-`Person` fallback for pose misses, retrain a 7-class Hardhat-only detector.

<details>
<summary>Raw output</summary>

```
split valid: 114 images
A seg: data\results_yolov8n_100e\kaggle\working\runs\detect\train\weights\best.pt  B pose: yolo26s-pose.pt

=== Image-level violation (NO-Hardhat present) ===
A (NO-Hardhat class): tp=33 fp=1 tn=76 fn=4  prec=0.971 rec=0.892 f1=0.930 n=114
B filtered (exclude inconclusive 51/114=44.7%): tp=34 fp=18 tn=11 fn=0  prec=0.654 rec=1.000 f1=0.791 n=63
B inconc-as-violation: tp=37 fp=66 tn=11 fn=0  prec=0.359 rec=1.000 f1=0.529
```

Artifacts: `scripts/eval_helmet_results_valid.json` (+ `_test.json` for test).

</details>
