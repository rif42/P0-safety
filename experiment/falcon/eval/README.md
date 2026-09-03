# Falcon Perception — Evaluation

This folder contains benchmark evaluation scripts for Falcon Perception.

## PBench

PBench is a grounded segmentation benchmark with 6 splits of increasing difficulty
(`level_0` → `level_4`) plus a `dense` split (multi-instance scenes).

### Quick start

```bash
# Default: streams level_0 from HuggingFace, first 100 samples
python eval/pbench.py

# Full level_0
python eval/pbench.py --split level_0 --limit 0

# All 6 splits in one run → prints a final summary table
python eval/pbench.py --split all --limit 0

# Save results to disk
python eval/pbench.py --split all --limit 0 --out-dir ./results/pbench/

# Local model export (skip HF download)
python eval/pbench.py --hf-local-dir /path/to/export --split level_1

# Resolution ablation
python eval/pbench.py --split level_0 --max-dimension 768

# See all options
python eval/pbench.py --help
```

### Evaluation protocol

1. **Force-resize** — each image is scaled so its longest edge equals
   `--max-dimension` (default 1024) using LANCZOS resampling before inference.
   This is different from the soft clamp used in the demo scripts, and is
   required to reproduce published PBench numbers.

2. **Inference** — the paged inference engine generates segmentation masks
   conditioned on the expression query.

3. **Mask alignment** — predicted masks are output at the upsampled inference
   resolution.  They are resized back to the **original image resolution**
   (nearest-neighbor) before scoring.  Ground-truth masks in PBench are at
   the original resolution.

4. **NMS** — greedy area-sorted NMS at IoU=0.5 is always applied to predicted
   masks before scoring.

### Metrics

| Metric | Description |
|---|---|
| **F1** | Mean of per-sample F1 scores (computed over positive GT samples only). Each sample's F1 is the average across all IoU thresholds. |
| **IL TP/TN/FP/FN** | Image-level classification counts. IL TP = GT has objects and model predicted at least one; IL FP = GT is empty but model predicted masks; etc. |

**IoU thresholds**: `[0.5, 0.55, 0.60, …, 0.95]` (10 thresholds) for
`level_0`–`level_4`; `[0.5]` only for the `dense` split.

Per-sample F1 uses **Hungarian matching** (optimal bipartite assignment) between
predicted and GT masks at each threshold, so every GT mask is matched to at
most one prediction.

### Output

Each split produces a JSON file (when `--out-dir` is set):

```
eval_results/pbench/
├── level_0_results.json
├── level_1_results.json
├── ...
└── summary.json          ← only when --split all
```

Example `level_0_results.json`:

```json
{
  "f1": 0.612,
  "il_tp": 95,
  "il_tn": 3,
  "il_fp": 1,
  "il_fn": 1,
  "n_samples": 100,
  "split": "level_0",
  "max_dimension": 1024,
  "wall_time_s": 84.2,
  "peak_gpu_gib": 18.4
}
```

### Using the metrics module independently

`eval/metrics.py` is a standalone pure-Python module with no PyTorch dependency.
Import it directly from notebooks or scripts:

```python
import sys
sys.path.insert(0, "eval/")
import metrics

# Evaluate a single sample
result = metrics.sample_f1(pred_rles, gt_rles, metrics.IOU_THRESHOLDS)
print(f"F1: {result['f1']:.3f}")

# Aggregate over a dataset
dataset_metrics = metrics.aggregate(per_sample_results, metrics.IOU_THRESHOLDS)
print(f"F1: {dataset_metrics['f1']:.3f}")
```

## YOLO demo-pics vs data/merged GT

Evaluate `yolo26m_merged_150ev2` (`runs/detect/yolo26m_merged_150ev2/weights/best.pt`)
against `demo-pics/` (sampled from `data/merged/`) with GT from
`data/merged/{train,val,test}/labels/` (resolved via `data/merged/merge_manifest.csv`).

```bash
# Default: IoU=0.5, conf=0.25 — prints per-class P/R/F1/Acc and challenging vs typical
python experiment/falcon/eval/eval_yolo_demo_pics.py --verbose

# Custom thresholds / outputs
python experiment/falcon/eval/eval_yolo_demo_pics.py --iou 0.5 --conf 0.25 \
  --out outputs/eval_demo_pics.json --csv-out outputs/eval_demo_pics.csv

# Help
python experiment/falcon/eval/eval_yolo_demo_pics.py --help
```

Metrics per class and micro-averaged: `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`,
`F1 = 2PR/(P+R)`, `accuracy (Jaccard) = TP/(TP+FP+FN)` at IoU 0.5.
Images in `demo-pics/` with no `merge_manifest.csv` entry (external web images)
are listed as unmatched and excluded from the counts.

## Falcon demo-pics vs data/merged GT (mask → bbox)

Run Falcon Perception on the same `demo-pics`/`data/merged` split but derive
bboxes from its masks so it can be scored with the same IoU counts as YOLO.
One text query per class (`person`, `helmet`, `gloves`, `boots`, `vest`,
`no-helmet`, `no-gloves`, `no-boots`, `no-vest` — order = class id, override
with `--queries`):

```bash
# Single image -> bboxes (uses existing visualization helper, no eval script):
python -c "from falcon_perception.visualization_utils import detections_from_sequence; \
            from falcon_perception import load_and_prepare_model, build_prompt_for_task; \
            from falcon_perception.paged_inference import PagedInferenceEngine, Sequence; \
            # load -> Sequence(image, text=build_prompt_for_task('person','segmentation')) -> engine.generate -> detections_from_sequence(seq)"

# Full demo-pics eval (mask->bbox at original resolution, same greedy IoU matching as YOLO):
python experiment/falcon/eval/eval_falcon_demo_pics.py --hf-local-dir /path/to/export --verbose
python experiment/falcon/eval/eval_falcon_demo_pics.py --hf-model-id tiiuae/Falcon-Perception --device cuda --dtype bfloat16 --out outputs/eval_falcon_demo_pics.json
python experiment/falcon/eval/eval_falcon_demo_pics.py --hf-local-dir /path/to/export --task detection --queries "person,helmet,gloves,boots,vest"
python experiment/falcon/eval/eval_falcon_demo_pics.py --help
```

Bbox derivation reuses `eval/metrics.py:resize_rle(nms(...))` (masks resized
to original `PIL.Image.size` before `pycocotools.mask.toBbox`/tight bounds)
and `visualization_utils.pair_bbox_entries(bboxes_raw)` as fallback when
`do_segmentation=False`. Same metrics/buckets as YOLO: per-class + micro
`precision/recall/F1/accuracy (Jaccard)` at `IoU=0.5`, with `challenging` vs
`typical` breakdown and 10 external unmatched images excluded.
