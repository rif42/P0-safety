"""P0 Safety — model performance: baseline classifier metrics/curves, and YOLO run metrics."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _APP_DIR.parent
STATS_PATH = _APP_DIR / "models" / "baseline_stats.json"
HISTORY_PATH = _APP_DIR / "models" / "baseline_history.json"
RUNS_DIR = _REPO_ROOT / "runs"

st.title("Model performance")

st.header("Baseline classifier")
if STATS_PATH.exists():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    col1, col2, col3 = st.columns(3)
    col1.metric("This model (val accuracy)", f"{stats['val_accuracy']:.0%}")
    col2.metric(f"Majority-class baseline ({stats['majority_class']})", f"{stats['majority_baseline_accuracy']:.0%}")
    col3.metric("Random-guess baseline", f"{stats['random_guess_accuracy']:.0%}")
else:
    st.info("No trained baseline model yet — run train_baseline_classifier.py first.")

if HISTORY_PATH.exists():
    with open(HISTORY_PATH) as f:
        history = json.load(f)
    st.subheader("Training curves — is this model overfitting?")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Loss per epoch")
        st.line_chart(pd.DataFrame({"train": history["loss"], "val": history["val_loss"]}))
    with col2:
        st.caption("Accuracy per epoch")
        st.line_chart(pd.DataFrame({"train": history["accuracy"], "val": history["val_accuracy"]}))
    st.caption(
        "Training loss/accuracy keep improving while validation plateaus early — a classic "
        "overfitting signature. The model is overconfident on out-of-distribution input."
    )

def _find_col(df, *keywords):
    """First column whose name contains all keywords, e.g. _find_col(df, 'train', 'box_loss')."""
    return next((c for c in df.columns if all(k in c for k in keywords)), None)


# The ONLY runs that show up on this page — nothing under runs/ is auto-discovered anymore.
# This is deliberate: the repo's runs/ folder can pick up stray training runs from teammates
# (e.g. someone's own experiment landing at runs/detect/runs/ppe_dev/yolo26n/) that aren't
# ready to show here. Add an entry only once a run is one you actually want on this page —
# dict order = display order, "path" is relative to the runs/ folder.
RUN_INFO = {
    "yolov8n_scratch": {
        "label": "YOLOv8n — trained from scratch",
        "path": "scratch/yolov8n_scratch",
    },
    "pretrained_100e": {
        "label": "YOLOv8n — pretrained (bundled with dataset)",
        "path": "pretrained_100e",
        "caption": (
            "Came bundled with the Kaggle dataset download, not trained by us. Started from "
            "COCO-pretrained yolov8n.pt weights and fine-tuned for 100 epochs on the same 10 "
            "classes as our data — a useful ceiling to compare our own runs against."
        ),
    },
    # "yolo26n": {
    #     "label": "YOLOv26n",
    #     "path": "detect/runs/ppe_dev/yolo26n",
    #     "caption": "Add once this run is finished and you've decided to show it.",
    # },
}

st.header("YOLO training runs")
runs_to_show = []
for key, info in RUN_INFO.items():
    results_path = RUNS_DIR / info["path"] / "results.csv"
    if results_path.exists():
        runs_to_show.append((info, results_path))

if not runs_to_show:
    st.info("No YOLO training runs configured yet — add an entry to RUN_INFO in this file.")
else:
    for info, results_path in runs_to_show:
        run_dir = results_path.parent
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        map50_95_col = next((c for c in df.columns if "mAP50-95" in c), None)
        map50_col = next((c for c in df.columns if "mAP50" in c and "mAP50-95" not in c), None)

        st.subheader(info.get("label", run_dir.name))
        if info.get("caption"):
            st.caption(info["caption"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Epochs completed", len(df))
        col2.metric("mAP50", f"{last[map50_col]:.3f}" if map50_col else "n/a")
        col3.metric("mAP50-95", f"{last[map50_95_col]:.3f}" if map50_95_col else "n/a")

        # Per-epoch curves, straight from results.csv (one row per epoch).
        box_train, box_val = _find_col(df, "train", "box_loss"), _find_col(df, "val", "box_loss")
        cls_train, cls_val = _find_col(df, "train", "cls_loss"), _find_col(df, "val", "cls_loss")
        dfl_train, dfl_val = _find_col(df, "train", "dfl_loss"), _find_col(df, "val", "dfl_loss")
        precision_col = _find_col(df, "precision")
        recall_col = _find_col(df, "recall")

        curve_col1, curve_col2 = st.columns(2)
        with curve_col1:
            if all([box_train, cls_train, dfl_train]):
                st.caption("Loss per epoch (box + cls + dfl, summed)")
                loss_data = {"train": df[box_train] + df[cls_train] + df[dfl_train]}
                if all([box_val, cls_val, dfl_val]):
                    loss_data["val"] = df[box_val] + df[cls_val] + df[dfl_val]
                st.line_chart(pd.DataFrame(loss_data))
            else:
                st.caption("Loss columns not found in results.csv")
        with curve_col2:
            if map50_col and map50_95_col:
                st.caption("mAP per epoch")
                st.line_chart(pd.DataFrame({"mAP50": df[map50_col], "mAP50-95": df[map50_95_col]}))
            else:
                st.caption("mAP columns not found in results.csv")

        if precision_col and recall_col:
            st.caption("Precision / recall per epoch")
            st.line_chart(pd.DataFrame({"precision": df[precision_col], "recall": df[recall_col]}))

        # Ultralytics also drops ready-made plots (PR curve, confusion matrix) in the run folder —
        # no need to rebuild these from scratch, just surface them.
        extra_plots = {
            "results.png": "All metrics, Ultralytics' own summary grid",
            "confusion_matrix.png": "Confusion matrix",
            "PR_curve.png": "Precision-recall curve",
        }
        available_plots = [(name, caption) for name, caption in extra_plots.items() if (run_dir / name).exists()]
        if available_plots:
            with st.expander("More plots from this run"):
                for name, caption in available_plots:
                    st.image(str(run_dir / name), caption=caption, width="stretch")

        st.divider()
