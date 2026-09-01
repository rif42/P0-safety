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
        "compare_label": "v8",
        "live_on_compare": True,
    },
    "yolo26s_css_100e": {
        "label": "YOLO26s — css-data, 100 epochs",
        "path": "yolo26s_css_100e",
        "caption": (
            "Trained by a teammate outside this repo, on the same css-data dataset and class "
            "list as pretrained_100e — a genuine apples-to-apples comparison. Different "
            "architecture (YOLO26, not YOLOv8n). Beats pretrained_100e on recall, mAP50 and "
            "mAP50-95 at the final epoch. Shown on Model Comparison as \"v26\"."
        ),
        "compare_label": "v26",
        "live_on_compare": True,
    },
    "yolo26s_merged_100e": {
        "label": "YOLO26s — merged dataset, 100 epochs",
        "path": "detect/yolo26s_merged_100e",
        "caption": (
            "Trained by a teammate outside this repo on a merged dataset (9 classes: person, "
            "helmet/no-helmet, vest/no-vest, gloves/no-gloves, boots/no-boots). Working Person "
            "class (0.83 recall) plus two PPE items no other run here tracks. Shown on Model "
            "Comparison and wired into the app's compliance logic."
        ),
        "compare_label": "merged",
        "live_on_compare": True,
    },
    "yolo26m_merged_150e": {
        "label": "YOLO26m — merged dataset, 150 epochs",
        "path": "detect/yolo26m_merged_150e",
        "caption": (
            "Same merged dataset/vocabulary as yolo26s_merged_100e above, but the larger "
            "YOLO26m backbone trained for the full 150 epochs (patience=20, ran to "
            "completion). Beats yolo26s_merged_100e on every aggregate metric and every "
            "per-class confusion-matrix diagonal. Now the app's default model on the Demo "
            "page — see Model Comparison for how it stacks up against yolo26m_mergedpeople_150e."
        ),
        "compare_label": "merged-m",
        "live_on_compare": True,
    },
    "yolo26m_mergedpeople_150e": {
        "label": "YOLO26m — merged + pseudo-labeled Person, 150 epochs",
        "path": "detect/yolo26m_mergedpeople_150e",
        "caption": (
            "Same run setup as yolo26m_merged_150e, trained instead on \"mergedpeople\": "
            "data/merged with ppe_detection_m's Person boxes filled in via pseudo-labeling "
            "(see person_pseudolabels_test.ipynb) rather than left absent. Early-stopped at "
            "134/150 epochs. Slightly better Person recall (0.88 vs 0.86) and aggregate "
            "mAP50-95/recall than yolo26m_merged_150e, but a real trade-off: its confusion "
            "matrix shows it also misclassifies far more true background as \"person\" (0.31 "
            "vs 0.16), driving its lower aggregate precision (0.912 vs 0.922) — why "
            "yolo26m_merged_150e, not this run, was chosen as the app's default."
        ),
        "compare_label": "mergedpeople",
        "live_on_compare": True,
    },
    "yolo26s_Altec_PPE_100e": {
        "label": "YOLO26s — Altec PPE dataset, 100 epochs",
        "path": "detect/yolo26s_Altec_PPE_100e",
        "caption": (
            "Trained by a teammate outside this repo on a different PPE dataset (10 classes: "
            "Face_masks, Face_shield, Glasses, Gloves, Helmet, Safety_shoes, Safety_vests, plus "
            "lowercase glasses/helmet duplicates) — no Person class at all, confirmed via its "
            "confusion matrix below. Can't drive a per-person compliance verdict, so it doesn't "
            "run live on Model Comparison — training metrics only, here."
        ),
        "compare_label": "Altec",
        "live_on_compare": False,  # no Person class — Model Comparison shows it as an N/A card
    },
}

# Latest runs/llm/ comparison (presence-detection F1, not training epochs — a
# different task shape than the YOLO runs below, so no epoch curves/confusion
# matrix). Transcribed from reports/llm_vs_yolo_comparison.md same as
# PER_CLASS below is transcribed from confusion matrices — update by hand if
# scripts/compare_models.py is re-run.
st.header("LLM / VLM comparison (latest run)")
st.caption("runs/llm/20260831_merged_n100_seed42_yolo-gemini — 100 test images, presence/absence per class. Full breakdown: reports/llm_vs_yolo_comparison.md")
LLM_COMPARISON = [
    {"Model": "YOLO26 (ours)", "Overall F1": 0.96, "Positive-class F1": 0.96, "Negative-class F1": 0.95},
    {"Model": "Gemini 3.6 Flash (API)", "Overall F1": 0.64, "Positive-class F1": 0.80, "Negative-class F1": 0.44},
]
st.dataframe(pd.DataFrame(LLM_COMPARISON), hide_index=True, width="stretch")
st.divider()

st.header("YOLO training runs")
runs_to_show = []
for key, info in RUN_INFO.items():
    results_path = RUNS_DIR / info["path"] / "results.csv"
    if results_path.exists():
        runs_to_show.append((key, info, results_path))

# Summary table — the 6 models currently on the Model Comparison page, side by side.
# yolov8n_scratch is deliberately excluded: it's shown lower on this page, but was never
# added to Model Comparison, so it's not part of "the 4 models" this table is answering for.
compare_rows = [(key, info, path) for key, info, path in runs_to_show if info.get("compare_label")]
if compare_rows:
    st.subheader("The 6 models on Model Comparison, side by side")
    st.caption(
        """
        mAP50 = average precision at a loose 0.5 IoU overlap threshold (is the box roughly in the right place)\n
        mAP50-95 = average across stricter thresholds from 0.5 to 0.95 (more demanding number)
        """
    )
    table_rows = []
    for key, info, results_path in compare_rows:
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        map50_95_col = next((c for c in df.columns if "mAP50-95" in c), None)
        map50_col = next((c for c in df.columns if "mAP50" in c and "mAP50-95" not in c), None)
        precision_col = _find_col(df, "precision")
        recall_col = _find_col(df, "recall")
        table_rows.append({
            "Model": info["compare_label"],
            "Run": info.get("label", key),
            "Epochs": len(df),
            "Precision": round(float(last[precision_col]), 3) if precision_col else None,
            "Recall": round(float(last[recall_col]), 3) if recall_col else None,
            "mAP50": round(float(last[map50_col]), 3) if map50_col else None,
            "mAP50-95": round(float(last[map50_95_col]), 3) if map50_95_col else None,
            "Live on Model Comparison": "Yes" if info.get("live_on_compare") else "No",
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    # Per-class numbers: results.csv is aggregate-only — Ultralytics never writes a per-class
    # CSV, the confusion matrix image is the only place these numbers exist. So this table is
    # manually transcribed from each run's confusion_matrix(_normalized).png diagonal (v8/v26/
    # merged/Altec read 2026-08-27, merged-m/mergedpeople added 2026-08-28) — NOT recomputed
    # live like the summary table above. If any of these 6 runs get retrained, re-read the new
    # confusion matrix and update this table by hand.
    #
    # Rows follow the app's own tracked slots (detector.SLOT_ITEMS) — person, then each PPE
    # item's present/absent pair. "—" means that model's training data has no matching class at
    # all (e.g. Altec has no Person, no negative classes for anything). Different models use
    # different words for the same slot (hardhat vs helmet) and different underlying datasets,
    # so a cell is that MODEL's own class name plus its own diagonal value — read each column on
    # its own terms rather than comparing raw numbers across columns as if it were one dataset.
    st.subheader("Per-class numbers")
    PER_CLASS = [
        {"Slot": "Person",                    "v8": "0.80", "v26": "0.83", "merged": "0.83", "merged-m": "0.86", "mergedpeople": "0.88", "Altec": "—"},
        {"Slot": "Head protection — present",  "v8": "0.76", "v26": "0.85", "merged": "0.94", "merged-m": "0.95", "mergedpeople": "0.95", "Altec": "0.89 / 0.78"},
        {"Slot": "Head protection — absent",   "v8": "0.62", "v26": "0.67", "merged": "0.88", "merged-m": "0.90", "mergedpeople": "0.88", "Altec": "—"},
        {"Slot": "Vest — present",             "v8": "0.78", "v26": "0.90", "merged": "0.76", "merged-m": "0.81", "mergedpeople": "0.81", "Altec": "0.70"},
        {"Slot": "Vest — absent",              "v8": "0.70", "v26": "0.74", "merged": "0.78", "merged-m": "0.81", "mergedpeople": "0.82", "Altec": "—"},
        {"Slot": "Mask — present",             "v8": "0.90", "v26": "0.95", "merged": "—",    "merged-m": "—",    "mergedpeople": "—",    "Altec": "0.83 / 0.70"},
        {"Slot": "Mask — absent",              "v8": "0.66", "v26": "0.70", "merged": "—",    "merged-m": "—",    "mergedpeople": "—",    "Altec": "—"},
        {"Slot": "Gloves — present",           "v8": "—",    "v26": "—",    "merged": "0.86", "merged-m": "0.88", "mergedpeople": "0.88", "Altec": "0.56"},
        {"Slot": "Gloves — absent",            "v8": "—",    "v26": "—",    "merged": "0.83", "merged-m": "0.83", "mergedpeople": "0.82", "Altec": "—"},
        {"Slot": "Boots — present",            "v8": "—",    "v26": "—",    "merged": "0.90", "merged-m": "0.91", "mergedpeople": "0.91", "Altec": "0.73"},
        {"Slot": "Boots — absent",             "v8": "—",    "v26": "—",    "merged": "0.86", "merged-m": "0.89", "mergedpeople": "0.89", "Altec": "—"},
    ]
    st.dataframe(pd.DataFrame(PER_CLASS), hide_index=True, width="stretch")
    st.caption(
        "Numbers are each class's diagonal in its confusion matrix — of every box that was "
        "truly that class, the fraction the model predicted correctly (≈ recall). Blank cells "
        "are classes that model's dataset never had, not a zero score. A cell with two numbers "
        "(Altec's Head protection / Mask rows) is two separate classes that model tracks for "
        "the same slot — e.g. Helmet and a lowercase duplicate \"helmet\" class, or Face_masks "
        "and Face_shield — see the confusion matrices below for the exact class names and full "
        "picture, including classes not in this table (Safety Cone/machinery/vehicle for v8/v26, "
        "Glasses for Altec)."
    )

    # Full confusion matrices below the table — the underlying image each number above came
    # from, plus every class (including the ones not in the table) with its confusions visible.
    st.subheader("Confusion matrices")
    cm_cols = st.columns(2)
    for i, (key, info, results_path) in enumerate(compare_rows):
        run_dir = results_path.parent
        cm_path = run_dir / "confusion_matrix_normalized.png"
        if not cm_path.exists():
            cm_path = run_dir / "confusion_matrix.png"
        with cm_cols[i % 2]:
            st.markdown(f"**{info['compare_label']}** — {info.get('label', key)}")
            if cm_path.exists():
                st.image(str(cm_path), width="stretch")
            else:
                st.info("No confusion matrix found for this run.")

    st.divider()

if not runs_to_show:
    st.info("No YOLO training runs configured yet — add an entry to RUN_INFO in this file.")
else:
    for key, info, results_path in runs_to_show:
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

        # results.csv is aggregate-only (one row per epoch, no per-class columns) — Ultralytics
        # never writes a per-class numeric table. The confusion matrix and the Box P/R/F1/PR
        # curves it drops in the run folder ARE the real per-class breakdown, so surface the
        # confusion matrix directly (most readable at a glance) and keep the rest one click away.
        cm_path = run_dir / "confusion_matrix_normalized.png"
        if not cm_path.exists():
            cm_path = run_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.caption("Per-class detection breakdown (confusion matrix — diagonal ≈ per-class recall)")
            st.image(str(cm_path), width="stretch")
        else:
            st.caption("No confusion matrix found for this run.")

        # Ultralytics also drops other ready-made plots in the run folder — no need to rebuild
        # these from scratch, just surface them.
        extra_plots = {
            "results.png": "All metrics, Ultralytics' own summary grid",
            "confusion_matrix.png": "Confusion matrix (raw counts)",
            "confusion_matrix_normalized.png": "Confusion matrix (normalized)",
            "BoxPR_curve.png": "Precision-recall curve, per class",
            "BoxP_curve.png": "Precision curve, per class",
            "BoxR_curve.png": "Recall curve, per class",
            "BoxF1_curve.png": "F1 curve, per class",
        }
        available_plots = [(name, caption) for name, caption in extra_plots.items() if (run_dir / name).exists()]
        if available_plots:
            with st.expander("More plots from this run (per-class curves, raw confusion matrix, summary grid)"):
                for name, caption in available_plots:
                    st.image(str(run_dir / name), caption=caption, width="stretch")

        st.divider()
