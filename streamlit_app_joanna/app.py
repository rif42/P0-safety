"""HI-VIS — Streamlit front end for the hardhat/no-hardhat baseline classifier."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from model_ppe_classifier import load_model, predict

STATS_PATH = Path(__file__).parent / "models" / "baseline_stats.json"
HISTORY_PATH = Path(__file__).parent / "models" / "baseline_history.json"

st.set_page_config(page_title="P0 Safety", page_icon="🦺")

st.title("🦺 P0 Safety")
st.caption(
    "Baseline VGG16 transfer-learning classifier (hardhat vs. no-hardhat only)."
)


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

if model is None:
    st.warning(
        "No trained model found at `models/baseline_classifier.keras`.\n\n"
        "Run `python train_baseline.py --data-dir data/css-data` first to train and "
        "save the baseline classifier, then reload this page."
    )
    st.stop()

if STATS_PATH.exists():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    st.subheader("How good is this, really?")
    col1, col2, col3 = st.columns(3)
    col1.metric("This model (val accuracy)", f"{stats['val_accuracy']:.0%}")
    col2.metric(f"Majority-class baseline ({stats['majority_class']})", f"{stats['majority_baseline_accuracy']:.0%}")
    col3.metric("Random-guess baseline", f"{stats['random_guess_accuracy']:.0%}")


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
        "overfitting signature. The metrics above (val accuracy, etc.) are the honest ones; "
        "the training-set numbers are not representative of real performance."
    )

uploaded_files = st.file_uploader(
    "Upload a photo, or select every photo in a folder at once",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Waiting for an image (or a batch of images) to classify.")
    st.stop()

st.warning(
    "This classifier was trained on tight crops around a head, not full "
    "site photos. It's overconfident on uncropped images.",
    icon="⚠️",
)

results = []
for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    label, confidence = predict(model, image)
    results.append({
        "file": uploaded_file.name,
        "prediction": label,
        "confidence": round(confidence, 3),
    })

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, width="stretch")
    with col2:
        st.metric(uploaded_file.name, label, f"{confidence:.2%} confidence")

if len(results) > 1:
    st.subheader("Batch results")
    df = pd.DataFrame(results)
    st.dataframe(df, width="stretch")
    st.download_button(
        "Download results as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
