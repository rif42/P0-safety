"""P0 Safety — interactive demo: upload photo(s), classify with the baseline model."""

import pandas as pd
import streamlit as st
from PIL import Image

from model_ppe_classifier import load_model, predict

st.title("Demo")
st.caption("Baseline VGG16 transfer-learning classifier (hardhat vs. no-hardhat only).")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

if model is None:
    st.warning(
        "No trained model found at `models/baseline_classifier.keras`.\n\n"
        "Run `python train_baseline_classifier.py` first to train and "
        "save the baseline classifier, then reload this page."
    )
    st.stop()

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
    "site photos. It's overconfident on uncropped images."
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
