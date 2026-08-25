"""Hardhat / no-hardhat classifier — model, loading, and inference logic from YOLO8.ipynb (Section 4)."""

from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

IMG_SIZE = (224, 224)
CLASS_NAMES = ["hardhat", "no_hardhat"]  # matches train_ds.class_names order in the notebook
MODEL_PATH = Path(__file__).parent / "models" / "baseline_classifier.keras"


def build_model():
    """Same architecture as YOLO8.ipynb, cell 16."""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False  # freeze the convolutional base

    model = Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # hardhat vs no-hardhat
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_model(model_path: Path = MODEL_PATH):
    """Load a trained model saved by train_baseline.py. Returns None if none exists yet."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and apply VGG16's expected preprocessing (mean-centered, BGR channel order) —
    must match whatever train_baseline_classifier.py applies to its training images."""
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict(model, image: Image.Image):
    """Return (label, confidence) for a single PIL image."""
    batch = preprocess_image(image)
    score = float(model.predict(batch, verbose=0)[0][0])
    # sigmoid output: near 0 -> hardhat, near 1 -> no_hardhat (per CLASS_NAMES order)
    label = CLASS_NAMES[1] if score >= 0.5 else CLASS_NAMES[0]
    confidence = score if score >= 0.5 else 1 - score
    return label, confidence
