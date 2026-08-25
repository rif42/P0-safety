"""Train the baseline hardhat/no-hardhat classifier from YOLO8.ipynb (Section 4) and save it to models/baseline_classifier.keras."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

from model_ppe_classifier import build_model, IMG_SIZE, MODEL_PATH, CLASS_NAMES

# --- Settings — edit these directly instead of passing CLI flags ---
# Anchored to the repo root (one level up from this file) so it works whether you run
# this from the repo root or from inside streamlit_app_joanna/.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "css-data"
EPOCHS = 30
BATCH_SIZE = 32
OUTPUT_PATH = MODEL_PATH  # where the trained model gets saved
# ---------------------------------------------------------------------

RANDOM_STATE = 42

# Full 10-class label set used by the YOLO-format .txt annotations in data/css-data.
YOLO_CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]


def parse_yolo_labels(label_path, class_names):
    """Read a YOLO-format .txt label file into (class_name, x, y, w, h) rows, 0-1 normalised."""
    rows = []
    if not Path(label_path).exists():
        return rows
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
            rows.append((class_names[cls_id], x, y, w, h))
    return rows


def crop_head_regions(images_dir, labels_dir, class_names, output_dir, padding=0.1):
    """Crop the Hardhat / NO-Hardhat boxes out of each image and sort the crops into
    output_dir/hardhat/ and output_dir/no_hardhat/ — the folder split
    image_dataset_from_directory uses to infer labels automatically."""
    output_dir = Path(output_dir)
    (output_dir / "hardhat").mkdir(parents=True, exist_ok=True)
    (output_dir / "no_hardhat").mkdir(parents=True, exist_ok=True)
    n_saved = {"hardhat": 0, "no_hardhat": 0}

    for label_path in Path(labels_dir).glob("*.txt"):
        image_path = Path(images_dir) / (label_path.stem + ".jpg")
        if not image_path.exists():
            continue
        try:
            img = Image.open(image_path)
        except OSError:
            continue
        w_img, h_img = img.size  # PIL gives (width, height) — opposite order from a cv2/numpy array

        for i, (cls_name, x, y, w, h) in enumerate(parse_yolo_labels(label_path, class_names)):
            if cls_name not in ("Hardhat", "NO-Hardhat"):
                continue

            bw, bh = w * (1 + padding), h * (1 + padding)
            x1 = max(int((x - bw / 2) * w_img), 0)
            y1 = max(int((y - bh / 2) * h_img), 0)
            x2 = min(int((x + bw / 2) * w_img), w_img)
            y2 = min(int((y + bh / 2) * h_img), h_img)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))
            folder = "hardhat" if cls_name == "Hardhat" else "no_hardhat"
            crop.save(output_dir / folder / f"{label_path.stem}_{i}.jpg")
            n_saved[folder] += 1

    return n_saved


def main():
    crop_dir = DATA_DIR / "baseline_crops"

    print("Cropping head regions from the training split...")
    counts = crop_head_regions(
        images_dir=DATA_DIR / "train" / "images",
        labels_dir=DATA_DIR / "train" / "labels",
        class_names=YOLO_CLASS_NAMES,
        output_dir=crop_dir,
    )
    print(f"Crops saved: {counts}")
    if sum(counts.values()) == 0:
        raise SystemExit(
            f"No crops were produced from {DATA_DIR}. Check the DATA_DIR constant at "
            "the top of this file points at the dataset root (the one containing "
            "train/images and train/labels)."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        crop_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", validation_split=0.2, subset="training", seed=RANDOM_STATE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        crop_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", validation_split=0.2, subset="validation", seed=RANDOM_STATE,
    )
    print(f"Class order: {train_ds.class_names} (expected: {CLASS_NAMES})")

    # Must match model_ppe_classifier.preprocess_image exactly, or training and inference
    # see different input distributions.
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    model = build_model()
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])

    loss, accuracy = model.evaluate(val_ds)
    print(f"val loss: {loss:.3f}, val accuracy: {accuracy:.3f}")

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved model to {output_path}")

    # Naive baselines, computed on this exact validation split so they're directly
    # comparable to the val accuracy above — not the wider training pool.
    val_labels = np.concatenate([y.numpy() for _, y in val_ds]).astype(int).ravel()
    n_hardhat = int((val_labels == 0).sum())
    n_no_hardhat = int((val_labels == 1).sum())
    total = n_hardhat + n_no_hardhat
    majority_class = CLASS_NAMES[0] if n_hardhat >= n_no_hardhat else CLASS_NAMES[1]
    majority_baseline_accuracy = max(n_hardhat, n_no_hardhat) / total
    random_guess_accuracy = 1 / len(CLASS_NAMES)

    stats = {
        "val_class_counts": {CLASS_NAMES[0]: n_hardhat, CLASS_NAMES[1]: n_no_hardhat},
        "majority_class": majority_class,
        "majority_baseline_accuracy": round(majority_baseline_accuracy, 4),
        "random_guess_accuracy": round(random_guess_accuracy, 4),
        "val_loss": round(float(loss), 4),
        "val_accuracy": round(float(accuracy), 4),
    }
    stats_path = output_path.parent / "baseline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved baseline stats to {stats_path}: {stats}")

    # Per-epoch train/val loss + accuracy — lets the app plot the training curves
    # (and show the overfitting gap) rather than just the final numbers above.
    history_path = output_path.parent / "baseline_history.json"
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
