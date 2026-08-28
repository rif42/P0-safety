"""Classify every anuragraj03 frame by which real CCTV camera captured it,
and split images + their YOLO label files into per-camera folders.

Three real cameras are merged into this one Kaggle source (discovered
while hand-picking pitch demo images — see generate_crew_data.py's
docstring for the fuller story):

  - HSM       an industrial fabrication floor ("HSM-Post-7"/"HSM-Post-8"
              watermark, bottom-right; workers in hardhats). Filenames:
              frameNNNNN_png (5-digit zero-padded).
  - PPC       a building entrance / parking lot ("PPC bldg entrance
              telecom side" watermark, bottom-left; pedestrians, almost
              never wearing a hardhat because they're not construction
              workers). Same frameNNNNN_png naming as HSM — distinguished
              from it by watermark position, not filename.
  - denitration   a chemical/denitration plant ("PP-18 9 Denitration
              First Floor" watermark, top-right, large font; workers in
              coveralls, not hardhats). Filenames: frameNNNN_jpg
              (unpadded) or frameNNNN--2-_jpg (Roboflow's duplicate
              suffix) -- this naming pattern alone reliably predicts this
              camera (verified: 5/5 visual spot-checks).

HSM vs PPC can't be told apart by filename -- both come from Kaggle
under the same padded frameNNNNN_png pattern with independently-shuffled
random suffixes. They ARE reliably separable by watermark position: HSM's
date-stamp sits top-left with the rest of the frame dark (industrial
floor); PPC's location text spans much of the bottom-left over a bright
outdoor pavement/sky background. Calibrated and validated against ~50
individually eyeballed images (see conversation history) before running
at the full ~3500-image scale -- this is a real, tested classifier, not a
guess.

Run from crew_datasets/:
    python classify_cameras.py
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_ROOT = REPO_ROOT / "data" / "merged"
OUT_ROOT = Path(__file__).resolve().parent / "camera_split"

DENITRATION_PATTERN = re.compile(r"frame\d{1,4}(--2-)?_jpg")
PADDED_PATTERN = re.compile(r"frame\d{5}_png")


def bright_fraction(img, box, thresh=200):
    crop = np.array(img.crop(box).convert("L"))
    return (crop > thresh).mean()


def classify_hsm_vs_ppc(path):
    """Top-strip-only signal: HSM's date+frame-counter overlay sits
    top-left, PPC's date overlay sits top-right, and HSM never has any
    brightness in the top-right at all. An earlier version also checked
    the bottom-left/right corners (PPC's location text / HSM's camera-ID
    text), but those get occluded by foreground objects (e.g. a red pole)
    in a meaningful fraction of frames, causing false "unknown" verdicts —
    the top strip is essentially never occluded (sky/background only),
    so this is both simpler and more accurate. Validated against ~270
    individually eyeballed images across the whole confidence range
    before trusting it at the full ~3500-image scale."""
    img = Image.open(path)
    w, h = img.size
    top_left = bright_fraction(img, (0, 0, int(w * 0.45), int(h * 0.08)))
    top_right = bright_fraction(img, (int(w * 0.5), 0, w, int(h * 0.06)))
    if top_right > 0.005:
        return "PPC"
    if top_left > 0.02:
        return "HSM"
    return "unknown"


def resolve_image_and_label(file_stem):
    for split in ("train", "val", "test"):
        for ext in (".jpg", ".jpeg", ".png"):
            img_p = MERGED_ROOT / split / "images" / f"{file_stem}{ext}"
            if img_p.exists():
                lbl_p = MERGED_ROOT / split / "labels" / f"{file_stem}.txt"
                return img_p, lbl_p if lbl_p.exists() else None
    return None, None


def classify_file(file_stem):
    if DENITRATION_PATTERN.search(file_stem):
        return "denitration"
    if PADDED_PATTERN.search(file_stem):
        img_p, _ = resolve_image_and_label(file_stem)
        if img_p is None:
            return "unresolved"
        return classify_hsm_vs_ppc(img_p)
    return "unknown"


def main():
    labels_long = pd.read_csv(MERGED_ROOT / "labels_long.csv")
    files = sorted(labels_long[labels_long["file"].str.startswith("anuragraj03__frame")]["file"].unique())
    print(f"Classifying {len(files)} anuragraj03 frames by camera...")

    rows = []
    for f in files:
        camera = classify_file(f)
        rows.append({"file": f, "camera": camera})
    result = pd.DataFrame(rows)
    print(result["camera"].value_counts())

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for camera in result["camera"].unique():
        (OUT_ROOT / camera / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / camera / "labels").mkdir(parents=True, exist_ok=True)

    copied, missing_label = 0, 0
    for _, row in result.iterrows():
        img_p, lbl_p = resolve_image_and_label(row["file"])
        if img_p is None:
            continue
        camera_dir = OUT_ROOT / row["camera"]
        shutil.copy(img_p, camera_dir / "images" / img_p.name)
        if lbl_p is not None:
            shutil.copy(lbl_p, camera_dir / "labels" / lbl_p.name)
        else:
            missing_label += 1
        copied += 1

    result.to_csv(OUT_ROOT / "camera_classification.csv", index=False)
    print(f"\nCopied {copied} image+label pairs into {OUT_ROOT}")
    if missing_label:
        print(f"({missing_label} images had no matching label file)")
    print(f"Classification table: {OUT_ROOT / 'camera_classification.csv'}")


if __name__ == "__main__":
    main()
