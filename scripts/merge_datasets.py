#!/usr/bin/env python3
"""
Merge the three raw PPE datasets under data/raw/ into one consistently
structured dataset/ tree, without altering any annotation content.

Each source currently uses its own split-naming convention and folder layout
on disk (train/valid/test vs train/val/test, images/labels nested differently).
This script normalizes all three onto the same layout:

    dataset/images/{train,val,test}/<source>__<original_name>.<ext>
    dataset/labels/{train,val,test}/<source>__<original_name>.txt

Files are copied, never moved or edited, so data/raw/ stays the untouched
source of truth. The source-name prefix guarantees no filename collisions
across datasets (verified none exist, but this makes it future-proof).

Class IDs are deliberately NOT remapped here: each source's label .txt files
use its own class numbering, and two of the three sources have no local
class-name file to remap from safely. See dataset/SOURCES.md for what's known
per source. Unifying classes into the project's positive-only schema
(person/hardhat/vest/boots/mask) is a separate follow-up step.
"""
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
OUT_ROOT = REPO_ROOT / "dataset"

# source name -> {split: (images_dir_relative_to_source, labels_dir_relative_to_source)}
SOURCES = {
    "anuragraj03": {
        "train": ("train/images", "train/labels"),
        "val": ("val/images", "val/labels"),
        "test": ("test/images", "test/labels"),
    },
    "ketakichalke-boots": {
        "train": ("images/train", "labels/train"),
        "val": ("images/val", "labels/val"),
        "test": ("images/test", "labels/test"),
    },
    "snehilsanyal-main": {
        "train": ("css-data/train/images", "css-data/train/labels"),
        "val": ("css-data/val/images", "css-data/val/labels"),
        "test": ("css-data/test/images", "css-data/test/labels"),
    },
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def merge():
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for source, splits in SOURCES.items():
        source_dir = RAW_ROOT / source
        if not source_dir.exists():
            print(f"skip {source}: not found at {source_dir}")
            continue

        for split, (img_rel, lbl_rel) in splits.items():
            img_dir = source_dir / img_rel
            lbl_dir = source_dir / lbl_rel
            if not img_dir.exists():
                print(f"skip {source}/{split}: {img_dir} missing")
                continue

            for image_path in sorted(img_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                merged_stem = f"{source}__{image_path.stem}"
                out_image = OUT_ROOT / "images" / split / f"{merged_stem}{image_path.suffix}"
                shutil.copy2(image_path, out_image)

                label_path = lbl_dir / f"{image_path.stem}.txt"
                has_label = label_path.exists()
                if has_label:
                    out_label = OUT_ROOT / "labels" / split / f"{merged_stem}.txt"
                    shutil.copy2(label_path, out_label)

                manifest_rows.append(
                    {
                        "merged_filename": f"{merged_stem}{image_path.suffix}",
                        "source": source,
                        "original_filename": image_path.name,
                        "split": split,
                        "has_label": has_label,
                    }
                )

    manifest_path = OUT_ROOT / "merge_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["merged_filename", "source", "original_filename", "split", "has_label"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    by_source_split = {}
    for row in manifest_rows:
        key = (row["source"], row["split"])
        by_source_split[key] = by_source_split.get(key, 0) + 1

    print(f"Merged {len(manifest_rows)} images into {OUT_ROOT}")
    for (source, split), count in sorted(by_source_split.items()):
        print(f"  {source:24s} {split:6s} {count}")
    missing_labels = sum(1 for row in manifest_rows if not row["has_label"])
    if missing_labels:
        print(f"  ({missing_labels} images have no matching label file — see merge_manifest.csv)")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    merge()
