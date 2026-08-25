#!/usr/bin/env python3
"""
Build dataset/ from data/raw/ in one pass: remap each label's class IDs to
the merged schema (dataset/class_mapping.py), keep only annotations whose
merged class is in --classes (default: the core classes 0-6), and copy over
only the images that still have at least one annotation after that
filtering. Images with none of the included classes present are excluded
entirely, not just given an empty label file.

dataset/images, dataset/labels, and dataset/merge_manifest.csv are fully
derived from data/raw/, so this script rebuilds them from scratch every run
rather than incrementally patching a previous run.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
OUT_ROOT = REPO_ROOT / "dataset"

sys.path.insert(0, str(OUT_ROOT))
from class_mapping import DEFAULT_INCLUDED_CLASSES, MERGED_CLASSES, SOURCE_CLASS_MAPS  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# source -> split -> (images_dir, labels_dir), relative to the source's raw folder
SOURCE_DIRS = {
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        default=",".join(str(c) for c in DEFAULT_INCLUDED_CLASSES),
        help="Comma-separated merged class IDs to include (default: core classes 0-6)",
    )
    return parser.parse_args()


def clean_output():
    for split in ("train", "val", "test"):
        img_dir = OUT_ROOT / "images" / split
        lbl_dir = OUT_ROOT / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)


def build(included_classes):
    clean_output()
    manifest_rows = []
    kept_counts = [0] * len(MERGED_CLASSES)
    images_seen = 0
    images_kept = 0

    for source, splits in SOURCE_DIRS.items():
        source_dir = RAW_ROOT / source
        if not source_dir.exists():
            print(f"skip {source}: not found at {source_dir}")
            continue
        class_map = SOURCE_CLASS_MAPS[source]

        for split, (img_rel, lbl_rel) in splits.items():
            img_dir = source_dir / img_rel
            lbl_dir = source_dir / lbl_rel
            if not img_dir.exists():
                print(f"skip {source}/{split}: {img_dir} missing")
                continue

            for image_path in sorted(img_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                images_seen += 1

                label_path = lbl_dir / f"{image_path.stem}.txt"
                kept_lines = []
                if label_path.exists():
                    for line in label_path.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        merged_id = class_map[int(parts[0])]
                        if merged_id in included_classes:
                            kept_lines.append(" ".join([str(merged_id)] + parts[1:]))

                if not kept_lines:
                    continue  # none of the included classes present -> drop the image

                images_kept += 1
                for line in kept_lines:
                    kept_counts[int(line.split()[0])] += 1

                merged_stem = f"{source}__{image_path.stem}"
                out_image = OUT_ROOT / "images" / split / f"{merged_stem}{image_path.suffix}"
                shutil.copy2(image_path, out_image)
                out_label = OUT_ROOT / "labels" / split / f"{merged_stem}.txt"
                out_label.write_text("\n".join(kept_lines) + "\n")

                manifest_rows.append(
                    {
                        "merged_filename": f"{merged_stem}{image_path.suffix}",
                        "source": source,
                        "original_filename": image_path.name,
                        "split": split,
                        "num_annotations": len(kept_lines),
                    }
                )

    manifest_path = OUT_ROOT / "merge_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["merged_filename", "source", "original_filename", "split", "num_annotations"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    by_source_split = {}
    for row in manifest_rows:
        key = (row["source"], row["split"])
        by_source_split[key] = by_source_split.get(key, 0) + 1

    included_names = [MERGED_CLASSES[c] for c in sorted(included_classes)]
    print(f"Included classes: {sorted(included_classes)} {included_names}")
    print(
        f"Images scanned: {images_seen}, kept: {images_kept}, "
        f"dropped (no included classes present): {images_seen - images_kept}"
    )
    for (source, split), count in sorted(by_source_split.items()):
        print(f"  {source:24s} {split:6s} {count}")
    print("Kept instance counts:")
    for class_id in sorted(included_classes):
        print(f"  {class_id:2d} {MERGED_CLASSES[class_id]:20s} {kept_counts[class_id]}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    args = parse_args()
    included = {int(c) for c in args.classes.split(",")}
    build(included)
