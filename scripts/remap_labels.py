#!/usr/bin/env python3
"""
Rewrite dataset/labels/*/*.txt so every source uses the same class IDs,
per dataset/class_mapping.py.

Reads original label content straight from data/raw/ (via
dataset/merge_manifest.csv, which records which raw source/file each merged
file came from) rather than from the already-copied dataset/labels/ files, so
this script is idempotent: re-running it always re-derives merged IDs from
the untouched raw IDs, instead of remapping an already-remapped file.

Bounding box coordinates are copied through unchanged; only the leading
class-id column on each line is rewritten.
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
DATASET_ROOT = REPO_ROOT / "dataset"
MANIFEST_PATH = DATASET_ROOT / "merge_manifest.csv"

sys.path.insert(0, str(DATASET_ROOT))
from class_mapping import MERGED_CLASSES, SOURCE_CLASS_MAPS  # noqa: E402

# source -> split -> labels dir relative to the source's raw folder
LABEL_DIRS = {
    "anuragraj03": {"train": "train/labels", "val": "val/labels", "test": "test/labels"},
    "ketakichalke-boots": {"train": "labels/train", "val": "labels/val", "test": "labels/test"},
    "snehilsanyal-main": {
        "train": "css-data/train/labels",
        "val": "css-data/val/labels",
        "test": "css-data/test/labels",
    },
}


def remap():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found — run scripts/merge_datasets.py first")

    merged_counts = [0] * len(MERGED_CLASSES)
    files_written = 0
    files_skipped = 0

    with MANIFEST_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["has_label"] != "True":
                continue

            source = row["source"]
            split = row["split"]
            class_map = SOURCE_CLASS_MAPS[source]
            label_dir = RAW_ROOT / source / LABEL_DIRS[source][split]
            original_stem = Path(row["original_filename"]).stem
            raw_label_path = label_dir / f"{original_stem}.txt"

            if not raw_label_path.exists():
                print(f"warning: expected {raw_label_path} but it's missing")
                files_skipped += 1
                continue

            out_lines = []
            for line in raw_label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                orig_id = int(parts[0])
                merged_id = class_map[orig_id]
                merged_counts[merged_id] += 1
                out_lines.append(" ".join([str(merged_id)] + parts[1:]))

            merged_stem = f"{source}__{original_stem}"
            out_path = DATASET_ROOT / "labels" / split / f"{merged_stem}.txt"
            out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            files_written += 1

    print(f"Remapped {files_written} label files ({files_skipped} skipped).")
    print("Merged class instance counts:")
    for class_id, name in enumerate(MERGED_CLASSES):
        print(f"  {class_id:2d} {name:20s} {merged_counts[class_id]}")


if __name__ == "__main__":
    remap()
