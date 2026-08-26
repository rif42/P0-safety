#!/usr/bin/env python3
"""
Build data/merged/ from data/raw/ through two persisted, inspectable stages
under data/build/:

  data/raw/<source>/...                     untouched, as-downloaded
  data/build/01_remap_and_filter/<source>/  per-source images+labels that
                                             survive class remap+filter
                                             (dataset/class_mapping.py),
                                             original filenames kept
  data/build/02_stratified_split/           split_assignment.csv: which
                                             split each surviving image was
                                             assigned to, and why
  data/merged/                              final training-ready dataset:
                                             images/labels renamed
                                             <source>__<original_name> and
                                             sorted into images|labels/{train,val,test}

Step 1 keeps only annotations whose merged class is in --classes (default:
the core classes 0-8) and drops any image left with none of them.

Step 2 assigns train/val/test via iterative stratification (Sechidis et al.
2011, hand-rolled here, no new dependency) instead of trusting whatever
split each source originally shipped with. The three sources ship wildly
different split ratios (e.g. one is ~95% train), and since different
classes concentrate in different sources, trusting the original per-source
splits left some classes with as little as ~2% of their instances in
val/test — too little to trust a recall estimate on. Iterative
stratification instead targets --split-ratios for every class individually,
prioritizing the rarest class first so it isn't starved by more common ones
co-occurring in the same images.

Every run rebuilds data/build/ and data/merged/ from data/raw/ from scratch
rather than incrementally patching a previous run.
"""
import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from class_mapping import DEFAULT_INCLUDED_CLASSES, MERGED_CLASSES, SOURCE_CLASS_MAPS

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
BUILD_ROOT = REPO_ROOT / "data" / "build"
STEP1_ROOT = BUILD_ROOT / "01_remap_and_filter"
STEP2_ROOT = BUILD_ROOT / "02_stratified_split"
MERGED_ROOT = REPO_ROOT / "data" / "merged"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "val", "test")

# source -> original_split -> (images_dir, labels_dir), relative to the
# source's raw folder. Only used to locate files on disk — the resulting
# split assignment comes from iterative_stratified_split(), not this key.
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
        help="Comma-separated merged class IDs to include (default: core classes 0-8)",
    )
    parser.add_argument(
        "--split-ratios",
        default="0.8,0.1,0.1",
        help="train,val,test ratios for the stratified split (default: 0.8,0.1,0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the stratified split")
    return parser.parse_args()


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def step1_remap_and_filter(included_classes):
    """Scan every source's raw files (across all its original splits, kept
    separate here — this stage doesn't merge sources or reassign splits),
    remap class IDs, and drop annotations/images that don't survive the
    class filter. Writes filtered copies to STEP1_ROOT and returns a dict
    merged_stem -> candidate info for step 2/3 to consume."""
    for source in SOURCE_DIRS:
        reset_dir(STEP1_ROOT / source)

    candidates = {}
    images_seen = 0

    for source, splits in SOURCE_DIRS.items():
        source_dir = RAW_ROOT / source
        if not source_dir.exists():
            print(f"skip {source}: not found at {source_dir}")
            continue
        class_map = SOURCE_CLASS_MAPS[source]

        for original_split, (img_rel, lbl_rel) in splits.items():
            img_dir = source_dir / img_rel
            lbl_dir = source_dir / lbl_rel
            if not img_dir.exists():
                print(f"skip {source}/{original_split}: {img_dir} missing")
                continue

            out_img_dir = STEP1_ROOT / source / original_split / "images"
            out_lbl_dir = STEP1_ROOT / source / original_split / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

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

                step1_image = out_img_dir / image_path.name
                shutil.copy2(image_path, step1_image)
                step1_label = out_lbl_dir / f"{image_path.stem}.txt"
                step1_label.write_text("\n".join(kept_lines) + "\n")

                merged_stem = f"{source}__{image_path.stem}"
                candidates[merged_stem] = {
                    "source": source,
                    "original_split": original_split,
                    "original_filename": image_path.name,
                    "suffix": image_path.suffix,
                    "step1_image": step1_image,
                    "step1_label": step1_label,
                    "kept_lines": kept_lines,
                    "classes": {int(line.split(maxsplit=1)[0]) for line in kept_lines},
                }

    return candidates, images_seen


def iterative_stratified_split(image_classes, ratios, seed=42):
    """Assign each image to a split so every class's instances land close to
    `ratios` in each split, per Sechidis et al. (2011): repeatedly pick the
    rarest not-yet-fully-assigned class, and hand its images to whichever
    split currently needs that class most, decrementing every class the
    image carries as it goes. This avoids the failure mode of a plain
    per-image random split, where a class concentrated in a handful of
    images can still land almost entirely in one split by chance.

    image_classes: dict image_id -> set of class ids present.
    ratios: dict split_name -> fraction (should sum to ~1).
    Returns: dict image_id -> split_name.
    """
    rng = random.Random(seed)
    split_names = list(ratios)

    class_totals = Counter()
    for classes in image_classes.values():
        class_totals.update(classes)

    desired = {s: {c: ratios[s] * total for c, total in class_totals.items()} for s in split_names}
    remaining_capacity = {s: ratios[s] * len(image_classes) for s in split_names}

    # A plain set's iteration order depends on Python's per-process string
    # hash randomization, which would silently make the "seeded" shuffle
    # below non-reproducible across runs (same seed, different result).
    # Always derive orderings from a sorted list instead, so `seed` is the
    # only source of randomness.
    unassigned = set(image_classes)
    assignment = {}

    while unassigned:
        class_counts = Counter()
        for img in sorted(unassigned):
            class_counts.update(image_classes[img])
        positive_classes = sorted(c for c, cnt in class_counts.items() if cnt > 0)

        if not positive_classes:
            # leftover images with none of their classes still "unprocessed"
            # (shouldn't happen since every candidate has >=1 class, but
            # handle it by remaining capacity so the loop always terminates)
            for img in sorted(unassigned):
                best = max(split_names, key=lambda s: (remaining_capacity[s], rng.random()))
                assignment[img] = best
                remaining_capacity[best] -= 1
                unassigned.discard(img)
            break

        target_class = min(positive_classes, key=lambda c: class_counts[c])
        imgs_with_class = sorted(img for img in unassigned if target_class in image_classes[img])
        rng.shuffle(imgs_with_class)

        for img in imgs_with_class:
            best_split = max(
                split_names,
                key=lambda s: (
                    round(desired[s].get(target_class, 0.0), 6),
                    round(remaining_capacity[s], 6),
                    rng.random(),
                ),
            )
            assignment[img] = best_split
            unassigned.discard(img)
            remaining_capacity[best_split] -= 1
            for c in image_classes[img]:
                desired[best_split][c] = desired[best_split].get(c, 0.0) - 1

    return assignment


def step2_stratified_split(candidates, ratios, seed):
    STEP2_ROOT.mkdir(parents=True, exist_ok=True)
    image_classes = {stem: c["classes"] for stem, c in candidates.items()}
    assignment = iterative_stratified_split(image_classes, ratios, seed=seed)

    assignment_path = STEP2_ROOT / "split_assignment.csv"
    with assignment_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["merged_stem", "source", "original_split", "split"])
        writer.writeheader()
        for stem, info in candidates.items():
            writer.writerow(
                {
                    "merged_stem": stem,
                    "source": info["source"],
                    "original_split": info["original_split"],
                    "split": assignment[stem],
                }
            )
    return assignment, assignment_path


def step3_materialize_merged(candidates, assignment):
    for split in SPLIT_NAMES:
        reset_dir(MERGED_ROOT / "images" / split)
        reset_dir(MERGED_ROOT / "labels" / split)

    manifest_rows = []
    label_rows = []
    kept_counts = [0] * len(MERGED_CLASSES)

    for merged_stem, info in candidates.items():
        split = assignment[merged_stem]

        out_image = MERGED_ROOT / "images" / split / f"{merged_stem}{info['suffix']}"
        shutil.copy2(info["step1_image"], out_image)
        out_label = MERGED_ROOT / "labels" / split / f"{merged_stem}.txt"
        shutil.copy2(info["step1_label"], out_label)

        for line in info["kept_lines"]:
            class_id = int(line.split(maxsplit=1)[0])
            kept_counts[class_id] += 1
            label_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": MERGED_CLASSES[class_id],
                    "source": info["source"],
                    "file": merged_stem,
                }
            )

        manifest_rows.append(
            {
                "merged_filename": f"{merged_stem}{info['suffix']}",
                "source": info["source"],
                "original_filename": info["original_filename"],
                "original_split": info["original_split"],
                "split": split,
                "num_annotations": len(info["kept_lines"]),
            }
        )

    manifest_path = MERGED_ROOT / "merge_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "merged_filename",
                "source",
                "original_filename",
                "original_split",
                "split",
                "num_annotations",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    labels_long_path = MERGED_ROOT / "labels_long.csv"
    with labels_long_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_id", "class_name", "source", "file"])
        writer.writeheader()
        writer.writerows(label_rows)

    return manifest_rows, label_rows, kept_counts, manifest_path, labels_long_path


def build(included_classes, ratios, seed):
    candidates, images_seen = step1_remap_and_filter(included_classes)
    assignment, assignment_path = step2_stratified_split(candidates, ratios, seed)
    manifest_rows, label_rows, kept_counts, manifest_path, labels_long_path = step3_materialize_merged(
        candidates, assignment
    )

    by_source_split = Counter((row["source"], row["split"]) for row in manifest_rows)
    images_kept = len(manifest_rows)

    included_names = [MERGED_CLASSES[c] for c in sorted(included_classes)]
    print(f"Included classes: {sorted(included_classes)} {included_names}")
    print(f"Split ratios: {ratios}  seed={seed}")
    print(
        f"Images scanned: {images_seen}, kept: {images_kept}, "
        f"dropped (no included classes present): {images_seen - images_kept}"
    )
    for (source, split), count in sorted(by_source_split.items()):
        print(f"  {source:24s} {split:6s} {count}")

    print("Kept instance counts, and their split (target vs. actual %):")
    class_split_counts = Counter((row["class_id"], row["split"]) for row in label_rows)
    for class_id in sorted(included_classes):
        total = kept_counts[class_id]
        pct = {
            s: (100 * class_split_counts[(class_id, s)] / total if total else 0.0) for s in SPLIT_NAMES
        }
        actual = "  ".join(f"{s}={pct[s]:5.1f}%" for s in SPLIT_NAMES)
        print(f"  {class_id:2d} {MERGED_CLASSES[class_id]:12s} n={total:6d}  {actual}")

    print(f"Step 1 (remap+filter) written to {STEP1_ROOT}")
    print(f"Step 2 (split assignment) written to {assignment_path}")
    print(f"Merged dataset written to {MERGED_ROOT}")
    print(f"  manifest: {manifest_path}")
    print(f"  per-annotation table: {labels_long_path}")


if __name__ == "__main__":
    args = parse_args()
    included = {int(c) for c in args.classes.split(",")}
    ratio_values = [float(v) for v in args.split_ratios.split(",")]
    if len(ratio_values) != 3 or abs(sum(ratio_values) - 1.0) > 1e-6:
        raise SystemExit(f"--split-ratios must be 3 comma-separated values summing to 1.0, got {args.split_ratios}")
    ratios = dict(zip(SPLIT_NAMES, ratio_values))
    build(included, ratios, args.seed)
