#!/usr/bin/env python3
"""
Flatten YOLO polygon label rows to bounding boxes, in place.

Ultralytics rejects any label file that mixes polygon rows (class x1 y1 x2
y2 ...) with detection rows (class xc yc w h) — see verify_image_label():

    assert not any(len(x) == 5 for x in lb), "labels mix segment and detection rows"

The whole image and every box in it is then dropped as "corrupt". On the
SuperVisor.v1 Roboflow export that silently cost 67/485 train images and
20/135 valid images (~14% of each split) before training even started.

Every polygon becomes its bounding rect, so a detection dataset comes out
uniformly 5-column and nothing is discarded. Dry run unless --write.

    python3 scripts/fix_mixed_labels.py <dataset_root>            # report
    python3 scripts/fix_mixed_labels.py <dataset_root> --write    # apply
    python3 scripts/fix_mixed_labels.py --selfcheck
"""

import sys
from pathlib import Path


def to_box(parts):
    """One label row -> 5-column detection row. Already-5-column rows pass through."""
    if len(parts) == 5:
        return parts
    coords = [min(max(float(v), 0.0), 1.0) for v in parts[1:]]  # roboflow can emit a hair out of range
    xs, ys = coords[0::2], coords[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [parts[0]] + [f"{v:.6f}" for v in ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)]


def fix(root, write=False):
    files = rows = recovered = 0
    for path in sorted(Path(root).rglob("labels/*.txt")):
        parts = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
        polys = [p for p in parts if len(p) != 5]
        if not polys:
            continue
        files += 1
        rows += len(polys)
        recovered += any(len(p) == 5 for p in parts)  # this file was the fatal mixed kind
        if write:
            path.write_text("".join(" ".join(to_box(p)) + "\n" for p in parts))
    verb = "Rewrote" if write else "Would rewrite"
    print(f"{verb} {files} label files, {rows} polygon rows -> boxes.")
    print(f"{recovered} images were mixed-format and would otherwise be dropped by Ultralytics.")
    if not write and files:
        print("Dry run — re-run with --write to apply.")


def selfcheck():
    assert to_box("9 0.5 0.4 0.1 0.2".split()) == "9 0.5 0.4 0.1 0.2".split(), "5-col rows must pass through untouched"
    # unit square traced anticlockwise from (0.2,0.3) to (0.6,0.7) -> centre 0.4,0.5 size 0.4,0.4
    box = to_box("2 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7".split())
    assert box[0] == "2", box
    assert [round(float(v), 6) for v in box[1:]] == [0.4, 0.5, 0.4, 0.4], box
    # out-of-range vertices clamp rather than producing a box Ultralytics will reject
    box = to_box("0 -0.1 0.5 1.4 0.9 0.3 0.2".split())
    assert [round(float(v), 6) for v in box[1:]] == [0.5, 0.55, 1.0, 0.7], box
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selfcheck" in args:
        selfcheck()
    elif args:
        fix(args[0], write="--write" in args)
    else:
        sys.exit(__doc__)
