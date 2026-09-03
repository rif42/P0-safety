#!/usr/bin/env python
"""Run perception_multi.py on every image in demo/assets/demo-pics.

Queries (default): helmet, safety vest, gloves, boots, person
Each image gets its own output folder under ./outputs/demo-pics/<subfolder>/<stem>/
so results don't overwrite each other.

Usage:
    python demo/run_demo_pics.py
    python demo/run_demo_pics.py --task detection --dry-run   # list what would run
    python demo/run_demo_pics.py --queries "helmet,person" --out-root ./outputs/ppe
    python demo/run_demo_pics.py --image-root demo/assets/demo-pics/typical
"""
import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_QUERIES = "helmet,safety vest,gloves,boots,person"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG", ".WEBP"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-root", default="demo/assets/demo-pics", help="root dir to glob recursively")
    ap.add_argument("--queries", default=DEFAULT_QUERIES, help='comma-separated queries (default: "%(default)s")')
    ap.add_argument("--task", default="detection", choices=["detection", "segmentation"])
    ap.add_argument("--out-root", default="outputs/demo-pics", help="per-image output goes to <out-root>/<relative-path-stem>/")
    ap.add_argument("--extra-args", default="", help='extra args forwarded to perception_multi.py, e.g. "--engine-type paged --compile"')
    ap.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = ap.parse_args()

    root = Path(args.image_root)
    if not root.exists():
        print(f"Image root not found: {root}", file=sys.stderr)
        sys.exit(1)

    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS)
    if not images:
        print(f"No images found under {root}")
        sys.exit(0)

    print(f"Found {len(images)} images under {root}")
    print(f"Queries : {args.queries}")
    print(f"Task    : {args.task}")
    print(f"Out root: {args.out_root}")
    if args.dry_run:
        print("(dry-run)\n")

    # perception_multi.py is sibling to this script
    script = Path(__file__).parent / "perception_multi.py"
    extra = args.extra_args.split() if args.extra_args else []

    for i, img in enumerate(images, 1):
        # outputs/demo-pics/challenging/foo.jpg -> outputs/demo-pics/challenging/foo/
        # outputs/demo-pics/typical/bar.jpg    -> outputs/demo-pics/typical/bar/
        rel = img.relative_to(root)
        out_dir = Path(args.out_root) / rel.parent / rel.stem
        cmd = [
            sys.executable, str(script),
            "--image", str(img),
            "--queries", args.queries,
            "--task", args.task,
            "--out-dir", str(out_dir),
            *extra,
        ]
        print(f"\n[{i}/{len(images)}] {img} -> {out_dir}")
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if args.dry_run:
            continue
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  [warn] failed with exit code {ret.returncode}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
