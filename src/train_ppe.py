#!/usr/bin/env python3
"""
train_ppe.py
============
Train the PPE detector from the command line, outside the Jupyter kernel.

Run long training this way, not in a notebook cell. The first run of this project
showed why: the training process starved the VM, the websocket timed out for 118
seconds, autosave then failed with ENOMEM, and when the OOM killer fired it took
the kernel - along with every cell output. A detached process writing to a log
file survives a dropped browser tab, a closed laptop and a restarted Jupyter, and
its output is still there afterwards to read.

Use the notebook for the dataset audit, evaluation and fusion. Use this for the
part that takes hours.

Typical use
-----------
    # see the plan and the memory budget without training anything
    python src/train_ppe.py --data /workspace/data/ppe_yolo26/data.yaml --dry-run

    # detached run, safe to close the browser
    nohup python src/train_ppe.py \
        --data /workspace/data/ppe_yolo26/data.yaml \
        --imgsz 960 --epochs 120 > runs/train.log 2>&1 &
    tail -f runs/train.log

    # pick up where an interrupted run left off
    python src/train_ppe.py --resume runs/ppe/detect_v1/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppe_memory import probe, recommend, report, preflight  # noqa: E402


def build_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="path to data.yaml")
    ap.add_argument("--resume", type=Path, default=None,
                    help="path to last.pt to resume an interrupted run")

    ap.add_argument("--imgsz", type=int, default=640,
                    help="training resolution. Raise it for small objects such "
                         "as boots; batch is rescaled automatically.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default=None, help="0, 0,1, or cpu")

    # Overrides. Left unset, each is derived from the measured machine.
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache", default=None, choices=["ram", "disk", "false"])
    ap.add_argument("--model", default=None, help="e.g. yolo26s.pt")

    ap.add_argument("--project", default="runs/ppe")
    ap.add_argument("--name", default="detect_v1")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the memory budget, then stop")
    ap.add_argument("--force", action="store_true",
                    help="train even if preflight says the machine is too small")
    return ap


def main() -> int:
    args = build_args().parse_args()

    facts = probe()
    rec = recommend(facts, imgsz=args.imgsz, base_imgsz=640)
    print(report(facts, rec), "\n")

    ok, msg = preflight(facts, rec)
    print(f"preflight: {msg}\n")
    if not ok and not args.force:
        print("Refusing to start. Fix the memory budget, or pass --force if you\n"
              "have a reason to think the probe is wrong.")
        return 2

    model_name = args.model or f"yolo26{rec.model_scale}.pt"
    batch = args.batch if args.batch is not None else rec.batch
    workers = args.workers if args.workers is not None else rec.workers
    if args.cache is None:
        cache = rec.cache
    else:
        cache = False if args.cache == "false" else args.cache

    plan = dict(model=model_name, data=str(args.data) if args.data else None,
                imgsz=args.imgsz, epochs=args.epochs, batch=batch,
                workers=workers, cache=cache,
                device=args.device if args.device is not None else (0 if facts.gpu_count else "cpu"))
    print("Plan:", json.dumps(plan, indent=2, default=str), "\n")

    if args.dry_run:
        print("Dry run - nothing trained.")
        return 0
    if not args.resume and not args.data:
        print("--data is required unless --resume is given.")
        return 2

    from ultralytics import YOLO

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = YOLO(str(args.resume))
        model.train(resume=True)
        return 0

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_plan.json").write_text(json.dumps(
        {"facts": facts.to_dict(), "recommendation": rec.to_dict(), "plan": plan},
        indent=2, default=str))

    model = YOLO(model_name)
    model.train(
        data     = str(args.data),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = batch,
        workers  = workers,
        cache    = cache,
        device   = plan["device"],
        patience = args.patience,
        project  = args.project,
        name     = args.name,
        exist_ok = True,

        optimizer = "auto",       # YOLO26 resolves this to MuSGD
        cos_lr    = True,
        warmup_epochs = 3.0,
        amp       = True,
        seed      = 0,
        plots     = True,

        # geometry - conservative, cameras are fixed and level
        degrees = 3.0, translate = 0.10, scale = 0.55, shear = 1.0,
        perspective = 0.0004, fliplr = 0.5, flipud = 0.0,
        # photometric - lighting and weather
        hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.45,
        # composition
        mosaic = 1.0, close_mosaic = 15, mixup = 0.05,
    )
    print(f"\nDone. Best weights: {out_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
