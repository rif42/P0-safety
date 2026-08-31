"""Video inference for YOLO26m merged PPE model.

Usage:
    python scripts/infer_video.py --source path/to/video.mp4
    python scripts/infer_video.py --source 0                      # webcam
    python scripts/infer_video.py --source video.mp4 --output out.mp4 --conf 0.25 --show

Defaults to runs/detect/yolo26m_merged_150e/weights/best.pt — override with
--weights or HIVIS_MODEL_PATH env var.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2

# Allow importing detector for colors when run from repo root or scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "yolo26m_merged_150e" / "weights" / "best.pt"

# Fallback palette if detector not importable
FALLBACK_COLORS = {
    "person": (140, 141, 144),
    "hardhat": (233, 155, 61),
    "helmet": (233, 155, 61),
    "vest": (107, 190, 47),
    "safety vest": (107, 190, 47),
    "gloves": (217, 184, 0),
    "boots": (255, 124, 176),
    "mask": (182, 196, 46),
    "no-hardhat": (66, 77, 255),
    "no-helmet": (66, 77, 255),
    "no-safety vest": (0, 138, 255),
    "no-vest": (0, 138, 255),
    "no-gloves": (60, 111, 255),
    "no-boots": (91, 24, 194),
    "no-mask": (176, 39, 156),
}

# Try to reuse detector colors if available (nice to keep consistent with Streamlit app)
DETECTOR_COLORS = None
try:
    sys.path.insert(0, str(REPO_ROOT / "streamlit_app"))
    import detector as _det

    DETECTOR_COLORS = {}
    for k, v in _det.CLASS_META.items():
        # CLASS_META keys are normalized (hardhat, nohardhat, vest, novest, ...)
        # model class names are like "helmet", "no-helmet", "vest", "no-vest", "person"
        hexc = v["color"].lstrip("#")
        r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
        DETECTOR_COLORS[k] = (b, g, r)  # cv2 BGR
    # map detector keys to YOLO raw names
    _KEY_TO_RAW = {
        "person": ["person"],
        "hardhat": ["hardhat", "helmet"],
        "nohardhat": ["no-hardhat", "no-helmet"],
        "vest": ["vest", "safety vest"],
        "novest": ["no-vest", "no-safety vest"],
        "gloves": ["gloves"],
        "nogloves": ["no-gloves"],
        "boots": ["boots"],
        "noboots": ["no-boots"],
        "mask": ["mask"],
        "nomask": ["no-mask"],
    }
    _RAW_TO_BGR = {}
    for dk, raws in _KEY_TO_RAW.items():
        if dk in DETECTOR_COLORS:
            for r in raws:
                _RAW_TO_BGR[r.lower()] = DETECTOR_COLORS[dk]
except Exception:
    _RAW_TO_BGR = {}


def _color_for(class_name: str):
    key = class_name.strip().lower()
    if key in _RAW_TO_BGR:
        return _RAW_TO_BGR[key]
    if key in FALLBACK_COLORS:
        # fallback is stored as RGB-ish, convert to BGR
        r, g, b = FALLBACK_COLORS[key]
        # FALLBACK_COLORS values above were actually picked as BGR-friendly; keep as-is for determinism
        # but ensure tuple is BGR — stored as (b,g,r) already except for fallback dict which is conceptual
        # For simplicity return as BGR directly:
        return FALLBACK_COLORS[key][::-1] if key not in _RAW_TO_BGR else FALLBACK_COLORS[key]
    # deterministic hash fallback
    h = hash(key) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26m video inference")
    p.add_argument("--source", required=True, help="Video path, image path, or 0 for webcam (also supports RTSP/YouTube via opencv)")
    p.add_argument("--weights", default=None, help="Path to best.pt (default: runs/detect/yolo26m_merged_150e/weights/best.pt or $HIVIS_MODEL_PATH)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (default 0.5)")
    p.add_argument("--output", default=None, help="Output video path (e.g. runs/predict/out.mp4). If omitted and --save given, auto-named.")
    p.add_argument("--save", action="store_true", help="Save annotated video (requires --output or auto-names to runs/predict/)")
    p.add_argument("--show", action="store_true", default=True, help="Show window (default on)")
    p.add_argument("--no-show", dest="show", action="store_false", help="Disable window display (useful on headless)")
    p.add_argument("--device", default=None, help="Device e.g. 0, cpu, cuda:0 (default auto)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    return p.parse_args()


def resolve_weights(args_weights):
    if args_weights:
        return Path(args_weights)
    env = os.environ.get("HIVIS_MODEL_PATH")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p
    return DEFAULT_WEIGHTS


def main():
    args = parse_args()

    weights = resolve_weights(args.weights)
    if not weights.exists():
        print(f"[error] weights not found: {weights}", file=sys.stderr)
        print(f"  Pass --weights or set HIVIS_MODEL_PATH. Default was {DEFAULT_WEIGHTS}", file=sys.stderr)
        sys.exit(1)

    # Normalize source: "0" -> 0 for webcam
    source = args.source.strip()
    cap_source = 0 if source == "0" else source
    if isinstance(cap_source, str) and not source.startswith("rtsp") and source != "0":
        # allow pathlib check for file existence hint
        if not Path(source).exists():
            print(f"[warn] source file does not exist: {source}", file=sys.stderr)

    print(f"[info] weights : {weights}")
    print(f"[info] source  : {source}")
    print(f"[info] conf    : {args.conf}  iou: {args.iou}  imgsz: {args.imgsz}  device: {args.device or 'auto'}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    # move to device if specified (YOLO handles auto otherwise)
    if args.device:
        try:
            model.to(args.device)
        except Exception as e:
            print(f"[warn] model.to({args.device}) failed: {e}", file=sys.stderr)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[error] could not open source: {source}", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 120:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Output writer setup
    writer = None
    out_path = None
    if args.save or args.output:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = REPO_ROOT / "runs" / "predict" / f"{Path(source).stem}_annotated.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # ensure size known; if w/h not yet known (webcam), defer creation until first frame
        if w > 0 and h > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                print(f"[warn] could not open VideoWriter for {out_path}, trying avc1", file=sys.stderr)
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            print(f"[info] saving to: {out_path}  ({w}x{h} @ {fps:.1f}fps)")

    # Warm up model.names for labels/colors
    names = getattr(model, "names", {})  # {id: name}

    frame_idx = 0
    t0 = time.time()
    infer_ms = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1

            # Lazily create writer if webcam size wasn't known at open time
            if writer is None and out_path is not None:
                fh, fw = frame.shape[:2]
                # writer expects (width, height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
                print(f"[info] saving to: {out_path}  ({fw}x{fh} @ {fps:.1f}fps)")

            t1 = time.time()
            results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
            t2 = time.time()
            infer_ms.append((t2 - t1) * 1000)

            r = results[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    color = _color_for(label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    txt = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # overlay stats
            cv2.putText(frame, f"frame {frame_idx}" + (f"/{total}" if total else ""), (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow("YOLO26m - video inference (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[info] quit by user")
                    break

            if frame_idx % 30 == 0:
                avg = sum(infer_ms[-30:]) / min(30, len(infer_ms))
                print(f"  ... frame {frame_idx}" + (f"/{total}" if total else "") + f"  avg infer {avg:.1f} ms")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    elapsed = time.time() - t0
    avg_ms = sum(infer_ms) / len(infer_ms) if infer_ms else 0
    print(f"[done] frames: {frame_idx}  elapsed: {elapsed:.1f}s  avg infer: {avg_ms:.1f} ms  avg fps: {frame_idx/elapsed:.1f}" if frame_idx else "[done] no frames read")
    if out_path is not None:
        print(f"[done] output: {out_path}" + ("" if writer else " (writer not opened - check codec)"))


if __name__ == "__main__":
    main()
