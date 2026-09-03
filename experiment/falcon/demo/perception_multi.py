"""Falcon Perception — multi-class single-pass detection/segmentation.

Detect multiple classes in **one batched forward pass** (one prefill + shared
decode). Same image, N different text queries — all tokenized together and
run through the model once. Far cheaper than looping N times.

This mirrors the reference snippet you provided, but uses the real repo
APIs and works on both CUDA and CPU. On CPU (e.g. Ryzen 8845HS) the
paged engine requires CUDA, so the batch engine is used automatically.

Usage
-----
    # default 4 classes on the bundled sample
    python demo/perception_multi.py --image demo/assets/sample.jpg

    # explicit classes (tyro list style)
    python demo/perception_multi.py --image demo/assets/sample.jpg --queries person --queries car --queries "traffic light" --queries dog

    # comma-separated shorthand
    python demo/perception_multi.py --image demo/assets/sample.jpg --queries "person,car,traffic light,dog"
    python demo/perception_multi.py --image demo/assets/sample.jpg --query "person,car"

    # detection (default, CPU-safe) vs segmentation (needs more RAM, OOMs at 1024 on CPU)
    python demo/perception_multi.py --image demo/assets/sample.jpg --task detection
    python demo/perception_multi.py --image demo/assets/sample.jpg --task segmentation --no-compile

    # force paged engine on CUDA
    python demo/perception_multi.py --image demo/assets/sample.jpg --engine-type paged --compile

    # custom output dir
    python demo/perception_multi.py --image demo/assets/sample.jpg --out-dir ./outputs/multi
"""

from pathlib import Path
from typing import Literal

import torch
import tyro

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    cuda_timed,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.data import load_image, stream_samples_from_hf_dataset
from falcon_perception.nvtx import nvtx_range

setup_torch_config()

# --- merged labeled overlay helpers ---
_PALETTE_UINT8_MERGED: list[tuple[int, int, int]] = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
]
_COLOR_NAMES_MERGED = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple", "springgreen", "deeppink"]


def _save_merged_labeled_image(
    pil_image,
    flat_detections: list[dict],
    class_names: list[str],
    out_path,
    *,
    task: str = "detection",
) -> None:
    """Save one image with ALL boxes/masks overlaid, each box labeled by class.

    - One consistent color per class (by order in class_names).
    - Bounding boxes are labeled with filled background + text.
    - For segmentation, masks are composited per-class (same color as its boxes).
    """
    from pathlib import Path as _P
    from PIL import Image as _Image, ImageDraw as _Draw, ImageFont as _Font
    import numpy as _np

    out_path = _P(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = pil_image.size
    base = _np.array(pil_image.convert("RGB"))  # (H,W,3) uint8

    # per-class color map
    cls_to_color = {c: _PALETTE_UINT8_MERGED[i % len(_PALETTE_UINT8_MERGED)] for i, c in enumerate(class_names)}
    # no detections -> just save copy
    if not flat_detections:
        pil_image.convert("RGB").save(str(out_path), quality=95)
        print(f"[merged] no detections, saved copy -> {out_path}")
        return

    # --- mask compositing (segmentation) — per-class color, index-map in one vectorised pass ---
    if task == "segmentation":
        masks = []
        det_for_mask = []  # parallel to masks
        for d in flat_detections:
            m = d.get("mask")
            if m is None:
                continue
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            m = _np.asarray(m)
            # binarize if logits/prob
            if m.dtype != _np.uint8:
                m = (m > 0.5).astype(_np.uint8) if m.max() <= 1.5 else (m > 127).astype(_np.uint8)
            if m.shape != (H, W):
                m = _np.array(_Image.fromarray(m.astype(_np.uint8)).resize((W, H), resample=_Image.NEAREST))
            masks.append(m)
            det_for_mask.append(d)

        if masks:
            # order largest first so smallest wins (overwrites)
            areas = [_np.sum(m) for m in masks]
            order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
            mask_idx = _np.full((H, W), -1, dtype=_np.int32)
            for ri, oi in enumerate(order):
                mask_idx[masks[oi] > 0] = ri
            has_mask = mask_idx >= 0
            if has_mask.any():
                ordered_colors = _np.array(
                    [cls_to_color[det_for_mask[order[i]]["label"]] for i in range(len(masks))],
                    dtype=_np.uint8,
                )  # (N,3)
                clamped = _np.where(has_mask, mask_idx, 0)
                fill_rgb = ordered_colors[clamped]  # (H,W,3)
                alpha = 0.35
                base = _np.where(
                    has_mask[:, :, None],
                    (alpha * fill_rgb.astype(_np.float32) + (1 - alpha) * base.astype(_np.float32) + 0.5).astype(_np.uint8),
                    base,
                )
                # thin border from index-map edges
                border = _np.zeros((H, W), dtype=bool)
                border[:, 1:] |= mask_idx[:, 1:] != mask_idx[:, :-1]
                border[:, :-1] |= mask_idx[:, 1:] != mask_idx[:, :-1]
                border[1:, :] |= mask_idx[1:, :] != mask_idx[:-1, :]
                border[:-1, :] |= mask_idx[1:, :] != mask_idx[:-1, :]
                border &= has_mask
                if border.any():
                    bright = _np.clip(0.65 * ordered_colors.astype(_np.float32) + 89.25, 0, 255).astype(_np.uint8)
                    border_rgb = bright[clamped]
                    base[border] = border_rgb[border]

    # --- draw labeled bboxes ---
    pil_out = _Image.fromarray(base)
    draw = _Draw.Draw(pil_out, "RGBA")
    try:
        font = _Font.load_default()
    except Exception:
        font = None

    for det in flat_detections:
        xy = det.get("xy")
        hw = det.get("hw")
        label = str(det.get("label") or "")
        if xy is None:
            continue
        cx, cy = float(xy["x"]) * W, float(xy["y"]) * H
        if hw and "w" in hw and "h" in hw:
            bw, bh = float(hw["w"]) * W, float(hw["h"]) * H
        else:
            bw = bh = 10
        x0 = max(0, int(round(cx - bw / 2)))
        y0 = max(0, int(round(cy - bh / 2)))
        x1 = min(W - 1, int(round(cx + bw / 2)))
        y1 = min(H - 1, int(round(cy + bh / 2)))
        color = cls_to_color.get(label, (255, 0, 0))
        # box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        # label bg + text
        if label:
            # text size
            if font is not None:
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = len(label) * 6 + 4, 10
            else:
                tw, th = len(label) * 6 + 4, 10
            pad = 2
            bg_x0, bg_y0 = x0, y0 - th - pad * 2
            bg_x1, bg_y1 = x0 + tw + pad * 2, y0
            # if not enough space above, draw inside top of box
            if bg_y0 < 0:
                bg_y0 = y0
                bg_y1 = y0 + th + pad * 2
                # keep inside image
                if bg_y1 > H:
                    bg_y1 = H
            bg_x1 = min(W, bg_x1)
            draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=color + (255,) if len(color) == 3 else color)
            # choose text color by luminance
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            tcol = (0, 0, 0) if lum > 150 else (255, 255, 255)
            draw.text((bg_x0 + pad, bg_y0 + pad // 2), label, fill=tcol, font=font)

    pil_out.save(str(out_path), quality=95)
    print(f"[merged] saved {out_path}  ({len(flat_detections)} boxes: {', '.join(class_names)})")


def _save_yolo_txt(
    flat_detections: list[dict],
    class_names: list[str],
    out_path,
    *,
    also_print: bool = True,
) -> None:
    """Save YOLO-format txt: one line per box ``<cls_id> <cx> <cy> <w> <h>`` (normalized 0-1).

    - class id is the index in ``class_names``.
    - cx/cy/w/h are taken directly from the normalized ``xy``/``hw`` dicts
      that Falcon already predicts (no pixel conversion needed).
    - Output is ``<out_path>`` (a ``.txt`` file). Parent dirs are created.
    - Empty detections -> empty file (valid YOLO: no objects).
    - Also prints lines to stdout when ``also_print`` is True.
    """
    from pathlib import Path as _P

    out_path = _P(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cls_to_id = {c: i for i, c in enumerate(class_names)}
    lines: list[str] = []
    for det in flat_detections:
        label = str(det.get("label") or "")
        if label not in cls_to_id:
            continue
        cid = cls_to_id[label]
        xy = det.get("xy") or {}
        hw = det.get("hw") or {}
        try:
            cx = float(xy.get("x", 0))
            cy = float(xy.get("y", 0))
            w = float(hw.get("w", 0))
            h = float(hw.get("h", 0))
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        # clamp 0-1 for safety
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        w = min(1.0, max(0.0, w))
        h = min(1.0, max(0.0, h))
        conf = det.get("conf")
        if conf is not None:
            try:
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(conf):.4f}")
            except Exception:
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        else:
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[yolo] saved {out_path}  ({len(lines)} boxes, classes={class_names})")
    if also_print:
        if lines:
            print("[yolo] YOLO format (cls_id cx cy w h normalized):")
            for ln in lines:
                print(f"  {ln}")
        else:
            print("[yolo] no detections -> empty file")
    # also print class map for reference
    print(f"[yolo] class map: {', '.join(f'{i}={n}' for i, n in enumerate(class_names))}")


@torch.inference_mode()
def main(
    image: str | None = None,
    queries: list[str] | None = None,
    query: str | None = None,
    task: Literal["segmentation", "detection"] = "detection",
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    device: str | None = None,
    dtype: Literal["bfloat16", "float32", "float"] = "float32",
    engine_type: Literal["batch", "paged"] = "batch",
    flex_attn_safe: bool = False,
    out_dir: str = "./outputs/",
    compile: bool = False,
    cudagraph: bool = False,
):
    """Run Falcon Perception on one image with multiple class queries in one pass.

    Each query becomes its own sequence/prompt. Batch engine packs all
    (image, prompt) pairs into one batch; paged engine creates one Sequence
    per class. Either way it's a single engine.generate() call.

    `--queries` accepts repeated flags or a single comma-separated string:
      --queries person --queries car        -> ["person", "car"]
      --queries "person,car,traffic light" -> ["person", "car", "traffic light"]
    `--query` is a backwards-compat alias for the same.
    """
    # --- normalize queries ---
    raw: list[str] = []
    if queries is not None and len(queries) > 0:
        # expand comma-separated entries (tyro passes ["person,car"] as one element)
        for q in queries:
            if "," in q:
                raw.extend([s.strip() for s in q.split(",") if s.strip()])
            elif q.strip():
                raw.append(q.strip())
    elif query is not None and query.strip():
        raw = [s.strip() for s in query.split(",") if s.strip()]
    else:
        raw = ["person", "car", "traffic light", "dog"]

    # remove empty / dedup preserve order
    seen: set[str] = set()
    classes: list[str] = []
    for q in raw:
        if q and q not in seen:
            seen.add(q)
            classes.append(q)

    if not classes:
        print("No queries provided.")
        return

    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1} if flex_attn_safe else {}

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id or PERCEPTION_MODEL_ID,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )
    resolved_device = model.device

    if task == "segmentation" and not model_args.do_segmentation:
        print("Model does not support segmentation (do_segmentation=False), falling back to detection.")
        task = "detection"

    if image is not None:
        pil_image = load_image(image).convert("RGB")
    else:
        print("No --image provided, loading a demo sample ...")
        sample = stream_samples_from_hf_dataset("tiiuae/PBench", split="level_1")[0]
        pil_image = sample["image"]
        sample_query = sample.get("expression") or sample.get("expressions") or "all objects"
        if isinstance(sample_query, list):
            sample_query = ", ".join(str(q) for q in sample_query) if sample_query else "all objects"
        print(f"  Sample query: {sample_query!r}")
        # if no explicit classes, use sample query as single class
        if queries is None and query is None:
            classes = [str(sample_query)]

    w, h = pil_image.size
    print(f"  Task    : {task}")
    print(f"  Classes : {classes}")
    print(f"  Image   : {w} x {h}  ({w*h/1e6:.1f} MP)")
    print(f"  Engine  : {engine_type}  device={resolved_device} dtype={dtype} compile={compile}")
    print()

    from falcon_perception.data import ImageProcessor

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "perception_input.jpg").parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out_path / "perception_input.jpg")

    # --- CPU fallback: paged engine requires CUDA ---
    if engine_type == "paged" and not torch.cuda.is_available():
        print("[warn] --engine-type paged requires CUDA; falling back to batch on CPU.")
        engine_type = "batch"

    if engine_type == "paged":
        # Reference snippet path, corrected to real API
        from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams, Sequence
        from falcon_perception.visualization_utils import render_paged_inference_outputs

        # Use batch size = num classes (paged engine batches Sequences)
        engine = PagedInferenceEngine(
            model, tokenizer, image_processor,
            max_batch_size=max(2, len(classes)),
            max_seq_length=8192,
            n_pages=128,
            page_size=128,
            prefill_length_limit=8192,
            enable_hr_cache=False,
            capture_cudagraph=cudagraph and torch.cuda.is_available(),
            kernel_options=kernel_options or None,
        )

        sampling_params = SamplingParams(stop_token_ids=stop_token_ids)

        sequences = [
            Sequence(
                text=build_prompt_for_task(cls_name, task),
                image=pil_image,
                min_image_size=256,
                max_image_size=1024,
                task=task,
                request_idx=i,
            )
            for i, cls_name in enumerate(classes)
        ]

        # Single generate() for all classes
        print(f"Running paged inference for {len(classes)} classes in one pass ...")
        with nvtx_range("Generate"):
            with cuda_timed() as t:
                engine.generate(sequences, sampling_params=sampling_params, use_tqdm=True, print_stats=True)
        print(f"Done in {t.elapsed:.1f}s")

        from falcon_perception.visualization_utils import pair_bbox_entries
        print(f"\n{'='*60}")
        print("Results (per class)")
        print("="*60)
        for cls_name, seq in zip(classes, sequences):
            aux = seq.output_aux
            n = 0
            if aux is not None:
                if task == "segmentation":
                    n = len(aux.masks_rle) if hasattr(aux, "masks_rle") else 0
                else:
                    n = len(pair_bbox_entries(aux.bboxes_raw)) if hasattr(aux, "bboxes_raw") else 0
            print(f"  {cls_name:20s} : {n} {'masks' if task=='segmentation' else 'boxes'}")

        render_paged_inference_outputs(sequences, image_processor, output_dir=out_dir, task=task)
        # --- merged labeled image + YOLO txt (all classes in one) ---
        flat: list[dict] = []
        try:
            from falcon_perception.visualization_utils import detections_from_sequence as _dets_from_seq
            for cls_name, seq in zip(classes, sequences):
                for d in _dets_from_seq(seq):
                    d2 = dict(d)
                    d2["label"] = cls_name
                    flat.append(d2)
            _save_merged_labeled_image(pil_image, flat, classes, out_path / "merged.jpg", task=task)
        except Exception as e:
            print(f"[merged] skip ({e})")
        try:
            _save_yolo_txt(flat, classes, out_path / "predictions_yolo.txt")
        except Exception as e:
            print(f"[yolo] skip ({e})")
        sub = "masks" if task == "segmentation" else "boxes"
        print(f"\n  Input image : {out_path / 'perception_input.jpg'}")
        print(f"  Output dir  : {out_path / sub}")
        print(f"  Merged      : {out_path / 'merged.jpg'}")
        print(f"  YOLO txt    : {out_path / 'predictions_yolo.txt'}")

    else:  # batch
        from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate
        from falcon_perception.visualization_utils import render_batch_inference_outputs

        prompts = [build_prompt_for_task(q, task) for q in classes]

        engine = BatchInferenceEngine(model, tokenizer, kernel_options=kernel_options or None)

        # One call packs all (image, prompt) pairs into a single batch
        batch_inputs = process_batch_and_generate(
            tokenizer,
            [(pil_image, p) for p in prompts],
            max_length=4096,
            min_dimension=256,
            max_dimension=1024,
        )
        batch_inputs = {
            k: (v.to(resolved_device) if torch.is_tensor(v) else v)
            for k, v in batch_inputs.items()
        }
        print(f"Batch tokens: {batch_inputs['tokens'].shape}  pixel_values: {batch_inputs['pixel_values'].shape}")
        print(f"Running batch inference for {len(classes)} classes in one pass ...")

        with cuda_timed() as t:
            _, aux_out = engine.generate(
                **batch_inputs,
                max_new_tokens=2048,
                temperature=0.0,
                stop_token_ids=stop_token_ids,
                seed=42,
                task=task,
            )
        print(f"Done in {t.elapsed:.1f}s")

        from falcon_perception.aux_output import AuxOutput
        from falcon_perception.visualization_utils import pair_bbox_entries

        print(f"\n{'='*60}")
        print("Results (per class)")
        print("="*60)
        for cls_name, aux in zip(classes, aux_out):
            if isinstance(aux, AuxOutput):
                if task == "segmentation":
                    # detection+seg share bboxes_raw length; masks are in auxiliary
                    n = len(pair_bbox_entries(aux.bboxes_raw))
                    kind = "masks"
                else:
                    n = len(pair_bbox_entries(aux.bboxes_raw))
                    kind = "boxes"
                print(f"  {cls_name:20s} : {n} {kind}")
            else:
                # fallback for raw list form
                n = len(aux) // (3 if task == "segmentation" else 2) if aux else 0
                print(f"  {cls_name:20s} : {n} {'masks' if task=='segmentation' else 'boxes'}")

        batch_inputs["__orig_images__"] = [pil_image] * len(classes)
        render_batch_inference_outputs(
            "BATCH", batch_inputs, aux_out, [], task, out_dir=out_dir, queries=classes,
        )
        # --- merged labeled image + YOLO txt (all classes in one) ---
        flat: list[dict] = []
        try:
            from falcon_perception.visualization_utils import detections_from_batch_aux as _dets_from_batch
            for idx, (cls_name, aux) in enumerate(zip(classes, aux_out)):
                hw = (pil_image.size[1], pil_image.size[0])  # (H,W)
                pm = batch_inputs.get("pixel_mask")
                pm1 = pm[idx, 0] if isinstance(pm, torch.Tensor) and pm.ndim >= 3 else None
                for d in _dets_from_batch(aux, pixel_mask_1hw=pm1, orig_hw=hw, segmentation=(task == "segmentation")):
                    d2 = dict(d)
                    d2["label"] = cls_name
                    flat.append(d2)
            _save_merged_labeled_image(pil_image, flat, classes, out_path / "merged.jpg", task=task)
        except Exception as e:
            print(f"[merged] skip ({e})")
        try:
            _save_yolo_txt(flat, classes, out_path / "predictions_yolo.txt")
        except Exception as e:
            print(f"[yolo] skip ({e})")

        print(f"\n  Input image : {out_path / 'perception_input.jpg'}")
        print(f"  Output dir  : {out_path / 'masks'}")
        print(f"  Merged      : {out_path / 'merged.jpg'}")
        print(f"  YOLO txt    : {out_path / 'predictions_yolo.txt'}")


if __name__ == "__main__":
    tyro.cli(main)
