"""Bulk 'all objects' detection over demo/assets/demo-pics (CPU-friendly).

One model load, batched by --batch-size images (BatchInferenceEngine, no CUDA needed).

Usage:
  python demo/perception_all_objects.py --limit 2        # smoke test
  python demo/perception_all_objects.py                  # all 51 images, detection
  python demo/perception_all_objects.py --task segmentation
  python demo/perception_all_objects.py --batch-size 4 --out-dir outputs/all_objects
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import torch
import tyro
from PIL import Image

from falcon_perception import build_prompt_for_task, load_and_prepare_model, setup_torch_config
from falcon_perception.visualization_utils import pair_bbox_entries

setup_torch_config()

SRC_GLOBS = ["demo/assets/demo-pics/challenging/*", "demo/assets/demo-pics/typical/*"]


def collect_images() -> list[Path]:
    paths: list[Path] = []
    for pat in SRC_GLOBS:
        paths.extend(Path(p) for p in glob.glob(pat))
    return sorted([p for p in paths if p.is_file()], key=lambda p: str(p))


@torch.inference_mode()
def main(
    query: str = "all objects",
    task: str = "detection",
    batch_size: int = 2,
    limit: int = 0,
    out_dir: str = "outputs/all_objects",
    max_dimension: int = 1024,
    min_dimension: int = 256,
):
    """Run `query` on every image in demo/assets/demo-pics.

    Args:
        query: open-vocabulary query; 'all objects' = exhaustive detection
        task: 'detection' (boxes) or 'segmentation' (boxes+masks)
        batch_size: images per forward pass
        limit: 0=all, else first N only (smoke test)
        out_dir: output root (json/ + overlays/)
    """
    images = collect_images()
    if limit and limit > 0:
        images = images[:limit]
    print(f"Found {len(images)} images")
    for p in images:
        print(f"  {p}")

    if not images:
        print("No images found.")
        return

    print(f"\nLoading model (cpu, task={task}, query={query!r}) ...")
    from falcon_perception import PERCEPTION_MODEL_ID
    model, tokenizer, model_args = load_and_prepare_model(hf_model_id=PERCEPTION_MODEL_ID, device="cpu", dtype="float32", compile=False)
    if task == "segmentation" and not model_args.do_segmentation:
        print("Model do_segmentation=False, falling back to detection.")
        task = "detection"

    from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate

    engine = BatchInferenceEngine(model, tokenizer)
    prompt = build_prompt_for_task(query, task)
    stop_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    per_image: list[dict] = []

    for start in range(0, len(images), batch_size):
        chunk_paths = images[start : start + batch_size]
        pils: list[Image.Image | None] = []
        for p in chunk_paths:
            try:
                pils.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[skip] {p}: {e}")
                pils.append(None)

        valid = [(img, prompt) for img in pils if img is not None]
        valid_paths = [p for p, img in zip(chunk_paths, pils) if img is not None]
        valid_pils = [img for img in pils if img is not None]
        if not valid:
            continue

        print(f"\n[{start+1:>3}/{len(images)}] batch {len(valid)}: {[p.name for p in valid_paths]}")
        batch_inputs = process_batch_and_generate(
            tokenizer, valid, max_length=4096, min_dimension=min_dimension, max_dimension=max_dimension
        )
        batch_inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch_inputs.items()}

        _, aux_out = engine.generate(
            **batch_inputs, max_new_tokens=2048, temperature=0.0, stop_token_ids=stop_ids, seed=42, task=task
        )

        for p, pil_img, aux in zip(valid_paths, valid_pils, aux_out):
            bboxes = pair_bbox_entries(aux.bboxes_raw)
            n = len(bboxes)
            total_boxes += n
            print(f"  {p.name}: {n} boxes")

            rel = p.relative_to(Path("demo/assets/demo-pics"))
            jdir = out_root / "json" / rel.parent
            jdir.mkdir(parents=True, exist_ok=True)
            with open(jdir / (p.stem + ".json"), "w", encoding="utf-8") as f:
                json.dump({"image": str(p), "query": query, "task": task, "boxes": bboxes, "masks_rle": aux.masks_rle if task == "segmentation" else []}, f, indent=2)

            # overlay: build detection dicts for overlay helper
            try:
                import numpy as np
                from falcon_perception.visualization_utils import decode_coco_rle, overlay_detections_on_image_v2

                dets = []
                for i, b in enumerate(bboxes):
                    d = {"xy": {"x": b["x"], "y": b["y"]}, "hw": {"w": b["w"], "h": b["h"]}}
                    if task == "segmentation" and i < len(aux.masks_rle):
                        m = decode_coco_rle(aux.masks_rle[i])
                        if m is not None:
                            d["mask"] = m
                    dets.append(d)
                overlay = overlay_detections_on_image_v2(np.array(pil_img), dets, draw_bbox=True, masks_are_binary=False)
                vdir = out_root / "overlays" / rel.parent
                vdir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(overlay).save(vdir / (p.stem + ".jpg"))
            except Exception as e:
                print(f"    [vis warn] {p.name}: {e}")

            per_image.append({"image": str(p), "boxes": n})

    print(f"\n{'='*60}")
    print(f"Done: {len(per_image)} images, {total_boxes} total boxes")
    print(f"JSON: {out_root/'json'}  Overlays: {out_root/'overlays'}")
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"query": query, "task": task, "images": len(per_image), "total_boxes": total_boxes, "per_image": per_image}, f, indent=2)
    print(f"Summary: {out_root/'summary.json'}")


if __name__ == "__main__":
    tyro.cli(main)
