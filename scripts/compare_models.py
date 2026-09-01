#!/usr/bin/env python3
"""
Run one or more models (our trained YOLO detector + image-focused
VLMs/LLMs) over the same sample of held-out test images, and write their
predictions to runs/llm/<run_name>/ for ModelComparison.ipynb to score and
visualize.

    python scripts/compare_models.py
    python scripts/compare_models.py --n-images -1 --models yolo,gemma4
    python scripts/compare_models.py --n-images 100 --models yolo,claude --include-cloud

--n-images -1 means "every image in the test split," not a sample.

Cloud models only run when --include-cloud is passed too, even if named in
--models — a deliberate double gate so a typo or reused command can't
accidentally spend API credits.

Images are sampled from data/merged/test/images only (never train/val) and
this script never modifies data/merged/ or the YOLO checkpoint.

Every non-YOLO model here is a chat-style LLM/VLM (Ollama-served, or
Claude) doing presence/classification only — this is deliberately an
LLM-only lineup, not a mix of detectors, so every entrant is judged on the
same terms. YOLO is the one fixed, non-LLM baseline they're compared
against.

Setup (installed into vision-data-env for this tool; not added to the
top-level requirements.txt, which targets the separate CUDA training env):
    pip install anthropic requests
ollama/claude/gemini adapters need their own separate setup (Ollama
installed + running with the relevant model(s) pulled — `ollama pull
llava`, `ollama pull qwen3-vl:4b`, `ollama pull gemma4:e4b`, `ollama pull
minicpm-v:8b`; ANTHROPIC_API_KEY or `ant auth login` for Claude;
GEMINI_API_KEY for Gemini — get one at https://aistudio.google.com/apikey).

Output, per run — folder name encodes timestamp, dataset, n_images, seed,
and models so a run is identifiable without opening it:
  runs/llm/<run_name>/run_manifest.json   models used, checkpoint/model ids,
                                           prompt template, sampled images
  runs/llm/<run_name>/detections.csv      one row per predicted box
                                           (grounding models only produce
                                           real bboxes)
  runs/llm/<run_name>/presence.csv        one row per (image, model, class)
                                           queryable by that model: present
                                           True/False, or a parse_error flag
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_adapters import ADAPTERS, DEFAULT_PROMPT_TEMPLATE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_ROOT = REPO_ROOT / "data" / "merged"
DATASET_NAME = MERGED_ROOT.name  # "merged" — folds into the run name/manifest
LLM_RUNS_ROOT = REPO_ROOT / "runs" / "llm"

DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "runs" / "detect" / "yolo26s_merged_100e" / "weights" / "best.pt"
DEFAULT_N_IMAGES = 20
DEFAULT_MODELS = "yolo,ollama,qwen3-vl,gemma4,minicpm-v"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n-images",
        type=int,
        default=DEFAULT_N_IMAGES,
        help=f"How many test images to sample (default: {DEFAULT_N_IMAGES}); -1 = every test image",
    )
    parser.add_argument(
        "--models", default=DEFAULT_MODELS, help="Comma-separated adapter names to run: " + ",".join(ADAPTERS)
    )
    parser.add_argument(
        "--include-cloud",
        action="store_true",
        help="Required in addition to naming a cloud model in --models, or the run aborts",
    )
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--ollama-model", default=ADAPTERS["ollama"]["default_model"])
    parser.add_argument("--qwen3-vl-model", default=ADAPTERS["qwen3-vl"]["default_model"])
    parser.add_argument("--gemma4-model", default=ADAPTERS["gemma4"]["default_model"])
    parser.add_argument("--minicpm-v-model", default=ADAPTERS["minicpm-v"]["default_model"])
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--claude-model", default=ADAPTERS["claude"]["default_model"])
    parser.add_argument("--gemini-model", default=ADAPTERS["gemini"]["default_model"])
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Overrides the prompt sent to every chat-style model (ollama/qwen3-vl/gemma4/minicpm-v/claude). "
        "Must contain {class_list} and {json_shape} placeholders. See model_adapters.DEFAULT_PROMPT_TEMPLATE.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None, help="Default: auto-built from timestamp/dataset/n/seed/models")
    return parser.parse_args()


def build_run_name(n_images, seed, model_names):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_part = "all" if n_images == -1 else str(n_images)
    return f"{timestamp}_{DATASET_NAME}_n{n_part}_seed{seed}_{'-'.join(model_names)}"


def sample_test_images(n_images, seed):
    """Stratified sample from the test split: reserves an even per-class
    quota (rarest class first) before topping up randomly, so a small
    n_images doesn't accidentally miss a rare class like vest/no-vest
    entirely. n_images=-1 means every test image.

    Each class is capped at floor(n_images / n_classes) images during the
    reservation pass — an earlier version let the single rarest class fill
    the *entire* n_images budget before ever considering the next class,
    which silently starved any class confined to a different raw source
    (e.g. gloves/boots, which only anuragraj03 labels, never got sampled
    because a same-image-count-or-larger class from snehilsanyal-main
    always sorted rarer and ate the whole quota first)."""
    labels_long = pd.read_csv(MERGED_ROOT / "labels_long.csv")
    test_labels = labels_long[labels_long["split"] == "test"]
    all_test_files = sorted(test_labels["file"].unique())

    if n_images == -1 or n_images >= len(all_test_files):
        return all_test_files

    rng = random.Random(seed)
    selected = []
    seen = set()
    class_counts = test_labels["class_id"].value_counts()
    class_ids_rarest_first = class_counts.sort_values().index.tolist()
    per_class_quota = max(1, n_images // len(class_ids_rarest_first))

    for class_id in class_ids_rarest_first:
        if len(selected) >= n_images:
            break
        candidates = test_labels[test_labels["class_id"] == class_id]["file"].unique().tolist()
        rng.shuffle(candidates)
        taken = 0
        for f in candidates:
            if taken >= per_class_quota or len(selected) >= n_images:
                break
            if f not in seen:
                seen.add(f)
                selected.append(f)
                taken += 1

    remaining_pool = [f for f in all_test_files if f not in seen]
    rng.shuffle(remaining_pool)
    for f in remaining_pool:
        if len(selected) >= n_images:
            break
        seen.add(f)
        selected.append(f)

    return sorted(selected)


def image_path_for(file_stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = MERGED_ROOT / "test" / "images" / f"{file_stem}{ext}"
        if p.exists():
            return p
    return None


def build_adapter(model_name, args):
    if model_name not in ADAPTERS:
        raise SystemExit(f"Unknown model '{model_name}'. Available: {', '.join(ADAPTERS)}")
    spec = ADAPTERS[model_name]

    if model_name == "yolo":
        return spec["cls"](args.yolo_weights)
    if model_name == "ollama":
        return spec["cls"](args.ollama_model, args.ollama_url, prompt_template=args.prompt_template)
    if model_name == "qwen3-vl":
        return spec["cls"](args.qwen3_vl_model, args.ollama_url, prompt_template=args.prompt_template)
    if model_name == "gemma4":
        return spec["cls"](args.gemma4_model, args.ollama_url, prompt_template=args.prompt_template)
    if model_name == "minicpm-v":
        return spec["cls"](args.minicpm_v_model, args.ollama_url, prompt_template=args.prompt_template)
    if model_name == "claude":
        return spec["cls"](args.claude_model, prompt_template=args.prompt_template)
    if model_name == "gemini":
        return spec["cls"](args.gemini_model, prompt_template=args.prompt_template)
    raise SystemExit(f"No constructor wired up for '{model_name}'")


def main():
    args = parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    unknown = [m for m in model_names if m not in ADAPTERS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}. Available: {', '.join(ADAPTERS)}")

    cloud_requested = [m for m in model_names if ADAPTERS[m]["is_cloud"]]
    if cloud_requested and not args.include_cloud:
        raise SystemExit(
            f"{cloud_requested} require(s) network calls to a paid API. "
            "Pass --include-cloud to confirm you want to spend API credits on this run."
        )

    run_name = args.run_name or build_run_name(args.n_images, args.seed, model_names)
    run_dir = LLM_RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    sampled_files = sample_test_images(args.n_images, args.seed)
    print(f"Sampled {len(sampled_files)} test images (n_images={args.n_images}, seed={args.seed})")

    adapters = {}
    for name in model_names:
        print(f"Loading {name}...")
        t0 = time.perf_counter()
        adapters[name] = build_adapter(name, args)
        print(f"  ready in {time.perf_counter() - t0:.1f}s")

    detection_rows = []
    presence_rows = []
    parse_failures = {name: 0 for name in model_names}

    total = len(sampled_files) * len(model_names)
    done = 0
    t_start = time.perf_counter()
    detections_path = run_dir / "detections.csv"
    presence_path = run_dir / "presence.csv"
    interrupted = False

    try:
        for file_stem in sampled_files:
            image_path = image_path_for(file_stem)
            if image_path is None:
                print(f"warning: no image found for {file_stem}, skipping")
                continue

            for name, adapter in adapters.items():
                done += 1
                detections = adapter.predict(image_path)

                if detections is None:  # unparseable model output
                    parse_failures[name] += 1
                    for cls in adapter.queryable_classes:
                        presence_rows.append(
                            {"file": file_stem, "model": name, "class_name": cls, "present": None, "parse_error": True}
                        )
                    continue

                present_classes = {d.class_name for d in detections if d.present}
                for cls in adapter.queryable_classes:
                    presence_rows.append(
                        {
                            "file": file_stem,
                            "model": name,
                            "class_name": cls,
                            "present": cls in present_classes,
                            "parse_error": False,
                        }
                    )
                for d in detections:
                    if d.bbox is None and not d.present:
                        continue  # a chat-model "false" isn't a detection row
                    detection_rows.append(
                        {
                            "file": file_stem,
                            "model": name,
                            "class_name": d.class_name,
                            "confidence": d.confidence,
                            "x1": d.bbox[0] if d.bbox else None,
                            "y1": d.bbox[1] if d.bbox else None,
                            "x2": d.bbox[2] if d.bbox else None,
                            "y2": d.bbox[3] if d.bbox else None,
                        }
                    )

                if done % 20 == 0 or done == total:
                    elapsed = time.perf_counter() - t_start
                    print(f"  {done}/{total} (image, model) pairs done, {elapsed:.0f}s elapsed")
                    # checkpoint: survive a kill, not just a clean Ctrl+C
                    pd.DataFrame(detection_rows).to_csv(detections_path, index=False)
                    pd.DataFrame(presence_rows).to_csv(presence_path, index=False)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted at {done}/{total} pairs — writing partial results and scoring what we have")

    status = "PARTIAL" if (interrupted or done < total) else "COMPLETE"
    new_run_name = f"{run_name}_{status}"
    new_run_dir = LLM_RUNS_ROOT / new_run_name
    run_dir.rename(new_run_dir)
    run_dir, run_name = new_run_dir, new_run_name
    detections_path = run_dir / "detections.csv"
    presence_path = run_dir / "presence.csv"

    pd.DataFrame(detection_rows).to_csv(detections_path, index=False)
    pd.DataFrame(presence_rows).to_csv(presence_path, index=False)

    manifest = {
        "run_name": run_name,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": DATASET_NAME,
        "n_images_requested": args.n_images,
        "n_images_sampled": len(sampled_files),
        "seed": args.seed,
        "models": model_names,
        "queryable_classes": {name: adapters[name].queryable_classes for name in model_names},
        "supports_grounding": {name: adapters[name].supports_grounding for name in model_names},
        "prompt_template": args.prompt_template,
        "config": {
            "yolo_weights": args.yolo_weights,
            "ollama_model": args.ollama_model,
            "qwen3_vl_model": args.qwen3_vl_model,
            "gemma4_model": args.gemma4_model,
            "minicpm_v_model": args.minicpm_v_model,
            "claude_model": args.claude_model,
            "gemini_model": args.gemini_model,
        },
        "parse_failures": parse_failures,
        "sampled_files": sampled_files,
    }
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone in {time.perf_counter() - t_start:.0f}s. Wrote:")
    print(f"  {manifest_path}")
    print(f"  {detections_path} ({len(detection_rows)} rows)")
    print(f"  {presence_path} ({len(presence_rows)} rows)")
    if any(parse_failures.values()):
        print(f"  parse failures (excluded from presence scoring): {parse_failures}")


if __name__ == "__main__":
    main()
