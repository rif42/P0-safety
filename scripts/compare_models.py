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


def run_comparison_steps(adapters, sampled_files, image_path_for=image_path_for, descriptive_prompt=None, skip_pairs=None):
    """Drives every (file, model) prediction call one at a time, yielding a
    step dict after each — so a caller can checkpoint, render, or bail
    between calls instead of only seeing anything once the whole loop
    finishes. Shared by main() below and streamlit_app/pages/live_compare.py,
    so the actual inference-and-row-building logic lives in exactly one
    place.

    `descriptive_prompt`, if given, also calls .describe() on any adapter
    that has one (claude, gemini) — same free-text/unscored capture
    ModelComparison.ipynb does; main() below doesn't pass one, so CLI runs
    are unaffected.

    `skip_pairs`, if given, is a set of (file, model) tuples to skip
    outright — no predict()/describe() call, no rows, just a `resumed=True`
    step carrying the counters forward. This is what makes resuming a
    paused Live Comparison run actually save the API calls it's resuming
    past, not just re-do them silently."""
    skip_pairs = skip_pairs or set()
    total = len(sampled_files) * len(adapters)
    done = 0
    done_per_model = {name: 0 for name in adapters}

    for file_stem in sampled_files:
        image_path = image_path_for(file_stem)
        if image_path is None:
            yield {
                "file": file_stem, "model": None, "done": done, "total": total,
                "done_per_model": dict(done_per_model), "skipped": True, "resumed": False, "parse_error": False,
                "presence_rows": [], "detection_rows": [], "descriptive_row": None,
            }
            continue

        for name, adapter in adapters.items():
            done += 1
            done_per_model[name] += 1

            if (file_stem, name) in skip_pairs:
                yield {
                    "file": file_stem, "model": name, "done": done, "total": total,
                    "done_per_model": dict(done_per_model), "skipped": False, "resumed": True, "parse_error": False,
                    "presence_rows": [], "detection_rows": [], "descriptive_row": None,
                }
                continue

            detections = adapter.predict(image_path)

            presence_rows = []
            detection_rows = []
            parse_error = detections is None
            if parse_error:  # unparseable model output
                for cls in adapter.queryable_classes:
                    presence_rows.append(
                        {"file": file_stem, "model": name, "class_name": cls, "present": None, "parse_error": True}
                    )
            else:
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

            descriptive_row = None
            if descriptive_prompt and hasattr(adapter, "describe"):
                text = adapter.describe(image_path, descriptive_prompt)
                descriptive_row = {"file": file_stem, "model": name, "response_text": text.strip()}

            yield {
                "file": file_stem, "model": name, "done": done, "total": total,
                "done_per_model": dict(done_per_model), "skipped": False, "resumed": False, "parse_error": parse_error,
                "presence_rows": presence_rows, "detection_rows": detection_rows, "descriptive_row": descriptive_row,
            }


def run_checklist_steps(adapters, sampled_files, image_path_for=image_path_for, items=None, skip_pairs=None):
    """Sibling to run_comparison_steps() for the per-person checklist prompt
    ("how many people, and for each: helmet/vest/gloves/boots?") — a
    different question shape (a list of people, not flat per-class
    booleans), so it's its own generator rather than a mode flag bolted
    onto run_comparison_steps(). Chat-style adapters answer via describe();
    a grounding adapter (YOLO) has no describe(), so it answers via its own
    predict() boxes instead, converted to the same shape by
    people_from_detections() — same loop, same output shape, either way.

    Each step's `people` is the list scripts/model_adapters.py's
    parse_person_checklist_json() (or people_from_detections()) returns, or
    None on a parse/detection failure — never a fabricated empty list, so
    "0 people" and "couldn't read the answer" stay distinguishable
    downstream."""
    from model_adapters import (
        CHECKLIST_ITEMS,
        parse_person_checklist_json,
        people_from_detections,
        render_checklist_prompt,
    )

    items = items or CHECKLIST_ITEMS
    prompt = render_checklist_prompt(items)
    skip_pairs = skip_pairs or set()
    total = len(sampled_files) * len(adapters)
    done = 0
    done_per_model = {name: 0 for name in adapters}

    for file_stem in sampled_files:
        image_path = image_path_for(file_stem)
        if image_path is None:
            yield {
                "file": file_stem, "model": None, "done": done, "total": total,
                "done_per_model": dict(done_per_model), "skipped": True, "resumed": False,
                "people": None, "raw_text": None,
            }
            continue

        for name, adapter in adapters.items():
            done += 1
            done_per_model[name] += 1

            if (file_stem, name) in skip_pairs:
                yield {
                    "file": file_stem, "model": name, "done": done, "total": total,
                    "done_per_model": dict(done_per_model), "skipped": False, "resumed": True,
                    "people": None, "raw_text": None,
                }
                continue

            raw_text = None
            if hasattr(adapter, "describe"):
                raw_text = adapter.describe(image_path, prompt)
                people = parse_person_checklist_json(raw_text, items)
            else:  # grounding model (YOLO) — no free-text interface, use its own boxes
                detections = adapter.predict(image_path)
                people = people_from_detections(detections, items) if detections is not None else None
            yield {
                "file": file_stem, "model": name, "done": done, "total": total,
                "done_per_model": dict(done_per_model), "skipped": False, "resumed": False,
                "people": people, "raw_text": raw_text,
            }


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
        for step in run_comparison_steps(adapters, sampled_files, image_path_for=image_path_for):
            if step["skipped"]:
                print(f"warning: no image found for {step['file']}, skipping")
                continue

            done, total = step["done"], step["total"]
            presence_rows.extend(step["presence_rows"])
            detection_rows.extend(step["detection_rows"])
            if step["parse_error"]:
                parse_failures[step["model"]] += 1

            if done % 20 == 0 or done == total:
                elapsed = time.perf_counter() - t_start
                per_model = " ".join(f"{n}:{c}/{len(sampled_files)}" for n, c in step["done_per_model"].items())
                print(f"  {done}/{total} pairs, {elapsed:.0f}s elapsed — {per_model}")
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
