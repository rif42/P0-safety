#!/usr/bin/env python3
"""
Run two different Gemini prompts over the same small sample of test images
and save the raw text side by side — a qualitative comparison, not a scored
one, since the "descriptive" prompt below produces free-text prose with no
per-class boolean to check against ground truth (unlike the structured JSON
prompt scripts/compare_models.py scores everyone else on).

    GEMINI_API_KEY=... python scripts/gemini_prompt_comparison.py
    GEMINI_API_KEY=... python scripts/gemini_prompt_comparison.py --n-images 20

Output: runs/llm/<run_name>/prompt_comparison.csv, one row per
(image, prompt_name) with the model's raw response text. See
ModelComparison.ipynb section 3 for a rendered side-by-side view.
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import DATASET_NAME, LLM_RUNS_ROOT, image_path_for, sample_test_images  # noqa: E402
from model_adapters import ADAPTERS, DEFAULT_PROMPT_TEMPLATE, render_prompt  # noqa: E402

DESCRIPTIVE_PROMPT = (
    "Describe this construction site photograph in one or two plain sentences "
    "for a site record: what the scene is, roughly how many people are visible "
    "and what they appear to be doing, and the setting. Describe only what is "
    "visible. Assess safety, compliance or risk. No preamble, no bullet "
    "points."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=12, help="Small on purpose — this is eyeballed, not scored")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemini-model", default=ADAPTERS["gemini"]["default_model"])
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = ADAPTERS["gemini"]["cls"](model=args.gemini_model)

    structured_prompt = render_prompt(DEFAULT_PROMPT_TEMPLATE, adapter.queryable_classes)
    prompts = {
        "1_new_descriptive": DESCRIPTIVE_PROMPT,
        "2_existing_structured": structured_prompt,
    }

    sampled_files = sample_test_images(args.n_images, args.seed)
    print(f"Sampled {len(sampled_files)} test images (n_images={args.n_images}, seed={args.seed})")

    rows = []
    total = len(sampled_files) * len(prompts)
    done = 0
    t_start = time.perf_counter()
    for file_stem in sampled_files:
        image_path = image_path_for(file_stem)
        if image_path is None:
            print(f"warning: no image found for {file_stem}, skipping")
            continue
        for prompt_name, prompt_text in prompts.items():
            done += 1
            text = adapter.describe(image_path, prompt_text)
            rows.append({"file": file_stem, "prompt_name": prompt_name, "response_text": text.strip()})
            print(f"  [{done}/{total}] {file_stem} / {prompt_name}: {text.strip()[:80]!r}")

    run_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{DATASET_NAME}_gemini_prompt_comparison_n{len(sampled_files)}_seed{args.seed}"
    run_dir = LLM_RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "prompt_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nDone in {time.perf_counter() - t_start:.0f}s. Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
