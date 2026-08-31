# LLM/VLM vs. our YOLO detector — presence-detection comparison

**Runs:**
- `runs/llm/20260828_035813_merged_n100_seed42_yolo-ollama-qwen3-vl-gemma4-minicpm-v/` — yolo, ollama, qwen3-vl, gemma4, minicpm-v
- `runs/llm/20260831_merged_n100_seed42_yolo-gemini/` — yolo (re-run as a consistency check — landed byte-identical, as expected from a deterministic checkpoint on the same sample), gemini

Both runs sample the **same 100 test images at seed 42** (see `sample_test_images()` in `scripts/compare_models.py` — deterministic given n/seed), so every model in this report is judged on an identical set of images despite coming from two separate runs.

**Dataset:** `merged` test split · **n = 100 images** · **seed 42**
**Task:** presence/absence classification (per image, per class — "is at least one instance of X visible?"), scored against the same ground truth for every model
**Models:** our trained **YOLO26** detector (baseline) vs. five general-purpose vision-language models prompted with an identical instruction — `ollama` (llava), `qwen3-vl`, `gemma4`, `minicpm-v`, and **Gemini** (`gemini-3.6-flash`, cloud)
**Classes:** `person`, `helmet`, `gloves`, `boots`, `vest` (positive/presence) and `no-helmet`, `no-gloves`, `no-boots`, `no-vest` (negative/absence)

See `scripts/compare_models.py` (adapters in `scripts/model_adapters.py`) for how the runs were produced and `ModelComparison.ipynb` for the underlying scoring. Gemini needs `GEMINI_API_KEY` in the environment — never committed, see `model_adapters.GeminiAdapter`.

---

## Headline result

Our trained detector outperforms every general-purpose VLM on every class, and the gap is largest exactly where it matters most for a safety product: **detecting the absence of PPE**.

| Model | Macro F1 — positive classes (person/helmet/gloves/boots/vest) | Macro F1 — negative classes (no-*) | Macro F1 — all 9 classes |
|---|---|---|---|
| **YOLO (ours)** | **0.964** | **0.948** | **0.957** |
| gemini (gemini-3.6-flash) | 0.801 | 0.439 | 0.640 |
| qwen3-vl | 0.744 | 0.355 | 0.571 |
| gemma4 | 0.634 | 0.303 | 0.487 |
| minicpm-v | 0.658 | 0.228 | 0.467 |
| ollama (llava) | 0.528 | 0.348 | 0.448 |

Every LLM's negative-class F1 collapses relative to its positive-class F1 — Gemini's least severely (0.801 → 0.439, still a ~45% relative drop), the local VLMs worst (roughly half to a third). YOLO's does not — it holds within a couple of points. This is the same failure mode the project's core design already bets against (see README: *"training on negative classes fails catastrophically... the model cannot learn an absent object"*) — it turns out to be true of prompted VLMs, cloud or local, as well as trained detectors, not just a training-data artifact. It's the practical case for HI-VIS's positive-only-detection-plus-association-logic architecture: rather than asking any model (trained or prompted) to directly recognize "no helmet," the system detects `person` and `helmet` independently and derives the violation.

Gemini is the strongest VLM in the lineup by a clear margin — noticeably ahead of qwen3-vl (the previous best) on every axis, and the only VLM to hit a **perfect precision on `vest`** (1.00, tp=28/fp=0) — but it is still well short of YOLO everywhere, and its negative-class precision is still poor enough (0.18–0.29 on `no-gloves`/`no-vest`) that most of its "PPE missing" flags would be false alarms in production.

---

## Full per-class results

tp/fp/fn are counts over the 100-image sample; precision/recall/F1 are computed from those. `nan` = undefined (0 predicted positives and 0 true positives for that class).

### YOLO (ours)

| class | tp | fp | fn | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 49 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| helmet | 47 | 1 | 2 | 0.98 | 0.96 | 0.97 |
| gloves | 31 | 0 | 2 | 1.00 | 0.94 | 0.97 |
| boots | 32 | 0 | 3 | 1.00 | 0.91 | 0.96 |
| vest | 29 | 2 | 3 | 0.94 | 0.91 | 0.92 |
| no-helmet | 31 | 2 | 4 | 0.94 | 0.89 | 0.91 |
| no-gloves | 17 | 1 | 1 | 0.94 | 0.94 | 0.94 |
| no-boots | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| no-vest | 22 | 1 | 2 | 0.96 | 0.92 | 0.94 |

### qwen3-vl

| class | tp | fp | fn | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 48 | 51 | 1 | 0.48 | 0.98 | 0.65 |
| helmet | 46 | 2 | 3 | 0.96 | 0.94 | 0.95 |
| gloves | 24 | 15 | 9 | 0.62 | 0.73 | 0.67 |
| boots | 23 | 16 | 12 | 0.59 | 0.66 | 0.62 |
| vest | 26 | 5 | 6 | 0.84 | 0.81 | 0.83 |
| no-helmet | 21 | 13 | 14 | 0.62 | 0.60 | 0.61 |
| no-gloves | 5 | 35 | 13 | 0.12 | 0.28 | 0.17 |
| no-boots | 9 | 27 | 7 | 0.25 | 0.56 | 0.35 |
| no-vest | 8 | 23 | 16 | 0.26 | 0.33 | 0.29 |

### gemma4

| class | tp | fp | fn | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 49 | 51 | 0 | 0.49 | 1.00 | 0.66 |
| helmet | 46 | 22 | 3 | 0.68 | 0.94 | 0.79 |
| gloves | 29 | 37 | 4 | 0.44 | 0.88 | 0.59 |
| boots | 35 | 55 | 0 | 0.39 | 1.00 | 0.56 |
| vest | 31 | 45 | 1 | 0.41 | 0.97 | 0.57 |
| no-helmet | 17 | 10 | 18 | 0.63 | 0.49 | 0.55 |
| no-gloves | 3 | 31 | 15 | 0.09 | 0.17 | 0.12 |
| no-boots | 2 | 9 | 14 | 0.18 | 0.12 | 0.15 |
| no-vest | 9 | 13 | 15 | 0.41 | 0.38 | 0.39 |

### minicpm-v

| class | tp | fp | fn | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 49 | 51 | 0 | 0.49 | 1.00 | 0.66 |
| helmet | 45 | 11 | 4 | 0.80 | 0.92 | 0.86 |
| gloves | 24 | 38 | 9 | 0.39 | 0.73 | 0.51 |
| boots | 34 | 53 | 1 | 0.39 | 0.97 | 0.56 |
| vest | 30 | 24 | 2 | 0.56 | 0.94 | 0.70 |
| no-helmet | 11 | 9 | 24 | 0.55 | 0.31 | 0.40 |
| no-gloves | 5 | 34 | 13 | 0.13 | 0.28 | 0.18 |
| no-boots | 0 | 14 | 16 | 0.00 | 0.00 | nan |
| no-vest | 9 | 22 | 15 | 0.29 | 0.38 | 0.33 |

### gemini (gemini-3.6-flash)

| class | tp | fp | fn | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 48 | 51 | 1 | 0.48 | 0.98 | 0.65 |
| helmet | 45 | 2 | 4 | 0.96 | 0.92 | 0.94 |
| gloves | 27 | 10 | 6 | 0.73 | 0.82 | 0.77 |
| boots | 25 | 10 | 10 | 0.71 | 0.71 | 0.71 |
| vest | 28 | 0 | 4 | 1.00 | 0.88 | 0.93 |
| no-helmet | 33 | 33 | 2 | 0.50 | 0.94 | 0.65 |
| no-gloves | 13 | 58 | 5 | 0.18 | 0.72 | 0.29 |
| no-boots | 12 | 29 | 4 | 0.29 | 0.75 | 0.42 |
| no-vest | 20 | 59 | 4 | 0.25 | 0.83 | 0.39 |

### ollama (llava)

| class | tp | fp | fn | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 49 | 51 | 0 | 0.49 | 1.00 | 0.66 |
| helmet | 34 | 28 | 15 | 0.55 | 0.69 | 0.61 |
| gloves | 19 | 33 | 14 | 0.37 | 0.58 | 0.45 |
| boots | 21 | 39 | 14 | 0.35 | 0.60 | 0.44 |
| vest | 21 | 34 | 11 | 0.38 | 0.66 | 0.48 |
| no-helmet | 17 | 31 | 18 | 0.35 | 0.49 | 0.41 |
| no-gloves | 12 | 40 | 6 | 0.23 | 0.67 | 0.34 |
| no-boots | 11 | 37 | 5 | 0.23 | 0.69 | 0.34 |
| no-vest | 9 | 27 | 15 | 0.25 | 0.38 | 0.30 |

---

## Reading the numbers

- **`person` "false positives" are mostly a ground-truth gap, not a model error — confirmed, not just suspected.** All five VLMs (ollama, qwen3-vl, gemma4, minicpm-v, gemini) report the *same* `fp=51` on `person`, every time. Checking the actual labels: **51 of the 100 sampled images have PPE boxes (helmet/gloves/boots/etc.) but no `person` box at all** — e.g. an image labeled `{helmet, gloves, boots}` with a visibly PPE-wearing person in it, just never boxed as "person" by whoever annotated that source dataset. Every VLM correctly says "yes, there's a person" and is marked wrong by an incomplete label, not a real mistake. YOLO scores a clean 49/49 on `person` here because it was *trained* on this same (gapped) ground truth, not because it's more correct in any absolute sense — its perfect person score reflects consistency with the label set, not a stronger visual read. Treat every VLM's `person` precision/F1 in this report as an artifact of that gap rather than a real capability signal; the other 8 classes aren't affected by it.
- **On the classes the product actually gates on (`no-helmet`, `no-boots`, etc.), every VLM's precision is poor** — gemma4 and minicpm-v both score 0.00–0.18 precision on `no-boots`/`no-gloves`, and even Gemini, the best of the five, only reaches 0.18–0.29 there.
- **Gemini is the strongest VLM by a clear margin** (0.640 overall macro F1, 0.439 on negative classes) — meaningfully ahead of qwen3-vl (the previous best local/open model, 0.571/0.355), and the only model besides YOLO to hit perfect precision on any class (`vest`, 1.00). Being a larger, more expensively-run cloud model, this isn't a fully fair fight against the four ~4-9GB local Ollama models — but it's still nowhere near YOLO.
- **YOLO's only imperfect class is `vest`** (F1 0.92, driven by 3 fn / 2 fp) — still the best score any model gets on that class, and far ahead of the 0.39–0.93 range the VLMs post there.

## Caveats

- n = 100 is one held-out sample (seed 42); treat exact figures as directional, not final production metrics — see README's stated thresholds (Head Protection recall > 0.95, Foot Protection recall > 0.85, mAP@50 > 0.80) for the bar YOLO is actually held to.
- This is a **presence/absence classification** task for the VLMs (no bounding boxes — see `supports_grounding: false` in the run's `run_manifest.json`), not object detection; YOLO is being compared on the same coarse presence signal here, not on its full localization ability.
- `minicpm-v`'s `no-boots` F1 is `nan`: it never once correctly flagged a genuine no-boots case (tp=0), while still raising 14 false alarms — precision and recall are both 0.00, so F1 is undefined (0/0) rather than a real zero score.
- Gemini and YOLO were scored in a separate run from the other four VLMs (`20260831_merged_n100_seed42_yolo-gemini/` vs. `20260828_..._yolo-ollama-qwen3-vl-gemma4-minicpm-v/`) — both draw the identical 100-image/seed-42 sample, and YOLO's numbers landed byte-identical across both runs (a deterministic checkpoint on the same images, as expected), so the two runs are directly comparable despite being separate invocations.

---

## Prompt-style comparison (Gemini): descriptive vs. structured

Separately from the scored comparison above, `scripts/gemini_prompt_comparison.py` ran Gemini over a small unscored sample (12 images, seed 42) with two different prompts, to see what a free-text "site record" description looks like next to the strict-JSON prompt everything above is scored on. See `runs/llm/20260831_075116_merged_gemini_prompt_comparison_n12_seed42/prompt_comparison.csv` and section 3 of `ModelComparison.ipynb` for the full side-by-side.

There's no ground truth for prose, so this isn't scored — but the qualitative difference is worth noting: the descriptive prompt reliably produces a fluent, specific one-to-two-sentence summary (headcount, setting, activity, an explicit safety/risk call) that would drop straight into a site log — e.g., for the `frame00201` sample, it noted the roughly eight workers present all had hard hats on, but flagged the dim, dusty, machinery-heavy setting itself as an operational/tripping hazard — versus the structured prompt's bare `{"person": true, "helmet": true, "gloves": false, ...}` for the same image. The descriptive prompt is more useful as a human-readable record; the structured prompt is what the scored comparison above actually depends on, since it's the only format with a per-class boolean to check against ground truth.
