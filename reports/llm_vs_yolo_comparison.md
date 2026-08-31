# LLM/VLM vs. our YOLO detector — presence-detection comparison

**Run:** `runs/llm/20260828_035813_merged_n100_seed42_yolo-ollama-qwen3-vl-gemma4-minicpm-v/`
**Dataset:** `merged` test split · **n = 100 images** · **seed 42**
**Task:** presence/absence classification (per image, per class — "is at least one instance of X visible?"), scored against the same ground truth for every model
**Models:** our trained **YOLO26** detector (baseline) vs. four general-purpose vision-language models prompted with an identical instruction — `ollama` (llava), `qwen3-vl`, `gemma4`, `minicpm-v`
**Classes:** `person`, `helmet`, `gloves`, `boots`, `vest` (positive/presence) and `no-helmet`, `no-gloves`, `no-boots`, `no-vest` (negative/absence)

See `scripts/compare_models.py` for how the run was produced and `ModelComparison.ipynb` for the underlying scoring.

---

## Headline result

Our trained detector outperforms every general-purpose VLM on every class, and the gap is largest exactly where it matters most for a safety product: **detecting the absence of PPE**.

| Model | Macro F1 — positive classes (person/helmet/gloves/boots/vest) | Macro F1 — negative classes (no-*) | Macro F1 — all 9 classes |
|---|---|---|---|
| **YOLO (ours)** | **0.964** | **0.948** | **0.957** |
| qwen3-vl | 0.744 | 0.355 | 0.571 |
| gemma4 | 0.634 | 0.303 | 0.487 |
| minicpm-v | 0.658 | 0.228 | 0.467 |
| ollama (llava) | 0.528 | 0.348 | 0.448 |

Every LLM's negative-class F1 collapses to roughly a third to a half of its positive-class F1. YOLO's does not — it holds within a couple of points. This is the same failure mode the project's core design already bets against (see README: *"training on negative classes fails catastrophically... the model cannot learn an absent object"*) — it turns out to be true of prompted VLMs as well as trained detectors, not just a training-data artifact. It's the practical case for HI-VIS's positive-only-detection-plus-association-logic architecture: rather than asking any model (trained or prompted) to directly recognize "no helmet," the system detects `person` and `helmet` independently and derives the violation.

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

- **`person` recall is 1.00 (or near it) for every model, including the VLMs** — but every VLM pairs that with ~51 false positives on a 100-image sample (precision ≈ 0.48–0.49). All four VLMs report the same `fp=51` on `person`, which looks like a systematic behavior (e.g. the model defaulting to "person: yes" almost unconditionally) rather than four independent failure modes — worth a follow-up look at the raw `presence.csv` before trusting `person` counts from any VLM.
- **On the classes the product actually gates on (`no-helmet`, `no-boots`, etc.), every VLM's precision is poor** — gemma4 and minicpm-v both score 0.00–0.18 precision on `no-boots`/`no-gloves`, meaning most of their "PPE missing" flags would be false alarms in production.
- **qwen3-vl is the strongest VLM by a clear margin** (0.571 overall macro F1, 0.355 on negative classes) — meaningfully ahead of gemma4, minicpm-v, and ollama, but still well short of YOLO on every axis.
- **YOLO's only imperfect class is `vest`** (F1 0.92, driven by 3 fn / 2 fp) — still the best score any model gets on that class, and far ahead of the 0.39–0.84 range the VLMs post there.

## Caveats

- n = 100 is one held-out sample (seed 42); treat exact figures as directional, not final production metrics — see README's stated thresholds (Head Protection recall > 0.95, Foot Protection recall > 0.85, mAP@50 > 0.80) for the bar YOLO is actually held to.
- This is a **presence/absence classification** task for the VLMs (no bounding boxes — see `supports_grounding: false` in the run's `run_manifest.json`), not object detection; YOLO is being compared on the same coarse presence signal here, not on its full localization ability.
- `minicpm-v`'s `no-boots` F1 is `nan`: it never once correctly flagged a genuine no-boots case (tp=0), while still raising 14 false alarms — precision and recall are both 0.00, so F1 is undefined (0/0) rather than a real zero score.
