# Merged dataset — source provenance

`dataset/` is built from `data/raw/` by `scripts/build_dataset.py`. Every file
is copied as `<source>__<original_name>`, so `dataset/merge_manifest.csv` can
always trace a merged file back to its origin.

By default the build only keeps the 9 core classes (0-8, see below) and
**excludes any image with none of them present** — not just its label file,
the image itself isn't copied. Pass `--classes` to include a different set
(e.g. `--classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18` for
everything). Re-running the script fully rebuilds `dataset/images`,
`dataset/labels`, and the manifest from `data/raw/` each time.

Class IDs inside each source's label `.txt` files are **not remapped** by the
merge script — they're copied verbatim. Do not compare class indices across
sources without checking this table first.

## anuragraj03 — 8 classes

Source: `data/raw/anuragraj03/dataset.yaml` (shipped with the dataset).

| ID | Class | Total instances |
|---|---|---|
| 0 | no-safety-glove | 6,498 |
| 1 | no-safety-helmet | 5,438 |
| 2 | no-safety-shoes | 5,408 |
| 3 | no-welding-glass | 4,094 |
| 4 | safety-glove | 7,472 |
| 5 | safety-helmet | 9,097 |
| 6 | safety-shoes | 9,342 |
| 7 | welding-glass | 4,383 |

Includes negative classes (`no-*`). The implementation plan calls for dropping
these before the final positive-only merge (negative-class training performs
catastrophically per the plan's own findings).

## snehilsanyal-main (css-data) — 10 classes

Source: recovered from `experiments/legacy-snehilsanyal-yolov8n_100e/kaggle/working/ppe_data.yaml`,
a `data.yaml` left over from a prior Kaggle training run on this same dataset
(paths inside it point at `/kaggle/input/construction-site-safety-image-dataset-roboflow/...`,
confirming it's the source for `css-data`). Not present anywhere in the
`css-data/` folder itself — only found because the legacy training run was
kept instead of deleted.

| ID | Class | Train instances |
|---|---|---|
| 0 | Hardhat | 2,958 |
| 1 | Mask | 1,359 |
| 2 | NO-Hardhat | 2,231 |
| 3 | NO-Mask | 2,946 |
| 4 | NO-Safety Vest | 3,785 |
| 5 | Person | 9,269 |
| 6 | Safety Cone | 3,198 |
| 7 | Safety Vest | 2,890 |
| 8 | machinery | 4,208 |
| 9 | vehicle | 1,453 |

Validated against the label data: `Person` (id 5) has by far the highest
instance count, consistent with every person in every image getting one box
while PPE items are per-item — strong confirmation the ID order is correct.

This is the Roboflow "Construction Site Safety v28 YOLOv5s" export (2,801 base
images + augmented versions). See `css-data/README.roboflow.txt`.

## ketakichalke-boots — 11 classes

No `data.yaml`/`classes.txt` ships with this source or appears anywhere else
in the repo — this mapping was provided directly (not recovered from a file),
confirmed against the dataset's own documentation.

Positive classes:

| ID | Class | Count |
|---|---|---|
| 0 | Helmet | 1,750 |
| 1 | Gloves | 1,461 |
| 2 | Vest | 1,632 |
| 3 | Boots | 1,613 |
| 4 | Goggles | 526 |
| 6 | Person | 2,265 |

Negative classes (missing equipment):

| ID | Class | Count |
|---|---|---|
| 7 | no_helmet | 485 |
| 8 | no_goggle | 411 |
| 9 | no_gloves | 556 |
| 10 | no_boots | 115 |

Additional:

| ID | Class | Count |
|---|---|---|
| 5 | none (areas with no relevant objects) | 800 |

## Merged class schema

`dataset/class_mapping.py` defines a single 19-class schema all three sources
are remapped into (labels only — bounding boxes are untouched, only the
class-id column is rewritten). Core classes (0–8: person, then
helmet/gloves/boots/vest paired with their negatives) come first, then
remaining classes are grouped as [matching positives][matching
negatives][unmatched singles], ending with the background marker `none`.
Instance counts below are from the full 19-class mapping, before the default
classes-0-8/image-dropping filter `build_dataset.py` applies:

| ID | Class | Instances | ID | Class | Instances |
|---|---|---|---|---|---|
| 0 | person | 12,117 | 10 | mask | 1,700 |
| 1 | helmet | 14,165 | 11 | welding-glass | 4,383 |
| 2 | gloves | 8,917 | 12 | no-goggle | 411 |
| 3 | boots | 10,939 | 13 | no-mask | 3,250 |
| 4 | vest | 4,753 | 14 | no-welding-glass | 4,094 |
| 5 | no-helmet | 8,350 | 15 | safety-cone | 3,502 |
| 6 | no-gloves | 7,054 | 16 | machinery | 5,346 |
| 7 | no-boots | 5,523 | 17 | vehicle | 1,628 |
| 8 | no-vest | 4,158 | 18 | none (no relevant objects) | 797 |
| 9 | goggles | 518 | | | |

Full per-source → merged ID mapping tables live in `class_mapping.py`.

**Judgment calls made, worth revisiting:**
- `anuragraj03`'s `safety-shoes`/`no-safety-shoes` were mapped to `boots`/
  `no-boots` — the merged schema has no separate "shoes" class. If shoes and
  boots should be distinguished, this needs to change.
- `welding-glass` (anuragraj03) and `goggles` (ketakichalke-boots) were kept
  as separate classes rather than merged, since they're different eye-
  protection items and weren't specified as equivalent.
- `none` (id 18, from ketakichalke-boots) isn't a real object class — it
  marks background regions. Training code should likely ignore/drop it
  rather than train a detector to predict it.

## Default build (classes 0-8 only)

Running `scripts/build_dataset.py` with no arguments keeps 13,284 of the
19,385 merged images (6,101 dropped for having none of the 9 core classes)
and only these instance counts:

| ID | Class | Instances |
|---|---|---|
| 0 | person | 12,117 |
| 1 | helmet | 14,165 |
| 2 | gloves | 8,917 |
| 3 | boots | 10,939 |
| 4 | vest | 4,753 |
| 5 | no-helmet | 8,350 |
| 6 | no-gloves | 7,054 |
| 7 | no-boots | 5,523 |
| 8 | no-vest | 4,158 |

Per-source image counts after filtering:

| Source | train | val | test |
|---|---|---|---|
| anuragraj03 | 4,804 | 2,369 | 2,017 |
| ketakichalke-boots | 1,131 | 143 | 141 |
| snehilsanyal-main | 2,535 | 84 | 60 |

## Before unifying to the project's positive-only training schema

`implementation plan.md` separately calls for a positive-only training
schema (`person`, `hardhat`, `vest`, `boots`, `mask`) with negatives dropped
entirely — a further reduction from the 19-class merged schema above, to be
applied at training-config time rather than baked into the label files.
