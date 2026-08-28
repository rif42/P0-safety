# Crew A — "The decline nobody was watching"

**Dataset:** `data/crew_a/HSM/` (images + labels, split by camera) · 90 photos · 90 unique source images (no reuse)
**Camera:** HSM-Post-8, a real industrial fabrication-floor CCTV camera (see `classify_cameras.py`)
**Window:** 6 simulated weeks, 2026-06-01 → 2026-07-10, weekdays only, 2–4 photos/day
**Filenames:** `<date>_<time>_<verdict>.jpg` — verdict is the real, computed overall compliance
result (see below), not just which pool the photo was drawn from

## The story

Crew A starts the monitoring window essentially perfect — full helmet, glove, and
boot compliance in week 1. Nobody flags anything, nobody's watching closely, and
over the next five weeks small lapses become normal. By week 6, helmet compliance
has dropped 16 points and the week ends on its worst day yet. Nothing dramatic
happens in any single photo — that's the point. This is the case for continuous
monitoring: a slow drift that a monthly inspection would miss entirely, because
by the time an inspector shows up, it already looks like "how this crew works."

## What's real and what's illustrative

- **Real:** every one of the 90 photos is a genuine frame from a real industrial
  CCTV camera (burned-in date/time and camera ID "HSM-Post-8" watermark intact),
  individually verified not to be off-domain content (see the source repo's
  contamination audit on `ketakichalke-boots`, which is why this dataset draws
  exclusively from `anuragraj03`'s HSM camera instead). Every compliance number
  is the dataset's own human-annotated ground truth on that photo — not a
  re-run model score, not fabricated.
- **Illustrative:** which week/day/time each real photo was assigned to. That's
  what produces the decline shape. Presented as "an illustrative monitoring
  scenario built from real, verified site photos," not literal footage of one
  named crew over 6 real weeks.

## Weekly stats (real ground truth on the assigned photos)

| Week | Helmet | Gloves | Boots | Photos |
|---|---|---|---|---|
| 1 | 100% | 100% | 100% | 14 |
| 2 | 99% | 75% | 100% | 15 |
| 3 | 90% | 86% | 97% | 12 |
| 4 | 91% | 93% | 95% | 17 |
| 5 | 92% | 91% | 98% | 16 |
| 6 | 84% | 88% | 96% | 16 |

**Headline numbers:**
- Helmet compliance: **100% → 84%** (week 1 → week 6), overall mean 93%
- Gloves compliance: **100% → 88%**, overall mean 90%, noisiest of the three metrics
- Boots compliance: **100% → 96%** — stayed strong the entire window; boots was never the risk here
- **Overall verdict (every annotated item at 100% for that photo): 54 compliant / 36 noncompliant.**
  This is stricter than any single metric above — a photo with perfect helmets but
  one worker missing gloves counts as noncompliant, which is what the filename on
  each image reflects.
- 59 photos drawn from the "compliant" *pool* (helmet-only criterion used to build
  the schedule), 31 from the "mixed compliance" pool — note this differs from the
  90-photo overall-verdict split above precisely because the pool tag only looked
  at helmets; the filename uses the real multi-item verdict instead

## The pitch line

> "Nobody was watching daily. By the time a drop like this shows up in a
> monthly inspection, it's already a habit — not an incident."

## Provenance

`generate_crew_data.py` (seed 2026) decides which real photo from the `HSM`
camera pool in `camera_split/` (1400 compliant + 95 mixed real photos, both
camera-classified and spot-verified — see `classify_cameras.py`'s docstring
for the full investigation that led to this pipeline) lands on which
crew/day/timestamp, and writes `data/manifest.csv`. `score_crew_data.py` then
scores each photo against `data/merged/labels_long.csv`'s real annotations,
computes the overall verdict, and materializes the final
`data/<crew>/<camera>/images|labels/` files with verdict-bearing filenames,
writing `data/manifest_scored.csv`. Re-run both scripts in order to regenerate
a fresh draw (same seed = same result).
