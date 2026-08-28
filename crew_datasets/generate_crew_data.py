"""HI-VIS pitch — synthetic two-crew CCTV dataset generator (v2).

Builds two illustrative "crew" photo sets for a compliance-trend pitch
story, from REAL, camera-verified construction-site CCTV frames — not
fabricated images or fabricated detections. What IS synthetic: which
crew/day/time each real photo is assigned to, and therefore the trend
shape you get when you plot compliance over time. What is NOT synthetic:
the photos themselves (genuine site camera captures, watermark and all)
and the compliance numbers on each one (computed from the dataset's own
human-annotated ground truth — see score_crew_data.py).

v2 change from the first round: rather than hand-picking and visually
verifying a handful of images one at a time (which only turned up 7 usable
"mixed compliance" photos), classify_cameras.py now reliably separates
every anuragraj03 frame by which of 3 real cameras captured it (see that
script's docstring for the full story — a red-carpet celebrity photo
mislabeled "no-helmet" in a DIFFERENT source is what started this whole
investigation). Restricting to the "HSM" camera (a real industrial
fabrication floor — 1500 classified photos) gives a MUCH bigger, still
fully real, still verified pool: 1400 "compliant" photos (helmet present,
no negative) and 95 "mixed" photos (some workers helmeted, some not, in
the same real frame) — 13x the diversity of the original 7-image pool,
with no risk of the off-domain contamination found in ketakichalke-boots
(excluded entirely) or the wrong-camera problem in anuragraj03's own PPC
entrance feed (also excluded — see classify_cameras.py).

Run from crew_datasets/, after classify_cameras.py:
    python generate_crew_data.py
"""

import random
import shutil
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_ROOT = REPO_ROOT / "data" / "merged"
CAMERA_SPLIT_DIR = Path(__file__).resolve().parent / "camera_split"
OUT_ROOT = Path(__file__).resolve().parent / "data"

SEED = 2026
CAMERA = "HSM"

# Weekly compliant-vs-mixed mix, week 1..6. Crew B mirrors Crew A but with
# a sharp step-change around week 4 (a toolbox talk / intervention) rather
# than a smooth climb -- a "before/after" jump reads better on stage than
# a gradual line. Crew A's decline accelerates in the final week (a "slow
# slide then a bad week" shape) rather than a straight line down.
CREW_SCHEDULES = {
    "crew_a": {
        "label": "Crew A — steady decline, then a bad week",
        "weekly_compliant_fraction": [0.95, 0.85, 0.70, 0.55, 0.45, 0.15],
    },
    "crew_b": {
        "label": "Crew B — flat, then a week-4 toolbox talk",
        "weekly_compliant_fraction": [0.20, 0.20, 0.25, 0.85, 0.90, 0.95],
    },
}

PHOTOS_PER_DAY = 3
DAYS_PER_WEEK = 5  # weekdays only
N_WEEKS = 6
START_DATE = datetime(2026, 6, 1, 0, 0)  # a Monday; arbitrary, illustrative


def load_pools():
    cam = pd.read_csv(CAMERA_SPLIT_DIR / "camera_classification.csv")
    hsm_files = set(cam[cam["camera"] == CAMERA]["file"])

    labels = pd.read_csv(MERGED_ROOT / "labels_long.csv")
    hsm_labels = labels[labels["file"].isin(hsm_files)]
    by_file = hsm_labels.groupby("file")["class_name"].apply(set)

    compliant = sorted(by_file[by_file.apply(lambda s: "helmet" in s and "no-helmet" not in s)].index)
    mixed = sorted(by_file[by_file.apply(lambda s: "helmet" in s and "no-helmet" in s)].index)
    return compliant, mixed


def resolve_image_path(file_stem):
    p = CAMERA_SPLIT_DIR / CAMERA / "images" / f"{file_stem}.jpg"
    return p if p.exists() else None


def business_hour_timestamp(day, rng):
    """A plausible, slightly irregular capture time within a working day —
    real CCTV doesn't sample on a perfect grid."""
    hour = rng.choice([8, 9, 10, 11, 13, 14, 15, 16, 17])
    minute = rng.randint(0, 59)
    return datetime.combine(day.date(), time(hour, minute))


def build_crew_manifest(crew_key, compliant_pool, mixed_pool, rng):
    schedule = CREW_SCHEDULES[crew_key]
    rows = []
    day = START_DATE
    week = 0
    weekday_count = 0
    used = set()
    while week < N_WEEKS:
        if day.weekday() < 5:  # Mon-Fri only
            compliant_frac = schedule["weekly_compliant_fraction"][week]
            n_photos = PHOTOS_PER_DAY + rng.choice([-1, 0, 0, 0, 1])
            n_photos = max(1, n_photos)
            for _ in range(n_photos):
                pool_name = "compliant" if rng.random() < compliant_frac else "mixed"
                pool = compliant_pool if pool_name == "compliant" else mixed_pool
                # Prefer not reusing a file already used by THIS crew, but
                # fall back to reuse once the pool (mixed especially) runs
                # low rather than erroring -- still real data either way.
                candidates = [f for f in pool if f not in used] or pool
                source_file = rng.choice(candidates)
                used.add(source_file)
                rows.append(
                    {
                        "crew": crew_key,
                        "week": week + 1,
                        "capture_datetime": business_hour_timestamp(day, rng),
                        "source_file": source_file,
                        "source_pool": pool_name,
                    }
                )
            weekday_count += 1
            if weekday_count == DAYS_PER_WEEK:
                weekday_count = 0
                week += 1
        day += timedelta(days=1)
    return pd.DataFrame(rows).sort_values("capture_datetime").reset_index(drop=True)


def materialize(manifest, crew_key):
    out_dir = OUT_ROOT / crew_key
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)

    filenames = []
    for _, row in manifest.iterrows():
        src_img = resolve_image_path(row["source_file"])
        src_lbl = CAMERA_SPLIT_DIR / CAMERA / "labels" / f"{row['source_file']}.txt"
        ts = row["capture_datetime"]
        base = f"{ts:%Y-%m-%d_%H-%M}_{row['source_pool']}"
        fname = f"{base}.jpg"
        i = 1
        while (out_dir / "images" / fname).exists():
            i += 1
            fname = f"{base}-{i}.jpg"
        shutil.copy(src_img, out_dir / "images" / fname)
        if src_lbl.exists():
            shutil.copy(src_lbl, out_dir / "labels" / fname.replace(".jpg", ".txt"))
        filenames.append(fname)
    manifest = manifest.copy()
    manifest["filename"] = filenames
    return manifest


def main():
    rng = random.Random(SEED)
    compliant_pool, mixed_pool = load_pools()
    print(f"Compliant pool ({CAMERA}): {len(compliant_pool)} images")
    print(f"Mixed pool ({CAMERA}): {len(mixed_pool)} images")

    all_manifests = []
    for crew_key in CREW_SCHEDULES:
        manifest = build_crew_manifest(crew_key, compliant_pool, mixed_pool, rng)
        manifest = materialize(manifest, crew_key)
        all_manifests.append(manifest)
        n_unique = manifest["source_file"].nunique()
        print(f"{crew_key}: {len(manifest)} photos ({n_unique} unique source images) across {manifest['week'].nunique()} weeks")

    combined = pd.concat(all_manifests, ignore_index=True)
    OUT_ROOT.mkdir(exist_ok=True)
    combined.to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"\nWrote {OUT_ROOT / 'manifest.csv'} ({len(combined)} rows)")
    print("Next: run score_crew_data.py to compute real ground-truth compliance for each photo.")


if __name__ == "__main__":
    main()
