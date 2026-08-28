"""HI-VIS pitch — score the two crew datasets and materialize the files.

Scores each photo from the dataset's own real, human-annotated ground
truth (data/merged/labels_long.csv) across three PPE items: helmet,
gloves, boots. This is real, not fabricated: real photos, real
annotations — only the crew/day assignment is synthetic (see
generate_crew_data.py).

v3: materialization now happens here (not in generate_crew_data.py),
because the output filename encodes each photo's real, computed overall
compliance verdict — "compliant" only if EVERY annotated item (helmet,
gloves, boots) is at 100% for that photo, "noncompliant" otherwise. This
matters: the previous version tagged filenames by which *pool* a photo was
drawn from ("compliant" pool = helmet criterion only), and 18/182 photos
(10%) disagreed once gloves/boots were also checked — e.g. a photo with
helmet_rate=1.0 but gloves_rate=0.0 was filed as "compliant", which isn't
true of the whole photo. The filename now reflects the actual multi-item
result, and images/labels are split into data/<crew>/<camera>/ subfolders
(camera from generate_crew_data.py's manifest, via classify_cameras.py) --
both crews currently draw entirely from the "HSM" camera, so there's one
subfolder today, but the structure holds if that changes.

Note on why this uses ground truth rather than live model re-detection:
an earlier attempt to run the actual trained detector (yolo26m_merged_150e,
via streamlit_app/detector.py) on this exact photo set found it produces
ZERO "person" detections on every single image, at any confidence
threshold — despite confidently detecting helmets/gloves/boots (>0.85
conf) in the same frames. This camera looks almost straight down from an
elevated mount; the model's "person" training examples are all ground-
level/eye-level shots. Since detector.assess() anchors every verdict to a
person box, it can't score this camera angle at all yet — a genuine,
useful domain-shift finding for the roadmap (more elevated-CCTV training
data), not a bug in this script. Ground truth sidesteps it cleanly because
the original annotators labeled items directly, with no person-box
dependency.

Run from crew_datasets/, after generate_crew_data.py:
    python score_crew_data.py
"""

import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_ROOT = REPO_ROOT / "data" / "merged"
CAMERA_SPLIT_DIR = Path(__file__).resolve().parent / "camera_split"
DATA_DIR = Path(__file__).resolve().parent / "data"

METRICS = {
    "helmet": ("helmet", "no-helmet"),
    "gloves": ("gloves", "no-gloves"),
    "boots": ("boots", "no-boots"),
}


def compliance_from_ground_truth(labels_long, source_file):
    rows = labels_long[labels_long["file"] == source_file]
    out = {}
    for metric, (pos, neg) in METRICS.items():
        n_pos = (rows["class_name"] == pos).sum()
        n_neg = (rows["class_name"] == neg).sum()
        total = n_pos + n_neg
        out[f"{metric}_rate"] = (n_pos / total) if total else None
        out[f"{metric}_n"] = int(total)
    return out


def overall_verdict(row):
    """"compliant" only if every annotated PPE item is at 100% for this
    photo -- a real, stricter, and more accurate check than any single
    metric (see module docstring)."""
    for metric in METRICS:
        rate = row[f"{metric}_rate"]
        if pd.notna(rate) and rate < 1.0:
            return "noncompliant"
    return "compliant"


def resolve_image_and_label(file_stem, camera):
    img_p = CAMERA_SPLIT_DIR / camera / "images" / f"{file_stem}.jpg"
    lbl_p = CAMERA_SPLIT_DIR / camera / "labels" / f"{file_stem}.txt"
    return (img_p if img_p.exists() else None), (lbl_p if lbl_p.exists() else None)


def materialize(scored):
    for crew in scored["crew"].unique():
        crew_dir = DATA_DIR / crew
        if crew_dir.exists():
            shutil.rmtree(crew_dir)

    filenames = []
    for _, row in scored.iterrows():
        img_p, lbl_p = resolve_image_and_label(row["source_file"], row["camera"])
        ts = pd.Timestamp(row["capture_datetime"])
        out_dir = DATA_DIR / row["crew"] / row["camera"]
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)

        base = f"{ts:%Y-%m-%d_%H-%M}_{row['verdict']}"
        fname = f"{base}.jpg"
        i = 1
        while (out_dir / "images" / fname).exists():
            i += 1
            fname = f"{base}-{i}.jpg"
        if img_p is not None:
            shutil.copy(img_p, out_dir / "images" / fname)
        if lbl_p is not None:
            shutil.copy(lbl_p, out_dir / "labels" / fname.replace(".jpg", ".txt"))
        filenames.append(fname)
    scored = scored.copy()
    scored["filename"] = filenames
    return scored


def main():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv", parse_dates=["capture_datetime"])
    labels_long = pd.read_csv(MERGED_ROOT / "labels_long.csv")

    scored_rows = [compliance_from_ground_truth(labels_long, f) for f in manifest["source_file"]]
    scored = pd.concat([manifest, pd.DataFrame(scored_rows)], axis=1)
    scored["verdict"] = scored.apply(overall_verdict, axis=1)

    scored = materialize(scored)

    out_path = DATA_DIR / "manifest_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(scored)} rows)")

    print("\nOverall verdict counts by crew:")
    print(scored.groupby(["crew", "verdict"]).size().unstack(fill_value=0))

    print("\nImages by crew/camera:")
    print(scored.groupby(["crew", "camera"]).size())

    for metric in METRICS:
        print(f"\nWeekly {metric} compliance rate by crew:")
        weekly = (
            scored.dropna(subset=[f"{metric}_rate"])
            .groupby(["crew", "week"])[f"{metric}_rate"]
            .mean()
            .unstack("crew")
            .round(2)
        )
        print(weekly)


if __name__ == "__main__":
    main()
