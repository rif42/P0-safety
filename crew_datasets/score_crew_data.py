"""HI-VIS pitch — score the two crew datasets' compliance.

Scores each photo from the dataset's own real, human-annotated ground
truth (data/merged/labels_long.csv) across three PPE items: helmet,
gloves, boots. This is real, not fabricated: real photos, real
annotations — only the crew/day assignment is synthetic (see
generate_crew_data.py).

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

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
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


def main():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv", parse_dates=["capture_datetime"])
    labels_long = pd.read_csv(REPO_ROOT / "data" / "merged" / "labels_long.csv")

    scored_rows = [compliance_from_ground_truth(labels_long, f) for f in manifest["source_file"]]
    scored = pd.concat([manifest, pd.DataFrame(scored_rows)], axis=1)

    out_path = DATA_DIR / "manifest_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(scored)} rows)")

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
