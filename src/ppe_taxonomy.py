"""
ppe_taxonomy.py
===============
Single source of truth for the PPE class taxonomy, the source-dataset registry,
and the name-driven mapping from any upstream dataset's labels onto our classes.

Design note
-----------
Mapping is done by CLASS NAME, never by class index. Upstream datasets get
re-exported, re-versioned and re-ordered without notice, so an index-based map
silently corrupts a whole training set. Names are normalised and looked up in
an alias table; anything unrecognised is reported in the audit rather than
guessed at.

Two upstream classes need special handling because they are not always mutually
exclusive with their positive counterpart:

  head  -> a 'head' box means 'no-helmet' ONLY if no helmet box overlaps it.
  foot  -> a 'foot' box means 'no-boots' ONLY if no boots box overlaps it.

Some datasets (SHWD / Hard Hat Workers) already annotate these exclusively, so
the subtraction is a no-op there and the audit will report zero drops. Others
(SH17) annotate the head region regardless of headgear - 11,985 head boxes
against 927 helmet boxes - so without subtraction every helmeted worker would
be labelled a violation. One code path handles both and tells you which you got.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Canonical taxonomy
# --------------------------------------------------------------------------
# Order IS the trained class index order. Do not reorder without retraining.
CANONICAL_CLASSES: List[str] = [
    "person",      # 0  full body - anchor for pose fusion
    "helmet",      # 1  hard hat, worn or otherwise present
    "no-helmet",   # 2  VIOLATION - uncovered head
    "vest",        # 3  hi-vis / reflective vest
    "no-vest",     # 4  VIOLATION - torso without hi-vis
    "boots",       # 5  safety footwear
    "no-boots",    # 6  VIOLATION - absent or non-safety footwear
]

CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# The three classes the whole exercise exists to report on. Metric selection,
# oversampling and threshold tuning are all driven off this list.
VIOLATION_CLASSES: List[str] = ["no-helmet", "no-vest", "no-boots"]
VIOLATION_IDS: List[int] = [CLASS_TO_ID[c] for c in VIOLATION_CLASSES]

# Positive/negative pairing, used by the fusion layer and the reconciliation table.
PPE_PAIRS: Dict[str, Dict[str, str]] = {
    "helmet": {"positive": "helmet", "negative": "no-helmet", "body_region": "head"},
    "vest":   {"positive": "vest",   "negative": "no-vest",   "body_region": "torso"},
    "boots":  {"positive": "boots",  "negative": "no-boots",  "body_region": "feet"},
}

# Pseudo-classes requiring overlap subtraction before they can be trusted.
HEAD_AUX = "__head__"
FOOT_AUX = "__foot__"
AUX_RESOLUTION = {
    # aux pseudo-class -> (canonical if unoccluded, positive class that cancels it)
    HEAD_AUX: ("no-helmet", "helmet"),
    FOOT_AUX: ("no-boots", "boots"),
}

DROP = None  # explicit sentinel: recognised upstream class, deliberately not used


# --------------------------------------------------------------------------
# Name normalisation
# --------------------------------------------------------------------------
_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalise(name: str) -> str:
    """'NO-Safety Vest' -> 'no_safety_vest'.  'Hard  Hat' -> 'hard_hat'."""
    return _NORM_RE.sub("_", str(name).strip().lower()).strip("_")


# --------------------------------------------------------------------------
# Global alias table  (normalised upstream name -> canonical class / aux / DROP)
# --------------------------------------------------------------------------
ALIASES: Dict[str, Optional[str]] = {}


def _register(canonical: Optional[str], *names: str) -> None:
    for n in names:
        ALIASES[normalise(n)] = canonical


_register("person", "person", "persons", "people", "worker", "workers", "human", "pedestrian")

_register("helmet",
          "helmet", "helmets", "hardhat", "hard hat", "hard-hat", "hat", "safety helmet",
          "safety_helmet", "with helmet", "helmet on", "wearing helmet", "hardhat on",
          "protective helmet", "safety hat")

_register("no-helmet",
          "no-hardhat", "no hardhat", "nohardhat", "no-helmet", "no helmet", "nohelmet",
          "no_hard_hat", "without helmet", "not wearing helmet", "helmet off",
          "bare head", "bare-head", "uncovered head", "no safety helmet")

_register("vest",
          "vest", "vests", "safety vest", "safety-vest", "safetyvest", "hi vis", "hivis",
          "hi-vis", "high vis", "high visibility vest", "reflective vest", "safety jacket",
          "with vest", "wearing vest", "reflective clothes", "safety_vest")

_register("no-vest",
          "no-safety vest", "no safety vest", "nosafetyvest", "no-vest", "no vest", "novest",
          "without vest", "not wearing vest", "no reflective vest", "no_safety_vest")

_register("boots",
          "boots", "boot", "safety boots", "safety-boots", "safety boot", "safety shoe",
          "safety shoes", "safety-shoes", "safetyshoe", "steel toe", "steel toe boots",
          "work boots", "protective footwear", "safety footwear")

_register("no-boots",
          "no-boots", "no boots", "noboots", "no shoes", "no_shoes", "noshoes",
          "without boots", "bare feet", "barefoot", "bare foot", "no safety shoes",
          "no safety footwear", "not wearing boots")

# Aux pseudo-classes - resolved by overlap subtraction, never used directly.
_register(HEAD_AUX, "head", "heads", "head region")
_register(FOOT_AUX, "foot", "feet")

# Recognised but deliberately unused. Listed explicitly so the audit can tell
# 'we know this class and chose to drop it' apart from 'we have never seen this'.
_register(DROP,
          "mask", "masks", "no-mask", "no mask", "no_mask", "face mask", "face_mask_medical",
          "face mask medical", "medical mask", "face guard", "face_guard", "faceguard",
          "glove", "gloves", "no-glove", "no glove", "no_glove", "no-gloves",
          "goggles", "no-goggles", "no goggles", "no_goggles", "glasses", "safety glasses",
          "eye protection", "ear", "ears", "earmuffs", "earmuff", "ear-mufs", "ear mufs",
          "ear protection",
          "face", "faces", "hand", "hands", "tool", "tools",
          "medical suit", "medical_suit", "safety suit", "safety_suit", "suit", "no-suit",
          "no_suit", "coverall", "coveralls",
          "safety cone", "safety-cone", "cone", "machinery", "machine", "vehicle",
          "vehicles", "truck", "excavator", "dumper", "loader",
          "fall", "falls", "falling", "fall detected", "fall_detected", "fallen",
          "harness", "no-harness", "lanyard", "scaffold", "ladder", "fire extinguisher",
          "first aid", "sign", "signage")


# --------------------------------------------------------------------------
# Source dataset registry
# --------------------------------------------------------------------------
@dataclass
class Source:
    """One upstream dataset.

    Attributes
    ----------
    key           short slug used in output filenames, must be filesystem-safe
    kind          'kaggle' | 'hf_snapshot' | 'hf_datasets' | 'url'
    ident         Kaggle slug 'owner/name', HF repo id, or download URL
    ann_format    'yolo' | 'voc' | 'coco' | 'hf_objects'  (best guess; the
                  converter sniffs the extracted tree and overrides this)
    licence       SPDX-ish string as published
    commercial_ok False for CC BY-NC-SA and friends. Systech is a commercial
                  consultancy, so NC sources are excluded by default and must
                  be opted into explicitly with --include-noncommercial.
    attribution   required credit line, if the licence demands one
    contributes   canonical classes this source is expected to supply
    overrides     source-specific name mapping that beats the global table.
                  Use this where a name means different things in different
                  datasets - e.g. 'shoes' is safety footwear in an industrial
                  PPE set but means any footwear at all in SH17.
    notes         anything a human should read before trusting this source
    """
    key: str
    kind: str
    ident: str
    ann_format: str
    licence: str
    commercial_ok: bool
    contributes: List[str] = field(default_factory=list)
    overrides: Dict[str, Optional[str]] = field(default_factory=dict)
    attribution: str = ""
    notes: str = ""
    hf_config: Optional[str] = None

    def resolve(self, upstream_name: str) -> Optional[str]:
        """Map one upstream class name onto a canonical class, aux token, or DROP.

        Returns the string '__UNKNOWN__' when the name is not recognised at all,
        so the caller can surface it instead of silently dropping data.
        """
        n = normalise(upstream_name)
        if n in self.overrides:
            return self.overrides[n]
        if n in ALIASES:
            return ALIASES[n]
        return "__UNKNOWN__"


# NOTE ON VERIFICATION
# --------------------
# Identifiers, class lists and licences below were checked against the dataset
# cards / source papers in Aug 2026. They are still treated as hints: the
# converter reads whatever class list actually ships with the download and maps
# by name, so a renamed or reordered upstream release degrades to an audit
# warning rather than a corrupted training set.

SOURCES: Dict[str, Source] = {

    "css": Source(
        key="css",
        kind="kaggle",
        ident="snehilsanyal/construction-site-safety-image-dataset-roboflow",
        ann_format="yolo",
        licence="CC BY 4.0 (Roboflow Universe export)",
        commercial_ok=True,
        contributes=["person", "helmet", "no-helmet", "vest", "no-vest"],
        notes=(
            "~2.8k construction-site images. The primary source of explicitly "
            "labelled negatives for helmet and vest. Ships classes: Hardhat, Mask, "
            "NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, "
            "machinery, vehicle. No footwear classes at all."
        ),
    ),

    "hardhat": Source(
        key="hardhat",
        kind="kaggle",
        ident="andrewmvd/hard-hat-detection",
        ann_format="voc",
        licence="CC0 1.0 (public domain)",
        commercial_ok=True,
        contributes=["person", "helmet", "no-helmet"],
        notes=(
            "~5k images, PASCAL VOC XML, classes helmet/head/person. Derived from "
            "the Safety Helmet Wearing Dataset, whose README states 'hat for "
            "positive object and person for negative object' - head and helmet are "
            "mutually exclusive here, so overlap subtraction is a no-op and the "
            "audit should report ~0 head boxes dropped. Strongest helmet source. "
            "Heavy negative skew in the parent set (9,044 helmeted vs 111,514 "
            "un-helmeted heads), so expect no-helmet to dominate."
        ),
    ),

    "kb_ppe": Source(
        key="kb_ppe",
        kind="hf_datasets",
        ident="keremberke/protective-equipment-detection",
        ann_format="coco",
        licence="CC BY 4.0",
        commercial_ok=True,
        contributes=["helmet", "no-helmet", "boots", "no-boots"],
        overrides={
            # Industrial PPE context: 'shoes' here means safety footwear and is
            # explicitly paired with a 'no_shoes' class, so the mapping is sound.
            "shoes": "boots",
            "no_shoes": "no-boots",
        },
        attribution="keremberke/protective-equipment-detection, CC BY 4.0",
        notes=(
            "~11,978 images across train/val/test, COCO annotations. Ships glove, "
            "goggles, helmet, mask, no_glove, no_goggles, no_helmet, no_mask, "
            "no_shoes, shoes. THE key source for footwear negatives - almost "
            "nothing else annotates a negative footwear class. No person or vest."
        ),
    ),

    "ppe6": Source(
        key="ppe6",
        kind="hf_snapshot",
        ident="51ddhesh/PPE_Detection",
        ann_format="yolo",
        licence="CC BY 4.0",
        commercial_ok=True,
        contributes=["helmet", "vest", "boots"],
        overrides={"safety_shoe": "boots"},
        attribution="51ddhesh/PPE_Detection, CC BY 4.0",
        notes=(
            "~16k annotated objects, YOLO folder layout with data.yaml. Classes "
            "Vest, Safety Shoe, Mask, Helmet, Goggles, Gloves - positives only. "
            "Useful for boots POSITIVES, which are otherwise thin."
        ),
    ),

    "sh17": Source(
        key="sh17",
        kind="kaggle",
        ident="mugheesahmad/sh17-dataset-for-ppe-detection",
        ann_format="yolo",
        licence="CC BY-NC-SA 4.0",
        commercial_ok=False,
        contributes=["person", "helmet", "no-helmet", "vest"],
        overrides={
            # SH17 'Shoes' is any footwear, not safety footwear. Mapping it to
            # 'boots' would teach the model that trainers are compliant PPE.
            # Dropped on purpose; 'foot' still yields no-boots after subtraction.
            "shoes": DROP,
        },
        attribution="SH17 (Mughees et al., arXiv:2407.04590), CC BY-NC-SA 4.0",
        notes=(
            "NON-COMMERCIAL LICENCE - excluded unless --include-noncommercial is "
            "passed. 8,099 images, 75,994 instances, 17 classes. The head class is "
            "defined as 'any view of the head: front, back, top or else' and there "
            "are 11,985 head boxes against only 927 helmet boxes, so head and "
            "helmet DO co-occur. Overlap subtraction is mandatory here or every "
            "helmeted worker becomes a false violation. Rich source of bare-head "
            "negatives from manufacturing settings."
        ),
    ),

    # --- registered Aug 2026, all identifiers fetched and class lists verified
    # --- against the datasets' own shipped files -------------------------------

    "keremberke_hh": Source(
        key="keremberke_hh",
        kind="hf_snapshot",
        ident="keremberke/hard-hat-detection",
        ann_format="coco",
        licence="CC BY 4.0",
        commercial_ok=True,
        contributes=["helmet", "no-helmet"],
        attribution="keremberke/hard-hat-detection, CC BY 4.0",
        notes=(
            "19,745 images (train 13,782 / valid 3,962 / test 2,001), COCO "
            "annotations. Verified categories verbatim from the shipped "
            "_annotations.coco.json: ['hardhat', 'no-hardhat']. The largest "
            "explicitly-labelled no-helmet pool we have - Roboflow 'Hard Hats' "
            "project (hard-hats-fhbh5), a different project from css. Roboflow "
            "pre-processing resized everything to 640x640."
        ),
    ),

    "ppe_v1": Source(
        key="ppe_v1",
        kind="kaggle",
        ident="beyzakucuk/ppe-detection-v1",
        ann_format="yolo",
        licence="MIT",
        commercial_ok=True,
        contributes=["person", "helmet", "vest", "boots"],
        notes=(
            "17,264 images (train 13,949 / valid 1,669 / test 1,646), YOLOv8 with "
            "data.yaml. Verified names verbatim: ['boots', 'gloves', 'goggles', "
            "'helmet', 'person', 'vest']. Positives only - no no-* classes - but "
            "the boots POSITIVES are valuable because footwear is otherwise thin. "
            "Roboflow export rbyz/ppe-6-classes v6; images are 3x brightness/"
            "exposure-augmented from ~5.7k sources, so near-duplicate leakage "
            "within the set is expected (dhash dedupe groups them into one split)."
        ),
    ),

    "ppe_nd": Source(
        key="ppe_nd",
        kind="kaggle",
        ident="ndomalau/personal-protective-equipment-ppe-dataset",
        ann_format="yolo",
        licence="CC BY 4.0",
        commercial_ok=True,
        contributes=["person", "helmet", "no-helmet", "vest", "no-vest"],
        attribution="ndomalau/personal-protective-equipment-ppe-dataset, CC BY 4.0",
        notes=(
            "4,060 images (train 3,248 / valid 406 / test 406), YOLO. Verified "
            "names verbatim from data.yaml: ['helmet', 'no_helmet', 'no_vest', "
            "'person', 'vest'] - both helmet AND vest negatives in one set, all "
            "mapping cleanly with no overrides needed. Good second opinion for "
            "the no-vest class that css currently owns."
        ),
    ),

    "ppe_combo": Source(
        key="ppe_combo",
        kind="kaggle",
        ident="shlokraval/ppe-dataset-yolov8",
        ann_format="yolo",
        licence="CC BY 4.0",
        commercial_ok=True,
        contributes=["person", "helmet", "no-helmet", "vest", "no-vest"],
        attribution="shlokraval/ppe-dataset-yolov8 (Roboflow personal-protective-"
                    "equipment-combined-model v4), CC BY 4.0",
        notes=(
            "44,002 images, YOLO. Verified names verbatim from data.yaml: "
            "['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', "
            "'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', "
            "'Person', 'Safety Cone', 'Safety Vest']. Heavy on NO-Hardhat and "
            "NO-Safety Vest negatives. CAUTION: this IS the Roboflow "
            "personal-protective-equipment-combined-model project that css's "
            "README lists as one of its image sources, so the two overlap - "
            "dhash dedupe handles the exact duplicates. 'Fall-Detected' is "
            "deliberately dropped. Kaggle lists the licence as Apache-2.0 but "
            "the dataset's own data.yaml says CC BY 4.0; both are commercial-safe."
        ),
    ),

    "hhvest": Source(
        key="hhvest",
        kind="kaggle",
        ident="muhammetzahitaydn/hardhat-vest-dataset-v3",
        ann_format="yolo",
        licence="CC0 1.0 (public domain)",
        commercial_ok=True,
        contributes=["person", "helmet", "no-helmet", "vest"],
        notes=(
            "22,141 images (train 17,248 / val 2,438 / test 2,455), YOLO. "
            "Verified classes verbatim from labels/classes.txt: helmet, vest, "
            "head, person. head (98k train boxes) dwarfs helmet (44k), so head "
            "and helmet co-occur and overlap subtraction is required - expect "
            "large aux_cancelled counts here; the surviving head boxes become "
            "no-helmet. No footwear. CC0 public domain, the cleanest possible "
            "licence for the consultancy."
        ),
    ),

    "ppe_waq": Source(
        key="ppe_waq",
        kind="kaggle",
        ident="waquarahmed1/ppe-dataset",
        ann_format="yolo",
        licence="CC BY-SA 4.0",
        commercial_ok=False,
        contributes=["helmet", "vest", "boots"],
        attribution="waquarahmed1/ppe-dataset, CC BY-SA 4.0",
        notes=(
            "SHARE-ALIKE LICENCE - commercial_ok=False on purpose. CC BY-SA is "
            "commercial-permitted but copyleft: any redistribution of derived "
            "material (arguably including trained weights) must stay BY-SA. "
            "Systech keeps its models proprietary, so this is excluded by "
            "default and only included with --include-noncommercial after a "
            "human signs off. Data itself is good: 11,777 images, YOLO, classes "
            "verbatim ['Helmet', 'Mask', 'Safety Vest', 'boots', 'glove'] - "
            "positives only, but a third source of boots positives."
        ),
    ),
}

COMMERCIAL_SOURCES = [k for k, s in SOURCES.items() if s.commercial_ok]
NONCOMMERCIAL_SOURCES = [k for k, s in SOURCES.items() if not s.commercial_ok]


def data_yaml_dict(root: str, kpt: bool = False) -> dict:
    """Ultralytics dataset YAML for the detection model."""
    d = {
        "path": root,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(CANONICAL_CLASSES)},
    }
    return d


if __name__ == "__main__":
    print(f"{len(CANONICAL_CLASSES)} canonical classes:")
    for i, c in enumerate(CANONICAL_CLASSES):
        flag = "  <-- VIOLATION" if c in VIOLATION_CLASSES else ""
        print(f"  {i}: {c}{flag}")
    print(f"\n{len(ALIASES)} alias entries registered.")
    print(f"Commercial-safe sources : {COMMERCIAL_SOURCES}")
    print(f"Non-commercial sources  : {NONCOMMERCIAL_SOURCES}")
