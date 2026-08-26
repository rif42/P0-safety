"""
Canonical merged class schema for dataset/, and the per-source remapping
tables used by scripts/build_dataset.py to translate each source's original
class IDs into it.

Numbering follows a fixed pattern: the requested core classes come first
(0-8: person, then helmet/gloves/boots/vest paired with their negatives in
the same relative order), then every remaining class is grouped as
[matching positives][matching negatives][unmatched singles], with the
background marker "none" last.
"""

MERGED_CLASSES = [
    "person",            # 0
    "helmet",            # 1
    "gloves",            # 2
    "boots",             # 3
    "vest",              # 4
    "no-helmet",         # 5
    "no-gloves",         # 6
    "no-boots",          # 7
    "no-vest",           # 8
    "goggles",           # 9
    "mask",              # 10
    "welding-glass",     # 11
    "no-goggle",         # 12
    "no-mask",           # 13
    "no-welding-glass",  # 14
    "safety-cone",       # 15
    "machinery",         # 16
    "vehicle",           # 17
    "none",              # 18  -- not a real object; "area with no relevant objects"
]

# The classes scripts/build_dataset.py includes when no --classes override is
# given: the core set (person, helmet/gloves/boots/vest, and their negatives).
DEFAULT_INCLUDED_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# original_class_id -> merged_class_id, per source.
SOURCE_CLASS_MAPS = {
    # data/raw/anuragraj03/dataset.yaml
    # NOTE: this source's "safety-shoes" / "no-safety-shoes" are treated as
    # equivalent to the merged schema's "boots" / "no-boots" (the schema has
    # no separate "shoes" class). Revisit if shoes and boots should stay
    # distinct. This source has no vest classes at all.
    "anuragraj03": {
        0: 6,   # no-safety-glove   -> no-gloves
        1: 5,   # no-safety-helmet  -> no-helmet
        2: 7,   # no-safety-shoes   -> no-boots
        3: 14,  # no-welding-glass  -> no-welding-glass
        4: 2,   # safety-glove      -> gloves
        5: 1,   # safety-helmet     -> helmet
        6: 3,   # safety-shoes      -> boots
        7: 11,  # welding-glass     -> welding-glass
    },
    # class list supplied directly (no local data.yaml for this source).
    # No "no-vest" class exists in this source.
    "ketakichalke-boots": {
        0: 1,   # Helmet    -> helmet
        1: 2,   # Gloves    -> gloves
        2: 4,   # Vest      -> vest
        3: 3,   # Boots     -> boots
        4: 9,   # Goggles   -> goggles
        5: 18,  # none      -> none
        6: 0,   # Person    -> person
        7: 5,   # no_helmet -> no-helmet
        8: 12,  # no_goggle -> no-goggle
        9: 6,   # no_gloves -> no-gloves
        10: 7,  # no_boots  -> no-boots
    },
    # experiments/legacy-snehilsanyal-yolov8n_100e/kaggle/working/ppe_data.yaml
    "snehilsanyal-main": {
        0: 1,   # Hardhat        -> helmet
        1: 10,  # Mask           -> mask
        2: 5,   # NO-Hardhat     -> no-helmet
        3: 13,  # NO-Mask        -> no-mask
        4: 8,   # NO-Safety Vest -> no-vest
        5: 0,   # Person         -> person
        6: 15,  # Safety Cone    -> safety-cone
        7: 4,   # Safety Vest    -> vest
        8: 16,  # machinery      -> machinery
        9: 17,  # vehicle        -> vehicle
    },
}
