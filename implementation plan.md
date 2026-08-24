# Implementation Plan: HI-VIS Automated AI Safety Compliance System

## 1. EXECUTIVE SUMMARY & STRATEGIC VISION

The construction sector remains the deadliest industry in the UK. Official HSE 2025/26 safety statistics report 25 construction fatalities, representing 20% of all worker deaths from a sector that employs only 6% of the workforce. At a fatal injury rate of 1.92 per 100,000 workers—4.8 times the all-industry average—the need for rigorous, automated monitoring has never been more critical. The HI-VIS system transitions safety monitoring from thin, inconsistent human spot inspections to a continuous, data-driven oversight model that utilizes existing site photography to generate searchable safety records, ensuring adherence to critical standards like **29 CFR 1926.100** (Head Protection) and **1926.96** (Foot Protection).

A critical strategic pivot in this implementation is the move from "Negative-Class Training" to "Algorithmic Compliance Logic." Experimental attempts to train models on negative classes (e.g., explicitly detecting "no_hardhat") have demonstrated catastrophic performance drops. Specifically, Sajjad’s dataset yielded a nearly unusable 1.8% mAP on the 'no_boots' class, as models struggle to learn the features of an absent object. HI-VIS adopts a superior "Positive-Only" detection path—identifying persons and their PPE (helmets, vests, boots, masks)—then applying a logical layer to determine non-compliance.

### Project Scope Definition

|   |   |   |
|---|---|---|
|Feature|Standard Site Photography (Input)|HI-VIS Searchable Safety Records (Output)|
|**Visibility**|Raw image files in unindexed folders.|Annotated visuals with localized PPE boxes.|
|**Analysis**|Manual, labor-intensive review.|Automated compliance verdict per person.|
|**Searchability**|None; chronologically filed.|Searchable database of dated exception logs.|
|**Accountability**|Inconsistent spot checks.|Documented contemporaneous safety evidence.|

This vision transforms passive photo folders into active safety evidence, creating a foundation of accountability through the rigorous data strategy detailed below.

## 2. DATA CURATION & MERGING STRATEGY

Dataset alignment is the strategic prerequisite for production-grade computer vision. Inconsistent labeling across sources leads to a "Missing Label Penalty" during backpropagation. If a boot is present but not labeled in a vest-only dataset, the engine treats those pixels as "background," causing gradient confusion that effectively penalizes the model for correctly identifying PPE.

### Thematic Analysis of Data Sources

We evaluated three primary Kaggle sources for their class distributions and annotation formats to ensure a unified pipeline:

|   |   |   |   |   |
|---|---|---|---|---|
|Source|Image Count|Class Count|Format|Primary PPE Focus|
|**snehilsanyal**|2,801|10|YOLO26|Hardhat, Vest (Includes Negatives)|
|**Sajjad**|1,405|4|YOLOv8|Helmets, Vests, Boots|
|**Anurag**|12,892|5|YOLOv8|General Construction PPE|

To bypass labeling conflicts and the "Missing Label Penalty," HI-VIS adopts the **Roboflow Construction PPE dataset** (8,845 images) as the "Clean Path." This dataset provides a unified ground truth for core PPE types (helmets, vests, and boots) across high-variance lighting. This clean foundation is required to achieve the robustness necessary for real-world site environments.

## 3. PREPROCESSING, TARGETED AUGMENTATION, & DATA CLEANING

Standardizing raw visual data ensures the model performs across heterogeneous site typologies. The HI-VIS pipeline standardizes annotation coordinates and image dimensions across merged Roboflow and SODA (Site Object Detection dAtaset) assets to ensure architectural compatibility for the YOLO26 backbone.

### Targeted Albumentations Pipeline

To simulate the environmental extremes of a live site, we employ a targeted pipeline:

- **Spatial Augmentations:** Rotations and scale shifts. Essential for recognizing workers captured from fixed security cameras at height or handheld devices at extreme low angles.
- **Environmental Augmentations:** Brightness/contrast shifts and synthetic shadow insertion. These simulate the harsh midday glare and deep shadows that often obscure PPE textures.
- **Noise Augmentations:** Motion blur. Replicates the reality of handheld site photography where motion artifacts are common.

**Strategic Impact:** These augmentations specifically optimize the detector for ground-level orientation and foot position. Given that boots are often the smallest and most occluded item required by **29 CFR 1926.96**, training the model to distinguish leather textures from ground-level shadows is critical for production reliability.

## 4. YOLO MODEL TRAINING ARCHITECTURE

For the 2026 production environment, we select **YOLO26** as the primary architecture. While legacy baselines like YOLOv8 provided speed, YOLO26 incorporates advanced spatial attention mechanisms necessary for detecting small-pixel objects.

**The "So What?" of Spatial Attention:** Respiratory and eye protection (**29 CFR 1926.102/101**) are historically the hardest to detect, with expected mAPs as low as 0.40–0.60. Spatial attention allows YOLO26 to focus feature extraction on high-occlusion regions, such as helmet straps and mask boundaries, which are often lost in standard global pooling layers.

### Hyperparameter Configuration & Infrastructure

Training is conducted on **on-premise GPU clusters** to ensure data security and stable gradient estimates, moving away from experimental edge environments:

- **Training Duration:** 100 epochs.
- **Optimization:** AdamW for superior weight decay handling in multi-class PPE detection.
- **Learning Rate:** Cosine schedule implementation for stable fine-tuning.
- **Batch Size:** Scaled to 32/64 to optimize throughput on industrial NVIDIA hardware.

These configurations prevent over-fitting to specific site backgrounds. However, raw detection is only the first phase; an association layer is required to move from "detected items" to "compliant individuals."

## 5. THE ASSOCIATION LAYER & ALGORITHMIC COMPLIANCE

In crowded construction scenes, "Spatial Ambiguity" leads to the **"Helmet Snatching"** problem: where a bounding box for Worker A's helmet overlaps Worker B's head. Traditional models may credit the wrong person with compliance.

### The Association Engine Logic

To resolve this, HI-VIS implements a post-processing layer:

1. **Head-Zone Centroid Distance:** Geometric constraints prioritize matching head-adjacent PPE (helmets/masks) to the person whose upper-body centroid is closest.
2. **Bipartite Matching (The Hungarian Algorithm):** A rigorous 1-to-1 matching rule is enforced. This prevents a single detected hardhat from being "shared" among multiple workers in a cluster.

This "Algorithmic Compliance" layer is more reliable than training noisy negative classes. By identifying a "Person" and logically verifying the presence of PPE within their specific spatial boundaries, we provide a logic-based over-watch that ensures compliance with **29 CFR 1926.100** is verified at an individual level.

## 6. EVALUATION PROTOCOL & PRODUCTION METRICS

A safety-critical system requires benchmarks that reflect life-safety priorities. In this context, **Recall on Violations** (minimizing False Negatives) is paramount.

### Safety-Critical Priority Analysis

A False Positive (a compliant worker flagged as non-compliant) is an administrative annoyance. A False Negative (missing a worker without a helmet) is a regulatory and safety failure. Therefore, we mandate the following **Minimum Acceptable Thresholds** for deployment:

- **Head Protection (1926.100) Recall:** > 0.95
- **Foot Protection (1926.96) Recall:** > 0.85
- **Overall mAP@50:** > 0.80

### Testing & Deployment

The testing regime utilizes a **Held-out Test Set** including SODA images to check for domain shift across different international PPE conventions. The final system output is a filterable **Exception Log**, which identifies the date, file, and specific missing PPE. These logs are then aggregated into the **Searchable Safety Records** defined in the strategic vision, providing safety officers with contemporaneous, actionable evidence of site compliance.