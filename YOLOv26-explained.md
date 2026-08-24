# YOLOv26 — Concise Explainer (for HI-VIS)

> One-paragraph version: **YOLO (You Only Look Once) is a single-pass object detector.** Give it an image, it predicts all bounding boxes + classes in one forward pass. YOLOv26 is the 2026 generation your plan selects — same YOLO idea, upgraded for tiny/occluded PPE (helmets, boots, masks) via spatial attention.

## 1. How YOLO Works (30s)

1.  **One look.** Unlike two-stage detectors (R-CNN), YOLO resizes the image to e.g. 640×640 and runs it through the network **once**.
2.  **Grid + anchors.** The image is divided into a grid. Each cell predicts N boxes: `(x, y, w, h, confidence, class)`.
3.  **NMS.** Overlapping boxes for the same object are merged via Non-Maximum Suppression — keep the best box.

Result: `~20-40ms` per image on GPU. Fast enough for batch photo folders.

## 2. Architecture: 3 Parts

```
Image → [ Backbone ] → [ Neck ] → [ Head ] → Boxes
```

| Part | Role | YOLOv26 Change (vs v8) |
|---|---|---|
| **Backbone** | Feature extractor (CNN). Turns pixels into feature maps. | Deeper CSP blocks + **spatial attention** — learns to focus on small regions (helmet straps, mask edges, boots vs shadow) instead of global pooling. |
| **Neck** (FPN/PAN) | Fuses multi-scale features (small objects need high-res, large objects need low-res). | Improved PAN for better small-object fusion. Critical for boots (fewest pixels). |
| **Head** | Predicts boxes + classes per scale. Anchor-free in v8/v26. | Decoupled head (box vs class separate) for fewer false positives. |

No new concept to learn — still `backbone → neck → head`.

## 3. Why YOLOv26 for HI-VIS

*   **Small-pixel objects.** Boots and masks are tiny/occluded. Spatial attention (plan §4) is the differentiator — 0.40-0.60 mAP on masks with v8 → materially higher with v26.
*   **Speed.** Single-pass = scan thousands of site photos without per-image reprocessing.
*   **Positive-only.** We **do not** train `no_hardhat` / `no_boots` (plan §1: 1.8% mAP — unlearnable). YOLOv26 only detects *present* objects: `person, hardhat, vest, boots, mask`. Compliance is logic after detection.

## 4. How HI-VIS Uses It

```
YOLOv26 detects → Association Layer decides compliance
```

1.  Detect all `person` + PPE boxes.
2.  **Association Engine** (plan §5): Hungarian matching + head-zone centroid distance assigns each PPE to exactly one person (fixes "helmet snatching").
3.  Rule: `if person has no hardhat in their zone → violation (29 CFR 1926.100)`.

Detection is perception. Compliance is logic.

## 5. Training in One Line

Pretrained COCO weights → fine-tune 100 epochs on Roboflow PPE (8,845 images) with Albumentations (rotate/scale, brightness/shadow, blur) → `best.pt`. Placeholder `yolov8n.pt` on Day 2 so team works in parallel (see `task parallelization.md`).

## 6. Strengths / Limits

**Strengths:** Real-time, strong on small PPE, simple deployment (`ultralytics` pip package, one `.pt` file).
**Limits:** Still struggles with extreme occlusion/crowds (needs association layer), biased to training distribution (needs SODA held-out test), boots recall will always lag helmets (set 0.85 vs 0.95 threshold).

---
*For the team: you don't need to understand the internals to integrate — `model("photo.jpg")` returns boxes; the rest is association logic.*
