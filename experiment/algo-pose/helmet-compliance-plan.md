# Helmet Compliance via Pose + Box (no NO-HELMET class)

> Assumption: no retraining now — ignore `NO-Hardhat`/`NO-Mask`/`NO-Safety Vest` at inference, use `Hardhat` boxes + pose head anchor. Retrain on filtered 7-class set only if geometric check proves insufficient.

1. Define head anchor from pose (reuse `sample_image` + existing dual inference)
   - inspect `results[0].keypoints` COCO-17 for this dataset — visible head keys (`nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`) per person, note occlusion rate on `data/css-data/test`
   - pick one head ROI formula — e.g. `center = mean(visible head kpts)`, `size = 1.8 * inter-eye distance` or `0.25 * person box width` fallback when <2 head kpts visible; document padding as single constant
   - use pose `results[0].boxes` as person box (has keypoints attached) to avoid cross-model matching for v1

2. Design helmet-on-head matching rule (no new deps, no new model)
   - input per person: `head_roi` (xyxy) + list of `Hardhat` boxes from `seg_model` (filter to `class_name == "Hardhat"` only, drop `NO-*`)
   - rule: `helmet worn = any Hardhat box with center inside expanded head_roi` (or `IoU(head_roi, helmet_box) > 0.1` — pick one, prefer center-in-box for small helmets)
   - constants at top of cell: `HEAD_PAD=1.4`, `CONF_THR=0.4`, `IOU_THR=0.1` — calibrate on ~5 test images, leave `# ponytail: fixed thresholds, learn per-scale if accuracy matters`
   - edge cases: no head kpts → fallback to top 15% of person box; multiple helmets → nearest center wins; multiple persons → greedy nearest helmet

3. Implement one compliance function + viz + table in `yolo-pose.ipynb` last cell (shortest diff)
   - `def is_helmet_on_head(person_kpts, helmet_boxes) -> bool` — ~15 lines, only new logic, testable via `__main__` assert on one positive/one negative crop
   - per-person classification: `compliant` / `violation` + reason string; color-code skeletons (green/red) via `result.plot()` override or `cv2.rectangle` on `head_roi`
   - table: `person_idx, head_conf, helmet_matched, max_helmet_conf, distance_to_head_center, verdict` — `display(df)`; no retraining, no new dependency

4. Validate and decide if retraining needed
   - run `combined_vertical` cell on 10–20 `test/images` diverse cases (far, occluded, side-view); compare verdict vs old `NO-Hardhat` baseline where available
   - tune `HEAD_PAD`/`CONF_THR` once from errors, note failure modes (hat vs helmet confusion, tiny helmets at distance) in markdown cell
   - outcome: if geometric check < target recall → plan follow-up to retrain `yolov8n`/`yolo26` on filtered 7-class set (drop `NO-*`) — not in this change
