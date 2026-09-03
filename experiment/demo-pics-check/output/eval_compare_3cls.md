# 4-Model Compare — Demo-pics GT Eval (IoU 0.5) — 3-class (person,helmet,vest)

- Source: `demo-pics` 48 images (15 challenging + 33 typical) — 38 matched, 10 unmatched (no GT in `data/merged/merge_manifest.csv`)
- GT: `data/merged` 9-class, **restricted to `0:person, 1:helmet, 4:vest`** (gloves `2` and boots `3` omitted from GT and preds; `no-*` also omitted)
- Models TL/TR/BL/BR: `yolo26m_merged_150ev2` / `yolov8n_scratch` / `yolo26m_css_300e` / `falcon_5cls`
- YOLO: `CONF=0.35` from `output/<model>/results.json`; Falcon: `experiment/falcon/outputs/demo-pics-5cls/**/predictions_yolo.txt` (no conf)
- Metrics: greedy IoU 0.5 match per class, `P=TP/(TP+FP) R=TP/(TP+FN) F1=2PR/(P+R) Acc=TP/(TP+FP+FN)` micro-averaged over the 3 classes
- Mapping: `Hardhat->helmet (1)`, `Person->person (0)`, `Safety Vest->vest (4)` for `snehilsanyal-main` models; `yolo26m_merged_*` and Falcon `0,1,4 -> 0,1,4` directly
- Note: gloves/boots are the rarest GT classes (`gloves 12, boots 49` across 38 images) and the noisiest for Falcon (`gloves 47 FP, boots 66 FP`); omitting them isolates head+body PPE

## Overall — 3-class (person,helmet,vest) — support 382 = 172+168+42

| Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26m_merged_150ev2 | 348 | 39 | 34 | 0.8992 | 0.9110 | 0.9051 (best YOLO) | 0.8266 |
| yolov8n_scratch | 190 | 88 | 192 | 0.6835 | 0.4974 | 0.5758 dF1 -0.329 | 0.4043 |
| yolo26m_css_300e | 257 | 67 | 125 | 0.7932 | 0.6728 | 0.7280 dF1 -0.177 | 0.5724 |
| falcon_5cls | 282 | 190 | 100 | 0.5975 | 0.7382 | 0.6604 dF1 -0.245 | 0.4930 |

## Per-class — F1 (3-class)

| class_id | name | yolo26m_merged_150ev2 F1 | yolov8n_scratch F1 | yolo26m_css_300e F1 | falcon_5cls F1 |
|---:|---|---:|---:|---:|---:|
| 0 | person | 0.8793 | 0.5695 | 0.7726 | 0.7632 |
| 1 | helmet | 0.9415 | 0.6028 | 0.6931 | 0.6250 |
| 4 | vest | 0.8608 | 0.5000 | 0.6829 | 0.4638 |

## Per-class — Precision / Recall / Accuracy

| class_id | name | yolo26m_merged_150ev2 P/R/Acc | yolov8n_scratch P/R/Acc | yolo26m_css_300e P/R/Acc | falcon_5cls P/R/Acc |
|---:|---|---:|---:|---:|---:|
| 0 | person | 0.8693/0.8895/0.7846 | 0.6615/0.5000/0.3981 | 0.8322/0.7209/0.6294 | 0.6971/0.8430/0.6170 |
| 1 | helmet | 0.9253/0.9583/0.8895 | 0.7456/0.5060/0.4315 | 0.7778/0.6250/0.5303 | 0.6250/0.6250/0.4545 |
| 4 | vest | 0.9189/0.8095/0.7556 | 0.5588/0.4524/0.3333 | 0.7000/0.6667/0.5185 | 0.3333/0.7619/0.3019 |

## By bucket — 3-class

| bucket | Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| challenging | yolo26m_merged_150ev2 | 132 | 17 | 14 | 0.8859 | 0.9041 | 0.8949 | 0.8098 |
| challenging | yolov8n_scratch | 61 | 42 | 85 | 0.5922 | 0.4178 | 0.4900 | 0.3245 |
| challenging | yolo26m_css_300e | 94 | 30 | 52 | 0.7581 | 0.6438 | 0.6963 | 0.5341 |
| challenging | falcon_5cls | 108 | 90 | 38 | 0.5455 | 0.7397 | 0.6279 | 0.4576 |
| typical | yolo26m_merged_150ev2 | 216 | 22 | 20 | 0.9076 | 0.9153 | 0.9114 | 0.8372 |
| typical | yolov8n_scratch | 129 | 46 | 107 | 0.7371 | 0.5466 | 0.6277 | 0.4574 |
| typical | yolo26m_css_300e | 163 | 37 | 73 | 0.8150 | 0.6907 | 0.7477 | 0.5971 |
| typical | falcon_5cls | 174 | 100 | 62 | 0.6350 | 0.7373 | 0.6824 | 0.5179 |

## Notes
- This file omits `gloves (2)` and `boots (3)` from both GT and predictions before matching. Compare with `eval_compare_4models.md` `Overall 5-class` (`F1 0.887 / 0.670 / 0.553 / 0.527`) to see the drag from rare classes: Falcon `gloves 0 TP 47 FP`, `boots 3 TP 66 FP`; YOLO merged is the only model trained on gloves/boots so it loses least when they are included.
- `yolov8n_scratch` / `yolo26m_css_300e` have no gloves/boots in their training schema (`snehilsanyal-main` -> `gloves/boots` are impossible), so their 5-class `FN 12+49=61` is unavoidable; 3-class removes that penalty.
- Falcon still has `P 0.597` / `R 0.738` on 3-class — `person` is strongest (`R 0.843`), `helmet` middle (`R 0.625`), `vest` weakest precision (`P 0.333`) due to false vest on `anuragraj03` images that never had vest GT (see `_gt_check/*__SIDE.jpg`).
- 38/48 matched; 10 unmatched are external web images with no manifest entry.