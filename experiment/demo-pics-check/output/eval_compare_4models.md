# 4-Model Compare — Demo-pics GT Eval (IoU 0.5)

- Source: `demo-pics` 48 images (15 challenging + 33 typical) — 38 matched, 10 unmatched (no GT in `data/merged/merge_manifest.csv`)
- GT: `data/merged` 9-class `person,helmet,gloves,boots,vest,no-helmet,no-gloves,no-boots,no-vest`
- Models TL/TR/BL/BR: `yolo26m_merged_150ev2` / `yolov8n_scratch` / `yolo26m_css_300e` / `falcon_5cls` (5-class `person,helmet,gloves,boots,vest`)
- Inference: YOLO `CONF=0.35` from `output/<model>/results.json`; Falcon from `experiment/falcon/outputs/demo-pics-5cls/**/predictions_yolo.txt` (no conf)
- Metrics: greedy IoU 0.5 match per class, `P=TP/(TP+FP) R=TP/(TP+FN) F1=2PR/(P+R) Acc=TP/(TP+FP+FN)`
- Mapping: `Hardhat->helmet (1)`, `Person->person (0)`, `Safety Vest->vest (4)` for snehilsanyal-main models; Falcon 0-4 -> merged 0-4 directly; `no-*` classes are FN for Falcon by construction

## Overall (9-class, YOLO-compatible)

| Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26m_merged_150ev2 | 480 | 57 | 56 | 0.8939 | 0.8955 | 0.8947 (best YOLO) | 0.8094 |
| yolov8n_scratch | 6 | 431 | 530 | 0.0137 | 0.0112 | 0.0123 dF1 -0.882 | 0.0062 |
| yolo26m_css_300e | 3 | 531 | 533 | 0.0056 | 0.0056 | 0.0056 dF1 -0.889 | 0.0028 |
| falcon_5cls | 285 | 303 | 251 | 0.4847 | 0.5317 | 0.5071 dF1 -0.388 | 0.3397 |

## Overall — 5-class restricted (0-4: person,helmet,gloves,boots,vest)

| Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26m_merged_150ev2 | 396 | 54 | 47 | 0.8800 | 0.8939 | 0.8869 | 0.7968 |
| yolov8n_scratch | 190 | 88 | 253 | 0.6835 | 0.4289 | 0.5270 | 0.3578 |
| yolo26m_css_300e | 257 | 67 | 186 | 0.7932 | 0.5801 | 0.6701 | 0.5039 |
| falcon_5cls | 285 | 303 | 158 | 0.4847 | 0.6433 | 0.5529 | 0.3820 |

## Per-class (9-class) — F1

| class_id | name | yolo26m_merged_150ev2 F1 | yolov8n_scratch F1 | yolo26m_css_300e F1 | falcon_5cls F1 |
|---:|---|---:|---:|---:|---:|
| 0 | person | 0.8793 | 0.0140 | 0.0130 | 0.7632 |
| 1 | helmet | 0.9415 | 0.0000 | 0.0000 | 0.6250 |
| 2 | gloves | 0.9565 | 0.0000 | 0.0000 | 0.0000 |
| 3 | boots | 0.7327 | 0.0000 | 0.0000 | 0.0508 |
| 4 | vest | 0.8608 | 0.0684 | 0.0164 | 0.4638 |
| 5 | no-helmet | 0.8727 | 0.0000 | 0.0000 | 0.0000 |
| 6 | no-gloves | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 7 | no-boots | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | no-vest | 0.9550 | 0.0000 | 0.0000 | 0.0000 |

## Per-class — Precision / Recall / Accuracy (9-class)

| class_id | name | yolo26m_merged_150ev2 P/R/Acc | yolov8n_scratch P/R/Acc | yolo26m_css_300e P/R/Acc | falcon_5cls P/R/Acc |
|---:|---|---:|---:|---:|---:|
| 0 | person | 0.8693/0.8895/0.7846 | 0.0175/0.0116/0.0070 | 0.0148/0.0116/0.0066 | 0.6971/0.8430/0.6170 |
| 1 | helmet | 0.9253/0.9583/0.8895 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.6250/0.6250/0.4545 |
| 2 | gloves | 1.0000/0.9167/0.9167 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 |
| 3 | boots | 0.7115/0.7551/0.5781 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0435/0.0612/0.0261 |
| 4 | vest | 0.9189/0.8095/0.7556 | 0.0533/0.0952/0.0354 | 0.0125/0.0238/0.0083 | 0.3333/0.7619/0.3019 |
| 5 | no-helmet | 0.9231/0.8276/0.7742 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 |
| 6 | no-gloves | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 |
| 7 | no-boots | 1.0000/1.0000/1.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 |
| 8 | no-vest | 0.9815/0.9298/0.9138 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 |

## By bucket (9-class)

| bucket | Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| challenging | yolo26m_merged_150ev2 | 188 | 18 | 19 | 0.9126 | 0.9082 | 0.9104 | 0.8356 |
| challenging | yolov8n_scratch | 2 | 154 | 205 | 0.0128 | 0.0097 | 0.0110 | 0.0055 |
| challenging | yolo26m_css_300e | 2 | 198 | 205 | 0.0100 | 0.0097 | 0.0098 | 0.0049 |
| challenging | falcon_5cls | 109 | 116 | 98 | 0.4844 | 0.5266 | 0.5046 | 0.3375 |
| typical | yolo26m_merged_150ev2 | 292 | 39 | 37 | 0.8822 | 0.8875 | 0.8848 | 0.7935 |
| typical | yolov8n_scratch | 4 | 277 | 325 | 0.0142 | 0.0122 | 0.0131 | 0.0066 |
| typical | yolo26m_css_300e | 1 | 333 | 328 | 0.0030 | 0.0030 | 0.0030 | 0.0015 |
| typical | falcon_5cls | 176 | 187 | 153 | 0.4848 | 0.5350 | 0.5087 | 0.3411 |

## Notes
- `outputs/eval_demo_pics.csv` (yolo26m_merged_150ev2 at conf 0.25) is the reference; this compare uses `CONF=0.35` from `check.py` results, so tiny delta is expected.
- Falcon `5cls` has 0 for `no-helmet/no-gloves/no-boots/no-vest` by construction -> drags 9-class overall down; use 5-class restricted row for apples-to-apples on `person/helmet/gloves/boots/vest`.
- `yolov8n_scratch` / `yolo26m_css_300e` were trained on `snehilsanyal-main` schema (Hardhat/Person/etc) — their 9-class table includes `machinery/vehicle` not in merged eval; 5-class mapped view remaps `Person->person, Hardhat->helmet, Safety Vest->vest`.