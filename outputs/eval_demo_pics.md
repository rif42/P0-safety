# yolo26m_merged_150ev2 — demo-pics Evaluation

> `runs/detect/yolo26m_merged_150ev2/weights/best.pt` vs `demo-pics/` with GT from `data/merged/` (via `data/merged/merge_manifest.csv`)

| Field | Value |
|---|---|
| **Weights** | `runs/detect/yolo26m_merged_150ev2/weights/best.pt` |
| **Source** | `demo-pics` (48 images: 15 challenging + 33 typical) |
| **GT root** | `data/merged` (`train`/`val`/`test`/`labels` + `merge_manifest.csv`) |
| **Matched / Unmatched** | 38 matched with GT / 10 unmatched (external, no GT) |
| **IoU threshold** | 0.5 (greedy per-class matching, conf-sorted) |
| **Conf threshold** | 0.25 |
| **Classes (nc=9)** | `person`, `helmet`, `gloves`, `boots`, `vest`, `no-helmet`, `no-gloves`, `no-boots`, `no-vest` |
| **Definitions** | `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`, `accuracy (Jaccard) = TP/(TP+FP+FN)` |
| **Date** | 2026-04-15 (from `outputs/eval_demo_pics.json`) |

---

## Overall (micro-averaged, 38 images)

| Precision | Recall | F1 | Accuracy (Jaccard) | TP | FP | FN | GT instances |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.869 | 0.912 | 0.890 | 0.802 | 489 | 74 | 47 | 536 |

## By bucket

| Bucket | Images (matched) | TP | FP | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| challenging | 13 | 190 | 26 | 17 | 0.880 | 0.918 | 0.898 | 0.815 |
| typical | 25 | 299 | 48 | 30 | 0.862 | 0.909 | 0.885 | 0.793 |

## Per-class

| # | Class | Support (GT) | TP | FP | FN | Precision | Recall | F1 | Accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | person | 172 | 156 | 24 | 16 | 0.867 | 0.907 | 0.886 | 0.796 |
| 1 | helmet | 168 | 163 | 15 | 5 | 0.916 | 0.970 | 0.942 | 0.891 |
| 2 | gloves | 12 | 12 | 2 | 0 | 0.857 | 1.000 | 0.923 | 0.857 |
| 3 | boots | 49 | 37 | 19 | 12 | 0.661 | 0.755 | 0.705 | 0.544 |
| 4 | vest | 42 | 34 | 5 | 8 | 0.872 | 0.810 | 0.840 | 0.723 |
| 5 | no-helmet | 29 | 25 | 3 | 4 | 0.893 | 0.862 | 0.877 | 0.781 |
| 6 | no-gloves | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | no-boots | 7 | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | no-vest | 57 | 55 | 6 | 2 | 0.902 | 0.965 | 0.932 | 0.873 |
| **overall** | — | **536** | **489** | **74** | **47** | **0.869** | **0.912** | **0.890** | **0.802** |

> `no-gloves` has 0 GT instances in the matched 38-image subset — all metrics are 0. Weakest class is `boots` (P 0.66 / R 0.76); strongest are `no-boots` (perfect on 7 instances) and `helmet` (P 0.916 / R 0.970).

---

## Per-image

| Image | Bucket | Split | GT | Pred | TP | FP | FN |
|---|---|---|---|---:|---:|---:|---:|---:|
| `-3766-_png_jpg.rf.81ead992d4b8047ec7beb28b7532cf0a.jpg` | challenging | train | 55 | 62 | 54 | 8 | 1 |
| `-4405-_png_jpg.rf.82b5c10b2acd1cfaa24259ada8e599fe.jpg` | challenging | train | 3 | 4 | 3 | 1 | 0 |
| `anuragraj03__005331_jpg.rf.6f33b1de6a36491745450d22f355c32a.jpg` | challenging | test | 16 | 18 | 16 | 2 | 0 |
| `anuragraj03__005354_jpg.rf.c1eefbf6cdbee9f086071435974241d0.jpg` | challenging | test | 1 | 1 | 1 | 0 | 0 |
| `anuragraj03__005360_jpg.rf.76cd4a716fa59edf807052ccfe690801.jpg` | challenging | test | 18 | 17 | 16 | 1 | 2 |
| `anuragraj03__frame00001_png.rf.8a0f5fb1d652ceb2340042962c04011a.jpg` | challenging | test | 21 | 20 | 18 | 2 | 3 |
| `anuragraj03__frame00121_png.rf.2c0882ee84e8074a67c70ca9e61d9ba2.jpg` | challenging | val | 27 | 27 | 26 | 1 | 1 |
| `anuragraj03__frame00271_png.rf.99575ef03b3806302ef8b71e87513080.jpg` | challenging | test | 16 | 16 | 16 | 0 | 0 |
| `ketakichalke-boots__image632.jpg` | challenging | val | 6 | 6 | 5 | 1 | 1 |
| `ketakichalke-boots__image645.jpg` | challenging | val | 21 | 18 | 12 | 6 | 9 |
| `ppe_0355_jpg.rf.508753d5b708536eca53de192b927c61.jpg` | challenging | train | 12 | 12 | 12 | 0 | 0 |
| `youtube-198_jpg.rf.e89faeb9765c6bd6cece5434d140f4af.jpg` | challenging | train | 3 | 3 | 3 | 0 | 0 |
| `youtube-342_jpg.rf.d7e55a17800f8d87313d7b6f33256ea9.jpg` | challenging | train | 8 | 12 | 8 | 4 | 0 |
| `000005_jpg.rf.96e9379ccae638140c4a90fc4b700a2b.jpg` | typical | train | 6 | 6 | 6 | 0 | 0 |
| `002551_jpg.rf.ce4b9f934161faa72c80dc6898d37b2d.jpg` | typical | train | 9 | 10 | 9 | 1 | 0 |
| `004063_jpg.rf.1b7cdc4035bcb24ef69b8798b444053e.jpg` | typical | train | 18 | 18 | 18 | 0 | 0 |
| `00596_jpg.rf.d030c5d98b937d080d75db1c1b269a84.jpg` | typical | train | 15 | 13 | 12 | 1 | 3 |
| `006463_jpg.rf.02f19082420ecc5537b9d59abbe6050c.jpg` | typical | train | 31 | 29 | 28 | 1 | 3 |
| `0_jpg.rf.2ff49f74309118f169e07aa12564df87.jpg` | typical | val | 13 | 14 | 13 | 1 | 0 |
| `2008_008320_jpg.rf.bd34011d46f82f9410d95f00e560b8ea.jpg` | typical | train | 4 | 4 | 4 | 0 | 0 |
| `anuragraj03__005447_jpg.rf.d6b948ff75c1201d032c20df018add4f.jpg` | typical | test | 7 | 8 | 7 | 1 | 0 |
| `anuragraj03__005491_jpg.rf.03afedd2b129bba220f187cac6baf072.jpg` | typical | test | 6 | 5 | 5 | 0 | 1 |
| `anuragraj03__005503_jpg.rf.75676b9f37499f5f6c4800b99678bfa2.jpg` | typical | test | 2 | 2 | 2 | 0 | 0 |
| `anuragraj03__005514_jpg.rf.357ef00ffb39721afc3bf9bea9dc6231.jpg` | typical | test | 22 | 21 | 20 | 1 | 2 |
| `anuragraj03__005518_jpg.rf.897f62ea1fbca703a5e94eba21b1cf8f.jpg` | typical | val | 13 | 14 | 12 | 2 | 1 |
| `anuragraj03__frame00001_png.rf.1bc96f2adf8cc42fe7d441bb169e5273.jpg` | typical | test | 11 | 14 | 10 | 4 | 1 |
| `anuragraj03__frame00031_png.rf.4ce9f7f0a0450f587d298dc47fae2ce8.jpg` | typical | val | 26 | 26 | 23 | 3 | 3 |
| `anuragraj03__frame00251_png.rf.c7baf25b891c45aff70e3fa12a7873c1.jpg` | typical | test | 15 | 17 | 14 | 3 | 1 |
| `anuragraj03__frame00281_png.rf.5ad3ea5aa3d13b16c25137ed015da273.jpg` | typical | test | 12 | 15 | 11 | 4 | 1 |
| `anuragraj03__helmet999532_jpg.rf.033e03a04792fbded6ab95d6b7d734a0.jpg` | typical | test | 5 | 6 | 5 | 1 | 0 |
| `class1_150_jpg.rf.5995dce34d38deb9eb0b6e36cae78f17.jpg` | typical | train | 6 | 6 | 6 | 0 | 0 |
| `ketakichalke-boots__image117.jpg` | typical | val | 29 | 29 | 25 | 4 | 4 |
| `ketakichalke-boots__image322.jpg` | typical | val | 17 | 23 | 15 | 8 | 2 |
| `ketakichalke-boots__image726.jpeg` | typical | val | 9 | 15 | 8 | 7 | 1 |
| `ketakichalke-boots__image962.jpg` | typical | val | 14 | 12 | 8 | 4 | 6 |
| `Mask2_mov-15_jpg.rf.026bd9a95154b1ead451a722a25ed130.jpg` | typical | val | 3 | 4 | 3 | 1 | 0 |
| `NX_img_177_jpg.rf.c03709e5fadfe2109411f05a9e9bc25f.jpg` | typical | train | 15 | 14 | 14 | 0 | 1 |
| `ppe_0665_jpg.rf.1dad479f7f54b2a7127cf18ef74ffd85.jpg` | typical | train | 21 | 22 | 21 | 1 | 0 |

---

## Unmatched (excluded from metrics)

10 images in `demo-pics/` have no entry in `data/merged/merge_manifest.csv` (external/web images, no GT):

- `demo-pics/challenging/f643eec0-dc10-11f0-bd2b-d531d865f86a.jpg.webp`
- `demo-pics/challenging/Incorrect-camera-lighting-for-PPE-detection-1024x576.webp`
- `demo-pics/typical/Building-Site-CCTV-Installation.jpg`
- `demo-pics/typical/Construction-site-security-team.jpg`
- `demo-pics/typical/Construction20CCTV.jpg`
- `demo-pics/typical/gettyimages-1354822857-640x640.jpg`
- `demo-pics/typical/Health-and-safety-on-a-construction.jpg`
- `demo-pics/typical/Health-and-Safety-on-building-site.jpg.webp`
- `demo-pics/typical/Incorrect-camera-positioning-with-obstructions-for-PPE-detection-1024x576.webp`
- `demo-pics/typical/Virtual-Trip-Wires-on-building-sites.jpg.webp`

---

## How to reproduce

```bash
python experiment/falcon/eval/eval_yolo_demo_pics.py --verbose
# custom thresholds / paths
python experiment/falcon/eval/eval_yolo_demo_pics.py --iou 0.5 --conf 0.25 \
  --weights runs/detect/yolo26m_merged_150ev2/weights/best.pt \
  --source demo-pics --data data/merged \
  --out outputs/eval_demo_pics.json
```

Artifacts: `outputs/eval_demo_pics.json`, `outputs/eval_demo_pics.csv`, `outputs/eval_demo_pics.md` — generated by `experiment/falcon/eval/eval_yolo_demo_pics.py`.
