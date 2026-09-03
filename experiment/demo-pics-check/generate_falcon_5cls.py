"""Generate falcon_5cls pred images + results.json/summary.csv from demo-pics-5cls predictions_yolo.txt."""
from pathlib import Path
import json, csv, cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "demo-pics"
FALCON_PRED_ROOT = ROOT / "experiment/falcon/outputs/demo-pics-5cls"
OUT_ROOT = Path(__file__).resolve().parent / "output" / "falcon_5cls"

# same as check.py
CONF_FALCON = 1.0
TILE_W, TILE_H = 960, 640

# names for 5-class
NAMES_5CLS = {0: "person", 1: "helmet", 2: "gloves", 3: "boots", 4: "vest"}
# BGR colors for cv2
COLORS_BGR = {
    0: (0, 0, 255),      # red (person)
    1: (0, 255, 0),      # green (helmet)
    2: (255, 0, 0),      # blue (gloves)
    3: (0, 255, 255),    # yellow (boots) BGR 0,255,255
    4: (255, 0, 255),    # magenta (vest)
}

def add_model_label(img: np.ndarray, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    banner_h = max(36, int(h * 0.065))
    font_scale = max(0.7, banner_h / 40 * 0.9)
    thickness = max(2, int(banner_h / 18))
    out = img.copy()
    cv2.rectangle(out, (0, 0), (w, banner_h), (30, 30, 30), -1)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if tw > w - 20:
        font_scale *= (w - 20) / tw
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(10, (w - tw) // 2)
    ty = (banner_h + th) // 2
    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out

def parse_yolo_txt(path: Path):
    out = []
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        parts=line.split()
        if len(parts)<5: continue
        try:
            cls = int(float(parts[0])); cx,cy,w,h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts)>=6 else CONF_FALCON
        except: continue
        out.append((cls,cx,cy,w,h,conf))
    return out

def main():
    IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}
    images = sorted(p for p in SRC.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    print(f"Found {len(images)} images in {SRC}")
    # clean
    import shutil
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_all = []
    rows = []
    for img in images:
        rel = img.relative_to(SRC)
        # map to falcon pred txt: use stem as in run_demo_pics.py -> out_dir = pred_root / rel.parent / rel.stem (stem = Path(name).stem)
        stem = Path(img.name).stem  # for .jpg.webp this is "xxx.jpg"
        pred_txt = FALCON_PRED_ROOT / rel.parent / stem / "predictions_yolo.txt"
        # fallback if not found, try alternative naming (full name minus suffixes)
        if not pred_txt.exists():
            # try with suffix handling for .webp double suffix: already stem handles; but also try without
            alt = FALCON_PRED_ROOT / rel.parent / (Path(img.name).stem.replace(".jpg","").replace(".jpeg","").replace(".png","")) / "predictions_yolo.txt"
            if alt.exists():
                pred_txt = alt
        # read image
        src_im = cv2.imread(str(img))
        if src_im is None:
            # try PIL fallback
            from PIL import Image
            try:
                pil = Image.open(img).convert("RGB")
                src_im = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"WARN cannot read {img}: {e}")
                continue
        h,w = src_im.shape[:2]
        dets = []
        pred_raw = parse_yolo_txt(pred_txt)
        # draw
        annotated = src_im.copy()
        for cls,cx,cy,bw,bh,conf in pred_raw:
            if cls not in NAMES_5CLS:
                continue
            # xywhn to xyxy
            x1 = int(round((cx - bw/2) * w))
            y1 = int(round((cy - bh/2) * h))
            x2 = int(round((cx + bw/2) * w))
            y2 = int(round((cy + bh/2) * h))
            x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1)); x2=max(0,min(x2,w-1)); y2=max(0,min(y2,h-1))
            color = COLORS_BGR.get(cls, (255,255,255))
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            label = f"{NAMES_5CLS[cls]}"
            # small label banner above box
            font_scale=0.5; thickness=1
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # background for label
            pad=2
            bg_x0, bg_y0 = x1, max(0, y1 - th - pad*2)
            bg_x1, bg_y1 = x1 + tw + pad*2, y1
            cv2.rectangle(annotated, (bg_x0,bg_y0), (bg_x1,bg_y1), color, -1)
            # text color: white or black based on luminance
            # simple: white
            cv2.putText(annotated, label, (bg_x0+pad, bg_y1 - pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
            # for results.json
            dets.append({"xyxy":[float(x1),float(y1),float(x2),float(y2)], "conf":float(conf), "cls_id":int(cls), "cls_name":NAMES_5CLS[cls]})
        # add banner
        annotated = add_model_label(annotated, "falcon_5cls")
        # save
        save_path = OUT_ROOT / rel.parent / f"{img.stem}_pred.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(save_path), annotated)
        if not ok:
            from PIL import Image
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(save_path)
        results_all.append({"image": str(rel).replace("\\","/"), "detections": dets})
        rows.append({"image": str(rel).replace("\\","/"), "count": len(dets)})
        print(f"  {rel}: {len(dets)} dets -> {save_path}")

    with open(OUT_ROOT / "results.json","w",encoding="utf-8") as f:
        json.dump(results_all,f,indent=2,ensure_ascii=False)
    with open(OUT_ROOT / "summary.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["image","count"])
        w.writeheader(); w.writerows(rows)
    total=sum(r["count"] for r in rows)
    print(f"[falcon_5cls] done: {len(images)} images, {total} total detections")
    print(f"  -> {OUT_ROOT}")
    # also verify 48 files
    pred_jpgs = list(OUT_ROOT.rglob("*_pred.jpg"))
    print(f"  pred jpgs: {len(pred_jpgs)}")
    # show histogram like before
    from collections import Counter
    cnt=Counter(r["count"] for r in rows)
    print(f"  histogram: {dict(cnt)}")

if __name__=="__main__":
    main()
