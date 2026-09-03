"""Generate ground_truth tiles from data/merged GT for demo-pics (for 2x2 compare)."""
from pathlib import Path
import csv, json, cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "demo-pics"
DATA = ROOT / "data/merged"
MANIFEST = DATA / "merge_manifest.csv"
OUT_ROOT = Path(__file__).resolve().parent / "output" / "ground_truth"

NAMES = ["person","helmet","gloves","boots","vest","no-helmet","no-gloves","no-boots","no-vest"]
# BGR — match falcon_5cls for 0-4, muted for 5-8
COLORS_BGR = {
    0: (0, 0, 255),      # person red
    1: (0, 255, 0),      # helmet green
    2: (255, 0, 0),      # gloves blue
    3: (0, 255, 255),    # boots yellow
    4: (255, 0, 255),    # vest magenta
    5: (0, 128, 255),    # no-helmet orange
    6: (255, 128, 0),    # no-gloves light-blue
    7: (128, 0, 128),    # no-boots purple
    8: (128, 128, 128),  # no-vest gray
}

def add_label(img, label):
    h, w = img.shape[:2]
    banner_h = max(36, int(h * 0.065))
    font_scale = max(0.7, banner_h / 40 * 0.9)
    thickness = max(2, int(banner_h / 18))
    out = img.copy()
    cv2.rectangle(out, (0, 0), (w, banner_h), (30, 30, 30), -1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if tw > w - 20:
        font_scale *= (w - 20) / tw
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(10, (w - tw) // 2)
    ty = (banner_h + th) // 2
    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out

def parse_yolo_txt(p):
    out=[]
    if not p.exists(): return out
    for line in open(p, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        parts=line.split()
        if len(parts)<5: continue
        try:
            cls=int(float(parts[0])); cx,cy,w,h=map(float, parts[1:5])
        except: continue
        out.append((cls,cx,cy,w,h))
    return out

def main():
    by_merged, by_original = {}, {}
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        by_merged[row["merged_filename"]] = row
        by_original[row["original_filename"]] = row
    IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}
    images = sorted(p for p in SRC.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    print(f"Found {len(images)} images in {SRC}")
    import shutil
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_all=[]; rows=[]
    for img in images:
        rel = img.relative_to(SRC)
        row = by_merged.get(img.name) or by_original.get(img.name)
        src_im = cv2.imread(str(img))
        if src_im is None:
            from PIL import Image
            try:
                pil = Image.open(img).convert("RGB")
                src_im = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"WARN cannot read {img}: {e}")
                continue
        h,w = src_im.shape[:2]
        dets=[]
        annotated = src_im.copy()
        if row is None:
            # no GT — leave image as-is with banner only
            pass
        else:
            label_path = DATA / row["split"] / "labels" / (Path(row["merged_filename"]).stem + ".txt")
            for cls,cx,cy,bw,bh in parse_yolo_txt(label_path):
                if cls < 0 or cls >= len(NAMES): continue
                x1 = int(round((cx - bw/2) * w)); y1 = int(round((cy - bh/2) * h))
                x2 = int(round((cx + bw/2) * w)); y2 = int(round((cy + bh/2) * h))
                x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1)); x2=max(0,min(x2,w-1)); y2=max(0,min(y2,h-1))
                color = COLORS_BGR.get(cls, (255,255,255))
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                label = NAMES[cls]
                font_scale=0.5; thickness=1
                (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                pad=2
                bg_x0, bg_y0 = x1, max(0, y1 - th - pad*2)
                bg_x1, bg_y1 = x1 + tw + pad*2, y1
                cv2.rectangle(annotated, (bg_x0,bg_y0), (bg_x1,bg_y1), color, -1)
                cv2.putText(annotated, label, (bg_x0+pad, bg_y1 - pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
                dets.append({"xyxy":[float(x1),float(y1),float(x2),float(y2)], "conf":1.0, "cls_id":int(cls), "cls_name":NAMES[cls]})
        # banner
        tag = "ground_truth" if row is not None else "ground_truth (no GT)"
        annotated = add_label(annotated, tag)
        save_path = OUT_ROOT / rel.parent / f"{img.stem}_pred.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(save_path), annotated)
        if not ok:
            from PIL import Image
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(save_path)
        results_all.append({"image": str(rel).replace("\\","/"), "detections": dets, "has_gt": row is not None})
        rows.append({"image": str(rel).replace("\\","/"), "count": len(dets), "has_gt": int(row is not None)})
        print(f"  {rel}: {len(dets)} GT boxes -> {save_path}")
    with open(OUT_ROOT / "results.json","w",encoding="utf-8") as f:
        json.dump(results_all,f,indent=2,ensure_ascii=False)
    with open(OUT_ROOT / "summary.csv","w",newline="",encoding="utf-8") as f:
        import csv as _csv
        w=_csv.DictWriter(f, fieldnames=["image","count","has_gt"])
        w.writeheader(); w.writerows(rows)
    total=sum(r["count"] for r in rows)
    print(f"[ground_truth] done: {len(images)} images, {total} total GT boxes ({sum(r['has_gt'] for r in rows)} with GT, {len(images)-sum(r['has_gt'] for r in rows)} without)")

if __name__ == "__main__":
    main()
