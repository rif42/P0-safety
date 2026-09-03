#!/usr/bin/env python
"""3-class (person,helmet,vest) eval — same sources as eval_compare_4models but omits gloves(2) and boots(3)."""
from pathlib import Path
import csv, json
from collections import defaultdict
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "demo-pics"
DATA = ROOT / "data/merged"
MANIFEST = DATA / "merge_manifest.csv"
PRED_FALCON = ROOT / "experiment/falcon/outputs/demo-pics-5cls"
OUT_DIR = ROOT / "experiment/demo-pics-check/output"
IOU_THR = 0.5
KEEP = [0,1,4]
NAMES = {0:"person",1:"helmet",4:"vest"}

def load_manifest(p):
    by_merged, by_original = {}, {}
    if not p.exists(): return by_merged, by_original
    for row in csv.DictReader(open(p, newline="", encoding="utf-8")):
        by_merged[row["merged_filename"]] = row
        by_original[row["original_filename"]] = row
    return by_merged, by_original

def resolve_gt(n, by_m, by_o):
    r = by_m.get(n) or by_o.get(n)
    if not r: return None, None, None
    return DATA / r["split"] / "labels" / (Path(r["merged_filename"]).stem + ".txt"), r["split"], r["merged_filename"]

def parse_txt(p):
    out=[]
    if not p or not p.exists(): return out
    for line in open(p, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        parts=line.split()
        if len(parts)<5: continue
        try: cls=int(float(parts[0])); cx,cy,w,h=map(float, parts[1:5])
        except: continue
        out.append((cls,cx,cy,w,h))
    return out

def xywhn(cx,cy,w,h,W,H): return [(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H]

def box_iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    if inter==0: return 0.0
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter+1e-9)

def evaluate_one(gt_by,pred_by,iou_thr):
    res={}
    for cls in set(gt_by) | set(pred_by):
        gts=gt_by.get(cls,[]); preds=pred_by.get(cls,[])
        if not gts: res[cls]=(0,len(preds),0); continue
        if not preds: res[cls]=(0,0,len(gts)); continue
        matched=[False]*len(gts); tp=0
        for pb in preds:
            best=0; bj=-1
            for j,gb in enumerate(gts):
                if matched[j]: continue
                v=box_iou(pb,gb)
                if v>best: best=v; bj=j
            if bj>=0 and best>=iou_thr:
                matched[bj]=True; tp+=1
        res[cls]=(tp,len(preds)-tp,len(gts)-tp)
    return res

def metrics(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0
    r=tp/(tp+fn) if tp+fn else 0
    f=2*p*r/(p+r) if p+r else 0
    a=tp/(tp+fp+fn) if tp+fp+fn else 0
    return p,r,f,a

by_merged, by_original = load_manifest(MANIFEST)
import yaml
ordered=["person","helmet","gloves","boots","vest","no-helmet","no-gloves","no-boots","no-vest"]
nc=9
if (DATA/"data.yaml").exists():
    d=yaml.safe_load(open(DATA/"data.yaml", encoding="utf-8"))
    if d.get("names"): ordered=list(d["names"]); nc=int(d.get("nc", len(ordered)))

IMG_EXTS={".jpg",".jpeg",".png",".webp",".bmp"}
images=[]
for bucket in ["challenging","typical"]:
    d=SRC/bucket
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                images.append((p,bucket))

MODELS_YOLO=["yolo26m_merged_150ev2","yolov8n_scratch","yolo26m_css_300e"]
yolo_preds={}
for m in MODELS_YOLO:
    data=json.load(open(OUT_DIR/m/"results.json", encoding="utf-8"))
    mp={}
    for entry in data:
        rel=entry["image"]
        by_cls=defaultdict(list)
        for det in entry.get("detections",[]):
            by_cls[int(det["cls_id"])].append(list(map(float, det["xyxy"])))
        mp[rel]=by_cls
    yolo_preds[m]=mp

falcon_map={}
for img,bucket in images:
    rel=f"{bucket}/{img.name}"
    stem=Path(img.name).stem
    pred_txt=PRED_FALCON/bucket/stem/"predictions_yolo.txt"
    if not pred_txt.exists():
        pred_txt=PRED_FALCON/bucket/Path(img.name).stem.replace(".jpg","")/"predictions_yolo.txt"
    raw=parse_txt(pred_txt)
    try:
        with Image.open(img) as im: W,H=im.size
    except: W,H=640,640
    by_cls=defaultdict(list)
    for cls,cx,cy,w,h in raw:
        by_cls[cls].append(xywhn(cx,cy,w,h,W,H))
    falcon_map[rel]=by_cls

ALL=["yolo26m_merged_150ev2","yolov8n_scratch","yolo26m_css_300e","falcon_5cls"]
results={m:{"total":defaultdict(lambda:[0,0,0]),"buckets":defaultdict(lambda:defaultdict(lambda:[0,0,0])),"n_matched":0,"n_unmatched":0} for m in ALL}

for img,bucket in images:
    rel=f"{bucket}/{img.name}"
    label_path,split,merged_name=resolve_gt(img.name, by_merged, by_original)
    is_matched=label_path is not None and label_path.exists()
    if is_matched:
        try:
            with Image.open(img) as im: W,H=im.size
        except:
            try:
                with Image.open(DATA/split/"images"/merged_name) as im: W,H=im.size
            except: is_matched=False
    if not is_matched:
        for m in ALL: results[m]["n_unmatched"]+=1
        continue
    gt_raw=parse_txt(label_path)
    gt_by=defaultdict(list)
    for cls,cx,cy,w,h in gt_raw:
        if cls in KEEP:
            gt_by[cls].append(xywhn(cx,cy,w,h,W,H))
    for m in ALL:
        if m=="falcon_5cls":
            pred_by={k:v for k,v in falcon_map.get(rel,{}).items() if k in KEEP}
        else:
            # for scratch/css need remap, for merged direct
            if m in ("yolov8n_scratch","yolo26m_css_300e"):
                remap={5:0,0:1,7:4}
                native=yolo_preds[m].get(rel,{})
                pred_by=defaultdict(list)
                for nid,boxes in native.items():
                    mid=remap.get(nid)
                    if mid in KEEP:
                        pred_by[mid].extend(boxes)
            else:
                pred_by={k:v for k,v in yolo_preds[m].get(rel,{}).items() if k in KEEP}
        per=evaluate_one(gt_by,pred_by,IOU_THR)
        for c,(tp,fp,fn) in per.items():
            results[m]["total"][c][0]+=tp; results[m]["total"][c][1]+=fp; results[m]["total"][c][2]+=fn
            results[m]["buckets"][bucket][c][0]+=tp; results[m]["buckets"][bucket][c][1]+=fp; results[m]["buckets"][bucket][c][2]+=fn
        results[m]["n_matched"]+=1

# compute outputs
outputs={}
for m in ALL:
    total=results[m]["total"]
    buckets=results[m]["buckets"]
    tp=sum(total[c][0] for c in KEEP); fp=sum(total[c][1] for c in KEEP); fn=sum(total[c][2] for c in KEEP)
    p,r,f,a=metrics(tp,fp,fn)
    per_class={}
    for c in KEEP:
        t,fpp,fnn=total.get(c,[0,0,0]); sup=t+fnn
        pp,rr,ff,aa=metrics(t,fpp,fnn)
        per_class[str(c)]={"name":ordered[c],"support":int(sup),"tp":int(t),"fp":int(fpp),"fn":int(fnn),"precision":pp,"recall":rr,"f1":ff,"accuracy":aa}
    by_bucket={}
    for bucket,d in buckets.items():
        btp=sum(d[c][0] for c in KEEP); bfp=sum(d[c][1] for c in KEEP); bfn=sum(d[c][2] for c in KEEP)
        pp,rr,ff,aa=metrics(btp,bfp,bfn)
        by_bucket[bucket]={"tp":int(btp),"fp":int(bfp),"fn":int(bfn),"precision":pp,"recall":rr,"f1":ff,"accuracy":aa}
    outputs[m]={"overall":{"tp":int(tp),"fp":int(fp),"fn":int(fn),"precision":p,"recall":r,"f1":f,"accuracy":a},"per_class":per_class,"by_bucket":by_bucket,"n_matched":results[m]["n_matched"],"n_unmatched":results[m]["n_unmatched"]}

print("3-class KEEP 0,1,4:", KEEP)
for m in ALL:
    o=outputs[m]["overall"]
    print(f"{m}: TP{o['tp']} FP{o['fp']} FN{o['fn']} P{o['precision']:.3f} R{o['recall']:.3f} F{o['f1']:.3f} Acc{o['accuracy']:.3f}")
    for c in KEEP:
        pc=outputs[m]["per_class"][str(c)]
        print(f"  {c} {ordered[c]}: tp{pc['tp']} fp{pc['fp']} fn{pc['fn']} P{pc['precision']:.3f} R{pc['recall']:.3f} F{pc['f1']:.3f}")

# save json/csv/md
out_json=OUT_DIR/"eval_compare_3cls.json"
out_csv=OUT_DIR/"eval_compare_3cls.csv"
out_md=OUT_DIR/"eval_compare_3cls.md"
out_json.parent.mkdir(parents=True, exist_ok=True)
json_out={"iou_threshold":IOU_THR,"source":str(SRC),"data":str(DATA),"manifest":str(MANIFEST),"keep_classes":KEEP,"keep_names":[ordered[c] for c in KEEP],"n_demo_images":len(images),"results":{}}
for m in ALL:
    json_out["results"][m]=outputs[m]
with open(out_json,"w",encoding="utf-8") as f: json.dump(json_out,f,indent=2,ensure_ascii=False)
print(f"Saved {out_json}")

with open(out_csv,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["model","class_id","name","support","tp","fp","fn","precision","recall","f1","accuracy","scope"])
    for m in ALL:
        for c in KEEP:
            r=outputs[m]["per_class"][str(c)]
            w.writerow([m,c,r["name"],r["support"],r["tp"],r["fp"],r["fn"],f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1']:.4f}",f"{r['accuracy']:.4f}","3-class"])
        o=outputs[m]["overall"]
        sup=sum(outputs[m]["per_class"][str(c)]["support"] for c in KEEP)
        w.writerow([m,"overall","person,helmet,vest",sup,o["tp"],o["fp"],o["fn"],f"{o['precision']:.4f}",f"{o['recall']:.4f}",f"{o['f1']:.4f}",f"{o['accuracy']:.4f}","3-class"])
print(f"Saved {out_csv}")

best=outputs["yolo26m_merged_150ev2"]["overall"]
def fmt(v): return f"{v:.4f}"
md=[]
md.append("# 4-Model Compare — Demo-pics GT Eval (IoU 0.5) — 3-class (person,helmet,vest)")
md.append("")
md.append("- Source: `demo-pics` 48 images (15 challenging + 33 typical) — 38 matched, 10 unmatched (no GT in `data/merged/merge_manifest.csv`)")
md.append("- GT: `data/merged` 9-class, **restricted to `0:person, 1:helmet, 4:vest`** (gloves `2` and boots `3` omitted from GT and preds; `no-*` also omitted)")
md.append("- Models TL/TR/BL/BR: `yolo26m_merged_150ev2` / `yolov8n_scratch` / `yolo26m_css_300e` / `falcon_5cls`")
md.append("- YOLO: `CONF=0.35` from `output/<model>/results.json`; Falcon: `experiment/falcon/outputs/demo-pics-5cls/**/predictions_yolo.txt` (no conf)")
md.append("- Metrics: greedy IoU 0.5 match per class, `P=TP/(TP+FP) R=TP/(TP+FN) F1=2PR/(P+R) Acc=TP/(TP+FP+FN)` micro-averaged over the 3 classes")
md.append("- Mapping: `Hardhat->helmet (1)`, `Person->person (0)`, `Safety Vest->vest (4)` for `snehilsanyal-main` models; `yolo26m_merged_*` and Falcon `0,1,4 -> 0,1,4` directly")
md.append("- Note: gloves/boots are the rarest GT classes (`gloves 12, boots 49` across 38 images) and the noisiest for Falcon (`gloves 47 FP, boots 66 FP`); omitting them isolates head+body PPE")
md.append("")
md.append("## Overall — 3-class (person,helmet,vest) — support 382 = 172+168+42")
md.append("")
md.append("| Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for m in ALL:
    o=outputs[m]["overall"]
    delta=f" dF1 {o['f1']-best['f1']:+.3f}" if m!="yolo26m_merged_150ev2" else " (best YOLO)"
    md.append(f"| {m} | {o['tp']} | {o['fp']} | {o['fn']} | {fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['f1'])}{delta} | {fmt(o['accuracy'])} |")
md.append("")
md.append("## Per-class — F1 (3-class)")
md.append("")
md.append("| class_id | name | yolo26m_merged_150ev2 F1 | yolov8n_scratch F1 | yolo26m_css_300e F1 | falcon_5cls F1 |")
md.append("|---:|---|---:|---:|---:|---:|")
for c in KEEP:
    row=f"| {c} | {ordered[c]} |"
    for m in ALL:
        row+=f" {fmt(outputs[m]['per_class'][str(c)]['f1'])} |"
    md.append(row)
md.append("")
md.append("## Per-class — Precision / Recall / Accuracy")
md.append("")
md.append("| class_id | name | yolo26m_merged_150ev2 P/R/Acc | yolov8n_scratch P/R/Acc | yolo26m_css_300e P/R/Acc | falcon_5cls P/R/Acc |")
md.append("|---:|---|---:|---:|---:|---:|")
for c in KEEP:
    def pra(m):
        r=outputs[m]["per_class"][str(c)]
        return f"{fmt(r['precision'])}/{fmt(r['recall'])}/{fmt(r['accuracy'])}"
    md.append(f"| {c} | {ordered[c]} | {pra('yolo26m_merged_150ev2')} | {pra('yolov8n_scratch')} | {pra('yolo26m_css_300e')} | {pra('falcon_5cls')} |")
md.append("")
md.append("## By bucket — 3-class")
md.append("")
md.append("| bucket | Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for bucket in ["challenging","typical"]:
    for m in ALL:
        b=outputs[m]["by_bucket"].get(bucket,{"tp":0,"fp":0,"fn":0,"precision":0,"recall":0,"f1":0,"accuracy":0})
        md.append(f"| {bucket} | {m} | {b['tp']} | {b['fp']} | {b['fn']} | {fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['f1'])} | {fmt(b['accuracy'])} |")
md.append("")
md.append("## Notes")
md.append("- This file omits `gloves (2)` and `boots (3)` from both GT and predictions before matching. Compare with `eval_compare_4models.md` `Overall 5-class` (`F1 0.887 / 0.670 / 0.553 / 0.527`) to see the drag from rare classes: Falcon `gloves 0 TP 47 FP`, `boots 3 TP 66 FP`; YOLO merged is the only model trained on gloves/boots so it loses least when they are included.")
md.append("- `yolov8n_scratch` / `yolo26m_css_300e` have no gloves/boots in their training schema (`snehilsanyal-main` -> `gloves/boots` are impossible), so their 5-class `FN 12+49=61` is unavoidable; 3-class removes that penalty.")
md.append("- Falcon still has `P 0.597` / `R 0.738` on 3-class — `person` is strongest (`R 0.843`), `helmet` middle (`R 0.625`), `vest` weakest precision (`P 0.333`) due to false vest on `anuragraj03` images that never had vest GT (see `_gt_check/*__SIDE.jpg`).")
md.append("- 38/48 matched; 10 unmatched are external web images with no manifest entry.")
with open(out_md,"w",encoding="utf-8") as f: f.write("\n".join(md))
print(f"Saved {out_md}")
print("\n".join(md))
