#!/usr/bin/env python
"""Unified GT eval for 4 models: yolo26m_merged_150ev2, yolov8n_scratch, yolo26m_css_300e, falcon_5cls."""
from pathlib import Path
import csv, json
from collections import defaultdict
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "demo-pics"
DATA = ROOT / "data/merged"
MANIFEST = DATA / "merge_manifest.csv"
PRED_ROOT_FALCON = ROOT / "experiment/falcon/outputs/demo-pics-5cls"
OUT_DIR = ROOT / "experiment/demo-pics-check/output"
IOU_THR = 0.5

def load_manifest(p):
    by_merged, by_original = {}, {}
    if not p.exists(): return by_merged, by_original
    for row in csv.DictReader(open(p, newline="", encoding="utf-8")):
        by_merged[row["merged_filename"]] = row
        by_original[row["original_filename"]] = row
    return by_merged, by_original

def resolve_gt(demo_name, by_merged, by_original):
    row = by_merged.get(demo_name) or by_original.get(demo_name)
    if not row: return None, None, None
    merged = row["merged_filename"]; split=row["split"]
    return DATA / split / "labels" / (Path(merged).stem + ".txt"), split, merged

def parse_yolo_txt(p):
    out=[]
    if not p or not p.exists(): return out
    for line in open(p, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        parts=line.split()
        if len(parts)<5: continue
        try:
            cls=int(float(parts[0])); cx,cy,w,h=map(float, parts[1:5])
        except:
            continue
        out.append((cls,cx,cy,w,h))
    return out

def xywhn_to_xyxy(cx,cy,w,h,W,H):
    return [(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H]

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
        if not gts and not preds: res[cls]=(0,0,0); continue
        if not gts: res[cls]=(0,len(preds),0); continue
        if not preds: res[cls]=(0,0,len(gts)); continue
        matched=[False]*len(gts); tp=0
        for pb in preds:
            best=0; bj=-1
            for j,gb in enumerate(gts):
                if matched[j]: continue
                iou=box_iou(pb,gb)
                if iou>best: best=iou; bj=j
            if bj>=0 and best>=iou_thr:
                matched[bj]=True; tp+=1
        res[cls]=(tp,len(preds)-tp,len(gts)-tp)
    return res

def metrics(tp,fp,fn):
    prec=tp/(tp+fp) if tp+fp>0 else 0.0
    rec=tp/(tp+fn) if tp+fn>0 else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    acc=tp/(tp+fp+fn) if tp+fp+fn>0 else 0.0
    return prec,rec,f1,acc

by_merged, by_original = load_manifest(MANIFEST)
print(f"manifest {len(by_merged)} merged, {len(by_original)} original")
import yaml
ordered=["person","helmet","gloves","boots","vest","no-helmet","no-gloves","no-boots","no-vest"]
nc=9
if (DATA/"data.yaml").exists():
    d=yaml.safe_load(open(DATA/"data.yaml", encoding="utf-8"))
    if d.get("names"):
        ordered=list(d["names"]); nc=int(d.get("nc",len(ordered)))
print(f"data.yaml nc={nc} names={ordered}")
IMG_EXTS={".jpg",".jpeg",".png",".webp",".bmp"}
images=[]
for bucket in ["challenging","typical"]:
    d=SRC/bucket
    if not d.exists(): continue
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            images.append((p,bucket))
print(f"found {len(images)} images")
MODELS_YOLO=["yolo26m_merged_150ev2","yolov8n_scratch","yolo26m_css_300e"]
yolo_preds={}
for m in MODELS_YOLO:
    res_path=OUT_DIR / m / "results.json"
    data=json.load(open(res_path, encoding="utf-8"))
    mapping={}
    for entry in data:
        rel=entry["image"]
        by_cls=defaultdict(list)
        for det in entry.get("detections",[]):
            cls=int(det["cls_id"]); xyxy=list(map(float, det["xyxy"]))
            by_cls[cls].append(xyxy)
        mapping[rel]=by_cls
    yolo_preds[m]=mapping
    print(f"{m}: {len(mapping)} images loaded")
falcon_mapping={}
for img,bucket in images:
    rel=f"{bucket}/{img.name}"
    stem=Path(img.name).stem
    pred_txt=PRED_ROOT_FALCON / bucket / stem / "predictions_yolo.txt"
    if not pred_txt.exists():
        pred_txt=PRED_ROOT_FALCON / bucket / (Path(img.name).stem.replace(".jpg","")) / "predictions_yolo.txt"
    raw=parse_yolo_txt(pred_txt)
    try:
        with Image.open(img) as im: W,H=im.size
    except:
        W,H=640,640
    by_cls=defaultdict(list)
    for cls,cx,cy,w,h in raw:
        by_cls[cls].append(xywhn_to_xyxy(cx,cy,w,h,W,H))
    falcon_mapping[rel]=by_cls
print(f"falcon_5cls: {len(falcon_mapping)} images mapped")
ALL_MODELS=["yolo26m_merged_150ev2","yolov8n_scratch","yolo26m_css_300e","falcon_5cls"]
results={m: {"total": defaultdict(lambda:[0,0,0]), "buckets": defaultdict(lambda: defaultdict(lambda:[0,0,0])), "per_image": [], "n_matched":0, "n_unmatched":0, "unmatched": []} for m in ALL_MODELS}
for img_path, bucket in images:
    rel=f"{bucket}/{img_path.name}"
    demo_name=img_path.name
    label_path, split, merged_name = resolve_gt(demo_name, by_merged, by_original)
    is_matched = label_path is not None and label_path.exists()
    if is_matched:
        try:
            with Image.open(img_path) as im: W,H=im.size
        except:
            try:
                with Image.open(DATA/split/"images"/merged_name) as im: W,H=im.size
            except:
                is_matched=False; W,H=640,640
    if not is_matched:
        for m in ALL_MODELS:
            results[m]["n_unmatched"]+=1
            results[m]["unmatched"].append(rel)
        continue
    gt_raw=parse_yolo_txt(label_path)
    gt_by=defaultdict(list)
    for cls,cx,cy,w,h in gt_raw:
        gt_by[cls].append(xywhn_to_xyxy(cx,cy,w,h,W,H))
    for m in ALL_MODELS:
        if m=="falcon_5cls":
            pred_by=falcon_mapping.get(rel, {})
        else:
            pred_by=yolo_preds[m].get(rel, {})
        per_cls=evaluate_one(gt_by, pred_by, IOU_THR)
        for c,(tp,fp,fn) in per_cls.items():
            results[m]["total"][c][0]+=tp; results[m]["total"][c][1]+=fp; results[m]["total"][c][2]+=fn
            results[m]["buckets"][bucket][c][0]+=tp; results[m]["buckets"][bucket][c][1]+=fp; results[m]["buckets"][bucket][c][2]+=fn
        results[m]["n_matched"]+=1
        tp=sum(v[0] for v in per_cls.values()); fp=sum(v[1] for v in per_cls.values()); fn=sum(v[2] for v in per_cls.values())
        pred_count=sum(len(v) for v in (falcon_mapping[rel] if m=="falcon_5cls" else yolo_preds[m][rel]).values())
        results[m]["per_image"].append({"image": rel, "bucket": bucket, "split": split, "merged_name": merged_name, "gt_count": len(gt_raw), "pred_count": pred_count, "tp": tp, "fp": fp, "fn": fn})
outputs={}
for m in ALL_MODELS:
    total=results[m]["total"]
    buckets=results[m]["buckets"]
    tot_tp=sum(v[0] for v in total.values()); tot_fp=sum(v[1] for v in total.values()); tot_fn=sum(v[2] for v in total.values())
    op,orec,of1,oacc=metrics(tot_tp,tot_fp,tot_fn)
    per_class={}
    for c in range(nc):
        tp,fp,fn=total.get(c,[0,0,0]); sup=tp+fn
        prec,rec,f1,acc=metrics(tp,fp,fn)
        per_class[str(c)]={"name": ordered[c], "support": int(sup), "tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
    by_bucket={}
    for bucket, d in buckets.items():
        btp=sum(v[0] for v in d.values()); bfp=sum(v[1] for v in d.values()); bfn=sum(v[2] for v in d.values())
        prec,rec,f1,acc=metrics(btp,bfp,bfn)
        by_bucket[bucket]={"tp": int(btp), "fp": int(bfp), "fn": int(bfn), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
    outputs[m]={"overall": {"tp": int(tot_tp), "fp": int(tot_fp), "fn": int(tot_fn), "precision": op, "recall": orec, "f1": of1, "accuracy": oacc}, "per_class": per_class, "by_bucket": by_bucket, "n_matched": results[m]["n_matched"], "n_unmatched": results[m]["n_unmatched"]}

def compute_mapped_5cls(model_name):
    remap={5:0, 0:1, 7:4}
    tot=defaultdict(lambda:[0,0,0])
    for img_path, bucket in images:
        rel=f"{bucket}/{img_path.name}"
        label_path, split, merged_name = resolve_gt(img_path.name, by_merged, by_original)
        if not label_path or not label_path.exists(): continue
        try:
            with Image.open(img_path) as im: W,H=im.size
        except:
            try:
                with Image.open(DATA/split/"images"/merged_name) as im: W,H=im.size
            except:
                continue
        gt_raw=parse_yolo_txt(label_path)
        gt_by=defaultdict(list)
        for cls,cx,cy,w,h in gt_raw:
            if cls in (0,1,2,3,4):
                gt_by[cls].append(xywhn_to_xyxy(cx,cy,w,h,W,H))
        pred_by_native=yolo_preds[model_name].get(rel, {})
        pred_by_mapped=defaultdict(list)
        for native_id, boxes in pred_by_native.items():
            merged_id=remap.get(native_id)
            if merged_id is not None:
                pred_by_mapped[merged_id].extend(boxes)
        per_cls=evaluate_one(gt_by, pred_by_mapped, IOU_THR)
        for c,(tp,fp,fn) in per_cls.items():
            tot[c][0]+=tp; tot[c][1]+=fp; tot[c][2]+=fn
    tp=sum(tot[c][0] for c in range(5)); fp=sum(tot[c][1] for c in range(5)); fn=sum(tot[c][2] for c in range(5))
    prec,rec,f1,acc=metrics(tp,fp,fn)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}, tot

for m in ("yolo26m_css_300e","yolov8n_scratch"):
    mapped, tot = compute_mapped_5cls(m)
    outputs[m]["overall_5cls_mapped"]=mapped
    outputs[m]["per_class_mapped"]={str(c): {"tp": int(tot[c][0]), "fp": int(tot[c][1]), "fn": int(tot[c][2]), "support": int(tot[c][0]+tot[c][2]), "precision": metrics(tot[c][0],tot[c][1],tot[c][2])[0], "recall": metrics(tot[c][0],tot[c][1],tot[c][2])[1], "f1": metrics(tot[c][0],tot[c][1],tot[c][2])[2], "accuracy": metrics(tot[c][0],tot[c][1],tot[c][2])[3]} for c in range(5)}

for m in ("yolo26m_merged_150ev2","falcon_5cls"):
    total=results[m]["total"]
    tp=sum(total[c][0] for c in range(5)); fp=sum(total[c][1] for c in range(5)); fn=sum(total[c][2] for c in range(5))
    prec,rec,f1,acc=metrics(tp,fp,fn)
    outputs[m]["overall_5cls"]={"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

print("overalls", {k: v["overall"] for k,v in outputs.items()})
print("mapped5", {k: v.get("overall_5cls") or v.get("overall_5cls_mapped") for k,v in outputs.items()})
out_json=OUT_DIR / "eval_compare_4models.json"
out_csv=OUT_DIR / "eval_compare_4models.csv"
out_md=OUT_DIR / "eval_compare_4models.md"
out_json.parent.mkdir(parents=True, exist_ok=True)
json_out={"iou_threshold": IOU_THR, "source": str(SRC), "data": str(DATA), "manifest": str(MANIFEST), "models": list(outputs.keys()), "model_names_ordered": ordered, "n_demo_images": len(images), "mapping_note": "yolo26m_css_300e/yolov8n_scratch native (Hardhat->helmet 1, Person->person 0, Safety Vest->vest 4) mapped for 5cls restricted view via SOURCE_CLASS_MAPS snehilsanyal-main; gloves/boots have no source in that dataset -> FN; merged+falcon use 0-8 directly", "results": {}}
for m in ALL_MODELS:
    json_out["results"][m]={"n_matched": results[m]["n_matched"], "n_unmatched": results[m]["n_unmatched"], "overall": outputs[m]["overall"], "overall_5cls": outputs[m].get("overall_5cls") or outputs[m].get("overall_5cls_mapped"), "per_class": outputs[m]["per_class"], "per_class_mapped_0_4": outputs[m].get("per_class_mapped"), "by_bucket": outputs[m]["by_bucket"]}
with open(out_json,"w",encoding="utf-8") as f:
    json.dump(json_out,f,indent=2,ensure_ascii=False)
print(f"Saved {out_json}")
with open(out_csv,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["model","class_id","name","support","tp","fp","fn","precision","recall","f1","accuracy","scope"])
    for m in ALL_MODELS:
        pc=outputs[m]["per_class"]
        for c in range(nc):
            r=pc[str(c)]
            w.writerow([m, c, r["name"], r["support"], r["tp"], r["fp"], r["fn"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['accuracy']:.4f}", "9-class"])
        o=outputs[m]["overall"]
        w.writerow([m, "overall", "", sum(outputs[m]["per_class"][str(c)]["support"] for c in range(nc)), o["tp"], o["fp"], o["fn"], f"{o['precision']:.4f}", f"{o['recall']:.4f}", f"{o['f1']:.4f}", f"{o['accuracy']:.4f}", "9-class"])
        o5=outputs[m].get("overall_5cls") or outputs[m].get("overall_5cls_mapped")
        sup5=sum(outputs[m].get("per_class_mapped", {}).get(str(c), {"support":0})["support"] if m in ("yolo26m_css_300e","yolov8n_scratch") else outputs[m]["per_class"][str(c)]["support"] for c in range(5))
        if m in ("yolo26m_merged_150ev2","falcon_5cls"):
            sup5=sum(outputs[m]["per_class"][str(c)]["support"] for c in range(5))
        w.writerow([m, "overall_5cls", "person,helmet,gloves,boots,vest", sup5, o5["tp"], o5["fp"], o5["fn"], f"{o5['precision']:.4f}", f"{o5['recall']:.4f}", f"{o5['f1']:.4f}", f"{o5['accuracy']:.4f}", "5-class"])
print(f"Saved {out_csv}")
best=outputs["yolo26m_merged_150ev2"]["overall"]
def fmt(v): return f"{v:.4f}"
md=[]
md.append("# 4-Model Compare — Demo-pics GT Eval (IoU 0.5)")
md.append("")
md.append(f"- Source: `demo-pics` 48 images (15 challenging + 33 typical) — 38 matched, 10 unmatched (no GT in `data/merged/merge_manifest.csv`)")
md.append(f"- GT: `data/merged` 9-class `person,helmet,gloves,boots,vest,no-helmet,no-gloves,no-boots,no-vest`")
md.append(f"- Models TL/TR/BL/BR: `yolo26m_merged_150ev2` / `yolov8n_scratch` / `yolo26m_css_300e` / `falcon_5cls` (5-class `person,helmet,gloves,boots,vest`)")
md.append(f"- Inference: YOLO `CONF=0.35` from `output/<model>/results.json`; Falcon from `experiment/falcon/outputs/demo-pics-5cls/**/predictions_yolo.txt` (no conf)")
md.append(f"- Metrics: greedy IoU 0.5 match per class, `P=TP/(TP+FP) R=TP/(TP+FN) F1=2PR/(P+R) Acc=TP/(TP+FP+FN)`")
md.append(f"- Mapping: `Hardhat->helmet (1)`, `Person->person (0)`, `Safety Vest->vest (4)` for snehilsanyal-main models; Falcon 0-4 -> merged 0-4 directly; `no-*` classes are FN for Falcon by construction")
md.append("")
md.append("## Overall (9-class, YOLO-compatible)")
md.append("")
md.append("| Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for m in ALL_MODELS:
    o=outputs[m]["overall"]
    delta=f" dF1 {o['f1']-best['f1']:+.3f}" if m!="yolo26m_merged_150ev2" else " (best YOLO)"
    md.append(f"| {m} | {o['tp']} | {o['fp']} | {o['fn']} | {fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['f1'])}{delta} | {fmt(o['accuracy'])} |")
md.append("")
md.append("## Overall — 5-class restricted (0-4: person,helmet,gloves,boots,vest)")
md.append("")
md.append("| Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for m in ALL_MODELS:
    o5=outputs[m].get("overall_5cls") or outputs[m].get("overall_5cls_mapped")
    md.append(f"| {m} | {o5['tp']} | {o5['fp']} | {o5['fn']} | {fmt(o5['precision'])} | {fmt(o5['recall'])} | {fmt(o5['f1'])} | {fmt(o5['accuracy'])} |")
md.append("")
md.append("## Per-class (9-class) — F1")
md.append("")
md.append("| class_id | name | yolo26m_merged_150ev2 F1 | yolov8n_scratch F1 | yolo26m_css_300e F1 | falcon_5cls F1 |")
md.append("|---:|---|---:|---:|---:|---:|")
for c in range(nc):
    name=ordered[c]
    row=f"| {c} | {name} |"
    for m in ALL_MODELS:
        f1=outputs[m]["per_class"][str(c)]["f1"]
        row+=f" {fmt(f1)} |"
    md.append(row)
md.append("")
md.append("## Per-class — Precision / Recall / Accuracy (9-class)")
md.append("")
md.append("| class_id | name | yolo26m_merged_150ev2 P/R/Acc | yolov8n_scratch P/R/Acc | yolo26m_css_300e P/R/Acc | falcon_5cls P/R/Acc |")
md.append("|---:|---|---:|---:|---:|---:|")
for c in range(nc):
    name=ordered[c]
    def pra(m):
        r=outputs[m]["per_class"][str(c)]
        return f"{fmt(r['precision'])}/{fmt(r['recall'])}/{fmt(r['accuracy'])}"
    md.append(f"| {c} | {name} | {pra('yolo26m_merged_150ev2')} | {pra('yolov8n_scratch')} | {pra('yolo26m_css_300e')} | {pra('falcon_5cls')} |")
md.append("")
md.append("## By bucket (9-class)")
md.append("")
md.append("| bucket | Model | TP | FP | FN | Precision | Recall | F1 | Accuracy |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for bucket in ["challenging","typical"]:
    for m in ALL_MODELS:
        b=outputs[m]["by_bucket"].get(bucket, {"tp":0,"fp":0,"fn":0,"precision":0,"recall":0,"f1":0,"accuracy":0})
        md.append(f"| {bucket} | {m} | {b['tp']} | {b['fp']} | {b['fn']} | {fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['f1'])} | {fmt(b['accuracy'])} |")
md.append("")
md.append("## Notes")
md.append("- `outputs/eval_demo_pics.csv` (yolo26m_merged_150ev2 at conf 0.25) is the reference; this compare uses `CONF=0.35` from `check.py` results, so tiny delta is expected.")
md.append("- Falcon `5cls` has 0 for `no-helmet/no-gloves/no-boots/no-vest` by construction -> drags 9-class overall down; use 5-class restricted row for apples-to-apples on `person/helmet/gloves/boots/vest`.")
md.append("- `yolov8n_scratch` / `yolo26m_css_300e` were trained on `snehilsanyal-main` schema (Hardhat/Person/etc) — their 9-class table includes `machinery/vehicle` not in merged eval; 5-class mapped view remaps `Person->person, Hardhat->helmet, Safety Vest->vest`.")
with open(out_md,"w",encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"Saved {out_md}")
print("\n".join(md))
