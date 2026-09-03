#!/usr/bin/env python
"""Eval Falcon predictions_yolo.txt under demo-pics-5cls vs GT (same metrics as eval_yolo_demo_pics.py)."""
from pathlib import Path
import csv, json
from collections import defaultdict
from PIL import Image

# reuse helpers
def load_manifest(p: Path):
    by_merged, by_original = {}, {}
    if not p.exists(): return by_merged, by_original
    for row in csv.DictReader(open(p, newline="")):
        by_merged[row["merged_filename"]] = row
        by_original[row["original_filename"]] = row
    return by_merged, by_original

def resolve_gt(demo_name, by_merged, by_original, data_root: Path):
    row = by_merged.get(demo_name) or by_original.get(demo_name)
    if not row: return None, None, None
    merged = row["merged_filename"]; split=row["split"]
    return data_root/split/"labels"/(Path(merged).stem+".txt"), split, merged

def parse_yolo_txt(p: Path):
    out=[]
    if not p or not p.exists(): return out
    for line in open(p):
        line=line.strip()
        if not line: continue
        parts=line.split()
        if len(parts)<5: continue
        cls=int(float(parts[0])); cx,cy,w,h=map(float, parts[1:5])
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
    for cls in set(gt_by)|set(pred_by):
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

source=Path("demo-pics")
data_root=Path("data/merged")
manifest=Path("data/merged/merge_manifest.csv")
pred_root=Path("experiment/falcon/outputs/demo-pics-5cls")
out_json=Path("experiment/falcon/outputs/eval_falcon_5cls.json")
out_csv=Path("experiment/falcon/outputs/eval_falcon_5cls.csv")
iou=0.5

by_merged,by_original=load_manifest(manifest)
print(f"manifest {len(by_merged)} merged, {len(by_original)} original")

# model names from data.yaml
import yaml
ordered=["person","helmet","gloves","boots","vest","no-helmet","no-gloves","no-boots","no-vest"]
nc=9
if (data_root/"data.yaml").exists():
    try:
        d=yaml.safe_load(open(data_root/"data.yaml"))
        if d.get("names"): ordered=list(d["names"]); nc=int(d.get("nc",len(ordered)))
        print(f"data.yaml nc={nc} names={ordered}")
    except Exception as e: print(f"warn data.yaml {e}")

# 5-class run only covers 0-4; keep full 9 for compatibility but only 0-4 will have preds
exts={".jpg",".jpeg",".png",".webp",".bmp"}
# collect images same as yolo eval: challenging+typical
images=[]
for bucket in ["challenging","typical"]:
    d=source/bucket
    if not d.exists(): continue
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in exts or "".join(p.suffixes).lower().endswith(".webp"):
            images.append((p,bucket))
# fallback for .jpg.webp double suffix already covered? handle by suffixes
# redo with rglob if needed already covered; for safety also capture .jpg.webp where suffix is .webp but we already include .webp
print(f"found {len(images)} images")

# validate pred inventory
missing_pred=[]
for img,bucket in images:
    stem=Path(img.name).stem  # matches run_demo_pics.py out_dir logic
    pred_txt=pred_root/bucket/stem/"predictions_yolo.txt"
    if not pred_txt.exists():
        missing_pred.append((img,pred_txt))
print(f"missing predictions: {len(missing_pred)}")
for m in missing_pred[:5]: print(f"  {m[0]} -> {m[1]}")

total=defaultdict(lambda:[0,0,0])
bucket_total=defaultdict(lambda: defaultdict(lambda:[0,0,0]))
n_matched=0; n_unmatched=0; unmatched=[]
per_image=[]

for img_path,bucket in images:
    demo_name=img_path.name
    label_path,split,merged_name=resolve_gt(demo_name, by_merged, by_original, data_root)
    if label_path is None:
        n_unmatched+=1; unmatched.append(str(img_path)); continue
    if not label_path.exists():
        n_unmatched+=1; unmatched.append(str(img_path)+f" -> missing {label_path}"); continue
    # image size
    try:
        with Image.open(img_path) as im: W,H=im.size
    except Exception:
        try:
            with Image.open(data_root/split/"images"/merged_name) as im: W,H=im.size
        except Exception:
            n_unmatched+=1; unmatched.append(str(img_path)); continue
    gt_raw=parse_yolo_txt(label_path)
    gt_by=defaultdict(list)
    for cls,cx,cy,w,h in gt_raw:
        # GT includes all 9 classes, but we will evaluate only 0-4 later; keep all for now
        gt_by[cls].append(xywhn_to_xyxy(cx,cy,w,h,W,H))
    # pred
    stem=Path(img_path.name).stem
    pred_txt=pred_root/bucket/stem/"predictions_yolo.txt"
    pred_raw=parse_yolo_txt(pred_txt)
    pred_by=defaultdict(list)
    for cls,cx,cy,w,h in pred_raw:
        # pred cls 0-4 maps directly to GT 0-4 (person,helmet,gloves,boots,vest)
        pred_by[cls].append(xywhn_to_xyxy(cx,cy,w,h,W,H))
    # For fair 5-class comparison, restrict GT to 0-4 as well? No: eval_yolo counts all GT 0-8.
    # But Falcon 5cls never predicts 5-8, so those GT will always be FN.
    # To make comparison apples-to-apples with YOLO overall, we should keep all GT classes,
    # but also produce a 5-class-restricted overall for insight. Here keep full 9-class totals (YOLO-compatible).
    per_cls=evaluate_one(gt_by,pred_by,iou)
    for c,(tp,fp,fn) in per_cls.items():
        total[c][0]+=tp; total[c][1]+=fp; total[c][2]+=fn
        bucket_total[bucket][c][0]+=tp; bucket_total[bucket][c][1]+=fp; bucket_total[bucket][c][2]+=fn
    n_matched+=1
    per_image.append((img_path,bucket,split,merged_name,len(gt_raw),len(pred_raw),sum(v[0] for v in per_cls.values()),sum(v[1] for v in per_cls.values()),sum(v[2] for v in per_cls.values())))

def metrics(tp,fp,fn):
    prec=tp/(tp+fp) if tp+fp>0 else 0
    rec=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
    acc=tp/(tp+fp+fn) if tp+fp+fn>0 else 0
    return prec,rec,f1,acc

tot_tp=sum(v[0] for v in total.values()); tot_fp=sum(v[1] for v in total.values()); tot_fn=sum(v[2] for v in total.values())
op,orec,of1,oacc=metrics(tot_tp,tot_fp,tot_fn)

# also compute 5-class restricted totals (0-4 only)
r5_tp=sum(total[c][0] for c in range(5)); r5_fp=sum(total[c][1] for c in range(5)); r5_fn=sum(total[c][2] for c in range(5))
r5p,r5r,r5f1,r5acc=metrics(r5_tp,r5_fp,r5_fn)

print("\n"+"="*78)
print(f"Falcon 5cls demo-pics eval  IoU={iou}  {n_matched}/{len(images)} matched ({n_unmatched} unmatched)")
print(f"  Overall (9-class, YOLO-compatible): P={op:.3f} R={orec:.3f} F1={of1:.3f} Acc={oacc:.3f} TP={tot_tp} FP={tot_fp} FN={tot_fn}")
print(f"  Restricted (classes 0-4 only):      P={r5p:.3f} R={r5r:.3f} F1={r5f1:.3f} Acc={r5acc:.3f} TP={r5_tp} FP={r5_fp} FN={r5_fn}")
for bucket,d in bucket_total.items():
    btp=sum(v[0] for v in d.values()); bfp=sum(v[1] for v in d.values()); bfn=sum(v[2] for v in d.values())
    prec,rec,f1,acc=metrics(btp,bfp,bfn)
    print(f"  [{bucket:12s}] 9-class P={prec:.3f} R={rec:.3f} F1={f1:.3f} Acc={acc:.3f} (TP={btp} FP={bfp} FN={bfn})")
print("-"*78)
print(f"{'cls':>3}  {'name':<12} {'sup':>5} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
print("-"*78)
per_class={}
for c in range(nc):
    tp,fp,fn=total.get(c,[0,0,0]); sup=tp+fn
    prec,rec,f1,acc=metrics(tp,fp,fn)
    per_class[str(c)]={"name":ordered[c] if c<len(ordered) else str(c),"support":sup,"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1,"accuracy":acc}
    print(f"{c:>3}  {ordered[c] if c<len(ordered) else str(c):<12} {sup:>5} {tp:>4} {fp:>4} {fn:>4}  {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {acc:>6.3f}")
print("="*78)

# save json/csv (same schema as yolo eval)
buckets={}
for bucket,d in bucket_total.items():
    btp=sum(v[0] for v in d.values()); bfp=sum(v[1] for v in d.values()); bfn=sum(v[2] for v in d.values())
    prec,rec,f1,acc=metrics(btp,bfp,bfn)
    buckets[bucket]={"tp":btp,"fp":bfp,"fn":bfn,"precision":prec,"recall":rec,"f1":f1,"accuracy":acc}

result={
 "model":"falcon-5cls (person,helmet,gloves,boots,vest) from predictions_yolo.txt",
 "source":str(source),"data":str(data_root),"manifest":str(manifest),
 "pred_root":str(pred_root),
 "iou_threshold":iou,"model_names":ordered,
 "n_demo_images":len(images),"n_matched_with_gt":n_matched,"n_unmatched_no_gt":n_unmatched,
 "unmatched":unmatched,
 "overall":{"tp":tot_tp,"fp":tot_fp,"fn":tot_fn,"precision":op,"recall":orec,"f1":of1,"accuracy":oacc},
 "overall_5cls":{"tp":r5_tp,"fp":r5_fp,"fn":r5_fn,"precision":r5p,"recall":r5r,"f1":r5f1,"accuracy":r5acc},
 "per_class":per_class,"by_bucket":buckets,
 "per_image":[{"image":str(img),"bucket":b,"split":s,"merged_name":m,"gt_count":gc,"pred_count":pc,"tp":tp,"fp":fp,"fn":fn} for img,b,s,m,gc,pc,tp,fp,fn in per_image]
}
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json,"w") as f: json.dump(result,f,indent=2)
print(f"Saved JSON -> {out_json}")
with open(out_csv,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["class_id","name","support","tp","fp","fn","precision","recall","f1","accuracy"])
    for c in range(nc):
        r=per_class[str(c)]
        w.writerow([c,r["name"],r["support"],r["tp"],r["fp"],r["fn"],f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1']:.4f}",f"{r['accuracy']:.4f}"])
    w.writerow([])
    w.writerow(["overall (9-class)","",tot_tp+tot_fn,tot_tp,tot_fp,tot_fn,f"{op:.4f}",f"{orec:.4f}",f"{of1:.4f}",f"{oacc:.4f}"])
    w.writerow(["overall_5cls (0-4)","",r5_tp+r5_fn,r5_tp,r5_fp,r5_fn,f"{r5p:.4f}",f"{r5r:.4f}",f"{r5f1:.4f}",f"{r5acc:.4f}"])
print(f"Saved CSV -> {out_csv}")
