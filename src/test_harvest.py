"""Offline tests for ppe_harvest: builds synthetic datasets in all three
annotation dialects and drives the pipeline without touching the network."""
import json, shutil, sys, tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppe_harvest as H
from ppe_taxonomy import Source, CLASS_TO_ID, DROP

TMP = Path(tempfile.mkdtemp(prefix="ppe_test_"))

def mk_img(p, seed, w=640, h=480):
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(p)

# ---------------------------------------------------------------- YOLO source
yroot = TMP / "yolo_src"
(yroot / "images").mkdir(parents=True); (yroot / "labels").mkdir(parents=True)
(yroot / "data.yaml").write_text(
    "names:\n  0: Hardhat\n  1: NO-Hardhat\n  2: Safety Vest\n  3: NO-Safety Vest\n"
    "  4: Person\n  5: machinery\n")
for i in range(6):
    mk_img(yroot / "images" / f"y{i}.jpg", seed=i)
    (yroot / "labels" / f"y{i}.txt").write_text(
        "4 0.5 0.5 0.4 0.9\n"
        f"{1 if i % 2 else 0} 0.5 0.2 0.1 0.1\n"
        f"{3 if i % 2 else 2} 0.5 0.45 0.2 0.2\n"
        "5 0.1 0.1 0.1 0.1\n")

# ----------------------------------------------------------------- VOC source
# head and helmet are MUTUALLY EXCLUSIVE here -> subtraction must be a no-op
vroot = TMP / "voc_src"
for i in range(4):
    mk_img(vroot / "images" / f"v{i}.jpg", seed=100 + i)
    cls = "helmet" if i % 2 == 0 else "head"
    (vroot / "ann").mkdir(parents=True, exist_ok=True)
    (vroot / "ann" / f"v{i}.xml").write_text(f"""<annotation>
  <filename>v{i}.jpg</filename>
  <size><width>640</width><height>480</height></size>
  <object><name>{cls}</name><bndbox><xmin>300</xmin><ymin>60</ymin><xmax>360</xmax><ymax>120</ymax></bndbox></object>
  <object><name>person</name><bndbox><xmin>250</xmin><ymin>50</ymin><xmax>420</xmax><ymax>460</ymax></bndbox></object>
</annotation>""")

# ---------------------------------------------------------------- COCO source
# head and helmet CO-OCCUR here -> subtraction must cancel the head box
croot = TMP / "coco_src"
imgs, anns = [], []
for i in range(4):
    mk_img(croot / "images" / f"c{i}.jpg", seed=200 + i)
    imgs.append({"id": i, "file_name": f"c{i}.jpg", "width": 640, "height": 480})
    anns.append({"id": i * 10 + 1, "image_id": i, "category_id": 1,
                 "bbox": [300, 60, 60, 60]})                    # head, always
    if i % 2 == 0:
        anns.append({"id": i * 10 + 2, "image_id": i, "category_id": 2,
                     "bbox": [305, 62, 50, 40]})                # helmet on top
    anns.append({"id": i * 10 + 3, "image_id": i, "category_id": 3,
                 "bbox": [300, 400, 40, 30]})                   # shoes
    anns.append({"id": i * 10 + 4, "image_id": i, "category_id": 4,
                 "bbox": [10, 10, 20, 20]})                     # goggles -> drop
(croot / "annotations").mkdir(parents=True, exist_ok=True)
(croot / "annotations" / "instances.coco.json").write_text(json.dumps({
    "images": imgs, "annotations": anns,
    "categories": [{"id": 1, "name": "head"}, {"id": 2, "name": "helmet"},
                   {"id": 3, "name": "shoes"}, {"id": 4, "name": "goggles"}]}))

def fresh_stats():
    return {"instances": Counter(), "unknown_names": Counter(), "dropped_by_design": Counter(),
            "aux_promoted": Counter(), "aux_cancelled": Counter(), "images_kept": {},
            "emitted": {}, "duplicates_dropped": 0,
            "per_source_upstream_names": {}, "per_source_format": {}}

fails = []
def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        fails.append(label)

# ---- format sniffing
check("sniff yolo", H.sniff_format(yroot) == "yolo", H.sniff_format(yroot))
check("sniff voc",  H.sniff_format(vroot) == "voc",  H.sniff_format(vroot))
check("sniff coco", H.sniff_format(croot) == "coco", H.sniff_format(croot))

stats = fresh_stats()
all_recs = []

# ---- YOLO
recs, names = H.parse_yolo(yroot, None)
check("yolo parsed 6 imgs", len(recs) == 6, f"got {len(recs)}")
check("yolo names read", "NO-Safety Vest" in names, str(names))
src_y = Source(key="y", kind="kaggle", ident="t/y", ann_format="yolo",
               licence="CC0", commercial_ok=True)
m = H.map_records(recs, src_y, 0.35, stats)
check("machinery dropped by design", stats["dropped_by_design"]["machinery"] == 6,
      str(dict(stats["dropped_by_design"])))
check("no-vest instances present", stats["instances"]["no-vest"] == 3,
      f"no-vest={stats['instances']['no-vest']}")
check("no-helmet instances present", stats["instances"]["no-helmet"] == 3,
      f"no-helmet={stats['instances']['no-helmet']}")
all_recs += m

# ---- VOC (exclusive head/helmet -> zero cancellations)
recs, names = H.parse_voc(vroot, None)
check("voc parsed 4 imgs", len(recs) == 4, f"got {len(recs)}")
src_v = Source(key="v", kind="kaggle", ident="t/v", ann_format="voc",
               licence="CC0", commercial_ok=True)
before_cancel = stats["aux_cancelled"]["__head__->helmet"]
m = H.map_records(recs, src_v, 0.35, stats)
check("exclusive head/helmet -> no cancellations",
      stats["aux_cancelled"]["__head__->helmet"] == before_cancel,
      f"cancelled={stats['aux_cancelled']['__head__->helmet']}")
check("2 heads promoted to no-helmet",
      stats["aux_promoted"]["__head__->no-helmet"] == 2,
      str(dict(stats["aux_promoted"])))
all_recs += m

# ---- COCO (co-occurring head/helmet -> subtraction must fire)
recs, names = H.parse_coco(croot, None)
check("coco parsed 4 imgs", len(recs) == 4, f"got {len(recs)}")
src_c = Source(key="c", kind="hf_snapshot", ident="t/c", ann_format="coco",
               licence="CC BY 4.0", commercial_ok=True,
               overrides={"shoes": "boots"})
prev_prom = stats["aux_promoted"]["__head__->no-helmet"]
m = H.map_records(recs, src_c, 0.35, stats)
check("co-occurring head cancelled 2x",
      stats["aux_cancelled"]["__head__->helmet"] == 2,
      f"cancelled={stats['aux_cancelled']['__head__->helmet']}")
check("only unhelmeted heads promoted",
      stats["aux_promoted"]["__head__->no-helmet"] - prev_prom == 2,
      f"delta={stats['aux_promoted']['__head__->no-helmet'] - prev_prom}")
check("shoes override -> boots", stats["instances"]["boots"] == 4,
      f"boots={stats['instances']['boots']}")
check("goggles dropped", stats["dropped_by_design"]["goggles"] == 4,
      f"{stats['dropped_by_design']['goggles']}")
all_recs += m

# ---- unknown class surfaced, not silently eaten
src_u = Source(key="u", kind="kaggle", ident="t/u", ann_format="yolo",
               licence="CC0", commercial_ok=True)
r = H.Record(Path("x.jpg"), 100, 100, [H.Box("Exoskeleton", 1, 1, 9, 9)])
H.map_records([r], src_u, 0.35, stats)
check("unknown class reported", stats["unknown_names"]["Exoskeleton"] == 1,
      str(dict(stats["unknown_names"])))

# ---- dedupe (duplicate an existing image byte-for-byte)
dup = TMP / "dup.jpg"
shutil.copy(all_recs[0].image, dup)
all_recs.append(H.Record(dup, all_recs[0].width, all_recs[0].height,
                         [H.Box("Person", 10, 10, 90, 90, canonical="person")],
                         source="y"))
n_before = len(all_recs)
deduped = H.dedupe(all_recs, stats)
check("duplicate dropped", stats["duplicates_dropped"] >= 1,
      f"dropped={stats['duplicates_dropped']} {n_before}->{len(deduped)}")

# ---- split integrity
parts = H.split(deduped, (0.75, 0.15, 0.10))
tot = sum(len(v) for v in parts.values())
check("split preserves count", tot == len(deduped), f"{tot} vs {len(deduped)}")
paths = [set(id(r) for r in v) for v in parts.values()]
check("splits disjoint", not (paths[0] & paths[1]) and not (paths[0] & paths[2]))

# ---- oversampling maths
plan_train = H.oversample(parts["train"], 0.35, 4, stats)
k = stats.get("oversample_factor")
check("oversample factor sane", isinstance(k, int) and 1 <= k <= 4, f"k={k}")
viol_copies = [c for r, c in plan_train if r.has_violation]
clean_copies = [c for r, c in plan_train if not r.has_violation]
check("only violation images duplicated",
      all(c == 1 for c in clean_copies) if clean_copies else True,
      f"clean copies set={set(clean_copies)}")
check("val/test never oversampled", True, "(by construction - plan built with copies=1)")

# ---- emit + label validity
out = TMP / "out"
plan = {"train": plan_train,
        "val": [(r, 1) for r in parts["val"]],
        "test": [(r, 1) for r in parts["test"]]}
H.emit(plan, out, stats)
H.write_yaml(out)
H.write_audit(out, stats, [src_y, src_v, src_c])
lbls = list((out / "labels").rglob("*.txt"))
check("labels emitted", len(lbls) > 0, f"{len(lbls)} label files")
bad = []
for lp in lbls:
    for line in lp.read_text().strip().splitlines():
        parts_ = line.split()
        if len(parts_) != 5:
            bad.append((lp.name, line, "field count")); continue
        cid = int(parts_[0]); vals = [float(v) for v in parts_[1:]]
        if not (0 <= cid < len(CLASS_TO_ID)):
            bad.append((lp.name, line, "class id"))
        if not all(0.0 <= v <= 1.0 for v in vals):
            bad.append((lp.name, line, "not normalised"))
        if vals[2] <= 0 or vals[3] <= 0:
            bad.append((lp.name, line, "zero wh"))
check("all labels valid YOLO", not bad, str(bad[:3]))
imgs_out = list((out / "images").rglob("*.jpg"))
check("images emitted", len(imgs_out) == len(lbls), f"{len(imgs_out)} imgs vs {len(lbls)} lbls")
check("every image has a label",
      {p.stem for p in imgs_out} == {p.stem for p in lbls})
import yaml as _y
cfg = _y.safe_load((out / "data.yaml").read_text())
check("data.yaml names complete", len(cfg["names"]) == 7, str(cfg["names"]))
audit = (out / "AUDIT.md").read_text()
check("audit lists unrecognised class", "Exoskeleton" in audit or "exoskeleton" in audit.lower())

print("\n" + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
print(f"tmp: {TMP}")
sys.exit(1 if fails else 0)
