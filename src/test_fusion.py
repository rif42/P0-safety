"""Synthetic-skeleton tests for ppe_fusion.

Builds COCO-17 skeletons at arbitrary position and scale, plus PPE boxes placed
either correctly on the body or deliberately wrong, and asserts the verdict for
every branch of the reconciliation table.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppe_fusion import (FusionConfig, Verdict, assess_frame, summarise,
                        compliance_rate, ITEMS)

IMG_H, IMG_W = 1080, 1920
CFG = FusionConfig()

# Proportions relative to torso length s (shoulder mid -> hip mid)
def skeleton(cx, sy, s, *, feet=True, face=True, torso=True, conf=0.9):
    """Return (kps (17,3), person_box). sy = shoulder-line y."""
    k = np.zeros((17, 3), dtype=float)
    def put(i, x, y, c=conf):
        k[i] = (x, y, c)
    if face:
        put(0,  cx,            sy - 0.35 * s)          # nose
        put(1,  cx - 0.06 * s, sy - 0.40 * s)          # L eye
        put(2,  cx + 0.06 * s, sy - 0.40 * s)          # R eye
        put(3,  cx - 0.11 * s, sy - 0.38 * s)          # L ear
        put(4,  cx + 0.11 * s, sy - 0.38 * s)          # R ear
    if torso:
        put(5,  cx - 0.22 * s, sy)                     # L shoulder
        put(6,  cx + 0.22 * s, sy)                     # R shoulder
        put(11, cx - 0.13 * s, sy + 1.00 * s)          # L hip
        put(12, cx + 0.13 * s, sy + 1.00 * s)          # R hip
    put(7,  cx - 0.26 * s, sy + 0.45 * s)
    put(8,  cx + 0.26 * s, sy + 0.45 * s)
    put(9,  cx - 0.28 * s, sy + 0.85 * s)              # L wrist
    put(10, cx + 0.28 * s, sy + 0.85 * s)              # R wrist
    put(13, cx - 0.13 * s, sy + 1.60 * s)
    put(14, cx + 0.13 * s, sy + 1.60 * s)
    if feet:
        put(15, cx - 0.13 * s, sy + 2.20 * s)          # L ankle
        put(16, cx + 0.13 * s, sy + 2.20 * s)          # R ankle
    ys = [v for v in k[:, 1] if v > 0]
    box = (cx - 0.35 * s, sy - 0.62 * s, cx + 0.35 * s,
           (sy + 2.30 * s) if feet else (sy + 1.70 * s))
    return k, box

def bx(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

def helmet_on(cx, sy, s):   return bx(cx, sy - 0.48 * s, 0.30 * s, 0.26 * s)
def helmet_in_hand(cx, sy, s): return bx(cx + 0.28 * s, sy + 0.88 * s, 0.26 * s, 0.24 * s)
def vest_on(cx, sy, s):     return bx(cx, sy + 0.45 * s, 0.40 * s, 0.80 * s)
def boots_on(cx, sy, s):
    return [bx(cx - 0.13 * s, sy + 2.28 * s, 0.18 * s, 0.14 * s),
            bx(cx + 0.13 * s, sy + 2.28 * s, 0.18 * s, 0.14 * s)]
def head_neg(cx, sy, s):    return bx(cx, sy - 0.38 * s, 0.28 * s, 0.30 * s)
def torso_neg(cx, sy, s):   return bx(cx, sy + 0.45 * s, 0.40 * s, 0.80 * s)
def feet_neg(cx, sy, s):
    return [bx(cx - 0.13 * s, sy + 2.26 * s, 0.18 * s, 0.14 * s),
            bx(cx + 0.13 * s, sy + 2.26 * s, 0.18 * s, 0.14 * s)]

fails = []
def check(label, got, want, extra=""):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<52} got={got}{'  ' + extra if extra else ''}")
    if not ok:
        fails.append(f"{label} (got {got}, want {want})")

def run(dets, kps, boxes, confs=None):
    kps = np.stack(kps); boxes = np.array(boxes, dtype=float)
    confs = np.array(confs if confs is not None else [0.9] * len(boxes))
    return assess_frame(dets, kps, boxes, confs, (IMG_H, IMG_W), CFG)

D = lambda c, b, k=0.9: {"cls": c, "conf": k, "box": b}

# ------------------------------------------------------------------ 1 compliant
cx, sy, s = 900.0, 300.0, 160.0
k, pb = skeleton(cx, sy, s)
dets = [D("helmet", helmet_on(cx, sy, s)), D("vest", vest_on(cx, sy, s))] + \
       [D("boots", b) for b in boots_on(cx, sy, s)]
a = run(dets, [k], [pb])[0]
check("compliant: helmet", a.findings["helmet"].verdict, Verdict.COMPLIANT,
      f"geo={a.findings['helmet'].geometry_score:.2f}")
check("compliant: vest", a.findings["vest"].verdict, Verdict.COMPLIANT,
      f"geo={a.findings['vest'].geometry_score:.2f}")
check("compliant: boots", a.findings["boots"].verdict, Verdict.COMPLIANT,
      f"geo={a.findings['boots'].geometry_score:.2f}")
check("compliant: person flagged compliant", a.compliant, True)
check("compliant: no violations", a.violations, [])

# ----------------------------------------------- 2 all three negatives detected
dets = [D("no-helmet", head_neg(cx, sy, s)), D("no-vest", torso_neg(cx, sy, s))] + \
       [D("no-boots", b) for b in feet_neg(cx, sy, s)]
a = run(dets, [k], [pb])[0]
for item in ITEMS:
    check(f"negatives: {item} -> VIOLATION", a.findings[item].verdict, Verdict.VIOLATION,
          f"conf={a.findings[item].confidence:.2f}")
check("negatives: all three reported", sorted(a.violations), ["boots", "helmet", "vest"])

# --------------------------------------- 3 helmet carried in hand, not worn
dets = [D("helmet", helmet_in_hand(cx, sy, s)), D("vest", vest_on(cx, sy, s))] + \
       [D("boots", b) for b in boots_on(cx, sy, s)]
a = run(dets, [k], [pb])[0]
check("carried helmet -> REVIEW_CARRIED", a.findings["helmet"].verdict,
      Verdict.REVIEW_CARRIED, f"geo={a.findings['helmet'].geometry_score:.2f}")
check("carried helmet: vest still compliant", a.findings["vest"].verdict, Verdict.COMPLIANT)

# --------------------- 4 helmet carried AND detector flags no-helmet -> VIOLATION
dets = [D("helmet", helmet_in_hand(cx, sy, s)), D("no-helmet", head_neg(cx, sy, s))]
a = run(dets, [k], [pb])[0]
check("carried + negative -> VIOLATION", a.findings["helmet"].verdict, Verdict.VIOLATION,
      f"conf={a.findings['helmet'].confidence:.2f}")

# ------------- 5 detector says no-helmet but helmet is anatomically in place
dets = [D("helmet", helmet_on(cx, sy, s)), D("no-helmet", head_neg(cx, sy, s))]
a = run(dets, [k], [pb])[0]
check("worn + negative -> REVIEW_DETECTOR_MISS", a.findings["helmet"].verdict,
      Verdict.REVIEW_DETECTOR_MISS)

# --------------------------------- 6 truncated at frame bottom -> feet unknown
cx2, s2 = 900.0, 300.0
sy2 = IMG_H - 2.25 * s2          # ankles fall below the frame
k2, pb2 = skeleton(cx2, sy2, s2, feet=False)
pb2 = (pb2[0], pb2[1], pb2[2], IMG_H - 1)
dets = [D("helmet", helmet_on(cx2, sy2, s2)), D("vest", vest_on(cx2, sy2, s2))]
a = run(dets, [k2], [pb2])[0]
check("truncated: boots -> INDETERMINATE", a.findings["boots"].verdict,
      Verdict.INDETERMINATE, a.findings["boots"].note)
check("truncated: helmet still judged", a.findings["helmet"].verdict, Verdict.COMPLIANT)
check("truncated: not counted as violation", "boots" in a.violations, False)
check("truncated flag set", a.truncated, True)

# ------------------------------- 7 no face keypoints -> head not observable
k3, pb3 = skeleton(cx, sy, s, face=False)
a = run([D("no-helmet", head_neg(cx, sy, s))], [k3], [pb3])[0]
check("no face kps: helmet -> INDETERMINATE", a.findings["helmet"].verdict,
      Verdict.INDETERMINATE, a.findings["helmet"].note)

# -------------- 8 bare region, nothing detected -> pose-only VIOLATION, low conf
a = run([], [k], [pb])[0]
check("bare person: helmet -> VIOLATION", a.findings["helmet"].verdict, Verdict.VIOLATION)
check("bare person: low confidence", a.findings["helmet"].confidence < 0.5, True,
      f"conf={a.findings['helmet'].confidence:.2f}")

# ------------------------------------------- 9 one boot only, both-feet rule on
dets = [D("boots", boots_on(cx, sy, s)[0])]
a = run(dets, [k], [pb])[0]
check("one boot only -> not compliant", a.findings["boots"].verdict != Verdict.COMPLIANT, True,
      a.findings["boots"].verdict.value)

# ------------------------- 10 two workers: helmet must go to the right person
cxA, cxB = 500.0, 1400.0
kA, pbA = skeleton(cxA, sy, s)
kB, pbB = skeleton(cxB, sy, s)
dets = [D("helmet", helmet_on(cxA, sy, s)), D("no-helmet", head_neg(cxB, sy, s))]
res = run(dets, [kA, kB], [pbA, pbB])
check("2 people: A compliant", res[0].findings["helmet"].verdict, Verdict.COMPLIANT)
check("2 people: B violation", res[1].findings["helmet"].verdict, Verdict.VIOLATION)

# ------------------------------------------ 11 scale invariance (near vs far)
verdicts = {}
for tag, (ccx, csy, cs) in {"near": (900.0, 200.0, 320.0),
                            "far":  (300.0, 400.0, 55.0)}.items():
    kk, bb = skeleton(ccx, csy, cs)
    dd = [D("helmet", helmet_on(ccx, csy, cs)), D("vest", vest_on(ccx, csy, cs))] + \
         [D("boots", b) for b in boots_on(ccx, csy, cs)]
    aa = run(dd, [kk], [bb])[0]
    verdicts[tag] = {i: aa.findings[i].verdict for i in ITEMS}
check("scale invariant near==far", verdicts["near"], verdicts["far"],
      f"near={ {k_: v.value for k_, v in verdicts['near'].items()} }")

# ------------------------------------- 12 low-confidence negative is filtered
a = run([D("no-helmet", head_neg(cx, sy, s), 0.20)], [k], [pb])[0]
check("negative below conf floor ignored", a.findings["helmet"].negative_flagged, False)

# ---------------------------------------------- 13 summary + honest denominator
res = run([D("helmet", helmet_on(cxA, sy, s)), D("no-helmet", head_neg(cxB, sy, s))],
          [kA, kB], [pbA, pbB])
summ = summarise(res)
check("summary counts 2 people", summ["people_assessed"], 2)
check("summary counts violations", summ["people_with_violation"], 2, "(B helmet, both bare feet/vest)")
check("helmet compliance rate 0.5", compliance_rate(res, "helmet"), 0.5)
k4, pb4 = skeleton(cx, sy, s, face=False)
check("no observable head -> rate is None not 1.0",
      compliance_rate(run([], [k4], [pb4]), "helmet"), None)

# ------------------------------------------------------- 14 row serialisation
row = res[0].to_row()
check("to_row has verdict columns",
      all(f"{i}_verdict" in row for i in ITEMS), True, str(sorted(row)[:6]))

print("\n" + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES:"))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
