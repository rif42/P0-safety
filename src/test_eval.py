"""Tests for ppe_eval using hand-built predictions and ground truth."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppe_eval import evaluate_at, sweep, pick_operating_point, box_size_report

NAMES = ["person", "helmet", "no-helmet"]
def b(x, y, w, h): return np.array([x, y, x + w, y + h], dtype=float)

# 4 images. no-helmet GT: img1, img2, img3 (3 instances). helmet GT: img1, img4.
gt = {
    "img1": [(1, b(.10, .10, .10, .10)), (2, b(.50, .10, .10, .10))],
    "img2": [(2, b(.20, .20, .10, .10))],
    "img3": [(2, b(.30, .30, .10, .10))],
    "img4": [(1, b(.40, .40, .10, .10))],
}
# predictions: strong on img1/img2 no-helmet, weak on img3, plus one FP on img4
preds = {
    "img1": [(1, 0.90, b(.10, .10, .10, .10)), (2, 0.85, b(.50, .10, .10, .10))],
    "img2": [(2, 0.55, b(.20, .20, .10, .10))],
    "img3": [(2, 0.20, b(.30, .30, .10, .10))],
    "img4": [(1, 0.80, b(.40, .40, .10, .10)), (2, 0.45, b(.70, .70, .10, .10))],
}

fails = []
def check(label, got, want, extra=""):
    ok = (got == want) if not isinstance(want, float) else abs(got - want) < 1e-6
    print(f"{'PASS' if ok else 'FAIL'}  {label:<50} got={got}{'  ' + extra if extra else ''}")
    if not ok: fails.append(label)

m = {x.name: x for x in evaluate_at(preds, gt, NAMES, 0.50)}
nh = m["no-helmet"]
check("t=0.50 no-helmet tp", nh.tp, 2)          # img1 + img2
check("t=0.50 no-helmet fn", nh.fn, 1)          # img3 below threshold
check("t=0.50 no-helmet fp", nh.fp, 0)          # img4 FP at 0.45, below threshold
check("t=0.50 recall", round(nh.recall, 3), 0.667)
check("t=0.50 precision", nh.precision, 1.0)

m = {x.name: x for x in evaluate_at(preds, gt, NAMES, 0.40)}
nh = m["no-helmet"]
check("t=0.40 lowers precision", round(nh.precision, 3), 0.667, "fp from img4 now counts")
check("t=0.40 recall unchanged", round(nh.recall, 3), 0.667)

m = {x.name: x for x in evaluate_at(preds, gt, NAMES, 0.15)}
check("t=0.15 recall rises", round(m["no-helmet"].recall, 3), 1.0)
check("t=0.15 precision falls", round(m["no-helmet"].precision, 3), 0.75)

# a badly localised prediction must not count as a hit
bad = {"img2": [(2, 0.99, b(.80, .80, .10, .10))]}
m = {x.name: x for x in evaluate_at(bad, {"img2": gt["img2"]}, NAMES, 0.5)}
check("poor localisation is FP not TP", (m["no-helmet"].tp, m["no-helmet"].fp), (0, 1))

# duplicate detections on one GT box: one TP, one FP
dup = {"img2": [(2, 0.9, b(.20, .20, .10, .10)), (2, 0.8, b(.205, .205, .10, .10))]}
m = {x.name: x for x in evaluate_at(dup, {"img2": gt["img2"]}, NAMES, 0.5)}
check("duplicate det -> 1 TP 1 FP", (m["no-helmet"].tp, m["no-helmet"].fp), (1, 1))

df = sweep(preds, gt, NAMES, thresholds=[0.15, 0.40, 0.50, 0.90])
check("sweep row count", len(df), 4 * 3)
check("sweep monotone recall",
      list(df[df["class"] == "no-helmet"].sort_values("threshold")["recall"]) ==
      sorted(df[df["class"] == "no-helmet"]["recall"], reverse=True), True)

op = pick_operating_point(df, "no-helmet", min_precision=0.70)
check("operating point respects precision floor", op["precision"] >= 0.70, True, str(op))
check("operating point maximises recall subject to floor", op["threshold"], 0.15,
      "p=0.75 r=1.00 clears a 0.70 floor")
op2 = pick_operating_point(df, "no-helmet", min_precision=0.99)
check("strict floor -> t=0.50", op2["threshold"], 0.50, str(op2))
op3 = pick_operating_point(df, "no-helmet", min_precision=1.01)
check("impossible floor -> None (a finding, not a crash)", op3, None)

sz = box_size_report(gt, NAMES)
check("box size report covers classes", set(sz["class"]), {"helmet", "no-helmet"})
row = sz[sz["class"] == "no-helmet"].iloc[0]
check("10%x10% box (64px @640) NOT flagged small", bool(row["small_object"]), False,
      f"median_area_pct={row['median_area_pct']} px_at_640={row['median_px_at_640']}")

# boots at site-camera distance: ~2% of frame width -> 0.04% area, ~13px at 640
gt_small = {f"s{i}": [(1, b(.4, .4, .02, .02))] for i in range(5)}
sz2 = box_size_report(gt_small, NAMES)
r2 = sz2.iloc[0]
check("2%x2% box IS flagged small", bool(r2["small_object"]), True,
      f"median_area_pct={r2['median_area_pct']} px_at_640={r2['median_px_at_640']}")
check("small-object px estimate sane", round(float(r2["median_px_at_640"]), 1), 12.8)

print("\n" + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
