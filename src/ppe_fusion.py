"""
ppe_fusion.py
=============
Geometric fusion of a PPE detector with a COCO-17 pose model, turning boxes
into per-person compliance verdicts.

Why this layer exists
---------------------
A detector that outputs `no-helmet` is answering "is there an uncovered head in
this picture". A safety report needs to answer "is worker 3 wearing their hard
hat". Those differ in three ways that matter commercially:

  1. A helmet dangling from a wrist is a `helmet` detection and reads as
     compliant. Anatomically it is not on the head.
  2. A worker whose legs are out of frame has no observable feet. Reporting a
     footwear violation there is a false accusation, and one false accusation
     costs more trust than ten missed detections.
  3. Violations must attach to a person to be countable, actionable, or
     defensible in a safety meeting.

So every PPE box is assigned to a person skeleton and tested against the body
region it claims to protect. The detector's own negative classes and the
pose-derived judgement are then reconciled; where they disagree the case is
flagged for human review rather than silently resolved.

All thresholds are expressed as multiples of the subject's torso length, so the
same configuration works on a worker 4 m from the camera and one 40 m away.

The core functions take plain numpy arrays and have no Ultralytics dependency,
so they are unit-testable. `assess_frame_ultralytics` adapts Results objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# COCO-17 keypoint indices
# --------------------------------------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

FACE_KPS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
TORSO_KPS = (L_SHO, R_SHO, L_HIP, R_HIP)
ANKLE_KPS = (L_ANK, R_ANK)

REGION_OF = {"helmet": "head", "no-helmet": "head",
             "vest": "torso", "no-vest": "torso",
             "boots": "feet", "no-boots": "feet"}

ITEMS = ("helmet", "vest", "boots")
NEGATIVE_OF = {"helmet": "no-helmet", "vest": "no-vest", "boots": "no-boots"}


class Verdict(str, Enum):
    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    REVIEW_CARRIED = "REVIEW_carried_not_worn"
    REVIEW_DETECTOR_MISS = "REVIEW_detector_disagrees"
    INDETERMINATE = "INDETERMINATE_not_observable"


@dataclass
class FusionConfig:
    # keypoint confidence below which a joint is treated as unseen
    kp_conf: float = 0.35
    # detector confidence floor, applied per class group
    det_conf_ppe: float = 0.35
    det_conf_neg: float = 0.40          # negatives drive reports, so hold them higher
    det_conf_person: float = 0.30

    # head test
    head_radius_scale: float = 0.42     # x torso length
    helmet_above_nose_tol: float = 0.05 # x torso length; helmet centre must sit above nose

    # torso test
    vest_torso_overlap: float = 0.30    # intersection / torso-quad-bbox area

    # feet test
    foot_radius_scale: float = 0.30     # x torso length, around each visible ankle
    boot_below_ankle_tol: float = 0.12  # x torso length; boot may sit slightly above ankle

    # assignment
    assign_max_scale: float = 1.20      # x torso length, max anchor distance to claim a box

    # observability
    edge_margin_px: int = 8             # person box within this of frame edge = truncated
    require_both_feet: bool = True      # both visible ankles must be shod

    # fallbacks when hips are missing
    fallback_scale_from_box: float = 0.32   # x person box height


@dataclass
class ItemFinding:
    item: str
    verdict: Verdict
    confidence: float
    observable: bool
    positive_worn: bool
    positive_present: bool          # positive box assigned but failed the anatomy test
    negative_flagged: bool          # detector emitted the negative class for this person
    geometry_score: float
    note: str = ""


@dataclass
class PersonAssessment:
    person_id: int
    box: Tuple[float, float, float, float]
    person_conf: float
    torso_scale: float
    truncated: bool
    findings: Dict[str, ItemFinding] = field(default_factory=dict)

    @property
    def violations(self) -> List[str]:
        return [k for k, f in self.findings.items() if f.verdict == Verdict.VIOLATION]

    @property
    def reviews(self) -> List[str]:
        return [k for k, f in self.findings.items()
                if f.verdict in (Verdict.REVIEW_CARRIED, Verdict.REVIEW_DETECTOR_MISS)]

    @property
    def compliant(self) -> bool:
        vs = [f.verdict for f in self.findings.values()]
        return bool(vs) and all(v == Verdict.COMPLIANT for v in vs)

    def to_row(self) -> dict:
        row = {"person_id": self.person_id,
               "x1": round(self.box[0], 1), "y1": round(self.box[1], 1),
               "x2": round(self.box[2], 1), "y2": round(self.box[3], 1),
               "person_conf": round(self.person_conf, 3),
               "torso_px": round(self.torso_scale, 1),
               "truncated": self.truncated}
        for item in ITEMS:
            f = self.findings.get(item)
            row[f"{item}_verdict"] = f.verdict.value if f else Verdict.INDETERMINATE.value
            row[f"{item}_conf"] = round(f.confidence, 3) if f else 0.0
        row["violations"] = "|".join(self.violations)
        return row


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _vis(kps: np.ndarray, idx: int, cfg: FusionConfig) -> bool:
    """kps: (17, 3) array of x, y, conf."""
    return bool(kps[idx, 2] >= cfg.kp_conf and np.isfinite(kps[idx, 0]))


def _mid(kps: np.ndarray, a: int, b: int, cfg: FusionConfig) -> Optional[np.ndarray]:
    va, vb = _vis(kps, a, cfg), _vis(kps, b, cfg)
    if va and vb:
        return (kps[a, :2] + kps[b, :2]) / 2.0
    if va:
        return kps[a, :2].copy()
    if vb:
        return kps[b, :2].copy()
    return None


def torso_scale(kps: np.ndarray, box: Sequence[float], cfg: FusionConfig) -> float:
    """Scale reference in pixels. Every threshold is a multiple of this."""
    sho = _mid(kps, L_SHO, R_SHO, cfg)
    hip = _mid(kps, L_HIP, R_HIP, cfg)
    if sho is not None and hip is not None:
        d = float(np.linalg.norm(sho - hip))
        if d > 1.0:
            return d
    return max(1.0, cfg.fallback_scale_from_box * (box[3] - box[1]))


def head_anchor(kps: np.ndarray, cfg: FusionConfig, scale: float
                ) -> Optional[Tuple[np.ndarray, float]]:
    pts = [kps[i, :2] for i in FACE_KPS if _vis(kps, i, cfg)]
    if len(pts) < 2:
        return None
    centre = np.mean(np.stack(pts), axis=0)
    return centre, cfg.head_radius_scale * scale


def torso_quad(kps: np.ndarray, cfg: FusionConfig) -> Optional[np.ndarray]:
    order = (L_SHO, R_SHO, R_HIP, L_HIP)
    if sum(_vis(kps, i, cfg) for i in order) < 3:
        return None
    pts = np.array([kps[i, :2] for i in order if _vis(kps, i, cfg)], dtype=float)
    return pts


def ankle_anchors(kps: np.ndarray, cfg: FusionConfig, scale: float
                  ) -> List[Tuple[int, np.ndarray, float]]:
    out = []
    for i in ANKLE_KPS:
        if _vis(kps, i, cfg):
            out.append((i, kps[i, :2].copy(), cfg.foot_radius_scale * scale))
    return out


def _box_centre(b: Sequence[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _poly_bbox(pts: np.ndarray) -> Tuple[float, float, float, float]:
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _inter_area(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


# --------------------------------------------------------------------------
# Anatomical tests - each returns a score in [0, 1]
# --------------------------------------------------------------------------
def score_helmet_on_head(box: Sequence[float], kps: np.ndarray,
                         cfg: FusionConfig, scale: float) -> float:
    anc = head_anchor(kps, cfg, scale)
    if anc is None:
        return 0.0
    centre, radius = anc
    d = float(np.linalg.norm(_box_centre(box) - centre))
    prox = max(0.0, 1.0 - d / max(radius, 1e-6))
    if prox <= 0.0:
        return 0.0
    # A worn helmet sits on the crown: its centre is above the nose, and its top
    # edge is above the eye line. A helmet held at chest height fails both.
    above = 1.0
    if _vis(kps, NOSE, cfg):
        dy = kps[NOSE, 1] - _box_centre(box)[1]      # positive when box is higher
        above = 1.0 if dy > -cfg.helmet_above_nose_tol * scale else 0.0
    eyes = [kps[i, 1] for i in (L_EYE, R_EYE) if _vis(kps, i, cfg)]
    crown = 1.0
    if eyes:
        crown = 1.0 if box[1] <= float(np.mean(eyes)) else 0.0
    return prox * above * crown


def score_vest_on_torso(box: Sequence[float], kps: np.ndarray,
                        cfg: FusionConfig, scale: float) -> float:
    quad = torso_quad(kps, cfg)
    if quad is None:
        return 0.0
    tb = _poly_bbox(quad)
    tarea = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1]))
    cover = _inter_area(box, tb) / tarea
    return float(min(1.0, cover / max(cfg.vest_torso_overlap, 1e-6))) if cover > 0 else 0.0


def score_boot_on_foot(box: Sequence[float], ankle: np.ndarray,
                       cfg: FusionConfig, scale: float) -> float:
    radius = cfg.foot_radius_scale * scale
    c = _box_centre(box)
    d = float(np.linalg.norm(c - ankle))
    prox = max(0.0, 1.0 - d / max(radius, 1e-6))
    if prox <= 0.0:
        return 0.0
    # Footwear sits at or below the ankle joint, never well above it.
    below = 1.0 if c[1] >= ankle[1] - cfg.boot_below_ankle_tol * scale else 0.0
    return prox * below


# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------
def region_observable(region: str, kps: np.ndarray, box: Sequence[float],
                      img_hw: Tuple[int, int], cfg: FusionConfig) -> Tuple[bool, str]:
    h, w = img_hw
    if region == "head":
        n = sum(_vis(kps, i, cfg) for i in FACE_KPS)
        return (n >= 2, "" if n >= 2 else f"only {n} face keypoints visible")
    if region == "torso":
        n = sum(_vis(kps, i, cfg) for i in TORSO_KPS)
        return (n >= 3, "" if n >= 3 else f"only {n} torso keypoints visible")
    if region == "feet":
        n = sum(_vis(kps, i, cfg) for i in ANKLE_KPS)
        if n == 0:
            return False, "no ankle keypoints visible"
        if box[3] >= h - cfg.edge_margin_px:
            return False, "person truncated at bottom of frame"
        return True, ""
    return False, "unknown region"


def is_truncated(box: Sequence[float], img_hw: Tuple[int, int], cfg: FusionConfig) -> bool:
    h, w = img_hw
    m = cfg.edge_margin_px
    return bool(box[0] <= m or box[1] <= m or box[2] >= w - m or box[3] >= h - m)


# --------------------------------------------------------------------------
# Assignment of PPE boxes to people
# --------------------------------------------------------------------------
def assign_boxes(ppe: List[dict], people: List[dict], cfg: FusionConfig
                 ) -> Dict[int, List[dict]]:
    """Assign each PPE box to at most one person; a person may receive many boxes.

    This is deliberately NOT a bipartite matching. A worker needs a helmet and a
    vest and two boots simultaneously, so capping each person at one box would
    silently discard most of the evidence. Each box independently claims its
    nearest eligible person, which also means two workers can never both be
    credited with the same helmet.

    Cost is the distance from the box centre to the anchor of the body region
    that class relates to, divided by that person's torso length, so a distant
    worker's helmet is not stolen by a nearer one.
    """
    out: Dict[int, List[dict]] = {j: [] for j in range(len(people))}
    if not ppe or not people:
        return out

    for d in ppe:
        region = REGION_OF.get(d["cls"])
        c = _box_centre(d["box"])
        best_j, best_cost = -1, float("inf")

        for j, p in enumerate(people):
            kps, scale, pbox = p["kps"], p["scale"], p["box"]
            anchors: List[np.ndarray] = []
            if region == "head":
                a = head_anchor(kps, cfg, scale)
                if a is not None:
                    anchors.append(a[0])
            elif region == "torso":
                q = torso_quad(kps, cfg)
                if q is not None:
                    anchors.append(q.mean(axis=0))
            elif region == "feet":
                anchors.extend(pt for _, pt, _r in ankle_anchors(kps, cfg, scale))

            inside = bool(pbox[0] <= c[0] <= pbox[2] and pbox[1] <= c[1] <= pbox[3])

            if anchors:
                d_norm = min(float(np.linalg.norm(c - a)) / max(scale, 1e-6)
                             for a in anchors)
                if d_norm <= cfg.assign_max_scale:
                    cost = d_norm
                elif inside:
                    # A helmet hanging from a wrist is nowhere near the head anchor
                    # but is plainly this worker's. Claim it just inside the gate so
                    # the anatomical test can fail it and report 'carried, not worn'
                    # instead of the box vanishing and the case reading as 'absent'.
                    cost = 0.95 * cfg.assign_max_scale
                else:
                    continue
            else:
                if not inside:
                    continue
                cost = 0.90 * cfg.assign_max_scale

            if cost < best_cost:
                best_cost, best_j = cost, j

        if best_j >= 0:
            out[best_j].append(d)
    return out


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------
def _reconcile(item: str, observable: bool, obs_note: str,
               positive_worn: bool, positive_present: bool, negative_flagged: bool,
               geo: float, det_conf: float) -> ItemFinding:
    if not observable:
        return ItemFinding(item, Verdict.INDETERMINATE, 0.0, False,
                           positive_worn, positive_present, negative_flagged, geo,
                           obs_note or "body region not observable")

    if positive_worn and negative_flagged:
        return ItemFinding(item, Verdict.REVIEW_DETECTOR_MISS,
                           min(1.0, 0.40 + 0.30 * geo), True, True, positive_present,
                           negative_flagged, geo,
                           "detector flagged a violation but the item is anatomically "
                           "in place - most likely a detector false positive")

    if positive_worn:
        return ItemFinding(item, Verdict.COMPLIANT, min(1.0, 0.5 * det_conf + 0.5 * geo),
                           True, True, positive_present, negative_flagged, geo,
                           "equipment detected in the correct anatomical position")

    if positive_present and negative_flagged:
        return ItemFinding(item, Verdict.VIOLATION, min(1.0, 0.65 + 0.35 * det_conf),
                           True, False, True, True, geo,
                           "equipment is present but not worn and the detector "
                           "corroborates the violation - worker has the equipment "
                           "and is not using it")

    if positive_present:
        return ItemFinding(item, Verdict.REVIEW_CARRIED, min(1.0, 0.45 + 0.35 * det_conf),
                           True, False, True, negative_flagged, geo,
                           "equipment detected but not on the body - carried, hanging, "
                           "or belonging to another worker")

    if negative_flagged:
        return ItemFinding(item, Verdict.VIOLATION, min(1.0, 0.55 + 0.45 * det_conf),
                           True, False, False, True, geo,
                           "detector flagged the negative class and no equipment was "
                           "found in position - corroborated violation")

    # Region is observable, nothing positive in place, detector did not flag it
    # either. Absence over an observable region is still a violation, but at
    # lower confidence because it rests on one line of evidence.
    return ItemFinding(item, Verdict.VIOLATION, 0.45, True, False, False, False, geo,
                       "no equipment found over an observable body region "
                       "(pose-derived only, detector did not corroborate)")


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def assess_frame(detections: List[dict], keypoints: np.ndarray,
                 person_boxes: np.ndarray, person_confs: np.ndarray,
                 img_hw: Tuple[int, int], cfg: Optional[FusionConfig] = None
                 ) -> List[PersonAssessment]:
    """
    detections   : list of {'cls': canonical class name, 'conf': float,
                            'box': (x1, y1, x2, y2)} from the PPE detector.
                   'person' entries are ignored - people come from the pose model.
    keypoints    : (N, 17, 3) array of x, y, conf from the pose model.
    person_boxes : (N, 4) person boxes from the pose model.
    person_confs : (N,) person confidences.
    img_hw       : (height, width) of the frame.
    """
    cfg = cfg or FusionConfig()
    keypoints = np.asarray(keypoints, dtype=float).reshape(-1, 17, 3)
    person_boxes = np.asarray(person_boxes, dtype=float).reshape(-1, 4)
    person_confs = np.asarray(person_confs, dtype=float).reshape(-1)

    people = []
    for i in range(len(person_boxes)):
        if person_confs[i] < cfg.det_conf_person:
            continue
        kps = keypoints[i]
        people.append({"idx": i, "box": person_boxes[i], "conf": float(person_confs[i]),
                       "kps": kps, "scale": torso_scale(kps, person_boxes[i], cfg)})

    ppe = []
    for d in detections:
        cls = d["cls"]
        if cls not in REGION_OF:
            continue
        floor = cfg.det_conf_neg if cls.startswith("no-") else cfg.det_conf_ppe
        if float(d["conf"]) < floor:
            continue
        ppe.append({"cls": cls, "conf": float(d["conf"]),
                    "box": np.asarray(d["box"], dtype=float)})

    assigned = assign_boxes(ppe, people, cfg)

    out: List[PersonAssessment] = []
    for j, p in enumerate(people):
        kps, scale, box = p["kps"], p["scale"], p["box"]
        mine = assigned.get(j, [])
        pa = PersonAssessment(person_id=p["idx"],
                              box=tuple(float(v) for v in box),
                              person_conf=p["conf"],
                              torso_scale=scale,
                              truncated=is_truncated(box, img_hw, cfg))

        for item in ITEMS:
            region = REGION_OF[item]
            observable, note = region_observable(region, kps, box, img_hw, cfg)
            pos = [d for d in mine if d["cls"] == item]
            neg = [d for d in mine if d["cls"] == NEGATIVE_OF[item]]

            geo, worn, best_conf, partial_note = 0.0, False, 0.0, ""
            if item == "helmet":
                for d in pos:
                    s = score_helmet_on_head(d["box"], kps, cfg, scale)
                    if s > geo:
                        geo, best_conf = s, d["conf"]
                worn = geo > 0.0
            elif item == "vest":
                for d in pos:
                    s = score_vest_on_torso(d["box"], kps, cfg, scale)
                    if s > geo:
                        geo, best_conf = s, d["conf"]
                worn = geo > 0.0
            else:  # boots - each box claims its nearest ankle, exclusively
                anchors = ankle_anchors(kps, cfg, scale)
                # A standing worker's ankles sit closer together than the search
                # radius, so without exclusive claiming one boot would satisfy both
                # feet and a single-booted worker would read as compliant.
                claims: Dict[int, List[dict]] = {i: [] for i, _pt, _r in anchors}
                for d in pos:
                    c = _box_centre(d["box"])
                    nearest, nd = -1, float("inf")
                    for i, pt, _r in anchors:
                        dd = float(np.linalg.norm(c - pt))
                        if dd < nd:
                            nd, nearest = dd, i
                    if nearest >= 0:
                        claims[nearest].append(d)
                per_foot = []
                for i, pt, _r in anchors:
                    best = 0.0
                    for d in claims[i]:
                        sc = score_boot_on_foot(d["box"], pt, cfg, scale)
                        if sc > best:
                            best = sc
                            best_conf = max(best_conf, d["conf"])
                    per_foot.append(best)
                if per_foot:
                    geo = float(np.mean(per_foot))
                    worn = (all(s > 0.0 for s in per_foot) if cfg.require_both_feet
                            else any(s > 0.0 for s in per_foot))
                    if not worn and any(s > 0.0 for s in per_foot):
                        partial_note = ("only one foot has identifiable safety "
                                        "footwear - the other may be occluded")

            if not pos:
                best_conf = max([d["conf"] for d in neg], default=0.0)
            elif neg:
                best_conf = max(best_conf, max(d["conf"] for d in neg))

            finding = _reconcile(
                item, observable, note, worn, bool(pos), bool(neg), geo, best_conf)
            if observable and partial_note:
                finding.note = f"{partial_note}; {finding.note}"
            pa.findings[item] = finding

        out.append(pa)
    return out


# --------------------------------------------------------------------------
# Ultralytics adapter
# --------------------------------------------------------------------------
def assess_frame_ultralytics(det_result, pose_result, class_names: Sequence[str],
                             cfg: Optional[FusionConfig] = None) -> List[PersonAssessment]:
    """Adapt a pair of Ultralytics Results objects for the same frame."""
    cfg = cfg or FusionConfig()
    detections = []
    if det_result is not None and det_result.boxes is not None:
        b = det_result.boxes
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for k in range(len(cls)):
            name = class_names[cls[k]] if cls[k] < len(class_names) else str(cls[k])
            detections.append({"cls": name, "conf": float(conf[k]), "box": xyxy[k]})

    if (pose_result is None or pose_result.keypoints is None
            or pose_result.keypoints.data is None or len(pose_result.keypoints.data) == 0):
        return []

    kp = pose_result.keypoints.data.cpu().numpy()          # (N, 17, 3)
    pb = pose_result.boxes.xyxy.cpu().numpy()
    pc = pose_result.boxes.conf.cpu().numpy()
    h, w = pose_result.orig_shape
    return assess_frame(detections, kp, pb, pc, (int(h), int(w)), cfg)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarise(assessments: List[PersonAssessment]) -> dict:
    tally = {item: {v.value: 0 for v in Verdict} for item in ITEMS}
    for a in assessments:
        for item, f in a.findings.items():
            tally[item][f.verdict.value] += 1
    n = len(assessments)
    viol_people = sum(1 for a in assessments if a.violations)
    return {
        "people_assessed": n,
        "people_with_violation": viol_people,
        "people_needing_review": sum(1 for a in assessments if a.reviews),
        "violation_rate": round(viol_people / n, 3) if n else 0.0,
        "by_item": tally,
    }


def compliance_rate(assessments: List[PersonAssessment], item: str) -> Optional[float]:
    """Compliance among people where the item was actually observable.

    Returns None when nothing was observable, which is the honest answer and
    stops an empty denominator being reported as 100 per cent compliance.
    """
    dec = [a.findings[item] for a in assessments
           if item in a.findings and a.findings[item].observable]
    if not dec:
        return None
    ok = sum(1 for f in dec if f.verdict == Verdict.COMPLIANT)
    return round(ok / len(dec), 3)


PALETTE = {
    Verdict.COMPLIANT: (0, 170, 0),
    Verdict.VIOLATION: (0, 0, 220),
    Verdict.REVIEW_CARRIED: (0, 170, 255),
    Verdict.REVIEW_DETECTOR_MISS: (0, 170, 255),
    Verdict.INDETERMINATE: (140, 140, 140),
}


def annotate(frame: np.ndarray, assessments: List[PersonAssessment]) -> np.ndarray:
    """Draw per-person verdicts. Requires cv2."""
    import cv2
    out = frame.copy()
    for a in assessments:
        x1, y1, x2, y2 = (int(v) for v in a.box)
        worst = Verdict.COMPLIANT
        for f in a.findings.values():
            if f.verdict == Verdict.VIOLATION:
                worst = Verdict.VIOLATION
                break
            if f.verdict in (Verdict.REVIEW_CARRIED, Verdict.REVIEW_DETECTOR_MISS):
                worst = f.verdict
            elif f.verdict == Verdict.INDETERMINATE and worst == Verdict.COMPLIANT:
                worst = Verdict.INDETERMINATE
        colour = PALETTE[worst]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        lines = [f"#{a.person_id}"]
        for item in ITEMS:
            f = a.findings.get(item)
            if not f:
                continue
            tag = {Verdict.COMPLIANT: "ok", Verdict.VIOLATION: "VIOL",
                   Verdict.REVIEW_CARRIED: "carried?",
                   Verdict.REVIEW_DETECTOR_MISS: "check",
                   Verdict.INDETERMINATE: "n/a"}[f.verdict]
            lines.append(f"{item}:{tag}")
        y = max(14, y1 - 6)
        for i, txt in enumerate(lines):
            cv2.putText(out, txt, (x1, y + i * 15 - (len(lines) - 1) * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


__all__ = ["FusionConfig", "Verdict", "ItemFinding", "PersonAssessment",
           "assess_frame", "assess_frame_ultralytics", "summarise",
           "compliance_rate", "annotate", "ITEMS", "NEGATIVE_OF", "REGION_OF"]
