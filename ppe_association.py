"""Geometric association of PPE detections to person detections.

Turns a flat list of YOLO detections into a per-person compliance verdict by
asking, for each person box, whether a plausible PPE box sits in the region of
that person where the item ought to be.

Design notes
------------
* **Containment, not IoU.** A hardhat box is roughly 1-3% of a person box by
  area, so the IoU between a correctly-worn hardhat and its wearer is about
  0.02. Thresholding on IoU rejects every true match. We instead use
  *containment*: the fraction of the smaller (PPE) box that falls inside the
  target region.
* **Zones, not whole-person boxes.** A hardhat resting on a workbench can be
  fully contained in a person box. Restricting the test to the head band of the
  person box removes most of those.
* **One-to-one assignment.** In a crowd, one hardhat can be contained in two
  overlapping person boxes. Candidates are scored and assigned greedily,
  highest score first, so each item is consumed once.
* **UNKNOWN is a first-class verdict.** A worker who is 40px tall, truncated at
  the frame edge, or crouching cannot be judged. Reporting NON_COMPLIANT in
  those cases is how a safety system loses the trust of its users.

Coordinates are absolute pixels in xyxy order, matching Ultralytics'
``result.boxes.xyxy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """A single YOLO detection in absolute pixel xyxy coordinates."""

    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def intersection_area(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def containment(inner: Detection, outer: Detection) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``.

    This is *not* IoU. IoU is symmetric and collapses to near-zero whenever the
    two boxes differ wildly in scale, which is exactly the PPE case.
    """
    if inner.area <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / inner.area


# ---------------------------------------------------------------------------
# Zone model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Zone:
    """Where on a person a given PPE item is expected to appear.

    ``top`` and ``bottom`` are fractions of the person box height, measured down
    from the top edge. ``top`` may be negative, which extends the zone *above*
    the person box: person detectors routinely clip at the hairline while the
    hardhat sits proud of it.

    ``min_rel_width`` / ``max_rel_width`` are sanity bounds on the item's width
    as a fraction of the person's width. They reject a distant lorry's high-vis
    panel being matched to a foreground worker.
    """

    top: float
    bottom: float
    min_rel_width: float
    max_rel_width: float

    def box_for(self, person: Detection) -> Detection:
        h = person.height
        return Detection(
            label="_zone",
            confidence=1.0,
            x1=person.x1,
            y1=person.y1 + self.top * h,
            x2=person.x2,
            y2=person.y1 + self.bottom * h,
        )


# Tuned against upright, roughly front-facing workers at >120px tall.
DEFAULT_ZONES: dict[str, Zone] = {
    "hardhat": Zone(top=-0.12, bottom=0.30, min_rel_width=0.20, max_rel_width=1.10),
    "vest": Zone(top=0.12, bottom=0.68, min_rel_width=0.40, max_rel_width=1.30),
}


# Datasets disagree on naming. Map raw class names onto canonical item keys.
DEFAULT_LABEL_MAP: dict[str, str] = {
    "person": "person",
    "worker": "person",
    "hardhat": "hardhat",
    "hard-hat": "hardhat",
    "helmet": "hardhat",
    "safetyhelmet": "hardhat",
    "safety vest": "vest",
    "safety-vest": "vest",
    "vest": "vest",
    "reflective-vest": "vest",
}


class Status(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Config:
    # Minimum fraction of the PPE box that must fall inside the zone.
    min_containment: float = 0.55
    # Detections below these confidences are discarded before association.
    min_person_confidence: float = 0.40
    min_item_confidence: float = 0.35
    # Below this height a person is too small to judge reliably.
    min_person_height_px: float = 90.0
    # Above this width/height ratio the person is probably crouching, bending or
    # sitting, and the vertical zone model no longer holds.
    max_person_aspect: float = 0.85
    # Distance (px) from the frame edge within which a box counts as truncated.
    edge_margin_px: float = 3.0
    # Score weights: zone containment, centre proximity, detector confidence.
    w_containment: float = 0.60
    w_proximity: float = 0.25
    w_confidence: float = 0.15


@dataclass
class ItemVerdict:
    item: str
    status: Status
    matched: Detection | None = None
    score: float = 0.0
    reason: str = ""


@dataclass
class PersonAssessment:
    person: Detection
    verdicts: dict[str, ItemVerdict] = field(default_factory=dict)

    @property
    def overall(self) -> Status:
        states = {v.status for v in self.verdicts.values()}
        if Status.NON_COMPLIANT in states:
            return Status.NON_COMPLIANT
        if Status.UNKNOWN in states:
            return Status.UNKNOWN
        return Status.COMPLIANT

    def explain(self) -> str:
        parts = []
        for item, v in sorted(self.verdicts.items()):
            detail = f"{v.score:.2f}" if v.matched else v.reason
            parts.append(f"{item}={v.status.value}({detail})")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Scoring and assignment
# ---------------------------------------------------------------------------


def _match_score(
    person: Detection, item: Detection, zone: Zone, cfg: Config
) -> float | None:
    """Score a candidate (person, item) pairing, or None if implausible."""
    zone_box = zone.box_for(person)

    c = containment(item, zone_box)
    if c < cfg.min_containment:
        return None

    if person.width <= 0:
        return None
    rel_w = item.width / person.width
    if not (zone.min_rel_width <= rel_w <= zone.max_rel_width):
        return None

    # Horizontal proximity of the item centre to the person's midline,
    # normalised by half the person's width. Vertical position is already
    # constrained by the zone, so weighting it again double-counts.
    ix, _ = item.centre
    px, _ = person.centre
    half_w = max(person.width / 2.0, 1e-6)
    proximity = max(0.0, 1.0 - abs(ix - px) / half_w)

    return (
        cfg.w_containment * c
        + cfg.w_proximity * proximity
        + cfg.w_confidence * item.confidence
    )


def _assign_greedy(
    people: Sequence[Detection],
    items: Sequence[Detection],
    zone: Zone,
    cfg: Config,
) -> dict[int, tuple[int, float]]:
    """Greedy one-to-one assignment, highest-scoring pair first.

    Returns ``{person_index: (item_index, score)}``.

    Greedy is used rather than the Hungarian algorithm because the candidate
    graph here is sparse and near-diagonal — the zone and size gates have
    already removed nearly all ambiguity — and it avoids a SciPy dependency.
    To swap in the optimal assignment, build a cost matrix of ``-score`` and
    call ``scipy.optimize.linear_sum_assignment``.
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, person in enumerate(people):
        for ii, item in enumerate(items):
            s = _match_score(person, item, zone, cfg)
            if s is not None:
                candidates.append((s, pi, ii))

    candidates.sort(reverse=True)

    assigned: dict[int, tuple[int, float]] = {}
    used_items: set[int] = set()
    for s, pi, ii in candidates:
        if pi in assigned or ii in used_items:
            continue
        assigned[pi] = (ii, s)
        used_items.add(ii)
    return assigned


# ---------------------------------------------------------------------------
# Judgement
# ---------------------------------------------------------------------------


def _unjudgeable(
    person: Detection, item_key: str, zone: Zone, frame: tuple[float, float], cfg: Config
) -> str | None:
    """Return a reason string if this person cannot be judged for this item."""
    frame_w, frame_h = frame

    if person.height < cfg.min_person_height_px:
        return "person too small"

    if person.height > 0 and person.width / person.height > cfg.max_person_aspect:
        return "non-upright pose"

    zone_box = zone.box_for(person)
    m = cfg.edge_margin_px
    if zone_box.y1 < m or zone_box.y2 > frame_h - m:
        return "zone outside frame"
    if person.x1 < m or person.x2 > frame_w - m:
        return "person truncated at frame edge"

    return None


def assess(
    detections: Iterable[Detection],
    frame_size: tuple[float, float],
    required: Sequence[str] = ("hardhat", "vest"),
    zones: dict[str, Zone] | None = None,
    label_map: dict[str, str] | None = None,
    cfg: Config | None = None,
) -> list[PersonAssessment]:
    """Assess PPE compliance for every person in a frame.

    Parameters
    ----------
    detections
        Flat list of detections for one frame.
    frame_size
        ``(width, height)`` of the source image, in pixels.
    required
        Canonical item keys every person must be wearing.
    """
    cfg = cfg or Config()
    zones = zones or DEFAULT_ZONES
    label_map = label_map or DEFAULT_LABEL_MAP

    canonical: list[tuple[str, Detection]] = []
    for d in detections:
        key = label_map.get(d.label.strip().lower())
        if key is not None:
            canonical.append((key, d))

    people = [
        d for k, d in canonical if k == "person" and d.confidence >= cfg.min_person_confidence
    ]
    assessments = [PersonAssessment(person=p) for p in people]

    for item_key in required:
        zone = zones[item_key]
        items = [
            d
            for k, d in canonical
            if k == item_key and d.confidence >= cfg.min_item_confidence
        ]
        assigned = _assign_greedy(people, items, zone, cfg)

        for pi, person in enumerate(people):
            match = assigned.get(pi)
            if match is not None:
                ii, score = match
                assessments[pi].verdicts[item_key] = ItemVerdict(
                    item=item_key,
                    status=Status.COMPLIANT,
                    matched=items[ii],
                    score=score,
                )
                continue

            # No match. Before calling it a breach, check we could have seen it.
            reason = _unjudgeable(person, item_key, zone, frame_size, cfg)
            assessments[pi].verdicts[item_key] = ItemVerdict(
                item=item_key,
                status=Status.UNKNOWN if reason else Status.NON_COMPLIANT,
                reason=reason or "no candidate in zone",
            )

    return assessments


# ---------------------------------------------------------------------------
# Ultralytics adapter
# ---------------------------------------------------------------------------


def from_ultralytics(result) -> list[Detection]:
    """Convert one Ultralytics ``Results`` object into ``Detection`` records."""
    names = result.names
    out: list[Detection] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(
            Detection(
                label=names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return out


def from_ultralytics_multi(*results, dedupe_iou: float | None = None) -> list[Detection]:
    """Merge detections from several Ultralytics ``Results`` objects on the same frame.

    Use this when no single model's classes cover both "person" and every PPE
    item — e.g. a PPE model fine-tuned on a dataset with no person class,
    run alongside a stock/COCO-pretrained model for the person boxes.

    Pass ``dedupe_iou`` (e.g. ``0.5``) when two sources can emit boxes for the
    same class — e.g. a detection model and a pose model both finding
    "person" — so one real person doesn't become two ``PersonAssessment``
    rows. Per-class NMS keeps the highest-confidence box of each overlapping
    cluster and drops the rest.
    """
    out: list[Detection] = []
    for result in results:
        out.extend(from_ultralytics(result))
    if dedupe_iou is not None:
        out = _nms_by_class(out, dedupe_iou)
    return out


def _nms_by_class(detections: list[Detection], iou_thres: float) -> list[Detection]:
    # ponytail: torch import lives here, not at module level, so ppe_association
    # stays importable (and testable) without torch/torchvision installed unless
    # dedupe is actually used.
    import torch
    from torchvision.ops import nms

    by_label: dict[str, list[Detection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)

    out: list[Detection] = []
    for group in by_label.values():
        boxes = torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in group], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in group], dtype=torch.float32)
        keep = nms(boxes, scores, iou_thres)
        out.extend(group[i] for i in keep.tolist())
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FRAME = (1280.0, 720.0)

    dets = [
        # Worker A: hardhat and vest, both correctly placed.
        Detection("person", 0.94, 200, 120, 320, 600),
        Detection("hardhat", 0.88, 232, 108, 292, 152),
        Detection("safety vest", 0.81, 206, 210, 316, 380),
        # Worker B: vest only. Should read NON_COMPLIANT on hardhat.
        Detection("person", 0.91, 500, 140, 615, 610),
        Detection("vest", 0.77, 508, 230, 610, 395),
        # Worker C: bare, but truncated at the right edge -> UNKNOWN.
        Detection("person", 0.86, 1150, 150, 1279, 640),
        # A hardhat on a bench, nowhere near anyone's head zone.
        Detection("hardhat", 0.72, 800, 560, 858, 600),
        # Worker D: far away and tiny -> UNKNOWN, not a breach.
        Detection("person", 0.55, 900, 300, 928, 372),
    ]

    for a in assess(dets, FRAME):
        p = a.person
        print(
            f"person @({p.x1:.0f},{p.y1:.0f},{p.x2:.0f},{p.y2:.0f}) "
            f"-> {a.overall.value:<14} {a.explain()}"
        )
