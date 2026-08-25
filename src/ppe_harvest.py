#!/usr/bin/env python3
"""
ppe_harvest.py
==============
Download PPE datasets from Kaggle / HuggingFace, convert every annotation
dialect to a single YOLO label set, merge, deduplicate, split, oversample the
violation classes, and emit an Ultralytics data.yaml plus a full audit trail.

Everything here is deterministic. The companion prompt in prompts/ is for a
smaller model to DISCOVER and register new sources; conversion is never left to
a language model, because a mis-mapped class is invisible until the model is
already trained on it.

Usage
-----
    # commercial-safe sources only (default)
    python ppe_harvest.py --out /data/ppe_yolo26

    # include the non-commercial SH17 set for research work
    python ppe_harvest.py --out /data/ppe_yolo26 --include-noncommercial

    # fast smoke test, 200 images per source, no downloads if cached
    python ppe_harvest.py --out /tmp/ppe_smoke --limit 200

    # audit only - resolve class names against the taxonomy and stop
    python ppe_harvest.py --out /tmp/ppe_audit --audit-only

Outputs
-------
    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt
    <out>/data.yaml            Ultralytics dataset config
    <out>/audit.json           machine-readable provenance + class mapping
    <out>/AUDIT.md             human-readable report - READ THIS
    <out>/ATTRIBUTION.txt      licence credit lines you are obliged to keep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppe_taxonomy import (  # noqa: E402
    AUX_RESOLUTION,
    CANONICAL_CLASSES,
    CLASS_TO_ID,
    DROP,
    FOOT_AUX,
    HEAD_AUX,
    SOURCES,
    Source,
    VIOLATION_CLASSES,
    normalise,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 1337


# ==========================================================================
# Records
# ==========================================================================
@dataclass
class Box:
    name: str                 # upstream class name, verbatim
    x1: float
    y1: float
    x2: float
    y2: float
    canonical: Optional[str] = None   # filled in by mapping stage


@dataclass
class Record:
    image: Path
    width: int
    height: int
    boxes: List[Box] = field(default_factory=list)
    source: str = ""
    dhash: int = 0

    @property
    def canonical_names(self) -> List[str]:
        return [b.canonical for b in self.boxes if b.canonical]

    @property
    def has_violation(self) -> bool:
        return any(c in VIOLATION_CLASSES for c in self.canonical_names)


# ==========================================================================
# Download
# ==========================================================================
def download(src: Source, cache: Path) -> Optional[Path]:
    dest = cache / src.key
    marker = dest / ".complete"
    if marker.exists():
        print(f"  [cache] {src.key} -> {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    try:
        if src.kind == "kaggle":
            path = _download_kaggle(src.ident, dest)
        elif src.kind in ("hf_snapshot", "hf_datasets"):
            path = _download_hf(src.ident, dest)
        else:
            print(f"  [skip] unsupported source kind '{src.kind}' for {src.key}")
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {src.key}: {type(exc).__name__}: {exc}")
        return None

    if path is None:
        return None
    _extract_archives(dest)
    marker.write_text("ok")
    return dest


def _download_kaggle(slug: str, dest: Path) -> Optional[Path]:
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub not installed. `pip install kagglehub`. "
            "Credentials: set KAGGLE_USERNAME and KAGGLE_KEY, or place "
            "kaggle.json at ~/.config/kaggle/kaggle.json"
        )
    print(f"  [kaggle] {slug}")
    got = Path(kagglehub.dataset_download(slug))
    # kagglehub caches elsewhere; symlink so our cache layout stays uniform
    link = dest / "payload"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            shutil.rmtree(link)
    try:
        link.symlink_to(got, target_is_directory=True)
    except OSError:
        shutil.copytree(got, link, dirs_exist_ok=True)
    return dest


def _download_hf(repo_id: str, dest: Path) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. `pip install huggingface_hub`")
    print(f"  [hf] {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest / "payload"),
        max_workers=8,
    )
    return dest


def _extract_archives(root: Path) -> None:
    for p in list(root.rglob("*")):
        if not p.is_file():
            continue
        stem_dir = p.with_suffix("").with_suffix("") if p.name.endswith(".tar.gz") else p.with_suffix("")
        try:
            if p.suffix == ".zip" and zipfile.is_zipfile(p):
                if stem_dir.exists():
                    continue
                print(f"    unzip {p.name}")
                with zipfile.ZipFile(p) as z:
                    z.extractall(stem_dir)
            elif p.name.endswith((".tar", ".tar.gz", ".tgz")) and tarfile.is_tarfile(p):
                if stem_dir.exists():
                    continue
                print(f"    untar {p.name}")
                with tarfile.open(p) as t:
                    t.extractall(stem_dir, filter="data")
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] could not extract {p.name}: {exc}")


# ==========================================================================
# Format sniffing
# ==========================================================================
def sniff_format(root: Path) -> str:
    """Decide what annotation dialect actually landed on disk."""
    if any(root.rglob("*.coco.json")) or _find_coco_json(root):
        return "coco"
    if any(root.rglob("*.xml")):
        return "voc"
    if list(root.rglob("labels/**/*.txt")) or list(root.rglob("*/labels/*.txt")):
        return "yolo"
    if any(root.rglob("*.parquet")):
        return "parquet"
    if any(root.rglob("*.txt")):
        return "yolo"
    return "unknown"


def _find_coco_json(root: Path) -> List[Path]:
    out = []
    for p in root.rglob("*.json"):
        if p.stat().st_size > 200_000_000:
            continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                head = fh.read(4096)
        except OSError:
            continue
        if '"annotations"' in head and '"categories"' in head:
            out.append(p)
        elif '"images"' in head and '"categories"' in head:
            out.append(p)
    return out


def _image_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            idx.setdefault(p.name, p)
    return idx


def _dims(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:  # noqa: BLE001
        return None


# ==========================================================================
# Parsers - each yields Record objects with upstream class names intact
# ==========================================================================
def parse_yolo(root: Path, limit: Optional[int]) -> Tuple[List[Record], List[str]]:
    names = _yolo_class_names(root)
    if not names:
        print("    [warn] no data.yaml / classes.txt found; YOLO indices cannot be named")
        return [], []
    imgs = _image_index(root)
    recs: List[Record] = []
    label_files = [p for p in root.rglob("*.txt")
                   if p.name.lower() not in {"classes.txt", "readme.txt", "notes.txt", "requirements.txt"}]
    for lp in label_files:
        img = _match_image(lp, imgs)
        if img is None:
            continue
        d = _dims(img)
        if not d:
            continue
        w, h = d
        boxes: List[Box] = []
        for line in lp.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                ci = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if ci < 0 or ci >= len(names):
                continue
            boxes.append(Box(names[ci],
                             (cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h))
        if boxes:
            recs.append(Record(img, w, h, boxes))
        if limit and len(recs) >= limit:
            break
    return recs, names


def _yolo_class_names(root: Path) -> List[str]:
    import yaml
    for cand in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        try:
            data = yaml.safe_load(cand.read_text(errors="ignore"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and "names" in data:
            n = data["names"]
            if isinstance(n, dict):
                return [n[k] for k in sorted(n, key=lambda x: int(x))]
            if isinstance(n, list):
                return list(n)
    for cand in root.rglob("classes.txt"):
        return [l.strip() for l in cand.read_text(errors="ignore").splitlines() if l.strip()]
    return []


def _match_image(label_path: Path, imgs: Dict[str, Path]) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        hit = imgs.get(label_path.stem + ext)
        if hit:
            return hit
    return None


def parse_voc(root: Path, limit: Optional[int]) -> Tuple[List[Record], List[str]]:
    imgs = _image_index(root)
    recs: List[Record] = []
    seen: set = set()
    for xp in root.rglob("*.xml"):
        try:
            tree = ET.parse(xp)
        except ET.ParseError:
            continue
        r = tree.getroot()
        size = r.find("size")
        w = h = 0
        if size is not None:
            try:
                w = int(float(size.findtext("width", "0")))
                h = int(float(size.findtext("height", "0")))
            except ValueError:
                pass
        img = imgs.get(r.findtext("filename", "") or "") or _match_image(xp, imgs)
        if img is None:
            continue
        if not (w and h):
            d = _dims(img)
            if not d:
                continue
            w, h = d
        boxes: List[Box] = []
        for obj in r.iter("object"):
            nm = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if not nm or bb is None:
                continue
            try:
                boxes.append(Box(nm,
                                 float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                                 float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0"))))
            except ValueError:
                continue
        if boxes and img not in seen:
            seen.add(img)
            recs.append(Record(img, w, h, boxes))
        if limit and len(recs) >= limit:
            break
    names = sorted({b.name for r_ in recs for b in r_.boxes})
    return recs, names


def parse_coco(root: Path, limit: Optional[int]) -> Tuple[List[Record], List[str]]:
    imgs = _image_index(root)
    recs: List[Record] = []
    names: set = set()
    for jp in _find_coco_json(root):
        try:
            data = json.loads(jp.read_text(errors="ignore"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(data, dict) or "annotations" not in data:
            continue
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        names.update(cats.values())
        by_img: Dict[int, List[dict]] = defaultdict(list)
        for a in data["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in data.get("images", []):
            path = imgs.get(Path(im.get("file_name", "")).name)
            if path is None:
                continue
            w = int(im.get("width") or 0)
            h = int(im.get("height") or 0)
            if not (w and h):
                d = _dims(path)
                if not d:
                    continue
                w, h = d
            boxes: List[Box] = []
            for a in by_img.get(im["id"], []):
                bb = a.get("bbox")
                if not bb or len(bb) < 4:
                    continue
                x, y, bw, bh = (float(v) for v in bb[:4])
                nm = cats.get(a.get("category_id"))
                if not nm:
                    continue
                boxes.append(Box(nm, x, y, x + bw, y + bh))
            if boxes:
                recs.append(Record(path, w, h, boxes))
            if limit and len(recs) >= limit:
                break
        if limit and len(recs) >= limit:
            break
    return recs, sorted(names)


PARSERS = {"yolo": parse_yolo, "voc": parse_voc, "coco": parse_coco}


# ==========================================================================
# Mapping stage: upstream names -> canonical, with aux overlap subtraction
# ==========================================================================
def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    bb = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    denom = min(aa, bb)          # containment-style: a small helmet inside a head box counts
    return inter / denom if denom > 0 else 0.0


def map_records(recs: List[Record], src: Source, aux_iou: float,
                stats: dict) -> List[Record]:
    """Resolve every box to a canonical class; drop what we cannot use."""
    kept: List[Record] = []
    for r in recs:
        for b in r.boxes:
            b.canonical = src.resolve(b.name)
            if b.canonical == "__UNKNOWN__":
                stats["unknown_names"][b.name] += 1
            elif b.canonical is DROP:
                stats["dropped_by_design"][b.name] += 1

        # --- aux subtraction -------------------------------------------------
        for aux, (neg_class, cancelling_pos) in AUX_RESOLUTION.items():
            aux_boxes = [b for b in r.boxes if b.canonical == aux]
            if not aux_boxes:
                continue
            pos_boxes = [b for b in r.boxes if b.canonical == cancelling_pos]
            for ab in aux_boxes:
                if any(_iou(ab, pb) >= aux_iou for pb in pos_boxes):
                    ab.canonical = DROP            # helmeted head / shod foot
                    stats["aux_cancelled"][f"{aux}->{cancelling_pos}"] += 1
                else:
                    ab.canonical = neg_class
                    stats["aux_promoted"][f"{aux}->{neg_class}"] += 1

        r.boxes = [b for b in r.boxes
                   if b.canonical in CLASS_TO_ID]
        if r.boxes:
            r.source = src.key
            kept.append(r)
            for b in r.boxes:
                stats["instances"][b.canonical] += 1
    stats["images_kept"][src.key] = len(kept)
    return kept


# ==========================================================================
# Dedupe
# ==========================================================================
def dhash(path: Path, size: int = 8) -> int:
    """Difference hash. Cheap, and good enough to catch re-encodes and the
    near-identical consecutive frames that video-derived datasets are full of."""
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.int16)
        diff = a[:, 1:] > a[:, :-1]
        bits = 0
        for v in diff.flatten():
            bits = (bits << 1) | int(v)
        return bits
    except Exception:  # noqa: BLE001
        return 0


def dedupe(recs: List[Record], stats: dict) -> List[Record]:
    seen: Dict[int, Record] = {}
    out: List[Record] = []
    for r in recs:
        r.dhash = dhash(r.image)
        if r.dhash and r.dhash in seen:
            # keep whichever copy carries more violation evidence
            incumbent = seen[r.dhash]
            if r.has_violation and not incumbent.has_violation:
                out[out.index(incumbent)] = r
                seen[r.dhash] = r
            stats["duplicates_dropped"] += 1
            continue
        if r.dhash:
            seen[r.dhash] = r
        out.append(r)
    return out


# ==========================================================================
# Split - group near-duplicates together so val is not contaminated
# ==========================================================================
def split(recs: List[Record], ratios: Tuple[float, float, float]) -> Dict[str, List[Record]]:
    rng = random.Random(SEED)
    groups: Dict[Tuple[str, int], List[Record]] = defaultdict(list)
    for r in recs:
        # masking the low 16 bits clusters visually near-identical frames
        groups[(r.source, r.dhash >> 16)].append(r)
    keys = list(groups)
    rng.shuffle(keys)

    total = len(recs)
    want_val = int(total * ratios[1])
    want_test = int(total * ratios[2])
    out = {"train": [], "val": [], "test": []}
    for k in keys:
        g = groups[k]
        if len(out["val"]) < want_val:
            out["val"].extend(g)
        elif len(out["test"]) < want_test:
            out["test"].extend(g)
        else:
            out["train"].extend(g)
    return out


# ==========================================================================
# Violation oversampling - TRAIN SPLIT ONLY
# ==========================================================================
def oversample(train: List[Record], target_share: float, cap: int,
               stats: dict) -> List[Tuple[Record, int]]:
    """Return (record, n_copies). Only images containing a violation class are
    duplicated, and only in train - duplicating val or test would inflate the
    very recall figure this project is meant to report honestly."""
    if target_share <= 0:
        return [(r, 1) for r in train]
    inst = Counter()
    for r in train:
        for b in r.boxes:
            inst[b.canonical] += 1
    total = sum(inst.values()) or 1
    viol = sum(inst[c] for c in VIOLATION_CLASSES)
    share = viol / total
    stats["violation_share_before"] = round(share, 4)
    if share >= target_share or viol == 0:
        stats["oversample_factor"] = 1
        return [(r, 1) for r in train]
    # solve for k in: (viol*k) / (total - viol + viol*k) = target
    non = total - viol
    k = (target_share * non) / (viol * (1 - target_share))
    k = max(1, min(cap, int(round(k))))
    stats["oversample_factor"] = k
    plan = [(r, k if r.has_violation else 1) for r in train]
    new_viol = viol * k
    stats["violation_share_after"] = round(new_viol / (non + new_viol), 4)
    return plan


# ==========================================================================
# Emit
# ==========================================================================
def emit(plan: Dict[str, List[Tuple[Record, int]]], out: Path, stats: dict) -> None:
    for sp in ("train", "val", "test"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    for sp, items in plan.items():
        n = 0
        for rec, copies in items:
            for c in range(copies):
                suffix = "" if c == 0 else f"_ovs{c}"
                base = f"{rec.source}_{rec.image.stem}{suffix}"
                ip = out / "images" / sp / f"{base}{rec.image.suffix.lower()}"
                lp = out / "labels" / sp / f"{base}.txt"
                _link_or_copy(rec.image, ip)
                lp.write_text(_yolo_lines(rec))
                n += 1
        stats["emitted"][sp] = n


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # hardlink: zero extra disk for duplicates
    except OSError:
        shutil.copy2(src, dst)


def _yolo_lines(r: Record) -> str:
    lines = []
    for b in r.boxes:
        cid = CLASS_TO_ID[b.canonical]
        cx = ((b.x1 + b.x2) / 2) / r.width
        cy = ((b.y1 + b.y2) / 2) / r.height
        w = (b.x2 - b.x1) / r.width
        h = (b.y2 - b.y1) / r.height
        if w <= 0 or h <= 0:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(w, 1.0), min(h, 1.0)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + "\n"


def write_yaml(out: Path) -> None:
    import yaml
    cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(CANONICAL_CLASSES)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def write_audit(out: Path, stats: dict, used: List[Source]) -> None:
    (out / "audit.json").write_text(json.dumps(stats, indent=2, default=str))

    inst = stats["instances"]
    total = sum(inst.values()) or 1
    lines = [
        "# PPE dataset build audit", "",
        f"Classes: {len(CANONICAL_CLASSES)} - {', '.join(CANONICAL_CLASSES)}", "",
        "## Sources used", "",
        "| key | identifier | licence | commercial | images kept |",
        "|---|---|---|---|---|",
    ]
    for s in used:
        lines.append(f"| {s.key} | `{s.ident}` | {s.licence} | "
                     f"{'yes' if s.commercial_ok else '**NO**'} | "
                     f"{stats['images_kept'].get(s.key, 0)} |")

    lines += ["", "## Class balance", "",
              "| class | instances | share |", "|---|---|---|"]
    for c in CANONICAL_CLASSES:
        mark = " **(violation)**" if c in VIOLATION_CLASSES else ""
        lines.append(f"| {c}{mark} | {inst.get(c, 0)} | {inst.get(c, 0) / total:.1%} |")

    lines += ["", "## Splits", ""]
    for sp, n in stats["emitted"].items():
        lines.append(f"- {sp}: {n} images")
    lines += ["",
              f"- duplicates dropped: {stats['duplicates_dropped']}",
              f"- violation instance share before oversampling: {stats.get('violation_share_before')}",
              f"- violation instance share after oversampling: {stats.get('violation_share_after')}",
              f"- oversample factor applied to train: {stats.get('oversample_factor')}",
              ""]

    lines += ["## Head / foot overlap subtraction", "",
              "A `head` box becomes `no-helmet` only when no helmet box overlaps it "
              "(same for `foot` / `boots`). A near-zero cancellation count means the "
              "source annotated those classes exclusively; a large count means it did "
              "not, and the subtraction just saved you from labelling helmeted workers "
              "as violations.", ""]
    for k, v in stats["aux_promoted"].items():
        lines.append(f"- promoted {k}: {v}")
    for k, v in stats["aux_cancelled"].items():
        lines.append(f"- cancelled {k}: {v}")

    if stats["unknown_names"]:
        lines += ["", "## UNRECOGNISED upstream class names", "",
                  "These were in the data and matched nothing in the taxonomy, so they "
                  "were skipped. Add them to `ppe_taxonomy.ALIASES` or to the source's "
                  "`overrides` if they matter.", ""]
        for k, v in sorted(stats["unknown_names"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{k}` x{v}")

    if stats["dropped_by_design"]:
        lines += ["", "## Dropped by design", "",
                  ", ".join(f"`{k}`({v})" for k, v in
                            sorted(stats["dropped_by_design"].items(), key=lambda x: -x[1]))]

    lines += ["", "## Known limitations", "",
              "1. Near-duplicate leakage. Several sources are video-derived. Images are "
              "deduplicated by difference hash and near-duplicates are grouped into the "
              "same split, but visually similar frames may still straddle train and val, "
              "so val metrics are mildly optimistic. Hold back a genuinely independent "
              "site-camera set before you quote a number to a client.",
              "2. Footwear is the weak leg of the taxonomy. Negative footwear labels come "
              "from essentially one source. Expect `no-boots` recall to lag helmet and "
              "vest, and treat footwear findings as advisory until you have site data.",
              "3. Domain gap. These are mostly stock and web images. Real site CCTV is "
              "lower resolution, worse lit and shot from further away. Fine-tune on a few "
              "hundred of your own frames before deployment.",
              "4. `no-vest` and `no-helmet` conflate 'not wearing' with 'cannot be seen'. "
              "The pose fusion layer exists to sort those two apart at inference time.",
              ""]
    (out / "AUDIT.md").write_text("\n".join(lines))

    attrib = ["Attribution required by the licences of the datasets used:", ""]
    for s in used:
        attrib.append(f"- {s.key}: {s.ident} - {s.licence}")
        if s.attribution:
            attrib.append(f"    {s.attribution}")
    nc = [s for s in used if not s.commercial_ok]
    if nc:
        attrib += ["", "WARNING - NON-COMMERCIAL SOURCES INCLUDED:", ""]
        for s in nc:
            attrib.append(f"  {s.key} ({s.ident}) is {s.licence}.")
        attrib += ["", "A model trained on these weights inherits the restriction. "
                       "Do not use it in commercial delivery work."]
    (out / "ATTRIBUTION.txt").write_text("\n".join(attrib) + "\n")


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("~/.cache/ppe_sources").expanduser())
    ap.add_argument("--sources", nargs="*", default=None,
                    help="source keys to use; default = all commercial-safe")
    ap.add_argument("--include-noncommercial", action="store_true",
                    help="opt in to CC BY-NC sources (SH17). Taints the model licence.")
    ap.add_argument("--limit", type=int, default=None, help="max images per source")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--aux-iou", type=float, default=0.35,
                    help="containment threshold for head/helmet and foot/boots subtraction")
    ap.add_argument("--violation-target", type=float, default=0.35,
                    help="target share of instances belonging to violation classes (0 disables)")
    ap.add_argument("--oversample-cap", type=int, default=4)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    keys = args.sources or [k for k, s in SOURCES.items()
                            if s.commercial_ok or args.include_noncommercial]
    if not args.include_noncommercial:
        keys = [k for k in keys if SOURCES[k].commercial_ok]
    unknown = [k for k in keys if k not in SOURCES]
    if unknown:
        print(f"Unknown source keys: {unknown}. Known: {list(SOURCES)}")
        return 2

    stats = {
        "instances": Counter(), "unknown_names": Counter(), "dropped_by_design": Counter(),
        "aux_promoted": Counter(), "aux_cancelled": Counter(), "images_kept": {},
        "emitted": {}, "duplicates_dropped": 0, "per_source_upstream_names": {},
        "per_source_format": {},
    }

    all_recs: List[Record] = []
    used: List[Source] = []
    args.cache.mkdir(parents=True, exist_ok=True)

    for k in keys:
        src = SOURCES[k]
        print(f"\n=== {k} :: {src.ident} ({src.licence}) ===")
        if not src.commercial_ok:
            print("  !! NON-COMMERCIAL LICENCE - model weights inherit the restriction")
        root = download(src, args.cache)
        if root is None:
            continue
        fmt = sniff_format(root)
        stats["per_source_format"][k] = fmt
        print(f"  format on disk: {fmt} (registry hint: {src.ann_format})")
        parser = PARSERS.get(fmt)
        if parser is None:
            print(f"  [skip] no parser for '{fmt}'")
            continue
        recs, upstream_names = parser(root, args.limit)
        stats["per_source_upstream_names"][k] = upstream_names
        print(f"  parsed {len(recs)} images, upstream classes: {upstream_names}")
        mapped = map_records(recs, src, args.aux_iou, stats)
        print(f"  kept {len(mapped)} images after mapping")
        for nm in upstream_names:
            res = src.resolve(nm)
            tag = {"__UNKNOWN__": "UNRECOGNISED", None: "drop"}.get(res, res)
            print(f"    {nm:<24} -> {tag}")
        all_recs.extend(mapped)
        used.append(src)

    if not all_recs:
        print("\nNothing parsed. Check credentials and source availability.")
        args.out.mkdir(parents=True, exist_ok=True)
        write_audit(args.out, stats, used)
        return 1

    print(f"\nTotal images before dedupe: {len(all_recs)}")
    all_recs = dedupe(all_recs, stats)
    print(f"After dedupe: {len(all_recs)} (dropped {stats['duplicates_dropped']})")

    if args.audit_only:
        args.out.mkdir(parents=True, exist_ok=True)
        write_audit(args.out, stats, used)
        print(f"\nAudit written to {args.out/'AUDIT.md'} (no images emitted)")
        return 0

    train_r = 1.0 - args.val - args.test
    parts = split(all_recs, (train_r, args.val, args.test))
    print(f"Split: train={len(parts['train'])} val={len(parts['val'])} test={len(parts['test'])}")

    plan = {
        "train": oversample(parts["train"], args.violation_target, args.oversample_cap, stats),
        "val": [(r, 1) for r in parts["val"]],
        "test": [(r, 1) for r in parts["test"]],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    emit(plan, args.out, stats)
    write_yaml(args.out)
    write_audit(args.out, stats, used)

    print(f"\nDone. {args.out}")
    print(f"  data.yaml     -> {args.out/'data.yaml'}")
    print(f"  READ THIS     -> {args.out/'AUDIT.md'}")
    for sp, n in stats["emitted"].items():
        print(f"  {sp}: {n} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
