"""HI-VIS — presentation helpers for the demo page: drawing detection overlays,
building the exception-log rows, and CSV export. Kept free of any Streamlit
imports so it can be unit-tested (and reused by other pages) on its own.
"""

import io

from PIL import Image, ImageDraw, ImageOps
from PIL.ExifTags import TAGS

import detector

VERDICT_META = {
    "ok":   {"label": "COMPLIANT",     "bg": "#FFFFFF", "fg": "#141414", "border": "#141414"},
    "non":  {"label": "NON-COMPLIANT", "bg": "#EFE600", "fg": "#141414", "border": "#141414"},
    "none": {"label": "NOT ASSESSED",  "bg": "#E4E5E2", "fg": "#4A4B47", "border": "#9B9D97"},
}


def load_image(file_bytes):
    """Open + auto-rotate per EXIF orientation, so overlay boxes (computed on
    the rotated image) line up with what's actually displayed."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def exif_datetime(file_bytes):
    """Best-effort capture timestamp from EXIF. Returns None (never a
    fabricated value) when the photo carries no timestamp — most
    screenshots, downloads, and messaging-app re-saves strip it."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        exif = img.getexif()
        if exif:
            for tag_id in (306,):  # base IFD "DateTime"
                if tag_id in exif and exif[tag_id]:
                    return str(exif[tag_id])
            try:
                sub = exif.get_ifd(0x8769)  # Exif SubIFD
                for tag_id in (0x9003, 0x9004):  # DateTimeOriginal, DateTimeDigitized
                    if tag_id in sub and sub[tag_id]:
                        return str(sub[tag_id])
            except Exception:
                pass
    except Exception:
        pass
    return None


def _hex_to_rgba(hexstr, alpha):
    hexstr = hexstr.lstrip("#")
    r, g, b = int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16)
    return (r, g, b, alpha)


def _box_px(box, w, h):
    x1, y1, x2, y2 = box
    return (x1 * w, y1 * h, x2 * w, y2 * h)


def draw_overlay(image, persons, selected_idx=None, show_boxes=True):
    """Return a copy of `image` with person + item bounding boxes drawn on
    it. `persons` is the list from detector.assess()["persons"]. If
    selected_idx is set, every other person's boxes are dimmed."""
    base = image.convert("RGBA")
    if not show_boxes or not persons:
        return base.convert("RGB")
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(min(w, h) / 220))

    for i, p in enumerate(persons):
        dim = selected_idx is not None and selected_idx != i
        alpha = 70 if dim else 255
        color = _hex_to_rgba(detector.CLASS_META["person"]["color"], alpha)
        draw.rectangle(_box_px(p["box"], w, h), outline=color, width=line_width)

        for slot in ("hardhat", "vest"):
            st_ = p["status"][slot]
            if st_["state"] == "notvisible" or not st_.get("box"):
                continue
            key = st_["class_key"]
            item_color = _hex_to_rgba(detector.CLASS_META[key]["color"], alpha)
            draw.rectangle(_box_px(st_["box"], w, h), outline=item_color, width=max(1, line_width - 1))

    return Image.alpha_composite(base, overlay).convert("RGB")


def badge_for(verdict):
    return VERDICT_META.get(verdict, VERDICT_META["none"])


def build_rows(items, threshold, rule_text):
    """items: list of {"name", "datetime", "assessment"} — one per uploaded
    photo. Returns one row per assessed person per tracked item (hardhat,
    vest), covering both compliant and non-compliant findings, so the
    verdict filter in the UI can slice either view from the same table."""
    rows = []
    for item in items:
        assessment = item["assessment"]
        for p_idx, p in enumerate(assessment["persons"]):
            missing = [slot for slot in ("hardhat", "vest") if p["status"][slot]["state"] == "missing"]
            if missing:
                for slot in missing:
                    st_ = p["status"][slot]
                    label = detector.CLASS_META[st_["class_key"]]["label"]
                    rows.append({
                        "file": item["name"], "datetime": item["datetime"] or "—",
                        "person": f"Person {p_idx + 1}", "type": slot,
                        "finding": f"{label} — absence detected",
                        "confidence": round(st_["conf"], 2), "verdict": "non-compliant",
                        "threshold": round(threshold, 2), "rule_set": rule_text,
                    })
            else:
                rows.append({
                    "file": item["name"], "datetime": item["datetime"] or "—",
                    "person": f"Person {p_idx + 1}", "type": "—",
                    "finding": "none — required PPE present",
                    "confidence": None, "verdict": "compliant",
                    "threshold": round(threshold, 2), "rule_set": rule_text,
                })
    return rows


def rows_to_csv(rows):
    import csv as _csv
    buf = io.StringIO()
    fields = ["file", "datetime", "person", "type", "finding", "confidence", "verdict", "threshold", "rule_set"]
    writer = _csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
