"""HI-VIS — presentation helpers for the demo page: drawing detection overlays,
building the exception-log rows, and CSV export. Kept free of any Streamlit
imports so it can be unit-tested (and reused by other pages) on its own.
"""

import base64
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

        for slot in p["status"]:
            st_ = p["status"][slot]
            if st_["state"] == "notvisible" or not st_.get("box"):
                continue
            key = st_["class_key"]
            item_color = _hex_to_rgba(detector.CLASS_META[key]["color"], alpha)
            draw.rectangle(_box_px(st_["box"], w, h), outline=item_color, width=max(1, line_width - 1))

    return Image.alpha_composite(base, overlay).convert("RGB")


def badge_for(verdict):
    return VERDICT_META.get(verdict, VERDICT_META["none"])


def icon_svg(kind, color, size=12, stroke=3):
    paths = {
        "check": '<path d="M4 12.5l5 5L20 6.5"/>',
        "cross": '<path d="M5 5l14 14M19 5L5 19"/>',
        "warn": '<path d="M12 3L2 21h20L12 3z"/><path d="M12 10v5"/>',
    }
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" style="vertical-align:-2px;flex:none">'
            f'{paths.get(kind, "")}</svg>')


def verdict_badge(verdict):
    meta = badge_for(verdict)
    kind = "check" if verdict == "ok" else "warn" if verdict == "non" else "cross"
    icon = icon_svg(kind, meta["fg"], 11, 2.6)
    return (f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:10.5px;'
            f'font-weight:700;letter-spacing:.5px;padding:2px 7px;background:{meta["bg"]};'
            f'color:{meta["fg"]};border:1px solid {meta["border"]};white-space:nowrap">'
            f'{icon}{meta["label"]}</span>')


def b64_image(img, max_dim=480, quality=82):
    im = img.copy()
    im.thumbnail((max_dim, max_dim))
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def flag_confidence(assessment, required):
    """The confidence number shown on a result tile — always the same 0-1
    metric the DETECTION CONFIDENCE THRESHOLD slider itself filters on, so a
    tile is never left blank just because nothing was flagged:
      - non-compliant: the strongest violation's confidence (what drove the verdict)
      - compliant: the strongest person-detection confidence in the photo
      - not assessed: the best person confidence found, even below threshold
      - genuinely nothing detected at all: None (the only real "—" case)
    """
    confs = [p["status"][slot]["conf"] for p in assessment["persons"] for slot in required
             if p["status"][slot]["state"] == "missing"]
    if confs:
        return max(confs)
    person_confs = [p["conf"] for p in assessment["persons"]]
    if person_confs:
        return max(person_confs)
    return assessment.get("best_person_conf")


# ---------------------------------------------------------------------------
# shared page chrome — style block + header banner, used by every page so
# they all look like one app instead of drifting apart. No Streamlit import
# needed here: callers do st.markdown(vh.HV_STYLE_CSS, unsafe_allow_html=True).
# ---------------------------------------------------------------------------

HV_STYLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:#E4E5E2; }
#MainMenu { visibility:hidden; }
.block-container { max-width:1400px; }
.hv-mono { font-family:'IBM Plex Mono',monospace; }
.hv-h1 { font-family:'Barlow Condensed',sans-serif; font-weight:800; letter-spacing:.5px; color:#141414; }
@keyframes hvspin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
@keyframes hvstripe { from { background-position:0 0; } to { background-position:28px 0; } }
[data-testid="stFileUploaderDropzone"] { background:#FFFFFF !important; border:2px dashed #141414 !important; border-radius:0 !important; }
[role="radiogroup"] label { border:1px solid #141414; padding:4px 12px; margin-right:0 !important; background:#FFFFFF; }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] { border-radius:0 !important; font-weight:600 !important; }
[data-testid="stBaseButton-secondary"] { border:1px solid #141414 !important; background:#FFFFFF !important; color:#141414 !important; }
[data-testid="stBaseButton-primary"] { border:1px solid #141414 !important; background:#141414 !important; color:#FFFFFF !important; }
[data-testid="stAlert"] { background:#FFFFFF !important; border:1px solid #C4C6C0 !important; border-radius:0 !important; }
[data-testid="stAlertContainer"] { background:#FFFFFF !important; color:#141414 !important; }
[data-testid="stAlertContentInfo"] { color:#141414 !important; }
[data-testid="stCaptionContainer"] { color:#4A4B47 !important; }
hr { border-color:#C4C6C0; }
</style>
"""


def header_html(subtitle, model_label=None):
    """The black HI-VIS banner every page opens with. `subtitle` is the
    all-caps label next to the logo (e.g. "PPE COMPLIANCE DETECTION");
    `model_label` is optional and renders as a dim mono tag on the right."""
    model_bit = ""
    if model_label:
        model_bit = (f'<div class="hv-mono" style="font-size:11px;color:#8D8F8A;border-left:1px solid #3A3B38;'
                      f'padding-left:14px">{model_label}</div>')
    return f"""
    <div style="background:#141414;color:#FFFFFF;display:flex;align-items:center;gap:16px;
         padding:14px 24px;margin:0 0 20px 0;flex-wrap:wrap">
      <div style="background:#EFE600;color:#141414;font-family:'Barlow Condensed',sans-serif;
           font-weight:800;font-size:24px;letter-spacing:1px;padding:2px 10px 4px;line-height:1">HI-VIS</div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:15px;
           letter-spacing:2.5px">{subtitle}</div>
      {model_bit}
    </div>
    """


def build_rows(items, threshold, rule_text, required=("hardhat", "vest")):
    """items: list of {"name", "datetime", "assessment"} — one per uploaded
    photo. Returns one row per assessed person per tracked item in
    `required` (any of hardhat/vest/gloves/boots), covering both compliant
    and non-compliant findings, so the verdict filter in the UI can slice
    either view from the same table. An item outside `required` never
    appears here even if the model found it missing — it just is not part
    of the rule."""
    rows = []
    for item in items:
        assessment = item["assessment"]
        for p_idx, p in enumerate(assessment["persons"]):
            missing = [slot for slot in required if p["status"][slot]["state"] == "missing"]
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
