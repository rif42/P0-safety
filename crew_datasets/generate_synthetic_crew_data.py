"""HI-VIS pitch — fully generated (drawn) two-crew CCTV dataset.

Replaces the earlier real-photo-based crew_a/crew_b datasets (which drew
from anuragraj03's HSM camera — see git history / classify_cameras.py for
that whole investigation) with fully synthetic, programmatically DRAWN
CCTV-style images. Nothing here is a real photograph: every worker figure,
every helmet/vest/glove/boot, and every watermark is rendered by PIL from
scratch. This trades "real photo" authenticity for full control over the
story: instead of being limited to whatever compliance states happen to
exist in a real photo pool, each day's compliance probabilities are driven
by a short list of named, presenter-callable EVENTS (a heatwave, a near-
miss, a toolbox talk...) so the pitch can point at a specific day on the
trend chart and explain *why* it moved, not just that it moved.

IMPORTANT — these are illustrative mockups, not real security footage.
Say so wherever this dataset is shown (labels, captions, the pitch deck)
— the CCTV-style timestamp/camera-ID watermark is deliberately styled
after real footage for narrative texture, not to pass as genuine.

Ground truth is exact by construction: every box drawn is a box labeled,
in the same 9-class schema as the rest of this project (see
data/merged/data.yaml) — person, helmet, gloves, boots, vest, no-helmet,
no-gloves, no-boots, no-vest.

Run from crew_datasets/:
    python generate_synthetic_crew_data.py
"""

import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_ROOT = Path(__file__).resolve().parent / "data"
CANVAS = 640
SEED = 2026

CLASS_IDS = {
    "person": 0,
    "helmet": 1,
    "gloves": 2,
    "boots": 3,
    "vest": 4,
    "no-helmet": 5,
    "no-gloves": 6,
    "no-boots": 7,
    "no-vest": 8,
}

PHOTOS_PER_DAY = 3
DAYS_PER_WEEK = 5
N_WEEKS = 6
N_DAYS = DAYS_PER_WEEK * N_WEEKS  # 30 weekdays, 0-indexed
START_DATE = datetime(2026, 6, 1, 0, 0)


@dataclass
class Event:
    name: str
    days: range  # 0-indexed weekday range this event is active
    effect: dict  # item -> probability delta (added to baseline, then clipped)
    note: str  # one-line explanation for the story file / chart annotation


CREWS = {
    "crew_a": {
        "camera": "CAM-A1",
        "label": "Crew A — the slow fade",
        "start_pct": {"helmet": 0.95, "vest": 0.95, "gloves": 0.93, "boots": 0.96},
        "end_pct": {"helmet": 0.72, "vest": 0.68, "gloves": 0.70, "boots": 0.80},
        "events": [
            Event(
                "Heatwave",
                range(10, 13),  # week 3, Wed-Fri
                {"vest": -0.35, "gloves": -0.15},
                "Three-day heatwave (week 3): workers shed hi-vis vests and gloves in the heat -- a sharp, isolated dip in exactly the days it was hot, not a permanent drop.",
            ),
            Event(
                "New apprentices join",
                range(25, 26),  # week 6, one day
                {"helmet": -0.20, "vest": -0.20, "gloves": -0.25, "boots": -0.20},
                "Two apprentices join the crew (week 6, day 1): a same-day dip across every item -- more people on site, less supervision per person, no induction yet.",
            ),
        ],
    },
    "crew_b": {
        "camera": "CAM-B1",
        "label": "Crew B — the turnaround",
        "start_pct": {"helmet": 0.45, "vest": 0.50, "gloves": 0.45, "boots": 0.55},
        "end_pct": {"helmet": 0.50, "vest": 0.55, "gloves": 0.50, "boots": 0.60},
        "events": [
            Event(
                "Rainy week",
                range(0, 5),  # week 1
                {"boots": -0.20, "gloves": -0.15},
                "A wet week 1: workers swap boots/gloves for whatever's dry, dragging those two metrics down before anything else changes.",
            ),
            Event(
                "Near-miss (dropped tool)",
                range(11, 15),  # week 3, Tue-Fri, short-lived
                {"helmet": 0.30},
                "A dropped tool narrowly misses an unhelmeted worker (week 3, day 2): a sharp but short-lived helmet-compliance spike as the crew self-corrects out of fear, fading back down by the end of the week without reinforcement.",
            ),
            Event(
                "Toolbox talk / safety stand-down",
                range(15, 30),  # from week 4 onward, permanent
                {"helmet": 0.40, "vest": 0.35, "gloves": 0.35, "boots": 0.30},
                "A formal safety stand-down at the start of week 4: unlike the near-miss spike, this holds -- every metric jumps and stays up through week 6.",
            ),
        ],
    },
}

PALETTE = {
    "helmet_on": [(232, 197, 71), (255, 255, 255), (240, 220, 60)],
    "helmet_off": [(120, 88, 62), (90, 65, 45), (140, 100, 70)],  # bare/hair tones
    "vest_on": (255, 140, 0),
    "vest_off": [(70, 90, 110), (100, 100, 100), (60, 110, 90), (110, 70, 70)],
    "glove_on": (235, 235, 225),
    "skin": [(190, 150, 120), (150, 110, 85), (210, 170, 140)],
    "boot_on": (30, 25, 22),
    "shoe_off": [(200, 200, 195), (150, 40, 40), (40, 40, 140)],
}


def clip01(x):
    return max(0.03, min(0.98, x))


def daily_probabilities(crew_key):
    """Returns {day_index: {item: probability}} for one crew's full timeline."""
    crew = CREWS[crew_key]
    out = {}
    for day in range(N_DAYS):
        t = day / (N_DAYS - 1)
        probs = {}
        for item in ("helmet", "vest", "gloves", "boots"):
            base = crew["start_pct"][item] + t * (crew["end_pct"][item] - crew["start_pct"][item])
            delta = sum(ev.effect.get(item, 0.0) for ev in crew["events"] if day in ev.days)
            probs[item] = clip01(base + delta)
        out[day] = probs
    return out


def business_hour_timestamp(day_date, rng):
    hour = rng.choice([8, 9, 10, 11, 13, 14, 15, 16, 17])
    minute = rng.randint(0, 59)
    return datetime.combine(day_date.date(), time(hour, minute))


def draw_background(rng):
    img = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = rng.randint(95, 115)
    for y in range(CANVAS):
        shade = base + int(20 * (y / CANVAS)) - 10
        draw.line([(0, y), (CANVAS, y)], fill=(shade, shade, shade - 5))
    # a few plank/seam lines for floor texture
    for _ in range(6):
        x = rng.randint(0, CANVAS)
        drift = rng.randint(-30, 30)
        c = base - rng.randint(5, 20)
        draw.line([(x, 0), (x + drift, CANVAS)], fill=(c, c, c - 5), width=rng.randint(1, 3))
    # a couple of generic industrial background shapes (crates)
    for _ in range(rng.randint(2, 4)):
        x0 = rng.randint(0, CANVAS - 80)
        y0 = rng.randint(0, 100)
        w, h = rng.randint(40, 90), rng.randint(30, 60)
        c = rng.randint(60, 90)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(c, c - 5, c - 10), outline=(30, 30, 30))
    return img


def draw_worker(draw, cx, cy, scale, state, rng):
    """Draws one worker (elevated/top-down view: mostly head+shoulders, feet
    peeking out below) and returns per-part bounding boxes in pixel coords."""
    boxes = {}

    torso_w, torso_h = int(46 * scale), int(34 * scale)
    torso_color = PALETTE["vest_on"] if state["vest"] else rng.choice(PALETTE["vest_off"])
    torso_box = [cx - torso_w // 2, cy - torso_h // 2, cx + torso_w // 2, cy + torso_h // 2]
    draw.ellipse(torso_box, fill=torso_color, outline=(20, 20, 20))
    if state["vest"]:
        draw.line([torso_box[0] + 6, cy, torso_box[2] - 6, cy], fill=(255, 220, 150), width=max(1, int(3 * scale)))

    head_r = int(15 * scale)
    head_cy = cy - torso_h // 2 - head_r + int(6 * scale)
    head_color = rng.choice(PALETTE["helmet_on"]) if state["helmet"] else rng.choice(PALETTE["helmet_off"])
    head_box = [cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r]
    draw.ellipse(head_box, fill=head_color, outline=(20, 20, 20))

    hand_r = int(7 * scale)
    hand_y = cy + int(4 * scale)
    hand_boxes = []
    for side in (-1, 1):
        hx = cx + side * (torso_w // 2 + hand_r - int(4 * scale))
        color = PALETTE["glove_on"] if state["gloves"] else rng.choice(PALETTE["skin"])
        hb = [hx - hand_r, hand_y - hand_r, hx + hand_r, hand_y + hand_r]
        draw.ellipse(hb, fill=color, outline=(20, 20, 20))
        hand_boxes.append(hb)

    foot_r = int(8 * scale)
    foot_y = cy + torso_h // 2 + foot_r
    foot_boxes = []
    for side in (-1, 1):
        fx = cx + side * int(12 * scale)
        color = PALETTE["boot_on"] if state["boots"] else rng.choice(PALETTE["shoe_off"])
        fb = [fx - foot_r, foot_y - foot_r, fx + foot_r, foot_y + foot_r]
        draw.ellipse(fb, fill=color, outline=(15, 15, 15))
        foot_boxes.append(fb)

    all_x = [torso_box[0], torso_box[2], head_box[0], head_box[2]] + [b[0] for b in hand_boxes + foot_boxes] + [
        b[2] for b in hand_boxes + foot_boxes
    ]
    all_y = [torso_box[1], torso_box[3], head_box[1], head_box[3]] + [b[1] for b in hand_boxes + foot_boxes] + [
        b[3] for b in hand_boxes + foot_boxes
    ]
    boxes["person"] = [min(all_x), min(all_y), max(all_x), max(all_y)]
    boxes["helmet" if state["helmet"] else "no-helmet"] = head_box
    boxes["vest" if state["vest"] else "no-vest"] = torso_box
    boxes["gloves" if state["gloves"] else "no-gloves"] = hand_boxes
    boxes["boots" if state["boots"] else "no-boots"] = foot_boxes
    return boxes


def add_cctv_look(img, np_rng):
    arr = np.array(img).astype(np.int16)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 12, 0, 255)  # slight green cast
    arr[:, :, 2] = np.clip(arr[:, :, 2] - 8, 0, 255)
    noise = np_rng.normal(0, 9, arr.shape[:2])[:, :, None]
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def stamp_watermark(img, ts, camera):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=15)
    date_text = f"{ts:%d-%m-%Y} {ts:%a} {ts:%I:%M:%S %p}"
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        draw.text((10 + dx, 8 + dy), date_text, font=font, fill=(0, 0, 0))
    draw.text((10, 8), date_text, font=font, fill=(255, 255, 255))

    cam_text = camera
    bbox = draw.textbbox((0, 0), cam_text, font=font)
    tw = bbox[2] - bbox[0]
    x, y = CANVAS - tw - 12, CANVAS - 26
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        draw.text((x + dx, y + dy), cam_text, font=font, fill=(0, 0, 0))
    draw.text((x, y), cam_text, font=font, fill=(255, 255, 255))
    return img


def render_photo(camera, ts, probs, rng, np_rng):
    img = draw_background(rng)
    draw = ImageDraw.Draw(img)
    n_workers = rng.randint(3, 6)
    label_lines = []

    positions = []
    attempts = 0
    while len(positions) < n_workers and attempts < 200:
        attempts += 1
        x = rng.randint(90, CANVAS - 90)
        y = rng.randint(150, CANVAS - 120)
        if all((x - px) ** 2 + (y - py) ** 2 > 90**2 for px, py in positions):
            positions.append((x, y))

    for x, y in positions:
        scale = rng.uniform(0.85, 1.25)
        state = {item: rng.random() < probs[item] for item in ("helmet", "vest", "gloves", "boots")}
        boxes = draw_worker(draw, x, y, scale, state, rng)
        for cls_name, box in boxes.items():
            box_list = box if isinstance(box[0], list) else [box]
            for b in box_list:
                x1, y1, x2, y2 = b
                xc, yc = (x1 + x2) / 2 / CANVAS, (y1 + y2) / 2 / CANVAS
                w, h = (x2 - x1) / CANVAS, (y2 - y1) / CANVAS
                label_lines.append(f"{CLASS_IDS[cls_name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    img = add_cctv_look(img, np_rng)
    img = stamp_watermark(img, ts, camera)
    return img, label_lines, n_workers


def photo_verdict_and_rates(label_lines):
    counts = {name: 0 for name in CLASS_IDS}
    id_to_name = {v: k for k, v in CLASS_IDS.items()}
    for line in label_lines:
        cls_id = int(line.split()[0])
        counts[id_to_name[cls_id]] += 1
    rates = {}
    for item in ("helmet", "vest", "gloves", "boots"):
        pos, neg = counts[item], counts[f"no-{item}"]
        total = pos + neg
        rates[f"{item}_rate"] = (pos / total) if total else None
    verdict = "compliant"
    for item in ("helmet", "vest", "gloves", "boots"):
        r = rates[f"{item}_rate"]
        if r is not None and r < 1.0:
            verdict = "noncompliant"
    return rates, verdict, counts["person"]


def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    rows = []
    for crew_key, crew in CREWS.items():
        out_dir = OUT_ROOT / crew_key / crew["camera"]
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)

        probs_by_day = daily_probabilities(crew_key)
        day = START_DATE
        day_idx = 0
        while day_idx < N_DAYS:
            if day.weekday() < 5:
                probs = probs_by_day[day_idx]
                n_photos = PHOTOS_PER_DAY + rng.choice([-1, 0, 0, 0, 1])
                for _ in range(max(1, n_photos)):
                    ts = business_hour_timestamp(day, rng)
                    img, label_lines, n_workers = render_photo(crew["camera"], ts, probs, rng, np_rng)
                    rates, verdict, n_person = photo_verdict_and_rates(label_lines)

                    base = f"{ts:%Y-%m-%d_%H-%M}_{verdict}"
                    fname = f"{base}.jpg"
                    i = 1
                    while (out_dir / "images" / fname).exists():
                        i += 1
                        fname = f"{base}-{i}.jpg"
                    img.save(out_dir / "images" / fname, quality=88)
                    (out_dir / "labels" / fname.replace(".jpg", ".txt")).write_text("\n".join(label_lines) + "\n")

                    rows.append(
                        {
                            "crew": crew_key,
                            "camera": crew["camera"],
                            "week": day_idx // DAYS_PER_WEEK + 1,
                            "day_index": day_idx,
                            "capture_datetime": ts,
                            "filename": fname,
                            "n_workers": n_person,
                            "verdict": verdict,
                            **rates,
                        }
                    )
                day_idx += 1
            day += timedelta(days=1)

        n_images = len(list((out_dir / "images").glob("*.jpg")))
        print(f"{crew_key} ({crew['camera']}): {n_images} generated photos")

    scored = pd.DataFrame(rows)
    OUT_ROOT.mkdir(exist_ok=True)
    scored.to_csv(OUT_ROOT / "manifest_scored.csv", index=False)
    print(f"\nWrote {OUT_ROOT / 'manifest_scored.csv'} ({len(scored)} rows)")

    print("\nOverall verdict counts by crew:")
    print(scored.groupby(["crew", "verdict"]).size().unstack(fill_value=0))

    for metric in ("helmet", "vest", "gloves", "boots"):
        print(f"\nWeekly {metric} compliance rate by crew:")
        weekly = (
            scored.dropna(subset=[f"{metric}_rate"])
            .groupby(["crew", "week"])[f"{metric}_rate"]
            .mean()
            .unstack("crew")
            .round(2)
        )
        print(weekly)


if __name__ == "__main__":
    main()
