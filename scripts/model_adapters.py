"""
One adapter class per model in the comparison, all implementing the same
interface: `predict(image_path) -> list[Detection]`. scripts/compare_models.py
loops over the same sampled images for whichever adapters are requested, so
adding a new model later means writing one adapter class, not touching the
orchestration logic.

Every adapter declares `queryable_classes`: the subset of the 9-class schema
it can actually be asked about. In this comparison every non-YOLO model is
a chat-style LLM/VLM (Ollama-served, or Claude) that reasons about absence
in text as readily as presence, so their queryable set is the full schema.

Only YOLO sets `supports_grounding = True` — its `bbox` values are real
predicted boxes worth IoU-scoring against ground truth. It's the sole
non-LLM entrant, included as the fixed baseline every LLM/VLM is compared
against. Every other adapter's `bbox` is always None (presence/
classification only); comparing them at the box level would be comparing a
coordinate to a guess.
"""
import base64
import json
import mimetypes
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load repo-root .env (ANTHROPIC_API_KEY / GEMINI_API_KEY) if present — no
# python-dotenv dependency for two lines. setdefault() so a real shell
# export still wins. Done here, not in every script/notebook that imports
# this module, since this is where those keys are actually read.
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


@dataclass
class Detection:
    class_name: str
    present: bool
    confidence: Optional[float] = None
    bbox: Optional[tuple] = None  # normalized (x1, y1, x2, y2), or None if not grounded


POSITIVE_CLASSES = ["person", "helmet", "gloves", "boots", "vest"]
NEGATIVE_CLASSES = ["no-helmet", "no-gloves", "no-boots", "no-vest"]
ALL_CLASSES = POSITIVE_CLASSES + NEGATIVE_CLASSES

# YOLOAdapter reads its raw class names straight off whatever checkpoint
# DEFAULT_YOLO_WEIGHTS points at — different training runs have used
# different words for the same slot (helmet vs hardhat, vest vs "safety
# vest"). This is the one place that needs to know about it; add a run's
# raw names here rather than retraining to match. (streamlit_app/detector.py
# has its own equivalent map — separate module, separate vocabulary, not
# reused here on purpose.) Unmapped raw names (e.g. mask/no-mask, not
# tracked by this comparison) pass through lowercased and are simply
# ignored by every consumer that only looks at ALL_CLASSES/CHECKLIST_ITEMS.
_RAW_CLASS_MAP = {
    "hardhat": "helmet", "no-hardhat": "no-helmet",
    "safety vest": "vest", "no-safety vest": "no-vest",
}


def _norm_class_name(raw):
    raw = raw.strip().lower()
    return _RAW_CLASS_MAP.get(raw, raw)

# Editable, visible prompt template for every chat-style (non-grounding)
# model. {class_list} and {json_shape} are filled in per-model from its own
# queryable_classes via render_prompt() — e.g. a notebook can print/edit
# PROMPT_TEMPLATE directly and pass it into an adapter's prompt_template=.
DEFAULT_PROMPT_TEMPLATE = (
    "You are reviewing a construction-site photo for PPE compliance. "
    "For each of these classes: {class_list} — report whether AT LEAST "
    "ONE INSTANCE is visible anywhere in the image. A 'no-X' class means "
    "a visible person who is clearly missing that item (e.g. 'no-helmet' "
    "= a visible person with no helmet on).\n"
    "Respond with ONLY a JSON object, no markdown fences, no other text, "
    "in exactly this shape (one boolean per class listed above):\n"
    "{json_shape}"
)


def render_prompt(template, classes):
    return template.format(
        class_list=", ".join(classes),
        json_shape="{" + ", ".join(f'"{c}": true/false' for c in classes) + "}",
    )


def _extract_json(text):
    """First `{...}` blob in text, parsed — or None if there isn't one or it
    doesn't parse. Shared by every JSON-shaped parser below so there's one
    place that knows how model text (markdown fences, chatter, ...) gets
    turned into a JSON object."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_presence_json(text, classes):
    """Returns dict[class_name -> bool], or None if the response couldn't be
    parsed as JSON at all (caller should record this as a parse failure, not
    as "model said false everywhere")."""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None
    return {c: bool(data.get(c, False)) for c in classes}


# Per-person checklist prompt: "how many people, and for each, which items —
# present or absent, how sure, and where?" — a second query shape alongside
# the flat per-image presence one above. Used by
# streamlit_app/pages/checklist_compare.py via each adapter's describe() for
# chat-style models, or people_from_detections() below for a grounding model
# (YOLO) — both produce the same list-of-people shape (see
# parse_person_checklist_json()'s docstring), so the rest of that page's
# scoring/visualization never needs to know which one produced it.
CHECKLIST_ITEMS = ["helmet", "vest", "gloves", "boots"]

PERSON_CHECKLIST_PROMPT_TEMPLATE = (
    "You are reviewing a construction-site photo for PPE compliance. "
    "Count every distinct person visible in the image — even partially "
    "visible ones. For EACH person: give their bounding box, then for each "
    "of {class_list}, report whether they have it (class \"X\") or not "
    "(class \"no-X\"), your confidence in that call from 0 to 1, and a "
    "bounding box — the item itself if present, or the region where it "
    "would be (e.g. the head, for a missing helmet) if absent. All boxes "
    "are [x1, y1, x2, y2], normalized 0-1 (top-left / bottom-right).\n"
    "Respond with ONLY a JSON object, no markdown fences, no other text, "
    "in exactly this shape — one entry per person, an empty list if you "
    "see nobody:\n"
    '{{"people": [{{"bbox": [x1,y1,x2,y2], "items": [{json_shape}]}}, ...]}}'
)


def render_checklist_prompt(items=CHECKLIST_ITEMS):
    item_shape = ", ".join(
        f'{{"class": "{item}"|"no-{item}", "confidence": 0.0-1.0, "bbox": [x1,y1,x2,y2]}}'
        for item in items
    )
    return PERSON_CHECKLIST_PROMPT_TEMPLATE.format(class_list=", ".join(items), json_shape=item_shape)


def _as_bbox(v):
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return tuple(float(x) for x in v)
        except (TypeError, ValueError):
            pass
    return None


def _as_confidence(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


def parse_person_checklist_json(text, items=CHECKLIST_ITEMS):
    """Returns a list of {"bbox": (x1,y1,x2,y2)|None, "items": [Detection, ...]},
    one per person the model claims to see (an empty list is a real,
    meaningful "saw nobody" answer) — or None if the response couldn't be
    read as {"people": [...]} at all (a parse failure, kept distinct from
    "0 people"). Each person carries exactly one Detection per slot in
    `items` — its class_name is either the slot ("helmet") or its negative
    ("no-helmet"), so the same class can repeat once per person without
    colliding across people. A slot the model never mentioned (or a
    malformed entry) defaults to an unstated negative (confidence/bbox
    None) rather than dropping the slot, so every person has the same
    item count downstream."""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return None
    raw_people = data.get("people")
    if not isinstance(raw_people, list):
        return None
    valid_names = {name for item in items for name in (item, f"no-{item}")}
    people = []
    for p in raw_people:
        p = p if isinstance(p, dict) else {}
        by_slot = {}
        for it in p.get("items", []) if isinstance(p.get("items"), list) else []:
            if not isinstance(it, dict) or it.get("class") not in valid_names:
                continue
            cls = it["class"]
            slot = cls[3:] if cls.startswith("no-") else cls
            by_slot[slot] = Detection(cls, not cls.startswith("no-"),
                                       _as_confidence(it.get("confidence")), _as_bbox(it.get("bbox")))
        people.append({
            "bbox": _as_bbox(p.get("bbox")),
            "items": [by_slot.get(item, Detection(f"no-{item}", False, None, None)) for item in items],
        })
    return people


def _containment(item_box, container_box):
    """Fraction of item_box's area that lies inside container_box — same
    idea as streamlit_app/detector.py's _containment(), duplicated locally
    since that module is tied to a different (demo-specific) YOLO weights
    registry and vocabulary, not YOLOAdapter's merged-dataset model."""
    ix1, iy1, ix2, iy2 = item_box
    cx1, cy1, cx2, cy2 = container_box
    ox1, oy1 = max(ix1, cx1), max(iy1, cy1)
    ox2, oy2 = min(ix2, cx2), min(iy2, cy2)
    ow, oh = max(0.0, ox2 - ox1), max(0.0, oy2 - oy1)
    inter = ow * oh
    item_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / item_area if item_area > 0 else 0.0


def people_from_detections(detections, items=CHECKLIST_ITEMS, containment_thresh=0.5):
    """The same per-person shape parse_person_checklist_json() returns,
    built from a grounding model's own real boxes instead of a prompted
    answer: each detected person, plus whichever item/no-item box (our
    trained YOLO detects both as real classes, each with its own confidence
    and box) is mostly contained in it, becomes that slot's Detection —
    highest-confidence one wins if both showed up. No matching box at all
    means an unstated negative, same convention as a chat model's
    unmentioned slot. Lets YOLO run through the exact same checklist
    scoring/visualization pipeline as every chat model, instead of being
    left out because it has no describe()."""
    persons = [d for d in detections if d.class_name == "person" and d.bbox is not None]
    other = [d for d in detections if d.class_name != "person" and d.bbox is not None]
    people = []
    for p in persons:
        entry_items = []
        for item in items:
            candidates = [d for d in other if d.class_name in (item, f"no-{item}")
                          and _containment(d.bbox, p.bbox) > containment_thresh]
            best = max(candidates, key=lambda d: d.confidence or 0.0) if candidates else None
            entry_items.append(best or Detection(f"no-{item}", False, None, None))
        people.append({"bbox": p.bbox, "items": entry_items})
    return people


class YOLOAdapter:
    """Our trained detector — the baseline every other model is compared
    against. Real boxes, all 9 classes, since it was trained on negatives
    too.

    _lock (per-INSTANCE, unlike OllamaAdapter's per-CLASS _SERVER_LOCK —
    different weight files are independent models, free to run in
    parallel; only calls sharing this one instance need serializing):
    checklist_compare.py's run_checklist_steps() submits every (file,
    model) pair to a thread pool at once, so a single popular YOLO weight
    can get a couple dozen concurrent .predict() calls on this exact
    object. Ultralytics' predict() isn't safe for that — enough
    concurrent callers reliably deadlocked with no exception and no CPU
    use (each call re-triggering .predictor's lazy setup, racing on the
    same internal state) rather than merely raising or slowing down."""

    name = "yolo"
    supports_grounding = True
    queryable_classes = ALL_CLASSES

    def __init__(self, weights_path, conf=0.25):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.class_names = self.model.names  # {id: name}
        self.conf = conf
        self._lock = threading.Lock()

    def predict(self, image_path):
        with self._lock:
            result = self.model.predict(str(image_path), conf=self.conf, verbose=False)[0]
        img_h, img_w = result.orig_shape
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=_norm_class_name(self.class_names[cls_id]),
                    present=True,
                    confidence=float(box.conf[0]),
                    bbox=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                )
            )
        return detections


class OllamaAdapter:
    """Chat-style local VLM via Ollama's REST API (presence/classification
    only — do not trust coordinates from a chat model). Requires Ollama
    installed and running separately (`brew install ollama`, `ollama pull
    <model>`, `ollama serve`) — not set up by this scaffold. Backs four
    registry entries (ollama/qwen3-vl/gemma4/minicpm-v below) that
    differ only in which model tag they pull — four different model
    families (LLaVA, Qwen, Gemma, MiniCPM) compared under identical
    prompting and scoring.

    _SERVER_LOCK is class-level, shared by every instance/model tag: the
    comparison runs every adapter concurrently (one thread per model per
    image), but they're four DIFFERENT model tags on the SAME local Ollama
    server. Fired at once, Ollama has to keep swapping which model is
    resident in VRAM between requests — confirmed directly (each model
    alone: ~30-45s; concurrent, "hangs" for minutes) — so this serializes
    every Ollama call to one at a time. YOLO and the cloud adapters aren't
    touched by this lock and keep running fully in parallel.
    ponytail: one global lock for every tag, not per-tag — simplest fix for
    "don't thrash one shared server," revisit if profiling ever shows two
    tags genuinely coexist in VRAM without evicting each other."""

    supports_grounding = False
    queryable_classes = ALL_CLASSES
    _SERVER_LOCK = threading.Lock()

    def __init__(self, model="llava", base_url="http://localhost:11434", prompt_template=DEFAULT_PROMPT_TEMPLATE):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompt_template = prompt_template

    def _generate(self, image_path, text_prompt):
        import requests

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        with self._SERVER_LOCK:  # one Ollama call at a time across every model tag — see class docstring
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": text_prompt,
                    "images": [image_b64],
                    "stream": False,
                    "format": "json",
                    # Backstop against a genuine runaway/looping generation eating
                    # the whole 300s timeout — NOT the fix for the "ollama/gemma4
                    # hang" bug (that was VRAM-thrashing from concurrent requests
                    # to different model tags on one server, fixed by _SERVER_LOCK
                    # above). 800 was too tight and silently cut off legitimate
                    # multi-person answers (observed: a real 95s gemma4 response
                    # came back empty; a real 5-person answer needed ~450 tokens).
                    # 1200 gives ~2.5x that headroom while still failing a true
                    # loop in well under a minute instead of several.
                    "options": {"num_predict": 1200},
                },
                timeout=300,  # cold model loads observed up to ~140s under load; leave headroom
            )
        response.raise_for_status()
        data = response.json()
        # Reasoning models (e.g. qwen3-vl) can emit their entire answer inside
        # Ollama's "thinking" field and leave "response" empty if they never
        # produce a distinct post-thinking answer — try both, prefer "response".
        return data.get("response", "") or data.get("thinking", "")

    def predict(self, image_path):
        text = self._generate(image_path, render_prompt(self.prompt_template, self.queryable_classes))
        presence = parse_presence_json(text, self.queryable_classes)
        if presence is None:
            return None  # caller records this as a parse failure
        return [Detection(class_name=c, present=present) for c, present in presence.items()]

    def describe(self, image_path, prompt):
        """Free-text/other-JSON-shape mode, same rationale as Claude/Gemini's
        describe() — outside predict()'s fixed presence schema. Used by the
        per-person checklist comparison, which asks a differently-shaped
        question ("how many people, and for each...") than the flat
        per-class presence prompt predict() is locked to."""
        return self._generate(image_path, prompt)


class GeminiAdapter:
    """Cloud, gated behind --include-cloud in compare_models.py. Presence/
    classification only, via a direct REST call (no google-genai SDK
    dependency, matching OllamaAdapter's raw-`requests` style). Needs
    GEMINI_API_KEY in the environment — not supplied by this scaffold, and
    never written to disk or committed.

    Default model is gemini-3.6-flash: gemini-2.5-flash (the prior default
    guess) returned a hard 404 as of this writing — "no longer available
    to new users" — with the API's own error message naming 3.6-flash as
    the replacement. Verified live against the real endpoint, not assumed."""

    name = "gemini"
    supports_grounding = False
    queryable_classes = ALL_CLASSES

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model="gemini-3.6-flash", prompt_template=DEFAULT_PROMPT_TEMPLATE, api_key=None):
        import os

        self.model = model
        self.prompt_template = prompt_template
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GeminiAdapter needs GEMINI_API_KEY set in the environment (or api_key= passed in).")

    def _generate(self, image_path, text_prompt, max_retries=5):
        import time

        import requests

        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        url = f"{self.API_BASE}/{self.model}:generateContent?key={self.api_key}"
        body = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                        {"text": text_prompt},
                    ]
                }
            ],
            # "low" thinking noticeably cuts latency/cost for this task
            # (verified: ~112 thoughts tokens on a trivial prompt at
            # default vs. ~69 at "low"); thinkingBudget:0 was rejected
            # outright (400 INVALID_ARGUMENT) on this model, so "low" is
            # the actual floor, not a guess.
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}},
        }
        # 429 (rate limit), 5xx (transient server overload), and network-level
        # timeouts/connection drops are all worth a retry — all observed live
        # from this endpoint under a 100-image run, not hypothetical.
        # Anything else (400/403/404) is a real problem and should raise
        # immediately rather than retry into the same wall 5 times.
        # timeout=90/backoff cap=15s (was 180/30): worst case with all 5
        # retries timing out was 1050s — long enough to look identical to a
        # genuine hang in the UI. 90/15 halves that ceiling; a request that's
        # actually going to succeed almost never takes anywhere near 90s.
        last_exc = None
        data = None
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=body, timeout=90)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 15))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_exc = requests.exceptions.HTTPError(
                    f"{response.status_code} on attempt {attempt + 1}/{max_retries}", response=response
                )
                time.sleep(min(2**attempt, 15))
                continue
            response.raise_for_status()
            data = response.json()
            break
        if data is None:
            raise last_exc
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def predict(self, image_path):
        text = self._generate(image_path, render_prompt(self.prompt_template, self.queryable_classes))
        presence = parse_presence_json(text, self.queryable_classes)
        if presence is None:
            return None
        return [Detection(class_name=c, present=present) for c, present in presence.items()]

    def describe(self, image_path, prompt):
        """Free-text mode for prompts that aren't the structured-JSON
        presence task (e.g. a plain-English site-record description) —
        deliberately outside the Detection/predict() interface, since
        there's no per-class boolean to score against ground truth here.
        Used by the descriptive-prompt comparison, not by
        compare_models.py's scored pipeline."""
        return self._generate(image_path, prompt)


class ClaudeAdapter:
    """Cloud, gated behind --include-cloud in compare_models.py. Presence/
    classification only. Needs ANTHROPIC_API_KEY (or an `ant auth login`
    profile) in the environment — not supplied by this scaffold."""

    name = "claude"
    supports_grounding = False
    queryable_classes = ALL_CLASSES

    def __init__(self, model="claude-haiku-4-5", prompt_template=DEFAULT_PROMPT_TEMPLATE):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.prompt_template = prompt_template

    def _generate(self, image_path, text_prompt):
        media_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    def predict(self, image_path):
        text = self._generate(image_path, render_prompt(self.prompt_template, self.queryable_classes))
        presence = parse_presence_json(text, self.queryable_classes)
        if presence is None:
            return None
        return [Detection(class_name=c, present=present) for c, present in presence.items()]

    def describe(self, image_path, prompt):
        """Free-text mode, same rationale as GeminiAdapter.describe() — no
        per-class boolean to score, so this is outside predict()'s scoring
        pipeline. Used to collect Claude's plain-language response per image
        alongside the structured call, for side-by-side viewing."""
        return self._generate(image_path, prompt)


# name -> {cls, is_cloud, default_model}. compare_models.py uses is_cloud to
# enforce the --include-cloud gate, and default_model to pick a model tag
# when the caller doesn't override one.
#
# qwen3-vl:4b (3.3GB), gemma4:e4b (9.6GB, released April 2026), and
# minicpm-v:8b are confirmed real Ollama library tags — verified via
# ollama.com/library before pulling, not guesses.
ADAPTERS = {
    "yolo": {"cls": YOLOAdapter, "is_cloud": False, "default_model": None},
    "ollama": {"cls": OllamaAdapter, "is_cloud": False, "default_model": "llava"},
    "qwen3-vl": {"cls": OllamaAdapter, "is_cloud": False, "default_model": "qwen3-vl:4b"},
    "gemma4": {"cls": OllamaAdapter, "is_cloud": False, "default_model": "gemma4:e4b"},
    "minicpm-v": {"cls": OllamaAdapter, "is_cloud": False, "default_model": "minicpm-v:8b"},
    "claude": {"cls": ClaudeAdapter, "is_cloud": True, "default_model": "claude-haiku-4-5"},
    "gemini": {"cls": GeminiAdapter, "is_cloud": True, "default_model": "gemini-3.6-flash"},
}
