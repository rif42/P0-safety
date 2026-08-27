"""
One adapter class per model in the comparison, all implementing the same
interface: `predict(image_path) -> list[Detection]`. scripts/compare_models.py
loops over the same sampled images for whichever adapters are requested, so
adding a new model later means writing one adapter class, not touching the
orchestration logic.

Every adapter declares `queryable_classes`: the subset of the 9-class schema
it can actually be asked about. This matters because grounding-only models
(Florence-2, or any dedicated open-vocab detector) can point at "helmet" but
cannot express "no-helmet" as a groundable phrase — that's an absence, not an
object. Asking them about negative classes and silently recording "not
found" as a confident "false" would be dishonest scoring. Chat-style models
(Ollama, Claude) CAN reason about absence in text, so their queryable set is
the full schema.

Only YOLO and Florence-2 set `supports_grounding = True` — their `bbox`
values are real predicted boxes worth IoU-scoring against ground truth.
Chat-style models' `bbox` is always None; comparing them at the box level
would be comparing a coordinate to a guess.
"""
import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Detection:
    class_name: str
    present: bool
    confidence: Optional[float] = None
    bbox: Optional[tuple] = None  # normalized (x1, y1, x2, y2), or None if not grounded


POSITIVE_CLASSES = ["person", "helmet", "gloves", "boots", "vest"]
NEGATIVE_CLASSES = ["no-helmet", "no-gloves", "no-boots", "no-vest"]
ALL_CLASSES = POSITIVE_CLASSES + NEGATIVE_CLASSES

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


def parse_presence_json(text, classes):
    """Returns dict[class_name -> bool], or None if the response couldn't be
    parsed as JSON at all (caller should record this as a parse failure, not
    as "model said false everywhere")."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return {c: bool(data.get(c, False)) for c in classes}


class YOLOAdapter:
    """Our trained detector — the baseline every other model is compared
    against. Real boxes, all 9 classes, since it was trained on negatives
    too."""

    name = "yolo"
    supports_grounding = True
    queryable_classes = ALL_CLASSES

    def __init__(self, weights_path, conf=0.25):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.class_names = self.model.names  # {id: name}
        self.conf = conf

    def predict(self, image_path):
        result = self.model.predict(str(image_path), conf=self.conf, verbose=False)[0]
        img_h, img_w = result.orig_shape
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=self.class_names[cls_id],
                    present=True,
                    confidence=float(box.conf[0]),
                    bbox=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                )
            )
        return detections


class Florence2Adapter:
    """Local, open-source, real grounding output via transformers — no API
    key, no new runtime beyond what's already installed. Only queryable on
    positive classes (see module docstring): one phrase-grounding call per
    class per image, so this is the slowest adapter (~5 calls x ~5s each on
    an M-series Mac)."""

    name = "florence2"
    supports_grounding = True
    queryable_classes = POSITIVE_CLASSES

    def __init__(self, model_id="microsoft/Florence-2-base", device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.task = "<CAPTION_TO_PHRASE_GROUNDING>"

    def predict(self, image_path):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        detections = []
        for cls_name in self.queryable_classes:
            inputs = self.processor(text=self.task + cls_name, images=image, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                generated_text, task=self.task, image_size=(image.width, image.height)
            )
            for x1, y1, x2, y2 in parsed.get(self.task, {}).get("bboxes", []):
                detections.append(
                    Detection(
                        class_name=cls_name,
                        present=True,
                        confidence=None,  # not a calibrated score for this task
                        bbox=(x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height),
                    )
                )
        return detections


class OllamaAdapter:
    """Chat-style local VLM via Ollama's REST API (presence/classification
    only — do not trust coordinates from a chat model). Requires Ollama
    installed and running separately (`brew install ollama`, `ollama pull
    <model>`, `ollama serve`) — not set up by this scaffold. Backs three
    registry entries (ollama/qwen3-vl/gemma3n below) that differ only in
    which model tag they pull; none have been live-tested in this
    environment since Ollama isn't installed here."""

    supports_grounding = False
    queryable_classes = ALL_CLASSES

    def __init__(self, model="llava", base_url="http://localhost:11434", prompt_template=DEFAULT_PROMPT_TEMPLATE):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompt_template = prompt_template

    def predict(self, image_path):
        import requests

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": render_prompt(self.prompt_template, self.queryable_classes),
                "images": [image_b64],
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()
        presence = parse_presence_json(response.json().get("response", ""), self.queryable_classes)
        if presence is None:
            return None  # caller records this as a parse failure
        return [Detection(class_name=c, present=present) for c, present in presence.items()]


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

    def predict(self, image_path):
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
                        {"type": "text", "text": render_prompt(self.prompt_template, self.queryable_classes)},
                    ],
                }
            ],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        presence = parse_presence_json(text, self.queryable_classes)
        if presence is None:
            return None
        return [Detection(class_name=c, present=present) for c, present in presence.items()]


# name -> {cls, is_cloud, default_model}. compare_models.py uses is_cloud to
# enforce the --include-cloud gate, and default_model to pick a model tag
# when the caller doesn't override one.
#
# qwen3-vl and gemma3n default tags ("qwen3-vl:4b", "gemma3n:e4b") are best
# guesses at Ollama's published naming for Qwen3-VL-4B and Gemma 3n E4B —
# verify with `ollama pull <tag>` (or `ollama list` after pulling) before
# relying on them; Ollama isn't installed in this environment so these
# haven't been live-tested.
ADAPTERS = {
    "yolo": {"cls": YOLOAdapter, "is_cloud": False, "default_model": None},
    "florence2": {"cls": Florence2Adapter, "is_cloud": False, "default_model": "microsoft/Florence-2-base"},
    "ollama": {"cls": OllamaAdapter, "is_cloud": False, "default_model": "llava"},
    "qwen3-vl": {"cls": OllamaAdapter, "is_cloud": False, "default_model": "qwen3-vl:4b"},
    "gemma3n": {"cls": OllamaAdapter, "is_cloud": False, "default_model": "gemma3n:e4b"},
    "claude": {"cls": ClaudeAdapter, "is_cloud": True, "default_model": "claude-haiku-4-5"},
}
