"""
Изчистване на мебели от снимка на помещение.

Стъпки:
1. GPT-4o Vision открива мебелите и връща bounding boxes (нормализирани 0..1).
2. OpenCV inpainting запълва маскираните зони с околната текстура (стена/под).
"""
import base64
import io
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np


def detect_furniture_boxes(base64_image: str) -> Optional[List[Dict]]:
    """
    Връща списък с {label, x, y, w, h} (нормализирани 0..1) за мебели/предмети,
    които трябва да се премахнат. None, ако OpenAI не е конфигуриран.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    prompt = """Ти си система за компютърно зрение. Задачата ти е да откриеш ВСИЧКИ мебели и предмети в стаята, които трябва да бъдат премахнати, за да остане празно помещение.

ПРЕМАХВАЙ (маркирай с box):
- мебели: шкафове, гардероби, комоди, легла, дивани, маси, столове, рафтове, ТВ шкафове
- техника: телевизори, монитори, лаптопи, лампи
- декорации и дребни предмети върху мебели и по стените: картини, постери, растения, играчки, кутии

НЕ МАРКИРАЙ (това е част от празната стая):
- стени, под, таван
- врати и каси на врати
- прозорци и пердета
- радиатори, контакти, ключове

За всяка мебел/предмет върни bounding box в НОРМАЛИЗИРАНИ координати (0.0 до 1.0):
- x, y = горен ляв ъгъл (като дял от ширината/височината на снимката)
- w, h = ширина и височина на кутията (като дял)
Кутията трябва да покрива ЦЕЛИЯ предмет, включително краката/основата му, с малък запас.

Върни САМО валиден JSON, без Markdown и обяснения:
{"boxes": [{"label": "tv", "x": 0.05, "y": 0.30, "w": 0.30, "h": 0.25}, ...]}
Ако няма мебели, върни {"boxes": []}."""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    data = json.loads(raw)
    boxes = data.get("boxes", [])
    valid = []
    for b in boxes:
        try:
            x = float(b["x"]); y = float(b["y"]); w = float(b["w"]); h = float(b["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        valid.append({
            "label": str(b.get("label", "item")),
            "x": min(max(x, 0.0), 1.0),
            "y": min(max(y, 0.0), 1.0),
            "w": min(max(w, 0.0), 1.0),
            "h": min(max(h, 0.0), 1.0),
        })
    return valid


def inpaint_furniture(image_rgb: np.ndarray, boxes: List[Dict]) -> np.ndarray:
    """Заличава маркираните зони чрез OpenCV inpainting."""
    h_img, w_img = image_rgb.shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    for b in boxes:
        x1 = int(b["x"] * w_img)
        y1 = int(b["y"] * h_img)
        x2 = int((b["x"] + b["w"]) * w_img)
        y2 = int((b["y"] + b["h"]) * h_img)
        pad_x = int(0.02 * w_img)
        pad_y = int(0.02 * h_img)
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(w_img, x2 + pad_x); y2 = min(h_img, y2 + pad_y)
        mask[y1:y2, x1:x2] = 255

    if not np.any(mask):
        return image_rgb

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(bgr, mask, inpaintRadius=15, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)


def encode_image_jpeg(image_rgb: np.ndarray, quality: int = 85) -> str:
    """Кодира numpy RGB изображение като base64 JPEG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
