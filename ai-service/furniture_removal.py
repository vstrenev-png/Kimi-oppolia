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
- мебели: шкафове, гардероби, комоди, скринове, легла, дивани, маси, столове, рафтове, ТВ шкафове
- техника: телевизори, монитори, лаптопи, лампи
- предмети, подпрени на стени: дъски, плоскости, панели, огледала
- помощни средства и инструменти: проходилки, стълби, сгъваеми столове
- декорации и дребни предмети върху мебели и по стените: картини, постери, растения, играчки, кутии, бутилки

НЕ МАРКИРАЙ (това е част от празната стая):
- стени, под, таван
- врати и каси на врати — НЕ покривай нито сантиметър от врата или касата!
- прозорци и пердета
- радиатори, контакти, ключове

За всяка мебел/предмет върни bounding box в НОРМАЛИЗИРАНИ координати (0.0 до 1.0):
- x, y = горен ляв ъгъл (като дял от ширината/височината на снимката)
- w, h = ширина и височина на кутията (като дял)
Кутията трябва да покрива ЦЕЛИЯ предмет плътно — от краката до върха, но БЕЗ да застъпва врати, прозорци и радиатори.

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


def _boxes_to_mask(boxes: List[Dict], w: int, h: int, pad_frac: float = 0.05) -> np.ndarray:
    """uint8 маска (255 = зона за заличаване) от нормализирани bounding boxes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x1 = int(b["x"] * w)
        y1 = int(b["y"] * h)
        x2 = int((b["x"] + b["w"]) * w)
        y2 = int((b["y"] + b["h"]) * h)
        pad_x = int(pad_frac * w)
        pad_y = int(pad_frac * h)
        # Ако кутията е близо до ръба, разширяваме до самия ръб
        if b["x"] + b["w"] > 0.80:
            x2 = w
        if b["y"] + b["h"] > 0.80:
            y2 = h
        if b["x"] < 0.20:
            x1 = 0
        if b["y"] < 0.20:
            y1 = 0
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
        mask[y1:y2, x1:x2] = 255
    return mask


def inpaint_furniture(image_rgb: np.ndarray, boxes: List[Dict]) -> np.ndarray:
    """Заличава маркираните зони чрез OpenCV inpainting."""
    h_img, w_img = image_rgb.shape[:2]
    mask = _boxes_to_mask(boxes, w_img, h_img, pad_frac=0.02)

    if not np.any(mask):
        return image_rgb

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(bgr, mask, inpaintRadius=15, flags=cv2.INPAINT_NS)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)


def composite_with_original(
    original_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    boxes: List[Dict],
) -> np.ndarray:
    """
    Връща оригиналните пиксели навсякъде ИЗВЪН маскираните зони.
    AI резултатът се използва само там, където е имало мебели —
    врати, прозорци, стени и всичко останало остава от оригиналната снимка.
    """
    h, w = original_rgb.shape[:2]
    gen = cv2.resize(generated_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    mask = _boxes_to_mask(boxes, w, h)
    # Мека граница само в тясна ивица (~±6px) около маската
    soft = cv2.GaussianBlur(mask, (13, 13), 0).astype(np.float32) / 255.0
    soft3 = soft[..., None]
    blended = np.empty_like(original_rgb, dtype=np.float32)
    np.multiply(gen, soft3, out=blended)
    blended += original_rgb.astype(np.float32) * (1.0 - soft3)
    out = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    # Гаранция: пикселите далеч от маската са пиксел-по-пиксел оригинални
    unchanged = soft < 0.01
    out[unchanged] = original_rgb[unchanged]
    return out


def encode_image_jpeg(image_rgb: np.ndarray, quality: int = 85) -> str:
    """Кодира numpy RGB изображение като base64 JPEG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generative_remove_furniture(image_rgb: np.ndarray, boxes: List[Dict]) -> Optional[np.ndarray]:
    """
    Генеративно премахване на мебели чрез OpenAI gpt-image-1 edit с маска.
    Връща изчистеното изображение или None при грешка.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI
    from PIL import Image

    h_img, w_img = image_rgb.shape[:2]

    # Ограничаваме до 1024px по дългата страна за цена/скорост
    scale = min(1.0, 1024 / max(h_img, w_img))
    new_w, new_h = int(w_img * scale), int(h_img * scale)
    pil_img = Image.fromarray(image_rgb).resize((new_w, new_h), Image.LANCZOS)

    if new_w >= new_h * 1.2:
        target_aspect, out_size = 1.5, "1536x1024"
    elif new_h >= new_w * 1.2:
        target_aspect, out_size = 2 / 3, "1024x1536"
    else:
        target_aspect, out_size = 1.0, "1024x1024"

    # Подравняваме съотношението с replicate-padding, за да НЯМА деформация
    # при връщането на резултата към оригиналния размер
    top = bottom = left = right = 0
    cur_aspect = new_w / new_h
    if abs(cur_aspect - target_aspect) > 0.02:
        arr = np.array(pil_img)
        if cur_aspect > target_aspect:
            target_h = int(round(new_w / target_aspect))
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            arr = cv2.copyMakeBorder(arr, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
        else:
            target_w = int(round(new_h * target_aspect))
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            arr = cv2.copyMakeBorder(arr, 0, 0, left, right, cv2.BORDER_REPLICATE)
        pil_img = Image.fromarray(arr)
    pad_w, pad_h = pil_img.size

    # Маска: RGBA, alpha=0 = зона за редактиране; padding зоната се пази
    mask_np = _boxes_to_mask(boxes, new_w, new_h)
    if top or bottom or left or right:
        mask_np = cv2.copyMakeBorder(mask_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    mask_arr = np.zeros((pad_h, pad_w, 4), dtype=np.uint8)
    mask_arr[..., 3] = 255 - mask_np
    mask = Image.fromarray(mask_arr, "RGBA")

    img_buf = io.BytesIO()
    pil_img.convert("RGBA").save(img_buf, format="PNG")
    img_buf.seek(0)
    img_buf.name = "room.png"
    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_buf.seek(0)
    mask_buf.name = "mask.png"

    prompt = (
        "Fill the masked areas by continuing the surrounding plain wall, floor or ceiling surface. "
        "STRICTLY FORBIDDEN: do not create windows, doors, radiators, curtains, furniture, decorations "
        "or any new features in the masked areas — only flat, empty wall/floor/ceiling matching the "
        "surrounding color, texture, perspective and lighting."
    )

    client = OpenAI(api_key=api_key)
    quality = os.environ.get("CLEAN_IMAGE_QUALITY", "medium")
    result = client.images.edit(
        model="gpt-image-1",
        image=img_buf,
        mask=mask_buf,
        prompt=prompt,
        size=out_size,
        quality=quality,
    )
    b64_out = result.data[0].b64_json
    generated = np.array(Image.open(io.BytesIO(base64.b64decode(b64_out))).convert("RGB"))

    # Връщаме към подплатения размер и изрязваме padding-а —
    # резултатът има същото съотношение като оригинала
    generated = cv2.resize(generated, (pad_w, pad_h), interpolation=cv2.INTER_LANCZOS4)
    if top or bottom or left or right:
        generated = generated[top:top + new_h, left:left + new_w]
    return generated
