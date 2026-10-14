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

Освен това върни и "protect" списък с bounding boxes (същия формат) на всички елементи, които трябва да останат НЕПИПНАТИ: врати с касите им, прозорци, пердета, радиатори. ЗАДЪЛЖИТЕЛНО включи в protect ВСЯКА видима врата, прозорец, перде или радиатор — дори частично видим. Protect кутията трябва да покрива ТОЧНО елемента с рамката/касата му — нито по-тясна, нито по-широка. НЕ включвай мебели или стени в protect кутиите.

Върни САМО валиден JSON, без Markdown и обяснения:
{"boxes": [{"label": "tv", "x": 0.05, "y": 0.30, "w": 0.30, "h": 0.25}, ...],
 "protect": [{"label": "door", "x": 0.60, "y": 0.15, "w": 0.15, "h": 0.70}, ...]}
Ако няма мебели, върни {"boxes": [], "protect": []}."""

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
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    data = json.loads(raw)

    def _parse(items):
        valid = []
        for b in items:
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

    return {"boxes": _parse(data.get("boxes", [])), "protect": _parse(data.get("protect", []))}


def _boxes_to_mask(boxes: List[Dict], w: int, h: int, pad_frac: float = 0.02, protect: Optional[List[Dict]] = None) -> np.ndarray:
    """uint8 маска (255 = зона за заличаване) от нормализирани bounding boxes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x1 = int(b["x"] * w)
        y1 = int(b["y"] * h)
        x2 = int((b["x"] + b["w"]) * w)
        y2 = int((b["y"] + b["h"]) * h)
        pad_x = int(pad_frac * w)
        pad_y = int(pad_frac * h)
        # Разширяваме до ръба само ако кутията наистина го докосва —
        # иначе рискуваме да погълнем ъгли на стени, които трябва да останат.
        # Към пода (долен ръб) сме по-щедри — там няма ъгли за пазене.
        if b["x"] + b["w"] > 0.93:
            x2 = w
        if b["y"] + b["h"] > 0.88:
            y2 = h
        if b["x"] < 0.07:
            x1 = 0
        if b["y"] < 0.07:
            y1 = 0
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
        mask[y1:y2, x1:x2] = 255

    # Изваждаме защитените зони (врати, прозорци, радиатори) от маската —
    # те гарантирано остават непипнати. Ако мебелна кутия само леко застъпва
    # защитен елемент (<30% от него), това е утечка → пълна защита.
    # Ако застъпването е голямо, мебелта е пред елемента → пазим само
    # свободната част на защитената зона.
    if protect:
        furniture = mask > 0
        pmask = np.zeros((h, w), dtype=np.uint8)
        for b in protect:
            px1 = max(0, int(b["x"] * w) - int(0.04 * w))
            py1 = max(0, int(b["y"] * h) - int(0.04 * h))
            px2 = min(w, int((b["x"] + b["w"]) * w) + int(0.04 * w))
            py2 = min(h, int((b["y"] + b["h"]) * h) + int(0.04 * h))
            pb = np.zeros((h, w), dtype=bool)
            pb[py1:py2, px1:px2] = True
            overlap = (pb & furniture).sum() / max(pb.sum(), 1)
            if overlap < 0.40:
                pmask[pb] = 255
            else:
                pmask[pb & ~furniture] = 255
        mask[pmask > 0] = 0
    return mask


def inpaint_furniture(image_rgb: np.ndarray, boxes: List[Dict], protect: Optional[List[Dict]] = None) -> np.ndarray:
    """Заличава маркираните зони чрез OpenCV inpainting."""
    h_img, w_img = image_rgb.shape[:2]
    mask = _boxes_to_mask(boxes, w_img, h_img, pad_frac=0.02, protect=protect)

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
    protect: Optional[List[Dict]] = None,
) -> np.ndarray:
    """
    Връща оригиналните пиксели навсякъде ИЗВЪН маскираните зони.
    AI резултатът се използва само там, където е имало мебели —
    врати, прозорци, стени и всичко останало остава от оригиналната снимка.
    """
    h, w = original_rgb.shape[:2]
    gen = cv2.resize(generated_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    mask = _boxes_to_mask(boxes, w, h, protect=protect)

    kernel = np.ones((31, 31), np.uint8)
    pmask = np.zeros((h, w), dtype=bool)
    if protect:
        for b in protect:
            px1 = max(0, int(b["x"] * w) - int(0.025 * w)); py1 = max(0, int(b["y"] * h) - int(0.025 * h))
            px2 = min(w, int((b["x"] + b["w"]) * w) + int(0.025 * w)); py2 = min(h, int((b["y"] + b["h"]) * h) + int(0.025 * h))
            pmask[py1:py2, px1:px2] = True

    # Цветова калибрация ПО ЗОНИ: извадка само от ЛЯВО и ДЯСНО на зоната
    # (същата стена, същата височина) — така подът/таванът/вратите не
    # замърсяват медианата. Медиана е робастна на малки чужди пиксели.
    n_labels, labels = cv2.connectedComponents(mask)
    gen_f = gen.astype(np.float32)
    band = 30
    for lbl in range(1, n_labels):
        comp_mask = labels == lbl
        ys = np.where(comp_mask.any(axis=1))[0]
        xs = np.where(comp_mask.any(axis=0))[0]
        if len(ys) == 0 or len(xs) == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        def _median_diff(col_start, col_end):
            cols = np.zeros((h, w), dtype=bool)
            col_start = max(0, col_start); col_end = min(w, col_end)
            if col_end <= col_start:
                return None
            cols[y0:y1 + 1, col_start:col_end] = True
            region = cols & (mask == 0) & ~pmask
            if region.sum() < 50:
                return None
            return (np.median(original_rgb[region].reshape(-1, 3), axis=0)
                    - np.median(gen[region].reshape(-1, 3), axis=0))

        d_left = _median_diff(x0 - band, x0)
        d_right = _median_diff(x1 + 1, x1 + 1 + band)
        if d_left is None and d_right is None:
            # Зона до ръба — fallback: пълен пръстен без пода/тавана
            comp = comp_mask.astype(np.uint8) * 255
            dil = cv2.dilate(comp, kernel, iterations=1) > 0
            ring = dil & (mask == 0) & ~pmask
            if ring.sum() < 50:
                continue
            offset = (np.median(original_rgb[ring].reshape(-1, 3), axis=0)
                      - np.median(gen[ring].reshape(-1, 3), axis=0))
        elif d_left is None:
            offset = d_right
        elif d_right is None:
            offset = d_left
        else:
            offset = (d_left + d_right) / 2
        gen_f[comp_mask] += offset[None, :]
    gen = np.clip(gen_f, 0, 255).astype(np.uint8)

    # Меко прехвърляне (~±6px) — преходът е мек, но не размазва върху вратите
    soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=6).astype(np.float32) / 255.0
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


def generative_remove_furniture(image_rgb: np.ndarray, boxes: List[Dict], protect: Optional[List[Dict]] = None) -> Optional[np.ndarray]:
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
    mask_np = _boxes_to_mask(boxes, new_w, new_h, protect=protect)
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
        "If a wall corner or the line where wall meets floor passes through a masked area, continue it "
        "straight and in the correct perspective. "
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


def _prepare_for_edit(image_rgb: np.ndarray):
    """Resize до 1024 + replicate-padding към поддържано съотношение."""
    from PIL import Image
    h_img, w_img = image_rgb.shape[:2]
    scale = min(1.0, 1024 / max(h_img, w_img))
    new_w, new_h = int(w_img * scale), int(h_img * scale)
    pil_img = Image.fromarray(image_rgb).resize((new_w, new_h), Image.LANCZOS)

    if new_w >= new_h * 1.2:
        target_aspect, out_size = 1.5, "1536x1024"
    elif new_h >= new_w * 1.2:
        target_aspect, out_size = 2 / 3, "1024x1536"
    else:
        target_aspect, out_size = 1.0, "1024x1024"

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
    return pil_img, out_size, (top, left, new_w, new_h)


def generative_remove_furniture_nomask(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Пълна AI реконструкция БЕЗ маска — gpt-image-1 със стриктен prompt да
    запази стаята идентична. Няма маскови шевове/петна, но изисква проверка.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    import io as _io
    from openai import OpenAI
    from PIL import Image

    pil_img, out_size, (top, left, new_w, new_h) = _prepare_for_edit(image_rgb)
    pad_w, pad_h = pil_img.size

    img_buf = _io.BytesIO()
    pil_img.convert("RGBA").save(img_buf, format="PNG")
    img_buf.seek(0)
    img_buf.name = "room.png"

    prompt = (
        "Remove ALL furniture and movable objects from this room photo: cabinets, wardrobes, dressers, "
        "TVs, shelves, tables, chairs, beds, sofas, boards, panels leaning on walls, walkers, ladders, "
        "boxes, bottles, lamps, paintings and decorations. The result must be the SAME room, empty. "
        "Keep absolutely IDENTICAL and unchanged: the walls (exact same color, texture, corners and angles), "
        "the floor, the ceiling, ALL doors with their exact frames, handles, locks and glass, windows with "
        "curtains, radiators, sockets, switches, the lighting and the perspective. "
        "Do NOT move, resize, redraw or restyle any door or window. Do NOT add anything new. "
        "Photorealistic, identical photo style and grain."
    )

    client = OpenAI(api_key=api_key)
    quality = os.environ.get("CLEAN_IMAGE_QUALITY", "medium")
    result = client.images.edit(
        model="gpt-image-1",
        image=img_buf,
        prompt=prompt,
        size=out_size,
        quality=quality,
    )
    b64_out = result.data[0].b64_json
    generated = np.array(Image.open(_io.BytesIO(base64.b64decode(b64_out))).convert("RGB"))
    generated = cv2.resize(generated, (pad_w, pad_h), interpolation=cv2.INTER_LANCZOS4)
    if top or bottom or left or right:
        generated = generated[top:top + new_h, left:left + new_w]
    return generated


def verify_cleaned_image(orig_b64: str, cleaned_b64: str) -> Optional[Dict]:
    """
    GPT-4o сравнява оригинала с изчистения резултат.
    structure_ok=false ако врата/прозорец/стена/ъгъл е променен.
    furniture_left=true ако има останали мебели.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    prompt = """Сравняваш две снимки на едно и също помещение: ПЪРВАТА е оригинал, ВТОРАТА е редактирана версия, от която трябва да са премахнати всички мебели.

Отговори със САМО валиден JSON:
{
 "structure_ok": true/false,
 "furniture_left": true/false,
 "details": "кратко описание"
}

structure_ok = true САМО ако:
- ВСИЧКИ врати (с рамки, каси, дръжки, стъкла) изглеждат идентично — същата позиция, форма, цвят, ширина на рамките
- прозорци, пердета, радиатори са идентични
- ъглите на стените и линиите под/таван са на същите места
- няма новодобавени елементи (нови прозорци, врати, мебели)

furniture_left = true ако във втората снимка остава каквато и да е мебел или предмет от първата.

Бъди СТРИКТЕН: всяка видима разлика във врата, прозорец или стена означава structure_ok=false."""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cleaned_b64}", "detail": "high"}},
                ],
            }
        ],
        max_tokens=300,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)
