"""
Интеграция с OpenAI GPT-4o Vision за AI измерване на помещение от снимка.
"""
import json
import math
import os
from typing import Dict, Optional


def estimate_with_openai_vision(
    base64_image: str,
    marker_type: str,
    custom_size_cm: float = 29.7,
) -> Optional[Dict]:
    """
    Изпраща снимка към OpenAI GPT-4o Vision и връща приблизителни размери на стаята.
    Ако OPENAI_API_KEY не е зададен, връща None.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai пакетът не е инсталиран.")

    marker_info = {
        "a4": "A4 лист с размери 21.0 x 29.7 cm",
        "cigarettes": "стандартна кутия цигари с дължина на по-дългата страна около 8.8 cm",
        "banknote": "разгъната банкнота от 50 евро/лева с дължина около 14.7 cm",
    }
    marker_desc = marker_info.get(marker_type, f"мащабен маркер с дължина {custom_size_cm} cm")

    prompt = f"""Ти си експерт по интериорни измервания и компютърно зрение. Твоята задача е да определиш реалните размери на помещението от една снимка.

ВАЖНИ ПРАВИЛА:
- В кадъра трябва да има {marker_desc}. Ако го виждаш, ЗАДЪЛЖИТЕЛНО го използвай като референция за мащаб.
- width_cm = разстояние между лявата и дясната стена (хоризонтална ширина).
- length_cm = дълбочина на стаята — разстояние от камерата до най-далечната видима стена.
- height_cm = височина от пода до тавана.
- Бъди консервативен. Типична стая е 250–600 cm в ширина/дължина и 220–300 cm във височина. Избягвай стойности над 800 cm, освен ако снимката ясно показва огромно помещение.
- Ако не виждаш маркер или снимката е неясна, върни приблизителни стойности и НИСКА confidence (под 0.4).

Задача:
1. Определи реалните размери на помещението в cm.
2. Изчисли px_per_cm според маркера (ако е видим).
3. Оцени confidence (0.0 до 1.0).

Върни САМО валиден JSON обект, без Markdown, без обяснения извън JSON:
{{
    "width_cm": <число>,
    "length_cm": <число>,
    "height_cm": <число>,
    "px_per_cm": <число>,
    "confidence": <число между 0 и 1>,
    "note": "кратко обяснение"
}}"""

    try:
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
            max_tokens=500,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()

        # Почистваме евентуални Markdown code blocks
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        data = json.loads(raw)

        def _float_or(value, default):
            try:
                return float(value) if value is not None else float(default)
            except (TypeError, ValueError):
                return float(default)

        return {
            "width_cm": _float_or(data.get("width_cm"), 400),
            "length_cm": _float_or(data.get("length_cm"), 400),
            "height_cm": _float_or(data.get("height_cm"), 260),
            "px_per_cm": _float_or(data.get("px_per_cm"), 10),
            "confidence": _float_or(data.get("confidence"), 0.5),
            "note": str(data.get("note", "AI измерване с OpenAI Vision.")),
            "marker_label": marker_info.get(marker_type, "Ръчен маркер"),
        }
    except Exception as exc:
        return {
            "width_cm": 400.0,
            "length_cm": 400.0,
            "height_cm": 260.0,
            "px_per_cm": 0.0,
            "confidence": 0.1,
            "note": f"OpenAI Vision грешка: {exc}. Проверете API ключа, кредитите и интернет връзката.",
            "marker_label": marker_info.get(marker_type, "Ръчен маркер"),
        }


MARKER_INFO = {
    "a4": ("A4 лист с размери 21.0 x 29.7 cm", 29.7),
    "cigarettes": ("стандартна кутия цигари (~8.8 cm дълга страна)", 8.8),
    "banknote": ("разгъната банкнота (~14.7 cm дълга страна)", 14.7),
}


def measure_with_vision_geometry(
    base64_image: str,
    marker_type: str,
    custom_size_cm: float,
    fov_deg: float,
    img_w: int,
    img_h: int,
) -> Optional[Dict]:
    """
    Геометрично измерване: GPT-4o засича маркера и линиите на стаята в
    пиксели, сметките се правят в код с пинхол модел и реален FOV от EXIF.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    marker_desc, marker_long = MARKER_INFO.get(
        marker_type, (f"мащабен маркер с дължина {custom_size_cm} cm", custom_size_cm)
    )

    prompt = f"""Ти си прецизна система за фотограметрия. Анализирай снимката и върни ГЕОМЕТРИЯТА ѝ в нормализирани координати (0.0-1.0 спрямо ширината/височината).

Търся {marker_desc} в кадъра.

Върни САМО валиден JSON:
{{
  "marker": {{"found": true/false, "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "on_floor": true/false, "at_far_wall": true/false}},
  "room": {{
    "floor_line_left_y": 0.0,
    "floor_line_right_y": 0.0,
    "left_wall_x": 0.0,
    "right_wall_x": 0.0,
    "ceiling_line_y": 0.0,
    "door_height_px_fraction": 0.0
  }},
  "notes": "кратко описание"
}}

Дефиниции (гледай ЗАДНАТА стена — тази срещу камерата):
- floor_line_left_y / floor_line_right_y: y на линията, където задната стена среща пода, при левия/десния край на стената
- left_wall_x / right_wall_x: x на лявата/дясната страна на задната стена при линията на пода
- ceiling_line_y: y където задната стена среща тавана
- door_height_px_fraction: ако има врата в кадъра — нейната височина като дял от височината на снимката (0.0 ако няма)
- marker.at_far_wall = true ако маркерът е точно до задната стена

Ако не виждаш маркер: "marker": {{"found": false}}. Ако не можеш да засечеш линиите — дай най-добрата си оценка."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                    ],
                }
            ],
            max_tokens=500,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
    except Exception:
        return None

    room = data.get("room") or {}
    marker = data.get("marker") or {}

    # Пинхол модел с реален FOV
    f_px = img_w / (2 * math.tan(math.radians(fov_deg) / 2))

    length_cm = None
    confidence = 0.3

    if marker.get("found"):
        m_w = abs(marker["x2"] - marker["x1"]) * img_w
        m_h = abs(marker["y2"] - marker["y1"]) * img_h
        marker_px = max(m_w, m_h)
        if marker_px > 5:
            dist_to_marker = f_px * marker_long / marker_px
            if marker.get("at_far_wall", True):
                length_cm = dist_to_marker
                confidence = 0.6
            else:
                # Маркер по-близо до камерата — задната стена е по-далеч
                length_cm = dist_to_marker * 1.4
                confidence = 0.45

    # Ширина на задната стена от нейния пикселен обхват
    width_cm = None
    if room.get("left_wall_x") is not None and room.get("right_wall_x") is not None:
        span_px = abs(room["right_wall_x"] - room["left_wall_x"]) * img_w
        if span_px > 50 and length_cm:
            width_cm = span_px * length_cm / f_px

    # Височина: линия под→таван на задната стена
    height_cm = None
    if room.get("ceiling_line_y") is not None:
        floor_y = max(room.get("floor_line_left_y") or 0, room.get("floor_line_right_y") or 0)
        ceil_span_px = abs(floor_y - room["ceiling_line_y"]) * img_h
        if ceil_span_px > 30 and length_cm:
            height_cm = ceil_span_px * length_cm / f_px

    # Врата като еталон (стандартна врата = 200 cm):
    # височината на стаята = 200 * (пиксели под→таван) / (пиксели на вратата)
    if height_cm is None and room.get("door_height_px_fraction", 0) > 0.15:
        door_px = room["door_height_px_fraction"] * img_h
        floor_y = max(room.get("floor_line_left_y") or 0, room.get("floor_line_right_y") or 0)
        if room.get("ceiling_line_y") is not None:
            ceil_span_px = abs(floor_y - room["ceiling_line_y"]) * img_h
            if ceil_span_px > 30 and door_px > 20:
                height_cm = 200.0 * ceil_span_px / door_px
                confidence = max(confidence, 0.5)

    if length_cm is None:
        return None

    # Реалистични граници
    width_cm = min(max(width_cm or 400, 150), 900)
    length_cm = min(max(length_cm, 150), 900)
    height_cm = min(max(height_cm or 260, 200), 400)

    return {
        "width_cm": round(width_cm, 1),
        "length_cm": round(length_cm, 1),
        "height_cm": round(height_cm, 1),
        "px_per_cm": round(f_px / max(length_cm, 1) , 2),
        "confidence": confidence,
        "note": f"Геометрично измерване (FOV {fov_deg:.0f}° от EXIF). {data.get('notes', '')} Приблизително — потвърдете с рулетка.",
        "marker_label": marker_desc,
        "debug": {"source": "vision_geometry", "fov_deg": fov_deg, "marker_found": bool(marker.get("found"))},
    }
