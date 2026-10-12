"""
Интеграция с OpenAI GPT-4o Vision за AI измерване на помещение от снимка.
"""
import json
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

    prompt = f"""Ти си експерт по интериорни измервания и компютърно зрение.

Анализирай приложената снимка на помещение. В кадъра има {marker_desc}, който можеш да използваш като референция за мащаб.

Задача:
1. Определи приблизителните реални размери на помещението: дължина, ширина и височина в сантиметри.
2. Изчисли мащаба px_per_cm (пиксели на сантиметър) според маркера.
3. Оцени confidence — колко сигурен си в резултата (от 0.0 до 1.0).

Върни САМО валиден JSON обект в следния формат, без Markdown, без обяснения извън JSON:
{{
    "width_cm": <число>,
    "length_cm": <число>,
    "height_cm": <число>,
    "px_per_cm": <число>,
    "confidence": <число между 0 и 1>,
    "note": "кратко обяснение на базата на какво си определил размерите"
}}

Ако не виждаш маркер или не можеш да определиш размерите, върни приблизителни стойности и ниска confidence."""

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

    return {
        "width_cm": float(data.get("width_cm", 400)),
        "length_cm": float(data.get("length_cm", 400)),
        "height_cm": float(data.get("height_cm", 260)),
        "px_per_cm": float(data.get("px_per_cm", 10)),
        "confidence": float(data.get("confidence", 0.5)),
        "note": str(data.get("note", "AI измерване с OpenAI Vision.")),
        "marker_label": marker_info.get(marker_type, "Ръчен маркер"),
    }
