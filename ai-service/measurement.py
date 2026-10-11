"""
Локално измерване на помещение от снимка с OpenCV.
Използва мащабен маркер (A4, банкнота, кутия цигари) за калибрация
и перспективна евристика за оценка на размерите.
"""
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


MARKER_SIZES_CM = {
    "a4": {"width": 21.0, "height": 29.7, "label": "A4 лист", "ratio": 29.7 / 21.0},
    "cigarettes": {"width": 8.8, "height": 5.5, "label": "Кутия цигари", "ratio": 8.8 / 5.5},
    "banknote": {"width": 14.7, "height": 8.2, "label": "Банкнота €50", "ratio": 14.7 / 8.2},
}


def detect_marker(
    image: np.ndarray,
    marker_type: str,
    custom_size_cm: float = 29.7,
) -> Optional[Dict]:
    """
    Опитва се да открие правоъгълен маркер в снимката.
    Връща dict с px_per_cm и bbox, или None ако не е намерен.
    """
    if marker_type in MARKER_SIZES_CM:
        marker = MARKER_SIZES_CM[marker_type]
        real_width = marker["width"]
        real_height = marker["height"]
        expected_ratio = marker["ratio"]
    else:
        real_width = custom_size_cm
        real_height = custom_size_cm
        expected_ratio = 1.0

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Намираме контурите
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h * 0.001):  # твърде малък
            continue
        if area > (w * h * 0.4):  # твърде голям
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue

        # Проверяваме дали е изпъкнал четириъгълник
        if not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        # Изчисляваме дължините на страните
        d = [
            np.linalg.norm(pts[i] - pts[(i + 1) % 4])
            for i in range(4)
        ]
        long = max(d[0], d[1], d[2], d[3])
        short = min(d[0], d[1], d[2], d[3])
        if short < 1:
            continue

        ratio = long / short
        # Проверяваме съотношението спрямо очакваното
        ratio_diff = abs(ratio - expected_ratio) / max(expected_ratio, 1e-6)
        if ratio_diff > 0.35:
            continue

        # Изчисляваме px_per_cm по дългата страна
        px_per_cm_long = long / max(real_height, real_width)
        px_per_cm_short = short / min(real_height, real_width)
        px_per_cm = (px_per_cm_long + px_per_cm_short) / 2

        candidates.append({
            "px_per_cm": px_per_cm,
            "ratio_diff": ratio_diff,
            "area": area,
            "bbox": pts,
            "long_px": long,
            "short_px": short,
        })

    if not candidates:
        return None

    # Избираме кандидата с най-малка разлика в съотношението и най-голяма площ
    candidates.sort(key=lambda c: (c["ratio_diff"], -c["area"]))
    best = candidates[0]
    return {
        "px_per_cm": best["px_per_cm"],
        "bbox": best["bbox"],
        "long_px": best["long_px"],
        "short_px": best["short_px"],
    }


def detect_floor_lines(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Открива ръбове и линии в снимката, които може да са ръбове на стая.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=max(image.shape[1] // 10, 80),
        maxLineGap=20,
    )
    return edges, lines if lines is not None else np.array([])


def estimate_room_from_image(
    image: np.ndarray,
    marker_type: str,
    custom_size_cm: float = 29.7,
) -> Dict:
    """
    Оценява размерите на стая от снимка.
    Ако открие маркер, го използва за калибрация.
    Връща dict с width_cm, length_cm, height_cm, px_per_cm, confidence, note.
    """
    h, w = image.shape[:2]

    # Опитваме се да открием маркер
    marker = detect_marker(image, marker_type, custom_size_cm)

    if marker is None:
        # Fallback: приемаме, че стаята заема целия кадър и използваме груба евристика
        return _fallback_estimate(h, w, marker_type, custom_size_cm)

    px_per_cm = marker["px_per_cm"]
    bbox = marker["bbox"]

    # Център на маркера в кадъра
    center_y = float(np.mean(bbox[:, 1]))
    # Нормализирана позиция: 0 = долу (близо до камерата), 1 = горе (далечна стена)
    # Ако маркерът е на пода, по-близо до дъното на кадъра = по-близо до камерата.
    normalized_y = center_y / h

    # Корекционен фактор за перспектива.
    # Маркерът обикновено е по-близо до камерата от задната стена.
    # Ако е в долната част, px_per_cm е завишен (обектите изглеждат по-големи),
    # затова увеличаваме оценените размери.
    # Ако е в горната половина, корекцията е по-малка.
    perspective_factor = 1.5 + normalized_y * 1.5  # варира от ~1.5 до ~3.0

    # Ширина на видимото поле на пода в cm, коригирана с перспектива
    visible_floor_width_cm = (w / px_per_cm) * perspective_factor

    # Височина на стаята — коригираме височината на кадъра
    visible_height_cm = (h / px_per_cm) * perspective_factor * 0.55

    # Дължина на стаята (дълбочина) — по-голяма от ширината, защото камерата е в ъгъл
    length_cm = visible_floor_width_cm * 0.95

    # Ширина на стаята
    width_cm = visible_floor_width_cm * 0.85

    # Височина
    height_cm = visible_height_cm

    # Ограничения за реалистичност
    width_cm = max(min(width_cm, 1200), 200)
    length_cm = max(min(length_cm, 1200), 200)
    height_cm = max(min(height_cm, 400), 220)

    return {
        "width_cm": round(width_cm, 1),
        "length_cm": round(length_cm, 1),
        "height_cm": round(height_cm, 1),
        "px_per_cm": round(px_per_cm, 2),
        "marker_label": MARKER_SIZES_CM.get(marker_type, {}).get("label", "Ръчен маркер"),
        "confidence": 0.65,
        "note": (
            "Измерване с мащабен маркер. "
            "Поставете маркерът равен на пода, близо до далечната стена, за най-добра точност. "
            "Резултатът е приблизителен — проверете и коригирайте ръчно."
        ),
    }


def _fallback_estimate(h: int, w: int, marker_type: str, custom_size_cm: float) -> Dict:
    """
    Груба евристика, когато маркер не е открит.
    """
    if marker_type in MARKER_SIZES_CM:
        marker = MARKER_SIZES_CM[marker_type]
        marker_long_edge = max(marker["width"], marker["height"])
    else:
        marker_long_edge = custom_size_cm

    assumed_marker_px = w * 0.10
    px_per_cm = assumed_marker_px / marker_long_edge

    fov_horizontal_deg = 70.0
    distance_to_far_wall_cm = 400.0

    fov_rad = math.radians(fov_horizontal_deg)
    visible_width_cm = 2 * distance_to_far_wall_cm * math.tan(fov_rad / 2)

    aspect = h / w
    visible_height_cm = visible_width_cm * aspect

    width_cm = round(visible_width_cm / 10) * 10
    length_cm = round((visible_width_cm * 0.85) / 10) * 10
    height_cm = round((visible_height_cm * 0.55) / 10) * 10

    width_cm = max(width_cm, 200)
    length_cm = max(length_cm, 200)
    height_cm = max(height_cm, 230)

    return {
        "width_cm": width_cm,
        "height_cm": height_cm,
        "length_cm": length_cm,
        "px_per_cm": round(px_per_cm, 2),
        "marker_label": MARKER_SIZES_CM.get(marker_type, {}).get("label", "Ръчен маркер"),
        "confidence": 0.35,
        "note": (
            "Маркерът не беше открит. Използват се фиксирани евристични стойности. "
            "Поставете A4 лист или банкнота на пода в кадъра за по-точно измерване."
        ),
    }
