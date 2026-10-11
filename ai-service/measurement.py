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
    Връща dict с px_per_cm, bbox и debug информация, или None ако не е намерен.
    Поддържа маркер както на пода, така и на стената.
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
    img_area = w * h
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    debug_info = {
        "tried_methods": [],
        "contours_checked": 0,
        "candidates_count": 0,
        "best_ratio_diff": None,
    }

    # Опитваме няколко метода за откриване на ръбове
    edge_methods = [
        ("canny_default", cv2.Canny(blurred, 50, 150)),
        ("canny_low", cv2.Canny(blurred, 20, 80)),
        ("canny_high", cv2.Canny(blurred, 100, 250)),
    ]

    # Адаптивен threshold — полезен, когато листът е по-светъл/по-тъмен от фона
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    edge_methods.append(("adaptive", adaptive))

    candidates = []
    for method_name, binary in edge_methods:
        debug_info["tried_methods"].append(method_name)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            debug_info["contours_checked"] += 1
            area = cv2.contourArea(cnt)
            if area < (img_area * 0.0005):  # по-малък праг
                continue
            if area > (img_area * 0.5):  # по-висок горен праг
                continue

            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue

            # Позволяваме по-груба апроксимация за перспективно изкривени маркери
            approx = cv2.approxPolyDP(cnt, 0.07 * peri, True)
            if len(approx) < 4 or len(approx) > 6:
                continue

            # Взимаме bounding quad — минимален обгръщащ четириъгълник
            pts = approx.reshape(-1, 2).astype(np.float32)
            if len(pts) > 4:
                # Ако имаме повече от 4 точки, взимаме minAreaRect
                rect = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect)
                pts = box.astype(np.float32)

            if not cv2.isContourConvex(pts.astype(np.int32)):
                continue

            # Изчисляваме дължините на страните
            d = [
                np.linalg.norm(pts[i] - pts[(i + 1) % 4])
                for i in range(4)
            ]
            long = max(d)
            short = min(d)
            if short < 1:
                continue

            ratio = long / short
            # Проверяваме съотношението спрямо очакваното и неговото обратно
            ratio_diff_normal = abs(ratio - expected_ratio) / max(expected_ratio, 1e-6)
            ratio_diff_inverse = abs(ratio - (1.0 / expected_ratio)) / max(1.0 / expected_ratio, 1e-6)
            ratio_diff = min(ratio_diff_normal, ratio_diff_inverse)

            if ratio_diff > 0.55:  # по-мек праг
                continue

            # Определяме коя страна е дълга/къса спрямо реалните размери
            if ratio > 1.0:
                px_per_cm_long = long / max(real_height, real_width)
                px_per_cm_short = short / min(real_height, real_width)
            else:
                px_per_cm_long = long / min(real_height, real_width)
                px_per_cm_short = short / max(real_height, real_width)
            px_per_cm = (px_per_cm_long + px_per_cm_short) / 2

            candidates.append({
                "px_per_cm": px_per_cm,
                "ratio_diff": ratio_diff,
                "area": area,
                "bbox": pts,
                "long_px": long,
                "short_px": short,
                "method": method_name,
            })

    debug_info["candidates_count"] = len(candidates)
    if candidates:
        debug_info["best_ratio_diff"] = min(c["ratio_diff"] for c in candidates)

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
        "debug": debug_info,
        "method": best["method"],
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
    Ако открие маркер, го използва за калибрация чрез пинхол камерна модел.
    Връща dict с width_cm, length_cm, height_cm, px_per_cm, confidence, note, debug.
    """
    h, w = image.shape[:2]

    # Опитваме се да открием маркер
    marker = detect_marker(image, marker_type, custom_size_cm)

    if marker is None:
        # Fallback: приемаме, че стаята заема целия кадър и използваме груба евристика
        result = _fallback_estimate(h, w, marker_type, custom_size_cm)
        result["debug"] = {"marker_found": False}
        return result

    px_per_cm = marker["px_per_cm"]
    bbox = marker["bbox"]
    debug = marker.get("debug", {})
    debug["marker_found"] = True
    debug["method"] = marker.get("method", "unknown")

    if marker_type in MARKER_SIZES_CM:
        marker_real_long = max(MARKER_SIZES_CM[marker_type]["width"], MARKER_SIZES_CM[marker_type]["height"])
    else:
        marker_real_long = custom_size_cm

    marker_image_long = marker["long_px"]

    # Пинхол модел: focal_length = image_width / (2 * tan(FOV/2))
    # Използваме 80° FOV, защото повечето телефони имат по-широко зрително поле.
    fov_horizontal_deg = 80.0
    fov_rad = math.radians(fov_horizontal_deg)
    focal_length_px = w / (2 * math.tan(fov_rad / 2))

    # Разстояние до маркера
    distance_to_marker_cm = focal_length_px * marker_real_long / marker_image_long

    # Позиция на маркера в кадъра - колко е „навътре“ в стаята
    center_y = float(np.mean(bbox[:, 1]))
    normalized_y = center_y / h  # 0 = горе (далеч), 1 = долу (близо)

    # Предполагаме, че задната стена е на няколко пъти по-далеч от маркера.
    # Ако маркерът е в горната част на кадъра (далеч), задната стена е малко по-далеч.
    # Ако е в долната част (близо), задната стена е значително по-далеч.
    depth_multiplier = 1.3 + (1 - normalized_y) * 1.2  # варира от 1.3 до 2.5
    distance_to_far_wall_cm = min(distance_to_marker_cm * depth_multiplier, 500.0)

    # Ширина на видимото поле на задната стена
    visible_width_cm = 2 * distance_to_far_wall_cm * math.tan(fov_rad / 2)

    # Височина на видимото поле
    aspect = h / w
    visible_height_cm = visible_width_cm * aspect

    # Оценка на размерите на стаята
    width_cm = visible_width_cm * 0.7
    length_cm = distance_to_far_wall_cm * 0.85
    height_cm = visible_height_cm * 0.45

    # Ограничения за реалистичност
    width_cm = max(min(width_cm, 800), 200)
    length_cm = max(min(length_cm, 800), 200)
    height_cm = max(min(height_cm, 320), 220)

    debug["distance_to_marker_cm"] = round(distance_to_marker_cm, 1)
    debug["distance_to_far_wall_cm"] = round(distance_to_far_wall_cm, 1)
    debug["focal_length_px"] = round(focal_length_px, 1)

    return {
        "width_cm": round(width_cm, 1),
        "length_cm": round(length_cm, 1),
        "height_cm": round(height_cm, 1),
        "px_per_cm": round(px_per_cm, 2),
        "marker_label": MARKER_SIZES_CM.get(marker_type, {}).get("label", "Ръчен маркер"),
        "confidence": 0.65,
        "note": (
            "Измерване с мащабен маркер. "
            "Поставете маркерът равен на пода или стената, близо до далечната стена, за най-добра точност. "
            "Резултатът е приблизителен — проверете и коригирайте ръчно."
        ),
        "debug": debug,
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
            "Поставете A4 лист или банкнота на пода или стената в кадъра за по-точно измерване."
        ),
    }
