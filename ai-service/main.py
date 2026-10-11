import base64
import io
import math
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image


app = FastAPI(title="Design-Code AI Measurement Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MARKER_SIZES_CM = {
    "a4": {"width": 21.0, "height": 29.7, "label": "A4 лист"},
    "cigarettes": {"width": 8.8, "height": 5.5, "label": "Кутия цигари"},
    "banknote": {"width": 14.7, "height": 8.2, "label": "Банкнота €50"},
}


class AnalyzeRequest(BaseModel):
    image: str
    marker_type: str = "a4"
    custom_size_cm: Optional[float] = 29.7
    use_ai: bool = False
    depth_model: str = "Intel/midas-small"


class AnalyzeResponse(BaseModel):
    width_cm: float
    height_cm: float
    length_cm: float
    px_per_cm: float
    marker_label: str
    confidence: float
    demo_model_url: Optional[str] = None
    note: str


def decode_image(b64_string: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def estimate_with_fallback(image: np.ndarray, marker_type: str, custom_size_cm: float):
    """
    Опростена евристика за демонстрация, когато AI моделите не са налични.
    """
    h, w = image.shape[:2]

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
        "px_per_cm": px_per_cm,
        "marker_label": MARKER_SIZES_CM.get(marker_type, {}).get("label", "Ръчен маркер"),
        "confidence": 0.55,
        "note": "Демонстрационно измерване без AI модели.",
    }


def estimate_with_ai(image: np.ndarray, marker_type: str, custom_size_cm: float, depth_model: str):
    """
    Опитва се да използва depth estimation + SAM за по-точни размери.
    Ако някой модел липсва, пада към fallback евристика.
    """
    try:
        depth = estimate_depth(image, depth_model)
        segment = segment_floor_and_walls(image)
        edges, lines = detect_edges(image)
        vp = find_vanishing_points(lines, image.shape[:2])

        h, w = image.shape[:2]

        # Използваме depth map за приблизителна дистанция до задната стена
        center_depth = float(depth[h // 2, w // 2])
        # Евристика: 1 depth единица ≈ 80 cm (ще се калибрира с маркер)
        far_wall_cm = max(center_depth * 80.0, 150.0)

        # Ширина на видимото поле на дълбочина far_wall_cm
        fov_horizontal_deg = 70.0
        fov_rad = math.radians(fov_horizontal_deg)
        visible_width_cm = 2 * far_wall_cm * math.tan(fov_rad / 2)

        aspect = h / w
        visible_height_cm = visible_width_cm * aspect

        width_cm = round(visible_width_cm / 10) * 10
        length_cm = round((far_wall_cm) / 10) * 10
        height_cm = round((visible_height_cm * 0.55) / 10) * 10

        width_cm = max(width_cm, 200)
        length_cm = max(length_cm, 200)
        height_cm = max(height_cm, 230)

        return {
            "width_cm": width_cm,
            "height_cm": height_cm,
            "length_cm": length_cm,
            "px_per_cm": round(w / visible_width_cm, 2),
            "marker_label": MARKER_SIZES_CM.get(marker_type, {}).get("label", "Ръчен маркер"),
            "confidence": 0.72,
            "note": (
                "AI измерване с depth estimation и SAM сегментация. "
                "Маркерът все още не се използва за финална калибрация."
            ),
        }
    except Exception as exc:
        result = estimate_with_fallback(image, marker_type, custom_size_cm)
        result["note"] = f"AI моделите не са налични ({exc}). Използва се fallback."
        return result


def estimate_room_dimensions(
    image: np.ndarray,
    marker_type: str,
    custom_size_cm: float,
    use_ai: bool = False,
    depth_model: str = "Intel/midas-small",
):
    if use_ai:
        return estimate_with_ai(image, marker_type, custom_size_cm, depth_model)
    return estimate_with_fallback(image, marker_type, custom_size_cm)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    image = decode_image(req.image)
    result = estimate_room_dimensions(
        image,
        req.marker_type,
        req.custom_size_cm,
        use_ai=req.use_ai,
        depth_model=req.depth_model,
    )

    # TODO: заменете с реален URL към GLB модел от вашия каталог
    result["demo_model_url"] = "https://threejs.org/examples/models/gltf/Parrot.glb"

    return AnalyzeResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
