import base64
import io
import math
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from measurement import estimate_room_from_image, MARKER_SIZES_CM
from vision_api import estimate_with_openai_vision
from furniture_removal import (
    detect_furniture_boxes,
    inpaint_furniture,
    generative_remove_furniture,
    encode_image_jpeg,
)


app = FastAPI(title="Design-Code AI Measurement Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    debug: Optional[Dict[str, Any]] = Field(default=None, description="Debug информация за marker detection")


def decode_image(b64_string: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def sanitize_base64(b64_string: str) -> str:
    """Премахва евентуален data URI prefix от base64 низа."""
    if "," in b64_string:
        return b64_string.split(",", 1)[1]
    return b64_string


def estimate_room_dimensions(
    image: np.ndarray,
    base64_image: str,
    marker_type: str,
    custom_size_cm: float,
    use_ai: bool = False,
    depth_model: str = "Intel/midas-small",
):
    """
    Оценява размерите на стая.
    По подразбиране използва OpenCV + маркер.
    use_ai=True използва OpenAI GPT-4o Vision, ако е наличен OPENAI_API_KEY.
    """
    if use_ai:
        ai_result = estimate_with_openai_vision(base64_image, marker_type, custom_size_cm)
        if ai_result is not None:
            ai_result["debug"] = {"source": "openai_vision"}
            return ai_result
        # Ако няма API ключ, падаме към локален алгоритъм с бележка
        result = estimate_room_from_image(image, marker_type, custom_size_cm)
        result["note"] += " OpenAI Vision API не е конфигуриран — използва се локален алгоритъм."
        return result

    return estimate_room_from_image(image, marker_type, custom_size_cm)


@app.get("/health")
def health():
    return {"status": "ok"}


class CleanRoomRequest(BaseModel):
    image: str


class CleanRoomResponse(BaseModel):
    cleaned_image: Optional[str] = None
    boxes_found: int = 0
    note: str


@app.post("/remove-furniture", response_model=CleanRoomResponse)
def remove_furniture(req: CleanRoomRequest):
    base64_image = sanitize_base64(req.image)
    image = decode_image(base64_image)

    try:
        boxes = detect_furniture_boxes(base64_image)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI откриване на мебели неуспешно: {exc}") from exc

    if boxes is None:
        raise HTTPException(status_code=503, detail="OpenAI API ключът не е конфигуриран на сървъра.")

    if not boxes:
        return CleanRoomResponse(
            cleaned_image=encode_image_jpeg(image),
            boxes_found=0,
            note="Не са открити мебели за премахване.",
        )

    engine = "opencv"
    cleaned = None
    try:
        cleaned = generative_remove_furniture(image, boxes)
        if cleaned is not None:
            engine = "gpt-image-1"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[remove-furniture] gpt-image-1 fallback: {exc}", flush=True)
        cleaned = None

    if cleaned is None:
        cleaned = inpaint_furniture(image, boxes)

    labels = ", ".join(sorted({b["label"] for b in boxes}))
    return CleanRoomResponse(
        cleaned_image=encode_image_jpeg(cleaned),
        boxes_found=len(boxes),
        note=f"Премахнати обекти ({len(boxes)}): {labels} [engine: {engine}]",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    base64_image = sanitize_base64(req.image)
    image = decode_image(base64_image)
    result = estimate_room_dimensions(
        image,
        base64_image,
        req.marker_type,
        req.custom_size_cm,
        use_ai=req.use_ai,
        depth_model=req.depth_model,
    )

    # demo_model_url се връща само ако има избран модел от каталога
    # Засега не добавяме автоматичен demo модел
    if "demo_model_url" not in result:
        result["demo_model_url"] = None

    return AnalyzeResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
