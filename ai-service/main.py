import base64
import io
import math
import os
from typing import Any, Dict, List, Optional

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
    composite_with_original,
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
        from PIL import ImageOps
        img_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(img_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
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


def _covered_by_protect(box: Dict[str, Any], protect: list, threshold: float = 0.5) -> bool:
    """True, ако кутията е покрита >threshold от защитена зона — т.е. това е
    прекласифициран защитен елемент (врата/прозорец), а не мебел."""
    bx1, by1 = box["x"], box["y"]
    bx2, by2 = box["x"] + box["w"], box["y"] + box["h"]
    b_area = box["w"] * box["h"]
    if b_area <= 0:
        return True
    for p in protect:
        ix1, iy1 = max(bx1, p["x"]), max(by1, p["y"])
        ix2, iy2 = min(bx2, p["x"] + p["w"]), min(by2, p["y"] + p["h"])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter / b_area > threshold:
            return True
    return False


class CleanRoomRequest(BaseModel):
    image: str


class CleanRoomResponse(BaseModel):
    cleaned_image: Optional[str] = None
    boxes_found: int = 0
    note: str
    boxes: Optional[List[Dict[str, Any]]] = None
    protect: Optional[List[Dict[str, Any]]] = None


@app.post("/remove-furniture", response_model=CleanRoomResponse)
def remove_furniture(req: CleanRoomRequest):
    base64_image = sanitize_base64(req.image)
    image = decode_image(base64_image)

    # Ограничаваме входа — телефонни снимки 12MP+ биха препълнили паметта
    h, w = image.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        base64_image = encode_image_jpeg(image)

    try:
        detection = detect_furniture_boxes(base64_image)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI откриване на мебели неуспешно: {exc}") from exc

    if detection is None:
        raise HTTPException(status_code=503, detail="OpenAI API ключът не е конфигуриран на сървъра.")

    boxes = detection["boxes"]
    protect = detection["protect"]

    if not boxes:
        return CleanRoomResponse(
            cleaned_image=encode_image_jpeg(image),
            boxes_found=0,
            note="Не са открити мебели за премахване.",
        )

    engine = "opencv"
    cleaned = None
    gen_error = ""
    try:
        generated = generative_remove_furniture(image, boxes, protect)
        if generated is not None:
            # AI променя само маскираните зони; всичко останало остава
            # пиксел-по-пиксел от оригиналната снимка (врати, прозорци, стени)
            cleaned = composite_with_original(image, generated, boxes, protect)
            engine = "gpt-image-1+composite"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        gen_error = str(exc)[:200]
        cleaned = None

    if cleaned is None:
        cleaned = inpaint_furniture(image, boxes, protect)

    # Допълнителни проверителни pass-ове (макс. 3 общо): дочистване на
    # пропуснати мебели. Защитните зони се НАТРУПВАТ — врата, маркирана в
    # първи pass, не може да бъде пипната в следващите.
    if "+" in engine:
        for pass_num in range(2, 4):
            try:
                detection_n = detect_furniture_boxes(encode_image_jpeg(cleaned))
                if not detection_n:
                    break
                # Филтрираме „мебели", които са прекласифицирани защитени
                # елементи (вратата сама на празна стена може да мине за шкаф)
                leftover = [
                    b for b in detection_n["boxes"]
                    if not _covered_by_protect(b, protect)
                ]
                protect = protect + detection_n["protect"]
                if not leftover:
                    break
                generated_n = generative_remove_furniture(cleaned, leftover, protect)
                if generated_n is None:
                    break
                cleaned = composite_with_original(cleaned, generated_n, leftover, protect)
                boxes = boxes + leftover
                engine += f", pass{pass_num}"
            except Exception:
                import traceback
                traceback.print_exc()
                break

    labels = ", ".join(sorted({b["label"] for b in boxes}))
    note = f"Премахнати обекти ({len(boxes)}): {labels} [engine: {engine}]"
    if gen_error:
        note += f" [gen err: {gen_error}]"
    return CleanRoomResponse(
        cleaned_image=encode_image_jpeg(cleaned),
        boxes_found=len(boxes),
        note=note,
        boxes=boxes,
        protect=protect,
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
