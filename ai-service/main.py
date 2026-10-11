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

from measurement import estimate_room_from_image, MARKER_SIZES_CM


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


def decode_image(b64_string: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def estimate_room_dimensions(
    image: np.ndarray,
    marker_type: str,
    custom_size_cm: float,
    use_ai: bool = False,
    depth_model: str = "Intel/midas-small",
):
    """
    Оценява размерите на стая.
    По подразбиране използва OpenCV + маркер.
    use_ai=True засега не е налично на този сървър и връща същия локален резултат с бележка.
    """
    result = estimate_room_from_image(image, marker_type, custom_size_cm)
    if use_ai:
        result["note"] += " AI моделите (depth/SAM) не са активирани на този сървър."
    return result


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
