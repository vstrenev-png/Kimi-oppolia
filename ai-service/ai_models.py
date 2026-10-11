"""
Опционални AI модели за по-точно измерване.
Тези функции се зареждат лениво при първа употреба, защото моделите са големи.
"""
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

_DEPTH_MODEL = None
_SAM_MODEL = None
_SAM_PREDICTOR = None


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_depth_model(model_name: str = "Intel/midas-small"):
    """Зарежда MiDaS/Depth Anything модел през Hugging Face transformers."""
    global _DEPTH_MODEL
    if _DEPTH_MODEL is not None:
        return _DEPTH_MODEL

    try:
        from transformers import pipeline
    except ImportError:
        raise RuntimeError("transformers не е инсталиран. Изпълни: pip install transformers")

    device = 0 if torch.cuda.is_available() else -1
    _DEPTH_MODEL = pipeline(
        task="depth-estimation",
        model=model_name,
        device=device,
    )
    return _DEPTH_MODEL


def estimate_depth(image: np.ndarray, model_name: str = "Intel/midas-small") -> np.ndarray:
    """
    Връща depth map като numpy масив със стойности в метри (приблизително).
    По-големи стойности = по-далечни обекти.
    """
    pipe = load_depth_model(model_name)
    pil_image = Image.fromarray(image)
    result = pipe(pil_image)
    depth = np.array(result["depth"])
    # Нормализираме до приблизителен мащаб (relative depth)
    depth_m = depth / depth.max() * 10.0
    return depth_m


def load_sam_model(model_type: str = "vit_b"):
    """Зарежда SAM (Segment Anything) модел."""
    global _SAM_MODEL, _SAM_PREDICTOR
    if _SAM_MODEL is not None:
        return _SAM_MODEL, _SAM_PREDICTOR

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise RuntimeError("segment-anything не е инсталиран. Провери requirements.txt")

    registry = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    checkpoint_name = registry.get(model_type, registry["vit_b"])
    checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint {checkpoint_path} не е намерен. "
            "Изтегли го от https://github.com/facebookresearch/segment-anything#model-checkpoints"
        )

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=get_device())
    _SAM_PREDICTOR = SamPredictor(sam)
    _SAM_MODEL = sam
    return _SAM_MODEL, _SAM_PREDICTOR


def segment_floor_and_walls(image: np.ndarray) -> dict:
    """
    Сегментира под, лява стена и задна стена от снимка на ъгъл на стая.
    Връща маски и bounding boxes.
    """
    _, predictor = load_sam_model("vit_b")
    predictor.set_image(image)

    h, w = image.shape[:2]
    # Точки за под, лява стена, задна стена — евристично в долната и страничните части
    point_coords = np.array([
        [w * 0.5, h * 0.75],   # под
        [w * 0.25, h * 0.45],  # лява стена
        [w * 0.65, h * 0.35],  # задна стена
    ])
    point_labels = np.array([1, 1, 1])

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_idx = int(scores.argmax())
    return {
        "mask": masks[best_idx],
        "scores": scores.tolist(),
        "image_shape": (h, w),
    }


def detect_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Открива ръбове с Canny и намира линии с Hough."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)
    return edges, lines if lines is not None else np.array([])


def find_vanishing_points(lines: np.ndarray, image_shape: Tuple[int, int]) -> dict:
    """
    Намира vanishing points от линии — полезно за корекция на перспектива.
    Връща координати или None, ако няма достатъчно линии.
    """
    if len(lines) < 4:
        return {}

    # Опростена клъстеризация по ъгъл
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    angles = np.array(angles)
    # TODO: реализирай клъстериране и пресичане на паралелни линии
    return {"count": len(lines), "angles_sample": angles[:5].tolist()}
