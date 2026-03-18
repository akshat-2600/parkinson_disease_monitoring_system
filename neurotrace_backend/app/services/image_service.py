"""
app/services/image_service.py
Shared image preprocessing + prediction for MRI and Spiral models.
Both models are expected to be CNN-based (Keras .h5).
"""
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model-specific image sizes ────────────────────────────────
MRI_IMAGE_SIZE    = (224, 224)   # e.g. VGG16 / ResNet input
SPIRAL_IMAGE_SIZE = (128, 128)   # spiral drawing model input

LABELS = ["No Parkinson's Detected", "Parkinson's Detected"]


def _load_and_preprocess(image_path: str, target_size: tuple, grayscale: bool = False) -> np.ndarray:
    """Load an image, resize, normalise to [0,1]."""
    try:
        from PIL import Image
        img = Image.open(image_path)
        if grayscale:
            img = img.convert("L")
            img = img.convert("RGB")   # keep 3-channel for CNN
        else:
            img = img.convert("RGB")
        img = img.resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)   # (1, H, W, C)
    except Exception as exc:
        raise ValueError(f"Image preprocessing failed for {image_path}: {exc}") from exc


def predict_mri(image_path: str, model) -> dict:
    """
    MRI scan → Parkinson's probability using a pre-trained Keras CNN.
    """
    X = _load_and_preprocess(image_path, MRI_IMAGE_SIZE, grayscale=False)
    raw = model.predict(X, verbose=0)

    # Support both binary sigmoid and softmax output
    if raw.shape[-1] == 1:
        prob_pd    = float(raw[0][0])
        prob_no_pd = 1.0 - prob_pd
    else:
        prob_no_pd = float(raw[0][0])
        prob_pd    = float(raw[0][1])

    has_pd     = prob_pd >= 0.5
    confidence = prob_pd if has_pd else prob_no_pd

    return {
        "has_parkinson": has_pd,
        "probability":   round(prob_pd, 4),
        "confidence":    round(confidence, 4),
        "severity":      round(prob_pd * 100, 2),
        "label":         LABELS[int(has_pd)],
        "model":         "mri",
    }


def predict_spiral(image_path: str, model) -> dict:
    """
    Spiral drawing image → Parkinson's probability using a pre-trained CNN.
    Spiral drawings are typically grayscale hand-drawn images.
    """
    X = _load_and_preprocess(image_path, SPIRAL_IMAGE_SIZE, grayscale=True)
    raw = model.predict(X, verbose=0)

    if raw.shape[-1] == 1:
        prob_pd    = float(raw[0][0])
        prob_no_pd = 1.0 - prob_pd
    else:
        prob_no_pd = float(raw[0][0])
        prob_pd    = float(raw[0][1])

    has_pd     = prob_pd >= 0.5
    confidence = prob_pd if has_pd else prob_no_pd

    return {
        "has_parkinson": has_pd,
        "probability":   round(prob_pd, 4),
        "confidence":    round(confidence, 4),
        "severity":      round(prob_pd * 100, 2),
        "label":         LABELS[int(has_pd)],
        "model":         "spiral",
    }