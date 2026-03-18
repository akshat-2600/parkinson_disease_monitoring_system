"""
app/services/model_loader.py
Singleton model registry — loads every ML model once at startup.
Supports: sklearn (.pkl), TensorFlow/Keras (.h5), PyTorch (.pt)
"""
import os
import logging
import joblib
from flask import current_app

logger = logging.getLogger(__name__)

# ── Lazy-import heavy frameworks ──────────────────────────────
def _load_keras_model(path):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except ImportError:
        logger.warning("TensorFlow not installed; cannot load Keras model at %s", path)
        return None

def _load_torch_model(path):
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except ImportError:
        logger.warning("PyTorch not installed; cannot load model at %s", path)
        return None


class ModelRegistry:
    """
    Central store for all loaded models.
    Call `ModelRegistry.load_all(app)` once at startup.
    Then access via `ModelRegistry.get("voice")` etc.
    """
    _models: dict = {}

    @classmethod
    def load_all(cls, app):
        cfg = app.config
        loaders = {
            "voice":       (cfg["VOICE_MODEL_PATH"],      "pkl"),
            "voice_scaler":(cfg["VOICE_SCALER_PATH"],     "pkl"),
            "clinical":    (cfg["CLINICAL_MODEL_PATH"],   "pkl"),
            "clinical_scaler":(cfg["CLINICAL_SCALER_PATH"],"pkl"),
            "mri":         (cfg["MRI_MODEL_PATH"],        "keras"),
            "spiral":      (cfg["SPIRAL_MODEL_PATH"],     "keras"),
            "motor":       (cfg["MOTOR_MODEL_PATH"],      "pkl"),
            "motor_scaler":(cfg["MOTOR_SCALER_PATH"],     "pkl"),
            "fusion":      (cfg["FUSION_MODEL_PATH"],     "pkl"),
            "timeseries":  (cfg["TIMESERIES_MODEL_PATH"], "pkl"),
        }

        for name, (path, kind) in loaders.items():
            cls._load_one(name, path, kind)

        loaded = [k for k, v in cls._models.items() if v is not None]
        missing = [k for k, v in cls._models.items() if v is None]
        logger.info("Models loaded: %s", loaded)
        if missing:
            logger.warning("Models NOT found (predictions for these will be skipped): %s", missing)

    @classmethod
    def _load_one(cls, name, path, kind):
        if not os.path.exists(path):
            logger.warning("Model file not found: %s", path)
            cls._models[name] = None
            return
        try:
            if kind == "pkl":
                cls._models[name] = joblib.load(path)
            elif kind == "keras":
                cls._models[name] = _load_keras_model(path)
            elif kind == "torch":
                cls._models[name] = _load_torch_model(path)
            logger.info("Loaded model '%s' from %s", name, path)
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", name, exc)
            cls._models[name] = None

    @classmethod
    def get(cls, name):
        return cls._models.get(name)

    @classmethod
    def is_available(cls, name):
        return cls._models.get(name) is not None

    @classmethod
    def status(cls):
        return {k: (v is not None) for k, v in cls._models.items()}