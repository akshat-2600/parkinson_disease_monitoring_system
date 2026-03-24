"""
app/services/model_loader.py
"""
import os
import logging
import joblib

logger = logging.getLogger(__name__)


def _load_keras(path):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except ImportError:
        logger.warning("TensorFlow not installed — cannot load Keras model at %s", path)
        return None
    except Exception as exc:
        logger.error("Keras model load failed (%s): %s", path, exc)
        return None


def _load_torch(path):
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except ImportError:
        logger.warning("PyTorch not installed — cannot load model at %s", path)
        return None
    except Exception as exc:
        logger.error("PyTorch model load failed (%s): %s", path, exc)
        return None


class ModelRegistry:
    """
    Central store for all loaded ML models.
    Missing models are stored as None — callers use is_available() before predicting.
    """
    _models: dict = {}

    # Map: registry name → (config key, format)
    MODEL_MANIFEST = [
        ("voice",            "VOICE_MODEL_PATH",       "pkl"),
        ("voice_scaler",     "VOICE_SCALER_PATH",      "pkl"),
        ("clinical",         "CLINICAL_MODEL_PATH",    "pkl"),
        ("clinical_scaler",  "CLINICAL_SCALER_PATH",   "pkl"),
        ("mri",              "MRI_MODEL_PATH",         "keras"),
        ("spiral",           "SPIRAL_MODEL_PATH",      "keras"),
        ("motor",                 "MOTOR_MODEL_PATH",       "pkl"),
        ("motor_scaler",          "MOTOR_SCALER_PATH",      "pkl"),
        ("voice_selector",        "VOICE_SELECTOR_PATH",    "pkl"),
        ("voice_selected_features","VOICE_SELECTED_FEATURES_PATH", "pkl"),
        ("fusion",                "FUSION_MODEL_PATH",      "pkl"),
        ("timeseries",            "TIMESERIES_MODEL_PATH",  "pkl"),
    ]

    @classmethod
    def load_all(cls, app):
        cfg = app.config
        for name, cfg_key, fmt in cls.MODEL_MANIFEST:
            path = cfg.get(cfg_key, "")
            cls._load_one(name, path, fmt)

        loaded  = [k for k, v in cls._models.items() if v is not None]
        missing = [k for k, v in cls._models.items() if v is None]

        logger.info("Models loaded successfully: %s", loaded if loaded else "none")
        if missing:
            logger.warning(
                "Models not found (predictions will be skipped for these): %s",
                missing
            )

    @classmethod
    def _load_one(cls, name, path, fmt):
        """Load a single model file. Sets None if file missing or load fails."""
        if not path:
            logger.warning("No path configured for model '%s'", name)
            cls._models[name] = None
            return

        if not os.path.exists(path):
            logger.warning("Model file not found: %s  (model='%s')", path, name)
            cls._models[name] = None
            return

        try:
            if fmt == "pkl":
                cls._models[name] = joblib.load(path)
            elif fmt == "keras":
                cls._models[name] = _load_keras(path)
            elif fmt == "torch":
                cls._models[name] = _load_torch(path)
            else:
                logger.error("Unknown model format '%s' for model '%s'", fmt, name)
                cls._models[name] = None
                return

            if cls._models[name] is not None:
                logger.info("Loaded model '%s' from %s", name, path)
            else:
                logger.warning("Model '%s' loaded but returned None (check file integrity)", name)

        except Exception as exc:
            logger.error("Failed to load model '%s' from %s: %s", name, path, exc)
            cls._models[name] = None

    @classmethod
    def get(cls, name):
        """Return model object or None if not loaded."""
        return cls._models.get(name)

    @classmethod
    def is_available(cls, name):
        """True if model is loaded and ready."""
        return cls._models.get(name) is not None

    @classmethod
    def status(cls):
        """Return {model_name: True/False} for all registered models."""
        return {k: (v is not None) for k, v in cls._models.items()}

    @classmethod
    def available_models(cls):
        """Return list of model names that are loaded."""
        return [k for k, v in cls._models.items() if v is not None]


def predict_safe(fn, *args, **kwargs):
    """
    Wraps a model prediction call and returns None on any exception.
    Use this so one failing model never crashes the entire fusion pipeline.

    Example:
        result = predict_safe(predict_from_audio, audio_path, model, scaler)
        if result is not None:
            modality_results["voice"] = result
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning("predict_safe caught error in %s: %s", fn.__name__, exc)
        return None