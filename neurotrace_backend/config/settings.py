"""
config/settings.py
Centralised, environment-aware configuration.
"""
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    # ── Core ────────────────────────────────────────────────
    SECRET_KEY            = os.getenv("SECRET_KEY", "dev-secret-change-me")
    FLASK_ENV             = os.getenv("FLASK_ENV", "development")

    # ── JWT ─────────────────────────────────────────────────
    JWT_SECRET_KEY                  = os.getenv("JWT_SECRET_KEY", "jwt-secret-change-me")
    JWT_ACCESS_TOKEN_EXPIRES        = timedelta(seconds=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 3600)))
    JWT_REFRESH_TOKEN_EXPIRES       = timedelta(seconds=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES", 2592000)))
    JWT_TOKEN_LOCATION              = ["headers", "query_string"]
    JWT_HEADER_NAME                 = "Authorization"
    JWT_HEADER_TYPE                 = "Bearer"

    # ── Database ────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI         = os.getenv("DATABASE_URL", "sqlite:///neurotrace.db")
    SQLALCHEMY_TRACK_MODIFICATIONS  = False

    # ── Uploads ─────────────────────────────────────────────
    UPLOAD_FOLDER       = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH  = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))  # 50 MB
    ALLOWED_AUDIO_EXT   = {"wav", "mp3", "mp4", "ogg", "flac", "m4a"}
    ALLOWED_IMAGE_EXT   = {"png", "jpg", "jpeg", "bmp", "tiff", "nii", "gz"}
    ALLOWED_DATA_EXT    = {"csv", "json", "xlsx"}

    # ── ML Model Paths ──────────────────────────────────────
    VOICE_MODEL_PATH             = os.getenv("VOICE_MODEL_PATH",             "ml_models/voice_model.pkl")
    VOICE_SCALER_PATH            = os.getenv("VOICE_SCALER_PATH",            "ml_models/voice_scaler.pkl")
    VOICE_SELECTOR_PATH          = os.getenv("VOICE_SELECTOR_PATH",          "ml_models/voice_selector.pkl")
    VOICE_SELECTED_FEATURES_PATH = os.getenv("VOICE_SELECTED_FEATURES_PATH", "ml_models/voice_feature_names.pkl")
    CLINICAL_MODEL_PATH          = os.getenv("CLINICAL_MODEL_PATH",          "ml_models/clinical_model.pkl")
    CLINICAL_SCALER_PATH  = os.getenv("CLINICAL_SCALER_PATH",  "ml_models/clinical_scaler.pkl")
    MRI_MODEL_PATH        = os.getenv("MRI_MODEL_PATH",        "ml_models/mri_model.h5")
    SPIRAL_MODEL_PATH     = os.getenv("SPIRAL_MODEL_PATH",     "ml_models/spiral_model.h5")
    MOTOR_MODEL_PATH      = os.getenv("MOTOR_MODEL_PATH",      "ml_models/motor_model.pkl")
    MOTOR_SCALER_PATH     = os.getenv("MOTOR_SCALER_PATH",     "ml_models/motor_scaler.pkl")
    FUSION_MODEL_PATH     = os.getenv("FUSION_MODEL_PATH",     "ml_models/fusion_model.pkl")
    TIMESERIES_MODEL_PATH = os.getenv("TIMESERIES_MODEL_PATH", "ml_models/timeseries_model.pkl")

    # ── CORS ────────────────────────────────────────────────
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False
    FLASK_ENV = "production"


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
}


def get_config():
    env = os.getenv("FLASK_ENV", "development")
    return CONFIG_MAP.get(env, DevelopmentConfig)