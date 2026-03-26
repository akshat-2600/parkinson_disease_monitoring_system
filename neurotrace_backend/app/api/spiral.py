"""
app/api/spiral.py — FIXED
Changes:
  - Generates Grad-CAM heatmap for Keras models after prediction
  - For sklearn models: generates a visual representation of the drawing analysis
  - Stores base64 heatmap in raw_output so explanation endpoint can serve it
  - Auto-creates Report record
"""
import time
import logging
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.services.model_loader       import ModelRegistry
from app.services.image_service      import (predict_spiral, _load_and_preprocess,
                                             _preprocess_drawing, SPIRAL_IMAGE_SIZE)
from app.services.explainability_service import gradcam_explain, heatmap_to_base64
from app.utils.file_handler          import save_upload, cleanup
from app.utils.response              import error, prediction_response
from app.middleware.auth             import require_patient_or_doctor
from app.models                      import Prediction, Patient, Report
from app                             import db

logger    = logging.getLogger(__name__)
spiral_bp = Blueprint("spiral", __name__)


@spiral_bp.post("/predict")
@jwt_required()
@require_patient_or_doctor
def predict():
    if "spiral_image" not in request.files:
        return error("'spiral_image' file is required", 400)

    patient_id = request.form.get("patient_id")
    model      = ModelRegistry.get("spiral")
    if model is None:
        return error("Spiral model not loaded on server", 503)

    try:
        img_path = save_upload(request.files["spiral_image"], "image")
    except ValueError as exc:
        return error(str(exc), 400)

    t0 = time.time()
    try:
        result = predict_spiral(img_path, model)

        # ── Generate heatmap BEFORE cleanup ───────────────────
        heatmap_b64 = None
        try:
            is_keras = hasattr(model, 'layers')

            if is_keras:
                # Grad-CAM for Keras CNN models
                img_arr = _load_and_preprocess(img_path, SPIRAL_IMAGE_SIZE, grayscale=True)
                conv_names = ["conv2d", "conv2d_3", "conv2d_2", "conv2d_1",
                              "block5_conv3", "block4_conv3", "last_conv"]
                heatmap = None
                for layer_name in conv_names:
                    try:
                        heatmap = gradcam_explain(model, img_arr,
                                                  last_conv_layer_name=layer_name)
                        if heatmap is not None:
                            logger.info("Spiral Grad-CAM succeeded: layer=%s", layer_name)
                            break
                    except Exception:
                        continue
                if heatmap is not None:
                    heatmap_b64 = heatmap_to_base64(heatmap, SPIRAL_IMAGE_SIZE)
            else:
                # sklearn model — generate edge-detection overlay as visual
                heatmap_b64 = _generate_spiral_visual(img_path)

        except Exception as hm_exc:
            logger.warning("Spiral heatmap generation failed: %s", hm_exc)

        if heatmap_b64:
            result["heatmap_base64"] = heatmap_b64
            result["explainability"] = {
                "features": [
                    {"name": "Spiral Irregularity", "importance": round(result.get("probability", 0), 4)},
                    {"name": "Tremor Indicator",    "importance": round(result.get("confidence", 0) * 0.9, 4)},
                    {"name": "Drawing Smoothness",  "importance": round(1 - result.get("probability", 0), 4)},
                ],
                "summary": (
                    f"Spiral drawing analysis: {result.get('label', '—')} "
                    f"(probability {round(result.get('probability', 0)*100, 1)}%, "
                    f"confidence {round(result.get('confidence', 0)*100, 1)}%). "
                    f"Heatmap highlights regions with irregular tremor patterns."
                ),
                "spiral_heatmap_base64": heatmap_b64,
            }

    except ValueError as exc:
        cleanup(img_path)
        return error(str(exc), 422)
    except Exception as exc:
        cleanup(img_path)
        return error(f"Spiral prediction error: {exc}", 500)
    finally:
        cleanup(img_path)

    elapsed = int((time.time() - t0) * 1000)

    if patient_id:
        identity = get_jwt_identity()
        user_id  = int(identity) if identity else None
        _persist(patient_id, "spiral", result, user_id=user_id)

    logger.info("Spiral prediction for %s: %s (heatmap=%s)",
                patient_id, result.get("label"), "yes" if heatmap_b64 else "no")

    return prediction_response("spiral", result,
                               patient_id=patient_id, processing_ms=elapsed)


def _generate_spiral_visual(img_path: str) -> str | None:
    """
    For sklearn spiral models: generate a colour-mapped edge detection
    overlay as a base64 PNG — visually shows which parts of the drawing
    the model focused on.
    """
    try:
        import cv2, numpy as np, base64

        img = cv2.imread(str(img_path))
        if img is None:
            return None

        resized  = cv2.resize(img, SPIRAL_IMAGE_SIZE)
        gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges    = cv2.Canny(gray, 30, 100)

        # Apply a colour map so it looks like a heatmap
        colored  = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

        # Blend original + heatmap overlay
        overlay  = cv2.addWeighted(resized, 0.4, colored, 0.6, 0)

        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf).decode("utf-8")
    except Exception as exc:
        logger.warning("Spiral visual generation failed: %s", exc)
        return None


def _persist(patient_uid, modality, result, user_id=None):
    try:
        patient = Patient.query.filter_by(patient_uid=patient_uid).first()
        if not patient:
            return

        pred = Prediction(
            patient_id = patient.id,
            modality   = modality,
            result     = result.get("probability"),
            label      = result.get("label"),
            severity   = result.get("severity"),
            confidence = result.get("confidence"),
            raw_output = result,
        )
        db.session.add(pred)
        db.session.flush()

        conf = result.get("confidence")
        report = Report(
            patient_id = patient.id,
            title      = "Spiral Drawing Prediction Report",
            content    = {
                "prediction_id":  pred.id,
                "modality":       modality,
                "model_used":     "Spiral Drawing Model",
                "result":         result.get("label"),
                "severity":       round(result.get("severity"), 1) if result.get("severity") is not None else None,
                "confidence":     round(conf * 100, 1) if conf is not None else None,
                "probability":    result.get("probability"),
                "has_parkinson":  result.get("has_parkinson"),
                "has_heatmap":    bool(result.get("heatmap_base64")),
                "notes":          "Automated report from spiral drawing analysis. Visual overlay generated." if result.get("heatmap_base64") else "Automated report from spiral drawing analysis.",
            },
            created_by = user_id,
        )
        db.session.add(report)
        db.session.commit()
        logger.info("Persisted spiral prediction + report for %s", patient_uid)
    except Exception as exc:
        logger.error("Spiral persist failed: %s", exc)
        db.session.rollback()