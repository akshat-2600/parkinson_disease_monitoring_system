"""
app/api/mri.py — FIXED
Changes:
  - Generates Grad-CAM heatmap after prediction (before cleanup)
  - Stores base64 heatmap in raw_output so explanation endpoint can serve it
  - Auto-creates Report record with heatmap reference
"""
import time
import logging
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.services.model_loader       import ModelRegistry
from app.services.image_service      import predict_mri, _load_and_preprocess, MRI_IMAGE_SIZE
from app.services.explainability_service import gradcam_explain, heatmap_to_base64
from app.utils.file_handler          import save_upload, cleanup
from app.utils.response              import error, prediction_response
from app.middleware.auth             import require_patient_or_doctor
from app.models                      import Prediction, Patient, Report
from app                             import db

logger = logging.getLogger(__name__)
mri_bp = Blueprint("mri", __name__)


@mri_bp.post("/predict")
@jwt_required()
@require_patient_or_doctor
def predict():
    if "mri_scan" not in request.files:
        return error("'mri_scan' image file is required", 400)

    patient_id = request.form.get("patient_id")
    model      = ModelRegistry.get("mri")
    if model is None:
        return error("MRI model not loaded on server", 503)

    try:
        img_path = save_upload(request.files["mri_scan"], "image")
    except ValueError as exc:
        return error(str(exc), 400)

    t0 = time.time()
    try:
        result = predict_mri(img_path, model)

        # ── Generate Grad-CAM heatmap BEFORE cleanup ──────────
        heatmap_b64 = None
        try:
            import numpy as np
            # Check if model is Keras (has layers attribute)
            is_keras = hasattr(model, 'layers')
            if is_keras:
                img_arr = _load_and_preprocess(img_path, MRI_IMAGE_SIZE, grayscale=False)
                # Try common last conv layer names
                conv_names = ["conv2d", "conv2d_3", "conv2d_2", "conv2d_1",
                              "block5_conv3", "block4_conv3", "last_conv"]
                heatmap = None
                for layer_name in conv_names:
                    try:
                        heatmap = gradcam_explain(model, img_arr,
                                                  last_conv_layer_name=layer_name)
                        if heatmap is not None:
                            logger.info("Grad-CAM succeeded using layer: %s", layer_name)
                            break
                    except Exception:
                        continue

                if heatmap is not None:
                    heatmap_b64 = heatmap_to_base64(heatmap, MRI_IMAGE_SIZE)
                    if heatmap_b64:
                        logger.info("MRI heatmap generated (%d bytes b64)", len(heatmap_b64))
                    else:
                        logger.warning("heatmap_to_base64 returned None")
                else:
                    logger.warning("Grad-CAM: no valid conv layer found in MRI model")
            else:
                logger.info("MRI model is not Keras — skipping Grad-CAM")
        except Exception as hm_exc:
            logger.warning("Grad-CAM heatmap generation failed: %s", hm_exc)

        # Store heatmap in result so it persists in raw_output
        if heatmap_b64:
            result["heatmap_base64"] = heatmap_b64
            result["explainability"] = {
                "features": [
                    {"name": "MRI CNN Confidence", "importance": round(result.get("confidence", 0), 4)},
                    {"name": "Parkinson Probability", "importance": round(result.get("probability", 0), 4)},
                ],
                "summary": (
                    f"MRI CNN analysis: {result.get('label', '—')} "
                    f"(probability {round(result.get('probability', 0)*100, 1)}%, "
                    f"confidence {round(result.get('confidence', 0)*100, 1)}%). "
                    f"Grad-CAM heatmap shows regions driving the prediction."
                ),
                "mri_heatmap_base64": heatmap_b64,
            }

    except ValueError as exc:
        cleanup(img_path)
        return error(str(exc), 422)
    except Exception as exc:
        cleanup(img_path)
        return error(f"MRI prediction error: {exc}", 500)
    finally:
        cleanup(img_path)

    elapsed = int((time.time() - t0) * 1000)

    if patient_id:
        identity = get_jwt_identity()
        user_id  = int(identity) if identity else None
        _persist(patient_id, "mri", result, user_id=user_id)

    logger.info("MRI prediction for %s: %s (heatmap=%s)",
                patient_id, result.get("label"), "yes" if heatmap_b64 else "no")

    return prediction_response("mri", result,
                               patient_id=patient_id, processing_ms=elapsed)


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
            title      = "MRI Scan Prediction Report",
            content    = {
                "prediction_id":  pred.id,
                "modality":       modality,
                "model_used":     "MRI CNN",
                "result":         result.get("label"),
                "severity":       round(result.get("severity"), 1) if result.get("severity") is not None else None,
                "confidence":     round(conf * 100, 1) if conf is not None else None,
                "probability":    result.get("probability"),
                "has_parkinson":  result.get("has_parkinson"),
                "has_heatmap":    bool(result.get("heatmap_base64")),
                "notes":          "Automated report from MRI CNN analysis. Grad-CAM heatmap generated." if result.get("heatmap_base64") else "Automated report from MRI CNN analysis.",
            },
            created_by = user_id,
        )
        db.session.add(report)
        db.session.commit()
        logger.info("Persisted MRI prediction + report for %s", patient_uid)
    except Exception as exc:
        logger.error("MRI persist failed: %s", exc)
        db.session.rollback()