"""
app/api/mri.py
MRI scan prediction endpoint.

POST /mri/predict   — upload MRI image → CNN prediction
"""
import time
from flask import Blueprint, request
from flask_jwt_extended import jwt_required

from app.services.model_loader  import ModelRegistry
from app.services.image_service import predict_mri
from app.utils.file_handler     import save_upload, cleanup
from app.utils.response         import success, error, prediction_response
from app.middleware.auth        import require_patient_or_doctor
from app.models                 import Prediction, Patient
from app                        import db

mri_bp = Blueprint("mri", __name__)


@mri_bp.post("/predict")
@jwt_required()
@require_patient_or_doctor
def predict():
    """
    POST /mri/predict
    Form-data: mri_scan (image file), patient_id (optional)
    """
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
    except ValueError as exc:
        return error(str(exc), 422)
    except Exception as exc:
        return error(f"MRI prediction error: {exc}", 500)
    finally:
        cleanup(img_path)

    elapsed = int((time.time() - t0) * 1000)
    if patient_id:
        _persist(patient_id, "mri", result)
    
    from flask import current_app
    current_app.logger.info(f"MRI prediction for {patient_id}: {result}")

    return prediction_response("mri", result, patient_id=patient_id, processing_ms=elapsed)


def _persist(patient_uid, modality, result):
    try:
        patient = Patient.query.filter_by(patient_uid=patient_uid).first()
        if not patient:
            return
        db.session.add(Prediction(
            patient_id=patient.id, modality=modality,
            result=result.get("probability"), label=result.get("label"),
            severity=result.get("severity"), confidence=result.get("confidence"),
            raw_output=result,
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()