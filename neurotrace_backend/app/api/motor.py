"""
app/api/motor.py
Motor examination score prediction endpoint.

POST /motor/predict         — JSON or CSV → motor model prediction
GET  /motor/features/schema — return feature column list
"""
import time
from flask import Blueprint, request
from flask_jwt_extended import jwt_required

from app.services.model_loader  import ModelRegistry
from app.services.motor_service import (
    predict_motor, parse_motor_csv, MOTOR_FEATURES, MOTOR_DEFAULTS
)
from app.utils.file_handler     import save_upload, cleanup
from app.utils.response         import success, error, prediction_response
from app.middleware.auth        import require_patient_or_doctor
from app.models                 import Prediction, Patient
from app                        import db

motor_bp = Blueprint("motor", __name__)


@motor_bp.post("/predict")
@jwt_required()
@require_patient_or_doctor
def predict():
    """
    POST /motor/predict
    Accepts JSON body or multipart with 'motor_data' CSV file.
    """
    patient_id = None
    input_data = {}
    source     = "json"
    content_type = request.content_type or ""

    if "application/json" in content_type:
        body       = request.get_json(silent=True) or {}
        patient_id = body.pop("patient_id", None)
        input_data = body

    elif "motor_data" in request.files:
        patient_id = request.form.get("patient_id")
        try:
            csv_path   = save_upload(request.files["motor_data"], "data")
            input_data = parse_motor_csv(csv_path)
            source     = "csv"
        except ValueError as exc:
            return error(str(exc), 422)
        finally:
            cleanup(csv_path)
    else:
        return error("Provide JSON body or 'motor_data' CSV file", 400)

    model  = ModelRegistry.get("motor")
    scaler = ModelRegistry.get("motor_scaler")
    if model is None:
        return error("Motor model not loaded on server", 503)

    t0 = time.time()
    try:
        result = predict_motor(input_data, model, scaler)
        result["input_source"] = source
    except Exception as exc:
        return error(f"Motor prediction error: {exc}", 500)

    elapsed = int((time.time() - t0) * 1000)
    if patient_id:
        _persist(patient_id, "motor", result)

    return prediction_response("motor", result, patient_id=patient_id, processing_ms=elapsed)


@motor_bp.get("/features/schema")
def feature_schema():
    return success(data={
        "features":    MOTOR_FEATURES,
        "count":       len(MOTOR_FEATURES),
        "defaults":    MOTOR_DEFAULTS,
        "description": "Motor examination feature columns from UPDRS III dataset",
    })


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