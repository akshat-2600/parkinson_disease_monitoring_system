"""
app/api/fusion.py
Fusion prediction engine — the primary API consumed by the frontend.

Endpoints (matching frontend fetch() calls exactly):
  POST /fusion/realtime_predict          — multi-file upload → instant fusion
  GET  /fusion/dashboard/<patient_id>    — full dashboard data
  GET  /fusion/explanation/<patient_id>  — explainability data
  GET  /fusion/recommendations/<patient_id> — personalised recommendations
  GET  /fusion/history/<patient_id>      — longitudinal history
"""
import time
import logging
from datetime import datetime, timezone, timedelta
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.services.model_loader       import ModelRegistry
from app.services.voice_service      import predict_from_audio, VOICE_FEATURES
from app.services.clinical_service   import predict_clinical, parse_csv_upload, CLINICAL_FEATURES
from app.services.image_service      import predict_mri, predict_spiral
from app.services.motor_service      import predict_motor, parse_motor_csv
from app.services.fusion_service     import (
    fuse_predictions, generate_risk_flags, generate_recommendations, severity_to_stage
)
from app.services.explainability_service import shap_explain, compute_attention_weights
from app.utils.file_handler          import save_upload, cleanup
from app.utils.response              import success, error
from app.middleware.auth             import require_patient_or_doctor
from app.models                      import Patient, Prediction, User
from app                             import db

logger    = logging.getLogger(__name__)
fusion_bp = Blueprint("fusion", __name__)


# ─────────────────────────────────────────────────────────────
# POST /fusion/realtime_predict
# ─────────────────────────────────────────────────────────────
@fusion_bp.post("/realtime_predict")
@jwt_required()
@require_patient_or_doctor
def realtime_predict():
    """
    Real-time multi-modal fusion prediction.
    Accepts any combination of:
      voice         (audio file)
      mri           (image file)
      spiral        (image file)
      clinical      (CSV file  OR included as JSON form field)
      motor         (CSV file)
      timeseries    (CSV file)
      patient_id    (form field, optional)

    At least one input modality is required.
    """
    t0         = time.time()
    patient_id = request.form.get("patient_id")
    results    = {}
    errors     = {}
    paths      = []   # track temp files for cleanup

    # ── 1. Voice ─────────────────────────────────────────────
    if "voice" in request.files:
        try:
            path = save_upload(request.files["voice"], "audio")
            paths.append(path)
            model  = ModelRegistry.get("voice")
            scaler = ModelRegistry.get("voice_scaler")
            if model:
                results["voice"] = predict_from_audio(path, model, scaler)
            else:
                errors["voice"] = "Voice model not loaded"
        except Exception as exc:
            errors["voice"] = str(exc)

    # ── 2. MRI ───────────────────────────────────────────────
    if "mri" in request.files:
        try:
            path = save_upload(request.files["mri"], "image")
            paths.append(path)
            model = ModelRegistry.get("mri")
            if model:
                results["mri"] = predict_mri(path, model)
            else:
                errors["mri"] = "MRI model not loaded"
        except Exception as exc:
            errors["mri"] = str(exc)

    # ── 3. Spiral ─────────────────────────────────────────────
    if "spiral" in request.files:
        try:
            path = save_upload(request.files["spiral"], "image")
            paths.append(path)
            model = ModelRegistry.get("spiral")
            if model:
                results["spiral"] = predict_spiral(path, model)
            else:
                errors["spiral"] = "Spiral model not loaded"
        except Exception as exc:
            errors["spiral"] = str(exc)

    # ── 4. Clinical (CSV) ─────────────────────────────────────
    if "clinical" in request.files:
        try:
            path = save_upload(request.files["clinical"], "data")
            paths.append(path)
            data  = parse_csv_upload(path)
            model = ModelRegistry.get("clinical")
            scl   = ModelRegistry.get("clinical_scaler")
            if model:
                results["clinical"] = predict_clinical(data, model, scl)
            else:
                errors["clinical"] = "Clinical model not loaded"
        except Exception as exc:
            errors["clinical"] = str(exc)

    # ── 5. Motor (CSV) ────────────────────────────────────────
    if "motor" in request.files:
        try:
            path = save_upload(request.files["motor"], "data")
            paths.append(path)
            data  = parse_motor_csv(path)
            model = ModelRegistry.get("motor")
            scl   = ModelRegistry.get("motor_scaler")
            if model:
                results["motor"] = predict_motor(data, model, scl)
            else:
                errors["motor"] = "Motor model not loaded"
        except Exception as exc:
            errors["motor"] = str(exc)

    # ── 6. Time-series (CSV) ──────────────────────────────────
    if "timeseries" in request.files:
        try:
            import numpy as np, pandas as pd
            path = save_upload(request.files["timeseries"], "data")
            paths.append(path)
            df    = pd.read_csv(path)
            X     = df.values.astype(np.float32).reshape(1, *df.shape)
            model = ModelRegistry.get("timeseries")
            if model:
                if hasattr(model, "predict_proba"):
                    proba   = model.predict_proba(X.reshape(1, -1))[0]
                    prob_pd = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    raw     = model.predict(X, verbose=0)
                    prob_pd = float(raw[0][0]) if raw.shape[-1] == 1 else float(raw[0][1])
                conf = prob_pd if prob_pd >= 0.5 else 1 - prob_pd
                results["timeseries"] = {
                    "has_parkinson": prob_pd >= 0.5,
                    "probability":   round(prob_pd, 4),
                    "confidence":    round(conf, 4),
                    "severity":      round(prob_pd * 100, 2),
                    "label":         "Parkinson's Detected" if prob_pd >= 0.5 else "No Parkinson's Detected",
                }
            else:
                errors["timeseries"] = "Time-series model not loaded"
        except Exception as exc:
            errors["timeseries"] = str(exc)

    # Cleanup all temp files
    for p in paths:
        cleanup(p)

    if not results:
        return error("No valid modality inputs could be processed", 422,
                     details={"modality_errors": errors})

    # ── Fusion ────────────────────────────────────────────────
    fusion_model = ModelRegistry.get("fusion")
    fused        = fuse_predictions(results, fusion_model)
    risk_flags   = generate_risk_flags(fused, results)
    recs         = generate_recommendations(fused)

    elapsed = int((time.time() - t0) * 1000)

    # Persist fusion prediction
    if patient_id:
        _persist(patient_id, "fusion", fused)

    return success(data={
        "severity":             fused["severity"],
        "stage":                fused["stage"],
        "label":                fused["label"],
        "has_parkinson":        fused["has_parkinson"],
        "confidence":           fused["confidence"],
        "probability":          fused["probability"],
        "fusion_method":        fused["fusion_method"],
        "modalities_used":      fused["modalities_used"],
        "modality_contributions": fused["modality_contributions"],
        "individual_results":   fused["individual_results"],
        "risk_flag":            "HIGH" if fused["severity"] >= 70 else ("MODERATE" if fused["severity"] >= 40 else "LOW"),
        "risk_description":     risk_flags[0]["msg"] if risk_flags else "No critical risk flags",
        "recommendation":       recs[0]["title"] + " — " + recs[0]["reasoning"] if recs else "",
        "explanation":          _build_explanation_summary(fused, results),
        "alerts":               risk_flags,
        "modality_errors":      errors if errors else None,
        "processing_time_ms":   elapsed,
    })


# ─────────────────────────────────────────────────────────────
# GET /fusion/dashboard/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/dashboard/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def dashboard(patient_id):
    """
    Full dashboard data for a patient.
    Aggregates latest predictions + patient profile.
    """
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    # Latest predictions per modality
    latest = {}
    for mod in ("voice", "clinical", "mri", "spiral", "motor", "fusion", "timeseries"):
        pred = (Prediction.query
                .filter_by(patient_id=patient.id, modality=mod)
                .order_by(Prediction.created_at.desc())
                .first())
        if pred:
            latest[mod] = pred.to_dict()

    fusion_pred = latest.get("fusion")
    severity    = fusion_pred["severity"] if fusion_pred else 0
    stage       = severity_to_stage(severity)

    # Modality confidence map
    modalities = {
        mod: round(latest[mod]["confidence"], 2) if latest.get(mod) else None
        for mod in ("voice", "clinical", "mri", "spiral", "motor")
    }

    # Risk indicators
    risks   = _build_risk_indicators(severity, latest)
    alerts  = _build_alerts(severity, latest)

    return success(data={
        # Patient info
        "patient_uid":   patient.patient_uid,
        "name":          f"{patient.user.first_name} {patient.user.last_name}".strip() if patient.user else patient.patient_uid,
        "initials":      _initials(patient.user),
        "age":           patient.age,
        "gender":        "Female" if patient.gender in ("F", "Female", 0) else "Male",
        "diagnosis":     patient.diagnosis or "Parkinson's Disease",
        "onset":         str(patient.onset_year) if patient.onset_year else "Unknown",
        "status":        _status_from_severity(severity),
        # Scores
        "severity":         round(severity, 1),
        "stage":            stage,
        "severity_change":  _severity_change(patient),
        "updrs":            _latest_updrs(latest),
        "fusion_confidence": fusion_pred["confidence"] if fusion_pred else None,
        # Charts
        "modalities":    {k: v for k, v in modalities.items() if v is not None},
        # Alerts & risks
        "alerts":        alerts,
        "risks":         risks,
    })


# ─────────────────────────────────────────────────────────────
# GET /fusion/explanation/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/explanation/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def explanation(patient_id):
    """SHAP-based explainability for the latest clinical prediction."""
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    latest_clinical = (Prediction.query
                       .filter_by(patient_id=patient.id, modality="clinical")
                       .order_by(Prediction.created_at.desc())
                       .first())

    feature_importance = []
    summary            = "No clinical prediction data available for this patient yet."
    attention          = []

    if latest_clinical and latest_clinical.raw_output:
        model  = ModelRegistry.get("clinical")
        scaler = ModelRegistry.get("clinical_scaler")
        if model:
            try:
                import numpy as np
                raw     = latest_clinical.raw_output
                vals    = [raw.get("features", {}).get(f, 0) for f in CLINICAL_FEATURES] if "features" in raw else [0] * len(CLINICAL_FEATURES)
                X       = np.array([vals])
                if scaler:
                    X = scaler.transform(X)
                expl    = shap_explain(model, X, CLINICAL_FEATURES)
                feature_importance = expl.get("top_features", [])
                summary            = expl.get("summary", "")
            except Exception as exc:
                logger.warning("Explanation generation failed: %s", exc)

    # Attention from latest fusion
    latest_fusion = (Prediction.query
                     .filter_by(patient_id=patient.id, modality="fusion")
                     .order_by(Prediction.created_at.desc())
                     .first())
    if latest_fusion and latest_fusion.raw_output:
        contrib = latest_fusion.raw_output.get("modality_contributions", {})
        attention = list(compute_attention_weights(contrib).values())

    return success(data={
        "features":           feature_importance,
        "attention":          attention,
        "summary":            summary,
        "mri_heatmap_url":    None,    # populated when Grad-CAM is run
        "spiral_heatmap_url": None,
    })


# ─────────────────────────────────────────────────────────────
# GET /fusion/recommendations/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/recommendations/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def recommendations(patient_id):
    """Personalised recommendations based on latest fusion prediction."""
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    latest_fusion = (Prediction.query
                     .filter_by(patient_id=patient.id, modality="fusion")
                     .order_by(Prediction.created_at.desc())
                     .first())

    fused_data = latest_fusion.raw_output if latest_fusion else {"severity": 30}
    recs       = generate_recommendations(fused_data)

    return success(data=recs)


# ─────────────────────────────────────────────────────────────
# GET /fusion/history/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/history/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def history(patient_id):
    """Longitudinal prediction history for trend charts."""
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    # Fetch last 12 months of fusion predictions
    since = datetime.now(timezone.utc) - timedelta(days=365)
    preds = (Prediction.query
             .filter(
                 Prediction.patient_id == patient.id,
                 Prediction.created_at >= since,
             )
             .order_by(Prediction.created_at.asc())
             .all())

    # Group by month label
    labels, severity, voice, mri, motor, updrs = [], [], [], [], [], []
    by_month: dict = {}

    for p in preds:
        key = p.created_at.strftime("%b %y")
        if key not in by_month:
            by_month[key] = {"fusion": [], "voice": [], "mri": [], "motor": [], "updrs": []}
        if p.modality == "fusion":
            by_month[key]["fusion"].append(p.severity or 0)
        elif p.modality == "voice":
            by_month[key]["voice"].append(p.confidence or 0)
        elif p.modality == "mri":
            by_month[key]["mri"].append(p.confidence or 0)
        elif p.modality == "motor":
            by_month[key]["motor"].append(p.confidence or 0)
            raw = p.raw_output or {}
            updrs_val = raw.get("updrs") or raw.get("hoehn_yahr_est", 0)
            by_month[key]["updrs"].append(float(updrs_val))

    def _avg(lst): return round(sum(lst) / len(lst), 2) if lst else None

    for month, vals in by_month.items():
        labels.append(month)
        severity.append(_avg(vals["fusion"]))
        voice.append(_avg(vals["voice"]))
        mri.append(_avg(vals["mri"]))
        motor.append(_avg(vals["motor"]))
        updrs.append(_avg(vals["updrs"]))

    # Build intervention timeline from report notes
    interventions = [
        {"date": p.created_at.strftime("%b %y"), "event": f"Fusion prediction recorded (severity {p.severity:.0f})"}
        for p in preds if p.modality == "fusion"
    ][:10]

    return success(data={
        "labels":        labels,
        "severity":      severity,
        "updrs":         updrs,
        "voice":         voice,
        "mri":           mri,
        "motor":         motor,
        "interventions": interventions,
    })


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
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
    except Exception as exc:
        logger.error("Fusion persist failed: %s", exc)
        db.session.rollback()


def _initials(user):
    if not user:
        return "??"
    fn = (user.first_name or "")[0].upper() if user.first_name else ""
    ln = (user.last_name  or "")[0].upper() if user.last_name  else ""
    return fn + ln or "??"


def _status_from_severity(s):
    if s >= 70: return "critical"
    if s >= 40: return "warning"
    return "stable"


def _severity_change(patient):
    preds = (Prediction.query
             .filter_by(patient_id=patient.id, modality="fusion")
             .order_by(Prediction.created_at.desc())
             .limit(2).all())
    if len(preds) == 2:
        return round(preds[0].severity - preds[1].severity, 1)
    return 0.0


def _latest_updrs(latest):
    motor = latest.get("motor", {})
    if motor and motor.get("raw_output"):
        return motor["raw_output"].get("updrs") or motor["raw_output"].get("hoehn_yahr_est")
    clinical = latest.get("clinical", {})
    if clinical and clinical.get("raw_output"):
        return clinical["raw_output"].get("updrs")
    return None


def _build_risk_indicators(severity, latest):
    risks = []
    if severity >= 70:
        risks.append({"name": "Falls Risk",         "level": "high"})
        risks.append({"name": "Motor Complications", "level": "high"})
    elif severity >= 40:
        risks.append({"name": "Falls Risk",     "level": "medium"})
        risks.append({"name": "Bradykinesia",   "level": "medium"})
    else:
        risks.append({"name": "Falls Risk",    "level": "low"})

    if latest.get("voice", {}).get("probability", 0) >= 0.7:
        risks.append({"name": "Dysarthria",    "level": "medium"})
    if latest.get("mri", {}).get("probability", 0) >= 0.8:
        risks.append({"name": "Dopamine Loss", "level": "high"})
    return risks


def _build_alerts(severity, latest):
    alerts = []
    if severity >= 70:
        alerts.append({"type": "critical", "msg": "Severity ≥ 70 — urgent clinical review required", "time": "Just now"})
    if latest.get("voice", {}).get("probability", 0) >= 0.75:
        alerts.append({"type": "warning", "msg": "Voice biomarkers indicate dysarthria progression", "time": "Latest scan"})
    if latest.get("mri", {}).get("probability", 0) >= 0.80:
        alerts.append({"type": "critical", "msg": "MRI: high probability of dopaminergic loss", "time": "Latest scan"})
    if not alerts:
        alerts.append({"type": "info", "msg": "No critical flags detected at this time", "time": "Now"})
    return alerts


def _build_explanation_summary(fused, results):
    mods  = fused.get("modalities_used", [])
    sev   = fused.get("severity", 0)
    stage = fused.get("stage", "—")
    parts = [f"Fusion analysis across {len(mods)} modality(ies): {', '.join(mods)}."]
    parts.append(f"Combined severity index: {sev:.1f}/100 ({stage}).")
    top   = max(fused.get("modality_contributions", {}).items(), key=lambda x: x[1], default=(None, 0))
    if top[0]:
        parts.append(f"Highest-contributing modality: {top[0]} ({top[1]}% weight).")
    return " ".join(parts)