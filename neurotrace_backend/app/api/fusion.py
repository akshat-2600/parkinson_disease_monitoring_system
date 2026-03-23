"""
app/api/fusion.py  (UPDATED)
════════════════════════════
Flask API blueprint for all fusion endpoints.
Uses FusionPredictor for the trained meta-model.

All prediction endpoints now follow this pattern:
  1. Run each available base model on uploaded inputs
  2. Collect probability outputs
  3. Pass to FusionPredictor → meta-model prediction
  4. Return unified result

Full endpoint list:
  POST /api/fusion/realtime_predict       — multi-file upload → instant fusion
  GET  /api/fusion/dashboard/<patient_id> — dashboard data
  GET  /api/fusion/explanation/<patient_id>
  GET  /api/fusion/recommendations/<patient_id>
  GET  /api/fusion/history/<patient_id>
"""
import time
import logging
from datetime import datetime, timezone, timedelta
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from app.services.model_loader        import ModelRegistry
from app.services.voice_service       import predict_from_audio, VOICE_FEATURES
from app.services.clinical_service    import predict_clinical, parse_csv_upload, CLINICAL_FEATURES
from app.services.image_service       import predict_mri, predict_spiral
from app.services.motor_service       import predict_motor, parse_motor_csv
from app.services.fusion_service      import (
    fuse_predictions, generate_risk_flags, generate_recommendations,
    severity_to_stage
)
from app.services.explainability_service import shap_explain, compute_attention_weights
from app.utils.file_handler           import save_upload, cleanup
from app.utils.response               import success, error
from app.middleware.auth              import require_patient_or_doctor
from app.models                       import Patient, Prediction, User
from app                              import db

logger    = logging.getLogger(__name__)
fusion_bp = Blueprint("fusion", __name__)


# ── Lazy-load the trained fusion meta-model ──────────────────
_fusion_predictor = None

def _get_fusion_predictor():
    """
    Load FusionPredictor singleton on first call.
    Falls back gracefully if fusion_model.pkl does not exist.
    """
    global _fusion_predictor
    if _fusion_predictor is None:
        try:
            from fusion.fusion_trainer import FusionPredictor
            from flask import current_app
            path = current_app.config.get("FUSION_MODEL_PATH", "ml_models/fusion_model.pkl")
            _fusion_predictor = FusionPredictor.from_path(path)
        except Exception as exc:
            logger.warning("Could not load FusionPredictor: %s — using built-in weighted average", exc)
            _fusion_predictor = None
    return _fusion_predictor


def _run_fusion(modality_results: dict) -> dict:
    """
    Core fusion logic.
    Tries meta-model first; falls back to weighted ensemble from fusion_service.
    """
    predictor = _get_fusion_predictor()

    if predictor is not None:
        # Use the trained meta-model
        fused = predictor.predict(modality_results)
        fused["fusion_method"] = "meta_model"
        # Add fields expected by dashboard
        fused.setdefault("stage", severity_to_stage(fused.get("severity", 0)))
        fused.setdefault("probability",  fused.get("final_probability", 0.5))
        fused.setdefault("has_parkinson", fused.get("probability", 0.5) >= 0.5)
        fused.setdefault("individual_results", {k: _slim(v) for k, v in modality_results.items() if v})
        fused.setdefault("modality_contributions",
                         {k: round(1/len(modality_results)*100, 1) for k in modality_results})
    else:
        # Fallback: use built-in weighted ensemble from fusion_service
        fusion_model = ModelRegistry.get("fusion")
        fused = fuse_predictions(modality_results, fusion_model)
        fused["fusion_method"] = "weighted_ensemble_fallback"

    return fused


def _slim(result: dict) -> dict:
    return {
        "probability": result.get("probability"),
        "confidence":  result.get("confidence"),
        "severity":    result.get("severity"),
        "label":       result.get("label"),
    }


# ─────────────────────────────────────────────────────────────
# POST /api/fusion/realtime_predict
# ─────────────────────────────────────────────────────────────
@fusion_bp.post("/realtime_predict")
@jwt_required()
@require_patient_or_doctor
def realtime_predict():
    t0         = time.time()
    patient_id = request.form.get("patient_id")
    results    = {}
    errors     = {}
    paths      = []

    # ── Voice ─────────────────────────────────────────────────
    if "voice" in request.files:
        try:
            path = save_upload(request.files["voice"], "audio"); paths.append(path)
            model  = ModelRegistry.get("voice")
            scaler = ModelRegistry.get("voice_scaler")
            if model:
                results["voice"] = predict_from_audio(path, model, scaler)
            else:
                errors["voice"] = "Voice model not loaded"
        except Exception as exc:
            errors["voice"] = str(exc)

    # ── MRI ───────────────────────────────────────────────────
    if "mri" in request.files:
        try:
            path = save_upload(request.files["mri"], "image"); paths.append(path)
            model = ModelRegistry.get("mri")
            if model:
                results["mri"] = predict_mri(path, model)
            else:
                errors["mri"] = "MRI model not loaded"
        except Exception as exc:
            errors["mri"] = str(exc)

    # ── Spiral ────────────────────────────────────────────────
    if "spiral" in request.files:
        try:
            path = save_upload(request.files["spiral"], "image"); paths.append(path)
            model = ModelRegistry.get("spiral")
            if model:
                results["spiral"] = predict_spiral(path, model)
            else:
                errors["spiral"] = "Spiral model not loaded"
        except Exception as exc:
            errors["spiral"] = str(exc)

    # ── Clinical ──────────────────────────────────────────────
    if "clinical" in request.files:
        try:
            path = save_upload(request.files["clinical"], "data"); paths.append(path)
            data  = parse_csv_upload(path)
            model = ModelRegistry.get("clinical")
            scl   = ModelRegistry.get("clinical_scaler")
            if model:
                results["clinical"] = predict_clinical(data, model, scl)
            else:
                errors["clinical"] = "Clinical model not loaded"
        except Exception as exc:
            errors["clinical"] = str(exc)

    # ── Motor ─────────────────────────────────────────────────
    if "motor" in request.files:
        try:
            path = save_upload(request.files["motor"], "data"); paths.append(path)
            data  = parse_motor_csv(path)
            model = ModelRegistry.get("motor")
            scl   = ModelRegistry.get("motor_scaler")
            if model:
                results["motor"] = predict_motor(data, model, scl)
            else:
                errors["motor"] = "Motor model not loaded"
        except Exception as exc:
            errors["motor"] = str(exc)

    # ── Time-series ────────────────────────────────────────────
    if "timeseries" in request.files:
        try:
            import numpy as np, pandas as pd
            path = save_upload(request.files["timeseries"], "data"); paths.append(path)
            df    = pd.read_csv(path)
            X     = df.values.astype(np.float32).reshape(1, *df.shape)
            model = ModelRegistry.get("timeseries")
            if model:
                proba   = model.predict_proba(X.reshape(1,-1))[0] if hasattr(model,"predict_proba") else None
                raw     = model.predict(X, verbose=0) if proba is None else None
                prob_pd = float(proba[1]) if proba is not None else float(raw[0][0])
                conf    = prob_pd if prob_pd >= 0.5 else 1 - prob_pd
                results["timeseries"] = {
                    "has_parkinson": prob_pd >= 0.5, "probability": round(prob_pd,4),
                    "confidence": round(conf,4), "severity": round(prob_pd*100,2),
                    "label": "Parkinson's Detected" if prob_pd >= 0.5 else "No Parkinson's Detected",
                }
            else:
                errors["timeseries"] = "Time-series model not loaded"
        except Exception as exc:
            errors["timeseries"] = str(exc)

    for p in paths:
        cleanup(p)

    if not results:
        return error("No valid modality inputs could be processed", 422,
                     details={"modality_errors": errors})

    # ── Run fusion (meta-model or weighted ensemble) ──────────
    fused      = _run_fusion(results)
    risk_flags = generate_risk_flags(fused, results)
    recs       = generate_recommendations(fused)

    elapsed = int((time.time() - t0) * 1000)

    if patient_id:
        _persist(patient_id, "fusion", fused)

    return success(data={
        # Core result
        "severity":          fused.get("severity"),
        "stage":             fused.get("stage"),
        "label":             fused.get("label"),
        "has_parkinson":     fused.get("has_parkinson"),
        "confidence":        fused.get("confidence"),
        "probability":       fused.get("final_probability") or fused.get("probability"),
        # Fusion details
        "fusion_method":     fused.get("fusion_method"),
        "simple_average":    fused.get("simple_average"),
        "weighted_average":  fused.get("weighted_average"),
        "meta_model_prob":   fused.get("meta_model_prob"),
        "modalities_used":   fused.get("modalities_used") or list(results.keys()),
        "modality_contributions": fused.get("modality_contributions"),
        "individual_results": fused.get("individual_results"),
        # Clinical info
        "risk_flag":         "HIGH" if (fused.get("severity",0) or 0) >= 70 else ("MODERATE" if (fused.get("severity",0) or 0) >= 40 else "LOW"),
        "risk_description":  risk_flags[0]["msg"] if risk_flags else "No critical risk flags",
        "recommendation":    recs[0]["title"] + " — " + recs[0]["reasoning"] if recs else "",
        "explanation":       _build_explanation(fused, results),
        "alerts":            risk_flags,
        "modality_errors":   errors or None,
        "processing_time_ms": elapsed,
    })


# ─────────────────────────────────────────────────────────────
# GET /api/fusion/dashboard/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/dashboard/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def dashboard(patient_id):
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    latest = {}
    for mod in ("voice","clinical","mri","spiral","motor","fusion","timeseries"):
        pred = (Prediction.query.filter_by(patient_id=patient.id, modality=mod)
                .order_by(Prediction.created_at.desc()).first())
        if pred:
            latest[mod] = pred.to_dict()

    fusion_pred = latest.get("fusion")
    severity    = fusion_pred["severity"] if fusion_pred else 0
    stage       = severity_to_stage(severity or 0)

    modalities = {
        mod: round(latest[mod]["confidence"], 2) if latest.get(mod) else None
        for mod in ("voice","clinical","mri","spiral","motor")
    }

    return success(data={
        "patient_uid":        patient.patient_uid,
        "name":               f"{patient.user.first_name} {patient.user.last_name}".strip() if patient.user else patient.patient_uid,
        "initials":           _initials(patient.user),
        "age":                patient.age,
        "gender":             "Female" if patient.gender in ("F","Female",0) else "Male",
        "diagnosis":          patient.diagnosis or "Parkinson's Disease",
        "onset":              str(patient.onset_year) if patient.onset_year else "Unknown",
        "status":             _status(severity or 0),
        "severity":           round(severity or 0, 1),
        "stage":              stage,
        "severity_change":    _sev_change(patient),
        "updrs":              _updrs(latest),
        "fusion_confidence":  fusion_pred["confidence"] if fusion_pred else None,
        "modalities":         {k: v for k, v in modalities.items() if v is not None},
        "alerts":             _build_alerts(severity or 0, latest),
        "risks":              _build_risks(severity or 0, latest),
    })


# ─────────────────────────────────────────────────────────────
# GET /api/fusion/explanation/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/explanation/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def explanation(patient_id):
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    features   = []
    summary    = "No prediction data yet."
    attention  = []

    latest_clinical = (Prediction.query.filter_by(patient_id=patient.id, modality="clinical")
                       .order_by(Prediction.created_at.desc()).first())
    if latest_clinical and latest_clinical.raw_output:
        model  = ModelRegistry.get("clinical")
        scaler = ModelRegistry.get("clinical_scaler")
        if model:
            try:
                import numpy as np
                raw = latest_clinical.raw_output
                vals = [raw.get("features", {}).get(f, 0) for f in CLINICAL_FEATURES]
                X    = np.array([vals])
                if scaler: X = scaler.transform(X)
                expl     = shap_explain(model, X, CLINICAL_FEATURES)
                features = expl.get("top_features", [])
                summary  = expl.get("summary", "")
            except Exception as exc:
                logger.warning("SHAP failed: %s", exc)

    latest_fusion = (Prediction.query.filter_by(patient_id=patient.id, modality="fusion")
                     .order_by(Prediction.created_at.desc()).first())
    if latest_fusion and latest_fusion.raw_output:
        contrib   = latest_fusion.raw_output.get("modality_contributions", {})
        attention = list(compute_attention_weights(contrib).values())

    return success(data={
        "features":           features,
        "attention":          attention,
        "summary":            summary,
        "mri_heatmap_url":    None,
        "spiral_heatmap_url": None,
    })


# ─────────────────────────────────────────────────────────────
# GET /api/fusion/recommendations/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/recommendations/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def recommendations(patient_id):
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    latest_fusion = (Prediction.query.filter_by(patient_id=patient.id, modality="fusion")
                     .order_by(Prediction.created_at.desc()).first())

    fused_data = latest_fusion.raw_output if latest_fusion else {"severity": 30}
    recs       = generate_recommendations(fused_data)
    return success(data=recs)


# ─────────────────────────────────────────────────────────────
# GET /api/fusion/history/<patient_id>
# ─────────────────────────────────────────────────────────────
@fusion_bp.get("/history/<patient_id>")
@jwt_required()
@require_patient_or_doctor
def history(patient_id):
    patient = Patient.query.filter_by(patient_uid=patient_id).first()
    if not patient:
        return error(f"Patient '{patient_id}' not found", 404)

    since = datetime.now(timezone.utc) - timedelta(days=365)
    preds = (Prediction.query
             .filter(Prediction.patient_id == patient.id, Prediction.created_at >= since)
             .order_by(Prediction.created_at.asc()).all())

    by_month = {}
    for p in preds:
        key = p.created_at.strftime("%b %y")
        if key not in by_month:
            by_month[key] = {"fusion":[], "voice":[], "mri":[], "motor":[], "updrs":[]}
        if p.modality == "fusion":     by_month[key]["fusion"].append(p.severity or 0)
        elif p.modality == "voice":    by_month[key]["voice"].append(p.confidence or 0)
        elif p.modality == "mri":      by_month[key]["mri"].append(p.confidence or 0)
        elif p.modality == "motor":
            by_month[key]["motor"].append(p.confidence or 0)
            by_month[key]["updrs"].append(float((p.raw_output or {}).get("updrs") or 0))

    def _avg(lst): return round(sum(lst)/len(lst), 2) if lst else None

    labels, severity, voice, mri, motor, updrs = [], [], [], [], [], []
    for month, vals in by_month.items():
        labels.append(month)
        severity.append(_avg(vals["fusion"]))
        voice.append(_avg(vals["voice"]))
        mri.append(_avg(vals["mri"]))
        motor.append(_avg(vals["motor"]))
        updrs.append(_avg(vals["updrs"]))

    interventions = [
        {"date": p.created_at.strftime("%b %y"), "event": f"Fusion prediction — severity {p.severity:.0f}"}
        for p in preds if p.modality == "fusion"
    ][:10]

    return success(data={"labels":labels,"severity":severity,"updrs":updrs,
                         "voice":voice,"mri":mri,"motor":motor,"interventions":interventions})


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
def _persist(patient_uid, modality, result):
    try:
        patient = Patient.query.filter_by(patient_uid=patient_uid).first()
        if not patient: return
        prob = result.get("final_probability") or result.get("probability")
        db.session.add(Prediction(
            patient_id=patient.id, modality=modality,
            result=prob, label=result.get("label"),
            severity=result.get("severity"), confidence=result.get("confidence"),
            raw_output=result,
        ))
        db.session.commit()
    except Exception as exc:
        logger.error("Persist failed: %s", exc); db.session.rollback()


def _initials(user):
    if not user: return "??"
    return ((user.first_name or "")[0] + (user.last_name or "")[0]).upper() or "??"


def _status(s):
    if s >= 70: return "critical"
    if s >= 40: return "warning"
    return "stable"


def _sev_change(patient):
    preds = (Prediction.query.filter_by(patient_id=patient.id, modality="fusion")
             .order_by(Prediction.created_at.desc()).limit(2).all())
    return round(preds[0].severity - preds[1].severity, 1) if len(preds) == 2 else 0.0


def _updrs(latest):
    for mod in ("motor","clinical"):
        raw = (latest.get(mod) or {}).get("raw_output") or {}
        val = raw.get("updrs") or raw.get("hoehn_yahr_est")
        if val: return val
    return None


def _build_risks(severity, latest):
    risks = []
    if severity >= 70:
        risks += [{"name":"Falls Risk","level":"high"},{"name":"Motor Complications","level":"high"}]
    elif severity >= 40:
        risks += [{"name":"Falls Risk","level":"medium"},{"name":"Bradykinesia","level":"medium"}]
    else:
        risks.append({"name":"Falls Risk","level":"low"})
    if (latest.get("voice") or {}).get("probability",0) >= 0.7:
        risks.append({"name":"Dysarthria","level":"medium"})
    if (latest.get("mri") or {}).get("probability",0) >= 0.8:
        risks.append({"name":"Dopamine Loss","level":"high"})
    return risks


def _build_alerts(severity, latest):
    alerts = []
    if severity >= 70:
        alerts.append({"type":"critical","msg":"Severity ≥ 70 — urgent clinical review required","time":"Just now"})
    if (latest.get("voice") or {}).get("probability",0) >= 0.75:
        alerts.append({"type":"warning","msg":"Voice biomarkers indicate dysarthria progression","time":"Latest scan"})
    if (latest.get("mri") or {}).get("probability",0) >= 0.80:
        alerts.append({"type":"critical","msg":"MRI: high probability of dopaminergic loss","time":"Latest scan"})
    if not alerts:
        alerts.append({"type":"info","msg":"No critical flags detected at this time","time":"Now"})
    return alerts


def _build_explanation(fused, results):
    mods  = fused.get("modalities_used") or list(results.keys())
    sev   = fused.get("severity", 0)
    stage = fused.get("stage", "—")
    method = fused.get("fusion_method","ensemble")
    parts = [
        f"Fusion analysis ({method}) across {len(mods)} modality(ies): {', '.join(mods)}.",
        f"Combined severity index: {sev:.1f}/100 ({stage}).",
    ]
    contribs = fused.get("modality_contributions") or {}
    if contribs:
        top = max(contribs.items(), key=lambda x: x[1], default=(None, 0))
        if top[0]: parts.append(f"Highest-contributing modality: {top[0]} ({top[1]}% weight).")
    if method == "meta_model":
        parts.append("Prediction made by trained meta-model (Logistic Regression/XGBoost).")
    return " ".join(parts)