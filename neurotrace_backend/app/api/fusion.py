"""
app/api/fusion.py 
"""
import time
import logging
from datetime import datetime, timezone, timedelta
from flask import Blueprint, request
from flask_jwt_extended import jwt_required

from app.services.model_loader         import ModelRegistry, predict_safe
from app.services.voice_service        import predict_from_audio, VOICE_FEATURES
from app.services.clinical_service     import predict_clinical, parse_csv_upload, CLINICAL_FEATURES
from app.services.image_service        import predict_mri, predict_spiral
from app.services.motor_service        import predict_motor, parse_motor_csv
from app.services.fusion_service       import (
    fuse_predictions, generate_risk_flags,
    generate_recommendations, severity_to_stage
)
from app.services.explainability_service import shap_explain, compute_attention_weights
from app.utils.file_handler            import save_upload, cleanup
from app.utils.response                import success, error
from app.middleware.auth               import require_patient_or_doctor
from app.models                        import Patient, Prediction
from app                               import db

logger    = logging.getLogger(__name__)
fusion_bp = Blueprint("fusion", __name__)


# ─────────────────────────────────────────────────────────────
# POST /api/fusion/realtime_predict
# ─────────────────────────────────────────────────────────────
@fusion_bp.post("/realtime_predict")
@jwt_required()
@require_patient_or_doctor
def realtime_predict():
    t0         = time.time()
    patient_id = request.form.get("patient_id")
    results    = {}          # modality → prediction dict
    skipped    = {}          # modality → reason skipped
    paths      = []

    # ── Voice ─────────────────────────────────────────────────
    if "voice" in request.files:
        model             = ModelRegistry.get("voice")
        scaler            = ModelRegistry.get("voice_scaler")
        selector          = ModelRegistry.get("voice_selector")
        selected_features = ModelRegistry.get("voice_selected_features")
        if model is None:
            skipped["voice"] = "Model file not loaded"
        else:
            try:
                path = save_upload(request.files["voice"], "audio")
                paths.append(path)
                r = predict_safe(predict_from_audio, path, model, scaler, selector, selected_features)
                if r: results["voice"] = r
                else: skipped["voice"] = "Feature extraction failed"
            except Exception as exc:
                skipped["voice"] = str(exc)

    # ── MRI ───────────────────────────────────────────────────
    if "mri" in request.files:
        model = ModelRegistry.get("mri")
        if model is None:
            skipped["mri"] = "Model file not loaded"
        else:
            try:
                path = save_upload(request.files["mri"], "image")
                paths.append(path)
                r = predict_safe(predict_mri, path, model)
                if r: results["mri"] = r
                else: skipped["mri"] = "Image processing failed"
            except Exception as exc:
                skipped["mri"] = str(exc)

    # ── Spiral ────────────────────────────────────────────────
    if "spiral" in request.files:
        model = ModelRegistry.get("spiral")
        if model is None:
            skipped["spiral"] = "Model file not loaded"
        else:
            try:
                path = save_upload(request.files["spiral"], "image")
                paths.append(path)
                r = predict_safe(predict_spiral, path, model)
                if r: results["spiral"] = r
                else: skipped["spiral"] = "Image processing failed"
            except Exception as exc:
                skipped["spiral"] = str(exc)

    # ── Clinical ──────────────────────────────────────────────
    if "clinical" in request.files:
        model  = ModelRegistry.get("clinical")
        scaler = ModelRegistry.get("clinical_scaler")
        if model is None:
            skipped["clinical"] = "Model file not loaded"
        else:
            try:
                path = save_upload(request.files["clinical"], "data")
                paths.append(path)
                data = parse_csv_upload(path)
                r    = predict_safe(predict_clinical, data, model, scaler)
                if r: results["clinical"] = r
                else: skipped["clinical"] = "Clinical data processing failed"
            except Exception as exc:
                skipped["clinical"] = str(exc)

    # ── Motor ─────────────────────────────────────────────────
    if "motor" in request.files:
        model  = ModelRegistry.get("motor")
        scaler = ModelRegistry.get("motor_scaler")
        if model is None:
            skipped["motor"] = "Model file not loaded"
        else:
            try:
                path = save_upload(request.files["motor"], "data")
                paths.append(path)
                data = parse_motor_csv(path)
                r    = predict_safe(predict_motor, data, model, scaler)
                if r: results["motor"] = r
                else: skipped["motor"] = "Motor data processing failed"
            except Exception as exc:
                skipped["motor"] = str(exc)

    # ── Time-series ───────────────────────────────────────────
    if "timeseries" in request.files:
        model = ModelRegistry.get("timeseries")
        if model is None:
            skipped["timeseries"] = "Model file not loaded"
        else:
            try:
                import numpy as np, pandas as pd
                path = save_upload(request.files["timeseries"], "data")
                paths.append(path)
                df   = pd.read_csv(path)
                X    = df.values.astype(np.float32)
                X_r  = X.reshape(1, -1) if hasattr(model, "predict_proba") else X.reshape(1, *X.shape)
                if hasattr(model, "predict_proba"):
                    proba   = model.predict_proba(X_r)[0]
                    prob_pd = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    raw     = model.predict(X_r, verbose=0)
                    prob_pd = float(raw[0][0]) if raw.shape[-1] == 1 else float(raw[0][1])
                conf = prob_pd if prob_pd >= 0.5 else 1 - prob_pd
                results["timeseries"] = {
                    "has_parkinson": prob_pd >= 0.5,
                    "probability":   round(prob_pd, 4),
                    "confidence":    round(conf, 4),
                    "severity":      round(prob_pd * 100, 2),
                    "label": "Parkinson's Detected" if prob_pd >= 0.5 else "No Parkinson's Detected",
                }
            except Exception as exc:
                skipped["timeseries"] = str(exc)

    # Cleanup temp files
    for p in paths:
        cleanup(p)

    # ── Require at least one successful modality ──────────────
    if not results:
        reasons = "; ".join(f"{k}: {v}" for k, v in skipped.items()) if skipped else "No files uploaded"
        return error(
            "No models could run a prediction. Check that model files exist in ml_models/.",
            422,
            details={"models_skipped": skipped, "reason": reasons}
        )

    # ── Fuse results ──────────────────────────────────────────
    fusion_model = ModelRegistry.get("fusion")
    fused        = fuse_predictions(results, fusion_model)
    risk_flags   = generate_risk_flags(fused, results)
    recs         = generate_recommendations(fused)

    elapsed = int((time.time() - t0) * 1000)

    if patient_id:
        _persist(patient_id, "fusion", fused)

    return success(data={
        "severity":               fused.get("severity"),
        "stage":                  fused.get("stage"),
        "label":                  fused.get("label"),
        "has_parkinson":          fused.get("has_parkinson"),
        "confidence":             fused.get("confidence"),
        "probability":            fused.get("probability"),
        "fusion_method":          fused.get("fusion_method"),
        "modalities_used":        fused.get("modalities_used") or list(results.keys()),
        "models_skipped":         skipped,
        "modality_contributions": fused.get("modality_contributions"),
        "individual_results":     {k: _slim(v) for k, v in results.items()},
        "risk_flag":              "HIGH" if (fused.get("severity") or 0) >= 70 else ("MODERATE" if (fused.get("severity") or 0) >= 40 else "LOW"),
        "risk_description":       risk_flags[0]["msg"] if risk_flags else "No critical risk flags",
        "recommendation":         f"{recs[0]['title']} — {recs[0]['reasoning']}" if recs else "",
        "explanation":            _build_explanation(fused, results, skipped),
        "alerts":                 risk_flags,
        "processing_time_ms":     elapsed,
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

    # Latest prediction per modality
    latest = {}
    for mod in ("voice","clinical","mri","spiral","motor","fusion","timeseries"):
        pred = (Prediction.query
                .filter_by(patient_id=patient.id, modality=mod)
                .order_by(Prediction.created_at.desc()).first())
        if pred:
            latest[mod] = pred

    fusion_pred = latest.get("fusion")

    # Return basic patient info even when no predictions exist
    base = {
        "patient_uid":  patient.patient_uid,
        "name":         f"{patient.user.first_name} {patient.user.last_name}".strip() if patient.user else patient.patient_uid,
        "initials":     _initials(patient.user),
        "age":          patient.age,
        "gender":       "Female" if patient.gender in ("F","Female") else ("Male" if patient.gender in ("M","Male") else patient.gender),
        "diagnosis":    patient.diagnosis or "Parkinson's Disease",
        "onset":        str(patient.onset_year) if patient.onset_year else None,
    }

    # Build modalities response with confidence for all available predictions
    modalities = {}
    for mod in ("voice","clinical","mri","spiral","motor"):
        p = latest.get(mod)
        if p and p.confidence is not None:
            modalities[mod] = {
                "confidence": round(p.confidence, 2),
                "label": p.label,
                "severity": round(p.severity, 1) if p.severity else None,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }

    if not fusion_pred and not modalities:
        # No predictions yet — return patient info with empty prediction fields
        return success(data={
            **base,
            "severity":          None,
            "stage":             None,
            "status":            "stable",
            "severity_change":   None,
            "updrs":             None,
            "fusion_confidence": None,
            "modalities":        modalities,
            "alerts":            [],
            "risks":             [],
            "has_predictions":   False,
        })

    severity = fusion_pred.severity if fusion_pred else (sum(p.get("severity", 0) for p in modalities.values()) / len(modalities) if modalities else 0)
    stage    = severity_to_stage(severity)

    return success(data={
        **base,
        "severity":          round(severity, 1),
        "stage":             stage,
        "status":            _status(severity),
        "severity_change":   _sev_change(patient),
        "updrs":             _updrs(latest),
        "fusion_confidence": fusion_pred.confidence if fusion_pred else None,
        "modalities":        modalities,
        "alerts":            _build_alerts(severity, latest),
        "risks":             _build_risks(severity, latest),
        "has_predictions":   bool(modalities or fusion_pred),
        "last_prediction_at": (fusion_pred.created_at if fusion_pred else max((p["created_at"] for p in modalities.values() if p.get("created_at")), default=None)).isoformat() if (fusion_pred or modalities) else None,
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

    latest_clinical = (Prediction.query
                       .filter_by(patient_id=patient.id, modality="clinical")
                       .order_by(Prediction.created_at.desc()).first())

    features  = []
    summary   = None
    attention = []

    if latest_clinical and latest_clinical.raw_output:
        model  = ModelRegistry.get("clinical")
        scaler = ModelRegistry.get("clinical_scaler")
        if model:
            try:
                import numpy as np
                raw  = latest_clinical.raw_output
                vals = [raw.get("features", {}).get(f, 0) for f in CLINICAL_FEATURES]
                X    = np.array([vals])
                if scaler: X = scaler.transform(X)
                expl     = shap_explain(model, X, CLINICAL_FEATURES)
                features = expl.get("top_features", [])
                summary  = expl.get("summary")
            except Exception as exc:
                logger.warning("SHAP explanation failed: %s", exc)

    latest_fusion = (Prediction.query
                     .filter_by(patient_id=patient.id, modality="fusion")
                     .order_by(Prediction.created_at.desc()).first())
    if latest_fusion and latest_fusion.raw_output:
        contrib   = latest_fusion.raw_output.get("modality_contributions", {})
        attention = list(compute_attention_weights(contrib).values()) if contrib else []

    # Return empty lists — frontend handles empty state
    return success(data={
        "features":           features,
        "attention":          attention,
        "summary":            summary,
        "mri_heatmap_url":    None,
        "spiral_heatmap_url": None,
        "has_data":           len(features) > 0,
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

    latest_fusion = (Prediction.query
                     .filter_by(patient_id=patient.id, modality="fusion")
                     .order_by(Prediction.created_at.desc()).first())

    if not latest_fusion:
        # No predictions yet — return empty list (frontend shows empty state)
        return success(data=[])

    fused_data = latest_fusion.raw_output or {"severity": 0}
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

    # No predictions → return empty arrays (frontend detects and shows empty state)
    if not preds:
        return success(data={
            "labels":        [],
            "severity":      [],
            "updrs":         [],
            "voice":         [],
            "mri":           [],
            "motor":         [],
            "interventions": [],
            "has_data":      False,
        })

    by_month: dict = {}
    for p in preds:
        key = p.created_at.strftime("%b %y")
        if key not in by_month:
            by_month[key] = {"fusion":[], "voice":[], "mri":[], "motor":[], "updrs":[]}
        if p.modality == "fusion":   by_month[key]["fusion"].append(p.severity or 0)
        elif p.modality == "voice":  by_month[key]["voice"].append(p.confidence or 0)
        elif p.modality == "mri":    by_month[key]["mri"].append(p.confidence or 0)
        elif p.modality == "motor":
            by_month[key]["motor"].append(p.confidence or 0)
            raw = p.raw_output or {}
            by_month[key]["updrs"].append(float(raw.get("updrs") or 0))

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
        {"date": p.created_at.strftime("%b %y"),
         "event": f"Fusion prediction — severity {p.severity:.0f}"}
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
        "has_data":      True,
    })


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
def _slim(r):
    return {k: r.get(k) for k in ("probability","confidence","severity","label","has_parkinson")}

def _persist(patient_uid, modality, result):
    try:
        patient = Patient.query.filter_by(patient_uid=patient_uid).first()
        if not patient: return
        prob = result.get("probability") or result.get("final_probability")
        db.session.add(Prediction(
            patient_id = patient.id, modality = modality,
            result     = prob,
            label      = result.get("label"),
            severity   = result.get("severity"),
            confidence = result.get("confidence"),
            raw_output = result,
        ))
        db.session.commit()
    except Exception as exc:
        logger.error("Prediction persist failed: %s", exc)
        db.session.rollback()

def _initials(user):
    if not user: return "??"
    return ((user.first_name or "")[0] + (user.last_name or "")[0]).upper() or "??"

def _status(s):
    if s >= 70: return "critical"
    if s >= 40: return "warning"
    return "stable"

def _sev_change(patient):
    preds = (Prediction.query
             .filter_by(patient_id=patient.id, modality="fusion")
             .order_by(Prediction.created_at.desc()).limit(2).all())
    if len(preds) == 2 and preds[0].severity and preds[1].severity:
        return round(preds[0].severity - preds[1].severity, 1)
    return None

def _updrs(latest):
    for mod in ("motor","clinical"):
        raw = (latest.get(mod) and getattr(latest[mod], 'raw_output', None)) or {}
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
    vp = getattr(latest.get("voice"), "result", None)
    if vp and vp >= 0.7:
        risks.append({"name":"Dysarthria","level":"medium"})
    mp = getattr(latest.get("mri"), "result", None)
    if mp and mp >= 0.8:
        risks.append({"name":"Dopamine Loss","level":"high"})
    return risks

def _build_alerts(severity, latest):
    alerts = []
    if severity >= 70:
        alerts.append({"type":"critical","msg":"Severity ≥ 70 — urgent clinical review required","time":"Latest prediction"})
    vp = getattr(latest.get("voice"), "result", None)
    if vp and vp >= 0.75:
        alerts.append({"type":"warning","msg":"Voice biomarkers indicate dysarthria progression","time":"Latest scan"})
    if not alerts:
        alerts.append({"type":"info","msg":"No critical flags detected at this time","time":"Latest prediction"})
    return alerts

def _build_explanation(fused, results, skipped):
    mods   = list(results.keys())
    sev    = fused.get("severity", 0)
    stage  = fused.get("stage", "—")
    method = fused.get("fusion_method", "ensemble")
    parts  = [f"Fusion ({method}) ran on {len(mods)} modality(ies): {', '.join(mods)}."]
    parts.append(f"Severity: {sev:.1f}/100 ({stage}).")
    if skipped:
        parts.append(f"Skipped ({len(skipped)} unavailable): {', '.join(skipped.keys())}.")
    return " ".join(parts)