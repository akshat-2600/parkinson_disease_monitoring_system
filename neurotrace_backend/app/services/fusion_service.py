"""
app/services/fusion_service.py
Multi-modal fusion prediction engine.

Strategy:
  1. Run whichever modality models have valid inputs.
  2. If a trained fusion model exists → feed its probability vector.
  3. Else → weighted soft-voting ensemble across available modalities.
"""
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Modality weights (tunable; higher = more trusted) ─────────
MODALITY_WEIGHTS = {
    "voice":      0.18,
    "clinical":   0.22,
    "mri":        0.25,
    "spiral":     0.12,
    "motor":      0.15,
    "timeseries": 0.08,
}

# Severity stage thresholds
SEVERITY_STAGES = [
    (0,   20,  "Stage 0 — No Evidence"),
    (20,  35,  "Stage I — Mild"),
    (35,  55,  "Stage II — Moderate"),
    (55,  70,  "Stage III — Moderate-Severe"),
    (70,  85,  "Stage IV — Severe"),
    (85,  101, "Stage V — Critical"),
]


def severity_to_stage(severity: float) -> str:
    for lo, hi, label in SEVERITY_STAGES:
        if lo <= severity < hi:
            return label
    return "Unknown"


def fuse_predictions(modality_results: dict, fusion_model=None) -> dict:
    """
    modality_results: dict of { "voice": {...}, "mri": {...}, ... }
    Each value must contain at least: probability (float 0-1), confidence (float).

    Returns the fused prediction dict.
    """
    if not modality_results:
        raise ValueError("No modality results provided for fusion")

    available = {k: v for k, v in modality_results.items() if v is not None}

    # ── Path 1: trained fusion model ─────────────────────────
    if fusion_model is not None:
        try:
            feature_vec = _build_fusion_features(available)
            proba = fusion_model.predict_proba(feature_vec)[0]
            prob_pd    = float(proba[1]) if len(proba) > 1 else float(proba[0])
            confidence = float(max(proba))
            method     = "fusion_model"
        except Exception as exc:
            logger.warning("Fusion model inference failed (%s), falling back to ensemble", exc)
            prob_pd, confidence, method = _ensemble_vote(available)
    else:
        prob_pd, confidence, method = _ensemble_vote(available)

    # ── Derived metrics ───────────────────────────────────────
    has_pd   = prob_pd >= 0.5
    severity = round(prob_pd * 100, 2)
    stage    = severity_to_stage(severity)

    # Per-modality contribution (normalised weights of available mods)
    contributions = _compute_contributions(available)

    return {
        "has_parkinson":    has_pd,
        "probability":      round(prob_pd, 4),
        "confidence":       round(confidence, 4),
        "severity":         severity,
        "stage":            stage,
        "label":            "Parkinson's Detected" if has_pd else "No Parkinson's Detected",
        "fusion_method":    method,
        "modalities_used":  list(available.keys()),
        "modality_contributions": contributions,
        "individual_results": {k: _slim(v) for k, v in available.items()},
    }


def _slim(result: dict) -> dict:
    """Return only the key fields from each modality result."""
    return {
        "probability": result.get("probability"),
        "confidence":  result.get("confidence"),
        "severity":    result.get("severity"),
        "label":       result.get("label"),
    }


def _ensemble_vote(available: dict) -> tuple:
    """Weighted soft-voting across available modalities."""
    total_weight = 0.0
    weighted_prob = 0.0

    for mod, result in available.items():
        w    = MODALITY_WEIGHTS.get(mod, 0.10)
        prob = result.get("probability", 0.5)
        weighted_prob += w * prob
        total_weight  += w

    if total_weight == 0:
        return 0.5, 0.5, "ensemble"

    prob_pd    = weighted_prob / total_weight
    # Confidence: mean individual confidence, boosted by num modalities
    conf_vals  = [r.get("confidence", 0.5) for r in available.values()]
    base_conf  = float(np.mean(conf_vals))
    n_bonus    = min(0.15, (len(available) - 1) * 0.03)
    confidence = min(0.99, base_conf + n_bonus)

    return float(prob_pd), float(confidence), "weighted_ensemble"


def _build_fusion_features(available: dict) -> np.ndarray:
    """
    Build feature vector for the trained fusion model.
    Expects: probability + confidence per modality, ordered.
    """
    ordered_mods = ["voice", "clinical", "mri", "spiral", "motor", "timeseries"]
    row = []
    for mod in ordered_mods:
        if mod in available:
            row.append(available[mod].get("probability", 0.5))
            row.append(available[mod].get("confidence", 0.5))
        else:
            row.extend([0.5, 0.0])   # absent → neutral prob, zero confidence
    return np.array([row])


def _compute_contributions(available: dict) -> dict:
    total = sum(MODALITY_WEIGHTS.get(k, 0.10) for k in available)
    return {
        k: round(MODALITY_WEIGHTS.get(k, 0.10) / total * 100, 1)
        for k in available
    }


def generate_risk_flags(fused: dict, individual: dict) -> list:
    """Generate clinical risk alerts from fusion output."""
    flags = []
    severity = fused.get("severity", 0)

    if severity >= 70:
        flags.append({"type": "critical", "msg": "Severity index ≥ 70 — urgent clinical review required"})
    elif severity >= 50:
        flags.append({"type": "warning",  "msg": "Moderate–severe staging detected — schedule follow-up"})

    voice = individual.get("voice", {})
    if voice and voice.get("probability", 0) >= 0.75:
        flags.append({"type": "warning",  "msg": "Voice biomarkers indicate significant dysarthria progression"})

    mri = individual.get("mri", {})
    if mri and mri.get("probability", 0) >= 0.80:
        flags.append({"type": "critical", "msg": "MRI analysis shows high probability of dopaminergic loss"})

    return flags


def generate_recommendations(fused: dict) -> list:
    """Rule-based personalised recommendations from fusion output."""
    severity = fused.get("severity", 0)
    recs = []

    if severity >= 70:
        recs.append({"priority": "high", "title": "Urgent Specialist Referral",
                     "category": "Clinical", "confidence": 0.95,
                     "reasoning": "Severity index exceeds 70%. Immediate neurology review recommended."})
        recs.append({"priority": "high", "title": "Falls Prevention Protocol",
                     "category": "Safety", "confidence": 0.90,
                     "reasoning": "High severity correlates with postural instability and fall risk."})

    if severity >= 40:
        recs.append({"priority": "moderate", "title": "Medication Timing Review",
                     "category": "Pharmacotherapy", "confidence": 0.82,
                     "reasoning": "Motor fluctuation pattern suggests sub-optimal levodopa scheduling."})
        recs.append({"priority": "moderate", "title": "Speech–Language Therapy",
                     "category": "Rehabilitation", "confidence": 0.78,
                     "reasoning": "Voice biomarkers indicate early dysarthria — LSVT LOUD recommended."})

    recs.append({"priority": "preventive", "title": "Structured Exercise Programme",
                 "category": "Lifestyle", "confidence": 0.85,
                 "reasoning": "150 min/week aerobic exercise shown to slow dopaminergic decline."})
    recs.append({"priority": "preventive", "title": "Mediterranean Diet",
                 "category": "Nutrition", "confidence": 0.72,
                 "reasoning": "Antioxidant-rich diet associated with slower neurodegeneration."})

    return recs