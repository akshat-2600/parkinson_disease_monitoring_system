"""
app/services/motor_service.py
Motor examination data → UPDRS III prediction.

Handles the large motor dataset with speech timing, tremor, rigidity,
finger tapping, gait columns etc.
"""
import numpy as np
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

# ── Feature columns from motor dataset (exclude ID + target) ─
MOTOR_FEATURES = [
    "Age (years)", "Gender",
    "Positive history of Parkinson disease in family",
    "Age of disease onset (years)",
    "Duration of disease from first symptoms (years)",
    "Antidepressant therapy", "Antiparkinsonian medication",
    "Antipsychotic medication", "Benzodiazepine medication",
    "Levodopa equivalent (mg/day)", "Clonazepam (mg/day)",
    # UPDRS III sub-items
    "18. Speech", "19. Facial Expression",
    "20. Tremor at Rest - head",
    "20. Tremor at Rest - RUE", "20. Tremor at Rest - LUE",
    "20. Tremor at Rest - RLE", "20. Tremor at Rest - LLE",
    "21. Action or Postural Tremor - RUE", "21. Action or Postural Tremor - LUE",
    "22. Rigidity - neck",
    "22. Rigidity - RUE", "22. Rigidity - LUE",
    "22. Rigidity - RLE", "22. Rigidity - LLE",
    "23.Finger Taps - RUE", "23.Finger Taps - LUE",
    "24. Hand Movements - RUE", "24. Hand Movements - LUE",
    "25. Rapid Alternating Movements - RUE", "25. Rapid Alternating Movements - LUE",
    "26. Leg Agility - RLE", "26. Leg Agility - LLE",
    "27. Arising from Chair", "28. Posture", "29. Gait",
    "30. Postural Stability", "31. Body Bradykinesia and Hypokinesia",
    # Speech timing features (session 1)
    "Entropy of speech timing (-)",
    "Rate of speech timing (-/min)",
    "Acceleration of speech timing (-/min2)",
    "Duration of pause intervals (ms)",
    "Duration of voiced intervals (ms)",
    "Gaping in-between voiced intervals (-/min)",
    "Duration of unvoiced stops (ms)",
    "Decay of unvoiced fricatives (\u2030/min)",
    "Relative loudness of respiration (dB)",
    "Pause intervals per respiration (-)",
    "Rate of speech respiration (-/min)",
    "Latency of respiratory exchange (ms)",
]

MOTOR_DEFAULTS = {
    "Age (years)": 65, "Gender": 0,
    "Positive history of Parkinson disease in family": 0,
    "Age of disease onset (years)": 60,
    "Duration of disease from first symptoms (years)": 5,
    "Antidepressant therapy": 0, "Antiparkinsonian medication": 1,
    "Antipsychotic medication": 0, "Benzodiazepine medication": 0,
    "Levodopa equivalent (mg/day)": 300, "Clonazepam (mg/day)": 0,
    "18. Speech": 0, "19. Facial Expression": 0,
    "20. Tremor at Rest - head": 0,
    "20. Tremor at Rest - RUE": 0, "20. Tremor at Rest - LUE": 0,
    "20. Tremor at Rest - RLE": 0, "20. Tremor at Rest - LLE": 0,
    "21. Action or Postural Tremor - RUE": 0, "21. Action or Postural Tremor - LUE": 0,
    "22. Rigidity - neck": 0,
    "22. Rigidity - RUE": 0, "22. Rigidity - LUE": 0,
    "22. Rigidity - RLE": 0, "22. Rigidity - LLE": 0,
    "23.Finger Taps - RUE": 0, "23.Finger Taps - LUE": 0,
    "24. Hand Movements - RUE": 0, "24. Hand Movements - LUE": 0,
    "25. Rapid Alternating Movements - RUE": 0, "25. Rapid Alternating Movements - LUE": 0,
    "26. Leg Agility - RLE": 0, "26. Leg Agility - LLE": 0,
    "27. Arising from Chair": 0, "28. Posture": 0, "29. Gait": 0,
    "30. Postural Stability": 0, "31. Body Bradykinesia and Hypokinesia": 0,
    "Entropy of speech timing (-)": 1.56,
    "Rate of speech timing (-/min)": 340, "Acceleration of speech timing (-/min2)": 10,
    "Duration of pause intervals (ms)": 160, "Duration of voiced intervals (ms)": 270,
    "Gaping in-between voiced intervals (-/min)": 50,
    "Duration of unvoiced stops (ms)": 25,
    "Decay of unvoiced fricatives (\u2030/min)": -1.5,
    "Relative loudness of respiration (dB)": -22,
    "Pause intervals per respiration (-)": 5,
    "Rate of speech respiration (-/min)": 17,
    "Latency of respiratory exchange (ms)": 200,
    "Overview of motor examination: Hoehn & Yahr scale (-)": 2.0,
    "Entropy of speech timing (-) .1": 1.56,
    "Rate of speech timing (-/min) .1": 340,
    "Acceleration of speech timing (-/min2) .1": 10,
    "Duration of pause intervals (ms) .1": 160,
    "Duration of voiced intervals (ms) .1": 270,
    "Gaping in-between voiced intervals (-/min) .1": 50,
    "Duration of unvoiced stops (ms) .1": 25,
    "Decay of unvoiced fricatives (‰/min) .1": -1.5,
    "Relative loudness of respiration (dB) .1": -22,
    "Pause intervals per respiration (-) .1": 5,
    "Rate of speech respiration (-/min) .1": 17,
    "Latency of respiratory exchange (ms) .1": 200,
}


def _canonical(name: str) -> str:
    """
    Normalize feature names so training names with extra spaces can map
    to API names (and vice-versa).
    """
    s = " ".join(str(name).replace("�", "‰").replace("\u2030", "‰").split()).strip().lower()
    s = re.sub(r"\s*\.\s*1$", " .1", s)
    return s


def _build_default_aliases():
    aliases = {}
    for k, v in MOTOR_DEFAULTS.items():
        aliases[_canonical(k)] = v

    # Common training variants with repeated spaces / punctuation.
    aliases.setdefault(_canonical("Overview  of  motor  examination:  Hoehn  &  Yahr  scale  (-)"), 2.0)
    aliases.setdefault(_canonical("Gaping  in-between  voiced  Intervals  (-/min)"), 50)
    aliases.setdefault(_canonical("Gaping  in-between  voiced  Intervals  (-/min) .1"), 50)
    return aliases


_MOTOR_DEFAULTS_ALIASED = _build_default_aliases()


def _to_float(value, default=0.0) -> float:
    """Coerce mixed API/CSV values (e.g., F/M, yes/no) into numeric form."""
    if value is None:
        return float(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return float(default)
    if s in {"m", "male", "true", "yes", "y"}:
        return 1.0
    if s in {"f", "female", "false", "no", "n"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return float(default)


def build_motor_vector(input_data: dict, expected_features=None) -> pd.DataFrame:
    """
    Build a feature dataframe aligned to expected model/scaler columns.
    If expected_features is None, falls back to MOTOR_FEATURES.
    """
    expected = list(expected_features) if expected_features is not None else list(MOTOR_FEATURES)
    input_alias = {_canonical(k): v for k, v in input_data.items()}

    row = {}
    for col in expected:
        key = _canonical(col)
        val = input_alias.get(key, _MOTOR_DEFAULTS_ALIASED.get(key, 0.0))
        row[col] = _to_float(val, default=_MOTOR_DEFAULTS_ALIASED.get(key, 0.0))
    return pd.DataFrame([row], columns=expected)


def predict_motor(input_data: dict, model, scaler=None) -> dict:
    """
    Motor examination data → UPDRS prediction + Parkinson's probability.
    """
    expected_features = None
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = list(MOTOR_FEATURES)

    X_df = build_motor_vector(input_data, expected_features=expected_features)
    input_keys_canon = {_canonical(k) for k in input_data.keys()}
    missing = [c for c in expected_features if _canonical(c) not in input_keys_canon]
    if missing:
        logger.info("Motor prediction: %d cols filled with defaults", len(missing))

    if scaler is not None:
        X = scaler.transform(X_df)
    else:
        X = X_df.values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        confidence = float(proba[label_idx])
        has_pd = bool(label_idx == 1)
        probability = float(proba[1]) if len(proba) > 1 else confidence
        severity = round(probability * 100, 2)
        extra = {}
    else:
        # Regression-style UPDRS model (e.g., RandomForestRegressor)
        updrs_pred = float(model.predict(X)[0])
        # Heuristic mapping to keep a fusion-friendly probability scale.
        probability = float(np.clip(updrs_pred / 108.0, 0.0, 1.0))
        has_pd = bool(updrs_pred >= 15.0)
        confidence = float(np.clip(abs(updrs_pred - 15.0) / 30.0, 0.5, 0.99))
        severity = round(probability * 100, 2)
        extra = {"updrs_predicted": round(updrs_pred, 2)}

    return {
        "has_parkinson": has_pd,
        "probability": probability,
        "confidence": confidence,
        "severity": severity,
        "label": "Parkinson's Detected" if has_pd else "No Parkinson's Detected",
        "hoehn_yahr_est": _estimate_hoehn_yahr(input_data),
        "missing_fields": missing,
        **extra,
    }


def _estimate_hoehn_yahr(data: dict) -> float:
    """Rough Hoehn & Yahr stage estimate from UPDRS sub-items."""
    updrs = sum([
        float(data.get("20. Tremor at Rest - RUE", 0)),
        float(data.get("20. Tremor at Rest - LUE", 0)),
        float(data.get("28. Posture", 0)),
        float(data.get("29. Gait", 0)),
        float(data.get("30. Postural Stability", 0)),
    ])
    if updrs <= 2:   return 1.0
    if updrs <= 5:   return 1.5
    if updrs <= 8:   return 2.0
    if updrs <= 12:  return 2.5
    if updrs <= 18:  return 3.0
    if updrs <= 24:  return 4.0
    return 5.0


def parse_motor_csv(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    drop_cols = ["Participant  code", "Overview  of  motor  examination:  Hoehn  &  Yahr  scale  (-)",
                 "Overview  of  motor  examination:  UPDRS  III  total  (-)"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if df.empty:
        raise ValueError("Motor CSV has no data rows")
    # Normalise column names (collapse multiple spaces)
    df.columns = [" ".join(c.split()) for c in df.columns]
    return df.iloc[0].to_dict()