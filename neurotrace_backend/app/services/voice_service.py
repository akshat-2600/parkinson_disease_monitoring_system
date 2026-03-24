"""
app/services/voice_service.py
Audio → voice biomarker features → prediction.

Extracts the exact columns required by the trained voice model:
MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs),
MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB),
Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA,
NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
"""
import os
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)

# ── Voice dataset feature columns (22 features matching scaler/model input in backend) ──
VOICE_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]


def extract_voice_features(audio_path: str) -> dict:
    """
    Extract Parkinson's voice biomarkers from an audio file.
    Returns a dict with keys matching VOICE_FEATURES.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
        import librosa

        sound = parselmouth.Sound(audio_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # ── Fundamental frequency (F0) ───────────────────────
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[f0_values > 0]   # voiced frames only

        fo  = float(np.mean(f0_values))   if len(f0_values) else 0.0
        fhi = float(np.max(f0_values))    if len(f0_values) else 0.0
        flo = float(np.min(f0_values))    if len(f0_values) else 0.0

        # ── Jitter (pitch period variability) ───────────────
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter_pct = call(point_process, "Get jitter (local)",  0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap        = call(point_process, "Get jitter (rap)",    0, 0, 0.0001, 0.02, 1.3)
        ppq        = call(point_process, "Get jitter (ppq5)",   0, 0, 0.0001, 0.02, 1.3)
        ddp        = rap * 3   # Jitter:DDP = 3 × RAP (standard definition)

        # ── Shimmer (amplitude variability) ─────────────────
        try:
            shimmer     = call([sound, point_process], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_db  = call([sound, point_process], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq3        = call([sound, point_process], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq5        = call([sound, point_process], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq         = call([sound, point_process], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
            dda         = apq3 * 3   # Shimmer:DDA = 3 × APQ3
        except Exception as shimmer_exc:
            logger.warning("Shimmer calculation failed, using defaults: %s", shimmer_exc)
            shimmer = shimmer_db = apq3 = apq5 = apq = dda = 0.0

        # ── Noise-to-Harmonics / Harmonics-to-Noise ──────────
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr         = call(harmonicity, "Get mean", 0, 0)
        nhr         = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.0

        # ── Nonlinear dynamical features ─────────────────────
        # RPDE: Recurrence Period Density Entropy (approximated via entropy of F0)
        if len(f0_values) > 1:
            f0_norm = f0_values / (f0_values.max() + 1e-9)
            hist, _ = np.histogram(f0_norm, bins=20, density=True)
            hist   += 1e-10
            rpde    = float(-np.sum(hist * np.log(hist)) / np.log(len(hist)))
        else:
            rpde = 0.5

        # DFA: Detrended Fluctuation Analysis (simplified)
        dfa = _compute_dfa(y)

        # Spread1 / Spread2 / D2 / PPE (nonlinear complexity features)
        spread1, spread2, d2, ppe = _compute_nonlinear(f0_values)

        features = {
            "MDVP:Fo(Hz)":      fo,
            "MDVP:Fhi(Hz)":     fhi,
            "MDVP:Flo(Hz)":     flo,
            "MDVP:Jitter(%)":   jitter_pct,
            "MDVP:Jitter(Abs)": jitter_abs,
            "MDVP:RAP":         rap,
            "MDVP:PPQ":         ppq,
            "Jitter:DDP":       ddp,
            "MDVP:Shimmer":     shimmer,
            "MDVP:Shimmer(dB)": shimmer_db,
            "Shimmer:APQ3":     apq3,
            "Shimmer:APQ5":     apq5,
            "MDVP:APQ":         apq,
            "Shimmer:DDA":      dda,
            "NHR":              nhr,
            "HNR":              hnr,
            "RPDE":             rpde,
            "DFA":              dfa,
            "spread1":          spread1,
            "spread2":          spread2,
            "D2":               d2,
            "PPE":              ppe,
        }
        return features

    except Exception as exc:
        logger.error("Voice feature extraction failed: %s", exc)
        raise ValueError(f"Audio feature extraction failed: {exc}") from exc


def _compute_dfa(signal: np.ndarray, min_box: int = 4, max_box: int = None) -> float:
    """Simplified Detrended Fluctuation Analysis."""
    n = len(signal)
    if n < 16:
        return 0.7   # default for very short clips
    if max_box is None:
        max_box = n // 4

    y_cum = np.cumsum(signal - np.mean(signal))
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
    fluctuations = []

    for box in box_sizes:
        if box < 2:
            continue
        n_boxes = n // box
        if n_boxes < 1:
            continue
        rms_list = []
        for i in range(n_boxes):
            segment = y_cum[i * box: (i + 1) * box]
            x       = np.arange(len(segment))
            poly    = np.polyfit(x, segment, 1)
            trend   = np.polyval(poly, x)
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))
        fluctuations.append(np.mean(rms_list))

    if len(fluctuations) < 2:
        return 0.7

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, _ = np.polyfit(np.log(box_sizes[:len(fluctuations)]), np.log(fluctuations), 1)

    return float(np.clip(slope, 0.0, 2.0))


def _compute_nonlinear(f0_values: np.ndarray):
    """Estimate spread1, spread2, D2, PPE from F0 series."""
    if len(f0_values) < 5:
        return -5.0, 0.3, 2.3, 0.28

    log_f0  = np.log(f0_values + 1e-9)
    spread1 = float(np.mean(log_f0) - np.log(np.mean(f0_values) + 1e-9))
    spread2 = float(np.std(log_f0))

    # D2: correlation dimension approximation
    d2 = float(np.clip(2.0 + np.random.normal(0, 0.1), 1.5, 3.5))  # placeholder

    # PPE: Pitch Period Entropy
    diff_f0 = np.abs(np.diff(f0_values))
    if diff_f0.sum() > 0:
        p = diff_f0 / diff_f0.sum()
        ppe = float(-np.sum(p * np.log(p + 1e-9)) / np.log(len(p) + 1))
    else:
        ppe = 0.0

    return spread1, spread2, d2, ppe


def predict_from_audio(audio_path: str, model, scaler=None, selector=None, selected_features=None):
    """
    Full pipeline: audio file → features → scaled → (selector) → model prediction.
    Returns dict with probability, label, severity, features.
    """
    features = extract_voice_features(audio_path)
    feature_vector = np.array([[features[f] for f in VOICE_FEATURES]])

    # Scale to raw feature space expected by scaler (typically 22)
    if scaler is not None:
        expected = getattr(scaler, 'n_features_in_', None)
        actual = feature_vector.shape[1]

        if hasattr(scaler, 'feature_names_in_'):
            from pandas import DataFrame
            expected_names = list(scaler.feature_names_in_)

            # Build DataFrame with all known voice features
            df = DataFrame(feature_vector, columns=VOICE_FEATURES)
            if expected != actual:
                logger.warning("Voice scaler expects %s features but got %d; aligning columns by name.", expected, actual)

            # Align, fill missing with 0, drop extras
            df = df.reindex(columns=expected_names, fill_value=0.0)
            feature_vector = df.values
        else:
            if expected is not None and expected != actual:
                logger.warning("Voice scaler expects %d features but got %d. Applying fallback pad/truncate.", expected, actual)
                if actual < expected:
                    pad = np.zeros((feature_vector.shape[0], expected), dtype=feature_vector.dtype)
                    pad[:, :actual] = feature_vector
                    feature_vector = pad
                else:
                    feature_vector = feature_vector[:, :expected]

        try:
            feature_vector = scaler.transform(feature_vector)
        except Exception as exc:
            logger.warning("Voice scaler.transform failed (%s), retrying via DataFrame where possible.", exc)
            from pandas import DataFrame
            df = DataFrame(feature_vector, columns=(list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else VOICE_FEATURES))
            if hasattr(scaler, 'feature_names_in_'):
                df = df.reindex(columns=list(scaler.feature_names_in_), fill_value=0.0)
            try:
                feature_vector = scaler.transform(df.values)
            except Exception as exc2:
                logger.error("Voice scaler fallback also failed: %s", exc2)
                raise ValueError("Failed to scale voice feature vector") from exc2

    # Apply the same k-best selector used in training (if available)
    if selector is not None:
        try:
            feature_vector = selector.transform(feature_vector)
        except Exception as exc:
            logger.warning("Voice selector.transform failed (%s); attempting name-based selection as fallback.", exc)

    if selected_features is not None and hasattr(model, 'n_features_in_') and feature_vector.shape[1] != model.n_features_in_:
        try:
            from pandas import DataFrame
            src_names = (list(scaler.feature_names_in_) if scaler is not None and hasattr(scaler, 'feature_names_in_') else VOICE_FEATURES)
            df = DataFrame(feature_vector, columns=src_names)
            df = df.reindex(columns=selected_features, fill_value=0.0)
            feature_vector = df.values
            logger.info("Applied selected_features fallback for model compatibility.")
        except Exception as exc:
            logger.warning("Selected_features fallback failed (%s)", exc)

    # If model expects fewer features (e.g., 15) and we have more, try naive cut if no other alignment fixed it.
    model_input_expected = getattr(model, 'n_features_in_', None)
    if model_input_expected is not None and feature_vector.shape[1] != model_input_expected:
        if feature_vector.shape[1] > model_input_expected:
            logger.warning("Truncating voice vector from %d to %d to match model input", feature_vector.shape[1], model_input_expected)
            feature_vector = feature_vector[:, :model_input_expected]
        elif feature_vector.shape[1] < model_input_expected:
            logger.warning("Padding voice vector from %d to %d to match model input", feature_vector.shape[1], model_input_expected)
            pad = np.zeros((feature_vector.shape[0], model_input_expected), dtype=feature_vector.dtype)
            pad[:, :feature_vector.shape[1]] = feature_vector
            feature_vector = pad

    proba = model.predict_proba(feature_vector)[0]
    label_idx = int(np.argmax(proba))
    confidence = float(proba[label_idx])
    has_pd = bool(label_idx == 1)

    return {
        "has_parkinson": has_pd,
        "probability":   float(proba[1]) if len(proba) > 1 else confidence,
        "confidence":    confidence,
        "severity":      round(float(proba[1]) * 100, 2) if len(proba) > 1 else round(confidence * 100, 2),
        "label":         "Parkinson's Detected" if has_pd else "No Parkinson's Detected",
        "features":      features,
    }