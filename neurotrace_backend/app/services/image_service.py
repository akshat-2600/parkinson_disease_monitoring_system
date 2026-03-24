"""
app/services/image_service.py
Shared image preprocessing + prediction for MRI and Spiral models.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Model-specific image sizes ────────────────────────────────
MRI_IMAGE_SIZE    = (96, 96)   # MRI model expects 96x96
SPIRAL_IMAGE_SIZE = (128, 128)   # spiral drawing model input

LABELS = ["No Parkinson's Detected", "Parkinson's Detected"]


def _load_and_preprocess(image_path: str, target_size: tuple, grayscale: bool = False) -> np.ndarray:
    """Load an image, resize, normalise to [0,1]."""
    try:
        from PIL import Image
        img = Image.open(image_path)
        if grayscale:
            img = img.convert("L")
            img = img.convert("RGB")   # keep 3-channel for CNN
        else:
            img = img.convert("RGB")
        img = img.resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)   # (1, H, W, C)
    except Exception as exc:
        raise ValueError(f"Image preprocessing failed for {image_path}: {exc}") from exc


def _preprocess_drawing(image_path: str, target_size: tuple = SPIRAL_IMAGE_SIZE):
    """Preprocess drawing exactly like training notebook pipeline."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return denoised, edges, contours
    except Exception as exc:
        raise ValueError(f"Drawing preprocessing failed: {exc}") from exc


def _extract_31_drawing_features(image_path: str) -> np.ndarray:
    """Extract 31 handcrafted features for sklearn spiral/wave models."""
    from scipy import ndimage
    from scipy.stats import entropy, skew, kurtosis
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

    denoised, edges, contours = _preprocess_drawing(image_path)

    # 1) Tremor frequency (3)
    profile = denoised.mean(axis=1).astype(np.float32)
    fft_vals = np.abs(np.fft.rfft(profile - profile.mean()))
    freqs = np.fft.rfftfreq(len(profile))
    dom_freq = float(freqs[np.argmax(fft_vals[1:]) + 1]) if len(fft_vals) > 1 else 0.0
    spectral_entropy = float(entropy(fft_vals + 1e-9))
    total_power = float(np.sum(fft_vals ** 2))
    high_freq_power = float(np.sum(fft_vals[len(fft_vals) // 4:] ** 2))
    tremor_index = float(high_freq_power / (total_power + 1e-9))

    # 2) Line smoothness (3)
    if contours and len(contours[0]) >= 5:
        pts = contours[0].squeeze().astype(float)
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curv = np.abs(dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2 + 1e-9) ** 1.5
        mean_curv = float(np.mean(curv))
        std_curv = float(np.std(curv))
        max_curv = float(np.max(curv))
    else:
        mean_curv = std_curv = max_curv = 0.0

    # 3) Spiral deviation (3)
    h, w = edges.shape
    cx, cy = w // 2, h // 2
    theta = np.linspace(0, 6 * np.pi, 500)
    r_max = min(cx, cy) * 0.9
    r_spiral = r_max * theta / (6 * np.pi)
    xs = (cx + r_spiral * np.cos(theta)).astype(int).clip(0, w - 1)
    ys = (cy + r_spiral * np.sin(theta)).astype(int).clip(0, h - 1)
    ideal = np.zeros((h, w), dtype=np.uint8)
    ideal[ys, xs] = 255
    dt_ideal = ndimage.distance_transform_edt(255 - ideal)
    edges_bin = (edges > 0).astype(np.uint8)
    masked = dt_ideal * edges_bin
    nz = masked[masked > 0]
    mean_dev = float(nz.mean()) if nz.size else 0.0
    std_dev = float(nz.std()) if nz.size else 0.0
    ys_e, xs_e = np.where(edges_bin > 0)
    if len(ys_e) > 10:
        dists = np.sqrt((xs_e - cx) ** 2 + (ys_e - cy) ** 2)
        circ_dev = float(np.std(dists) / (np.mean(dists) + 1e-9))
    else:
        circ_dev = 0.0

    # 4) Fractal dimension (1)
    def _box_count(img, box_size):
        hh, ww = img.shape
        n_rows = int(np.ceil(hh / box_size))
        n_cols = int(np.ceil(ww / box_size))
        count = 0
        for r in range(n_rows):
            for c in range(n_cols):
                patch = img[r * box_size:(r + 1) * box_size, c * box_size:(c + 1) * box_size]
                if patch.any():
                    count += 1
        return count

    sizes = [2, 4, 8, 16, 32]
    counts = [_box_count(edges_bin, s) for s in sizes]
    valid = [(s, c) for s, c in zip(sizes, counts) if c > 0]
    if len(valid) >= 2:
        log_s = np.log([v[0] for v in valid])
        log_c = np.log([v[1] for v in valid])
        fractal_dim = float(abs(np.polyfit(log_s, log_c, 1)[0]))
    else:
        fractal_dim = 0.0

    # 5) Texture (15)
    lbp = local_binary_pattern(denoised, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    glcm = graycomatrix(
        denoised, [1], [0, np.pi / 2], levels=256, symmetric=True, normed=True
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    dissim = float(graycoprops(glcm, "dissimilarity").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())

    # 6) Statistical moments (6)
    flat = denoised.astype(float).ravel()
    moments = [
        float(np.mean(flat)),
        float(np.std(flat)),
        float(skew(flat)),
        float(kurtosis(flat)),
        float(np.percentile(flat, 25)),
        float(np.percentile(flat, 75)),
    ]

    features = [
        dom_freq, spectral_entropy, tremor_index,
        mean_curv, std_curv, max_curv,
        mean_dev, std_dev, circ_dev,
        fractal_dim,
        *hist.tolist(),
        contrast, dissim, homogeneity, energy, correlation,
        *moments,
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)


def _predict_probabilities(model, X: np.ndarray, is_keras_input: bool):
    """Return (prob_no_pd, prob_pd) for keras/sklearn-like models."""
    if is_keras_input:
        raw = model.predict(X, verbose=0)
        if raw.shape[-1] == 1:
            prob_pd = float(raw[0][0])
            return 1.0 - prob_pd, prob_pd
        return float(raw[0][0]), float(raw[0][1])

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 1:
            prob_pd = float(p[0][0])
            return 1.0 - prob_pd, prob_pd
        return float(p[0][0]), float(p[0][1])

    pred = int(model.predict(X)[0])
    prob_pd = float(pred)
    return 1.0 - prob_pd, prob_pd


def predict_mri(image_path: str, model) -> dict:
    """
    MRI scan → Parkinson's probability using a pre-trained Keras CNN.
    """
    X = _load_and_preprocess(image_path, MRI_IMAGE_SIZE, grayscale=False)
    raw = model.predict(X, verbose=0)

    # Support both binary sigmoid and softmax output
    if raw.shape[-1] == 1:
        prob_pd    = float(raw[0][0])
        prob_no_pd = 1.0 - prob_pd
    else:
        prob_no_pd = float(raw[0][0])
        prob_pd    = float(raw[0][1])

    has_pd     = prob_pd >= 0.5
    confidence = prob_pd if has_pd else prob_no_pd

    return {
        "has_parkinson": has_pd,
        "probability":   round(prob_pd, 4),
        "confidence":    round(confidence, 4),
        "severity":      round(prob_pd * 100, 2),
        "label":         LABELS[int(has_pd)],
        "model":         "mri",
    }


def predict_spiral(image_path: str, model) -> dict:
    """
    Spiral drawing image → Parkinson's probability using a pre-trained CNN.
    Spiral drawings are typically grayscale hand-drawn images.
    """
    # Keras CNN expects image tensor; sklearn models expect 31 handcrafted features.
    is_keras = "tensorflow" in type(model).__module__ or "keras" in type(model).__module__
    if is_keras:
        X = _load_and_preprocess(image_path, SPIRAL_IMAGE_SIZE, grayscale=True)
    else:
        X = _extract_31_drawing_features(image_path)
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"Spiral model expects {expected} features but extracted {X.shape[1]}"
            )

    prob_no_pd, prob_pd = _predict_probabilities(model, X, is_keras_input=is_keras)

    has_pd     = prob_pd >= 0.5
    confidence = prob_pd if has_pd else prob_no_pd

    return {
        "has_parkinson": has_pd,
        "probability":   round(prob_pd, 4),
        "confidence":    round(confidence, 4),
        "severity":      round(prob_pd * 100, 2),
        "label":         LABELS[int(has_pd)],
        "model":         "spiral",
    }