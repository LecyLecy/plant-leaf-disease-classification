from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd


IMAGE_SIZE = (224, 224)


def prepare_uploaded_image(image_rgb: np.ndarray, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Resize uploaded RGB image into the same size used during training.
    """
    image_rgb = cv2.resize(image_rgb, image_size, interpolation=cv2.INTER_AREA)
    return image_rgb


def preprocess_image(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply noise reduction and color space conversion.
    This follows the preprocessing pipeline used in the notebook.
    """
    gaussian = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    denoised = cv2.medianBlur(gaussian, 3)

    hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)

    return {
        "rgb": image_rgb,
        "denoised": denoised,
        "hsv": hsv,
        "lab": lab,
        "gray": gray,
    }


def create_leaf_mask(hsv: np.ndarray) -> np.ndarray:
    """
    Create leaf mask using HSV saturation and value channels.
    """
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = ((saturation > 25) & (value > 35)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def otsu_spot_mask(lab: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """
    Segment disease spot candidates using Otsu thresholding on LAB channels.
    """
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    _, otsu_a = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_b = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    otsu_mask = cv2.bitwise_or(otsu_a, otsu_b)
    otsu_mask = cv2.bitwise_and(otsu_mask, leaf_mask)

    return otsu_mask


def kmeans_spot_mask(lab: np.ndarray, leaf_mask: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Segment disease spot candidates using K-Means on LAB A-B channels.
    """
    leaf_pixels = leaf_mask > 0

    if leaf_pixels.sum() < 20:
        return np.zeros(leaf_mask.shape, dtype=np.uint8)

    ab_pixels = lab[:, :, 1:3][leaf_pixels].astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)

    _, labels, centers = cv2.kmeans(
        ab_pixels,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    cluster_scores = centers[:, 0] + centers[:, 1]
    disease_cluster = int(np.argmax(cluster_scores))

    result = np.zeros(leaf_mask.shape, dtype=np.uint8)
    result[leaf_pixels] = (labels.flatten() == disease_cluster).astype(np.uint8) * 255

    return result


def segment_disease_spots(processed: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine Otsu, K-Means, and morphology to create disease spot mask.
    """
    hsv = processed["hsv"]
    lab = processed["lab"]

    leaf_mask = create_leaf_mask(hsv)
    otsu_mask = otsu_spot_mask(lab, leaf_mask)
    kmask = kmeans_spot_mask(lab, leaf_mask, k=3)

    combined = cv2.bitwise_or(otsu_mask, kmask)
    combined = cv2.bitwise_and(combined, leaf_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    return leaf_mask, combined


def safe_skew(values: np.ndarray) -> float:
    """
    Calculate skewness without scipy dependency.
    """
    values = values.astype(np.float32)
    std = values.std()

    if std < 1e-6:
        return 0.0

    centered = values - values.mean()
    return float(np.mean((centered / std) ** 3))


def color_moments(image: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Extract mean, standard deviation, and skewness from each color channel.
    """
    features = {}
    valid = mask > 0

    if valid.sum() == 0:
        valid = np.ones(mask.shape, dtype=bool)

    for channel_idx in range(image.shape[2]):
        values = image[:, :, channel_idx][valid].astype(np.float32)

        features[f"{prefix}_ch{channel_idx}_mean"] = float(values.mean())
        features[f"{prefix}_ch{channel_idx}_std"] = float(values.std())
        features[f"{prefix}_ch{channel_idx}_skew"] = safe_skew(values)

    return features


def glcm_features(gray: np.ndarray, mask: np.ndarray, levels: int = 16) -> Dict[str, float]:
    """
    Calculate simple GLCM texture features using four neighbor directions.
    """
    quantized = np.clip((gray.astype(np.float32) / 256 * levels).astype(np.int32), 0, levels - 1)
    valid = mask > 0
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for dy, dx in offsets:
        y_start = max(0, dy)
        y_end = gray.shape[0] + min(0, dy)
        x_start = max(0, dx)
        x_end = gray.shape[1] + min(0, dx)

        current = quantized[y_start:y_end, x_start:x_end]
        neighbor = quantized[y_start - dy:y_end - dy, x_start - dx:x_end - dx]

        current_mask = valid[y_start:y_end, x_start:x_end]
        neighbor_mask = valid[y_start - dy:y_end - dy, x_start - dx:x_end - dx]

        pair_mask = current_mask & neighbor_mask

        for i, j in zip(current[pair_mask].ravel(), neighbor[pair_mask].ravel()):
            glcm[i, j] += 1

    if glcm.sum() == 0:
        glcm += 1

    glcm /= glcm.sum()

    i_idx, j_idx = np.indices(glcm.shape)

    contrast = np.sum(((i_idx - j_idx) ** 2) * glcm)
    dissimilarity = np.sum(np.abs(i_idx - j_idx) * glcm)
    homogeneity = np.sum(glcm / (1.0 + np.abs(i_idx - j_idx)))
    energy = np.sqrt(np.sum(glcm ** 2))

    mean_i = np.sum(i_idx * glcm)
    mean_j = np.sum(j_idx * glcm)
    std_i = np.sqrt(np.sum(((i_idx - mean_i) ** 2) * glcm))
    std_j = np.sqrt(np.sum(((j_idx - mean_j) ** 2) * glcm))

    correlation = np.sum(
        ((i_idx - mean_i) * (j_idx - mean_j) * glcm) / (std_i * std_j + 1e-8)
    )

    return {
        "glcm_contrast": float(contrast),
        "glcm_dissimilarity": float(dissimilarity),
        "glcm_homogeneity": float(homogeneity),
        "glcm_energy": float(energy),
        "glcm_correlation": float(correlation),
    }


def lbp_histogram(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate 16-bin LBP histogram using 8-neighbor radius 1.
    """
    center = gray[1:-1, 1:-1]

    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)

    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    valid = mask[1:-1, 1:-1] > 0

    if valid.sum() == 0:
        valid = np.ones_like(center, dtype=bool)

    hist, _ = np.histogram(lbp[valid], bins=16, range=(0, 256), density=True)

    return {f"lbp_bin_{idx:02d}": float(value) for idx, value in enumerate(hist)}


def shape_features(spot_mask: np.ndarray, leaf_mask: np.ndarray) -> Dict[str, float]:
    """
    Extract disease area and shape features from segmentation masks.
    """
    leaf_area = max(int((leaf_mask > 0).sum()), 1)
    spot_area = int((spot_mask > 0).sum())

    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]

    return {
        "disease_area_ratio": float(spot_area / leaf_area),
        "spot_count": int(len(contours)),
        "largest_spot_ratio": float((max(contour_areas) if contour_areas else 0.0) / leaf_area),
    }


def extract_features_from_rgb(image_rgb: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract handcrafted features from uploaded RGB image.

    Returns:
        features, processed_rgb, leaf_mask, spot_mask
    """
    image_rgb = prepare_uploaded_image(image_rgb)
    processed = preprocess_image(image_rgb)
    leaf_mask, spot_mask = segment_disease_spots(processed)

    features = {}
    features.update(shape_features(spot_mask, leaf_mask))
    features.update(color_moments(processed["hsv"], leaf_mask, "hsv"))
    features.update(color_moments(processed["lab"], leaf_mask, "lab"))
    features.update(glcm_features(processed["gray"], leaf_mask, levels=16))
    features.update(lbp_histogram(processed["gray"], leaf_mask))

    return features, processed["rgb"], leaf_mask, spot_mask


def create_spot_overlay(image_rgb: np.ndarray, spot_mask: np.ndarray) -> np.ndarray:
    """
    Create red overlay visualization for disease spot mask.
    """
    overlay = image_rgb.copy()
    overlay[spot_mask > 0] = [255, 60, 40]

    blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    return blended


def predict_uploaded_image(image_rgb: np.ndarray, model_bundle) -> Dict[str, object]:
    """
    Run feature extraction and prediction using saved model bundle.
    """
    features, processed_rgb, leaf_mask, spot_mask = extract_features_from_rgb(image_rgb)

    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=model_bundle.feature_columns, fill_value=0)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    prediction_encoded = model_bundle.model.predict(feature_df)[0]
    predicted_label = model_bundle.label_encoder.inverse_transform([prediction_encoded])[0]

    confidence = None
    probabilities = None

    if hasattr(model_bundle.model, "predict_proba"):
        probabilities = model_bundle.model.predict_proba(feature_df)[0]
        confidence = float(np.max(probabilities))

    overlay = create_spot_overlay(processed_rgb, spot_mask)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": features,
        "processed_rgb": processed_rgb,
        "leaf_mask": leaf_mask,
        "spot_mask": spot_mask,
        "overlay": overlay,
    }