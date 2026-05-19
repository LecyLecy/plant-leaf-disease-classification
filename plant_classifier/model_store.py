from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle


@dataclass
class ModelBundle:
    """
    Container for storing a trained model together with its supporting objects.

    Attributes:
        variant: Model variant, for example "full".
        model_name: Human-readable model name.
        model: Trained machine learning model.
        label_encoder: Fitted LabelEncoder used to decode prediction labels.
        feature_columns: List of feature column names used during training.
        metrics: Evaluation metrics saved with the model.
        metadata: Additional information such as random_state or notes.
        saved_at: UTC timestamp when the model bundle was saved.
    """

    variant: str
    model_name: str
    model: Any
    label_encoder: Any
    feature_columns: List[str]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    saved_at: Optional[str] = None


def save_model_bundle(
    path: str | Path,
    variant: str,
    model_name: str,
    model: Any,
    label_encoder: Any,
    feature_columns: List[str],
    metrics: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a trained model and its supporting objects into a pickle file.

    Args:
        path: Output pickle file path.
        variant: Model variant name.
        model_name: Human-readable model name.
        model: Trained machine learning model.
        label_encoder: Fitted label encoder.
        feature_columns: Feature columns used during model training.
        metrics: Optional evaluation metrics.
        metadata: Optional additional information.

    Returns:
        Path to the saved pickle file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "variant": variant,
        "model_name": model_name,
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": list(feature_columns),
        "metrics": metrics or {},
        "metadata": metadata or {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(path, "wb") as file:
        pickle.dump(payload, file)

    return path


def load_model_bundle(path: str | Path) -> ModelBundle:
    """
    Load a saved model bundle from a pickle file.

    Args:
        path: Pickle file path.

    Returns:
        ModelBundle object.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    with open(path, "rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, ModelBundle):
        return payload

    if not isinstance(payload, dict):
        raise TypeError(
            "Invalid model bundle format. Expected a dictionary or ModelBundle object."
        )

    required_keys = [
        "variant",
        "model_name",
        "model",
        "label_encoder",
        "feature_columns",
    ]

    missing_keys = [key for key in required_keys if key not in payload]

    if missing_keys:
        raise KeyError(f"Missing required keys in model bundle: {missing_keys}")

    return ModelBundle(
        variant=payload["variant"],
        model_name=payload["model_name"],
        model=payload["model"],
        label_encoder=payload["label_encoder"],
        feature_columns=list(payload["feature_columns"]),
        metrics=payload.get("metrics", {}),
        metadata=payload.get("metadata", {}),
        saved_at=payload.get("saved_at"),
    )