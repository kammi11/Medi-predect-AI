"""Save/load trained models and scalers with joblib."""

import joblib
from pathlib import Path

from src.config import MODELS_DIR


def save_model(model, name: str) -> Path:
    """Save a fitted model (or scaler) to models/<name>.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """Load a previously saved model/scaler from models/<name>.joblib."""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def model_exists(name: str) -> bool:
    return (MODELS_DIR / f"{name}.joblib").exists()
