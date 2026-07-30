"""Single-patient prediction engine used by the Prediction page."""

import pandas as pd

from src.config import ALL_FEATURES
from src.model_utils import load_model


def build_patient_dataframe(inputs: dict) -> pd.DataFrame:
    """Convert a dict of form inputs into a single-row dataframe with the
    correct column order expected by the model."""
    return pd.DataFrame([[inputs[col] for col in ALL_FEATURES]], columns=ALL_FEATURES)


def get_risk_level(probability: float) -> str:
    """Bucket a predicted probability into a plain-language risk level."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    else:
        return "High"


def get_next_steps(risk_level: str) -> list[str]:
    """Educational, non-diagnostic suggestions based on risk level.
    This is NOT medical advice -- always shown alongside a disclaimer."""
    common = ["Maintain a balanced diet and regular exercise", "Avoid smoking and limit alcohol"]

    if risk_level == "Low":
        return common + ["Continue routine annual checkups"]
    elif risk_level == "Moderate":
        return common + ["Consider discussing these results with a physician", "Monitor blood pressure and cholesterol regularly"]
    else:
        return common + ["Consult a cardiologist promptly for a full clinical evaluation", "Do not delay seeking professional medical advice"]


def predict_patient(inputs: dict, model_name: str = "best_model") -> dict:
    """Run the full prediction pipeline for one patient: build row, scale,
    predict, and package a result dict ready for display."""
    model = load_model(model_name)
    scaler = load_model("scaler")

    patient_df = build_patient_dataframe(inputs)
    patient_scaled = pd.DataFrame(
        scaler.transform(patient_df), columns=patient_df.columns
    )

    prediction = int(model.predict(patient_scaled)[0])
    probability = float(model.predict_proba(patient_scaled)[0][1])
    risk_level = get_risk_level(probability)

    return {
        "prediction": prediction,
        "probability": round(probability, 3),
        "risk_level": risk_level,
        "next_steps": get_next_steps(risk_level),
    }
