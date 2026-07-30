"""Central configuration: paths, column names, and constants used across the project."""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "heart.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "heart_clean.csv"
MODELS_DIR = BASE_DIR / "models"

# --- Target column ---
TARGET_COLUMN = "target"

# --- Feature groups ---
NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Human-readable labels (for UI dropdowns) ---
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels (0-3)",
    "thal": "Thalassemia",
}

# --- Model training constants ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
