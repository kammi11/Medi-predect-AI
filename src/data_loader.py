"""Load, save, and summarize the heart disease dataset. No UI code here."""

import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    return pd.read_csv(path)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load a dataset from a Streamlit UploadedFile object."""
    return pd.read_csv(uploaded_file)


def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    """Save a cleaned/processed dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return key stats about the dataset for display on the Dataset page."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
