"""Data cleaning: missing values, duplicates, dtype fixes, and IQR outlier removal."""

import pandas as pd

from src.config import NUMERIC_FEATURES


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> tuple[pd.DataFrame, dict]:
    """Fill missing values. Numeric columns use median/mean, categorical use mode.

    Returns the cleaned dataframe plus a report of how many values were filled
    per column, so the UI can show a clear before/after summary.
    """
    df = df.copy()
    report = {}

    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            continue

        if col in NUMERIC_FEATURES:
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
        else:
            fill_value = df[col].mode().iloc[0]

        df[col] = df[col].fillna(fill_value)
        report[col] = {"filled": n_missing, "fill_value": round(float(fill_value), 2)}

    return df, report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows. Returns cleaned df and count removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are numeric and categorical columns are integer-coded.

    Handles cases where values got read in as strings (e.g. ' 1' with a stray space,
    or numbers stored as objects after a manual CSV edit).
    """
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Remove rows with outliers in numeric columns using the 1.5*IQR rule.

    For each column: Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1.
    A value is an outlier if it falls below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    Only applied to true numeric features (age, blood pressure, cholesterol, etc.)
    -- not to categorical columns like sex or chest pain type, where "outliers"
    are just valid category codes.
    """
    if columns is None:
        columns = NUMERIC_FEATURES

    df = df.copy()
    report = {}
    before = len(df)

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        report[col] = {"lower_bound": round(lower, 1), "upper_bound": round(upper, 1), "outliers": outlier_count}

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    df = df.reset_index(drop=True)
    report["total_rows_removed"] = before - len(df)
    return df, report


def clean_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline in order and collect a combined report."""
    df = fix_data_types(df)
    df, missing_report = handle_missing_values(df)
    df, duplicates_removed = remove_duplicates(df)
    df, outlier_report = remove_outliers_iqr(df)

    full_report = {
        "missing_values": missing_report,
        "duplicates_removed": duplicates_removed,
        "outliers": outlier_report,
    }
    return df, full_report
