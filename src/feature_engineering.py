"""Feature engineering: encoding, scaling, feature selection, train/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from src.config import TARGET_COLUMN, ALL_FEATURES, RANDOM_STATE, TEST_SIZE


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """The source dataset already encodes categoricals as integers (sex, cp,
    fbs, restecg, exang, slope, ca, thal), so no one-hot encoding is needed
    here. This function exists as a hook in case a raw/string-labeled dataset
    is used instead -- it one-hot encodes any remaining object columns.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        df = pd.get_dummies(df, columns=object_cols, drop_first=True)
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on training data only, transform both sets.

    Fitting only on train avoids leaking test-set statistics into training,
    which would give an overly optimistic (and dishonest) evaluation.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def select_features(X: pd.DataFrame, y: pd.Series, k: int = 10) -> list[str]:
    """Select the top-k features by ANOVA F-value against the target.

    Useful for showing which features matter most, and for optionally
    training on a reduced feature set to fight overfitting on a small dataset.
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()].tolist()
    return selected


def split_data(df: pd.DataFrame, feature_columns: list[str] | None = None):
    """Split into train/test sets. Stratified on target to preserve class balance."""
    if feature_columns is None:
        feature_columns = ALL_FEATURES

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
