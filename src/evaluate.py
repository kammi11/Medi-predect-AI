"""Model evaluation: metrics, confusion matrix, ROC curve, comparison table."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute the full metrics suite for one fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_curve": roc_curve(y_test, y_proba),  # (fpr, tpr, thresholds)
    }


def evaluate_all_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate every trained model. Returns {model_name: metrics_dict}."""
    return {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a summary dataframe: one row per model, one column per metric."""
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 3),
            "Precision": round(metrics["precision"], 3),
            "Recall": round(metrics["recall"], 3),
            "F1 Score": round(metrics["f1"], 3),
            "ROC-AUC": round(metrics["roc_auc"], 3),
        })
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)


def select_best_model(results: dict, metric: str = "f1") -> str:
    """Return the name of the model with the highest score on the given metric."""
    return max(results.items(), key=lambda kv: kv[1][metric])[0]


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Return feature importances for tree-based models, or coefficients for
    Logistic Regression. Returns None for models without either (e.g. SVM)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
