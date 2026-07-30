"""Train and compare Logistic Regression, Decision Tree, Random Forest, and SVM."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.config import RANDOM_STATE


def get_model_definitions() -> dict:
    """Return a fresh dict of model name -> untrained estimator.

    probability=True on SVC is required so predict_proba() works later for
    the risk-probability display on the Prediction page.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    }


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Fit all four models and return them in a dict keyed by name."""
    models = get_model_definitions()
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained
