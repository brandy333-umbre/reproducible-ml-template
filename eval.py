
from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    try:
        ll = log_loss(y_true, y_proba)
    except Exception:
        ll = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": float(ll),
    }


def evaluate_model(model_bundle, data) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for split_name in ["val", "test"]:
        X = getattr(data, f"X_{split_name}")
        y = getattr(data, f"y_{split_name}")
        y_pred = model_bundle.predict(X)
        y_proba = model_bundle.predict_proba(X)
        out[split_name] = evaluate_predictions(y, y_pred, y_proba)

    return out
