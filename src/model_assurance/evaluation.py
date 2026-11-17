"""Binary-classification metrics and per-case proper scoring rules."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_losses(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if (
        y.shape != p.shape
        or y.ndim != 1
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Invalid binary labels/probabilities")
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(q) + (1 - y) * np.log1p(-q)), (p - y) ** 2


def evaluate(y, p):
    logloss, brier = binary_losses(y, p)
    labels = (np.asarray(p) >= 0.5).astype(int)
    return {
        "log_loss": float(logloss.mean()),
        "brier": float(brier.mean()),
        "accuracy": float(accuracy_score(y, labels)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, labels, labels=[0, 1]).tolist(),
    }
