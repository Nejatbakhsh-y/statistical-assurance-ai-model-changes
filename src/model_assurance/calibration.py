"""Descriptive calibration summaries; no calibrator is fitted on test cases."""

import numpy as np


def reliability_bins(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities or length")
    indexes = np.minimum((p * bins).astype(int), bins - 1)
    return [
        {
            "bin": j,
            "count": int(sum(indexes == j)),
            "mean_probability": float(p[indexes == j].mean()) if np.any(indexes == j) else None,
            "observed_rate": float(y[indexes == j].mean()) if np.any(indexes == j) else None,
        }
        for j in range(bins)
    ]
