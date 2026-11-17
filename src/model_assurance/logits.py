"""Scores and binary margins are comparable only in a shared parameterization."""

import numpy as np
from scipy.special import expit, softmax


def scale_binary_score(score, factor):
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all() or factor <= 0 or not np.isfinite(factor):
        raise ValueError("Scores must be finite and factor strictly positive")
    return score * factor


def probability_from_binary_score(score):
    return expit(np.asarray(score, dtype=float))


def center_multiclass_logits(logits):
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Expected finite two-dimensional logits")
    return z - z.mean(axis=1, keepdims=True)


def softmax_probs(logits):
    return softmax(np.asarray(logits, dtype=float), axis=1)
