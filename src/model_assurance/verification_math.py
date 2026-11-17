"""Independent arithmetic for semantic evidence verification (not coverage validation)."""

import numpy as np
from scipy.special import expit


def require_equal(actual, expected, context: str) -> None:
    """Reject missing/extra keys, wrong primitive types, nonfinite or changed values."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{context}: missing or unsupported fields")
        for key, value in expected.items():
            require_equal(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{context}: invalid list shape")
        for index, value in enumerate(expected):
            require_equal(actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{context}: changed value or type")
    elif isinstance(expected, (int, np.integer)):
        if type(actual) is not int or actual != expected:
            raise ValueError(f"{context}: invalid integer")
    else:
        if type(actual) not in (float, int) or not np.isfinite(actual):
            raise ValueError(f"{context}: expected finite number")
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{context}: calculated value mismatch")


def loss(y, score):
    return np.logaddexp(0, (1 - 2 * np.asarray(y)) * np.asarray(score))


def contrasts(y, scores):
    values = loss(y, scores)
    return (values[..., 1:, :] - values[..., :1, :]).mean(axis=-3)


def interval(draws, point, alpha):
    return {
        "estimate": float(point),
        "lower": float(np.quantile(draws, alpha / 2)),
        "upper": float(np.quantile(draws, 1 - alpha / 2)),
        "confidence": 1 - alpha,
        "replicates": len(draws),
        "expected_draws_per_tail": len(draws) * alpha / 2,
    }


def category(ci, n, c):
    """Independently check the historical illustrative gate, never authorize deployment."""
    if c["safety_veto"]:
        value, reason = "inconclusive", "Safety veto prevents favorable assurance"
    elif n < c["minimum_test_cases"]:
        value, reason = "inconclusive", "Too few unique test cases"
    elif ci["expected_draws_per_tail"] < c["minimum_tail_draws"]:
        value, reason = "inconclusive", "Insufficient Monte Carlo tail resolution"
    elif ci["upper"] < -c["minimum_improvement"]:
        value, reason = "improvement", "Upper bound clears minimum improvement"
    elif ci["lower"] > c["degradation_margin"]:
        value, reason = "degradation", "Lower bound exceeds degradation margin"
    elif ci["upper"] < c["noninferiority_margin"]:
        value, reason = (
            "non_inferiority",
            "Upper bound is below prespecified loss margin; not equivalence",
        )
    else:
        value, reason = "inconclusive", "Bounds do not resolve prespecified criteria"
    return {
        "category": value,
        "reason": reason,
        "automatic_deployment_allowed": False,
        "human_review_required": True,
    }


def model_metrics(y, score):
    """No calls to the experiment's metric or calibration implementation."""
    y, p = np.asarray(y), expit(score)
    z = p >= 0.5
    tn, fp = int(np.sum((y == 0) & ~z)), int(np.sum((y == 0) & z))
    fn, tp = int(np.sum((y == 1) & ~z)), int(np.sum((y == 1) & z))
    bins = []
    indices = np.minimum(np.floor(10 * p).astype(int), 9)
    for j in range(10):
        mask = indices == j
        bins.append(
            {
                "bin": j,
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()) if mask.any() else None,
                "observed_rate": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "metrics": {
            "log_loss": float(loss(y, score).mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "accuracy": float(np.mean(z == y)),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        },
        "calibration": {
            "ece_10_equal_width": float(
                sum(
                    b["count"] * abs(b["mean_probability"] - b["observed_rate"])
                    for b in bins
                    if b["count"]
                )
                / len(y)
            ),
            "mean_probability_minus_prevalence": float(p.mean() - y.mean()),
            "bins": bins,
        },
    }


def changes(y, baseline, score):
    a, b = expit(baseline), expit(score)
    return {
        "mean_absolute_score_change": float(np.abs(score - baseline).mean()),
        "signed_margin_change": float(((2 * y - 1) * (score - baseline)).mean()),
        "mean_absolute_probability_change": float(np.abs(b - a).mean()),
        "disagreement_rate": float(np.mean((a >= 0.5) != (b >= 0.5))),
        "accuracy_difference": float(np.mean((b >= 0.5) == y) - np.mean((a >= 0.5) == y)),
    }
