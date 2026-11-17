"""Phase 3 paired uncertainty and nonbinding, prespecified assurance categories."""

import numpy as np

from .calibration import reliability_bins
from .evaluation import binary_losses


def stratified_resample(ids: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve class counts; inference conditions on the empirical class mixture."""
    ids, y = np.asarray(ids), np.asarray(y)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) < 2
            or np.any(ids < 0) or np.any(ids >= len(y)) or set(y[ids]) != {0, 1}):
        raise ValueError("Need valid indices containing both binary classes")
    return np.concatenate([rng.choice(ids[y[ids] == c], size=np.sum(y[ids] == c), replace=True)
                           for c in (0, 1)])


def percentile_interval(draws: np.ndarray, estimate: float, alpha: float) -> dict:
    """No finite-sample or unconditional bootstrap-coverage guarantee is asserted."""
    d = np.asarray(draws, dtype=float)
    if d.ndim != 1 or len(d) < 100 or not np.isfinite(d).all():
        raise ValueError("Need at least 100 finite bootstrap effects")
    if not np.isfinite(estimate) or not 0 < alpha < 1:
        raise ValueError("Invalid estimate or alpha")
    return {"estimate": float(estimate), "lower": float(np.quantile(d, alpha/2)),
            "upper": float(np.quantile(d, 1-alpha/2)), "confidence": 1-alpha,
            "replicates": len(d), "expected_draws_per_tail": len(d)*alpha/2}


def assurance_category(ci: dict, n_test: int, config: dict) -> dict:
    """Classify log-loss bounds; never produce a deployment permission."""
    lower, upper = ci["lower"], ci["upper"]
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Invalid uncertainty bounds")
    if config["safety_veto"]:
        status, reason = "inconclusive", "Safety veto prevents favorable assurance"
    elif n_test < config["minimum_test_cases"]:
        status, reason = "inconclusive", "Too few unique test cases"
    elif ci["expected_draws_per_tail"] < config["minimum_tail_draws"]:
        status, reason = "inconclusive", "Insufficient Monte Carlo tail resolution"
    elif upper < -config["minimum_improvement"]:
        status, reason = "improvement", "Upper bound clears minimum improvement"
    elif lower > config["degradation_margin"]:
        status, reason = "degradation", "Lower bound exceeds degradation margin"
    elif upper < config["noninferiority_margin"]:
        status, reason = "non_inferiority", "Upper bound is below prespecified loss margin; not equivalence"
    else:
        status, reason = "inconclusive", "Bounds do not resolve prespecified criteria"
    return {"category": status, "reason": reason, "automatic_deployment_allowed": False,
            "human_review_required": True}


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Equal-width ECE and calibration-in-the-large are descriptive, not tests."""
    binary_losses(y, p)
    if len(y) == 0:
        raise ValueError("Calibration metrics require nonempty observations")
    bins = reliability_bins(y, p)
    ece = sum(r["count"] / len(y) * abs(r["mean_probability"]-r["observed_rate"])
              for r in bins if r["count"])
    return {"ece_10_equal_width": float(ece),
            "mean_probability_minus_prevalence": float(np.mean(p)-np.mean(y)),
            "bins": bins}
