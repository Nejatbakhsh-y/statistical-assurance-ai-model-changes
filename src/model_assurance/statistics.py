"""Subject-level paired inference; does not include training-run uncertainty."""

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap(diff, *, replicates=2000, confidence=0.95, seed=42):
    d = np.asarray(diff, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Need at least two finite paired differences")
    if not (0 < confidence < 1) or replicates < 100:
        raise ValueError("Invalid confidence or replication budget")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for i in range(replicates):
        samples[i] = d[rng.integers(0, len(d), size=len(d))].mean()
    return {
        "estimate": float(d.mean()),
        "lower": float(np.quantile(samples, (1 - confidence) / 2)),
        "upper": float(np.quantile(samples, 1 - (1 - confidence) / 2)),
        "confidence": confidence,
        "replicates": replicates,
    }


def exact_mcnemar(y, baseline, challenger):
    y, a, b = map(np.asarray, (y, baseline, challenger))
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("Paired observations must have equal shape")
    x = int(np.sum((a == y) & (b != y)))
    z = int(np.sum((a != y) & (b == y)))
    return {
        "baseline_only_correct": x,
        "challenger_only_correct": z,
        "p_value_two_sided": float(binomtest(min(x, z), x + z, 0.5).pvalue) if x + z else 1.0,
    }
