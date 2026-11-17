"""Normal-approximation planning for a *paired*, independent-case mean difference."""

from math import ceil, isfinite

from scipy.stats import norm


def paired_required_cases(sd, effect, alpha=0.05, power=0.8):
    if (
        not all(isfinite(x) for x in (sd, effect, alpha, power))
        or sd <= 0
        or effect <= 0
        or not 0 < alpha < 1
        or not 0.5 < power < 1
    ):
        raise ValueError("Finite positive SD/effect, alpha in (0,1), planning power in (0.5,1)")
    n = ((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / effect) ** 2
    return ceil(n)
