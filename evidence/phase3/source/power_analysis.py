"""Normal-approximation planning for a *paired*, independent-case mean difference."""
from math import ceil

from scipy.stats import norm


def paired_required_cases(sd, effect, alpha=0.05, power=0.8):
    if sd <= 0 or effect <= 0 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("SD and effect positive; alpha and power between zero and one")
    n = ((norm.ppf(1-alpha/2)+norm.ppf(power))*sd/effect)**2
    return ceil(n)
