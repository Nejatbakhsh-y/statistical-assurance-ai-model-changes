# Exploratory Phase 3 evidence

Conditional research classifications only. Coverage, power and false-release control have not been established. Human review is mandatory; automatic deployment authorization is false.

| Dataset | Challenger | Mean log-loss change | Joint interval | Assurance |
|---|---|---:|---|---|
| banknote | tuned_logistic | -0.02926603 | [-0.03799119, -0.01877655] | improvement |
| banknote | calibrated_forest | 0.06499755 | [0.04485821, 0.13301433] | degradation |
| ionosphere | tuned_logistic | 0.02271787 | [-0.34210108, 0.12169791] | inconclusive |
| ionosphere | calibrated_forest | 0.04774743 | [-0.46519099, 0.18713404] | inconclusive |

See the repository scientific limitations and dataset-specific reports. These are external tasks, not frozen-model site validation. Source snapshots are provenance evidence, not the current verifier implementation. A detached semantic-verification record does not certify coverage.
