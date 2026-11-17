# Archived empirical findings

All categories are exploratory and conditional. Coverage, power and false-release control have not been established. Human review is mandatory; no automatic deployment authorization.

| Phase/task | Challenger | Mean log-loss change | Interval | Interpretation |
|---|---|---:|---|---|
| 1 / breast cancer | Fixed transform | 0.02169919 | [0.01320463, 0.02835873] | 95%, additional evaluation |
| 2 / breast cancer | regularized_logistic | 0.03393258 | [0.01885932, 0.04894589] | 97.5%, exploratory |
| 2 / breast cancer | calibrated_forest | 0.22830303 | [0.02869641, 0.72857768] | 97.5%, exploratory |
| 3 / banknote | tuned_logistic | -0.02926603 | [-0.03799119, -0.01877655] | 98.75%, conditional improvement |
| 3 / banknote | calibrated_forest | 0.06499755 | [0.04485821, 0.13301433] | 98.75%, conditional degradation |
| 3 / ionosphere | tuned_logistic | 0.02271787 | [-0.34210108, 0.12169791] | 98.75%, conditional inconclusive |
| 3 / ionosphere | calibrated_forest | 0.04774743 | [-0.46519099, 0.18713404] | 98.75%, conditional inconclusive |

Negative change favors challenger. The seven rows are not one controlled family. Phase 3 alone allocates alpha across four primary contrasts. Separate domains are not independent-site validation of a frozen medical model.

The proposed evidence is linked to immutable local runs by [lineage](../evidence/lineage.json). Model data and raw score arrays are copied unchanged; redacted environment fields and regenerated narrative are explicitly listed derivatives. Original archives were not rewritten. SHA256 does not authenticate a publisher.

Phase 1/2 use clipped probability loss. Phase 3 uses score-space loss and a different forest sigmoid input; these are not causal ablations of an unchanged intervention. Phase 2 forest saturation is influenced by one high-leverage case. Ionosphere has only 88 primary test cases and unstable split/tail behavior. No case was deleted or tuning performed against these outcomes in Phase 4.
