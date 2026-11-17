> Historical Phase 1 scope. For current setup and Phase 2/3 inference, see the phase-specific protocols and reproducibility guide.

# Methodology
Primary contrast: per-observation challenger minus baseline log loss; negative values mean improvement. The paired nonparametric case-bootstrap percentile interval is conditional on the fitted estimator and evaluation sample; it assumes subjects are independent. McNemar's exact test is an exploratory paired classification diagnostic. The confidence bound is compared to a prespecified minimum improvement; additional evaluation is recommended when evidence is ambiguous. The same held-out set is never used for fitting or selecting temperature. Further studies should use independent datasets and repeated training seeds.
