# Phase 2 independent-development validation report



This is an internal validation analysis, not an independent third-party audit.

Dataset reuse makes all findings exploratory. Mandatory human review applies.



Evaluation subjects: 171; paired comparison records: 18.

Python: 3.11.9; packages: {"PyYAML": "6.0.3", "matplotlib": "3.11.2", "numpy": "2.4.6", "scikit-learn": "1.9.1", "scipy": "1.17.1"}.



## Primary effects



Negative challenger-minus-baseline log loss favors challenger. Each interval is 97.5% at default configuration, targeting approximate 95% family coverage for two primary contrasts.



- regularized_logistic: mean 0.03393258; interval [0.01885932, 0.04894589]; mean Brier difference 0.00615702; mean accuracy difference 0.00000000; split SD 0.00337969; within-split seed SD 0.00000000.

  Descriptive level diagnostics: mean absolute log-odds change 2.85359180; mean absolute probability change 0.04119890; mean decision disagreement 0.01559454. Largest subject-mean paired loss difference 0.41412406 at dataset subject ID 414; saturated challenger probabilities across fits/subjects: 0.

- calibrated_forest: mean 0.22830303; interval [0.02869641, 0.72857768]; mean Brier difference 0.02083271; mean accuracy difference -0.02729045; split SD 0.03433976; within-split seed SD 0.03049940.

  Descriptive level diagnostics: mean absolute log-odds change 13.15100208; mean absolute probability change 0.06372551; mean decision disagreement 0.04678363. Largest subject-mean paired loss difference 28.79122290 at dataset subject ID 135; saturated challenger probabilities across fits/subjects: 302.



## Interpretation and limitations



Intervals resample independent evaluation subjects after averaging loss differences across fits. They describe expected loss over this fixed finite fit collection, not the loss of averaged probabilities. No run-row pooling or independent-run t-test is used. Overlapping development splits make seed/split SDs descriptive, not variance components or confidence intervals. Fits are held fixed: training-population uncertainty is excluded.

Logistic scores and calibrated forest log odds share positive-class orientation, but raw score magnitudes across model families have no intrinsic equivalence. Margin shifts are descriptive. Per-fit proper scores, reliability-bin counts, decision metrics and exploratory McNemar diagnostics are in metrics.json; unadjusted McNemar p-values support no confirmatory claims. Mean-prediction reliability describes a finite ensemble and differs from average single-fit calibration.

Calibration uses disjoint development rows; no evaluation fitting or temperature/hyperparameter selection occurs. Sigmoid calibration is not guaranteed to improve calibration. Reliability bins can be sparse. No subgroup, temporal-shift, external-dataset or safety audit has been established.

Numerical convention: proper-score evaluation clips probabilities to [1e-15, 1-1e-15], matching Phase 1; forest probability-derived log odds use the same clipping before sigmoid calibration. Saturated errors can dominate loss and widen bootstrap intervals. This is evidence to investigate tail risk and numerical sensitivity on future data, not permission to retune on this holdout. Saturation counts include both correct and incorrect extremes; largest-subject effects are post-hoc diagnostics.

The mathematical identities and disjoint-index checks are established properties; bootstrap coverage is an approximation under independent subjects, not a finite-sample guarantee. General superiority, false-release-rate control, and publication-level novelty remain untested hypotheses.

Recommendation: freeze candidate choices before fresh external data, replicate datasets and independently sampled training cohorts, examine subgroup/shift risk, budget bootstrap tails and subject counts using defensible pilot effects, and simulate interval coverage before any confirmatory release gate. Do not tune against these reported evaluation results.



## Reproducibility



Configuration, dataset hash, package versions, exact subject partitions and all seeds are in metrics.json. Per-case paired predictions are in predictions.csv. Existing output directories are never overwritten. Phase 1 outputs are untouched.


## Executed verification

19 pytest tests passed; Ruff and pip check passed. All 3,078 CSV comparison rows, per-fit metrics and primary bootstrap intervals were independently recalculated from saved predictions. Original Phase 1 and archive SHA256 manifests were checked unchanged. Logs and dependency snapshot: reports/phase2-20261117-075627-011. See verification.json.
