# Phase 2 protocol: exploratory scientific robustness

This extension is independently developed internal validation, not a third-party
audit or external replication. The breast-cancer data were already examined in
Phase 1; a newly frozen partition does not make them fresh confirmatory data.

Freeze `configs/phase2.yaml` before examining Phase 2 outputs. A stratified 30%
evaluation holdout is fixed with seed 2026. The remaining development subjects
are partitioned into 75% training and 25% calibration with split seeds 11, 22,
33. Cross these with training seeds 101, 202, 303. All preprocessing is trained
on training rows only. Baseline logistic regression uses C=1; a separately
fitted challenger uses C=0.1. A seeded 100-tree forest uses minimum leaf size 3;
a regularized sigmoid on its clipped raw probability log odds is fitted on
calibration rows only (C=1). No hyperparameter search uses evaluation outcomes.
Class 1 is the dataset's benign class; clinical utility is not inferred.

The primary estimand for each of the two challengers is mean paired held-out
log loss, averaged equally across the nine specified fits **within subject**.
Conditional on the fitted model collection, resample subjects jointly using
5,000 paired percentile-bootstrap replicates. With two primary comparisons,
each interval has confidence 1-(1-family_confidence)/2. The union bound motivates
Bonferroni family coverage only if individual intervals have their stated
coverage; percentile-bootstrap finite-sample coverage is not guaranteed.
Subjects are assumed independent and representative. This estimand is not the
loss of an ensemble's averaged probabilities, nor unconditional training risk.

Overlapping development splits are dependent. Report across-split SD of seed
means and mean within-split SD across seeds descriptively; do not attach a run
t-test, pool all subject/run rows, or interpret these SDs as additive variance
components. Logistic training is deterministic here; its seed SD may be zero.
Random forest seed variation represents algorithmic randomness, not independent
data collection. Nine fits are a reproducible pilot budget, not a power claim.

Report log-odds and signed-margin shifts descriptively, acknowledging distinct
cross-family score parameterizations. Probability endpoints include log loss,
Brier, reliability-bin counts and per-case changes. Decision endpoints use a
fixed 0.5 threshold, disagreement, paired correctness and exact McNemar.
Secondary endpoints and McNemar p-values are exploratory without confirmatory
significance declarations. Calibration bins may be sparse; sigmoid calibration
has no guaranteed improvement. Report reliability of mean predictions as an
ensemble diagnostic separately from per-fit diagnostics.

Results are computed from saved predictions and metrics. Preserve Phase 1 files
and reject existing Phase 2 output directories. Human review is mandatory; do
not approve release using reused data. Next-stage hypotheses concern external
replication, subgroup/shift harm, training-population uncertainty and coverage
simulation. They are untested, not theorems or novelty claims.

Implementation references: [SciPy paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
and [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
