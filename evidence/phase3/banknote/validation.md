# Phase 3: Banknote Authentication

Lohweg, V. (2012). Banknote Authentication. UCI Machine Learning Repository. DOI: 10.24432/C55P57. Source: https://archive.ics.uci.edu/dataset/267/banknote+authentication. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Snapshot: Content-addressed UCI snapshot retrieved 2026-11-17; no upstream release number advertised; SHA256 `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9`. Integrity verifies pinned bytes, not a publisher signature. Positive class: Original UCI class 1; semantic label not inferred.

Cleaning: {"raw_rows": 1372, "unique_rows": 1348, "removed_duplicate_raw_ids": [47, 139, 190, 195, 226, 268, 284, 300, 315, 320, 345, 351, 352, 404, 427, 436, 476, 498, 604, 615, 657, 691, 716, 727]}. Partition counts: {"train": 606, "validation": 202, "calibration": 203, "test": 337}. Counts refer to unique feature rows; raw source IDs remain in metrics and NPZ.

## Primary joint uncertainty and assurance

Negative challenger-minus-baseline log loss favors the challenger.

- tuned_logistic: effect -0.02926603; 98.750% interval [-0.03799119, -0.01877655]; assurance **improvement** (Upper bound clears minimum improvement). Expected bootstrap draws per tail: 12.50. Descriptive split effects: [-0.029266026200445907, -0.02876426690848286, -0.02296245045505339].

  Primary-split descriptive mean ECE: 0.01402689; paired Brier change: -0.00451429; level diagnostics: {"accuracy_difference": 0.0029673590504450953, "disagreement_rate": 0.002967359050445104, "mean_absolute_probability_change": 0.02693789320285246, "mean_absolute_score_change": 2.856681780014199, "signed_margin_change": 2.854806888570251}. Per-fit baseline/challenger calibration bins, proper scores and confusion matrices are in metrics.json.

  evaluation_only_ci: [-0.03746350, -0.02212716] (diagnostic decomposition; no extra assurance decisions).

  training_validation_calibration_only_ci: [-0.03271450, -0.02133391] (diagnostic decomposition; no extra assurance decisions).

- calibrated_forest: effect 0.06499755; 98.750% interval [0.04485821, 0.13301433]; assurance **degradation** (Lower bound exceeds degradation margin). Expected bootstrap draws per tail: 12.50. Descriptive split effects: [0.06499755322412984, 0.05716586620056542, 0.05604298375561673].

  Primary-split descriptive mean ECE: 0.06667527; paired Brier change: 0.01217879; level diagnostics: {"accuracy_difference": -0.011869436201780492, "disagreement_rate": 0.02967359050445104, "mean_absolute_probability_change": 0.06971898288421416, "mean_absolute_score_change": 1.8184829092954864, "signed_margin_change": -1.6790098599843415}. Per-fit baseline/challenger calibration bins, proper scores and confusion matrices are in metrics.json.

  evaluation_only_ci: [0.03979950, 0.09490302] (diagnostic decomposition; no extra assurance decisions).

  training_validation_calibration_only_ci: [0.05701620, 0.11752947] (diagnostic decomposition; no extra assurance decisions).

## Dataset-specific independence limitation

No original banknote specimen IDs supplied; image/specimen clustering cannot be excluded

## Assumptions, limitations and research recommendations

These datasets are external tasks, independently collected from the Phase 1/2 medical dataset and each other; this validates a method across domains, not transfer of one frozen breast-cancer model. No third-party audit, deployment approval, regulatory validation or general superiority is established.

Primary inference uses only the first prespecified split. Other whole-data splits overlap and are descriptive sensitivity analyses, never pooled as independent runs. Bootstrap replicates resample original training, validation, calibration and test rows within disjoint roles and refit the complete selection/calibration pipeline. Repeated predictions are never counted as independent subjects. Training-only intervals fix test rows; evaluation-only intervals fix fitted models; joint intervals vary both. These are not additive variance components. The two fixed training seeds define a finite algorithmic average; uncertainty over arbitrary new seeds is not fully integrated.

Resampling is stratified: class counts in every role are held fixed. Intervals condition on observed class mixtures and role-specific empirical source pools. They do not establish new-site, temporal, cluster, or unseen-prevalence performance. Percentile-bootstrap coverage is approximate; validation selection is nonsmooth, small calibration samples can be unstable, and Monte Carlo tail counts are recorded. Independence of unique rows and representativeness are assumptions, not verified facts. Missing source grouping IDs preclude cluster validation. Removing exact feature duplicates cannot establish absence of hidden dependence.

Four primary joint log-loss intervals (two datasets by two challengers) use Bonferroni alpha allocation. Family coverage follows only if the component intervals have their nominal coverage, which is not guaranteed here. Secondary Brier, ECE, reliability, margin and decision diagnostics are descriptive; no uncorrected secondary significance claims are made. A non-inferiority category is margin-specific and is not equivalence. Illustrative loss margins are not domain-certified utility thresholds. All categories require human review and block automatic deployment.

Score shifts are descriptive: logistic native log odds and sigmoid-calibrated forest log odds share class orientation but not an intrinsic cross-family scale. Primary log loss is calculated stably from scores using logaddexp, without Phase 1/2 probability clipping caps; it is therefore not a bitwise-identical numerical estimand at saturated probabilities. Calibration diagnostics use ten fixed equal-width bins; ECE depends on bin choice and sparse counts. Forest sigmoid calibration uses raw bounded forest probability, selected before these datasets' test results, rather than Phase 2's extreme raw log odds. No guaranteed calibration improvement is claimed.

Next research: acquire specimen/session group identifiers, independent sites/cohorts and temporal tests; validate practical margins; simulate selection-aware interval coverage; check bootstrap-tail stability; prespecify subgroup and shift safety endpoints and their multiplicity; replicate with independently sampled training populations. Do not retune against reported test outcomes. Public benchmark familiarity limits any claim of a prospectively untouched population. No new theorem or publication-level novelty is asserted.

## Reproducibility

metrics.json contains configuration, provenance, exact partitions, fit metrics and selections. predictions.npz contains observed and refit scores, sampled role IDs, target labels and bootstrap effects. Parent environment.json and command logs record actual execution. No existing output is overwritten. Human review is mandatory; automatic_deployment_allowed is always false.
