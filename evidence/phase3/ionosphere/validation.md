# Phase 3: Ionosphere

Sigillito, V., Wing, S., Hutton, L., and Baker, K. (1989). Ionosphere. UCI Machine Learning Repository. DOI: 10.24432/C5W01B. Source: https://archive.ics.uci.edu/dataset/52/ionosphere. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Snapshot: Content-addressed UCI snapshot retrieved 2026-11-17; no upstream release number advertised; SHA256 `46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd`. Integrity verifies pinned bytes, not a publisher signature. Positive class: Good radar return (g).

Cleaning: {"raw_rows": 351, "unique_rows": 350, "removed_duplicate_raw_ids": [248]}. Partition counts: {"train": 156, "validation": 53, "calibration": 53, "test": 88}. Counts refer to unique feature rows; raw source IDs remain in metrics and NPZ.

## Primary joint uncertainty and assurance

Negative challenger-minus-baseline log loss favors the challenger.

- tuned_logistic: effect 0.02271787; 98.750% interval [-0.34210108, 0.12169791]; assurance **inconclusive** (Bounds do not resolve prespecified criteria). Expected bootstrap draws per tail: 12.50. Descriptive split effects: [0.022717867321160448, -0.09993809870070632, -0.15524542665394378].

  Primary-split descriptive mean ECE: 0.09746274; paired Brier change: 0.00014047; level diagnostics: {"accuracy_difference": -0.011363636363636354, "disagreement_rate": 0.03409090909090909, "mean_absolute_probability_change": 0.08849151647992899, "mean_absolute_score_change": 1.810708351183803, "signed_margin_change": -1.6253318126215992}. Per-fit baseline/challenger calibration bins, proper scores and confusion matrices are in metrics.json.

  evaluation_only_ci: [-0.05693430, 0.08955826] (diagnostic decomposition; no extra assurance decisions).

  training_validation_calibration_only_ci: [-0.23644582, 0.07684382] (diagnostic decomposition; no extra assurance decisions).

- calibrated_forest: effect 0.04774743; 98.750% interval [-0.46519099, 0.18713404]; assurance **inconclusive** (Bounds do not resolve prespecified criteria). Expected bootstrap draws per tail: 12.50. Descriptive split effects: [0.047747430199992634, -0.10214007274731561, -0.23824710129818794].

  Primary-split descriptive mean ECE: 0.19842095; paired Brier change: -0.00185210; level diagnostics: {"accuracy_difference": 0.0511363636363637, "disagreement_rate": 0.08522727272727273, "mean_absolute_probability_change": 0.1897892400684346, "mean_absolute_score_change": 2.6575026746952224, "signed_margin_change": -2.1757337052562016}. Per-fit baseline/challenger calibration bins, proper scores and confusion matrices are in metrics.json.

  evaluation_only_ci: [-0.08043915, 0.16020034] (diagnostic decomposition; no extra assurance decisions).

  training_validation_calibration_only_ci: [-0.31880752, 0.13684246] (diagnostic decomposition; no extra assurance decisions).

## Dataset-specific independence limitation

Single Goose Bay radar system; session/time grouping unavailable, so independence of returns is unverified

## Assumptions, limitations and research recommendations

These datasets are external tasks, independently collected from the Phase 1/2 medical dataset and each other; this validates a method across domains, not transfer of one frozen breast-cancer model. No third-party audit, deployment approval, regulatory validation or general superiority is established.

Primary inference uses only the first prespecified split. Other whole-data splits overlap and are descriptive sensitivity analyses, never pooled as independent runs. Bootstrap replicates resample original training, validation, calibration and test rows within disjoint roles and refit the complete selection/calibration pipeline. Repeated predictions are never counted as independent subjects. Training-only intervals fix test rows; evaluation-only intervals fix fitted models; joint intervals vary both. These are not additive variance components. The two fixed training seeds define a finite algorithmic average; uncertainty over arbitrary new seeds is not fully integrated.

Resampling is stratified: class counts in every role are held fixed. Intervals condition on observed class mixtures and role-specific empirical source pools. They do not establish new-site, temporal, cluster, or unseen-prevalence performance. Percentile-bootstrap coverage is approximate; validation selection is nonsmooth, small calibration samples can be unstable, and Monte Carlo tail counts are recorded. Independence of unique rows and representativeness are assumptions, not verified facts. Missing source grouping IDs preclude cluster validation. Removing exact feature duplicates cannot establish absence of hidden dependence.

Four primary joint log-loss intervals (two datasets by two challengers) use Bonferroni alpha allocation. Family coverage follows only if the component intervals have their nominal coverage, which is not guaranteed here. Secondary Brier, ECE, reliability, margin and decision diagnostics are descriptive; no uncorrected secondary significance claims are made. A non-inferiority category is margin-specific and is not equivalence. Illustrative loss margins are not domain-certified utility thresholds. All categories require human review and block automatic deployment.

Score shifts are descriptive: logistic native log odds and sigmoid-calibrated forest log odds share class orientation but not an intrinsic cross-family scale. Primary log loss is calculated stably from scores using logaddexp, without Phase 1/2 probability clipping caps; it is therefore not a bitwise-identical numerical estimand at saturated probabilities. Calibration diagnostics use ten fixed equal-width bins; ECE depends on bin choice and sparse counts. Forest sigmoid calibration uses raw bounded forest probability, selected before these datasets' test results, rather than Phase 2's extreme raw log odds. No guaranteed calibration improvement is claimed.

Next research: acquire specimen/session group identifiers, independent sites/cohorts and temporal tests; validate practical margins; simulate selection-aware interval coverage; check bootstrap-tail stability; prespecify subgroup and shift safety endpoints and their multiplicity; replicate with independently sampled training populations. Do not retune against reported test outcomes. Public benchmark familiarity limits any claim of a prospectively untouched population. No new theorem or publication-level novelty is asserted.

## Reproducibility

metrics.json contains configuration, provenance, exact partitions, fit metrics and selections. predictions.npz contains observed and refit scores, sampled role IDs, target labels and bootstrap effects. Parent environment.json and command logs record actual execution. No existing output is overwritten. Human review is mandatory; automatic_deployment_allowed is always false.
