# Scientific capabilities, evidence and limitations

## Implemented established methods

Paired proper-score differences, percentile bootstrap intervals, descriptive
reliability/ECE, McNemar diagnostics and Bonferroni allocation are established
methods. The engineering contribution is a reproducible comparison pipeline;
this integration is not a new theorem or independently certified audit.

Phase 1/2 estimate paired mean differences conditional on fixed fitted models.
Phase 2 averages seed-specific losses within subjects, not probabilities or
independent repeated predictions. Phase 3 resamples role-specific empirical
training/validation/calibration pools and paired test cases with class counts
fixed. Its finite-seed average and empirical refit quantiles must not be confused
with established coverage for a population learning-procedure risk. Frozen-model
risk and expected risk over new training samples are different targets.

## Empirical evidence

The [summary](validation-summary.md) records actual archived findings. These are
pilot benchmark results. Later overlapping splits are descriptive, not independent
replicated subjects or sites. Different source domains do not prove independent
rows. No model was tuned against these reported test outcomes in Phase 4.

## Theoretical limits

**Confidence-interval coverage, statistical power and false-release control have
not been established.** Bonferroni needs valid component coverage. Bootstrap can
be biased after selection or with small samples/rare losses. More resamples improve
numerical resolution, not the validity of assumptions. Case minima and practical
margins are illustrative, not adequate-power or domain-utility guarantees.

Banknote lacks specimen IDs and Ionosphere lacks session grouping. Exact duplicate
removal cannot establish independence. No cluster, temporal, site, fairness,
adversarial, subgroup or shift assurance is established. Non-inferiority is
margin-specific, not equivalence. ECE depends on binning and is descriptive.

Cross-family logit magnitudes have no intrinsic equivalence. Positive scaling
preserves mathematical signs but a rounded probability predicate may differ near
zero. Phase 1/2 clip probabilities at 1e-15; Phase 3 uses score-space loss.
Phase 2 forest saturation/high-leverage cases were not removed or retuned away.
Phase 3 changed its forest sigmoid input before its test outcomes; cross-phase
results are not an ablation of one unchanged intervention. Algorithm/calibration
causal attribution would need budget-matched ablations.

## Untested research and human review

Future work needs timestamped protocols, utility-justified margins, selection-aware
coverage/power simulations, group IDs, new sites/cohorts, shift/subgroup checks
with multiplicity, and systematic related work. Existing pilots must not be
retrospectively labeled preregistered or publication-level novelty.

Every classification is exploratory and conditional. Human review is mandatory;
automatic deployment authorization is false. No deployment mechanism exists.
