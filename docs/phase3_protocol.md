> Historical Phase 3 execution protocol. Phase 4 read-only verification and safe entry points supersede its execution instructions; the archived experiment is unchanged.

# Phase 3: external tasks and paired refit assurance

This protocol extends the existing method to two independently collected public
tasks; it does not test transfer of the original breast-cancer model to new
feature spaces. Banknote wavelet features and Goose Bay radar returns have
different origins from each other and Phase 1/2. Public benchmark familiarity
precludes a claim of prospectively untouched populations or third-party audit.

## Provenance and integrity

- [UCI Banknote Authentication, ID 267](https://archive.ics.uci.edu/dataset/267/banknote+authentication):
  Lohweg (2012), DOI 10.24432/C55P57; 1,372 source rows, four features.
- [UCI Ionosphere, ID 52](https://archive.ics.uci.edu/dataset/52/ionosphere):
  Sigillito, Wing, Hutton and Baker (1989), DOI 10.24432/C5W01B;
  351 source rows, 34 features.

UCI lists both under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
These dataset licenses do not grant a license to this research repository.
`configs/datasets-phase3.json` pins official HTTPS download URLs, raw-byte
length, SHA256, schema, target mapping, citations and license URLs. No upstream
release number is advertised; the version is explicitly a content-addressed
snapshot retrieved 2026-11-17. Initial hashes were calculated from official
HTTPS downloads before fitting; they are not independently published checksums
or cryptographic proof of source authenticity. Cache mismatches fail closed.

Feature-identical duplicates are removed deterministically before splitting,
retaining first-occurrence raw IDs. Conflicting labels fail rather than being
silently resolved. This changes the target to unique-feature records. Numeric
parsing, schema validation and duplicate detection are mechanical checks, not
learned preprocessing. No specimen/session IDs are available: radar session
dependence and repeated banknote specimens cannot be excluded. Row independence
is an explicit unverified assumption; exact deduplication does not prove it.

## Prespecified modeling and partitions

Freeze `configs/phase3.yaml` before observing test outcomes. Three split seeds
(1701, 2903, 4307) each define 45% training, 15% validation, 15% calibration and
25% test, with stratification and rounding. Use seeds 101 and 202 for fitting.
Only the first split supplies primary inference. Later, overlapping splits are
descriptive sensitivity analyses, never independent replicated subjects.

Median imputation is trained only on training rows. Logistic preprocessing also
standardizes using training statistics. Baseline logistic C=1 is fixed. The
logistic challenger selects C from [0.1, 1, 10] using validation log loss only;
ties choose the first listed candidate. Models are not refitted on validation
or calibration rows. The forest uses 50 trees and minimum leaf size 3, with a
regularized sigmoid (C=1) fitted solely on calibration rows. Its sigmoid input
is raw forest probability, bounded in [0,1], rather than Phase 2's potentially
extreme raw log odds. This change is fixed before new test results. It has no
guaranteed calibration benefit. The logistic challenger does not fit a
calibrator; the calibration role remains excluded from its fitting and selection.
No decision threshold optimization occurs: threshold 0.5 is fixed.

Primary log loss is computed directly from finite scores as
`logaddexp(0, s) - y*s`. Unlike Phase 1/2 probability clipping, extreme errors
are not capped by a 1e-15 floor. Brier, ECE, reliability bins, signed margins,
probability changes, confusion matrices, accuracy, precision, recall and F1 are
secondary descriptive diagnostics. No secondary significance declarations are
made. Native logistic scores and calibrated forest log odds share positive
class orientation but no guaranteed cross-family score equivalence.

## Training and evaluation uncertainty

Let d[j,i] be challenger-minus-baseline loss for training seed j and test
subject i, and let d[i] average losses across the two fixed seeds. The point
estimate is the mean of d[i] on the primary test subjects, not loss of averaged
probabilities. Fixed seed averages do not integrate all future seed variability.

For each of 2,000 refit bootstrap replicates, resample rows **within each** of
the four disjoint roles, preserving role class counts. Refit preprocessing,
validation selection, base models and sigmoid calibrator. Predict all original
test subjects for both models jointly, then evaluate on the resampled test IDs.
One paired effect per dataset/challenger is produced by each whole-procedure
bootstrap draw. Models share the same training draw and test draw. Repeated
predictions are never treated as independent data points, and prefit run rows
are never bootstrapped as though independently collected training populations.
Four worker processes execute deterministic per-replicate RNG streams with one
BLAS thread per worker; task scheduling does not define a scientific replicate.

The same refit predictions evaluated on the unresampled test set give a
training/validation/calibration-only sensitivity interval. Separately, 10,000
paired subject draws with original fits fixed give an evaluation-only interval.
These are diagnostic decompositions, not additive variance components or extra
confirmatory tests. Original split effects and per-seed metrics describe
additional sensitivity without pooling overlaps into standard errors.

The bootstrap approximates sampling from role-specific empirical source pools
at their observed class mixtures. It does not model new-site shift, prevalence
uncertainty, unknown clusters, or a fully unconditional training population.
Selection is nonsmooth; small calibration samples and tail quantiles can be
unstable. No finite-sample coverage theorem is claimed.

## Multiplicity and assurance criteria

The primary family contains four contrasts: two challengers by two datasets.
Allocate family alpha 0.05 with Bonferroni, yielding individual 98.75% joint
percentile intervals. A union-bound argument requires correct component
coverage; bootstrap approximation error prevents a guaranteed familywise claim.
At 2,000 replicates, expected draws per tail are 12.5. Report this resolution,
and withhold categorical assurance if it is below the configured minimum 10.
This minimum is a computational guard, not a precision or power theorem.

Negative loss difference favors the challenger. In order:

1. Safety veto, fewer than 80 unique test cases, or insufficient tail resolution:
   inconclusive; mandatory human review.
2. Upper bound < -0.01: improvement.
3. Lower bound > 0.02: degradation.
4. Otherwise upper bound < 0.02: non-inferiority at the specified margin.
5. Otherwise inconclusive.

Strict inequalities make equality at a boundary inconclusive unless another
criterion applies. Non-inferiority is not equivalence. Margins and case minimum
are illustrative research settings, not domain-certified utility requirements.
All categories have `automatic_deployment_allowed=false` and
`human_review_required=true`. No code path approves deployment.

## Execution and preservation

`run-phase3.ps1` restores/verifies the R: alias, optionally installs declared
dependencies, snapshots prior artifact hashes, records executed commands and
package versions, runs all tests/lint/dependency checks, executes the Phase 1 CLI
in an isolated verification directory, then executes Phase 3 in a new timestamped
directory. It verifies preservation of prior files and recalculates stored
effects/intervals from saved scores before marking output verified. Dataset caches
and all generated outputs stay under ignored results/reports directories.
It never runs the publishing script, changes registry settings, or replaces an
existing run. Test fixtures are synthetic and never presented as public-data
experimental findings. Online fetches are not required for unit/integration tests.

Each new run also archives the executed Python source beside its hash manifest.
The verifier checks that archive when available, so later repository changes do
not prevent verification of historical outputs. Report-only rendering corrections
can be delivered as additional files while preserving original report bytes.

These are implemented established tools plus empirical investigations, not a new
theory of false-release-rate control. Further work requires group/session IDs,
fresh cohorts, domain-specific practical margins, selection-aware coverage
simulation, independent sites, subgroup/shift diagnostics and prespecified
related multiplicity. Do not tune on these held-out results.
