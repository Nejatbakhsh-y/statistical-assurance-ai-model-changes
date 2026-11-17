# Instructions for Codex

## Mission
Maintain a statistically defensible, independently developed model-change validation research prototype. Do not claim Rutgers or NSF sponsorship.

## Repository boundaries
Read README.md, configuration, source, and tests before changes. Python package code lives under `src/model_assurance`, tests under `tests`, user configurations under `configs`. All generated results stay ignored under `results`, `figures`, and `reports`. Keep scripts idempotent. Never delete or overwrite unrelated or untracked work.

## Scientific correctness
The example challenger is a deterministic monotone score transform, not independent training. Do not use a held-out test set for model fitting or temperature selection. Resample paired subjects jointly; document case versus seed variability. Disallow pseudoreplication, uncorrected multiplicity, and unsupported equivalence claims. State assumptions for logit comparability, power, calibration, and inference. Decisions remain advisory with mandatory human review. Separate established methods from new hypotheses, and never invent publication-level novelty, benchmarks, or tests.

## Engineering
Use Python >=3.10, type-checked or well-documented interfaces, clear errors, stable CLI, pyproject.toml, pytest, Ruff, and GitHub CI. Pin or bound dependencies appropriately. Run `python -m pytest -q`, `ruff check src tests`, and `python -m model_assurance.cli run --config configs/experiment.yaml` after meaningful changes. Correct failures instead of disabling tests.

## Reproducibility and publication
Preserve explicit seeds, fixed held-out sets, software versions, and structured output. Do not commit credentials, personal data, virtual environments, generated results, or unsupported claims. A LICENSE requires author approval. Before push, inspect `git diff`, run checks, read README for consistency, and request authorization for remote changes. No force-push. Completion means runnable scripts, reproducible example, passing local tests, and an accurately bounded validation report.
