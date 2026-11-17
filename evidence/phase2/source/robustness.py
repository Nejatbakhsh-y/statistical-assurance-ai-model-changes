"""Exploratory repeated-development evaluation with a fixed subject holdout.

Inference conditions on all fitted models. Runs are never treated as subjects.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit, logit
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import reliability_bins
from .evaluation import binary_losses, evaluate
from .statistics import exact_mcnemar, paired_bootstrap

NAMES = ("regularized_logistic", "calibrated_forest")


def validate_config(c: dict) -> None:
    """Reject invalid budgets and seeds before fitting or creating outputs."""
    required = {
        "evaluation_seed", "test_size", "calibration_size", "split_seeds",
        "training_seeds", "bootstrap_seed", "bootstrap_replicates",
        "family_confidence", "logistic_c", "forest_trees", "forest_min_leaf",
        "calibration_c",
    }
    if set(c) != required:
        raise ValueError(f"Configuration keys must be exactly {sorted(required)}")
    for key in ("test_size", "calibration_size", "family_confidence"):
        if not isinstance(c[key], (float, int)) or not 0 < c[key] < 1:
            raise ValueError(f"{key} must be between zero and one")
    for key in ("logistic_c", "calibration_c"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("evaluation_seed", "bootstrap_seed", "bootstrap_replicates",
                "forest_trees", "forest_min_leaf"):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if c["bootstrap_replicates"] < 100 or min(c["forest_trees"], c["forest_min_leaf"]) < 1:
        raise ValueError("Invalid bootstrap or forest budget")
    for key in ("split_seeds", "training_seeds"):
        seeds = c[key]
        if (not isinstance(seeds, list) or len(seeds) < 2
                or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError(f"{key} requires at least two distinct valid seeds")
    if max(c["evaluation_seed"], c["bootstrap_seed"]) >= 2**32:
        raise ValueError("Seeds must be less than 2**32")


def partitions(y: np.ndarray, c: dict) -> tuple[np.ndarray, list[dict]]:
    """Freeze global subject IDs; calibration and training never include evaluation IDs."""
    dev, test = train_test_split(
        np.arange(len(y)), test_size=c["test_size"],
        stratify=y, random_state=c["evaluation_seed"],
    )
    splits = []
    for seed in c["split_seeds"]:
        train, cal = train_test_split(
            dev, test_size=c["calibration_size"], stratify=y[dev], random_state=seed,
        )
        splits.append({"split_seed": seed, "train": train.tolist(), "calibration": cal.tolist()})
    return test, splits


def fit_partition(x: np.ndarray, y: np.ndarray, split: dict, test: np.ndarray,
                  seed: int, c: dict) -> tuple[dict, LogisticRegression]:
    """Fit preprocessing/models on training only, sigmoid on calibration only.

    Evaluation IDs are used solely for overlap validation, never for fitting.
    Returns fitted estimators to permit independent leakage checks.
    """
    train = np.asarray(split["train"])
    cal = np.asarray(split["calibration"])
    test = np.asarray(test)
    groups = (train, cal, test)
    for ids in groups:
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or len(ids) == 0 or len(set(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= len(y))):
            raise ValueError("Invalid or duplicate subject IDs")
    if any(set(a) & set(b) for a, b in ((train, cal), (train, test), (cal, test))):
        raise ValueError("Training, calibration and evaluation subjects must be disjoint")
    models = {}
    for name, strength in (("baseline", 1.0), (NAMES[0], c["logistic_c"])):
        models[name] = make_pipeline(
            StandardScaler(), LogisticRegression(C=strength, max_iter=2000, random_state=seed),
        ).fit(x[train], y[train])
    models[NAMES[1]] = RandomForestClassifier(
        n_estimators=c["forest_trees"], min_samples_leaf=c["forest_min_leaf"],
        random_state=seed, n_jobs=1,
    ).fit(x[train], y[train])
    raw = probability_logit(models[NAMES[1]].predict_proba(x[cal])[:, 1])
    calibrator = LogisticRegression(C=c["calibration_c"], max_iter=2000).fit(
        raw[:, None], y[cal],
    )
    return models, calibrator


def probability_logit(p: np.ndarray) -> np.ndarray:
    """Numerically bounded probability-derived log odds; not native forest logits."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite one-dimensional probabilities in [0,1]")
    return logit(np.clip(p, 1e-15, 1 - 1e-15))


def scores(models: dict, calibrator: LogisticRegression, x: np.ndarray) -> dict:
    """Binary scores have the same positive class; cross-model scales remain descriptive."""
    result = {name: models[name].decision_function(x) for name in ("baseline", NAMES[0])}
    raw = probability_logit(models[NAMES[1]].predict_proba(x)[:, 1])
    result[NAMES[1]] = calibrator.decision_function(raw[:, None])
    return result


def subject_interval(differences: np.ndarray, c: dict) -> dict:
    """Average fits within subject, then resample paired subjects, not run rows."""
    d = np.asarray(differences, dtype=float)
    if d.ndim != 2 or d.shape[0] < 1:
        raise ValueError("Expected fit-by-subject paired differences")
    confidence = 1 - (1 - c["family_confidence"]) / len(NAMES)
    return paired_bootstrap(
        d.mean(axis=0), replicates=c["bootstrap_replicates"],
        confidence=confidence, seed=c["bootstrap_seed"],
    )


def run(config_path: str | Path, output: str | Path) -> dict:
    """Write a new output directory; refuse to overwrite any prior run."""
    c = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(c, dict):
        raise TypeError("Expected a configuration mapping")
    validate_config(c)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    data = load_breast_cancer()
    x, y = data.data, data.target
    test, splits = partitions(y, c)
    runs, predictions = [], []
    losses = {name: [] for name in NAMES}
    probabilities = {name: [] for name in ("baseline", *NAMES)}
    for split in splits:
        for seed in c["training_seeds"]:
            models, calibrator = fit_partition(x, y, split, test, seed, c)
            s = scores(models, calibrator, x[test])
            p = {name: expit(value) for name, value in s.items()}
            for name, values in probabilities.items():
                values.append(p[name])
            la, ba = binary_losses(y[test], p["baseline"])
            for name in NAMES:
                lb, bb = binary_losses(y[test], p[name])
                losses[name].append(lb - la)
                run_result = {
                    "split_seed": split["split_seed"], "training_seed": seed, "challenger": name,
                    "baseline_metrics": evaluate(y[test], p["baseline"]),
                    "challenger_metrics": evaluate(y[test], p[name]),
                    "logloss_difference": float((lb - la).mean()),
                    "brier_difference": float((bb - ba).mean()),
                    "accuracy_difference": float(np.mean((p[name] >= .5) == y[test])
                                                 - np.mean((p["baseline"] >= .5) == y[test])),
                    "mean_absolute_log_odds_change": float(np.abs(s[name] - s["baseline"]).mean()),
                    "mean_signed_margin_change": float(((2*y[test]-1)*(s[name]-s["baseline"])).mean()),
                    "mean_absolute_probability_change": float(np.abs(p[name]-p["baseline"]).mean()),
                    "disagreement": float(np.mean((p[name] >= .5) != (p["baseline"] >= .5))),
                    "mcnemar_exploratory": exact_mcnemar(y[test], p["baseline"] >= .5, p[name] >= .5),
                    "reliability": reliability_bins(y[test], p[name]),
                    "baseline_reliability": reliability_bins(y[test], p["baseline"]),
                }
                runs.append(run_result)
                for i, subject in enumerate(test):
                    predictions.append({
                        "subject_id": int(subject), "true_label": int(y[subject]),
                        "split_seed": split["split_seed"], "training_seed": seed, "challenger": name,
                        "baseline_score": float(s["baseline"][i]), "challenger_score": float(s[name][i]),
                        "baseline_probability": float(p["baseline"][i]),
                        "challenger_probability": float(p[name][i]),
                        "paired_logloss_difference": float(lb[i]-la[i]),
                    })
    summary = {}
    for name in NAMES:
        d = np.asarray(losses[name])
        grid = d.mean(axis=1).reshape(len(splits), len(c["training_seeds"]))
        summary[name] = {
            "paired_logloss_ci": subject_interval(d, c),
            "largest_subject_mean_loss_difference": float(d.mean(axis=0).max()),
            "largest_subject_id": int(test[np.argmax(d.mean(axis=0))]),
            "saturated_probability_count": int(np.sum(
                (np.asarray(probabilities[name]) <= 1e-15)
                | (np.asarray(probabilities[name]) >= 1-1e-15))),
            "split_mean_sd_descriptive": float(grid.mean(axis=1).std(ddof=1)),
            "mean_within_split_seed_sd_descriptive": float(grid.std(axis=1, ddof=1).mean()),
            "fit_effect_range": [float(grid.min()), float(grid.max())],
            "mean_brier_difference": float(np.mean([r["brier_difference"] for r in runs
                                                    if r["challenger"] == name])),
            "mean_accuracy_difference": float(np.mean([r["accuracy_difference"] for r in runs
                                                       if r["challenger"] == name])),
            "mean_prediction_reliability_descriptive": reliability_bins(
                y[test], np.mean(probabilities[name], axis=0)),
        }
    result = {
        "configuration": c, "dataset": "sklearn breast_cancer; previously inspected dataset",
        "dataset_sha256": hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_evaluation_subjects": len(test), "evaluation_ids": test.tolist(), "splits": splits,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "scikit-learn", "matplotlib", "PyYAML")},
        "python": platform.python_version(), "summary": summary, "runs": runs,
        "inference": "Conditional on fixed fitted models; paired subject bootstrap; two-primary-endpoint Bonferroni intervals; approximate coverage only",
        "recommendation": "Additional evaluation and mandatory human review; exploratory reused data cannot authorize release",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write_report(result, output / "validation.md")
    return result


def write_report(result: dict, path: Path) -> None:
    """Render findings exclusively from calculated outputs; no third-party audit claim."""
    lines = ["# Phase 2 independent-development validation report", "",
             "This is an internal validation analysis, not an independent third-party audit.",
             "Dataset reuse makes all findings exploratory. Mandatory human review applies.", "",
             f"Evaluation subjects: {result['n_evaluation_subjects']}; paired comparison records: {len(result['runs'])}.",
             f"Python: {result['python']}; packages: {json.dumps(result['versions'], sort_keys=True)}.",
             "", "## Primary effects", "",
             "Negative challenger-minus-baseline log loss favors challenger. Each interval is 97.5% at default configuration, targeting approximate 95% family coverage for two primary contrasts.", ""]
    for name, s in result["summary"].items():
        ci = s["paired_logloss_ci"]
        lines.append(f"- {name}: mean {ci['estimate']:.8f}; interval [{ci['lower']:.8f}, {ci['upper']:.8f}]; mean Brier difference {s['mean_brier_difference']:.8f}; mean accuracy difference {s['mean_accuracy_difference']:.8f}; split SD {s['split_mean_sd_descriptive']:.8f}; within-split seed SD {s['mean_within_split_seed_sd_descriptive']:.8f}.")
        matching = [r for r in result["runs"] if r["challenger"] == name]
        lines.append(
            f"  Descriptive level diagnostics: mean absolute log-odds change {np.mean([r['mean_absolute_log_odds_change'] for r in matching]):.8f}; "
            f"mean absolute probability change {np.mean([r['mean_absolute_probability_change'] for r in matching]):.8f}; "
            f"mean decision disagreement {np.mean([r['disagreement'] for r in matching]):.8f}. "
            f"Largest subject-mean paired loss difference {s['largest_subject_mean_loss_difference']:.8f} "
            f"at dataset subject ID {s['largest_subject_id']}; saturated challenger probabilities across fits/subjects: {s['saturated_probability_count']}."
        )
    lines += ["", "## Interpretation and limitations", "",
              "Intervals resample independent evaluation subjects after averaging loss differences across fits. They describe expected loss over this fixed finite fit collection, not the loss of averaged probabilities. No run-row pooling or independent-run t-test is used. Overlapping development splits make seed/split SDs descriptive, not variance components or confidence intervals. Fits are held fixed: training-population uncertainty is excluded.",
              "Logistic scores and calibrated forest log odds share positive-class orientation, but raw score magnitudes across model families have no intrinsic equivalence. Margin shifts are descriptive. Per-fit proper scores, reliability-bin counts, decision metrics and exploratory McNemar diagnostics are in metrics.json; unadjusted McNemar p-values support no confirmatory claims. Mean-prediction reliability describes a finite ensemble and differs from average single-fit calibration.",
              "Calibration uses disjoint development rows; no evaluation fitting or temperature/hyperparameter selection occurs. Sigmoid calibration is not guaranteed to improve calibration. Reliability bins can be sparse. No subgroup, temporal-shift, external-dataset or safety audit has been established.",
              "Numerical convention: proper-score evaluation clips probabilities to [1e-15, 1-1e-15], matching Phase 1; forest probability-derived log odds use the same clipping before sigmoid calibration. Saturated errors can dominate loss and widen bootstrap intervals. This is evidence to investigate tail risk and numerical sensitivity on future data, not permission to retune on this holdout. Saturation counts include both correct and incorrect extremes; largest-subject effects are post-hoc diagnostics.",
              "The mathematical identities and disjoint-index checks are established properties; bootstrap coverage is an approximation under independent subjects, not a finite-sample guarantee. General superiority, false-release-rate control, and publication-level novelty remain untested hypotheses.",
              "Recommendation: freeze candidate choices before fresh external data, replicate datasets and independently sampled training cohorts, examine subgroup/shift risk, budget bootstrap tails and subject counts using defensible pilot effects, and simulate interval coverage before any confirmatory release gate. Do not tune against these reported evaluation results.",
              "", "## Reproducibility", "",
              "Configuration, dataset hash, package versions, exact subject partitions and all seeds are in metrics.json. Per-case paired predictions are in predictions.csv. Existing output directories are never overwritten. Phase 1 outputs are untouched."]
    path.write_text("\n\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--output", required=True, help="New directory under results, reports or figures")
    args = parser.parse_args()
    output = Path(args.output)
    if output.is_absolute() or ".." in output.parts or output.parts[0] not in ("results", "reports", "figures"):
        parser.error("Output must be a relative subdirectory of results, reports or figures")
    result = run(args.config, output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
