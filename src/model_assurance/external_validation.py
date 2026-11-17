"""External-task validation with paired refit and evaluation-subject bootstraps."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .assurance import (
    assurance_category,
    calibration_metrics,
    percentile_interval,
    stratified_resample,
)
from .evaluation import evaluate
from .evidence import seal
from .external_data import load_dataset, load_manifest, sha256

MODELS = ("baseline", "tuned_logistic", "calibrated_forest")


def validate_config(c: dict, manifest: dict) -> None:
    """Fail before data access for malformed partitions, margins, budgets or seeds."""
    keys = {
        "datasets",
        "split_seeds",
        "training_seeds",
        "workers",
        "test_fraction",
        "validation_fraction",
        "calibration_fraction",
        "logistic_candidates",
        "forest_trees",
        "forest_min_leaf",
        "calibration_c",
        "bootstrap_seed",
        "refit_replicates",
        "evaluation_replicates",
        "family_alpha",
        "minimum_improvement",
        "noninferiority_margin",
        "degradation_margin",
        "minimum_test_cases",
        "minimum_tail_draws",
        "safety_veto",
    }
    if not isinstance(c, dict) or set(c) != keys:
        raise ValueError("Phase 3 configuration keys differ from the documented schema")
    if (
        not isinstance(c["datasets"], list)
        or len(c["datasets"]) < 2
        or len(set(c["datasets"])) != len(c["datasets"])
        or any(n not in manifest for n in c["datasets"])
    ):
        raise ValueError("At least two distinct pinned datasets are required")
    for name in ("split_seeds", "training_seeds"):
        values = c[name]
        if (
            not isinstance(values, list)
            or len(values) < 2
            or any(type(v) is not int or not 0 <= v < 2**32 for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid distinct {name}")
    for name in (
        "workers",
        "forest_trees",
        "forest_min_leaf",
        "bootstrap_seed",
        "refit_replicates",
        "evaluation_replicates",
        "minimum_test_cases",
        "minimum_tail_draws",
    ):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if c["bootstrap_seed"] >= 2**32 or min(c["refit_replicates"], c["evaluation_replicates"]) < 100:
        raise ValueError("Invalid bootstrap seed or replication budget")
    for name in (
        "test_fraction",
        "validation_fraction",
        "calibration_fraction",
        "family_alpha",
        "minimum_improvement",
        "noninferiority_margin",
        "degradation_margin",
        "calibration_c",
    ):
        v = c[name]
        if type(v) not in (int, float) or not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and positive")
    fractions = [c[n] for n in ("test_fraction", "validation_fraction", "calibration_fraction")]
    if sum(fractions) >= 1 or not c["family_alpha"] < 1:
        raise ValueError("Partition fractions must leave training rows and alpha must be below one")
    if c["degradation_margin"] < c["noninferiority_margin"]:
        raise ValueError("Degradation margin must not be below the non-inferiority margin")
    candidates = c["logistic_candidates"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("Invalid distinct logistic candidates")
    if type(c["safety_veto"]) is not bool:
        raise ValueError("safety_veto must be boolean")


def make_partition(y: np.ndarray, seed: int, c: dict) -> dict:
    """Four disjoint roles; percentages refer to the complete deduplicated dataset."""
    dev, test = train_test_split(
        np.arange(len(y)), test_size=c["test_fraction"], stratify=y, random_state=seed
    )
    trainval, cal = train_test_split(
        dev,
        test_size=c["calibration_fraction"] / (1 - c["test_fraction"]),
        stratify=y[dev],
        random_state=seed + 1,
    )
    train, val = train_test_split(
        trainval,
        test_size=c["validation_fraction"] / (1 - c["test_fraction"] - c["calibration_fraction"]),
        stratify=y[trainval],
        random_state=seed + 2,
    )
    return {"train": train, "validation": val, "calibration": cal, "test": test}


def validate_partition(x: np.ndarray, y: np.ndarray, roles: dict) -> None:
    """Duplicates within a bootstrap role are permitted, never across roles."""
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not np.isin(y, [0, 1]).all()
        or np.isinf(x).any()
        or set(roles) != {"train", "validation", "calibration", "test"}
    ):
        raise ValueError("Invalid binary data or role mapping")
    sets = []
    for ids in roles.values():
        ids = np.asarray(ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or len(ids) < 2
            or np.any(ids < 0)
            or np.any(ids >= len(y))
            or set(y[ids]) != {0, 1}
        ):
            raise ValueError("Every role requires valid indices with both classes")
        sets.append(set(ids))
    if sum(map(len, sets)) != len(set.union(*sets)):
        raise ValueError("Leakage: role overlap")


def stable_losses(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Bernoulli log loss from finite log odds, without probability-clipping caps."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if (
        score.ndim < 1
        or score.shape[-1] != len(y)
        or y.ndim != 1
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(score).all()
    ):
        raise ValueError("Finite scores and matching binary labels required")
    return np.logaddexp(0, (1 - 2 * y) * score)


def fit_models(x: np.ndarray, y: np.ndarray, roles: dict, seed: int, c: dict) -> dict:
    """Selection sees validation only; forest sigmoid sees calibration only."""
    validate_partition(x, y, roles)
    train, val, cal = (roles[n] for n in ("train", "validation", "calibration"))

    def logistic(strength):
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=strength, max_iter=2000, random_state=seed),
        )

    baseline = logistic(1.0).fit(x[train], y[train])
    candidates, validation_losses = [], []
    for strength in c["logistic_candidates"]:
        model = logistic(strength).fit(x[train], y[train])
        candidates.append(model)
        validation_losses.append(
            float(stable_losses(y[val], model.decision_function(x[val])).mean())
        )
    best = int(np.argmin(validation_losses))  # deterministic first-candidate tie break
    forest = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=c["forest_trees"],
            min_samples_leaf=c["forest_min_leaf"],
            n_jobs=1,
            random_state=seed,
        ),
    ).fit(x[train], y[train])
    # Bounded raw probabilities avoid the infinite log-odds inputs observed in Phase 2.
    raw = forest.predict_proba(x[cal])[:, 1]
    sigmoid = LogisticRegression(C=c["calibration_c"], max_iter=2000).fit(raw[:, None], y[cal])
    return {
        "baseline": baseline,
        "tuned_logistic": candidates[best],
        "forest": forest,
        "sigmoid": sigmoid,
        "selected_c": c["logistic_candidates"][best],
        "validation_losses": validation_losses,
    }


def predict_scores(models: dict, x: np.ndarray) -> np.ndarray:
    raw = models["forest"].predict_proba(x)[:, 1]
    return np.asarray(
        [
            models["baseline"].decision_function(x),
            models["tuned_logistic"].decision_function(x),
            models["sigmoid"].decision_function(raw[:, None]),
        ]
    )


def paired_effects(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Average seed-specific paired losses within each subject, not probabilities."""
    if scores.ndim != 3 or scores.shape[1] != len(MODELS) or scores.shape[0] < 1:
        raise ValueError("Expected seed-by-model-by-subject scores")
    losses = stable_losses(y, scores)
    return (losses[:, 1:, :] - losses[:, :1, :]).mean(axis=0)


def describe_fit(y: np.ndarray, score: np.ndarray) -> dict:
    p = expit(score)
    metrics = evaluate(y, p)
    metrics["log_loss"] = float(stable_losses(y, score).mean())
    return {"metrics": metrics, "calibration": calibration_metrics(y, p)}


def refit_draw(
    b: int, x: np.ndarray, y: np.ndarray, primary: dict, c: dict, dataset_index: int
) -> tuple:
    """Independent RNG streams make process scheduling irrelevant to resampling."""
    rng = np.random.default_rng(np.random.SeedSequence([c["bootstrap_seed"], dataset_index, b + 1]))
    resampled = {n: stratified_resample(ids, y, rng) for n, ids in primary.items()}
    fit_roles = {**resampled, "test": primary["test"]}
    score, selected = [], []
    # One BLAS thread per process avoids nested numerical thread oversubscription.
    with threadpool_limits(limits=1):
        for seed in c["training_seeds"]:
            fitted = fit_models(x, y, fit_roles, seed, c)
            score.append(predict_scores(fitted, x[primary["test"]]))
            selected.append(fitted["selected_c"])
    return np.asarray(score), resampled, selected


def evaluate_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    raw_ids: np.ndarray,
    cleaning: dict,
    spec: dict,
    c: dict,
    family_size: int,
    output: Path,
    dataset_index: int,
) -> dict:
    """Primary first split; all later splits are descriptive sensitivity analyses."""
    roles_list, observed, records = [], [], []
    for split_seed in c["split_seeds"]:
        roles = make_partition(y, split_seed, c)
        roles_list.append(roles)
        seed_scores = []
        for seed in c["training_seeds"]:
            fitted = fit_models(x, y, roles, seed, c)
            s = predict_scores(fitted, x[roles["test"]])
            seed_scores.append(s)
            record = {
                "split_seed": split_seed,
                "training_seed": seed,
                "selected_c": fitted["selected_c"],
                "validation_losses": fitted["validation_losses"],
                "models": {n: describe_fit(y[roles["test"]], s[i]) for i, n in enumerate(MODELS)},
            }
            p = expit(s)
            record["changes"] = {
                n: {
                    "mean_absolute_score_change": float(np.abs(s[i] - s[0]).mean()),
                    "signed_margin_change": float(
                        ((2 * y[roles["test"]] - 1) * (s[i] - s[0])).mean()
                    ),
                    "mean_absolute_probability_change": float(np.abs(p[i] - p[0]).mean()),
                    "disagreement_rate": float(np.mean((p[i] >= 0.5) != (p[0] >= 0.5))),
                    "accuracy_difference": float(
                        np.mean((p[i] >= 0.5) == y[roles["test"]])
                        - np.mean((p[0] >= 0.5) == y[roles["test"]])
                    ),
                }
                for i, n in enumerate(MODELS)
                if i
            }
            records.append(record)
        observed.append(np.asarray(seed_scores))
    observed = np.asarray(observed)
    primary = roles_list[0]
    yt = y[primary["test"]]
    d = paired_effects(yt, observed[0])
    observed_effect = d.mean(axis=1)
    alpha = c["family_alpha"] / family_size
    rng_eval = np.random.default_rng(
        np.random.SeedSequence([c["bootstrap_seed"], dataset_index, 0])
    )
    eval_draws = np.empty((c["evaluation_replicates"], 2))
    for b in range(c["evaluation_replicates"]):
        ids = stratified_resample(np.arange(len(yt)), yt, rng_eval)
        eval_draws[b] = d[:, ids].mean(axis=1)
    shape = (c["refit_replicates"], len(c["training_seeds"]), len(MODELS), len(yt))
    refit_scores = np.empty(shape)
    joint, training = np.empty((c["refit_replicates"], 2)), np.empty((c["refit_replicates"], 2))
    sampled_roles = {
        n: np.empty((c["refit_replicates"], len(ids)), dtype=np.int32) for n, ids in primary.items()
    }
    selected = np.empty((c["refit_replicates"], len(c["training_seeds"])))
    task = partial(refit_draw, x=x, y=y, primary=primary, c=c, dataset_index=dataset_index)
    pool = ProcessPoolExecutor(max_workers=c["workers"]) if c["workers"] > 1 else None
    try:
        draws = (
            pool.map(task, range(c["refit_replicates"]), chunksize=10)
            if pool
            else map(task, range(c["refit_replicates"]))
        )
        for b, (score, resampled, choices) in enumerate(draws):
            refit_scores[b], selected[b] = score, choices
            for n, ids in resampled.items():
                sampled_roles[n][b] = ids
            db = paired_effects(yt, score)
            training[b] = db.mean(axis=1)
            positions = {int(subject): pos for pos, subject in enumerate(primary["test"])}
            test_positions = [positions[int(i)] for i in resampled["test"]]
            joint[b] = db[:, test_positions].mean(axis=1)
            if (b + 1) % 50 == 0:
                print(
                    f"{name}: paired training/calibration/validation refits {b + 1}/{c['refit_replicates']}",
                    flush=True,
                )
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    summary = {}
    for i, challenger in enumerate(MODELS[1:]):
        ci = percentile_interval(joint[:, i], observed_effect[i], alpha)
        sensitivity = [
            float(paired_effects(y[r["test"]], s)[i].mean())
            for r, s in zip(roles_list, observed, strict=True)
        ]
        summary[challenger] = {
            "joint_refit_and_test_ci": ci,
            "evaluation_only_ci": percentile_interval(eval_draws[:, i], observed_effect[i], alpha),
            "training_validation_calibration_only_ci": percentile_interval(
                training[:, i], observed_effect[i], alpha
            ),
            "assurance": assurance_category(ci, len(yt), c),
            "split_effects_descriptive": sensitivity,
            "split_sd_descriptive": float(np.std(sensitivity, ddof=1)),
        }
    result = {
        "dataset": name,
        "provenance": spec,
        "cleaning": cleaning,
        "n_test": len(yt),
        "partition_counts": {n: len(ids) for n, ids in primary.items()},
        "model_order": list(MODELS),
        "raw_row_ids": raw_ids.tolist(),
        "partitions": [{n: ids.tolist() for n, ids in r.items()} for r in roles_list],
        "summary": summary,
        "fits": records,
        "configuration": c,
        "family_size": family_size,
        "individual_alpha": alpha,
    }
    output.mkdir()
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "predictions.npz",
        observed_scores=observed,
        refit_scores=refit_scores,
        test_labels=np.asarray([y[r["test"]] for r in roles_list]),
        test_ids=np.asarray([r["test"] for r in roles_list]),
        raw_row_ids=raw_ids,
        joint_draws=joint,
        training_draws=training,
        evaluation_draws=eval_draws,
        selected_c=selected,
        **sampled_roles,
    )
    write_dataset_report(result, output / "validation.md")
    return result


LIMITATIONS = """These datasets are external tasks, independently collected from the Phase 1/2 medical dataset and each other; this validates a method across domains, not transfer of one frozen breast-cancer model. No third-party audit, deployment approval, regulatory validation or general superiority is established.

Primary inference uses only the first prespecified split. Other whole-data splits overlap and are descriptive sensitivity analyses, never pooled as independent runs. Bootstrap replicates resample original training, validation, calibration and test rows within disjoint roles and refit the complete selection/calibration pipeline. Repeated predictions are never counted as independent subjects. Training-only intervals fix test rows; evaluation-only intervals fix fitted models; joint intervals vary both. These are not additive variance components. The two fixed training seeds define a finite algorithmic average; uncertainty over arbitrary new seeds is not fully integrated.

Resampling is stratified: class counts in every role are held fixed. Intervals condition on observed class mixtures and role-specific empirical source pools. They do not establish new-site, temporal, cluster, or unseen-prevalence performance. Percentile-bootstrap coverage is approximate; validation selection is nonsmooth, small calibration samples can be unstable, and Monte Carlo tail counts are recorded. Independence of unique rows and representativeness are assumptions, not verified facts. Missing source grouping IDs preclude cluster validation. Removing exact feature duplicates cannot establish absence of hidden dependence.

Four primary joint log-loss intervals (two datasets by two challengers) use Bonferroni alpha allocation. Family coverage follows only if the component intervals have their nominal coverage, which is not guaranteed here. Secondary Brier, ECE, reliability, margin and decision diagnostics are descriptive; no uncorrected secondary significance claims are made. A non-inferiority category is margin-specific and is not equivalence. Illustrative loss margins are not domain-certified utility thresholds. All categories require human review and block automatic deployment.

Score shifts are descriptive: logistic native log odds and sigmoid-calibrated forest log odds share class orientation but not an intrinsic cross-family scale. Primary log loss is calculated stably from scores using logaddexp, without Phase 1/2 probability clipping caps; it is therefore not a bitwise-identical numerical estimand at saturated probabilities. Calibration diagnostics use ten fixed equal-width bins; ECE depends on bin choice and sparse counts. Forest sigmoid calibration uses raw bounded forest probability, selected before these datasets' test results, rather than Phase 2's extreme raw log odds. No guaranteed calibration improvement is claimed.

Next research: acquire specimen/session group identifiers, independent sites/cohorts and temporal tests; validate practical margins; simulate selection-aware interval coverage; check bootstrap-tail stability; prespecify subgroup and shift safety endpoints and their multiplicity; replicate with independently sampled training populations. Do not retune against reported test outcomes. Public benchmark familiarity limits any claim of a prospectively untouched population. No new theorem or publication-level novelty is asserted."""


def write_dataset_report(r: dict, path: Path) -> None:
    spec = r["provenance"]
    lines = [
        f"# Phase 3: {spec['name']}",
        f"{spec['citation']} DOI: {spec['doi']}. Source: {spec['source']}. License: [{spec['license']}]({spec['license_url']}).",
        f"Snapshot: {spec['version']}; SHA256 `{spec['sha256']}`. Integrity verifies pinned bytes, not a publisher signature. Positive class: {spec['positive_class']}.",
        f"Cleaning: {json.dumps(r['cleaning'])}. Partition counts: {json.dumps(r['partition_counts'])}. Counts refer to unique feature rows; raw source IDs remain in metrics and NPZ.",
        "## Primary joint uncertainty and assurance",
        "Negative challenger-minus-baseline log loss favors the challenger.",
    ]
    for name, s in r["summary"].items():
        ci = s["joint_refit_and_test_ci"]
        lines.append(
            f"- {name}: effect {ci['estimate']:.8f}; {100 * ci['confidence']:.3f}% interval [{ci['lower']:.8f}, {ci['upper']:.8f}]; assurance **{s['assurance']['category']}** ({s['assurance']['reason']}). Expected bootstrap draws per tail: {ci['expected_draws_per_tail']:.2f}. Descriptive split effects: {s['split_effects_descriptive']}."
        )
        matching = [f for f in r["fits"] if f["split_seed"] == r["configuration"]["split_seeds"][0]]
        ece = np.mean([f["models"][name]["calibration"]["ece_10_equal_width"] for f in matching])
        brier = np.mean(
            [
                f["models"][name]["metrics"]["brier"] - f["models"]["baseline"]["metrics"]["brier"]
                for f in matching
            ]
        )
        change = {
            k: float(np.mean([f["changes"][name][k] for f in matching]))
            for k in matching[0]["changes"][name]
        }
        lines.append(
            f"  Primary-split descriptive mean ECE: {ece:.8f}; paired Brier change: {brier:.8f}; level diagnostics: {json.dumps(change, sort_keys=True)}. Per-fit baseline/challenger calibration bins, proper scores and confusion matrices are in metrics.json."
        )
        for label in ("evaluation_only_ci", "training_validation_calibration_only_ci"):
            bound = s[label]
            lines.append(
                f"  {label}: [{bound['lower']:.8f}, {bound['upper']:.8f}] (diagnostic decomposition; no extra assurance decisions)."
            )
    lines += [
        "## Dataset-specific independence limitation",
        spec["independence_limit"],
        "## Assumptions, limitations and research recommendations",
        LIMITATIONS,
        "## Reproducibility",
        "metrics.json contains configuration, provenance, exact partitions, fit metrics and selections. predictions.npz contains observed and refit scores, sampled role IDs, target labels and bootstrap effects. Parent environment.json and command logs record actual execution. No existing output is overwritten. Human review is mandatory; automatic_deployment_allowed is always false.",
    ]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path,
    manifest_path: str | Path,
    cache: str | Path,
    output: str | Path,
    offline: bool = False,
) -> dict:
    manifest = load_manifest(manifest_path)
    c = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_config(c, manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "configuration.yaml")
    (output / "status.json").write_text('{"status":"incomplete"}\n')
    environment = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "argv": sys.argv,
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
        "configuration": c,
        "config_sha256": sha256(Path(config_path)),
        "manifest_sha256": sha256(Path(manifest_path)),
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source").mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, output / "source" / source.name)
    results = {}
    for i, name in enumerate(c["datasets"]):
        x, y, raw_ids, cleaning = load_dataset(name, manifest, cache, download=not offline)
        results[name] = evaluate_dataset(
            name,
            x,
            y,
            raw_ids,
            cleaning,
            manifest[name],
            c,
            len(c["datasets"]) * 2,
            output / name,
            i,
        )
    integrated = {
        "datasets": {n: r["summary"] for n, r in results.items()},
        "family_size": len(results) * 2,
        "automatic_deployment_allowed": False,
        "human_review_required": True,
    }
    (output / "integrated.json").write_text(
        json.dumps(integrated, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Phase 3 integrated scientific validation",
        "Independent external tasks; not an independent third-party audit.",
        f"Family: {integrated['family_size']} primary comparisons; family alpha {c['family_alpha']}. No cross-dataset population pooling or universal-success claim.",
        "| Dataset | Challenger | Mean log-loss change | Joint interval | Assurance |",
        "|---|---|---:|---|---|",
    ]
    for name, r in results.items():
        for model, s in r["summary"].items():
            ci = s["joint_refit_and_test_ci"]
            lines.append(
                f"| {name} | {model} | {ci['estimate']:.8f} | [{ci['lower']:.8f}, {ci['upper']:.8f}] | {s['assurance']['category']} |"
            )
    lines += [
        "",
        "## Prespecified criteria",
        f"Improvement: upper < -{c['minimum_improvement']}; degradation: lower > {c['degradation_margin']}; otherwise non-inferiority: upper < {c['noninferiority_margin']}; otherwise inconclusive. Minimum unique test cases: {c['minimum_test_cases']}; minimum expected bootstrap draws per tail: {c['minimum_tail_draws']}. Safety veto overrides favorable evidence. These are advisory categories, never automatic approvals.",
        "## Assumptions and remaining research",
        LIMITATIONS,
        "## Reproducibility",
        "Dataset-specific validation.md files include provenance, attribution, integrity, results and limitations. environment.json records packages, source/config hashes and CLI invocation. Scores and role bootstrap draws are saved for independent recalculation. Every prior Phase 1/2 artifact remains separate. Review command logs and verification.json before treating this run as verified.",
    ]
    (output / "validation.md").write_text(
        "\n\n".join(lines).replace("|\n\n|", "|\n|") + "\n", encoding="utf-8"
    )
    (output / "status.json").write_text('{"status":"experiment_complete_verification_pending"}\n')
    seal(output)
    return integrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase3.yaml")
    parser.add_argument("--manifest", default="configs/datasets-phase3.json")
    parser.add_argument("--cache", default="results/external-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    for value in (args.cache, args.output):
        path = Path(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 2
            or path.parts[0] not in ("results", "reports", "figures", "evidence")
        ):
            parser.error(
                "Cache/output must be new relative subdirectories under results, reports or figures"
            )
    result = run(args.config, args.manifest, args.cache, args.output, args.offline)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
