"""Read-only semantic verification. Passing is not statistical coverage validation."""

import argparse
import importlib.metadata
import json
import platform
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .external_data import load_dataset, load_manifest, sha256
from .external_validation import MODELS, fit_models, make_partition, predict_scores, validate_config
from .verification_math import (
    category,
    changes,
    contrasts,
    interval,
    loss,
    model_metrics,
    require_equal,
)


def read_json(path):
    def bad(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8-sig"), parse_constant=bad, object_pairs_hook=pairs
        )
    except (KeyError, TypeError, FileNotFoundError) as exc:
        raise ValueError(f"Missing or malformed evidence: {path}") from exc


def safe_member(root, relative):
    if not isinstance(relative, str) or ":" in relative:
        raise ValueError("Unsafe manifest member")
    p = PurePosixPath(relative.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Unsafe manifest member")
    target = root / Path(*p.parts)
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Manifest member escapes evidence directory")
    return target


def check_hashes(root):
    manifest = read_json(root / "artifact-hashes.json")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("A complete artifact integrity manifest is required")
    normalized = {key.replace("\\", "/") for key in manifest}
    if len(normalized) != len(manifest):
        raise ValueError("Duplicate normalized manifest members")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "artifact-hashes.json"
    }
    if normalized != actual:
        raise ValueError("Artifact integrity manifest has missing or unlisted files")
    for name, digest in manifest.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or sha256(safe_member(root, name)) != digest
        ):
            raise ValueError(f"Artifact integrity mismatch: {name}")


def selection(x, y, train, validation, c):
    """Replay deterministic validation selection independently of experiment fit_models."""
    scores = []
    for strength in c["logistic_candidates"]:
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=strength, max_iter=2000),
        )
        model.fit(x[train], y[train])
        s = model.decision_function(x[validation])
        scores.append(float(loss(y[validation], s).mean()))
    return c["logistic_candidates"][int(np.argmin(scores))], scores


def draw_ids(ids, y, rng):
    return np.concatenate(
        [
            rng.choice(ids[y[ids] == label], int(np.sum(y[ids] == label)), replace=True)
            for label in (0, 1)
        ]
    )


def require_array(a, shape, context, integer=False):
    if a.shape != shape or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError(f"{context}: invalid array shape/type/values")
    if integer and not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{context}: integer IDs required")


def close(a, b, context):
    if np.asarray(a).shape != np.asarray(b).shape or not np.allclose(a, b, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{context}: prediction-level reconstruction mismatch")


def report_checks(root, metrics):
    """Validate numeric result rows in canonical reports; historical copies remain hash-protected."""
    integrated_text = (root / "validation.md").read_text(encoding="utf-8")
    for name, m in metrics.items():
        text = (root / name / "validation.md").read_text(encoding="utf-8")
        for model, s in m["summary"].items():
            ci = s["joint_refit_and_test_ci"]
            row = (
                f"| {name} | {model} | {ci['estimate']:.8f} | "
                f"[{ci['lower']:.8f}, {ci['upper']:.8f}] | {s['assurance']['category']} |"
            )
            if integrated_text.splitlines().count(row) != 1:
                raise ValueError("Canonical integrated report result mismatch")
            primary = (
                f"- {model}: effect {ci['estimate']:.8f}; {100 * ci['confidence']:.3f}% interval "
                f"[{ci['lower']:.8f}, {ci['upper']:.8f}]; assurance **{s['assurance']['category']}**"
            )
            if primary not in text:
                raise ValueError("Canonical dataset report result mismatch")


def verify(
    output,
    manifest_path="configs/datasets-phase3.json",
    cache="results/external-cache",
    checks=None,
    *,
    config_path=None,
    record=None,
):
    """Read-only by default; optional records are exclusive-create outside the archive.

    All metric fields and interval metadata are recomputed; original model bundles
    and every bootstrap validation choice are replayed. Stored bootstrap forest
    scores are reconstructed statistically, not all retrained. No coverage claim.
    """
    root = Path(output)
    if record is not None and Path(record).exists():
        raise FileExistsError("Refusing to replace a detached verification record")
    if record is not None and Path(record).resolve().is_relative_to(root.resolve()):
        raise ValueError("Verification records must be outside the immutable archive")
    check_hashes(root)
    env = read_json(root / "environment.json")
    keys = {
        "python",
        "executable",
        "platform",
        "argv",
        "packages",
        "configuration",
        "config_sha256",
        "manifest_sha256",
        "source_sha256",
    }
    if set(env) != keys:
        raise ValueError("Unsupported environment schema")
    if env["python"] != platform.python_version():
        raise ValueError("Replay requires the recorded Python patch version")
    if not isinstance(env["argv"], list) or not all(isinstance(x, str) for x in env["argv"]):
        raise ValueError("Malformed execution arguments")
    if not isinstance(env["packages"], dict) or not isinstance(env["source_sha256"], dict):
        raise ValueError("Malformed environment metadata")
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib", "PyYAML", "threadpoolctl"):
        installed = importlib.metadata.version(package)
        if env["packages"].get(package) != installed:
            raise ValueError(f"Replay requires archived {package}=={env['packages'].get(package)}")
    manifest = load_manifest(manifest_path)
    if sha256(Path(manifest_path)) != env["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from execution")
    c = env["configuration"]
    validate_config(c, manifest)
    config_file = root / "configuration.yaml"
    if not config_file.exists():
        config_file = Path(config_path or "configs/phase3.yaml")
    if (
        sha256(config_file) != env["config_sha256"]
        or yaml.safe_load(config_file.read_text(encoding="utf-8")) != c
    ):
        raise ValueError("Original configuration bytes/content are required")
    sources = root / "source"
    if not sources.is_dir() or not env["source_sha256"]:
        raise ValueError("Archived executed source is required")
    if set(env["source_sha256"]) != {p.name for p in sources.iterdir() if p.is_file()}:
        raise ValueError("Source snapshot file set mismatch")
    for name, digest in env["source_sha256"].items():
        if "/" in name or not name.endswith(".py") or sha256(safe_member(sources, name)) != digest:
            raise ValueError("Executed source snapshot mismatch")
    family = len(c["datasets"]) * (len(MODELS) - 1)
    alpha = c["family_alpha"] / family
    all_metrics = {}
    report = {
        "schema": "semantic-verification-v2",
        "saved_scores_and_intervals_recalculated": True,
        "read_only": True,
        "coverage_power_false_release_control_established": False,
        "interpretation": "Exploratory classifications conditional on unvalidated statistical assumptions",
        "human_review_gate_verified": True,
        "automatic_deployment_allowed": False,
        "verifier_sha256": sha256(Path(__file__)),
        "test_checks": "not supplied",
        "datasets": {},
    }
    if checks is not None:
        # Historical text logs are supplementary, never authenticated execution attestations.
        checks = Path(checks)
        counts = re.findall(
            r"(\d+) passed", (checks / "pytest.txt").read_text(encoding="utf-8-sig")
        )
        if not counts or "All checks passed!" not in (checks / "ruff.txt").read_text(
            encoding="utf-8-sig"
        ):
            raise ValueError("Supplementary check logs are missing success evidence")
        if "No broken requirements found" not in (checks / "pip-check.txt").read_text(
            encoding="utf-8-sig"
        ):
            raise ValueError("Supplementary dependency log missing")
        report["test_checks"] = {
            "pytest_passed": int(counts[-1]),
            "scope": "supplementary unauthenticated logs",
        }
    for di, name in enumerate(c["datasets"]):
        m = read_json(root / name / "metrics.json")
        expected_keys = {
            "dataset",
            "provenance",
            "cleaning",
            "n_test",
            "partition_counts",
            "model_order",
            "raw_row_ids",
            "partitions",
            "summary",
            "fits",
            "configuration",
            "family_size",
            "individual_alpha",
        }
        if set(m) != expected_keys:
            raise ValueError("Unsupported dataset metrics schema")
        x, y, ids, cleaning = load_dataset(name, manifest, cache, download=False)
        require_equal(m["dataset"], name, "dataset")
        for key, expected in (
            ("provenance", manifest[name]),
            ("cleaning", cleaning),
            ("configuration", c),
            ("model_order", list(MODELS)),
            ("family_size", family),
            ("individual_alpha", alpha),
            ("raw_row_ids", ids.tolist()),
        ):
            require_equal(m[key], expected, key)
        roles = [make_partition(y, seed, c) for seed in c["split_seeds"]]
        require_equal(
            m["partitions"],
            [{k: v.tolist() for k, v in r.items()} for r in roles],
            "partitions/seeds",
        )
        primary = roles[0]
        yt = y[primary["test"]]
        n, splits, seeds, reps = (
            len(yt),
            len(roles),
            len(c["training_seeds"]),
            c["refit_replicates"],
        )
        require_equal(m["n_test"], n, "n_test")
        require_equal(
            m["partition_counts"], {k: len(v) for k, v in primary.items()}, "partition counts"
        )
        with np.load(root / name / "predictions.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        shapes = {
            "observed_scores": (splits, seeds, 3, n),
            "refit_scores": (reps, seeds, 3, n),
            "test_labels": (splits, n),
            "test_ids": (splits, n),
            "raw_row_ids": (len(ids),),
            "joint_draws": (reps, 2),
            "training_draws": (reps, 2),
            "evaluation_draws": (c["evaluation_replicates"], 2),
            "selected_c": (reps, seeds),
            **{k: (reps, len(v)) for k, v in primary.items()},
        }
        if set(arrays) != set(shapes):
            raise ValueError("Missing or unsupported prediction arrays")
        for key, shape in shapes.items():
            require_array(
                arrays[key], shape, key, key in {"test_labels", "test_ids", "raw_row_ids", *primary}
            )
        close(arrays["raw_row_ids"], ids, "raw IDs")
        close(arrays["test_ids"], [r["test"] for r in roles], "test IDs")
        close(arrays["test_labels"], [y[r["test"]] for r in roles], "test labels")
        observed = arrays["observed_scores"]
        expected_fits = []
        # First validate all score-derived fields before expensive independent refitting.
        for si, split_seed in enumerate(c["split_seeds"]):
            for ki, seed in enumerate(c["training_seeds"]):
                index = si * seeds + ki
                if not isinstance(m["fits"], list) or len(m["fits"]) != splits * seeds:
                    raise ValueError("Fit record count mismatch")
                f = m["fits"][index]
                if set(f) != {
                    "split_seed",
                    "training_seed",
                    "selected_c",
                    "validation_losses",
                    "models",
                    "changes",
                }:
                    raise ValueError("Unsupported fit fields")
                expected = {
                    "split_seed": split_seed,
                    "training_seed": seed,
                    "selected_c": f["selected_c"],
                    "validation_losses": f["validation_losses"],
                    "models": {
                        model: model_metrics(y[roles[si]["test"]], observed[si, ki, j])
                        for j, model in enumerate(MODELS)
                    },
                    "changes": {
                        model: changes(
                            y[roles[si]["test"]], observed[si, ki, 0], observed[si, ki, j]
                        )
                        for j, model in enumerate(MODELS)
                        if j
                    },
                }
                require_equal(f, expected, f"{name}.fit[{index}]")
                expected_fits.append(expected)
        d = contrasts(yt, observed[0])
        training = contrasts(yt, arrays["refit_scores"]).mean(axis=-1)
        db = contrasts(yt, arrays["refit_scores"])
        joint = np.empty((reps, 2))
        positions = {int(v): i for i, v in enumerate(primary["test"])}
        for b in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([c["bootstrap_seed"], di, b + 1]))
            for role, original_ids in primary.items():
                close(arrays[role][b], draw_ids(original_ids, y, rng), f"{role} RNG stream")
            ix = [positions[int(i)] for i in arrays["test"][b]]
            joint[b] = db[b][:, ix].mean(axis=1)
        rng = np.random.default_rng(np.random.SeedSequence([c["bootstrap_seed"], di, 0]))
        evaluation = np.array(
            [
                d[:, draw_ids(np.arange(n), yt, rng)].mean(axis=1)
                for _ in range(c["evaluation_replicates"])
            ]
        )
        for key, values in (
            ("joint_draws", joint),
            ("training_draws", training),
            ("evaluation_draws", evaluation),
        ):
            close(arrays[key], values, key)
        expected_summary = {}
        for j, model in enumerate(MODELS[1:]):
            ci = interval(joint[:, j], d[j].mean(), alpha)
            split_effects = [
                float(contrasts(y[r["test"]], observed[si])[j].mean()) for si, r in enumerate(roles)
            ]
            expected_summary[model] = {
                "joint_refit_and_test_ci": ci,
                "evaluation_only_ci": interval(evaluation[:, j], d[j].mean(), alpha),
                "training_validation_calibration_only_ci": interval(
                    training[:, j], d[j].mean(), alpha
                ),
                "assurance": category(ci, n, c),
                "split_effects_descriptive": split_effects,
                "split_sd_descriptive": float(np.std(split_effects, ddof=1)),
            }
        require_equal(m["summary"], expected_summary, f"{name}.summary")
        with threadpool_limits(limits=1):
            for si, role in enumerate(roles):
                choice, val_losses = selection(x, y, role["train"], role["validation"], c)
                for ki, seed in enumerate(c["training_seeds"]):
                    f = m["fits"][si * seeds + ki]
                    require_equal(f["selected_c"], choice, "validation selection")
                    require_equal(f["validation_losses"], val_losses, "validation losses")
                    fitted = fit_models(x, y, role, seed, c)
                    close(
                        observed[si, ki],
                        predict_scores(fitted, x[role["test"]]),
                        "original fitted scores",
                    )
            # Every stored bootstrap C is deterministic for this LBFGS selector.
            for b in range(reps):
                choice, _ = selection(x, y, arrays["train"][b], arrays["validation"][b], c)
                close(
                    arrays["selected_c"][b],
                    np.full(seeds, choice),
                    "bootstrap validation selection",
                )
        all_metrics[name] = m
        report["datasets"][name] = {
            "unique_test_cases": n,
            "metric_records_recalculated": splits * seeds * 3,
            "intervals_verified": 6,
            "bootstrap_selections_replayed": reps,
            "original_fit_bundles_replayed": splits * seeds,
        }
    integrated = {
        "datasets": {name: m["summary"] for name, m in all_metrics.items()},
        "family_size": family,
        "automatic_deployment_allowed": False,
        "human_review_required": True,
    }
    require_equal(read_json(root / "integrated.json"), integrated, "integrated")
    report_checks(root, all_metrics)
    report["input_manifest_sha256"] = sha256(root / "artifact-hashes.json")
    if record is not None:
        record = Path(record)
        if record.resolve().is_relative_to(root.resolve()):
            raise ValueError("Verification records must be outside the immutable archive")
        record.parent.mkdir(parents=True, exist_ok=True)
        with record.open("x", encoding="utf-8") as f:
            json.dump(
                {
                    **report,
                    "record_id": uuid.uuid4().hex,
                    "verified_utc": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="configs/datasets-phase3.json")
    parser.add_argument("--cache", "--data-dir", dest="cache", default="results/external-cache")
    parser.add_argument("--config")
    parser.add_argument("--checks")
    parser.add_argument("--record", help="New detached record, outside the input archive")
    parser.add_argument(
        "--read-only", action="store_true", help="Default; do not change archive files"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.output,
                args.manifest,
                args.cache,
                args.checks,
                config_path=args.config,
                record=args.record,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
