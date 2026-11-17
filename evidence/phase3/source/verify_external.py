"""Recalculate saved Phase 3 statistics from scores, not reported summary values."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import expit

from .assurance import assurance_category, stratified_resample
from .external_data import load_dataset, load_manifest, sha256
from .external_validation import MODELS, validate_partition


def verify(output: str | Path, manifest_path: str | Path = "configs/datasets-phase3.json",
           cache: str | Path = "results/external-cache", checks: str | Path | None = None) -> dict:
    """Mark a run verified only after direct reconstruction and integrity checks.

    An optional checks directory must contain successful pytest, Ruff and pip logs.
    Without it, verification explicitly makes no claim about those commands.
    """
    output = Path(output)
    if (output/"artifact-hashes.json").exists():
        for relative, digest in json.loads((output/"artifact-hashes.json").read_text()).items():
            if sha256(output/relative) != digest:
                raise ValueError(f"Saved artifact integrity mismatch: {relative}")
    environment = json.loads((output/"environment.json").read_text())
    manifest = load_manifest(manifest_path)
    if environment["manifest_sha256"] != sha256(Path(manifest_path)):
        raise ValueError("Manifest changed after execution")
    for name, digest in environment["source_sha256"].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise ValueError(f"Source changed after execution: {name}")
    c = environment["configuration"]
    integrated = json.loads((output/"integrated.json").read_text())
    if integrated["automatic_deployment_allowed"] or not integrated["human_review_required"]:
        raise ValueError("Human-review gate violated")
    report = {"saved_scores_and_intervals_recalculated": True, "datasets": {},
              "human_review_gate_verified": True, "test_checks": "not supplied"}
    if checks is not None:
        checks = Path(checks)
        pytest = (checks/"pytest.txt").read_text(encoding="utf-8-sig")
        counts = re.findall(r"(\d+) passed", pytest)
        if not counts or "failed" in pytest or "All checks passed!" not in (checks/"ruff.txt").read_text(encoding="utf-8-sig"):
            raise ValueError("Missing successful test/lint evidence")
        if "No broken requirements found" not in (checks/"pip-check.txt").read_text(encoding="utf-8-sig"):
            raise ValueError("Missing successful dependency evidence")
        report["test_checks"] = {"pytest_passed": int(counts[-1]), "ruff": "passed",
                                 "pip_check": "passed", "logs": str(checks)}
    for di, name in enumerate(c["datasets"]):
        metrics = json.loads((output/name/"metrics.json").read_text())
        x, y, raw_ids, cleaning = load_dataset(name, manifest, cache, download=False)
        if metrics["provenance"] != manifest[name] or metrics["cleaning"] != cleaning:
            raise ValueError("Dataset provenance or deduplication mismatch")
        with np.load(output/name/"predictions.npz", allow_pickle=False) as a:
            np.testing.assert_array_equal(a["raw_row_ids"], raw_ids)
            roles = [{n: np.asarray(ids) for n, ids in r.items()} for r in metrics["partitions"]]
            for split, r in enumerate(roles):
                validate_partition(x, y, r)
                if len(set.union(*(set(ids) for ids in r.values()))) != len(y):
                    raise ValueError("Incomplete original partition")
                np.testing.assert_array_equal(a["test_ids"][split], r["test"])
                np.testing.assert_array_equal(a["test_labels"][split], y[r["test"]])
            labels = y[roles[0]["test"]]
            # Independently reconstruct proper scores and seed-averaged contrasts.
            original = a["observed_scores"][0]
            loss = np.logaddexp(0, original)-labels*original
            effects = (loss[:, 1:, :]-loss[:, :1, :]).mean(axis=0)
            point = effects.mean(axis=1)
            alpha = c["family_alpha"]/(len(c["datasets"])*2)
            sb = a["refit_scores"]
            lb = np.logaddexp(0, sb)-labels*sb
            db = (lb[:, :, 1:, :]-lb[:, :, :1, :]).mean(axis=1)
            training = db.mean(axis=2)
            positions = {int(i): j for j, i in enumerate(roles[0]["test"])}
            joint = np.empty_like(training)
            for b in range(c["refit_replicates"]):
                for role, ids in roles[0].items():
                    draw = a[role][b]
                    if not set(draw) <= set(ids):
                        raise ValueError("Resampled IDs escaped their original role")
                    np.testing.assert_array_equal(np.bincount(y[draw]), np.bincount(y[ids]))
                indices = [positions[int(i)] for i in a["test"][b]]
                joint[b] = db[b][:, indices].mean(axis=1)
            np.testing.assert_allclose(training, a["training_draws"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(joint, a["joint_draws"], rtol=1e-12, atol=1e-12)
            rng = np.random.default_rng(np.random.SeedSequence([c["bootstrap_seed"], di, 0]))
            evaluation = np.empty((c["evaluation_replicates"], 2))
            for b in range(c["evaluation_replicates"]):
                ids = stratified_resample(np.arange(len(labels)), labels, rng)
                evaluation[b] = effects[:, ids].mean(axis=1)
            np.testing.assert_allclose(evaluation, a["evaluation_draws"], rtol=1e-12, atol=1e-12)
            for i, model in enumerate(MODELS[1:]):
                summary = metrics["summary"][model]
                if summary != integrated["datasets"][name][model]:
                    raise ValueError("Integrated report mismatch")
                for key, values in (("joint_refit_and_test_ci", joint),
                                    ("evaluation_only_ci", evaluation),
                                    ("training_validation_calibration_only_ci", training)):
                    ci = summary[key]
                    expected = [point[i], *np.quantile(values[:, i], [alpha/2, 1-alpha/2])]
                    np.testing.assert_allclose([ci["estimate"], ci["lower"], ci["upper"]], expected)
                if summary["assurance"] != assurance_category(summary["joint_refit_and_test_ci"], len(labels), c):
                    raise ValueError("Assurance gate mismatch")
            # Also verify all per-fit reported log losses and Brier scores from stored scores.
            for split, r in enumerate(roles):
                for seed_index, seed in enumerate(c["training_seeds"]):
                    f = next(f for f in metrics["fits"] if f["split_seed"] == c["split_seeds"][split]
                             and f["training_seed"] == seed)
                    for k, model in enumerate(MODELS):
                        s = a["observed_scores"][split, seed_index, k]
                        yy = y[r["test"]]
                        expected_loss = np.mean(np.logaddexp(0, s)-yy*s)
                        np.testing.assert_allclose(f["models"][model]["metrics"]["log_loss"], expected_loss)
                        np.testing.assert_allclose(f["models"][model]["metrics"]["brier"], np.mean((expit(s)-yy)**2))
            report["datasets"][name] = {"unique_source_rows": len(y), "unique_test_cases": len(labels),
                                        "refit_replicates": len(joint), "score_array_shape": list(sb.shape),
                                        "intervals_verified": 6, "four_role_leakage_checks": "passed"}
    (output/"verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (output/"status.json").write_text('{"status":"verified"}\n')
    hashes = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
              if p.is_file() and p.name != "artifact-hashes.json"}
    (output/"artifact-hashes.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="configs/datasets-phase3.json")
    parser.add_argument("--cache", default="results/external-cache")
    parser.add_argument("--checks")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.manifest, args.cache, args.checks), indent=2))


if __name__ == "__main__":
    main()
