import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from model_assurance.assurance import (
    assurance_category,
    calibration_metrics,
    percentile_interval,
    stratified_resample,
)
from model_assurance.external_data import load_dataset, parse_data, verify_bytes
from model_assurance.external_validation import (
    fit_models,
    make_partition,
    paired_effects,
    predict_scores,
    run,
    stable_losses,
    validate_config,
    validate_partition,
)
from model_assurance.verify_external import verify


@pytest.fixture
def config():
    c = yaml.safe_load((Path(__file__).parents[1] / "configs/phase3.yaml").read_text())
    c.update(
        datasets=["fixture_a", "fixture_b"],
        split_seeds=[17, 29],
        training_seeds=[101, 202],
        forest_trees=3,
        logistic_candidates=[0.1, 1.0],
        refit_replicates=100,
        evaluation_replicates=100,
        workers=1,
    )
    return c


@pytest.fixture
def data():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(120, 4))
    y = (x[:, 0] + rng.normal(size=len(x)) * 0.4 > 0).astype(int)
    return x, y


def fixture_spec(content, rows, features):
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": rows,
        "features": features,
        "labels": {"0": 0, "1": 1},
        "name": "Synthetic test fixture",
        "citation": "Synthetic test data; not public evidence",
        "doi": "not applicable",
        "source": "fixture://synthetic",
        "license": "test-only",
        "license_url": "fixture://synthetic",
        "url": "fixture://synthetic",
        "version": "test-v1",
        "positive_class": "Synthetic class 1",
        "independence_limit": "Synthetic test only",
    }


def test_integrity_schema_dedup_and_conflicts(tmp_path):
    content = b"1,2,0\n3,4,1\n1,2,0\n"
    spec = fixture_spec(content, 3, 2)
    x, y, ids, info = parse_data(content, spec)
    assert x.shape == (2, 2) and y.tolist() == [0, 1] and ids.tolist() == [0, 1]
    assert info["removed_duplicate_raw_ids"] == [2]
    with pytest.raises(ValueError, match="integrity"):
        verify_bytes(content + b"x", spec)
    changed = {**spec, "features": 3}
    with pytest.raises(ValueError, match="schema"):
        parse_data(content, changed)
    conflicting = b"1,2,0\n1,2,1\n"
    with pytest.raises(ValueError, match="Conflicting"):
        parse_data(conflicting, fixture_spec(conflicting, 2, 2))
    (tmp_path / "fixture.data").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_dataset("fixture", {"fixture": spec}, tmp_path, download=False)
    assert (tmp_path / "fixture.data").read_bytes() == b"corrupt"
    with pytest.raises(FileNotFoundError):
        load_dataset("missing", {"missing": spec}, tmp_path, download=False)


def test_four_roles_bootstrap_counts_and_dependence(config, data):
    x, y = data
    a = make_partition(y, 17, config)
    b = make_partition(y, 17, config)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert sum(map(len, a.values())) == len(y)
    validate_partition(x, y, a)
    rng = np.random.default_rng(4)
    draws = {n: stratified_resample(ids, y, rng) for n, ids in a.items()}
    validate_partition(x, y, draws)
    for name in a:
        assert set(draws[name]) <= set(a[name])
        np.testing.assert_array_equal(np.bincount(y[draws[name]]), np.bincount(y[a[name]]))
    bad = deepcopy(a)
    bad["validation"][0] = bad["train"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_partition(x, y, bad)
    with pytest.raises(ValueError):
        stratified_resample(np.array([0.5, 1.5]), y, rng)
    with pytest.raises(ValueError):
        stratified_resample(np.flatnonzero(y == 0), y, rng)


def test_preprocessing_selection_calibration_and_test_leakage(config, data):
    x, y = data
    roles = make_partition(y, 17, config)
    x = x.copy()
    x[roles["train"][0], 0] = np.nan
    a = fit_models(x, y, roles, 101, config)
    probe = np.zeros((5, 4))
    xx, yy = x.copy(), y.copy()
    xx[roles["test"]] = 1e8
    yy[roles["test"]] = 1 - yy[roles["test"]]
    b = fit_models(xx, yy, roles, 101, config)
    assert a["selected_c"] == b["selected_c"]
    np.testing.assert_array_equal(predict_scores(a, probe), predict_scores(b, probe))
    np.testing.assert_allclose(
        a["baseline"].named_steps["simpleimputer"].statistics_,
        np.nanmedian(x[roles["train"]], axis=0),
    )
    assert a["baseline"].named_steps["standardscaler"].n_samples_seen_ == len(roles["train"])
    # Calibration-label changes cannot change a trained base model or validation selection.
    yc = y.copy()
    yc[roles["calibration"]] = 1 - yc[roles["calibration"]]
    calibrated = fit_models(x, yc, roles, 101, config)
    assert calibrated["selected_c"] == a["selected_c"]
    np.testing.assert_array_equal(
        calibrated["baseline"].predict_proba(probe), a["baseline"].predict_proba(probe)
    )
    np.testing.assert_array_equal(
        calibrated["forest"].predict_proba(probe), a["forest"].predict_proba(probe)
    )
    # Validation features are not used to fit either preprocessing or forest.
    xv = x.copy()
    xv[roles["validation"]] = 1e9
    selected = fit_models(xv, y, roles, 101, config)
    np.testing.assert_array_equal(
        a["baseline"].predict_proba(probe), selected["baseline"].predict_proba(probe)
    )
    np.testing.assert_array_equal(
        a["forest"].predict_proba(probe), selected["forest"].predict_proba(probe)
    )


def test_stable_scores_pairing_and_calibration():
    assert stable_losses(np.array([0, 1]), np.array([1000.0, -1000.0])).tolist() == [1000.0, 1000.0]
    s = np.array([[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]])
    np.testing.assert_allclose(
        paired_effects(np.array([0, 1]), s),
        paired_effects(np.array([0, 1]), np.repeat(s, 10, axis=0)),
    )
    zero = percentile_interval(np.zeros(100), 0, 0.0125)
    assert zero["lower"] == zero["upper"] == 0
    assert zero["confidence"] == 0.9875
    assert calibration_metrics(np.array([0, 1]), np.array([0.0, 1.0]))["ece_10_equal_width"] == 0
    with pytest.raises(ValueError):
        percentile_interval(np.full(100, np.nan), 0, 0.05)
    with pytest.raises(ValueError):
        stable_losses(np.array([0, 1]), np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        calibration_metrics(np.array([0, 1]), np.array([-0.1, 0.2]))


@pytest.mark.parametrize(
    "bounds,category",
    [
        ((-0.1, -0.02), "improvement"),
        ((-0.01, 0.01), "non_inferiority"),
        ((0.03, 0.1), "degradation"),
        ((-0.1, 0.1), "inconclusive"),
        ((-0.01, 0.02), "inconclusive"),
    ],
)
def test_prespecified_gates_never_authorize_deployment(config, bounds, category):
    ci = {"lower": bounds[0], "upper": bounds[1], "expected_draws_per_tail": 20}
    result = assurance_category(ci, 100, config)
    assert result["category"] == category
    assert result["human_review_required"] and not result["automatic_deployment_allowed"]
    assert assurance_category(ci, 20, config)["category"] == "inconclusive"
    assert (
        assurance_category({**ci, "expected_draws_per_tail": 1}, 100, config)["category"]
        == "inconclusive"
    )
    config["safety_veto"] = True
    assert assurance_category(ci, 100, config)["category"] == "inconclusive"


@pytest.mark.parametrize(
    "key,value",
    [
        ("datasets", ["fixture_a", "fixture_a"]),
        ("refit_replicates", 99),
        ("test_fraction", 0.9),
        ("family_alpha", 1),
        ("safety_veto", "false"),
        ("training_seeds", [101, 101]),
        ("calibration_c", float("nan")),
        ("logistic_candidates", [0]),
        ("minimum_improvement", -0.01),
    ],
)
def test_invalid_configuration(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config, {"fixture_a": {}, "fixture_b": {}})


def test_offline_integration_reproducibility_and_artifacts(config, data, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    x, y = data
    text = "".join(
        ",".join(map(str, [*row, int(label)])) + "\n" for row, label in zip(x, y, strict=True)
    )
    content = text.encode()
    spec = fixture_spec(content, len(y), x.shape[1])
    manifest = {n: spec for n in config["datasets"]}
    for name in manifest:
        (cache / f"{name}.data").write_bytes(content)
    cfg, man = tmp_path / "config.yaml", tmp_path / "manifest.json"
    cfg.write_text(yaml.safe_dump(config))
    man.write_text(json.dumps(manifest))
    a = run(cfg, man, cache, tmp_path / "a", offline=True)
    # Process scheduling must not change data draws, scores or inference.
    config["workers"] = 2
    cfg.write_text(yaml.safe_dump(config))
    b = run(cfg, man, cache, tmp_path / "b", offline=True)
    assert a == b and a["family_size"] == 4 and not a["automatic_deployment_allowed"]
    assert (tmp_path / "a/validation.md").exists()
    for name in manifest:
        with (
            np.load(tmp_path / f"a/{name}/predictions.npz") as aa,
            np.load(tmp_path / f"b/{name}/predictions.npz") as bb,
        ):
            for key in aa.files:
                np.testing.assert_array_equal(aa[key], bb[key])
            labels, original = aa["test_labels"][0], aa["observed_scores"][0]
            effect = paired_effects(labels, original).mean(axis=1)
            assert (
                effect[0]
                == a["datasets"][name]["tuned_logistic"]["joint_refit_and_test_ci"]["estimate"]
            )
            # Every resampled fitting ID remains outside the original test set.
            for role in ("train", "validation", "calibration"):
                assert not set(aa[role].ravel()) & set(aa["test_ids"][0])
    checked = verify(tmp_path / "a", man, cache)
    assert checked["saved_scores_and_intervals_recalculated"]
    assert checked["test_checks"] == "not supplied"
    target = tmp_path / "a/fixture_a/metrics.json"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="integrity"):
        verify(tmp_path / "a", man, cache)
    with pytest.raises(FileExistsError):
        run(cfg, man, cache, tmp_path / "a", offline=True)


@pytest.fixture(scope="module")
def strict_evidence(tmp_path_factory):
    """One real, small offline run shared by semantic corruption regressions."""
    root = tmp_path_factory.mktemp("strict-evidence")
    c = yaml.safe_load((Path(__file__).parents[1] / "configs/phase3.yaml").read_text())
    c.update(
        datasets=["fixture_a", "fixture_b"],
        split_seeds=[17, 29],
        training_seeds=[101, 202],
        forest_trees=3,
        logistic_candidates=[0.1, 1.0],
        refit_replicates=100,
        evaluation_replicates=100,
        workers=1,
    )
    rng = np.random.default_rng(91)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] > 0).astype(int)
    content = "".join(
        ",".join(map(str, [*row, int(label)])) + "\n" for row, label in zip(x, y, strict=True)
    ).encode()
    cache = root / "datasets"
    cache.mkdir()
    manifest = {name: fixture_spec(content, len(y), 4) for name in c["datasets"]}
    for name in manifest:
        (cache / f"{name}.data").write_bytes(content)
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(c))
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    evidence = root / "evidence"
    run(config, path, cache, evidence, offline=True)
    return evidence, path, cache


def test_strict_verification_read_only_and_detached_records(strict_evidence, tmp_path):
    evidence, manifest, cache = strict_evidence
    before = {
        p.relative_to(evidence): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in evidence.rglob("*")
        if p.is_file()
    }
    result = verify(evidence, manifest, cache, record=tmp_path / "verification.json")
    assert result["read_only"] and not result["coverage_power_false_release_control_established"]
    assert result["datasets"]["fixture_a"]["bootstrap_selections_replayed"] == 100
    assert before == {
        p.relative_to(evidence): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in evidence.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        verify(evidence, manifest, cache, record=tmp_path / "verification.json")
    with pytest.raises(ValueError, match="outside"):
        verify(evidence, manifest, cache, record=evidence / "new-record.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "ece",
        "bin",
        "calibration_bias",
        "precision",
        "confusion",
        "change",
        "validation_loss",
        "selected_c",
        "confidence",
        "replicates",
        "tail",
        "ci_lower",
        "category",
        "gate",
        "split_sd",
        "split_effect",
        "family",
        "alpha",
        "config",
        "missing",
        "unsupported",
        "nan",
        "shape",
        "bootstrap_choice",
        "rng",
        "report",
        "raw_label",
        "missing_manifest",
    ],
)
def test_semantic_verifier_rejects_falsified_evidence(strict_evidence, tmp_path, mutation):
    import shutil

    source, manifest, cache = strict_evidence
    evidence = tmp_path / "evidence"
    shutil.copytree(source, evidence)
    path = evidence / "fixture_a/metrics.json"
    m = json.loads(path.read_text())
    f = m["fits"][0]
    summary = m["summary"]["tuned_logistic"]
    ci = summary["joint_refit_and_test_ci"]
    if mutation == "ece":
        f["models"]["tuned_logistic"]["calibration"]["ece_10_equal_width"] = 0.987654321
    elif mutation == "bin":
        f["models"]["tuned_logistic"]["calibration"]["bins"][0]["count"] += 1
    elif mutation == "calibration_bias":
        f["models"]["tuned_logistic"]["calibration"]["mean_probability_minus_prevalence"] += 0.1
    elif mutation == "precision":
        f["models"]["tuned_logistic"]["metrics"]["precision"] += 0.1
    elif mutation == "confusion":
        f["models"]["tuned_logistic"]["metrics"]["confusion_matrix"][0][0] += 1
    elif mutation == "change":
        f["changes"]["tuned_logistic"]["disagreement_rate"] += 0.1
    elif mutation == "validation_loss":
        f["validation_losses"][0] += 0.1
    elif mutation == "selected_c":
        f["selected_c"] = 1234.0
    elif mutation == "confidence":
        ci["confidence"] = 0.123
    elif mutation == "replicates":
        ci["replicates"] = 1
    elif mutation == "tail":
        ci["expected_draws_per_tail"] = 1234.0
    elif mutation == "ci_lower":
        ci["lower"] -= 0.1
    elif mutation == "category":
        summary["assurance"]["category"] = "unsupported"
    elif mutation == "gate":
        summary["assurance"]["automatic_deployment_allowed"] = True
    elif mutation == "split_sd":
        summary["split_sd_descriptive"] += 0.1
    elif mutation == "split_effect":
        summary["split_effects_descriptive"][0] += 0.1
    elif mutation == "family":
        m["family_size"] = 1
    elif mutation == "alpha":
        m["individual_alpha"] = 0.5
    elif mutation == "config":
        m["configuration"]["test_fraction"] = 0.5
    elif mutation == "missing":
        del ci["confidence"]
    elif mutation == "unsupported":
        ci["unknown"] = True
    path.write_text(json.dumps(m))
    if mutation in ("nan", "shape", "bootstrap_choice", "rng", "raw_label"):
        ap = evidence / "fixture_a/predictions.npz"
        with np.load(ap) as a:
            arrays = {k: a[k] for k in a.files}
        if mutation == "nan":
            arrays["refit_scores"][0, 0, 0, 0] = np.nan
        elif mutation == "shape":
            arrays["observed_scores"] = arrays["observed_scores"][:1]
        elif mutation == "bootstrap_choice":
            arrays["selected_c"][0, 0] = 1234.0
        elif mutation == "rng":
            arrays["train"][0, 0] = arrays["test_ids"][0, 0]
        elif mutation == "raw_label":
            arrays["test_labels"][0, 0] = 1 - arrays["test_labels"][0, 0]
        np.savez_compressed(ap, **arrays)
    if mutation == "report":
        rp = evidence / "validation.md"
        rp.write_text(rp.read_text().replace("| tuned_logistic |", "| false_model |"))
    # Rehash the scratch fixture to exercise semantic checks, not only byte integrity.
    hp = evidence / "artifact-hashes.json"
    hp.write_text(
        json.dumps(
            {
                p.relative_to(evidence).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in evidence.rglob("*")
                if p.is_file() and p != hp
            }
        )
    )
    if mutation == "missing_manifest":
        hp.unlink()
    with pytest.raises(ValueError):
        verify(evidence, manifest, cache)
