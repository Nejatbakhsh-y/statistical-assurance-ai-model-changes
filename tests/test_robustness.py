from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import expit
from sklearn.datasets import load_breast_cancer

from model_assurance.robustness import (
    fit_partition,
    partitions,
    probability_logit,
    run,
    scores,
    subject_interval,
    validate_config,
)
from model_assurance.statistics import paired_bootstrap


@pytest.fixture
def config():
    c = yaml.safe_load((Path(__file__).parents[1] / "configs/phase2.yaml").read_text())
    c.update(
        split_seeds=[11, 22], training_seeds=[101, 202], forest_trees=5, bootstrap_replicates=100
    )
    return c


@pytest.mark.parametrize(
    "key,value",
    [
        ("test_size", 1),
        ("calibration_size", float("nan")),
        ("logistic_c", -1),
        ("forest_trees", 0),
        ("split_seeds", [1, 1]),
        ("training_seeds", [1]),
        ("bootstrap_replicates", 99),
        ("evaluation_seed", -1),
    ],
)
def test_bad_configuration(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_partitions_are_fixed_disjoint_and_complete(config):
    y = load_breast_cancer().target
    test, splits = partitions(y, config)
    other, again = partitions(y, config)
    np.testing.assert_array_equal(test, other)
    assert splits == again
    for split in splits:
        train, cal = set(split["train"]), set(split["calibration"])
        assert not train & cal and not train & set(test) and not cal & set(test)
        assert train | cal | set(test) == set(range(len(y)))


def test_no_evaluation_leakage_or_calibration_preprocessing_leakage(config):
    data = load_breast_cancer()
    x, y = data.data.copy(), data.target.copy()
    test, splits = partitions(y, config)
    split = splits[0]
    a, ca = fit_partition(x, y, split, test, 101, config)
    probe = x[test[:5]].copy()
    # Poison evaluation features and labels, holding the partition fixed.
    x[test] = 1e9
    y[test] = 1 - y[test]
    b, cb = fit_partition(x, y, split, test, 101, config)
    for name in scores(a, ca, probe):
        np.testing.assert_array_equal(scores(a, ca, probe)[name], scores(b, cb, probe)[name])
    scaler = a["baseline"].named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, data.data[split["train"]].mean(axis=0))
    assert scaler.n_samples_seen_ == len(split["train"])
    invalid = deepcopy(split)
    invalid["calibration"][0] = invalid["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        fit_partition(x, y, invalid, test, 101, config)
    invalid = deepcopy(split)
    invalid["train"][0] = 1.5
    with pytest.raises(ValueError, match="subject IDs"):
        fit_partition(x, y, invalid, test, 101, config)


def test_probability_logit_and_subject_resampling(config):
    p = np.array([0.01, 0.3, 0.5, 0.9])
    np.testing.assert_allclose(expit(probability_logit(p)), p)
    assert np.isfinite(probability_logit(np.array([0.0, 1.0]))).all()
    for bad in ([np.nan], [-0.1], [[0.5]]):
        with pytest.raises(ValueError):
            probability_logit(np.array(bad))
    d = np.array([[0.1, 0.2, -0.3], [0.4, -0.2, 0.6]])
    ci = subject_interval(d, config)
    assert ci == paired_bootstrap(d.mean(axis=0), confidence=0.975, replicates=100, seed=904)
    # Duplicating fits must not manufacture independent subjects or narrower intervals.
    assert ci == subject_interval(np.tile(d, (2, 1)), config)
    assert subject_interval(np.zeros((2, 3)), config)["upper"] == 0


def test_phase2_reproducible_and_refuses_overwrite(config, tmp_path):
    source = tmp_path / "phase2.yaml"
    source.write_text(yaml.safe_dump(config))
    a = run(source, tmp_path / "a")
    b = run(source, tmp_path / "b")
    assert a == b
    assert (tmp_path / "a/predictions.csv").read_bytes() == (
        tmp_path / "b/predictions.csv"
    ).read_bytes()
    assert (tmp_path / "a/validation.md").exists()
    assert len(a["runs"]) == 8
    assert len(set(a["evaluation_ids"])) == a["n_evaluation_subjects"]
    with pytest.raises(FileExistsError):
        run(source, tmp_path / "a")
