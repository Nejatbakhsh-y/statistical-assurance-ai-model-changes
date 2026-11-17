import numpy as np
import pytest

from model_assurance.calibration import reliability_bins
from model_assurance.decision_framework import recommend
from model_assurance.evaluation import binary_losses, evaluate
from model_assurance.logits import (
    center_multiclass_logits,
    probability_from_binary_score,
    scale_binary_score,
    softmax_probs,
)
from model_assurance.power_analysis import paired_required_cases
from model_assurance.statistics import exact_mcnemar, paired_bootstrap


def test_softmax_shift_invariance():
    z = np.array([[1.0, 2.0, 3.0], [5.0, -2.0, 0.0]])
    np.testing.assert_allclose(softmax_probs(z), softmax_probs(z + 10))
    np.testing.assert_allclose(center_multiclass_logits(z).mean(axis=1), 0, atol=1e-12)


def test_probabilities_and_unchanged_decisions():
    x = np.array([-2.0, -0.5, 0.2, 3.0])
    a = probability_from_binary_score(x)
    b = probability_from_binary_score(scale_binary_score(x, 0.7))
    assert np.all((a > 0) & (a < 1)) and np.any(a != b)
    np.testing.assert_array_equal(a >= 0.5, b >= 0.5)


def test_metrics_and_calibration():
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    assert evaluate(y, p)["accuracy"] == 1
    assert evaluate(y, p)["confusion_matrix"] == [[1, 0], [0, 1]]
    assert evaluate(y, p)["brier"] == pytest.approx(0.01)
    assert sum(z["count"] for z in reliability_bins(y, p)) == 2


def test_bootstrap_reproducible_and_pairing():
    d = np.array([-0.3, 0.1, -0.2, 0.4])
    assert paired_bootstrap(d, seed=13) == paired_bootstrap(d, seed=13)
    with pytest.raises(ValueError):
        paired_bootstrap([float("nan"), 1])
    with pytest.raises(ValueError):
        exact_mcnemar([1], [1, 0], [1])


def test_identical_models_and_invalid_probs():
    y = np.array([1, 0, 1])
    a = np.array([1, 0, 0])
    assert exact_mcnemar(y, a, a)["p_value_two_sided"] == 1
    with pytest.raises(ValueError):
        binary_losses(y, [0.1, 1.2, 0.4])


def test_release_gate_and_budget():
    cfg = {
        "minimum_cases": 100,
        "minimum_logloss_improvement": 0.0,
        "rejection_tolerance": 0.02,
        "safety_veto": False,
    }
    assert recommend({"lower": -0.04, "upper": -0.01}, 120, cfg)["status"] == "recommend_approval"
    assert recommend({"lower": 0.03, "upper": 0.06}, 120, cfg)["status"] == "recommend_rejection"
    assert (
        recommend({"lower": -0.04, "upper": 0.04}, 120, cfg)["status"]
        == "recommend_additional_evaluation"
    )
    assert (
        recommend({"lower": -0.04, "upper": -0.01}, 20, cfg)["status"]
        == "recommend_additional_evaluation"
    )
    assert paired_required_cases(0.2, 0.05) > 0


def test_numerical_boundaries_and_power_domain():
    from model_assurance.external_validation import stable_losses

    x = np.array([-4e-16, 0.0, 4e-16])
    np.testing.assert_array_equal(x >= 0, scale_binary_score(x, 0.7) >= 0)
    assert stable_losses(np.array([1]), np.array([40.0]))[0] > 0
    assert stable_losses(np.array([1]), np.array([40.0]))[0] == pytest.approx(np.exp(-40))
    for sd, effect, power in [(np.nan, 0.1, 0.8), (0.2, np.inf, 0.8), (0.2, 0.1, 0.1)]:
        with pytest.raises(ValueError):
            paired_required_cases(sd, effect, power=power)


def test_end_to_end(tmp_path):
    from pathlib import Path

    from model_assurance.cli import run

    source = Path(__file__).resolve().parents[1] / "configs/experiment.yaml"
    r = run(source, tmp_path)
    assert r["decision_disagreement_rate"] == 0.0
    assert r["mean_absolute_probability_change"] > 0
    assert (tmp_path / "reports/validation.html").exists()
    assert (tmp_path / "figures/reliability.png").exists()
    assert (tmp_path / "results/case_predictions.csv").exists()
