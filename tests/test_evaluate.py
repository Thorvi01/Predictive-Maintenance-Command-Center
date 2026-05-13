# tests/test_evaluate.py
# Tests for evaluate.py — ECE calibration metric and NASA asymmetric score.
# Run with: pytest tests/test_evaluate.py -v

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import compute_ece, compute_sharpness
from train import nasa_score


# ── 1. ECE on a perfectly calibrated distribution ────────────────

def test_ece_perfect_calibration():
    """
    A model whose stated confidence exactly matches actual coverage
    should have ECE = 0.0.

    We construct this synthetically: predictions are the true values
    with Gaussian noise of known std. If we set std to match the actual
    noise, the model is perfectly calibrated by construction.
    """
    np.random.seed(42)
    n = 10_000
    true_rul = np.random.uniform(0, 125, n)

    # Model std = 10. Predictions are true ± N(0, 10).
    sigma = 10.0
    mean_preds = true_rul + np.random.normal(0, sigma, n)
    std_preds  = np.full(n, sigma)   # model knows its own noise

    ece, conf_levels, actual_cov = compute_ece(
        true_rul, mean_preds, std_preds, n_bins=10
    )

    # Perfect calibration → ECE ≈ 0. Allow 0.05 tolerance for sampling noise.
    assert ece < 0.05, \
        f"ECE should be near 0 for perfectly calibrated model, got {ece:.4f}"

def test_ece_overconfident_model():
    """
    A model that always reports std = 1 but actually has error std = 20
    is severely overconfident. ECE should be large (> 0.2).
    """
    np.random.seed(0)
    n = 5_000
    true_rul   = np.random.uniform(0, 125, n)
    mean_preds = true_rul + np.random.normal(0, 20, n)   # large actual error
    std_preds  = np.ones(n)                               # tiny stated uncertainty

    ece, _, _ = compute_ece(true_rul, mean_preds, std_preds, n_bins=10)

    assert ece > 0.2, \
        f"Overconfident model should have ECE > 0.2, got {ece:.4f}"

def test_ece_underconfident_model():
    """
    A model that always reports std = 100 but actually has error std = 1
    is severely underconfident. ECE should also be large (> 0.2).
    """
    np.random.seed(1)
    n = 5_000
    true_rul   = np.random.uniform(0, 125, n)
    mean_preds = true_rul + np.random.normal(0, 1, n)     # small actual error
    std_preds  = np.full(n, 100.0)                        # huge stated uncertainty

    ece, _, _ = compute_ece(true_rul, mean_preds, std_preds, n_bins=10)

    assert ece > 0.2, \
        f"Underconfident model should have ECE > 0.2, got {ece:.4f}"

def test_ece_returns_three_values():
    """compute_ece must return (ece, confidence_levels, actual_coverages)."""
    y    = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    mean = np.array([11.0, 21.0, 29.0, 41.0, 51.0])
    std  = np.array([ 2.0,  2.0,  2.0,  2.0,  2.0])

    result = compute_ece(y, mean, std)
    assert len(result) == 3, "compute_ece must return exactly 3 values"

def test_ece_confidence_levels_and_coverages_same_length():
    """Returned confidence_levels and actual_coverages must be the same length."""
    y    = np.random.uniform(0, 100, 200)
    mean = y + np.random.normal(0, 5, 200)
    std  = np.full(200, 5.0)

    _, conf_levels, actual_cov = compute_ece(y, mean, std, n_bins=8)

    assert len(conf_levels) == len(actual_cov), \
        "confidence_levels and actual_coverages have different lengths"
    assert len(conf_levels) == 8

def test_ece_non_negative():
    """ECE must always be non-negative."""
    np.random.seed(5)
    y    = np.random.uniform(0, 125, 500)
    mean = y + np.random.normal(0, 10, 500)
    std  = np.abs(np.random.normal(8, 3, 500))

    ece, _, _ = compute_ece(y, mean, std)
    assert ece >= 0, f"ECE must be >= 0, got {ece}"

def test_ece_actual_coverage_between_0_and_1():
    """All actual coverage values must be valid probabilities in [0, 1]."""
    np.random.seed(3)
    y    = np.random.uniform(0, 125, 1000)
    mean = y + np.random.normal(0, 12, 1000)
    std  = np.full(1000, 12.0)

    _, _, actual_cov = compute_ece(y, mean, std, n_bins=10)

    assert np.all(actual_cov >= 0) and np.all(actual_cov <= 1), \
        f"Coverage values must be in [0,1]. Got min={actual_cov.min():.3f}, max={actual_cov.max():.3f}"


# ── 2. NASA asymmetric score ──────────────────────────────────────

def test_nasa_score_late_penalty_greater_than_early():
    """
    NASA score penalizes late predictions (underestimating RUL)
    more heavily than early predictions (overestimating RUL).

    Same absolute error of d cycles, but:
      - early:  pred > true  (you said engine has more life than it does)
      - late:   pred < true  (you said engine has less life than it does)

    Wait — NASA score convention:
      d = pred - true
      d > 0 → late prediction (you predicted higher RUL than reality) → dangerous
      d < 0 → early prediction (you predicted lower RUL than reality) → safe
    """
    y_true = np.array([50.0])
    error  = 10.0

    # Late prediction: pred > true → d = +10
    y_late  = np.array([50.0 + error])
    # Early prediction: pred < true → d = -10
    y_early = np.array([50.0 - error])

    score_late  = nasa_score(y_true, y_late)
    score_early = nasa_score(y_true, y_early)

    assert score_late > score_early, \
        f"Late prediction score ({score_late:.2f}) should be > " \
        f"early prediction score ({score_early:.2f}) for same absolute error"

def test_nasa_score_zero_error_is_zero():
    """Perfect predictions (d=0) should give score = 0."""
    y_true = np.array([10.0, 30.0, 70.0, 100.0])
    y_pred = y_true.copy()

    score = nasa_score(y_true, y_pred)

    assert abs(score) < 1e-9, \
        f"Perfect predictions should score 0, got {score:.6f}"

def test_nasa_score_positive():
    """NASA score must always be >= 0 (it's a sum of exp terms minus 1 each)."""
    np.random.seed(7)
    y_true = np.random.uniform(10, 100, 100)
    y_pred = y_true + np.random.normal(0, 15, 100)

    score = nasa_score(y_true, y_pred)
    assert score >= 0, f"NASA score must be non-negative, got {score}"

def test_nasa_score_larger_errors_give_larger_score():
    """
    Larger prediction errors must give a larger NASA score
    (worse = higher score, since lower is better).
    """
    y_true = np.array([50.0, 50.0, 50.0])

    score_small = nasa_score(y_true, y_true + 5)    # small late error
    score_large = nasa_score(y_true, y_true + 20)   # large late error

    assert score_large > score_small, \
        "Larger errors must give a higher (worse) NASA score"

def test_nasa_score_asymmetry_ratio():
    """
    Quantify the asymmetry: a late error of d should score higher
    than an early error of d. Verify the ratio is consistent with
    the formula: exp(d/10)-1 vs exp(d/13)-1.
    """
    d = 20.0
    y_true = np.array([50.0])

    score_late  = nasa_score(y_true, y_true + d)  # d > 0 → dangerous
    score_early = nasa_score(y_true, y_true - d)  # d < 0 → safe

    # By formula: late = exp(20/10)-1 ≈ 6.39, early = exp(20/13)-1 ≈ 3.60
    expected_late  = np.exp(d / 10) - 1
    expected_early = np.exp(d / 13) - 1

    assert abs(score_late  - expected_late)  < 1e-6, \
        f"Late formula wrong: expected {expected_late:.4f}, got {score_late:.4f}"
    assert abs(score_early - expected_early) < 1e-6, \
        f"Early formula wrong: expected {expected_early:.4f}, got {score_early:.4f}"

def test_nasa_score_sum_over_fleet():
    """
    NASA score for a fleet is the sum of individual engine scores.
    Verify linearity: score([A, B]) == score([A]) + score([B]).
    """
    y_true = np.array([40.0, 80.0])
    y_pred = np.array([45.0, 70.0])

    score_combined = nasa_score(y_true, y_pred)
    score_engine1  = nasa_score(np.array([40.0]), np.array([45.0]))
    score_engine2  = nasa_score(np.array([80.0]), np.array([70.0]))

    assert abs(score_combined - (score_engine1 + score_engine2)) < 1e-9, \
        "NASA score must be additive across engines"


# ── 3. Sharpness tests ────────────────────────────────────────────

def test_sharpness_returns_mean_and_widths():
    """compute_sharpness must return (mean_width, per_engine_widths)."""
    std_preds = np.array([5.0, 10.0, 15.0])
    mean_width, widths = compute_sharpness(std_preds)

    assert isinstance(mean_width, float), "mean_width must be a float"
    assert widths.shape == (3,), f"widths shape wrong: {widths.shape}"

def test_sharpness_width_scales_with_std():
    """Larger std must give wider CI (sharpness = CI width, smaller = better)."""
    std_narrow = np.full(100, 2.0)
    std_wide   = np.full(100, 20.0)

    mean_narrow, _ = compute_sharpness(std_narrow)
    mean_wide,   _ = compute_sharpness(std_wide)

    assert mean_wide > mean_narrow, \
        "Wider std must produce wider CI"

def test_sharpness_mean_equals_average_of_widths():
    """Returned mean_width must equal the mean of per-engine widths."""
    std_preds = np.random.uniform(3, 15, 200)
    mean_width, widths = compute_sharpness(std_preds)

    assert abs(mean_width - widths.mean()) < 1e-6, \
        "mean_width must equal widths.mean()"
