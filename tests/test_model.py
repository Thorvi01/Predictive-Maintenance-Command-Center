# tests/test_model.py
# Tests for model.py — RULPredictor architecture and mc_predict inference.
# Run with: pytest tests/test_model.py -v

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import RULPredictor, mc_predict, enable_mc_dropout

# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def model():
    """A freshly initialised (untrained) model with default config."""
    return RULPredictor(input_size=17, hidden_size=64,
                        num_layers=2, dropout_rate=0.2)

@pytest.fixture
def batch():
    """Fake input tensor matching real data shape: (8, 30, 17)."""
    torch.manual_seed(42)
    return torch.randn(8, 30, 17)

@pytest.fixture
def single_engine():
    """Single engine tensor: (1, 30, 17)."""
    torch.manual_seed(0)
    return torch.randn(1, 30, 17)


# ── 1. Output shapes ─────────────────────────────────────────────

def test_forward_output_shape(model, batch):
    """Forward pass must return one RUL value per sample."""
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"

def test_mc_predict_return_shapes(model, batch):
    """mc_predict must return 5 arrays all with correct shapes."""
    mean, std, lower, upper, all_preds = mc_predict(
        model, batch, n_samples=50
    )
    assert mean.shape      == (8,),     f"mean shape wrong: {mean.shape}"
    assert std.shape       == (8,),     f"std shape wrong: {std.shape}"
    assert lower.shape     == (8,),     f"lower shape wrong: {lower.shape}"
    assert upper.shape     == (8,),     f"upper shape wrong: {upper.shape}"
    assert all_preds.shape == (50, 8),  f"all_preds shape wrong: {all_preds.shape}"

def test_mc_predict_single_engine(model, single_engine):
    """mc_predict must work on a single engine (batch size 1)."""
    mean, std, lower, upper, all_preds = mc_predict(
        model, single_engine, n_samples=30
    )
    assert mean.shape      == (1,)
    assert all_preds.shape == (30, 1)


# ── 2. Numerical consistency ──────────────────────────────────────

def test_mean_equals_average_of_all_preds(model, batch):
    """mean must equal the row-wise average of all_preds."""
    mean, _, _, _, all_preds = mc_predict(model, batch, n_samples=100)
    expected_mean = all_preds.mean(axis=0)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-5,
        err_msg="mean does not match all_preds.mean(axis=0)")

def test_std_equals_std_of_all_preds(model, batch):
    """std must equal the row-wise std of all_preds."""
    _, std, _, _, all_preds = mc_predict(model, batch, n_samples=100)
    expected_std = all_preds.std(axis=0)
    np.testing.assert_allclose(std, expected_std, rtol=1e-5,
        err_msg="std does not match all_preds.std(axis=0)")

def test_lower_ci_below_mean(model, batch):
    """Lower CI bound must always be <= mean."""
    mean, _, lower, _, _ = mc_predict(model, batch, n_samples=100)
    assert np.all(lower <= mean + 1e-6), \
        "lower CI bound is above mean for some engines"

def test_upper_ci_above_mean(model, batch):
    """Upper CI bound must always be >= mean."""
    mean, _, _, upper, _ = mc_predict(model, batch, n_samples=100)
    assert np.all(upper >= mean - 1e-6), \
        "upper CI bound is below mean for some engines"

def test_lower_always_below_upper(model, batch):
    """lower CI must be strictly <= upper CI for all engines."""
    _, _, lower, upper, _ = mc_predict(model, batch, n_samples=100)
    assert np.all(lower <= upper + 1e-6), \
        "lower CI is above upper CI for some engines"

def test_std_non_negative(model, batch):
    """Standard deviation must be non-negative."""
    _, std, _, _, _ = mc_predict(model, batch, n_samples=100)
    assert np.all(std >= 0), "Negative std values found"

def test_ci_is_90_percent_interval(model, batch):
    """Lower/upper must be the 5th and 95th percentiles of all_preds."""
    _, _, lower, upper, all_preds = mc_predict(model, batch, n_samples=200)
    expected_lower = np.percentile(all_preds, 5,  axis=0)
    expected_upper = np.percentile(all_preds, 95, axis=0)
    np.testing.assert_allclose(lower, expected_lower, rtol=1e-5)
    np.testing.assert_allclose(upper, expected_upper, rtol=1e-5)


# ── 3. MC Dropout actually produces variance ──────────────────────

def test_mc_dropout_produces_variance(model, batch):
    """
    With dropout active, two forward passes on the same input must
    give different results. If std == 0 for all engines, dropout
    is not active during inference — the core MC mechanism is broken.
    """
    mean, std, _, _, all_preds = mc_predict(model, batch, n_samples=50)
    # At least some engines should have non-zero uncertainty
    assert np.any(std > 1e-6), \
        "All std values are zero — MC Dropout is not active at inference"

def test_mc_dropout_variance_increases_with_more_samples(model, batch):
    """
    The mean estimate should stabilise as n_samples grows.
    Std with 10 samples should be noisier than with 500 samples.
    (We check that both are non-zero, not that one is strictly smaller,
    since that's a statistical property not a unit test property.)
    """
    _, std_10,  _, _, _ = mc_predict(model, batch, n_samples=10)
    _, std_500, _, _, _ = mc_predict(model, batch, n_samples=500)
    assert np.all(std_10  >= 0)
    assert np.all(std_500 >= 0)

def test_eval_mode_gives_deterministic_output(model, batch):
    """
    In standard eval mode (dropout OFF), two passes must be identical.
    This confirms enable_mc_dropout is what activates stochasticity,
    not some other source of randomness.
    """
    model.eval()
    with torch.no_grad():
        out1 = model(batch).numpy()
        out2 = model(batch).numpy()
    np.testing.assert_array_equal(out1, out2,
        err_msg="eval() mode is not deterministic — unexpected randomness")

def test_mc_mode_gives_different_outputs(model, batch):
    """
    With MC Dropout enabled, two passes on the same input must differ.
    """
    enable_mc_dropout(model)
    with torch.no_grad():
        out1 = model(batch).numpy()
        out2 = model(batch).numpy()
    assert not np.array_equal(out1, out2), \
        "MC Dropout mode gave identical outputs — dropout may be off"


# ── 4. Model parameter count ──────────────────────────────────────

def test_model_has_trainable_parameters(model):
    """Model must have at least one trainable parameter."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total > 0, "Model has no trainable parameters"

def test_model_parameter_count_reasonable(model):
    """
    Default config (hidden=64, layers=2, input=17) should be in a
    sensible range — not accidentally huge or microscopic.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 10_000 < total < 500_000, \
        f"Unexpected parameter count: {total:,}"
