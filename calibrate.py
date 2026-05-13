# calibrate.py
# Temperature scaling for MC Dropout uncertainty calibration.
#
# Problem: mc_predict gives ECE = 0.20, 90% CI coverage = 54%.
#          The model is overconfident — its stated uncertainty is too narrow.
#
# Fix: learn a scalar temperature T on the validation set.
#      Multiply every predicted std by T before computing CIs.
#      T > 1 widens the CIs → coverage improves.
#      T is fit by minimising Negative Log-Likelihood on validation data.
#
# Usage:
#   python calibrate.py
#   → saves temperature to models/temperature.pt
#   → reports ECE before and after

import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import RULPredictor, mc_predict, enable_mc_dropout
from dataset import get_dataloaders
from evaluate import compute_ece, compute_sharpness


# ── 1. Temperature Scaler ────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Learns a single scalar T that multiplies predicted std.

    Before calibration: CI = mean ± z * std
    After calibration:  CI = mean ± z * (T * std)

    T > 1 → wider CIs → fixes overconfidence
    T < 1 → narrower CIs → would fix underconfidence
    T = 1 → no change

    This is the simplest possible post-hoc calibration method.
    It has one parameter, can't overfit, and doesn't touch the LSTM weights.
    """

    def __init__(self):
        super().__init__()
        # Start at 1.0 (no change). Optimiser will move it.
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def scale(self, std: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to a numpy std array."""
        T = self.temperature.item()
        return std * T

    def forward(self, std_tensor: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to a torch std tensor."""
        return std_tensor * self.temperature


# ── 2. Gaussian NLL loss ─────────────────────────────────────────

def gaussian_nll(y_true: torch.Tensor,
                 mean:   torch.Tensor,
                 std:    torch.Tensor) -> torch.Tensor:
    """
    Negative Log-Likelihood under a Gaussian distribution.

    NLL = 0.5 * [ log(2π σ²) + (y - μ)² / σ² ]

    Minimising NLL jointly optimises accuracy (via the residual term)
    and calibration (via the log σ term). If T is too small, σ is
    too small, so (y-μ)²/σ² blows up. If T is too large, log(σ²)
    grows. The optimum is where both are balanced → well-calibrated.

    PyTorch has nn.GaussianNLLLoss but we write it explicitly so the
    maths is visible.
    """
    var = std ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (y_true - mean) ** 2 / var)
    return nll.mean()


# ── 3. Fit temperature on validation set ─────────────────────────

def fit_temperature(model:      RULPredictor,
                    val_loader,
                    device:     str = 'cpu',
                    n_mc:       int = 100,
                    lr:         float = 0.01,
                    max_steps:  int = 200,
                    patience:   int = 20) -> TemperatureScaler:
    """
    Optimises T to minimise Gaussian NLL on the validation set.

    Args:
        model:      trained RULPredictor (weights frozen)
        val_loader: DataLoader for validation split
        device:     'cpu' or 'cuda'
        n_mc:       MC Dropout samples per engine
        lr:         learning rate for T (Adam)
        max_steps:  maximum optimisation steps
        patience:   stop if NLL doesn't improve for this many steps

    Returns:
        TemperatureScaler with fitted temperature
    """

    print("Fitting temperature scaler on validation set...")
    print(f"  MC samples per engine: {n_mc}")

    # ── Collect all validation predictions ──────────────────────
    # We run MC inference once and cache the results.
    # No need to re-run for every optimisation step.
    print("  Running MC inference on validation set...")

    X_val = torch.cat([X for X, _ in val_loader]).to(device)
    y_val = torch.cat([y for _, y in val_loader]).numpy()

    mean_np, std_np, _, _, _ = mc_predict(
        model, X_val, n_samples=n_mc, device=device
    )

    # Apply temperature scaling for calibrated uncertainty
    std_np   = std_np * T
    lower_np = mean_np - 1.645 * std_np
    upper_np = mean_np + 1.645 * std_np

    mean_np = np.clip(mean_np, 0, None)

    # Convert to tensors for gradient-based optimisation
    mean_t = torch.tensor(mean_np, dtype=torch.float32)
    std_t  = torch.tensor(std_np,  dtype=torch.float32)
    y_t    = torch.tensor(y_val,   dtype=torch.float32)

    # ── Optimise temperature ─────────────────────────────────────
    scaler    = TemperatureScaler()
    optimiser = torch.optim.Adam(scaler.parameters(), lr=lr)

    best_nll   = float('inf')
    best_T     = 1.0
    patience_c = 0

    print(f"\n  {'Step':>6}  {'NLL':>10}  {'T':>8}")
    print(f"  {'-'*28}")

    for step in range(max_steps):
        optimiser.zero_grad()

        # Scale std by current temperature
        scaled_std = scaler(std_t)

        # Clamp to prevent log(0) — T could theoretically go negative
        scaled_std = torch.clamp(scaled_std, min=1e-6)

        loss = gaussian_nll(y_t, mean_t, scaled_std)
        loss.backward()
        optimiser.step()

        # Clamp T to positive values — negative temperature is meaningless
        with torch.no_grad():
            scaler.temperature.clamp_(min=0.01, max=20.0)

        current_nll = loss.item()
        current_T   = scaler.temperature.item()

        if step % 20 == 0:
            print(f"  {step:>6}  {current_nll:>10.4f}  {current_T:>8.4f}")

        # Early stopping
        if current_nll < best_nll - 1e-6:
            best_nll   = current_nll
            best_T     = current_T
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                print(f"  Early stopping at step {step}")
                break

    # Restore best T
    with torch.no_grad():
        scaler.temperature.fill_(best_T)

    print(f"\n  ✓ Fitted temperature T = {best_T:.4f}")
    if best_T > 1.0:
        print(f"  → T > 1: model was overconfident, CIs widened by {best_T:.2f}×")
    else:
        print(f"  → T < 1: model was underconfident, CIs narrowed by {best_T:.2f}×")

    return scaler


# ── 4. Evaluate calibration before/after ─────────────────────────

def evaluate_calibration(y_true:     np.ndarray,
                          mean_preds: np.ndarray,
                          std_preds:  np.ndarray,
                          label:      str = "") -> dict:
    """
    Computes ECE, CI coverage, and sharpness. Prints a summary.
    Returns a dict of metrics.
    """
    from scipy import stats

    ece, conf_levels, actual_cov = compute_ece(y_true, mean_preds, std_preds)
    mean_sharpness, _            = compute_sharpness(std_preds)

    z90     = stats.norm.ppf(0.95)
    lower   = mean_preds - z90 * std_preds
    upper   = mean_preds + z90 * std_preds
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100

    rmse = np.sqrt(np.mean((mean_preds - y_true) ** 2))

    tag = f" [{label}]" if label else ""
    print(f"\n  Calibration{tag}:")
    print(f"    ECE:            {ece:.4f}  (target < 0.05)")
    print(f"    90% CI Coverage:{coverage:.1f}%  (target ≈ 90%)")
    print(f"    Mean CI Width:  {mean_sharpness:.2f} cycles")
    print(f"    RMSE:           {rmse:.2f} cycles")

    return {
        'ece': ece, 'coverage': coverage,
        'sharpness': mean_sharpness, 'rmse': rmse
    }


# ── 5. Main: fit, evaluate, save ─────────────────────────────────

def run_calibration(model_path: str = 'models/rul_predictor.pt',
                    out_path:   str = 'models/temperature.pt',
                    n_mc:       int = 100):
    """
    Full calibration pipeline:
    1. Load trained model
    2. Run MC inference on val set, compute baseline ECE
    3. Fit temperature T on val set
    4. Re-evaluate ECE with scaled std
    5. Save T to disk
    """

    device = 'cpu'

    # ── Load model ───────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location=device,
                             weights_only=False)
    config = checkpoint['config']
    model  = RULPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded (saved RMSE: {checkpoint['test_rmse']:.2f})")

    # ── Load data ────────────────────────────────────────────────
    _, val_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        window_size=config['window_size']
    )

    # ── Baseline: MC inference on test set ───────────────────────
    print("\nBaseline MC inference on test set...")
    X_test = torch.cat([X for X, _ in test_loader]).to(device)
    y_test = torch.cat([y for _, y in test_loader]).numpy()

    mean_raw, std_raw, _, _, _ = mc_predict(
        model, X_test, n_samples=n_mc, device=device
    )

    # Apply temperature scaling for calibrated uncertainty
    std_raw   = std_raw * T
    lower_raw = mean_raw - 1.645 * std_raw
    upper_raw = mean_raw + 1.645 * std_raw

    mean_raw = np.clip(mean_raw, 0, None)

    baseline = evaluate_calibration(y_test, mean_raw, std_raw,
                                     label="before calibration")

    # ── Fit temperature on validation set ────────────────────────
    scaler = fit_temperature(model, val_loader, device=device, n_mc=n_mc)
    T      = scaler.temperature.item()

    # ── Apply temperature to test predictions ────────────────────
    std_calibrated = scaler.scale(std_raw)

    calibrated = evaluate_calibration(y_test, mean_raw, std_calibrated,
                                       label="after calibration")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("CALIBRATION SUMMARY")
    print("="*50)
    print(f"  Temperature T:      {T:.4f}")
    print(f"  ECE:                {baseline['ece']:.4f}  →  {calibrated['ece']:.4f}",
          "✓" if calibrated['ece'] < 0.05 else "⚠ still above 0.05")
    print(f"  90% CI Coverage:    {baseline['coverage']:.1f}%  →  {calibrated['coverage']:.1f}%",
          "✓" if abs(calibrated['coverage'] - 90) < 10 else "⚠ still far from 90%")
    print(f"  RMSE (unchanged):   {calibrated['rmse']:.2f} cycles")
    print(f"  CI Width:           {baseline['sharpness']:.1f}  →  {calibrated['sharpness']:.1f} cycles")

    # ── Save temperature ──────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        'temperature': T,
        'baseline_ece': baseline['ece'],
        'calibrated_ece': calibrated['ece'],
        'baseline_coverage': baseline['coverage'],
        'calibrated_coverage': calibrated['coverage'],
    }, out_path)
    print(f"\n  ✓ Temperature saved → {out_path}")
    print(f"  Run 'python evaluate.py' to regenerate calibration report with T={T:.4f}")

    return scaler


# ── 6. Helper: load saved temperature ────────────────────────────

def load_temperature(path: str = 'models/temperature.pt') -> float:
    """
    Loads the saved temperature scalar.
    Use this in mc_predict wrappers to apply calibration at inference time.

    Returns T as a float. Returns 1.0 (no-op) if file not found.
    """
    if not os.path.exists(path):
        return 1.0
    data = torch.load(path, map_location='cpu', weights_only=False)
    return float(data['temperature'])


if __name__ == '__main__':
    run_calibration()
