# evaluate.py
# Calibration evaluation for MC Dropout RUL model.
# Produces: reliability diagram, ECE score, sharpness, prediction scatter plot.
# These plots are your portfolio differentiator — visualizing uncertainty quality.

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from dataset import get_dataloaders
from model  import RULPredictor, mc_predict


# ── 1. Load trained model ────────────────────────────────────────
def load_model(model_path='models/rul_predictor.pt', device='cpu'):
    """
    Loads the saved model weights and config from disk.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config     = checkpoint['config']

    model = RULPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded — saved RMSE: {checkpoint['test_rmse']:.2f}")
    return model, config


# ── 2. ECE score ─────────────────────────────────────────────────
def compute_ece(y_true, mean_preds, std_preds,
                confidence_levels=None, n_bins=10):
    """
    Expected Calibration Error for regression with Gaussian uncertainty.

    For each confidence level p (e.g. 0.9 = 90% CI):
      - Compute the interval [mean ± z*std] where z is the Gaussian z-score
      - Count what fraction of true values fall inside → actual coverage
      - Compare to stated confidence p → gap

    ECE = weighted average of |actual_coverage - stated_confidence|

    Args:
        y_true:      true RUL values, shape (N,)
        mean_preds:  MC mean predictions, shape (N,)
        std_preds:   MC std predictions, shape (N,)

    Returns:
        ece:               scalar ECE score (lower = better)
        confidence_levels: list of confidence levels used
        actual_coverages:  list of actual coverages at each level
    """
    from scipy import stats

    if confidence_levels is None:
        # 10 evenly spaced confidence levels from 10% to 90%
        confidence_levels = np.linspace(0.1, 0.9, n_bins)

    actual_coverages = []

    for p in confidence_levels:
        # z-score for this confidence level (two-sided interval)
        # e.g. p=0.9 → z=1.645 → interval is mean ± 1.645*std
        z = stats.norm.ppf((1 + p) / 2)

        lower = mean_preds - z * std_preds
        upper = mean_preds + z * std_preds

        # Fraction of true values inside this interval
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        actual_coverages.append(coverage)

    actual_coverages = np.array(actual_coverages)

    # ECE = average absolute gap, weighted by bin width
    ece = np.mean(np.abs(actual_coverages - confidence_levels))

    return ece, confidence_levels, actual_coverages


# ── 3. Sharpness ─────────────────────────────────────────────────
def compute_sharpness(std_preds, confidence_level=0.9):
    """
    Sharpness = average width of the 90% confidence interval.

    Narrower = sharper = more useful predictions.
    A model that always says CI = [0, 125] is calibrated but not sharp.

    Returns mean CI width in RUL cycles.
    """
    from scipy import stats
    z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_widths = 2 * z * std_preds   # width = upper - lower = 2*z*std
    return ci_widths.mean(), ci_widths


# ── 4. All plots ─────────────────────────────────────────────────
def plot_calibration_report(y_true, mean_preds, std_preds,
                             save_dir='logs'):
    """
    Creates a 2x2 figure with 4 diagnostic plots:
      [0,0] Reliability diagram
      [0,1] Confidence interval width distribution (sharpness)
      [1,0] Prediction vs truth scatter with error bars
      [1,1] Residuals vs uncertainty
    """
    os.makedirs(save_dir, exist_ok=True)

    from scipy import stats

    # Compute metrics
    ece, conf_levels, actual_cov = compute_ece(
        y_true, mean_preds, std_preds
    )
    mean_sharpness, ci_widths = compute_sharpness(std_preds)

    # 90% CI bounds for scatter plot
    z90   = stats.norm.ppf(0.95)   # 1.645
    lower = mean_preds - z90 * std_preds
    upper = mean_preds + z90 * std_preds
    coverage_90 = np.mean((y_true >= lower) & (y_true <= upper)) * 100

    print(f"\n{'='*45}")
    print(f"  ECE Score:        {ece:.4f}  (lower = better, 0 = perfect)")
    print(f"  Mean CI Width:    {mean_sharpness:.2f} cycles  (sharpness)")
    print(f"  90% CI Coverage:  {coverage_90:.1f}%  (ideal = 90%)")
    print(f"  Test RMSE:        "
          f"{np.sqrt(np.mean((mean_preds - y_true)**2)):.2f} cycles")
    print(f"{'='*45}\n")

    # ── Figure setup ──
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.38, wspace=0.32)

    ACCENT  = '#00C4FF'
    WARN    = '#FF6B6B'
    SUCCESS = '#00E676'
    GRID    = '#2a2a2a'
    TEXT    = '#E0E0E0'
    BG      = '#1a1a1a'

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TEXT, fontsize=12,
                     fontweight='bold', pad=10)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    # ── Plot 1: Reliability Diagram ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, f'Reliability Diagram  (ECE = {ece:.4f})')

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], '--', color='#555555',
             linewidth=1.5, label='Perfect calibration', zorder=1)

    # Actual calibration curve
    ax1.plot(conf_levels, actual_cov, 'o-',
             color=ACCENT, linewidth=2, markersize=7,
             label='MC Dropout model', zorder=3)

    # Fill area between ideal and actual
    ax1.fill_between(conf_levels, conf_levels, actual_cov,
                     alpha=0.15, color=WARN,
                     label='Calibration gap')

    ax1.set_xlabel('Stated Confidence Level')
    ax1.set_ylabel('Actual Coverage')
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8, facecolor='#222222',
               labelcolor=TEXT, framealpha=0.8)

    # Annotation explaining the gap
    ax1.annotate('Below diagonal =\noverconfident\n(CIs too narrow)',
                 xy=(0.6, 0.45), fontsize=8, color=WARN,
                 ha='center',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='#2a1a1a', alpha=0.8))

    # ── Plot 2: CI Width Distribution (Sharpness) ────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, f'CI Width Distribution  '
                  f'(Mean = {mean_sharpness:.1f} cycles)')

    ax2.hist(ci_widths, bins=20, color=ACCENT,
             alpha=0.75, edgecolor='#003344')
    ax2.axvline(mean_sharpness, color=WARN, linewidth=2,
                linestyle='--',
                label=f'Mean = {mean_sharpness:.1f} cycles')
    ax2.set_xlabel('90% CI Width (cycles)')
    ax2.set_ylabel('Count')
    ax2.legend(fontsize=8, facecolor='#222222',
               labelcolor=TEXT, framealpha=0.8)

    # ── Plot 3: Prediction vs Truth with Error Bars ───────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, f'Predicted vs True RUL  '
                  f'(Coverage = {coverage_90:.0f}%)')

    sort_idx = np.argsort(y_true)
    y_sorted    = y_true[sort_idx]
    mean_sorted = mean_preds[sort_idx]
    lower_sorted = lower[sort_idx]
    upper_sorted = upper[sort_idx]

    # Which engines are covered by CI
    covered = (y_sorted >= lower_sorted) & (y_sorted <= upper_sorted)

    # Error bars
    yerr = np.array([
        mean_sorted - lower_sorted,
        upper_sorted - mean_sorted
    ])
    ax3.errorbar(
        range(len(y_sorted)), mean_sorted,
        yerr=yerr,
        fmt='none', ecolor='#004455', alpha=0.4,
        capsize=0, linewidth=1
    )

    # Scatter — green if covered, red if not
    ax3.scatter(np.where(covered)[0],  mean_sorted[covered],
                color=SUCCESS, s=25, zorder=4,
                label=f'Inside 90% CI ({covered.sum()})')
    ax3.scatter(np.where(~covered)[0], mean_sorted[~covered],
                color=WARN, s=25, zorder=4,
                label=f'Outside 90% CI ({(~covered).sum()})')

    # True RUL line
    ax3.plot(range(len(y_sorted)), y_sorted,
             color='#FFCC00', linewidth=1.5,
             alpha=0.8, label='True RUL', zorder=3)

    ax3.set_xlabel('Engine (sorted by true RUL)')
    ax3.set_ylabel('RUL (cycles)')
    ax3.legend(fontsize=7, facecolor='#222222',
               labelcolor=TEXT, framealpha=0.8)

    # ── Plot 4: Residuals vs Uncertainty ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'Residuals vs Uncertainty')

    residuals = np.abs(mean_preds - y_true)

    ax4.scatter(std_preds, residuals,
                color=ACCENT, alpha=0.5, s=20, zorder=3)

    # Ideal line: residual should scale with std
    max_std = std_preds.max()
    ideal_x = np.linspace(0, max_std, 100)
    ax4.plot(ideal_x, ideal_x * z90, '--',
             color=WARN, linewidth=1.5,
             label=f'Ideal (residual = {z90:.2f}×std)')

    ax4.set_xlabel('Predicted Uncertainty (std, cycles)')
    ax4.set_ylabel('Absolute Residual |pred − true|')
    ax4.legend(fontsize=8, facecolor='#222222',
               labelcolor=TEXT, framealpha=0.8)

    ax4.annotate(
        'Points above line =\nuncertainty underestimated',
        xy=(std_preds.max() * 0.55, residuals.max() * 0.85),
        fontsize=8, color=WARN,
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='#2a1a1a', alpha=0.8)
    )

    # ── Main title ───────────────────────────────────────────────
    fig.suptitle(
        'MC Dropout Calibration Report — NASA C-MAPSS FD001',
        color=TEXT, fontsize=14, fontweight='bold', y=0.98
    )

    # ── Save ─────────────────────────────────────────────────────
    out_path = os.path.join(save_dir, 'calibration_report.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved → {out_path}")

    return ece, mean_sharpness, coverage_90


# ── 5. Main ──────────────────────────────────────────────────────
def evaluate_calibration(
    model_path='models/rul_predictor.pt',
    n_mc_samples=100
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, config = load_model(model_path, str(device))

    # Load test data
    _, _, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        window_size=config['window_size']
    )

    # Collect all test samples
    X_test = torch.cat([X for X, _ in test_loader])
    y_test = torch.cat([y for _, y in test_loader]).numpy()

    # MC Dropout inference
    print(f"Running {n_mc_samples} MC forward passes...")
    mean_preds, std_preds, lower, upper, all_preds = mc_predict(
        model, X_test, n_samples=n_mc_samples, device=str(device)
    )

    mean_preds = np.clip(mean_preds, 0, None)

    # Plot and compute metrics
    ece, sharpness, coverage = plot_calibration_report(
        y_test, mean_preds, std_preds
    )

    return ece, sharpness, coverage


if __name__ == '__main__':
    evaluate_calibration()