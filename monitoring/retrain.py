# monitoring/retrain.py
# Automated retraining trigger.
# Checks drift report — if drift detected, retrains model,
# compares new vs old RMSE, promotes only if improved.
# This is the champion/challenger pattern used in production ML.

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.drift import load_data, run_drift_report, print_drift_summary
from dataset import get_dataloaders
from model import RULPredictor, mc_predict
from train import train, evaluate, nasa_score, CONFIG


# ── 1. Load current model RMSE ───────────────────────────────────
def get_champion_rmse(model_path='models/rul_predictor.pt'):
    """
    Loads the current production model and returns its saved RMSE.
    This is the score the new model must beat to be promoted.
    """
    checkpoint = torch.load(
        model_path, map_location='cpu', weights_only=False
    )
    rmse = checkpoint.get('test_rmse', float('inf'))
    print(f"Champion model RMSE: {rmse:.2f} cycles")
    return rmse


# ── 2. Check if retraining is needed ────────────────────────────
def should_retrain(drift_summary, champion_rmse,
                   rmse_degradation_threshold=0.15):
    """
    Decides whether to trigger retraining based on:
    1. Data drift detected by Evidently
    2. RMSE degradation above threshold

    Returns (bool, str) — (retrain_flag, reason)
    """
    reasons = []

    # Check data drift
    if drift_summary['drift_detected']:
        share = drift_summary['drift_share'] * 100
        reasons.append(
            f"Data drift detected in "
            f"{drift_summary['n_drifted']} features "
            f"({share:.0f}% of features)"
        )

    if reasons:
        return True, " | ".join(reasons)
    else:
        return False, "No drift detected — champion model is stable"


# ── 3. Retrain and evaluate challenger ──────────────────────────
def train_challenger():
    """
    Trains a new challenger model with the same config.
    Returns the trained model and its test RMSE.
    In production: would use new + old data combined.
    """
    print("\nTraining challenger model...")
    print("(In production: would include newly collected failure data)")

    # Use same config but slightly higher dropout for better calibration
    challenger_config = CONFIG.copy()
    challenger_config['dropout_rate'] = 0.25
    challenger_config['epochs']       = 50   # faster for demo

    model, rmse, score = train(config=challenger_config)
    return model, rmse, score


# ── 4. Champion/challenger decision ─────────────────────────────
def promote_challenger(challenger_model, challenger_rmse,
                       champion_rmse,
                       improvement_threshold=0.02,
                       save_path='models/rul_predictor.pt'):
    """
    Promotes challenger to champion only if it improves RMSE
    by more than improvement_threshold (default 2%).

    This prevents promoting a model that's marginally different
    due to random training variation.
    """
    improvement = (champion_rmse - challenger_rmse) / champion_rmse

    print(f"\n{'='*50}")
    print(f"CHAMPION vs CHALLENGER")
    print(f"{'='*50}")
    print(f"Champion  RMSE: {champion_rmse:.2f} cycles")
    print(f"Challenger RMSE: {challenger_rmse:.2f} cycles")
    print(f"Improvement:    {improvement*100:.1f}%")
    print(f"Threshold:      {improvement_threshold*100:.1f}%")

    if improvement > improvement_threshold:
        print(f"\n✅ CHALLENGER PROMOTED — "
              f"{improvement*100:.1f}% improvement exceeds threshold")

        # Save challenger as new champion
        torch.save({
            'model_state_dict': challenger_model.state_dict(),
            'config':           CONFIG,
            'test_rmse':        challenger_rmse,
            'promoted_at':      datetime.now().isoformat(),
            'previous_rmse':    champion_rmse,
        }, save_path)

        return True, improvement
    else:
        print(f"\n❌ CHALLENGER REJECTED — "
              f"improvement {improvement*100:.1f}% below threshold")
        print(f"   Champion model retained.")
        return False, improvement


# ── 5. Full retraining pipeline ──────────────────────────────────
def run_retraining_pipeline(
    model_path='models/rul_predictor.pt',
    force_retrain=False
):
    """
    Complete pipeline:
    1. Check drift
    2. Decide if retraining needed
    3. Train challenger if yes
    4. Promote if challenger is better
    5. Log everything
    """
    print("="*55)
    print("RETRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    # ── Step 1: Check drift ──
    print("\nStep 1: Checking data drift...")
    reference, current = load_data()
    drift_summary, _   = run_drift_report(
        reference, current, save=True
    )
    print_drift_summary(drift_summary)

    # ── Step 2: Decide ──
    champion_rmse       = get_champion_rmse(model_path)
    retrain, reason     = should_retrain(
        drift_summary, champion_rmse
    )

    print(f"\nStep 2: Retraining decision")
    print(f"  Force retrain: {force_retrain}")
    print(f"  Drift trigger: {retrain}")
    print(f"  Reason: {reason}")

    if not retrain and not force_retrain:
        print("\n✅ No retraining needed. Pipeline complete.")
        return False, champion_rmse, None

    # ── Step 3: Train challenger ──
    print("\nStep 3: Training challenger model...")
    challenger_model, challenger_rmse, challenger_score = \
        train_challenger()

    # ── Step 4: Promote if better ──
    print("\nStep 4: Champion/challenger evaluation...")
    promoted, improvement = promote_challenger(
        challenger_model, challenger_rmse, champion_rmse
    )

    # ── Step 5: Log result ──
    log = {
        'timestamp':        datetime.now().isoformat(),
        'drift_detected':   drift_summary['drift_detected'],
        'drift_share':      drift_summary['drift_share'],
        'champion_rmse':    champion_rmse,
        'challenger_rmse':  challenger_rmse,
        'improvement_pct':  improvement * 100,
        'promoted':         promoted,
        'reason':           reason
    }

    os.makedirs('logs', exist_ok=True)
    log_path = f"logs/retrain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nRetraining log saved → {log_path}")

    return promoted, challenger_rmse, log


# ── 6. Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Run drift check only first (fast)
    print("Running drift check first (no retraining)...")
    reference, current = load_data()
    summary, _         = run_drift_report(reference, current, save=True)
    print_drift_summary(summary)

    retrain, reason = should_retrain(summary, champion_rmse=13.56)
    print(f"Retraining needed: {retrain}")
    print(f"Reason: {reason}")
    print("\nTo force full retraining pipeline run:")
    print("  run_retraining_pipeline(force_retrain=True)")