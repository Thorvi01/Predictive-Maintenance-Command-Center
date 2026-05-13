# monitoring/drift.py
# Detects when incoming sensor data drifts from training distribution.
# Uses Evidently AI to compare reference (train) vs current (test) data.
# Triggers retraining alert when drift is detected.

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Configuration ────────────────────────────────────────────────
DRIFT_THRESHOLD  = 0.1
REPORT_DIR       = 'logs/drift_reports'
REFERENCE_PATH   = 'data/processed/train_FD001.csv'
CURRENT_PATH     = 'data/processed/test_FD001.csv'

SENSOR_COLS = [
    'setting_1', 'setting_2', 'setting_3',
    's2', 's3', 's4', 's7', 's8', 's9',
    's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'
]


# ── 1. Load data ─────────────────────────────────────────────────
def load_data(reference_path=REFERENCE_PATH,
              current_path=CURRENT_PATH):
    """
    Loads reference (training) and current (production) sensor data.
    In a real system, current_path would be live incoming sensor data.
    For this project we simulate it using the test set.
    """
    reference = pd.read_csv(reference_path)[SENSOR_COLS]
    current   = pd.read_csv(current_path)[SENSOR_COLS]

    reference = reference.sample(min(2000, len(reference)), random_state=42)
    current   = current.sample(min(500,  len(current)),  random_state=42)

    return reference, current


# ── 2. Extract feature drifts from Evidently report dict ─────────
def _extract_feature_drifts(report_dict):
    """
    Evidently changes its internal dict structure between versions.
    This function tries multiple known paths to extract per-feature
    drift scores robustly.
    """
    feature_drifts = {}

    for metric in report_dict.get('metrics', []):
        result = metric.get('result', {})

        # Path 1: drift_by_columns (most common)
        columns_data = result.get('drift_by_columns', {})

        # Path 2: some versions nest under 'columns'
        if not columns_data:
            columns_data = result.get('columns', {})

        # Path 3: list format
        if not columns_data and isinstance(result.get('columns'), list):
            for item in result['columns']:
                col = item.get('column_name', '')
                if col:
                    columns_data[col] = item

        for col, info in columns_data.items():
            if isinstance(info, dict) and col in SENSOR_COLS:
                feature_drifts[col] = {
                    'drift_score':    float(info.get('drift_score',
                                           info.get('p_value', 0))),
                    'drift_detected': bool(info.get('drift_detected', False)),
                    'stattest':       str(info.get('stattest_name',
                                         info.get('stattest', 'unknown')))
                }

        if feature_drifts:
            break   # found data, stop searching

    return feature_drifts


# ── 3. Fallback: scipy KS test if Evidently extraction fails ─────
def _scipy_drift_fallback(reference, current):
    """
    If Evidently doesn't return per-feature data, fall back to
    scipy Kolmogorov-Smirnov test for each sensor column.
    KS test p-value < 0.05 → distributions are significantly different.
    """
    from scipy import stats

    feature_drifts = {}
    for col in SENSOR_COLS:
        stat, p_value = stats.ks_2samp(
            reference[col].dropna(),
            current[col].dropna()
        )
        feature_drifts[col] = {
            'drift_score':    round(float(stat), 4),
            'drift_detected': bool(p_value < 0.05),
            'stattest':       'kolmogorov-smirnov'
        }
    return feature_drifts


# ── 4. Run drift report ──────────────────────────────────────────
def run_drift_report(reference, current, save=True):
    """
    Uses Evidently AI DataDriftPreset to compare distributions.
    Falls back to scipy KS test if Evidently extraction fails.
    Returns drift summary dict and Evidently report object.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Run Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report_dict = report.as_dict()

    # Try to extract feature-level drift scores
    feature_drifts = _extract_feature_drifts(report_dict)

    # Fallback to scipy if Evidently extraction returned nothing
    if not feature_drifts:
        print("  (Using scipy KS test fallback for per-feature scores)")
        feature_drifts = _scipy_drift_fallback(reference, current)

    n_drifted   = sum(1 for v in feature_drifts.values()
                      if v['drift_detected'])
    drift_share = n_drifted / len(feature_drifts) if feature_drifts else 0
    drift_flag  = drift_share > 0.3  # flag if >30% features drifted

    summary = {
        'timestamp':      datetime.now().isoformat(),
        'n_features':     len(feature_drifts),
        'n_drifted':      n_drifted,
        'drift_share':    drift_share,
        'drift_detected': drift_flag,
        'feature_drifts': feature_drifts
    }

    if save:
        ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = os.path.join(REPORT_DIR, f'drift_report_{ts}.html')
        json_path = os.path.join(REPORT_DIR, f'drift_summary_{ts}.json')
        report.save_html(html_path)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved HTML report → {html_path}")
        print(f"Saved JSON summary → {json_path}")

    return summary, report


# ── 5. Print drift summary ───────────────────────────────────────
def print_drift_summary(summary):
    """Prints a clean drift report to console."""

    flag = "🔴 DRIFT DETECTED" if summary['drift_detected'] \
           else "✅ NO DRIFT"

    print(f"\n{'='*50}")
    print(f"DATA DRIFT REPORT — {summary['timestamp'][:19]}")
    print(f"{'='*50}")
    print(f"Overall Status:   {flag}")
    print(f"Features drifted: {summary['n_drifted']} / "
          f"{summary['n_features']} "
          f"({summary['drift_share']*100:.0f}%)")

    if summary['feature_drifts']:
        print(f"\nPer-feature drift scores:")
        print(f"{'Feature':<15} {'Score':>8} {'Drifted':>10}")
        print("-" * 36)
        for feat, info in sorted(
            summary['feature_drifts'].items(),
            key=lambda x: x[1]['drift_score'],
            reverse=True
        ):
            drifted = "⚠ YES" if info['drift_detected'] else "  no"
            print(f"{feat:<15} {info['drift_score']:>8.4f} {drifted:>10}")

    if summary['drift_detected']:
        print(f"\n⚠  ACTION REQUIRED: Significant data drift detected.")
        print(f"   Recommendation: Trigger model retraining pipeline.")
    else:
        print(f"\n✓  Model input distribution is stable.")
        print(f"   No retraining required at this time.")
    print(f"{'='*50}\n")


# ── 6. Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Running drift detection...")
    print(f"Reference: training data ({REFERENCE_PATH})")
    print(f"Current:   test data ({CURRENT_PATH})")
    print("(In production: current = live incoming sensor stream)\n")

    reference, current = load_data()
    print(f"Reference samples: {len(reference)}")
    print(f"Current samples:   {len(current)}")

    summary, report = run_drift_report(reference, current)
    print_drift_summary(summary)