# tests/test_preprocessing.py
# Tests for data_preprocessing.py — RUL labeling, sensor dropping,
# normalization, and output shape correctness.
# Run with: pytest tests/test_preprocessing.py -v

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    add_rul,
    normalize,
    DROP_SENSORS,
    USEFUL_SENSORS,
    COLUMNS,
)
from dataset import build_windows, build_test_windows, FEATURE_COLS, WINDOW_SIZE


# ── Fixtures ──────────────────────────────────────────────────────

def make_engine_df(engine_id, n_cycles, sensor_val=0.5):
    """
    Creates a minimal DataFrame for one engine with n_cycles rows.
    All sensors set to sensor_val so we can control variance easily.
    """
    rows = []
    for cycle in range(1, n_cycles + 1):
        row = {'engine_id': engine_id, 'cycle': cycle}
        for col in COLUMNS:
            if col not in ('engine_id', 'cycle'):
                row[col] = sensor_val
        rows.append(row)
    return pd.DataFrame(rows)

def make_multi_engine_df(engine_specs, sensor_val=0.5):
    """
    engine_specs: list of (engine_id, n_cycles) tuples.
    Returns a combined DataFrame with all engines.
    """
    frames = [make_engine_df(eid, n, sensor_val) for eid, n in engine_specs]
    return pd.concat(frames, ignore_index=True)


# ── 1. RUL cap tests ──────────────────────────────────────────────

def test_rul_cap_clips_at_125():
    """
    Engines with > 125 cycles must have their early-life RUL capped at 125.
    """
    df = make_engine_df(engine_id=1, n_cycles=200)
    df = add_rul(df, rul_cap=125)
    assert df['RUL'].max() == 125, \
        f"RUL max should be 125, got {df['RUL'].max()}"

def test_rul_cap_custom_value():
    """add_rul must respect arbitrary rul_cap values."""
    df = make_engine_df(engine_id=1, n_cycles=300)
    df = add_rul(df, rul_cap=100)
    assert df['RUL'].max() == 100

def test_rul_minimum_is_zero():
    """
    The last cycle of any engine must have RUL = 0
    (engine fails at last recorded cycle in training set).
    """
    df = make_engine_df(engine_id=1, n_cycles=50)
    df = add_rul(df, rul_cap=125)
    last_cycle = df[df['cycle'] == df['cycle'].max()]
    assert int(last_cycle['RUL'].values[0]) == 0, \
        "Last cycle must have RUL = 0"

def test_rul_decreases_monotonically():
    """
    Within a single engine, RUL must decrease (or stay flat when capped)
    as cycles increase — it must never increase.
    """
    df = make_engine_df(engine_id=1, n_cycles=150)
    df = add_rul(df, rul_cap=125)
    df = df.sort_values('cycle').reset_index(drop=True)
    rul_values = df['RUL'].values
    # diff should be <= 0 everywhere (RUL can stay flat when capped)
    diffs = np.diff(rul_values)
    assert np.all(diffs <= 0), \
        f"RUL is not monotonically decreasing. Max increase: {diffs.max()}"

def test_rul_short_engine_no_cap():
    """
    An engine with exactly 50 cycles and rul_cap=125:
    the first cycle must have RUL = 49 (not capped).
    """
    df = make_engine_df(engine_id=1, n_cycles=50)
    df = add_rul(df, rul_cap=125)
    first_cycle_rul = df[df['cycle'] == 1]['RUL'].values[0]
    assert first_cycle_rul == 49, \
        f"Expected 49, got {first_cycle_rul}"

def test_rul_multiple_engines_independent():
    """
    RUL calculation must be per-engine. Two engines with different
    cycle counts must each have their own correct max RUL.
    """
    df = make_multi_engine_df([(1, 50), (2, 200)])
    df = add_rul(df, rul_cap=125)

    eng1_max = df[df['engine_id'] == 1]['RUL'].max()
    eng2_max = df[df['engine_id'] == 2]['RUL'].max()

    # Engine 1 has 50 cycles: raw max RUL = 49, not capped
    assert eng1_max == 49, f"Engine 1 max RUL: expected 49, got {eng1_max}"
    # Engine 2 has 200 cycles: raw max RUL = 199, capped to 125
    assert eng2_max == 125, f"Engine 2 max RUL: expected 125, got {eng2_max}"


# ── 2. Zero-variance sensor dropping ─────────────────────────────

def test_drop_sensors_not_in_useful():
    """
    Every sensor in DROP_SENSORS must be absent from USEFUL_SENSORS.
    """
    for s in DROP_SENSORS:
        assert s not in USEFUL_SENSORS, \
            f"{s} is in DROP_SENSORS but also appears in USEFUL_SENSORS"

def test_useful_sensors_count():
    """
    FD001 has 21 sensors total. We drop 7. Expect 14 useful sensors.
    """
    all_sensors = [c for c in COLUMNS if c.startswith('s') and not c.startswith('setting')]
    assert len(all_sensors) == 21, f"Expected 21 sensors, got {len(all_sensors)}"
    assert len(USEFUL_SENSORS) == 17, \
    f"Expected 17 useful features (3 settings + 14 sensors), got {len(USEFUL_SENSORS)}"

def test_useful_sensors_match_feature_cols():
    """
    FEATURE_COLS in dataset.py includes operational settings + useful sensors.
    The sensor subset of FEATURE_COLS must match USEFUL_SENSORS exactly.
    """
    sensor_feature_cols = [c for c in FEATURE_COLS if c.startswith('s')]
    assert sorted(sensor_feature_cols) == sorted(USEFUL_SENSORS), \
        f"Sensor mismatch between dataset.py and data_preprocessing.py.\n" \
        f"  dataset.py:            {sorted(sensor_feature_cols)}\n" \
        f"  data_preprocessing.py: {sorted(USEFUL_SENSORS)}"

def test_normalize_does_not_include_dropped_sensors():
    """
    After normalization, the dropped sensors should not appear in the
    DataFrame output (they should have been removed before normalize is called).
    """
    df = make_multi_engine_df([(1, 50), (2, 50)])
    df = add_rul(df, rul_cap=125)
    keep_cols = ['engine_id', 'cycle'] + USEFUL_SENSORS + ['RUL']
    train_df = df[keep_cols].copy()
    test_df  = df[keep_cols].copy()

    train_out, test_out, _ = normalize(train_df.copy(), test_df.copy())

    for s in DROP_SENSORS:
        assert s not in train_out.columns, f"Dropped sensor {s} still in train"
        assert s not in test_out.columns,  f"Dropped sensor {s} still in test"


# ── 3. Normalization ──────────────────────────────────────────────

def test_normalize_range_zero_to_one():
    """
    After MinMaxScaler, all useful sensor values in train
    must be in [0, 1].
    """
    df1 = make_engine_df(1, 60, sensor_val=0.2)
    df2 = make_engine_df(2, 60, sensor_val=0.8)
    df  = pd.concat([df1, df2], ignore_index=True)
    df  = add_rul(df, rul_cap=125)
    keep_cols = ['engine_id', 'cycle'] + USEFUL_SENSORS + ['RUL']
    train_df = df[keep_cols].copy()
    test_df  = df[keep_cols].copy()

    train_out, _, _ = normalize(train_df, test_df)

    for col in USEFUL_SENSORS:
        col_min = train_out[col].min()
        col_max = train_out[col].max()
        assert col_min >= -1e-6, f"{col} min {col_min:.4f} below 0"
        assert col_max <= 1 + 1e-6, f"{col} max {col_max:.4f} above 1"

def test_normalize_fit_on_train_only():
    """
    Scaler must be fit on train only. If test has values outside the
    train range, they should fall outside [0,1] — that's correct behaviour.
    This test verifies the scaler is NOT re-fit on test data.
    """
    # Train: all sensors at 0.5 → scaler learns min=max=0.5 (degenerate)
    # But with two engines with different values we get a real range.
    train_df = make_multi_engine_df([(1, 40), (2, 40)], sensor_val=0.3)
    train_df = add_rul(train_df, rul_cap=125)
    # Test: sensors at a completely different value
    test_df  = make_multi_engine_df([(3, 40), (4, 40)], sensor_val=0.7)
    test_df  = add_rul(test_df, rul_cap=125)

    keep = ['engine_id', 'cycle'] + USEFUL_SENSORS + ['RUL']
    train_df = train_df[keep].copy()
    test_df  = test_df[keep].copy()

    # Give train a real range by using two different sensor values
    train_df_varied = pd.concat([
        make_engine_df(1, 40, sensor_val=0.0),
        make_engine_df(2, 40, sensor_val=1.0),
    ], ignore_index=True)
    train_df_varied = add_rul(train_df_varied, rul_cap=125)[keep].copy()

    _, test_out, scaler = normalize(train_df_varied, test_df)

    # Scaler was fit on [0, 1] range. test sensor_val=0.7 should map to ~0.7
    for col in USEFUL_SENSORS:
        val = test_out[col].iloc[0]
        assert abs(val - 0.7) < 0.05, \
            f"Expected ~0.7 after normalize, got {val:.3f} for {col}"


# ── 4. Sliding window output shapes ──────────────────────────────

def test_build_windows_shape():
    """
    build_windows must return X of shape (N, window_size, n_features)
    and y of shape (N,).
    """
    df = make_multi_engine_df([(1, 50), (2, 60), (3, 40)])
    df = add_rul(df, rul_cap=125)
    keep = ['engine_id', 'cycle'] + USEFUL_SENSORS + ['RUL']
    # Add settings columns (zeros) so FEATURE_COLS is satisfied
    df['setting_1'] = 0.0
    df['setting_2'] = 0.0
    df['setting_3'] = 0.0

    X, y = build_windows(df, window_size=WINDOW_SIZE)

    assert X.ndim == 3, f"X must be 3D, got {X.ndim}D"
    assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"
    assert X.shape[1] == WINDOW_SIZE, \
        f"Window size mismatch: expected {WINDOW_SIZE}, got {X.shape[1]}"
    assert X.shape[2] == len(FEATURE_COLS), \
        f"Feature count mismatch: expected {len(FEATURE_COLS)}, got {X.shape[2]}"
    assert X.shape[0] == y.shape[0], \
        "X and y must have the same number of samples"

def test_build_windows_correct_window_count():
    """
    For one engine with n_cycles, the number of windows is
    n_cycles - window_size + 1.
    """
    n_cycles = 50
    df = make_engine_df(1, n_cycles)
    df = add_rul(df, rul_cap=125)
    df['setting_1'] = 0.0
    df['setting_2'] = 0.0
    df['setting_3'] = 0.0

    X, y = build_windows(df, window_size=WINDOW_SIZE)

    expected = n_cycles - WINDOW_SIZE + 1
    assert X.shape[0] == expected, \
        f"Expected {expected} windows, got {X.shape[0]}"

def test_build_windows_skips_short_engines():
    """
    Engines with fewer cycles than window_size must be skipped.
    """
    df = pd.concat([
        make_engine_df(1, 10),   # too short — skip
        make_engine_df(2, 50),   # long enough
    ], ignore_index=True)
    df = add_rul(df, rul_cap=125)
    df['setting_1'] = 0.0
    df['setting_2'] = 0.0
    df['setting_3'] = 0.0

    X, y = build_windows(df, window_size=WINDOW_SIZE)

    # Only engine 2 contributes: 50 - 30 + 1 = 21 windows
    assert X.shape[0] == 21, \
        f"Expected 21 windows (only engine 2), got {X.shape[0]}"

def test_build_test_windows_one_per_engine():
    """
    build_test_windows must return exactly one window per engine,
    regardless of how long each engine's history is.
    """
    n_engines = 5
    df = make_multi_engine_df([(i, 50) for i in range(1, n_engines + 1)])
    df = add_rul(df, rul_cap=125)
    df['setting_1'] = 0.0
    df['setting_2'] = 0.0
    df['setting_3'] = 0.0

    X, y = build_test_windows(df, window_size=WINDOW_SIZE)

    assert X.shape[0] == n_engines, \
        f"Expected {n_engines} test windows, got {X.shape[0]}"
    assert X.shape == (n_engines, WINDOW_SIZE, len(FEATURE_COLS))

def test_build_test_windows_pads_short_engines():
    """
    Engines shorter than window_size must be zero-padded at the front,
    not dropped (unlike training windows).
    """
    df = make_engine_df(1, n_cycles=10)   # shorter than WINDOW_SIZE=30
    df = add_rul(df, rul_cap=125)
    df['setting_1'] = 0.0
    df['setting_2'] = 0.0
    df['setting_3'] = 0.0

    X, y = build_test_windows(df, window_size=WINDOW_SIZE)

    assert X.shape[0] == 1, "Short engine should be kept (padded), not dropped"
    assert X.shape[1] == WINDOW_SIZE

def test_window_rul_label_is_last_cycle_rul():
    """
    The RUL label for each window must correspond to the last
    cycle in that window, not the first or the middle.
    """
    df = make_engine_df(1, n_cycles=50)
    df = add_rul(df, rul_cap=125)
    df['setting_1'] = 0.0
    df['setting_2'] = 0.0
    df['setting_3'] = 0.0
    df = df.sort_values('cycle').reset_index(drop=True)

    X, y = build_windows(df, window_size=WINDOW_SIZE)

    # First window spans cycles 1–30. RUL at cycle 30 of a 50-cycle engine:
    # max_cycle = 50, so RUL = 50 - 30 = 20 (under cap of 125)
    expected_first_label = 50 - WINDOW_SIZE
    assert abs(y[0] - expected_first_label) < 1e-5, \
        f"First window RUL: expected {expected_first_label}, got {y[0]}"
