# data_preprocessing.py
# Loads raw NASA C-MAPSS FD001 data, computes RUL labels,
# normalizes sensors, and saves cleaned data to data/processed/

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ── 1. Column names ──────────────────────────────────────────────
# The raw file has no headers. These are the 26 columns in order.
COLUMNS = [
    'engine_id', 'cycle',
    'setting_1', 'setting_2', 'setting_3',
    's1','s2','s3','s4','s5','s6','s7','s8','s9','s10',
    's11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21'
]

# These 7 sensors are constant in FD001 — zero variance, useless for learning
DROP_SENSORS = ['s1','s5','s6','s10','s16','s18','s19']

# The 14 sensors that actually change and carry information
USEFUL_SENSORS = [s for s in COLUMNS if s.startswith('s') and s not in DROP_SENSORS]

# ── 2. Load raw data ─────────────────────────────────────────────
def load_raw(path):
    """
    Reads a C-MAPSS text file into a DataFrame.
    The file is space-separated with no header row.
    """
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    return df

# ── 3. Compute RUL labels ────────────────────────────────────────
def add_rul(df, rul_cap=125):
    """
    For each engine, RUL at cycle t = (max cycle of that engine) - t.
    We cap RUL at 125 to avoid noisy early-life data.

    Example: engine runs for 200 cycles total.
      - At cycle 1:   raw RUL = 199, capped to 125
      - At cycle 100: raw RUL = 100, kept as 100
      - At cycle 190: raw RUL = 10,  kept as 10
    """
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=rul_cap)
    df.drop(columns=['max_cycle'], inplace=True)
    return df

# ── 4. Add RUL for test set using ground truth file ──────────────
def add_rul_test(test_df, rul_path):
    """
    Test set engines are cut off before failure — we don't know
    their max cycle. The RUL_FD001.txt file gives us the true RUL
    at the last observed cycle for each engine.
    """
    rul_true = pd.read_csv(rul_path, header=None, names=['RUL_true'])
    rul_true['engine_id'] = rul_true.index + 1  # engines are 1-indexed

    # Get only the last cycle row for each engine
    last_cycles = test_df.groupby('engine_id').last().reset_index()
    last_cycles = last_cycles.merge(rul_true, on='engine_id')

    # Add RUL to full test set by working backwards from last cycle
    test_df = test_df.merge(rul_true, on='engine_id')
    max_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    test_df = test_df.merge(max_cycles, on='engine_id')
    test_df['RUL'] = test_df['RUL_true'] + (test_df['max_cycle'] - test_df['cycle'])
    test_df['RUL'] = test_df['RUL'].clip(upper=125)
    test_df.drop(columns=['RUL_true', 'max_cycle'], inplace=True)
    return test_df

# ── 5. Normalize sensors ─────────────────────────────────────────
def normalize(train_df, test_df):
    """
    Fit MinMaxScaler on training data only (important!), then
    apply the same scale to test data.

    Why fit on train only: if we used test data to compute the scale,
    we'd be leaking future information into training — cheating.
    """
    scaler = MinMaxScaler()
    train_df[USEFUL_SENSORS] = scaler.fit_transform(train_df[USEFUL_SENSORS])
    test_df[USEFUL_SENSORS] = scaler.transform(test_df[USEFUL_SENSORS])
    return train_df, test_df, scaler

# ── 6. Main pipeline ─────────────────────────────────────────────
def preprocess(data_dir='data/raw', out_dir='data/processed'):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading raw data...")
    train_df = load_raw(os.path.join(data_dir, 'train_FD001.txt'))
    test_df  = load_raw(os.path.join(data_dir, 'test_FD001.txt'))
    rul_path = os.path.join(data_dir, 'RUL_FD001.txt')

    print("Computing RUL labels...")
    train_df = add_rul(train_df)
    test_df  = add_rul_test(test_df, rul_path)

    print("Dropping useless sensors...")
    keep_cols = ['engine_id', 'cycle'] + USEFUL_SENSORS + ['RUL']
    train_df = train_df[keep_cols]
    test_df  = test_df[keep_cols]

    print("Normalizing sensors...")
    train_df, test_df, scaler = normalize(train_df, test_df)

    print("Saving processed data...")
    train_df.to_csv(os.path.join(out_dir, 'train_FD001.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir,  'test_FD001.csv'),  index=False)

    print(f"\nDone!")
    print(f"  Train shape: {train_df.shape}  → {len(train_df['engine_id'].unique())} engines")
    print(f"  Test shape:  {test_df.shape}   → {len(test_df['engine_id'].unique())} engines")
    print(f"  Sensors kept: {USEFUL_SENSORS}")
    print(f"  RUL range (train): {train_df['RUL'].min():.0f} – {train_df['RUL'].max():.0f}")

    return train_df, test_df

if __name__ == '__main__':
    preprocess()