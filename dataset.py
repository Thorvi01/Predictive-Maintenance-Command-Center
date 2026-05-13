# dataset.py
# Builds sliding window sequences from processed C-MAPSS data.
# Each window = 30 consecutive cycles of sensor readings → one RUL label.

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Sensors we use as input features (same order as preprocessing)
FEATURE_COLS = [
    'setting_1', 'setting_2', 'setting_3',
    's2', 's3', 's4', 's7', 's8', 's9',
    's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'
]

WINDOW_SIZE = 30  # how many cycles per sequence


# ── 1. Build windows from a DataFrame ───────────────────────────
def build_windows(df, window_size=WINDOW_SIZE):
    """
    Slides a window of `window_size` cycles over each engine's data.
    Returns:
        X: numpy array of shape (num_windows, window_size, num_features)
        y: numpy array of shape (num_windows,) — RUL at end of each window
    """
    X_list, y_list = [], []

    for engine_id, group in df.groupby('engine_id'):
        # Sort by cycle — must be in time order
        group = group.sort_values('cycle').reset_index(drop=True)

        features = group[FEATURE_COLS].values  # shape: (num_cycles, 17)
        labels   = group['RUL'].values          # shape: (num_cycles,)

        num_cycles = len(group)

        # Can only make a window if we have at least window_size cycles
        if num_cycles < window_size:
            continue

        # Slide window one step at a time
        for start in range(num_cycles - window_size + 1):
            end = start + window_size
            X_list.append(features[start:end])   # 30 rows of sensors
            y_list.append(labels[end - 1])        # RUL at the last cycle

    X = np.array(X_list, dtype=np.float32)  # (N, 30, 17)
    y = np.array(y_list, dtype=np.float32)  # (N,)

    return X, y


# ── 2. Build test windows (one per engine — last 30 cycles) ─────
def build_test_windows(df, window_size=WINDOW_SIZE):
    """
    For test set we only predict once per engine — at the last
    observed cycle. So we take only the final window per engine.
    """
    X_list, y_list = [], []

    for engine_id, group in df.groupby('engine_id'):
        group = group.sort_values('cycle').reset_index(drop=True)
        features = group[FEATURE_COLS].values
        labels   = group['RUL'].values

        if len(group) < window_size:
            # Pad with zeros at the front if engine is too short
            pad_len = window_size - len(group)
            features = np.pad(features, ((pad_len, 0), (0, 0)), mode='constant')
            labels   = np.pad(labels,   (pad_len, 0),           mode='constant')

        # Take last window only
        X_list.append(features[-window_size:])
        y_list.append(labels[-1])

    X = np.array(X_list, dtype=np.float32)  # (100, 30, 17)
    y = np.array(y_list, dtype=np.float32)  # (100,)

    return X, y


# ── 3. PyTorch Dataset class ─────────────────────────────────────
class RULDataset(Dataset):
    """
    Wraps numpy arrays into a PyTorch Dataset.
    PyTorch's DataLoader needs this format to batch and shuffle data.
    """
    def __init__(self, X, y):
        # Convert numpy → PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # How many samples total
        return len(self.X)

    def __getitem__(self, idx):
        # Returns one (input_window, rul_label) pair
        return self.X[idx], self.y[idx]


# ── 4. Convenience function — returns DataLoaders ready to use ───
def get_dataloaders(
    train_path='data/processed/train_FD001.csv',
    test_path='data/processed/test_FD001.csv',
    window_size=WINDOW_SIZE,
    batch_size=64,
    val_split=0.2
):
    """
    Loads processed CSVs, builds windows, splits train into
    train/val, and returns three DataLoaders.

    val_split=0.2 means 20% of training windows → validation set.
    """
    print("Building sliding windows...")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Build windows
    X_train_full, y_train_full = build_windows(train_df, window_size)
    X_test, y_test             = build_test_windows(test_df, window_size)

    print(f"  Total training windows: {len(X_train_full)}")
    print(f"  Test windows (1 per engine): {len(X_test)}")

    # Train / validation split
    # Important: we split by index, not randomly across engines
    # to avoid data leakage between overlapping windows
    n_val   = int(len(X_train_full) * val_split)
    n_train = len(X_train_full) - n_val

    X_train, y_train = X_train_full[:n_train], y_train_full[:n_train]
    X_val,   y_val   = X_train_full[n_train:], y_train_full[n_train:]

    print(f"  Train windows: {len(X_train)}")
    print(f"  Val windows:   {len(X_val)}")

    # Wrap in Dataset + DataLoader
    train_loader = DataLoader(RULDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(RULDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(RULDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ── 5. Quick test — run this file directly to verify ────────────
if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders()

    # Grab one batch and print shapes
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nOne training batch:")
    print(f"  X shape: {X_batch.shape}  → (batch_size, window, features)")
    print(f"  y shape: {y_batch.shape}  → (batch_size,)")
    print(f"  RUL range in batch: {y_batch.min():.1f} – {y_batch.max():.1f}")

    X_test_batch, y_test_batch = next(iter(test_loader))
    print(f"\nOne test batch:")
    print(f"  X shape: {X_test_batch.shape}")
    print(f"  y shape: {y_test_batch.shape}")
    print("\nDataset pipeline working correctly!")