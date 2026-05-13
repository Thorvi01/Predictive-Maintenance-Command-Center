# model.py
# Two-layer LSTM with MC Dropout for uncertainty quantification.
# The key architectural choice: dropout stays ON during inference.

import torch
import torch.nn as nn
import numpy as np


# ── 1. The LSTM model ────────────────────────────────────────────
class RULPredictor(nn.Module):
    """
    Two-layer stacked LSTM with dropout between layers.

    Architecture:
        Input  → (batch, 30, 17)
        LSTM 1 → (batch, 30, hidden_size)
        Dropout
        LSTM 2 → (batch, 30, hidden_size)
        Dropout
        Take last timestep → (batch, hidden_size)
        FC layer → (batch, 1)
        Output → (batch,)  ← predicted RUL

    dropout_rate: fraction of neurons randomly zeroed each forward pass.
                  0.2 means 20% of neurons are switched off.
    """
    def __init__(self, input_size=17, hidden_size=64,
                 num_layers=2, dropout_rate=0.2):
        super(RULPredictor, self).__init__()

        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.dropout_rate = dropout_rate

        # LSTM layers
        # dropout param in nn.LSTM applies between layers (not after last)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq, features)
            dropout=dropout_rate    # dropout between LSTM layers
        )

        # Extra dropout layer after LSTM — this is the MC Dropout layer
        # We will keep this active during inference (the key trick)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Final fully connected layer: hidden_size → 1 (RUL prediction)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Standard forward pass.
        x shape: (batch_size, sequence_length, input_size)
                 e.g. (64, 30, 17)
        """
        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size)
        # We ignore hidden state (h_n, c_n) — not needed for prediction
        lstm_out, _ = self.lstm(x)

        # Take only the LAST timestep's output
        # Shape: (batch, hidden_size)
        # Why last? It has seen all 30 cycles and carries the most information
        last_step = lstm_out[:, -1, :]

        # Apply dropout — during training AND inference (MC Dropout)
        dropped = self.dropout(last_step)

        # Linear layer → single RUL value per sample
        # Shape: (batch, 1) → squeeze to (batch,)
        out = self.fc(dropped).squeeze(-1)

        return out


# ── 2. Enable MC Dropout at inference time ───────────────────────
def enable_mc_dropout(model):
    """
    PyTorch's model.eval() turns OFF all dropout layers.
    This function keeps dropout ON for MC sampling.

    Call this instead of model.eval() when doing MC inference.
    """
    model.train()  # sets dropout to active mode

    # But we don't want BatchNorm layers updating their stats
    # So freeze any BatchNorm layers (we don't have them here,
    # but this is best practice for MC Dropout in general)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or \
           isinstance(module, nn.BatchNorm2d):
            module.eval()


# ── 3. MC Dropout inference ──────────────────────────────────────
def mc_predict(model, X_tensor, n_samples=100, device='cpu'):
    """
    Runs N stochastic forward passes with dropout active.
    Each pass gives a slightly different prediction.

    Args:
        model:    trained RULPredictor
        X_tensor: input tensor shape (batch, 30, 17)
        n_samples: number of MC samples (100 is standard)
        device:   'cpu' or 'cuda'

    Returns:
        mean:  shape (batch,) — best RUL estimate
        std:   shape (batch,) — uncertainty (higher = less confident)
        lower: shape (batch,) — 5th percentile  (lower bound of 90% CI)
        upper: shape (batch,) — 95th percentile (upper bound of 90% CI)
        all_preds: shape (n_samples, batch) — all raw predictions
    """
    enable_mc_dropout(model)
    X_tensor = X_tensor.to(device)

    predictions = []

    with torch.no_grad():  # no gradient computation — saves memory
        for _ in range(n_samples):
            pred = model(X_tensor)           # shape: (batch,)
            predictions.append(pred.cpu().numpy())

    # Stack into (n_samples, batch)
    all_preds = np.array(predictions)

    mean  = all_preds.mean(axis=0)
    std   = all_preds.std(axis=0)
    lower = np.percentile(all_preds, 5,  axis=0)  # 5th percentile
    upper = np.percentile(all_preds, 95, axis=0)  # 95th percentile

    return mean, std, lower, upper, all_preds


# ── 4. Quick test — run this file directly ───────────────────────
if __name__ == '__main__':
    # Create a model with default settings
    model = RULPredictor(input_size=17, hidden_size=64,
                         num_layers=2, dropout_rate=0.2)

    print("=== Model Architecture ===")
    print(model)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Test with a fake batch — shape matches real data
    fake_batch = torch.randn(64, 30, 17)  # 64 samples, 30 cycles, 17 sensors

    # Standard forward pass (training mode)
    model.train()
    output = model(fake_batch)
    print(f"\n=== Standard forward pass ===")
    print(f"Input shape:  {fake_batch.shape}")
    print(f"Output shape: {output.shape}  → one RUL per sample")

    # MC Dropout inference
    print(f"\n=== MC Dropout inference (100 samples) ===")
    mean, std, lower, upper, all_preds = mc_predict(
        model, fake_batch, n_samples=100
    )
    print(f"Mean RUL (first 5 engines): {mean[:5].round(1)}")
    print(f"Std  dev (first 5 engines): {std[:5].round(2)}")
    print(f"90% CI lower (first 5):     {lower[:5].round(1)}")
    print(f"90% CI upper (first 5):     {upper[:5].round(1)}")
    print(f"\nAll predictions shape: {all_preds.shape} → (n_samples, batch)")
    print("\nModel working correctly!")