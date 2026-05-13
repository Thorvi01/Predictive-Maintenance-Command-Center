# train.py
# Training loop for RULPredictor with:
#   - MSE loss + Adam optimizer
#   - Learning rate scheduler
#   - Early stopping
#   - MLflow experiment tracking

import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
import os
import time

from dataset import get_dataloaders
from model import RULPredictor, mc_predict

# ── 1. Configuration ─────────────────────────────────────────────
# All hyperparameters in one place — easy to change and track
CONFIG = {
    # Model
    'input_size':   17,
    'hidden_size':  64,
    'num_layers':   2,
    'dropout_rate': 0.2,

    # Training
    'epochs':       100,
    'batch_size':   64,
    'learning_rate': 0.001,
    'val_split':    0.2,

    # Early stopping
    # Stop if validation loss doesn't improve for this many epochs
    'patience':     15,

    # Learning rate scheduler
    # Reduce LR by factor 0.5 if val loss plateaus for 7 epochs
    'lr_patience':  7,
    'lr_factor':    0.5,

    # Data
    'window_size':  30,
    'rul_cap':      125,
}

# ── 2. Evaluation helper ─────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    """
    Runs model on a DataLoader in eval mode (no dropout).
    Returns average loss and RMSE over all batches.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss  = criterion(preds, y_batch)

            total_loss += loss.item() * len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    rmse     = np.sqrt(np.mean(
        (np.array(all_preds) - np.array(all_targets)) ** 2
    ))
    return avg_loss, rmse


# ── 3. NASA asymmetric score ─────────────────────────────────────
def nasa_score(y_true, y_pred):
    """
    NASA's competition scoring function.
    Penalizes late predictions (underestimating RUL) more than early ones.

    If prediction > truth (you said engine has MORE life than it does):
        penalty is exponential — very bad, engine could fail unexpectedly
    If prediction < truth (you said engine has LESS life than it does):
        penalty is smaller — you'll do a preventive check earlier, but safe

    Lower score is better.
    """
    d = np.array(y_pred) - np.array(y_true)
    score = np.sum(
        np.where(d < 0,
                 np.exp(-d / 13) - 1,   # early prediction (safe)
                 np.exp( d / 10) - 1)   # late prediction  (dangerous)
    )
    return score


# ── 4. Main training function ────────────────────────────────────
def train(config=CONFIG):
    # Detect GPU — use it if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # ── Data ──
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        window_size=config['window_size']
    )

    # ── Model ──
    model = RULPredictor(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    # ── Loss, optimizer, scheduler ──
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )
    # Reduces learning rate when val loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config['lr_patience'],
        factor=config['lr_factor']
    )

    # ── MLflow experiment ──
    mlflow.set_experiment("rul-predictor")

    with mlflow.start_run():
        # Log all hyperparameters
        mlflow.log_params(config)

        # ── Training loop ──
        best_val_loss  = float('inf')
        patience_count = 0
        best_model_state = None

        print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>10} "
              f"{'Val RMSE':>10} {'LR':>10}")
        print("-" * 55)

        for epoch in range(1, config['epochs'] + 1):
            # ── Train one epoch ──
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()              # clear old gradients
                preds = model(X_batch)             # forward pass
                loss  = criterion(preds, y_batch)  # compute loss
                loss.backward()                    # compute gradients
                optimizer.step()                   # update weights

                train_loss += loss.item() * len(y_batch)

            train_loss /= len(train_loader.dataset)

            # ── Validate ──
            val_loss, val_rmse = evaluate(model, val_loader,
                                          criterion, device)

            # ── LR scheduler step ──
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            # ── Log to MLflow ──
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss':   val_loss,
                'val_rmse':   val_rmse,
                'lr':         current_lr
            }, step=epoch)

            # ── Print progress ──
            print(f"{epoch:>5} {train_loss:>12.4f} {val_loss:>10.4f} "
                  f"{val_rmse:>10.2f} {current_lr:>10.6f}")

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_count   = 0
                # Save best model weights in memory
                best_model_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
            else:
                patience_count += 1
                if patience_count >= config['patience']:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {config['patience']} epochs)")
                    break

        # ── Restore best model ──
        model.load_state_dict(best_model_state)
        print(f"\nBest val loss: {best_val_loss:.4f}")

        # ── Final test evaluation ──
        print("\nEvaluating on test set...")
        test_loss, test_rmse = evaluate(model, test_loader,
                                        criterion, device)

        # MC Dropout prediction on test set
        X_test_all = torch.cat([X for X, _ in test_loader])
        y_test_all = torch.cat([y for _, y in test_loader]).numpy()

        mean_preds, std_preds, lower, upper, _ = mc_predict(
            model, X_test_all, n_samples=100, device=str(device)
        )

        # Clip negative predictions to 0 (RUL can't be negative)
        mean_preds = np.clip(mean_preds, 0, None)

        # NASA score
        score = nasa_score(y_test_all, mean_preds)

        # Coverage: what % of true RUL values fall inside our 90% CI
        coverage = np.mean(
            (y_test_all >= lower) & (y_test_all <= upper)
        ) * 100

        print(f"\n{'='*40}")
        print(f"  Test RMSE:        {test_rmse:.2f} cycles")
        print(f"  NASA Score:       {score:.2f} (lower=better)")
        print(f"  Mean uncertainty: {std_preds.mean():.2f} cycles (avg std)")
        print(f"  90% CI Coverage:  {coverage:.1f}% (ideal=90%)")
        print(f"{'='*40}")

        # Log final metrics
        mlflow.log_metrics({
            'test_rmse':   test_rmse,
            'nasa_score':  score,
            'mean_std':    float(std_preds.mean()),
            'ci_coverage': coverage
        })

        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config':           config,
            'test_rmse':        test_rmse,
            'nasa_score':       score,
        }, 'models/rul_predictor.pt')

        mlflow.pytorch.log_model(model, "model")
        print("\nModel saved to models/rul_predictor.pt")
        print("Run 'mlflow ui' to view experiment results in browser")

    return model, test_rmse, score


# ── 5. Run training ──────────────────────────────────────────────
if __name__ == '__main__':
    start = time.time()
    model, rmse, score = train()
    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")