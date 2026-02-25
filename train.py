"""
Train MLP model for proton NMR chemical shift prediction.

This module implements the training pipeline for a multi-layer perceptron (MLP)
that predicts 1H NMR chemical shifts from molecular descriptors. The model
learns the relationship between local chemical environment features and
nuclear magnetic shielding effects.

Training methodology:
- Architecture: MLP with ReLU activations
- Loss function: Huber loss (robust to outliers)
- Optimizer: AdamW with learning rate scheduling
- Early stopping to prevent overfitting
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

DATASET_PATH = "datasets/proton_dataset.npz"
MODEL_OUT = "proton_mlp.pt"
PLOTS_DIR = "training_plots"

BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
RANDOM_STATE = 42
EARLY_STOP_PATIENCE = 20
MIN_DELTA = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(PLOTS_DIR, exist_ok=True)


class ProtonMLP(nn.Module):
    """
    Multi-layer perceptron for proton NMR chemical shift prediction.
    
    Architecture designed for regression of continuous chemical shift values
    from molecular descriptor inputs. Uses ReLU activation functions for
    non-linear mapping capability.
    
    Parameters
    ----------
    n_features : int
        Number of input features (molecular descriptors)
    """
    
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()


def evaluate(model, loader, device):
    """Evaluate model, return MAE, R2 and predictions."""
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            true.append(yb.numpy())
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    return mean_absolute_error(true, preds), r2_score(true, preds), preds, true


def save_plots(val_mae_history, val_r2_history, val_preds, val_true, output_dir):
    """Generate and save training plots."""
    # 1. Val MAE evolution
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_mae_history) + 1), val_mae_history, 'b-', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE (ppm)")
    plt.title("Validation MAE Evolution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_mae_evolution.png"), dpi=150)
    plt.close()

    # 2. Val R2 evolution
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_r2_history) + 1), val_r2_history, 'g-', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Val R2")
    plt.title("Validation R2 Evolution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_r2_evolution.png"), dpi=150)
    plt.close()
    
    # 3. Predicted vs True scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(val_true, val_preds, s=5, alpha=0.3, c='steelblue')
    lims = [min(val_true.min(), val_preds.min()), max(val_true.max(), val_preds.max())]
    plt.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
    plt.xlabel("True δ (ppm)")
    plt.ylabel("Predicted δ (ppm)")
    plt.title("Predicted vs True Chemical Shifts")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_vs_true.png"), dpi=150)
    plt.close()


def main():
    print(f"Using device: {DEVICE}")

    # Load dataset
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    print(f"Dataset loaded: X={X.shape}, y={y.shape}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    # Model setup
    model = ProtonMLP(X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    loss_fn = nn.HuberLoss()

    # Training loop
    best_val_mae, epochs_no_improve = np.inf, 0
    val_mae_history = []
    val_r2_history = []

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_mae, val_r2, val_preds, val_true = evaluate(model, val_loader, DEVICE)
        val_mae_history.append(val_mae)
        val_r2_history.append(val_r2)

        print(f"Epoch {epoch:03d} | Val MAE: {val_mae:.5f} ppm | Val R2: {val_r2:.5f}")
        scheduler.step(val_mae)

        if val_mae < best_val_mae - MIN_DELTA:
            best_val_mae, epochs_no_improve = val_mae, 0
            torch.save(model.state_dict(), MODEL_OUT)
            best_preds, best_true = val_preds, val_true
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Generate plots
    save_plots(val_mae_history, val_r2_history, best_preds, best_true, PLOTS_DIR)
    
    print(f"Best Val MAE: {best_val_mae:.4f} ppm | Model saved to: {MODEL_OUT}")
    print(f"Plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
