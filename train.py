"""
train.py
---------
Training pipeline for the supervised driving model.

Workflow:
  1. Load all CSV data from data/
  2. Split into train / validation (80/20)
  3. Compute normalization stats from TRAIN split only (no leakage)
  4. Normalize both splits
  5. Train DrivingModel with Adam + ReduceLROnPlateau
  6. Save best model (lowest val loss) and normalization stats
  7. Print final evaluation metrics and save training curves plot

Usage:
  python train.py
  python train.py --epochs 200 --batch-size 128 --data-dir data --model-dir models
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ai_model import (
    DrivingModel,
    compute_normalization_stats,
    load_normalization_stats,
    normalize,
    save_normalization_stats,
)
from data_collector import DataCollector


# ===========================================================================
# Data loading
# ===========================================================================


def load_and_prepare_data(
    data_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cpu",
) -> tuple:
    """
    Load data, split, normalise (train stats only), return tensors.

    Returns:
        X_train, X_val, y_train, y_val : torch.Tensors on device
        norm_mean, norm_std             : np.ndarrays
    """
    print("\n[Train] Loading data...")
    X_np, y_np = DataCollector.load_all(data_dir)

    print(f"[Train] Dataset: {X_np.shape[0]} samples, {X_np.shape[1]} features")

    # Split FIRST to prevent leakage into normalization stats
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Compute stats from training split only
    norm_mean, norm_std = compute_normalization_stats(X_tr)

    X_tr_norm = normalize(X_tr, norm_mean, norm_std).astype(np.float32)
    X_val_norm = normalize(X_val, norm_mean, norm_std).astype(np.float32)

    X_train = torch.tensor(X_tr_norm, device=device)
    X_val_t = torch.tensor(X_val_norm, device=device)
    y_train = torch.tensor(y_tr, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    print(f"[Train] Train: {len(X_train)}  |  Val: {len(X_val_t)}")
    return X_train, X_val_t, y_train, y_val_t, norm_mean, norm_std


# ===========================================================================
# DataLoaders
# ===========================================================================


def build_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Wrap tensors in TensorDataset / DataLoader."""
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ===========================================================================
# Training step
# ===========================================================================


def train_epoch(
    model: DrivingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


# ===========================================================================
# Evaluation
# ===========================================================================


def evaluate(
    model: DrivingModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """
    Evaluate the model on a DataLoader.

    Returns a dict with:
      loss, mae_steering, mae_acceleration, r2_steering, r2_acceleration
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            total_loss += criterion(pred, y_batch).item() * len(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return {
        "loss": total_loss / len(loader.dataset),
        "mae_steering": float(mean_absolute_error(targets[:, 0], preds[:, 0])),
        "mae_acceleration": float(mean_absolute_error(targets[:, 1], preds[:, 1])),
        "r2_steering": float(r2_score(targets[:, 0], preds[:, 0])),
        "r2_acceleration": float(r2_score(targets[:, 1], preds[:, 1])),
    }


# ===========================================================================
# Plotting
# ===========================================================================


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str = "models/training_curves.png",
) -> None:
    """Save a plot of training vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", color="steelblue")
    ax.plot(epochs, val_losses, label="Val loss", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Train] Training curves saved to: {save_path}")


# ===========================================================================
# Main
# ===========================================================================


def main(
    data_dir: str = "data",
    model_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device_name: str = "cpu",
) -> None:
    device = torch.device(device_name)
    os.makedirs(model_dir, exist_ok=True)

    # ---- Data ----
    X_train, X_val, y_train, y_val, norm_mean, norm_std = load_and_prepare_data(
        data_dir=data_dir, device=device_name
    )

    n_rays = X_train.shape[1] - 1  # last feature is speed
    train_loader, val_loader = build_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    # ---- Model ----
    model = DrivingModel(n_rays=n_rays).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, verbose=True
    )

    print(f"\n[Train] Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"[Train] Training for {epochs} epochs  |  batch={batch_size}  |  lr={learning_rate}\n")

    # ---- Training loop ----
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(model_dir, "driving_model.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device_name)
        val_metrics = evaluate(model, val_loader, criterion, device_name)
        val_loss = val_metrics["loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{epochs}  "
                f"train_loss={train_loss:.5f}  "
                f"val_loss={val_loss:.5f}  "
                f"mae_steer={val_metrics['mae_steering']:.4f}  "
                f"mae_accel={val_metrics['mae_acceleration']:.4f}"
            )

    # ---- Save normalization stats ----
    norm_path = os.path.join(model_dir, "norm_stats.npz")
    save_normalization_stats(norm_mean, norm_std, norm_path)

    # ---- Final evaluation on best model ----
    best_model = DrivingModel.load(best_model_path, device=device_name)
    train_metrics = evaluate(best_model, train_loader, criterion, device_name)
    val_metrics = evaluate(best_model, val_loader, criterion, device_name)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best model)")
    print("=" * 60)
    print(f"  Train MSE loss      : {train_metrics['loss']:.5f}")
    print(f"  Val   MSE loss      : {val_metrics['loss']:.5f}")
    print(f"  Train MAE steering  : {train_metrics['mae_steering']:.4f}")
    print(f"  Val   MAE steering  : {val_metrics['mae_steering']:.4f}")
    print(f"  Train MAE accel     : {train_metrics['mae_acceleration']:.4f}")
    print(f"  Val   MAE accel     : {val_metrics['mae_acceleration']:.4f}")
    print(f"  Train R² steering   : {train_metrics['r2_steering']:.4f}")
    print(f"  Val   R² steering   : {val_metrics['r2_steering']:.4f}")
    print(f"  Train R² accel      : {train_metrics['r2_acceleration']:.4f}")
    print(f"  Val   R² accel      : {val_metrics['r2_acceleration']:.4f}")
    print("=" * 60)

    # ---- Plots ----
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(model_dir, "training_curves.png"),
    )

    print(f"\n[Train] Done. Best model: {best_model_path}")
    print(f"[Train] Run: python client.py --mode ai")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the driving AI model")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV session files")
    parser.add_argument("--model-dir", default="models", help="Output directory for model artifacts")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cpu / cuda)",
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
    )
