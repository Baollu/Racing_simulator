"""
ai_model.py
-----------
Neural network model for supervised driving prediction.

Architecture:
  Input:    nbRay + 1  (ray distances + speed)
  Hidden 1: 64, ReLU + BatchNorm1d
  Hidden 2: 64, ReLU + BatchNorm1d
  Hidden 3: 32, ReLU
  Output:   2,  Tanh  (steering in [-1,1], acceleration in [-1,1])

Also provides normalization utilities that must be used consistently
between training (train.py) and inference (client.py).
"""

import os

import numpy as np
import torch
import torch.nn as nn


# ===========================================================================
# Model
# ===========================================================================


class DrivingModel(nn.Module):
    """
    Small MLP for predicting (steering, acceleration) from ray + speed observations.

    Keeping the model small is intentional: a large dataset + small model
    generalises better than a large model + small dataset for this task.
    """

    def __init__(self, n_rays: int = 10) -> None:
        super().__init__()
        self.n_rays = n_rays
        input_size = n_rays + 1  # rays + speed

        self.network = nn.Sequential(
            # Hidden layer 1
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Hidden layer 2
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Hidden layer 3
            nn.Linear(64, 32),
            nn.ReLU(),
            # Output: 2 continuous values in [-1, 1]
            nn.Linear(32, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_rays + 1) float32 tensor
        Returns:
            (batch_size, 2) float32 tensor — [steering, acceleration]
        """
        return self.network(x)

    def predict(self, obs: np.ndarray) -> tuple[float, float]:
        """
        Single-sample inference convenience method.

        Args:
            obs: 1-D numpy array of shape (n_rays + 1,), already normalised
        Returns:
            (steering, acceleration) as Python floats
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = self.network(x)
        steering = float(out[0, 0].item())
        acceleration = float(out[0, 1].item())
        return steering, acceleration

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and architecture metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"n_rays": self.n_rays, "state_dict": self.state_dict()}, path)
        print(f"[DrivingModel] Model saved to: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DrivingModel":
        """Load model from a .pt file produced by save()."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(n_rays=checkpoint["n_rays"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        print(f"[DrivingModel] Model loaded from: {path}  (n_rays={checkpoint['n_rays']})")
        return model


# ===========================================================================
# Normalization utilities
# ===========================================================================


def compute_normalization_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and standard deviation.

    Args:
        X: (N, F) float32 array
    Returns:
        mean: (F,) float32
        std:  (F,) float32
    """
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    return mean, std


def normalize(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Z-score normalisation.  std is clipped to 1e-8 to avoid division by zero
    for features that are nearly constant (e.g. a ray always at max distance).

    Args:
        X:    (N, F) or (F,) float32 array
        mean: (F,) float32
        std:  (F,) float32
    Returns:
        normalised array of the same shape as X
    """
    safe_std = np.clip(std, 1e-8, None)
    return (X - mean) / safe_std


def save_normalization_stats(mean: np.ndarray, std: np.ndarray, path: str) -> None:
    """Save mean and std arrays to a .npz file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    np.savez(path, mean=mean, std=std)
    print(f"[Normalization] Stats saved to: {path}")


def load_normalization_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mean and std arrays from a .npz file."""
    data = np.load(path)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    print(f"[Normalization] Stats loaded from: {path}")
    return mean, std
