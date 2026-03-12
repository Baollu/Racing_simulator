"""
data_collector.py
------------------
Records observation+action pairs to timestamped CSV files during manual
driving, and provides utilities to load that data back for training.

CSV schema (one row per simulation step):
  ray_0, ray_1, ..., ray_{n-1}, speed, steering, acceleration
"""

import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd


class DataCollector:
    """
    Writes driving data to a new CSV file each session.

    Usage:
        dc = DataCollector(data_dir="data")
        dc.record(obs_array, steering, acceleration)   # called every step
        dc.close()                                     # called on exit
    """

    FLUSH_INTERVAL = 100  # flush to disk every N steps

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(data_dir, f"session_{timestamp}.csv")

        self._file_handle = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file_handle)
        self._step_count = 0
        self._n_rays: int | None = None  # inferred on first record() call

        print(f"[DataCollector] Recording to: {self.filepath}")

    def _write_header(self, n_rays: int) -> None:
        ray_cols = [f"ray_{i}" for i in range(n_rays)]
        header = ray_cols + ["speed", "steering", "acceleration"]
        self._writer.writerow(header)

    def record(
        self,
        observations: np.ndarray,
        steering: float,
        acceleration: float,
    ) -> None:
        """
        Append one row to the CSV.

        Args:
            observations: 1-D array of shape (nbRay + 1,)
                          last element is speed, preceding elements are ray distances
            steering:     float in [-1.0, 1.0]
            acceleration: float in [-1.0, 1.0]
        """
        obs = observations.flatten()

        # Lazily write header once we know the observation size
        if self._n_rays is None:
            self._n_rays = len(obs) - 1  # last element is speed
            self._write_header(self._n_rays)

        row = list(obs) + [steering, acceleration]
        self._writer.writerow(row)
        self._step_count += 1

        if self._step_count % self.FLUSH_INTERVAL == 0:
            self._file_handle.flush()

    def close(self) -> None:
        """Flush and close the file handle."""
        self._file_handle.flush()
        self._file_handle.close()
        print(f"[DataCollector] Session closed. {self._step_count} steps recorded -> {self.filepath}")

    # ------------------------------------------------------------------
    # Static loading utilities (used by train.py and eda.py)
    # ------------------------------------------------------------------

    @staticmethod
    def load_all(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
        """
        Read all CSV files in data_dir and concatenate into a single dataset.

        Returns:
            X: np.ndarray of shape (N, nbRay + 1)  — observations [rays..., speed]
            y: np.ndarray of shape (N, 2)           — [steering, acceleration]

        Raises:
            FileNotFoundError: if data_dir does not exist
            ValueError: if no CSV files are found
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        csv_files = sorted(
            f for f in os.listdir(data_dir) if f.endswith(".csv")
        )
        if not csv_files:
            raise ValueError(f"No CSV files found in: {data_dir}")

        frames = []
        for fname in csv_files:
            path = os.path.join(data_dir, fname)
            df = pd.read_csv(path)
            frames.append(df)
            print(f"[DataCollector] Loaded {len(df):>6} rows from {fname}")

        data = pd.concat(frames, ignore_index=True)
        print(f"[DataCollector] Total: {len(data)} rows across {len(csv_files)} file(s)")

        # Last two columns are labels; everything else is input
        X = data.iloc[:, :-2].to_numpy(dtype=np.float32)
        y = data.iloc[:, -2:].to_numpy(dtype=np.float32)
        return X, y

    @staticmethod
    def load_as_tensors(
        data_dir: str = "data",
        device: str = "cpu",
    ):
        """
        Convenience wrapper: loads data and returns PyTorch float32 tensors.

        Returns:
            X_tensor: torch.Tensor of shape (N, nbRay + 1)
            y_tensor: torch.Tensor of shape (N, 2)
        """
        import torch

        X, y = DataCollector.load_all(data_dir)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        return X_tensor, y_tensor
