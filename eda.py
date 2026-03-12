"""
eda.py
-------
Exploratory Data Analysis for collected driving data.

Produces diagnostic plots to validate data quality before training:
  1. Action distributions (steering / acceleration histograms)
  2. Ray distance distributions (box plots per ray)
  3. Correlation heatmap (which rays correlate with actions?)
  4. Time-series sample (first 500 steps, validates smooth driving)
  5. Summary statistics printed to stdout

Output plots are saved to eda_output/ directory.

Usage:
  python eda.py
  python eda.py --data-dir data --save-dir eda_output
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ===========================================================================
# Data loading
# ===========================================================================


def load_data(data_dir: str = "data") -> pd.DataFrame:
    """Load all session CSV files and concatenate into one DataFrame."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in: {data_dir}")

    frames = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"[EDA] Loaded {len(df)} rows from {len(csv_files)} session file(s)")
    return df


# ===========================================================================
# Plots
# ===========================================================================


def plot_action_distributions(df: pd.DataFrame, save_dir: str = "eda_output") -> None:
    """Histograms of steering and acceleration distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, color in zip(axes, ["steering", "acceleration"], ["steelblue", "tomato"]):
        series = df[col]
        ax.hist(series, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(series.mean(), color="black", linestyle="--", linewidth=1.2, label=f"mean={series.mean():.3f}")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Count")
        ax.set_title(f"{col.capitalize()} distribution  (std={series.std():.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Action Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "action_distributions.png")


def plot_ray_distributions(df: pd.DataFrame, save_dir: str = "eda_output") -> None:
    """Box plots of all ray distance columns to spot uninformative sensors."""
    ray_cols = [c for c in df.columns if c.startswith("ray_")]
    if not ray_cols:
        print("[EDA] No ray columns found, skipping ray distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(ray_cols) * 0.9), 5))
    df[ray_cols].plot.box(ax=ax, grid=True, color=dict(boxes="steelblue", whiskers="gray", medians="tomato", caps="gray"))
    ax.set_xlabel("Ray index")
    ax.set_ylabel("Distance")
    ax.set_title("Ray Distance Distributions\n(nearly-constant rays are uninformative)")
    fig.tight_layout()
    _save(fig, save_dir, "ray_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = "eda_output") -> None:
    """Pearson correlation heatmap: which rays correlate most with actions?"""
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.7), max(8, len(corr) * 0.6)))
    sns.heatmap(
        corr,
        ax=ax,
        annot=len(corr) <= 20,  # only annotate if not too many columns
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        square=True,
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "correlation_heatmap.png")


def plot_time_series_sample(
    df: pd.DataFrame,
    n_steps: int = 500,
    save_dir: str = "eda_output",
) -> None:
    """Line plot of steering and acceleration for the first n_steps rows."""
    sample = df.head(n_steps)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    axes[0].plot(sample["steering"].values, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Steering")
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sample["acceleration"].values, color="tomato", linewidth=0.8)
    axes[1].set_ylabel("Acceleration")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Action Time Series — first {min(n_steps, len(df))} steps", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "time_series_sample.png")


# ===========================================================================
# Summary statistics
# ===========================================================================


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print descriptive statistics and data-quality indicators."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total steps     : {len(df)}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")

    steer_zero_pct = (df["steering"] == 0.0).mean() * 100
    accel_zero_pct = (df["acceleration"] == 0.0).mean() * 100
    print(f"\n  Steering  == 0  : {steer_zero_pct:.1f}%  (straight-driving bias indicator)")
    print(f"  Accel     == 0  : {accel_zero_pct:.1f}%")

    if steer_zero_pct > 80:
        print(
            "\n  [WARNING] More than 80% of steps have zero steering.\n"
            "  The model may struggle on corners. Consider collecting\n"
            "  more cornering data or using weighted sampling in train.py."
        )

    print("\n--- describe() ---")
    print(df.describe().to_string())
    print("=" * 60)


# ===========================================================================
# Helpers
# ===========================================================================


def _save(fig: plt.Figure, save_dir: str, filename: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================


def main(data_dir: str = "data", save_dir: str = "eda_output") -> None:
    df = load_data(data_dir)
    print_summary_statistics(df)
    plot_action_distributions(df, save_dir)
    plot_ray_distributions(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    plot_time_series_sample(df, save_dir=save_dir)
    print(f"\n[EDA] All plots saved to: {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory data analysis of collected driving data")
    parser.add_argument("--data-dir", default="data", help="Directory containing session CSV files")
    parser.add_argument("--save-dir", default="eda_output", help="Directory for output plots")
    args = parser.parse_args()

    main(data_dir=args.data_dir, save_dir=args.save_dir)
