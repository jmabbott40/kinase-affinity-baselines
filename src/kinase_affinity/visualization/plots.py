"""Visualization functions for model evaluation and data analysis.

Standard plots for molecular property prediction:
    - Predicted vs actual scatter plots
    - Residual distributions
    - Calibration diagrams
    - Learning curves
    - Feature importance bar charts
    - Split comparison heatmaps

All plots use a consistent style and save to results/figures/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIGURES_DIR = Path("results/figures")


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: Path | None = None,
) -> None:
    """Scatter plot of predicted vs actual pActivity values.

    Includes diagonal reference line, R², and RMSE annotation.
    """
    raise NotImplementedError("TODO: Implement scatter plot")


def plot_calibration_diagram(
    expected: np.ndarray,
    observed: np.ndarray,
    title: str = "Calibration Diagram",
    save_path: Path | None = None,
) -> None:
    """Plot calibration curve (expected vs observed coverage)."""
    raise NotImplementedError("TODO: Implement calibration plot")


def plot_selective_prediction(
    retention: np.ndarray,
    rmse: np.ndarray,
    title: str = "Selective Prediction",
    save_path: Path | None = None,
) -> None:
    """Plot RMSE vs retention fraction curve."""
    raise NotImplementedError("TODO: Implement selective prediction plot")


def plot_split_comparison(
    results: dict,
    metric: str = "rmse",
    save_path: Path | None = None,
) -> None:
    """Heatmap comparing model performance across split strategies."""
    raise NotImplementedError("TODO: Implement split comparison heatmap")
