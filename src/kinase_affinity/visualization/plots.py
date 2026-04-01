"""Visualization functions for model evaluation and data analysis.

Standard plots for molecular property prediction:
    - Predicted vs actual scatter plots
    - Calibration diagrams
    - Selective prediction curves
    - Split comparison heatmaps

All plots use a consistent style and save to results/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("results/figures")

# Consistent styling
MODEL_COLORS = {
    "random_forest": "#2196F3",
    "xgboost": "#4CAF50",
    "elasticnet": "#FF9800",
    "mlp": "#9C27B0",
}

SPLIT_MARKERS = {
    "random": "o",
    "scaffold": "s",
    "target": "^",
}


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: Path | None = None,
    rmse: float | None = None,
    r2: float | None = None,
) -> plt.Figure:
    """Scatter plot of predicted vs actual pActivity values.

    Includes diagonal reference line, R², and RMSE annotation.
    Uses hexbin for large datasets to avoid overplotting.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    n = len(y_true)
    if n > 5000:
        hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap="Blues",
                        mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, c="#2196F3")

    # Diagonal reference
    lims = [
        min(np.min(y_true), np.min(y_pred)) - 0.5,
        max(np.max(y_true), np.max(y_pred)) + 0.5,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("True pActivity")
    ax.set_ylabel("Predicted pActivity")
    ax.set_title(title)

    # Annotate metrics
    text_parts = []
    if rmse is not None:
        text_parts.append(f"RMSE = {rmse:.3f}")
    if r2 is not None:
        text_parts.append(f"R² = {r2:.3f}")
    if text_parts:
        ax.text(0.05, 0.95, "\n".join(text_parts),
                transform=ax.transAxes, verticalalignment="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white",
                                       alpha=0.8))

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration_diagram(
    expected: np.ndarray,
    observed: np.ndarray,
    title: str = "Calibration Diagram",
    save_path: Path | None = None,
    miscal_area: float | None = None,
) -> plt.Figure:
    """Plot calibration curve (expected vs observed coverage).

    The diagonal represents perfect calibration. Points above the diagonal
    indicate overconfident predictions (intervals too narrow); points below
    indicate underconfident predictions (intervals too wide).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(expected, observed, "o-", color="#2196F3", markersize=6,
            label="Model")
    ax.fill_between(expected, expected, observed, alpha=0.15, color="#2196F3")

    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")

    if miscal_area is not None:
        ax.text(0.05, 0.95, f"Miscal. area = {miscal_area:.3f}",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white",
                                       alpha=0.8))

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_selective_prediction(
    retention: np.ndarray,
    rmse: np.ndarray,
    title: str = "Selective Prediction",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot RMSE vs retention fraction curve.

    A steep descent from right (all predictions) to left (most confident)
    indicates that uncertainty estimates are informative.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(retention, rmse, "o-", color="#2196F3", markersize=4)

    # Shade improvement region
    baseline_rmse = rmse[-1]  # RMSE at 100% retention
    ax.axhline(baseline_rmse, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline RMSE = {baseline_rmse:.3f}")

    ax.set_xlabel("Retention fraction")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_split_comparison(
    summary_df,
    metric: str = "test_rmse",
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap comparing model performance across split strategies.

    Shows how each model degrades from random to scaffold to target splits.
    """
    pivot = summary_df.pivot(index="model", columns="split", values=metric)

    # Reorder for consistent display
    split_order = [s for s in ["random", "scaffold", "target"] if s in pivot.columns]
    model_order = [m for m in ["random_forest", "xgboost", "elasticnet", "mlp"]
                   if m in pivot.index]
    pivot = pivot.loc[model_order, split_order]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(split_order)))
    ax.set_xticklabels(split_order)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order)

    # Annotate cells
    for i in range(len(model_order)):
        for j in range(len(split_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > np.nanmedian(pivot.values) else "black")

    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"Model × Split: {metric}")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_target_histogram(
    target_metrics_df,
    metric: str = "rmse",
    title: str = "Per-Target RMSE Distribution",
    save_path: Path | None = None,
) -> plt.Figure:
    """Histogram of per-target metrics.

    Shows the distribution of model performance across different kinase
    targets — reveals whether aggregate metrics hide high variance.
    """
    values = target_metrics_df[metric].dropna().values

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(values, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(values), color="red", linestyle="--",
               label=f"Median = {np.median(values):.3f}")
    ax.axvline(np.mean(values), color="orange", linestyle="--",
               label=f"Mean = {np.mean(values):.3f}")

    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Number of targets")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_uncertainty_correlation(
    abs_error: np.ndarray,
    uncertainty: np.ndarray,
    title: str = "Error vs Uncertainty",
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of absolute error vs predicted uncertainty.

    Strong positive correlation means the model "knows what it doesn't know."
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    n = len(abs_error)
    if n > 5000:
        hb = ax.hexbin(uncertainty, abs_error, gridsize=50, cmap="Oranges",
                        mincnt=1, linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(uncertainty, abs_error, alpha=0.3, s=8, c="#FF9800")

    ax.set_xlabel("Predicted uncertainty (std)")
    ax.set_ylabel("Absolute error")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
