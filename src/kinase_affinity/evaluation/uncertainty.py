"""Uncertainty quantification and calibration analysis.

A well-calibrated model's confidence should correlate with its actual
accuracy. This module provides tools to assess and visualize calibration:

    - Calibration curves: do 90% prediction intervals contain 90% of true values?
    - Miscalibration area: numerical measure of calibration quality
    - Error-uncertainty correlation: do uncertain predictions have higher errors?
    - Selective prediction: how much does accuracy improve if we abstain on
      high-uncertainty predictions?

These analyses are key to understanding when to trust model predictions,
which is critical in drug discovery where false positives waste
experimental resources.

Usage:
    from kinase_affinity.evaluation.uncertainty import calibration_curve
    expected, observed = calibration_curve(y_true, y_pred, y_std)
"""

from __future__ import annotations

import numpy as np


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute regression calibration curve.

    For each confidence level (e.g., 50%, 60%, ..., 95%), compute the
    fraction of true values that fall within the corresponding prediction
    interval. A perfectly calibrated model should have expected = observed.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted means.
    y_std : np.ndarray
        Predicted standard deviations.
    n_bins : int
        Number of confidence levels to evaluate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (expected_coverage, observed_coverage) arrays of shape (n_bins,).
    """
    raise NotImplementedError("TODO: Implement calibration curve")


def miscalibration_area(
    expected: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Compute the miscalibration area (area between calibration curve and diagonal).

    Lower is better. Zero means perfect calibration.

    Parameters
    ----------
    expected : np.ndarray
        Expected coverage fractions.
    observed : np.ndarray
        Observed coverage fractions.

    Returns
    -------
    float
        Miscalibration area.
    """
    raise NotImplementedError("TODO: Implement miscalibration area")


def error_uncertainty_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, float]:
    """Assess correlation between prediction error and uncertainty.

    A good uncertainty estimate should have high correlation with
    absolute prediction error — uncertain predictions should have
    larger errors on average.

    Parameters
    ----------
    y_true, y_pred, y_std : np.ndarray
        True values, predictions, and uncertainty estimates.

    Returns
    -------
    dict[str, float]
        Pearson and Spearman correlation between |error| and uncertainty.
    """
    raise NotImplementedError("TODO: Implement error-uncertainty correlation")


def selective_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMSE as a function of retention fraction.

    If we only keep the most confident predictions (lowest uncertainty),
    how much does RMSE improve? This shows the value of uncertainty
    estimation for practical decision-making.

    Parameters
    ----------
    y_true, y_pred, y_std : np.ndarray
        True values, predictions, and uncertainty estimates.
    n_points : int
        Number of retention fractions to evaluate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (retention_fraction, rmse_at_fraction) arrays.
    """
    raise NotImplementedError("TODO: Implement selective prediction curve")
