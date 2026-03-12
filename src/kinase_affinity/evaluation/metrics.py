"""Regression and classification evaluation metrics.

Metrics selected for their relevance to drug discovery:

Regression (predicting pActivity values):
    - RMSE: Root mean squared error (penalizes large errors)
    - MAE: Mean absolute error (robust to outliers)
    - R²: Coefficient of determination (explained variance)
    - Pearson R: Linear correlation between predicted and actual
    - Spearman ρ: Rank correlation (important for compound ranking)

Classification (active/inactive at pActivity threshold):
    - AUROC: Area under ROC curve (overall discrimination)
    - AUPRC: Area under precision-recall curve (better for imbalanced data)
    - Precision@k: Precision in top-k predictions (drug discovery use case:
      "if I test my top 100 predictions, how many are truly active?")
    - Enrichment factor: fold improvement over random selection

Usage:
    from kinase_affinity.evaluation.metrics import compute_regression_metrics
    metrics = compute_regression_metrics(y_true, y_pred)
"""

from __future__ import annotations

import numpy as np


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True pActivity values.
    y_pred : np.ndarray
        Predicted pActivity values.

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name → value.
    """
    raise NotImplementedError("TODO: Implement regression metrics")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1).
    y_pred_proba : np.ndarray
        Predicted probabilities of the positive class.
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name → value.
    """
    raise NotImplementedError("TODO: Implement classification metrics")


def precision_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
) -> float:
    """Compute precision in the top-k predictions.

    Answers: "If I select the k compounds with the highest predicted
    activity, what fraction are truly active?"

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores (higher = more likely active).
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    float
        Precision in top-k.
    """
    raise NotImplementedError("TODO: Implement precision@k")


def enrichment_factor(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fraction: float = 0.01,
) -> float:
    """Compute enrichment factor at a given fraction.

    EF = (hits_in_top_fraction / n_top) / (total_hits / n_total)

    An EF of 10 at 1% means the model finds 10x more actives in
    its top 1% than random selection would.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores.
    fraction : float
        Top fraction to evaluate (e.g., 0.01 for top 1%).

    Returns
    -------
    float
        Enrichment factor.
    """
    raise NotImplementedError("TODO: Implement enrichment factor")
