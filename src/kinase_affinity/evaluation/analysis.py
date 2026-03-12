"""Error analysis and failure mode identification.

Goes beyond aggregate metrics to understand *why* models fail:

    - Activity cliffs: pairs of structurally similar compounds with
      very different activities (hard for fingerprint models)
    - Scaffold rarity: do models perform worse on rare scaffolds?
    - Per-target breakdown: which kinases are easier/harder to predict?
    - Noise impact: are compounds flagged as "noisy" during curation
      systematically harder to predict?

This analysis drives scientific insight and guides model improvement.

Usage:
    from kinase_affinity.evaluation.analysis import find_worst_predictions
    worst = find_worst_predictions(y_true, y_pred, df, top_n=50)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_worst_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Identify the top-N worst predicted compounds.

    Parameters
    ----------
    y_true : np.ndarray
        True pActivity values.
    y_pred : np.ndarray
        Predicted pActivity values.
    df : pd.DataFrame
        Original dataset with compound metadata.
    top_n : int
        Number of worst predictions to return.

    Returns
    -------
    pd.DataFrame
        Worst predicted compounds with error, true/predicted values,
        and metadata for analysis.
    """
    raise NotImplementedError("TODO: Implement worst prediction identification")


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_ids: np.ndarray,
) -> pd.DataFrame:
    """Compute metrics broken down by target.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted pActivity values.
    target_ids : np.ndarray
        Target identifiers for each measurement.

    Returns
    -------
    pd.DataFrame
        Metrics (RMSE, R², count) per target.
    """
    raise NotImplementedError("TODO: Implement per-target metrics")


def noise_impact_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_noisy: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compare model performance on noisy vs clean compounds.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted values.
    is_noisy : np.ndarray
        Boolean array indicating noisy compounds (high measurement variance).

    Returns
    -------
    dict
        {'clean': {metrics}, 'noisy': {metrics}}
    """
    raise NotImplementedError("TODO: Implement noise impact analysis")
