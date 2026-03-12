"""Dataset curation: activity normalization, duplicate handling, quality filters.

This module takes standardized molecule data and produces a clean, analysis-ready
dataset with the following steps:

    1. Convert activities to pActivity (-log10 of molar value)
    2. Handle duplicate measurements via median aggregation
    3. Flag noisy compounds (high variance across measurements)
    4. Apply quality filters (pActivity range, etc.)
    5. Add binary active/inactive classification labels
    6. Save versioned dataset to data/processed/v{version}/

All thresholds and parameters are config-driven (configs/dataset_v1.yaml).

Usage:
    python -m kinase_affinity.data.curate
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def convert_to_pactivity(df: pd.DataFrame) -> pd.DataFrame:
    """Convert standard_value (nM) to pActivity (-log10 M).

    pActivity = -log10(standard_value * 1e-9)
              = 9 - log10(standard_value)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'standard_value' column in nM units.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'pactivity' column.
    """
    raise NotImplementedError("TODO: Implement pActivity conversion")


def handle_duplicates(
    df: pd.DataFrame,
    group_cols: list[str],
    aggregation: str = "median",
    noise_std_threshold: float = 1.0,
    min_measurements: int = 3,
) -> pd.DataFrame:
    """Collapse duplicate measurements for the same compound-target pair.

    Parameters
    ----------
    df : pd.DataFrame
        Activity data with potential duplicates.
    group_cols : list[str]
        Columns defining a unique measurement (e.g.,
        ['std_smiles', 'target_chembl_id', 'standard_type']).
    aggregation : str
        Aggregation method ('median' or 'mean').
    noise_std_threshold : float
        If std of pActivity > this threshold and n >= min_measurements,
        flag the compound as noisy.
    min_measurements : int
        Minimum number of measurements to assess noise.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with 'pactivity' (aggregated),
        'n_measurements', 'pactivity_std', and 'is_noisy' columns.
    """
    raise NotImplementedError("TODO: Implement duplicate handling")


def apply_quality_filters(
    df: pd.DataFrame,
    pactivity_min: float = 3.0,
    pactivity_max: float = 12.0,
) -> pd.DataFrame:
    """Remove measurements outside plausible pActivity range.

    Parameters
    ----------
    df : pd.DataFrame
        Curated activity data.
    pactivity_min : float
        Minimum pActivity (removes very weak binders / likely inactive).
    pactivity_max : float
        Maximum pActivity (removes likely erroneous very potent values).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    raise NotImplementedError("TODO: Implement quality filters")


def add_classification_labels(
    df: pd.DataFrame,
    threshold: float = 6.0,
) -> pd.DataFrame:
    """Add binary active/inactive labels based on pActivity threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'pactivity' column.
    threshold : float
        pActivity threshold. >= threshold is 'active'.
        Default 6.0 corresponds to IC50 <= 1 μM.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'is_active' boolean column.
    """
    raise NotImplementedError("TODO: Implement classification labeling")


def main() -> None:
    """Run the full curation pipeline."""
    with open("configs/dataset_v1.yaml") as f:
        config = yaml.safe_load(f)

    version = config["version"]
    output_dir = PROCESSED_DIR / version
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running curation pipeline (dataset %s)...", version)
    # TODO: Load raw data, apply standardization, curate, and save
    raise NotImplementedError("TODO: Implement full curation pipeline")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
