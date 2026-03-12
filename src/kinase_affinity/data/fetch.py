"""Fetch kinase bioactivity data from ChEMBL.

This module queries the ChEMBL database for protein kinase targets and their
associated bioactivity measurements (IC50, Ki, Kd). Raw data is saved as a
parquet file for downstream processing.

Data flow:
    1. Query ChEMBL for all human protein kinase single-protein targets
    2. For each target, fetch bioactivity records matching quality filters
    3. Combine into a single DataFrame with columns:
       - molecule_chembl_id, canonical_smiles, target_chembl_id, target_pref_name
       - standard_type (IC50/Ki/Kd), standard_value, standard_units
       - pchembl_value, assay_chembl_id, assay_confidence_score
    4. Save to data/raw/chembl_kinase_activities.parquet

Usage:
    python -m kinase_affinity.data.fetch
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
DEFAULT_CONFIG = Path("configs/dataset_v1.yaml")


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Load dataset configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_kinase_targets(organism: str = "Homo sapiens") -> pd.DataFrame:
    """Fetch all single-protein kinase targets from ChEMBL.

    Parameters
    ----------
    organism : str
        Species filter. Default is human targets only.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: target_chembl_id, target_pref_name,
        target_type, organism.
    """
    raise NotImplementedError("TODO: Implement ChEMBL target query")


def fetch_bioactivities(
    target_chembl_ids: list[str],
    activity_types: list[str],
    assay_confidence_min: int = 7,
) -> pd.DataFrame:
    """Fetch bioactivity measurements for given targets.

    Parameters
    ----------
    target_chembl_ids : list[str]
        ChEMBL target IDs to query.
    activity_types : list[str]
        Activity types to include (e.g., ["IC50", "Ki", "Kd"]).
    assay_confidence_min : int
        Minimum assay confidence score (7 = direct single protein).

    Returns
    -------
    pd.DataFrame
        Raw bioactivity records.
    """
    raise NotImplementedError("TODO: Implement ChEMBL bioactivity query")


def main() -> None:
    """Run the full data fetching pipeline."""
    config = load_config()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching kinase targets from ChEMBL...")
    targets = fetch_kinase_targets(organism=config["source"]["organism"])
    logger.info("Found %d kinase targets", len(targets))

    logger.info("Fetching bioactivity data...")
    activities = fetch_bioactivities(
        target_chembl_ids=targets["target_chembl_id"].tolist(),
        activity_types=config["activity"]["types"],
        assay_confidence_min=config["activity"]["assay_confidence_min"],
    )
    logger.info("Fetched %d activity records", len(activities))

    output_path = RAW_DATA_DIR / "chembl_kinase_activities.parquet"
    activities.to_parquet(output_path, index=False)
    logger.info("Saved raw data to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
