"""Create filtered dataset subsets for ablation studies.

Primary use case: the "ESM-92" subset containing only the 92 kinase targets
with real ESM-2 protein embeddings. This enables a clean protein-aware vs.
ligand-only comparison where every target has a genuine embedding, rather than
the full 507-target dataset where ~82% of targets receive an arbitrary
fallback embedding.

Usage:
    python -m kinase_affinity.data.subset
    python -m kinase_affinity.data.subset --endpoint IC50  # IC50-only subset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kinase_affinity.data.splits import (
    random_split,
    save_splits,
    scaffold_split,
    target_split,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")


def create_esm_subset(
    dataset_version: str = "v1",
    output_tag: str = "esm92",
) -> pd.DataFrame:
    """Filter curated dataset to targets with real ESM-2 embeddings.

    Reads the target_index.json produced by the ESM-2 embedding pipeline
    to identify which targets have genuine protein embeddings. Filters the
    curated activity data to only those targets.

    Parameters
    ----------
    dataset_version : str
        Source dataset version.
    output_tag : str
        Tag for output files (e.g., "esm92").

    Returns
    -------
    pd.DataFrame
        Filtered dataset.
    """
    data_dir = DATA_DIR / dataset_version
    features_dir = data_dir / "features"

    # Load the target index from ESM-2 embedding computation
    target_index_path = features_dir / "target_index.json"
    if not target_index_path.exists():
        raise FileNotFoundError(
            f"target_index.json not found at {target_index_path}. "
            "ESM-2 embeddings must be computed first "
            "(python -m kinase_affinity.features.protein_embeddings)."
        )

    with open(target_index_path) as f:
        target_to_row = json.load(f)

    targets_with_embeddings = set(target_to_row.keys())
    logger.info("Targets with real ESM-2 embeddings: %d", len(targets_with_embeddings))

    # Load curated dataset
    curated_path = data_dir / "curated_activities.parquet"
    df = pd.read_parquet(curated_path)
    logger.info("Full dataset: %d records, %d targets",
                len(df), df["target_chembl_id"].nunique())

    # Filter to targets with real embeddings
    mask = df["target_chembl_id"].isin(targets_with_embeddings)
    df_subset = df[mask].reset_index(drop=True)

    n_targets_subset = df_subset["target_chembl_id"].nunique()
    n_compounds_subset = df_subset["std_smiles"].nunique()

    logger.info("ESM subset: %d records, %d targets, %d compounds",
                len(df_subset), n_targets_subset, n_compounds_subset)
    logger.info("  Retained %.1f%% of records",
                len(df_subset) / len(df) * 100)

    # Save subset
    output_dir = data_dir / f"subsets/{output_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = output_dir / "curated_activities.parquet"
    df_subset.to_parquet(subset_path, index=False)
    logger.info("Saved subset to %s", subset_path)

    # Save summary statistics
    summary = {
        "subset_tag": output_tag,
        "source_version": dataset_version,
        "n_records": len(df_subset),
        "n_records_full": len(df),
        "record_retention_pct": round(len(df_subset) / len(df) * 100, 1),
        "n_targets": n_targets_subset,
        "n_targets_full": df["target_chembl_id"].nunique(),
        "n_compounds": n_compounds_subset,
        "n_compounds_full": df["std_smiles"].nunique(),
        "targets_included": sorted(targets_with_embeddings & set(df_subset["target_chembl_id"].unique())),
        "pactivity_mean": round(float(df_subset["pactivity"].mean()), 3),
        "pactivity_std": round(float(df_subset["pactivity"].std()), 3),
        "active_fraction": round(float(df_subset["is_active"].mean()), 3),
        "endpoint_distribution": df_subset["standard_type"].value_counts().to_dict(),
    }
    with open(output_dir / "subset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", output_dir / "subset_summary.json")

    return df_subset


def create_endpoint_subset(
    dataset_version: str = "v1",
    endpoint: str = "IC50",
) -> pd.DataFrame:
    """Filter curated dataset to a single endpoint type.

    IC50 represents ~77% of the data. Running on the IC50-only subset
    verifies that mixed-endpoint effects don't drive the main conclusions.

    Parameters
    ----------
    dataset_version : str
        Source dataset version.
    endpoint : str
        Activity type to retain ("IC50", "Ki", or "Kd").

    Returns
    -------
    pd.DataFrame
        Filtered dataset.
    """
    data_dir = DATA_DIR / dataset_version
    output_tag = endpoint.lower()

    df = pd.read_parquet(data_dir / "curated_activities.parquet")
    logger.info("Full dataset: %d records", len(df))

    df_subset = df[df["standard_type"] == endpoint].reset_index(drop=True)
    logger.info("%s subset: %d records (%.1f%% of full)",
                endpoint, len(df_subset), len(df_subset) / len(df) * 100)

    output_dir = data_dir / f"subsets/{output_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_subset.to_parquet(output_dir / "curated_activities.parquet", index=False)

    summary = {
        "subset_tag": output_tag,
        "endpoint": endpoint,
        "n_records": len(df_subset),
        "n_records_full": len(df),
        "record_retention_pct": round(len(df_subset) / len(df) * 100, 1),
        "n_targets": df_subset["target_chembl_id"].nunique(),
        "n_compounds": df_subset["std_smiles"].nunique(),
        "pactivity_mean": round(float(df_subset["pactivity"].mean()), 3),
        "active_fraction": round(float(df_subset["is_active"].mean()), 3),
    }
    with open(output_dir / "subset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return df_subset


def generate_splits_for_subset(
    dataset_version: str = "v1",
    subset_tag: str = "esm92",
    seeds: list[int] | None = None,
) -> None:
    """Generate all split strategies for a dataset subset.

    Parameters
    ----------
    dataset_version : str
        Source dataset version.
    subset_tag : str
        Subset identifier (e.g., "esm92", "ic50").
    seeds : list[int], optional
        Seeds to use for splits. Defaults to [42].
    """
    if seeds is None:
        seeds = [42]

    data_dir = DATA_DIR / dataset_version
    subset_dir = data_dir / f"subsets/{subset_tag}"
    df = pd.read_parquet(subset_dir / "curated_activities.parquet")
    logger.info("Generating splits for %s subset (%d records)", subset_tag, len(df))

    for seed in seeds:
        split_dir = subset_dir / "splits"
        if len(seeds) > 1:
            split_dir = subset_dir / f"splits_seed{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)

        # Random split
        r_split = random_split(df, seed=seed)
        save_splits(r_split, split_dir / "random_split.json")

        # Scaffold split
        s_split = scaffold_split(df, seed=seed)
        save_splits(s_split, split_dir / "scaffold_split.json")

        # Target split
        t_split = target_split(df, seed=seed)
        save_splits(t_split, split_dir / "target_split.json")

        logger.info("Splits generated for seed=%d: %s", seed, split_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create dataset subsets for ablation studies",
    )
    parser.add_argument("--dataset-version", default="v1")
    subparsers = parser.add_subparsers(dest="command")

    # ESM-92 subset
    esm_parser = subparsers.add_parser("esm", help="Create ESM-92 target subset")
    esm_parser.add_argument("--output-tag", default="esm92")

    # Endpoint subset
    ep_parser = subparsers.add_parser("endpoint", help="Create endpoint subset")
    ep_parser.add_argument("--type", default="IC50", choices=["IC50", "Ki", "Kd"])

    # Generate splits
    split_parser = subparsers.add_parser("splits", help="Generate splits for subset")
    split_parser.add_argument("--subset-tag", required=True)
    split_parser.add_argument("--seeds", type=int, nargs="+", default=[42])

    # All: create ESM-92 subset + splits
    subparsers.add_parser("all", help="Create ESM-92 subset and generate splits")

    args = parser.parse_args()

    if args.command == "esm":
        df = create_esm_subset(args.dataset_version, args.output_tag)
        generate_splits_for_subset(args.dataset_version, args.output_tag)
    elif args.command == "endpoint":
        df = create_endpoint_subset(args.dataset_version, args.type)
        generate_splits_for_subset(args.dataset_version, args.type.lower())
    elif args.command == "splits":
        generate_splits_for_subset(args.dataset_version, args.subset_tag, args.seeds)
    elif args.command == "all":
        create_esm_subset(args.dataset_version)
        generate_splits_for_subset(args.dataset_version, "esm92")
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
