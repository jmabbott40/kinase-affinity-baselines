"""Generate supplementary tables for the preprint.

Extracts reproducibility-critical details from the codebase and data
into self-contained tables suitable for a supplementary methods section.

Tables generated:
    S1: Target-family assignment for target split
    S2: Hyperparameter search spaces and selected values
    S3: Training budget and compute per model
    S4: Dataset composition by endpoint type and split
    S5: ESM-2 embedding coverage by target family

Usage:
    python scripts/generate_supplement_tables.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "supplement_tables"


def table_s1_target_family_assignments(dataset_version: str = "v1") -> pd.DataFrame:
    """S1: Which kinase families are in train/val/test for target split.

    Reads the target split indices and curated data to determine
    which families were assigned to which fold.
    """
    data_dir = DATA_DIR / dataset_version
    df = pd.read_parquet(data_dir / "curated_activities.parquet")

    split_path = data_dir / "splits" / "target_split.json"
    if not split_path.exists():
        logger.warning("Target split not found at %s", split_path)
        return pd.DataFrame()

    with open(split_path) as f:
        split = json.load(f)

    rows = []
    for fold_name, indices in split.items():
        subset = df.iloc[indices]
        family_counts = subset.groupby("kinase_group").agg(
            n_targets=("target_chembl_id", "nunique"),
            n_records=("target_chembl_id", "count"),
        ).reset_index()
        family_counts["fold"] = fold_name
        rows.append(family_counts)

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["fold", "n_records"], ascending=[True, False])

    # Also create a target-level table
    target_rows = []
    for fold_name, indices in split.items():
        subset = df.iloc[indices]
        targets = subset.groupby(["target_chembl_id", "pref_name", "gene_symbol", "kinase_group"]).size().reset_index(name="n_records")
        targets["fold"] = fold_name
        target_rows.append(targets)

    target_result = pd.concat(target_rows, ignore_index=True)
    target_result = target_result.sort_values(["fold", "kinase_group", "n_records"], ascending=[True, True, False])

    return result, target_result


def table_s2_hyperparameter_details(dataset_version: str = "v1") -> pd.DataFrame:
    """S2: Search spaces, selected values, and selection criteria."""
    rows = []

    # RF — no search, fixed params
    rows.extend([
        {"model": "Random Forest", "parameter": "n_estimators", "search_space": "500 (fixed)", "selected": "500", "criterion": "N/A"},
        {"model": "Random Forest", "parameter": "max_depth", "search_space": "None (fixed)", "selected": "None (unlimited)", "criterion": "N/A"},
        {"model": "Random Forest", "parameter": "max_features", "search_space": "sqrt (fixed)", "selected": "sqrt", "criterion": "N/A"},
        {"model": "Random Forest", "parameter": "min_samples_leaf", "search_space": "2 (fixed)", "selected": "2", "criterion": "N/A"},
    ])

    # XGBoost — tuned
    rows.extend([
        {"model": "XGBoost", "parameter": "max_depth", "search_space": "[4, 6, 8, 10, 12]", "selected": "TBD*", "criterion": "Val RMSE"},
        {"model": "XGBoost", "parameter": "learning_rate", "search_space": "[0.05, 0.1]", "selected": "TBD*", "criterion": "Val RMSE"},
        {"model": "XGBoost", "parameter": "n_estimators", "search_space": "[300, 500]", "selected": "TBD*", "criterion": "Val RMSE"},
        {"model": "XGBoost", "parameter": "subsample", "search_space": "0.8 (fixed)", "selected": "0.8", "criterion": "N/A"},
        {"model": "XGBoost", "parameter": "colsample_bytree", "search_space": "0.8 (fixed)", "selected": "0.8", "criterion": "N/A"},
    ])

    # ElasticNet — tuned
    rows.extend([
        {"model": "ElasticNet", "parameter": "alpha", "search_space": "[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]", "selected": "TBD*", "criterion": "Val RMSE"},
        {"model": "ElasticNet", "parameter": "l1_ratio", "search_space": "[0.1, 0.3, 0.5, 0.7, 0.9]", "selected": "TBD*", "criterion": "Val RMSE"},
        {"model": "ElasticNet", "parameter": "max_iter", "search_space": "10000 (fixed)", "selected": "10000", "criterion": "N/A"},
    ])

    # MLP baseline
    rows.extend([
        {"model": "MLP (baseline)", "parameter": "hidden_layers", "search_space": "[256, 128] (fixed)", "selected": "[256, 128]", "criterion": "N/A"},
        {"model": "MLP (baseline)", "parameter": "n_ensemble", "search_space": "3 (fixed)", "selected": "3", "criterion": "N/A"},
        {"model": "MLP (baseline)", "parameter": "max_iter", "search_space": "200 (fixed)", "selected": "200", "criterion": "N/A"},
    ])

    # ESM-FP MLP
    rows.extend([
        {"model": "ESM-FP MLP", "parameter": "hidden_dims", "search_space": "[512, 256, 128] (fixed)", "selected": "[512, 256, 128]", "criterion": "N/A"},
        {"model": "ESM-FP MLP", "parameter": "dropout", "search_space": "0.3 (fixed)", "selected": "0.3", "criterion": "N/A"},
        {"model": "ESM-FP MLP", "parameter": "learning_rate", "search_space": "0.001 (fixed)", "selected": "0.001", "criterion": "N/A"},
        {"model": "ESM-FP MLP", "parameter": "batch_size", "search_space": "1024 (fixed)", "selected": "1024", "criterion": "N/A"},
        {"model": "ESM-FP MLP", "parameter": "max_epochs", "search_space": "100 (fixed)", "selected": "100", "criterion": "Early stop (patience=10)"},
    ])

    # GIN
    rows.extend([
        {"model": "GIN", "parameter": "gnn_layers", "search_space": "3 (fixed)", "selected": "3", "criterion": "N/A"},
        {"model": "GIN", "parameter": "hidden_dim", "search_space": "128 (fixed)", "selected": "128", "criterion": "N/A"},
        {"model": "GIN", "parameter": "dropout", "search_space": "0.3 (fixed)", "selected": "0.3", "criterion": "N/A"},
        {"model": "GIN", "parameter": "learning_rate", "search_space": "0.001 (fixed)", "selected": "0.001", "criterion": "N/A"},
    ])

    # Fusion
    rows.extend([
        {"model": "Fusion", "parameter": "gnn_layers", "search_space": "3 (fixed)", "selected": "3", "criterion": "N/A"},
        {"model": "Fusion", "parameter": "protein_projection_dim", "search_space": "256 (fixed)", "selected": "256", "criterion": "N/A"},
        {"model": "Fusion", "parameter": "learning_rate", "search_space": "0.0005 (fixed)", "selected": "0.0005", "criterion": "N/A"},
        {"model": "Fusion", "parameter": "max_epochs", "search_space": "100 (fixed)", "selected": "100", "criterion": "Early stop (patience=15)"},
    ])

    # Try to fill in TBD* from tuning results
    tuning_dir = RESULTS_DIR / "tuning"
    if tuning_dir.exists():
        for tuning_file in tuning_dir.glob("*_best_params.json"):
            try:
                with open(tuning_file) as f:
                    best = json.load(f)
                model_name = tuning_file.stem.replace("_best_params", "")
                for row in rows:
                    if row["selected"] == "TBD*":
                        param = row["parameter"]
                        if param in best:
                            row["selected"] = str(best[param])
            except Exception:
                pass

    return pd.DataFrame(rows)


def table_s3_training_compute() -> pd.DataFrame:
    """S3: Training time and compute for each model."""
    rows = []

    # Load from metric JSONs
    tables_dir = RESULTS_DIR / "tables"
    for json_path in sorted(tables_dir.glob("*_metrics.json")):
        try:
            with open(json_path) as f:
                m = json.load(f)
            rows.append({
                "model": m.get("model", json_path.stem),
                "split": m.get("split", "unknown"),
                "train_time_s": m.get("train_time_seconds", None),
                "best_epoch": m.get("best_epoch", "N/A"),
                "n_train": m.get("n_train", None),
                "n_test": m.get("n_test", None),
            })
        except Exception:
            pass

    if not rows:
        logger.warning("No metric JSONs found in %s", tables_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "split"])
    return df


def table_s4_endpoint_composition(dataset_version: str = "v1") -> pd.DataFrame:
    """S4: IC50/Ki/Kd breakdown by split fold."""
    data_dir = DATA_DIR / dataset_version
    df = pd.read_parquet(data_dir / "curated_activities.parquet")

    rows = []
    for split_name in ["random", "scaffold", "target"]:
        split_path = data_dir / "splits" / f"{split_name}_split.json"
        if not split_path.exists():
            continue

        with open(split_path) as f:
            split = json.load(f)

        for fold_name, indices in split.items():
            subset = df.iloc[indices]
            type_counts = subset["standard_type"].value_counts()
            total = len(subset)

            rows.append({
                "split": split_name,
                "fold": fold_name,
                "n_records": total,
                "n_IC50": int(type_counts.get("IC50", 0)),
                "n_Ki": int(type_counts.get("Ki", 0)),
                "n_Kd": int(type_counts.get("Kd", 0)),
                "pct_IC50": round(type_counts.get("IC50", 0) / total * 100, 1),
                "pct_active": round(subset["is_active"].mean() * 100, 1),
                "pactivity_mean": round(float(subset["pactivity"].mean()), 2),
                "pactivity_std": round(float(subset["pactivity"].std()), 2),
            })

    return pd.DataFrame(rows)


def table_s5_esm_coverage(dataset_version: str = "v1") -> pd.DataFrame:
    """S5: ESM-2 embedding coverage by kinase family.

    Shows which families have good vs. poor sequence coverage.
    """
    data_dir = DATA_DIR / dataset_version
    df = pd.read_parquet(data_dir / "curated_activities.parquet")

    # Load target index if available
    target_index_path = data_dir / "features" / "target_index.json"
    if not target_index_path.exists():
        logger.warning("target_index.json not found — ESM coverage table unavailable")
        return pd.DataFrame()

    with open(target_index_path) as f:
        target_to_row = json.load(f)

    targets_with_emb = set(target_to_row.keys())

    # Group by kinase family
    target_info = df.groupby("target_chembl_id").agg(
        kinase_group=("kinase_group", "first"),
        n_records=("pactivity", "count"),
    ).reset_index()

    target_info["has_embedding"] = target_info["target_chembl_id"].isin(targets_with_emb)

    # Aggregate by family
    family_stats = target_info.groupby("kinase_group").agg(
        total_targets=("target_chembl_id", "count"),
        targets_with_emb=("has_embedding", "sum"),
        total_records=("n_records", "sum"),
    ).reset_index()

    family_stats["targets_without_emb"] = family_stats["total_targets"] - family_stats["targets_with_emb"]
    family_stats["coverage_pct"] = (family_stats["targets_with_emb"] / family_stats["total_targets"] * 100).round(1)

    # Records with embeddings per family
    target_info_emb = target_info[target_info["has_embedding"]]
    records_with = target_info_emb.groupby("kinase_group")["n_records"].sum().reset_index()
    records_with.columns = ["kinase_group", "records_with_emb"]

    family_stats = family_stats.merge(records_with, on="kinase_group", how="left")
    family_stats["records_with_emb"] = family_stats["records_with_emb"].fillna(0).astype(int)
    family_stats["record_coverage_pct"] = (family_stats["records_with_emb"] / family_stats["total_records"] * 100).round(1)

    family_stats = family_stats.sort_values("total_records", ascending=False)

    return family_stats


def generate_all_tables(dataset_version: str = "v1") -> None:
    """Generate all supplement tables and save to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating supplement tables to %s", OUTPUT_DIR)

    # S1: Target family assignments
    logger.info("\n--- Table S1: Target Family Assignments ---")
    try:
        family_df, target_df = table_s1_target_family_assignments(dataset_version)
        family_df.to_csv(OUTPUT_DIR / "S1_target_family_assignments.csv", index=False)
        target_df.to_csv(OUTPUT_DIR / "S1_target_level_assignments.csv", index=False)
        logger.info("Saved S1 (%d family rows, %d target rows)",
                    len(family_df), len(target_df))
    except Exception as e:
        logger.warning("S1 failed: %s", e)

    # S2: Hyperparameters
    logger.info("\n--- Table S2: Hyperparameter Details ---")
    hp_df = table_s2_hyperparameter_details(dataset_version)
    hp_df.to_csv(OUTPUT_DIR / "S2_hyperparameters.csv", index=False)
    logger.info("Saved S2 (%d rows)", len(hp_df))

    # S3: Training compute
    logger.info("\n--- Table S3: Training Compute ---")
    compute_df = table_s3_training_compute()
    if not compute_df.empty:
        compute_df.to_csv(OUTPUT_DIR / "S3_training_compute.csv", index=False)
        logger.info("Saved S3 (%d rows)", len(compute_df))

    # S4: Endpoint composition
    logger.info("\n--- Table S4: Endpoint Composition ---")
    endpoint_df = table_s4_endpoint_composition(dataset_version)
    endpoint_df.to_csv(OUTPUT_DIR / "S4_endpoint_composition.csv", index=False)
    logger.info("Saved S4 (%d rows)", len(endpoint_df))

    # S5: ESM coverage
    logger.info("\n--- Table S5: ESM-2 Embedding Coverage ---")
    esm_df = table_s5_esm_coverage(dataset_version)
    if not esm_df.empty:
        esm_df.to_csv(OUTPUT_DIR / "S5_esm_coverage.csv", index=False)
        logger.info("Saved S5 (%d rows)", len(esm_df))

    logger.info("\nAll supplement tables generated.")


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate supplement tables")
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args()
    generate_all_tables(args.dataset_version)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
