"""Stronger selectivity baselines for the JAK case study.

Addresses reviewer Gap #4: the pooled ligand-only vs. protein-aware
selectivity comparison is asymmetric because ligand-only models can't
encode target identity. This script adds three stronger non-protein
baselines:

1. **Per-target models**: Separate RF/MLP trained per JAK member.
   At prediction time, predict with all 4 models and compare.
   This is the strongest non-protein selectivity baseline.

2. **One-hot target-conditioned**: Pooled model with Morgan FP + 4-dim
   one-hot target indicator. Tests whether simple target identity
   (without ESM-2 sequence info) suffices for selectivity.

3. **Pairwise delta models**: Predict ΔpActivity between JAK pairs.
   Directly models selectivity rather than absolute affinity.

Usage:
    python scripts/run_selectivity_baselines.py
    python scripts/run_selectivity_baselines.py --split scaffold
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from kinase_affinity.features import load_morgan_fingerprints

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

JAK_TARGETS = {
    "CHEMBL2835": "JAK1",
    "CHEMBL2971": "JAK2",
    "CHEMBL2148": "JAK3",
    "CHEMBL3553": "TYK2",
}
JAK_TARGET_IDS = sorted(JAK_TARGETS.keys())
JAK_NAMES = [JAK_TARGETS[t] for t in JAK_TARGET_IDS]


def _load_jak_data(
    dataset_version: str = "v1",
    split_strategy: str = "scaffold",
) -> tuple[pd.DataFrame, dict, np.ndarray, list[str]]:
    """Load JAK subset with features and splits.

    Returns
    -------
    tuple
        (jak_df, split_indices, fp_matrix, smiles_list)
    """
    data_dir = DATA_DIR / dataset_version
    df = pd.read_parquet(data_dir / "curated_activities.parquet")

    # Filter to JAK targets
    jak_df = df[df["target_chembl_id"].isin(JAK_TARGET_IDS)].copy()
    jak_df["jak_name"] = jak_df["target_chembl_id"].map(JAK_TARGETS)
    logger.info("JAK data: %d records across %d targets",
                len(jak_df), jak_df["target_chembl_id"].nunique())

    # Load splits
    with open(data_dir / "splits" / f"{split_strategy}_split.json") as f:
        full_split = json.load(f)

    # Map full-dataset indices to JAK rows
    jak_original_idx = set(jak_df.index.tolist())
    split_indices = {}
    for fold_name, indices in full_split.items():
        fold_idx = [i for i in indices if i in jak_original_idx]
        split_indices[fold_name] = fold_idx

    logger.info("JAK split sizes: train=%d, val=%d, test=%d",
                len(split_indices["train"]), len(split_indices["val"]),
                len(split_indices["test"]))

    # Load fingerprints
    fp_matrix, smiles_list = load_morgan_fingerprints(dataset_version)

    return jak_df, split_indices, fp_matrix, smiles_list


def _get_features(
    df: pd.DataFrame,
    indices: list[int],
    fp_matrix: np.ndarray,
    smiles_to_row: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Get fingerprint features and targets for a set of indices."""
    subset = df.loc[indices]
    fp_rows = np.array([smiles_to_row[s] for s in subset["std_smiles"].values])
    X = fp_matrix[fp_rows].astype(np.float32)
    y = subset["pactivity"].values
    return X, y


# -----------------------------------------------------------------------
# Baseline 1: Per-target models
# -----------------------------------------------------------------------

def run_per_target_baseline(
    jak_df: pd.DataFrame,
    split_indices: dict,
    fp_matrix: np.ndarray,
    smiles_list: list[str],
    model_type: str = "random_forest",
) -> dict:
    """Train separate models per JAK target, predict selectivity.

    For each JAK member, train a model on only that target's training data.
    At prediction time, predict with each model and compare predictions
    across targets for the same compound.

    Parameters
    ----------
    model_type : str
        "random_forest" or "mlp".

    Returns
    -------
    dict
        Selectivity metrics.
    """
    smiles_to_row = {s: i for i, s in enumerate(smiles_list)}
    models = {}

    # Train one model per JAK target
    for tid, tname in JAK_TARGETS.items():
        # Filter training data to this target
        train_idx = [
            i for i in split_indices["train"]
            if i in jak_df.index and jak_df.loc[i, "target_chembl_id"] == tid
        ]

        if len(train_idx) < 10:
            logger.warning("Skipping %s: only %d training samples", tname, len(train_idx))
            continue

        X_train, y_train = _get_features(jak_df, train_idx, fp_matrix, smiles_to_row)

        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=500, random_state=42, n_jobs=-1,
            )
        elif model_type == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128), max_iter=200,
                random_state=42, early_stopping=True,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        models[tid] = model
        logger.info("  Trained %s model for %s (%d samples)",
                    model_type, tname, len(train_idx))

    # Predict selectivity on test set
    return _evaluate_selectivity_per_target(
        models, jak_df, split_indices["test"], fp_matrix, smiles_to_row,
        model_label=f"per_target_{model_type}",
    )


def _evaluate_selectivity_per_target(
    models: dict,
    jak_df: pd.DataFrame,
    test_indices: list[int],
    fp_matrix: np.ndarray,
    smiles_to_row: dict[str, int],
    model_label: str = "per_target",
) -> dict:
    """Evaluate selectivity using per-target models.

    For each compound in test set that appears against 2+ JAK members:
    - Get true pActivity for each JAK
    - Predict with each target-specific model
    - Compare predicted ranking to true ranking
    """
    test_df = jak_df.loc[test_indices].copy()

    # Find compounds tested against multiple JAKs
    compound_targets = test_df.groupby("std_smiles")["target_chembl_id"].apply(set)
    multi_jak = compound_targets[compound_targets.apply(len) >= 2]
    logger.info("  Multi-JAK test compounds: %d", len(multi_jak))

    correct_top1 = 0
    rank_corrs = []
    n_evaluated = 0

    for smiles, targets in multi_jak.items():
        targets = sorted(targets)
        # Only use targets that have trained models
        targets = [t for t in targets if t in models]
        if len(targets) < 2:
            continue

        # Get fingerprint for this compound
        fp_row = smiles_to_row[smiles]
        X = fp_matrix[fp_row:fp_row + 1].astype(np.float32)

        # Get true pActivities
        compound_data = test_df[test_df["std_smiles"] == smiles]
        true_pacts = {}
        for t in targets:
            rows = compound_data[compound_data["target_chembl_id"] == t]
            if len(rows) > 0:
                true_pacts[t] = rows["pactivity"].values[0]

        if len(true_pacts) < 2:
            continue

        # Predict with each target's model
        pred_pacts = {}
        for t in targets:
            if t in true_pacts:
                pred_pacts[t] = float(models[t].predict(X)[0])

        # Evaluate
        true_best = max(true_pacts, key=true_pacts.get)
        pred_best = max(pred_pacts, key=pred_pacts.get)
        if true_best == pred_best:
            correct_top1 += 1

        n_evaluated += 1

        # Rank correlation (if >= 3 targets)
        if len(true_pacts) >= 3:
            common = sorted(true_pacts.keys())
            true_vals = [true_pacts[t] for t in common]
            pred_vals = [pred_pacts[t] for t in common]
            rho, _ = stats.spearmanr(true_vals, pred_vals)
            if not np.isnan(rho):
                rank_corrs.append(rho)

    top1_acc = correct_top1 / max(n_evaluated, 1)
    mean_rho = float(np.mean(rank_corrs)) if rank_corrs else float("nan")

    result = {
        "model": model_label,
        "n_compounds": n_evaluated,
        "top1_accuracy": round(top1_acc, 4),
        "mean_rank_corr": round(mean_rho, 4) if not np.isnan(mean_rho) else None,
        "n_with_rank_corr": len(rank_corrs),
    }
    logger.info("  %s: top1=%.1f%%, rho=%.3f, n=%d",
                model_label, top1_acc * 100, mean_rho, n_evaluated)
    return result


# -----------------------------------------------------------------------
# Baseline 2: One-hot target-conditioned model
# -----------------------------------------------------------------------

def run_onehot_baseline(
    jak_df: pd.DataFrame,
    split_indices: dict,
    fp_matrix: np.ndarray,
    smiles_list: list[str],
    model_type: str = "random_forest",
) -> dict:
    """Train a pooled model with FP + one-hot target encoding.

    Concatenates Morgan FP (2048) with a 4-dim one-hot vector indicating
    which JAK member the compound is tested against. This gives the model
    explicit target identity without protein sequence information.

    Parameters
    ----------
    model_type : str
        "random_forest" or "mlp".

    Returns
    -------
    dict
        Selectivity metrics.
    """
    smiles_to_row = {s: i for i, s in enumerate(smiles_list)}
    target_to_onehot_idx = {t: i for i, t in enumerate(JAK_TARGET_IDS)}

    def _build_onehot_features(indices):
        subset = jak_df.loc[indices]
        fp_rows = np.array([smiles_to_row[s] for s in subset["std_smiles"].values])
        X_fp = fp_matrix[fp_rows].astype(np.float32)

        # One-hot encode target
        onehot = np.zeros((len(subset), len(JAK_TARGET_IDS)), dtype=np.float32)
        for i, tid in enumerate(subset["target_chembl_id"].values):
            if tid in target_to_onehot_idx:
                onehot[i, target_to_onehot_idx[tid]] = 1.0

        X = np.concatenate([X_fp, onehot], axis=1)
        y = subset["pactivity"].values
        return X, y

    # Train
    X_train, y_train = _build_onehot_features(split_indices["train"])

    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1,
        )
    elif model_type == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128), max_iter=200,
            random_state=42, early_stopping=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    logger.info("  Trained %s one-hot model (%d features, %d samples)",
                model_type, X_train.shape[1], len(X_train))

    # Evaluate selectivity
    test_df = jak_df.loc[split_indices["test"]].copy()
    compound_targets = test_df.groupby("std_smiles")["target_chembl_id"].apply(set)
    multi_jak = compound_targets[compound_targets.apply(len) >= 2]
    logger.info("  Multi-JAK test compounds: %d", len(multi_jak))

    correct_top1 = 0
    rank_corrs = []
    n_evaluated = 0

    for smiles, targets in multi_jak.items():
        targets = sorted(targets)
        fp_row = smiles_to_row[smiles]
        X_fp = fp_matrix[fp_row:fp_row + 1].astype(np.float32)

        compound_data = test_df[test_df["std_smiles"] == smiles]
        true_pacts = {}
        pred_pacts = {}

        for tid in targets:
            rows = compound_data[compound_data["target_chembl_id"] == tid]
            if len(rows) == 0:
                continue
            true_pacts[tid] = rows["pactivity"].values[0]

            # Build input with one-hot for this target
            onehot = np.zeros((1, len(JAK_TARGET_IDS)), dtype=np.float32)
            onehot[0, target_to_onehot_idx[tid]] = 1.0
            X_input = np.concatenate([X_fp, onehot], axis=1)
            pred_pacts[tid] = float(model.predict(X_input)[0])

        if len(true_pacts) < 2:
            continue

        true_best = max(true_pacts, key=true_pacts.get)
        pred_best = max(pred_pacts, key=pred_pacts.get)
        if true_best == pred_best:
            correct_top1 += 1
        n_evaluated += 1

        if len(true_pacts) >= 3:
            common = sorted(true_pacts.keys())
            rho, _ = stats.spearmanr(
                [true_pacts[t] for t in common],
                [pred_pacts[t] for t in common],
            )
            if not np.isnan(rho):
                rank_corrs.append(rho)

    top1_acc = correct_top1 / max(n_evaluated, 1)
    mean_rho = float(np.mean(rank_corrs)) if rank_corrs else float("nan")

    result = {
        "model": f"onehot_{model_type}",
        "n_compounds": n_evaluated,
        "top1_accuracy": round(top1_acc, 4),
        "mean_rank_corr": round(mean_rho, 4) if not np.isnan(mean_rho) else None,
        "n_with_rank_corr": len(rank_corrs),
    }
    logger.info("  onehot_%s: top1=%.1f%%, rho=%.3f, n=%d",
                model_type, top1_acc * 100, mean_rho, n_evaluated)
    return result


# -----------------------------------------------------------------------
# Baseline 3: Pairwise delta model
# -----------------------------------------------------------------------

def run_pairwise_baseline(
    jak_df: pd.DataFrame,
    split_indices: dict,
    fp_matrix: np.ndarray,
    smiles_list: list[str],
    model_type: str = "random_forest",
) -> dict:
    """Train a model to predict ΔpActivity between JAK pairs.

    For each compound tested on 2+ JAKs, create training pairs:
        input = FP
        output = pActivity(JAK_A) - pActivity(JAK_B)

    This directly models selectivity.

    Returns
    -------
    dict
        Selectivity metrics.
    """
    smiles_to_row = {s: i for i, s in enumerate(smiles_list)}

    jak_pairs = list(itertools.combinations(JAK_TARGET_IDS, 2))
    logger.info("  Training pairwise models for %d target pairs", len(jak_pairs))

    pair_models = {}
    for t1, t2 in jak_pairs:
        pair_name = f"{JAK_TARGETS[t1]}_vs_{JAK_TARGETS[t2]}"

        # Build pairwise training data from training fold
        train_df = jak_df.loc[split_indices["train"]]

        # Find compounds tested on both targets in training set
        t1_data = train_df[train_df["target_chembl_id"] == t1].set_index("std_smiles")
        t2_data = train_df[train_df["target_chembl_id"] == t2].set_index("std_smiles")
        shared_smiles = t1_data.index.intersection(t2_data.index)

        if len(shared_smiles) < 20:
            logger.warning("  Skipping %s: only %d shared compounds", pair_name, len(shared_smiles))
            continue

        X_pair = np.array([
            fp_matrix[smiles_to_row[s]].astype(np.float32)
            for s in shared_smiles
        ])
        y_delta = np.array([
            t1_data.loc[s, "pactivity"].iloc[0] if isinstance(t1_data.loc[s, "pactivity"], pd.Series) else t1_data.loc[s, "pactivity"]
            - (t2_data.loc[s, "pactivity"].iloc[0] if isinstance(t2_data.loc[s, "pactivity"], pd.Series) else t2_data.loc[s, "pactivity"])
            for s in shared_smiles
        ])

        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        else:
            model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)

        model.fit(X_pair, y_delta)
        pair_models[(t1, t2)] = model
        logger.info("    %s: %d pairs, delta mean=%.2f, std=%.2f",
                    pair_name, len(shared_smiles), y_delta.mean(), y_delta.std())

    if not pair_models:
        return {"model": f"pairwise_{model_type}", "error": "no pairs trained"}

    # Evaluate selectivity using pairwise predictions
    test_df = jak_df.loc[split_indices["test"]].copy()
    compound_targets = test_df.groupby("std_smiles")["target_chembl_id"].apply(set)
    multi_jak = compound_targets[compound_targets.apply(len) >= 2]

    correct_top1 = 0
    rank_corrs = []
    n_evaluated = 0

    for smiles, targets in multi_jak.items():
        targets = sorted(targets)
        if len(targets) < 2:
            continue

        fp_row = smiles_to_row[smiles]
        X = fp_matrix[fp_row:fp_row + 1].astype(np.float32)

        compound_data = test_df[test_df["std_smiles"] == smiles]
        true_pacts = {}
        for tid in targets:
            rows = compound_data[compound_data["target_chembl_id"] == tid]
            if len(rows) > 0:
                true_pacts[tid] = rows["pactivity"].values[0]

        if len(true_pacts) < 2:
            continue

        # Reconstruct relative scores from pairwise deltas
        # Use a simple scoring approach: score(t) = sum of predicted deltas
        scores = {t: 0.0 for t in true_pacts}
        for t1, t2 in itertools.combinations(sorted(true_pacts.keys()), 2):
            pair_key = (t1, t2)
            rev_key = (t2, t1)
            if pair_key in pair_models:
                delta = float(pair_models[pair_key].predict(X)[0])
                scores[t1] += delta
                scores[t2] -= delta
            elif rev_key in pair_models:
                delta = float(pair_models[rev_key].predict(X)[0])
                scores[t2] += delta
                scores[t1] -= delta

        pred_best = max(scores, key=scores.get)
        true_best = max(true_pacts, key=true_pacts.get)
        if pred_best == true_best:
            correct_top1 += 1
        n_evaluated += 1

        if len(true_pacts) >= 3:
            common = sorted(true_pacts.keys())
            rho, _ = stats.spearmanr(
                [true_pacts[t] for t in common],
                [scores[t] for t in common],
            )
            if not np.isnan(rho):
                rank_corrs.append(rho)

    top1_acc = correct_top1 / max(n_evaluated, 1)
    mean_rho = float(np.mean(rank_corrs)) if rank_corrs else float("nan")

    result = {
        "model": f"pairwise_{model_type}",
        "n_compounds": n_evaluated,
        "top1_accuracy": round(top1_acc, 4),
        "mean_rank_corr": round(mean_rho, 4) if not np.isnan(mean_rho) else None,
        "n_with_rank_corr": len(rank_corrs),
    }
    logger.info("  pairwise_%s: top1=%.1f%%, rho=%.3f, n=%d",
                model_type, top1_acc * 100, mean_rho, n_evaluated)
    return result


# -----------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------

def run_all_selectivity_baselines(
    split_strategy: str = "scaffold",
    dataset_version: str = "v1",
) -> pd.DataFrame:
    """Run all selectivity baselines and save results.

    Parameters
    ----------
    split_strategy : str
        Split to evaluate on.
    dataset_version : str
        Dataset version.

    Returns
    -------
    pd.DataFrame
        Results table.
    """
    logger.info("=" * 70)
    logger.info("SELECTIVITY BASELINES — split=%s", split_strategy)
    logger.info("=" * 70)

    jak_df, split_indices, fp_matrix, smiles_list = _load_jak_data(
        dataset_version, split_strategy,
    )

    results = []

    # Per-target RF
    logger.info("\n--- Per-target Random Forest ---")
    results.append(run_per_target_baseline(
        jak_df, split_indices, fp_matrix, smiles_list, "random_forest",
    ))

    # Per-target MLP
    logger.info("\n--- Per-target MLP ---")
    results.append(run_per_target_baseline(
        jak_df, split_indices, fp_matrix, smiles_list, "mlp",
    ))

    # One-hot RF
    logger.info("\n--- One-hot RF ---")
    results.append(run_onehot_baseline(
        jak_df, split_indices, fp_matrix, smiles_list, "random_forest",
    ))

    # One-hot MLP
    logger.info("\n--- One-hot MLP ---")
    results.append(run_onehot_baseline(
        jak_df, split_indices, fp_matrix, smiles_list, "mlp",
    ))

    # Pairwise RF
    logger.info("\n--- Pairwise RF ---")
    results.append(run_pairwise_baseline(
        jak_df, split_indices, fp_matrix, smiles_list, "random_forest",
    ))

    # Save
    results_df = pd.DataFrame(results)
    results_df["split"] = split_strategy

    output_dir = RESULTS_DIR / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"selectivity_baselines_{split_strategy}.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("\nSaved results to %s", output_path)

    logger.info("\n" + "=" * 70)
    logger.info("SELECTIVITY BASELINES SUMMARY")
    logger.info("=" * 70)
    logger.info("\n%s", results_df[
        ["model", "n_compounds", "top1_accuracy", "mean_rank_corr"]
    ].to_string(index=False))

    return results_df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run stronger selectivity baselines for JAK case study",
    )
    parser.add_argument("--split", default="scaffold",
                        choices=["random", "scaffold"])
    parser.add_argument("--dataset-version", default="v1")
    args = parser.parse_args()

    run_all_selectivity_baselines(args.split, args.dataset_version)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
