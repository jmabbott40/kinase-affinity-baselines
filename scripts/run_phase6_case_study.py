#!/usr/bin/env python3
"""Phase 6: JAK Family Case Study — Deep Dive Analysis.

Performs a comprehensive analysis of model performance on the JAK kinase
subfamily (JAK1, JAK2, JAK3, TYK2), examining:
    1. Per-target model comparison across all 7 models
    2. Activity cliff detection via Tanimoto similarity
    3. Worst prediction failure mode analysis
    4. Selectivity prediction accuracy
    5. Easy vs hard target characterization

Usage:
    python scripts/run_phase6_case_study.py

Outputs:
    results/tables/phase6_jak_*.csv
    results/figures/phase6_jak_*.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed/v1")
RESULTS_DIR = Path("results")
PRED_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ── Constants ──────────────────────────────────────────────────────────
JAK_TARGETS = {
    "CHEMBL2835": "JAK1",
    "CHEMBL2971": "JAK2",
    "CHEMBL2148": "JAK3",
    "CHEMBL3553": "TYK2",
}
JAK_GENES = ["JAK1", "JAK2", "JAK3", "TYK2"]

ALL_MODELS = [
    "random_forest", "xgboost", "elasticnet", "mlp",
    "esm_fp_mlp", "gnn", "fusion",
]
# Only random and scaffold; target split has 0 JAK test compounds
CASE_SPLITS = ["random", "scaffold"]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "elasticnet": "ElasticNet",
    "mlp": "MLP (baseline)",
    "esm_fp_mlp": "ESM-FP MLP",
    "gnn": "GIN",
    "fusion": "GIN+ESM Fusion",
}

MODEL_COLORS = {
    "random_forest": "#2196F3",
    "xgboost": "#4CAF50",
    "elasticnet": "#FF9800",
    "mlp": "#9C27B0",
    "esm_fp_mlp": "#E91E63",
    "gnn": "#00BCD4",
    "fusion": "#FF5722",
}

JAK_COLORS = {
    "JAK1": "#1f77b4",
    "JAK2": "#ff7f0e",
    "JAK3": "#2ca02c",
    "TYK2": "#d62728",
}


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load curated activities and split indices."""
    df = pd.read_parquet(DATA_DIR / "curated_activities.parquet")
    splits = {}
    for split in CASE_SPLITS:
        with open(DATA_DIR / "splits" / f"{split}_split.json") as f:
            splits[split] = json.load(f)
    return df, splits


def load_predictions(model: str, split: str) -> dict[str, np.ndarray]:
    """Load .npz prediction arrays."""
    path = PRED_DIR / f"{model}_{split}.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}


def extract_jak_predictions(df, splits, model, split):
    """Get JAK-specific true/pred values from a model's test predictions."""
    test_idx = splits[split]["test"]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    preds = load_predictions(model, split)
    y_true = preds["y_test_true"]
    y_pred = preds["y_test_pred"]
    y_std = preds["y_test_std"] if "y_test_std" in preds else None

    # Filter to JAK
    jak_mask = test_df["gene_symbol"].isin(JAK_GENES)
    jak_df = test_df[jak_mask].copy()
    jak_df["y_true"] = y_true[jak_mask.values]
    jak_df["y_pred"] = y_pred[jak_mask.values]
    jak_df["abs_error"] = np.abs(jak_df["y_true"] - jak_df["y_pred"])
    if y_std is not None:
        jak_df["y_std"] = y_std[jak_mask.values]

    return jak_df


# ═══════════════════════════════════════════════════════════════════════
# 1. Per-Target Model Comparison
# ═══════════════════════════════════════════════════════════════════════

def compute_per_target_metrics(jak_df):
    """Compute regression metrics per JAK target from a single experiment."""
    from kinase_affinity.evaluation.metrics import compute_regression_metrics

    results = []
    for gene in JAK_GENES:
        sub = jak_df[jak_df["gene_symbol"] == gene]
        if len(sub) < 5:
            continue
        metrics = compute_regression_metrics(
            sub["y_true"].values, sub["y_pred"].values,
        )
        metrics["gene_symbol"] = gene
        metrics["n_compounds"] = len(sub)
        results.append(metrics)
    return results


def run_per_target_comparison(df, splits):
    """Compare all 7 models on each JAK member."""
    logger.info("=" * 60)
    logger.info("STEP 1: Per-target model comparison")
    logger.info("=" * 60)

    all_rows = []
    for split in CASE_SPLITS:
        for model in ALL_MODELS:
            jak_df = extract_jak_predictions(df, splits, model, split)
            metrics_list = compute_per_target_metrics(jak_df)
            for m in metrics_list:
                m["model"] = model
                m["split"] = split
                all_rows.append(m)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(TABLES_DIR / "phase6_jak_per_target.csv", index=False)
    logger.info("  Saved per-target metrics: %d rows", len(results_df))

    # Generate heatmaps
    for split in CASE_SPLITS:
        _plot_jak_heatmap(results_df, split)

    return results_df


def _plot_jak_heatmap(results_df, split):
    """Heatmap: model × JAK target colored by RMSE."""
    sub = results_df[results_df["split"] == split]
    pivot = sub.pivot(index="model", columns="gene_symbol", values="rmse")

    model_order = [m for m in ALL_MODELS if m in pivot.index]
    col_order = [g for g in JAK_GENES if g in pivot.columns]
    pivot = pivot.loc[model_order, col_order]

    display_labels = [MODEL_DISPLAY.get(m, m) for m in model_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(display_labels, fontsize=10)

    # Separator between baselines and deep models
    n_baselines = sum(1 for m in model_order if m in
                      {"random_forest", "xgboost", "elasticnet", "mlp"})
    if n_baselines < len(model_order):
        ax.axhline(n_baselines - 0.5, color="white", linewidth=2.5)

    for i in range(len(model_order)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val > np.nanmedian(pivot.values)
                        else "black")

    # Gold border on best per column
    for j in range(len(col_order)):
        col = pivot.values[:, j]
        valid = ~np.isnan(col)
        if valid.any():
            best_idx = np.nanargmin(col)
            ax.add_patch(plt.Rectangle(
                (j - 0.5, best_idx - 0.5), 1, 1,
                fill=False, edgecolor="gold", linewidth=3,
            ))

    fig.colorbar(im, ax=ax, label="RMSE")
    ax.set_title(
        f"JAK Family: Model × Target RMSE ({split.capitalize()} Split)",
        fontsize=13,
    )
    fig.tight_layout()
    path = FIGURES_DIR / f"phase6_jak_heatmap_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved heatmap: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# 2. Activity Distribution Violins
# ═══════════════════════════════════════════════════════════════════════

def plot_activity_distributions(df):
    """Violin plots of pActivity for each JAK member."""
    logger.info("=" * 60)
    logger.info("STEP 2: Activity distributions")
    logger.info("=" * 60)

    jak = df[df["gene_symbol"].isin(JAK_GENES)].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(JAK_GENES))
    parts = ax.violinplot(
        [jak[jak["gene_symbol"] == g]["pactivity"].values for g in JAK_GENES],
        positions=positions, showmeans=True, showmedians=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(JAK_COLORS.values())[i])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(JAK_GENES, fontsize=12, fontweight="bold")
    ax.set_ylabel("pActivity", fontsize=11)
    ax.set_title("pActivity Distribution per JAK Member", fontsize=13)
    ax.axhline(6.0, color="red", linestyle="--", alpha=0.5,
               label="Active threshold (pActivity ≥ 6)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = FIGURES_DIR / "phase6_jak_violins.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# 3. Compound Overlap Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_compound_overlap(df):
    """Heatmap showing shared compounds between JAK members."""
    logger.info("=" * 60)
    logger.info("STEP 3: Compound overlap")
    logger.info("=" * 60)

    jak = df[df["gene_symbol"].isin(JAK_GENES)]
    sets = {g: set(jak[jak["gene_symbol"] == g]["std_smiles"]) for g in JAK_GENES}

    n = len(JAK_GENES)
    overlap = np.zeros((n, n), dtype=int)
    for i, g1 in enumerate(JAK_GENES):
        for j, g2 in enumerate(JAK_GENES):
            overlap[i, j] = len(sets[g1] & sets[g2])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap, cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_xticklabels(JAK_GENES, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(JAK_GENES, fontsize=11, fontweight="bold")

    for i in range(n):
        for j in range(n):
            pct = overlap[i, j] / min(overlap[i, i], overlap[j, j]) * 100
            label = f"{overlap[i, j]:,}\n({pct:.0f}%)" if i != j else f"{overlap[i, j]:,}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10,
                    color="white" if overlap[i, j] > np.median(overlap) else "black")

    fig.colorbar(im, ax=ax, label="Shared compounds")
    ax.set_title("Compound Overlap Between JAK Members", fontsize=13)
    fig.tight_layout()

    path = FIGURES_DIR / "phase6_jak_compound_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# 4. Activity Cliff Detection
# ═══════════════════════════════════════════════════════════════════════

def _tanimoto_matrix(fps):
    """Compute pairwise Tanimoto similarity for binary fingerprint matrix."""
    # fps: (n, d) uint8 array
    fps = fps.astype(np.float32)
    dot = fps @ fps.T
    norms = np.sum(fps, axis=1)
    denom = norms[:, None] + norms[None, :] - dot
    # Avoid division by zero
    denom = np.maximum(denom, 1e-10)
    return dot / denom


def detect_activity_cliffs(df, splits, split="random",
                           tanimoto_thresh=0.85, delta_thresh=1.5,
                           top_n=20):
    """Find activity cliff pairs in the JAK test set."""
    logger.info("=" * 60)
    logger.info("STEP 4: Activity cliff detection")
    logger.info("=" * 60)

    # Load fingerprints and index
    fp_data = np.load(DATA_DIR / "features" / "morgan_fp.npz")
    fps = fp_data["fingerprints"]
    with open(DATA_DIR / "features" / "smiles_index.json") as f:
        smiles_list = json.load(f)
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}

    # Get JAK test data
    test_idx = splits[split]["test"]
    test_df = df.iloc[test_idx].reset_index(drop=True)
    jak_test = test_df[test_df["gene_symbol"].isin(JAK_GENES)].copy()

    all_cliffs = []
    for gene in JAK_GENES:
        sub = jak_test[jak_test["gene_symbol"] == gene].copy()
        sub = sub.drop_duplicates(subset="std_smiles").reset_index(drop=True)

        # Get fingerprints for this subset
        fp_indices = []
        valid_rows = []
        for i, smi in enumerate(sub["std_smiles"]):
            if smi in smiles_to_idx:
                fp_indices.append(smiles_to_idx[smi])
                valid_rows.append(i)

        if len(fp_indices) < 2:
            continue

        sub = sub.iloc[valid_rows].reset_index(drop=True)
        sub_fps = fps[fp_indices]

        logger.info("  %s: computing Tanimoto for %d compounds...", gene, len(sub))
        sim = _tanimoto_matrix(sub_fps)

        # Find cliff pairs
        pacts = sub["pactivity"].values
        n = len(sub)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= tanimoto_thresh:
                    delta = abs(pacts[i] - pacts[j])
                    if delta >= delta_thresh:
                        all_cliffs.append({
                            "gene_symbol": gene,
                            "smiles_1": sub.iloc[i]["std_smiles"],
                            "smiles_2": sub.iloc[j]["std_smiles"],
                            "pactivity_1": pacts[i],
                            "pactivity_2": pacts[j],
                            "delta_pactivity": delta,
                            "tanimoto": sim[i, j],
                        })

    cliffs_df = pd.DataFrame(all_cliffs)
    if len(cliffs_df) > 0:
        cliffs_df = cliffs_df.sort_values("delta_pactivity", ascending=False)
        cliffs_df.to_csv(TABLES_DIR / "phase6_jak_activity_cliffs.csv", index=False)
        logger.info("  Found %d activity cliff pairs", len(cliffs_df))
        logger.info("  Top cliff: delta=%.2f, Tanimoto=%.3f (%s)",
                     cliffs_df.iloc[0]["delta_pactivity"],
                     cliffs_df.iloc[0]["tanimoto"],
                     cliffs_df.iloc[0]["gene_symbol"])
    else:
        logger.info("  No activity cliffs found")
        return cliffs_df

    # Plot: Tanimoto vs delta pActivity scatter (all pairs above tanimoto_thresh)
    _plot_cliff_scatter(df, splits, split, smiles_to_idx, fps, cliffs_df)

    return cliffs_df


def _plot_cliff_scatter(df, splits, split, smiles_to_idx, fps, cliffs_df):
    """Scatter: Tanimoto vs |ΔpActivity| for JAK test pairs."""
    test_idx = splits[split]["test"]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, gene in zip(axes.flat, JAK_GENES):
        sub = test_df[test_df["gene_symbol"] == gene].drop_duplicates(
            subset="std_smiles"
        ).reset_index(drop=True)

        fp_indices = []
        valid_rows = []
        for i, smi in enumerate(sub["std_smiles"]):
            if smi in smiles_to_idx:
                fp_indices.append(smiles_to_idx[smi])
                valid_rows.append(i)
        sub = sub.iloc[valid_rows].reset_index(drop=True)

        if len(sub) < 2:
            ax.set_title(f"{gene}: insufficient data")
            continue

        sub_fps = fps[fp_indices]
        sim = _tanimoto_matrix(sub_fps)
        pacts = sub["pactivity"].values
        n = len(sub)

        # Sample pairs for plotting (too many for full scatter)
        tani_vals = []
        delta_vals = []
        np.random.seed(42)
        ij_upper = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if sim[i, j] >= 0.5]
        if len(ij_upper) > 10000:
            sampled = np.random.choice(len(ij_upper), 10000, replace=False)
            ij_upper = [ij_upper[s] for s in sampled]

        for i, j in ij_upper:
            tani_vals.append(sim[i, j])
            delta_vals.append(abs(pacts[i] - pacts[j]))

        tani_vals = np.array(tani_vals)
        delta_vals = np.array(delta_vals)

        ax.scatter(tani_vals, delta_vals, alpha=0.15, s=4,
                   c=JAK_COLORS[gene], label="All pairs")

        # Highlight cliffs
        cliff_mask = (tani_vals >= 0.85) & (delta_vals >= 1.5)
        if cliff_mask.any():
            ax.scatter(tani_vals[cliff_mask], delta_vals[cliff_mask],
                       alpha=0.8, s=20, c="red", marker="^",
                       label=f"Cliffs (n={cliff_mask.sum()})")

        ax.axhline(1.5, color="red", linestyle="--", alpha=0.3)
        ax.axvline(0.85, color="red", linestyle="--", alpha=0.3)
        ax.set_xlabel("Tanimoto Similarity")
        ax.set_ylabel("|ΔpActivity|")
        ax.set_title(f"{gene} (n={len(sub)} compounds)", fontsize=12)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0.45, 1.02)

    fig.suptitle("Activity Cliffs in JAK Family (Random Split Test Set)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = FIGURES_DIR / "phase6_jak_activity_cliffs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# 5. Worst Prediction Deep Dive
# ═══════════════════════════════════════════════════════════════════════

def analyze_worst_predictions(df, splits, split="random", top_n=20):
    """Identify and characterize worst JAK predictions."""
    logger.info("=" * 60)
    logger.info("STEP 5: Worst prediction analysis")
    logger.info("=" * 60)

    all_worst = []
    for model in ALL_MODELS:
        jak_df = extract_jak_predictions(df, splits, model, split)
        jak_df = jak_df.sort_values("abs_error", ascending=False)
        top = jak_df.head(top_n).copy()
        top["model"] = model
        all_worst.append(top)

    worst_df = pd.concat(all_worst, ignore_index=True)

    save_cols = ["model", "gene_symbol", "std_smiles", "target_chembl_id",
                 "standard_type", "y_true", "y_pred", "abs_error",
                 "is_noisy", "pactivity_std", "n_measurements"]
    save_cols = [c for c in save_cols if c in worst_df.columns]
    worst_df[save_cols].to_csv(
        TABLES_DIR / "phase6_jak_worst_predictions.csv", index=False,
    )
    logger.info("  Saved %d worst predictions across %d models",
                 len(worst_df), len(ALL_MODELS))

    # Failure mode breakdown
    logger.info("  Failure mode analysis:")
    for model in ["random_forest", "esm_fp_mlp", "fusion"]:
        sub = worst_df[worst_df["model"] == model]
        n_noisy = sub["is_noisy"].sum() if "is_noisy" in sub.columns else 0
        by_gene = sub["gene_symbol"].value_counts().to_dict()
        by_type = sub["standard_type"].value_counts().to_dict() if "standard_type" in sub.columns else {}
        logger.info("    %s: noisy=%d, genes=%s, types=%s",
                     model, n_noisy, by_gene, by_type)

    # Scatter: true vs predicted for best model (RF) on JAK
    _plot_jak_scatter(df, splits, split)

    return worst_df


def _plot_jak_scatter(df, splits, split):
    """True vs predicted scatter for each JAK member, best baseline vs best deep."""
    models_to_plot = ["random_forest", "esm_fp_mlp"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for row, model in enumerate(models_to_plot):
        jak_df = extract_jak_predictions(df, splits, model, split)
        for col, gene in enumerate(JAK_GENES):
            ax = axes[row, col]
            sub = jak_df[jak_df["gene_symbol"] == gene]

            if len(sub) == 0:
                ax.set_visible(False)
                continue

            ax.hexbin(sub["y_true"], sub["y_pred"], gridsize=30,
                      cmap="Blues", mincnt=1, linewidths=0.2)
            lims = [
                min(sub["y_true"].min(), sub["y_pred"].min()) - 0.5,
                max(sub["y_true"].max(), sub["y_pred"].max()) + 0.5,
            ]
            ax.plot(lims, lims, "k--", alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Highlight worst 5
            worst = sub.nlargest(5, "abs_error")
            ax.scatter(worst["y_true"], worst["y_pred"],
                       c="red", s=40, zorder=5, marker="x", linewidths=2)

            from kinase_affinity.evaluation.metrics import compute_regression_metrics
            m = compute_regression_metrics(sub["y_true"].values, sub["y_pred"].values)
            ax.text(0.05, 0.95, f"RMSE={m['rmse']:.3f}\nR²={m['r2']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if row == 0:
                ax.set_title(gene, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{MODEL_DISPLAY[model]}\nPredicted pActivity",
                              fontsize=10)
            ax.set_xlabel("True pActivity" if row == 1 else "")

    fig.suptitle(f"JAK Family: Predicted vs Actual ({split.capitalize()} Split)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = FIGURES_DIR / f"phase6_jak_scatter_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# 6. Selectivity Prediction
# ═══════════════════════════════════════════════════════════════════════

def analyze_selectivity(df, splits, split="random"):
    """Test whether models correctly predict compound selectivity profiles."""
    logger.info("=" * 60)
    logger.info("STEP 6: Selectivity prediction analysis")
    logger.info("=" * 60)

    test_idx = splits[split]["test"]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    jak_test = test_df[test_df["gene_symbol"].isin(JAK_GENES)]

    # Find compounds present in test set for 2+ JAK members
    smi_gene_counts = jak_test.groupby("std_smiles")["gene_symbol"].nunique()
    multi_jak_smiles = set(smi_gene_counts[smi_gene_counts >= 2].index)
    logger.info("  Compounds in test set against 2+ JAKs: %d", len(multi_jak_smiles))

    smi_4jak = set(smi_gene_counts[smi_gene_counts == 4].index)
    logger.info("  Compounds in test set against all 4 JAKs: %d", len(smi_4jak))

    if len(multi_jak_smiles) < 10:
        logger.warning("  Too few multi-JAK compounds for selectivity analysis")
        return None

    # For each model, check selectivity ranking accuracy
    sel_results = []
    for model in ALL_MODELS:
        preds = load_predictions(model, split)
        y_pred_all = preds["y_test_pred"]

        correct_top1 = 0
        correct_rank = 0
        total = 0
        rank_correlations = []

        for smi in multi_jak_smiles:
            smi_rows = jak_test[jak_test["std_smiles"] == smi]
            if len(smi_rows) < 2:
                continue

            genes_present = smi_rows["gene_symbol"].values
            true_pacts = smi_rows["pactivity"].values
            pred_indices = smi_rows.index.values
            pred_pacts = y_pred_all[pred_indices]

            # Rank correlation
            if len(genes_present) >= 3:
                rho, _ = stats.spearmanr(true_pacts, pred_pacts)
                if not np.isnan(rho):
                    rank_correlations.append(rho)

            # Top-1 accuracy: does model predict most potent JAK correctly?
            true_best = genes_present[np.argmax(true_pacts)]
            pred_best = genes_present[np.argmax(pred_pacts)]
            if true_best == pred_best:
                correct_top1 += 1
            total += 1

        sel_results.append({
            "model": model,
            "split": split,
            "n_compounds": total,
            "top1_accuracy": correct_top1 / total if total > 0 else 0,
            "mean_rank_corr": np.mean(rank_correlations) if rank_correlations else np.nan,
            "median_rank_corr": np.median(rank_correlations) if rank_correlations else np.nan,
            "n_with_rank_corr": len(rank_correlations),
        })
        logger.info("  %s: top-1 acc=%.1f%%, mean rank ρ=%.3f (n=%d)",
                     model,
                     sel_results[-1]["top1_accuracy"] * 100,
                     sel_results[-1]["mean_rank_corr"],
                     sel_results[-1]["n_with_rank_corr"])

    sel_df = pd.DataFrame(sel_results)
    sel_df.to_csv(TABLES_DIR / "phase6_jak_selectivity.csv", index=False)

    # Plot selectivity bar chart
    _plot_selectivity_bars(sel_df, split)

    return sel_df


def _plot_selectivity_bars(sel_df, split):
    """Bar chart of selectivity prediction accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = [m for m in ALL_MODELS if m in sel_df["model"].values]
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#888") for m in models]
    labels = [MODEL_DISPLAY.get(m, m) for m in models]

    # Top-1 accuracy
    vals = [sel_df[sel_df["model"] == m]["top1_accuracy"].values[0] * 100
            for m in models]
    bars = ax1.bar(x, vals, color=colors, alpha=0.8, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax1.set_title("Most-Potent JAK Prediction", fontsize=12)
    ax1.set_ylim(0, 100)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.1f}%", ha="center", fontsize=9)

    # Separator line
    n_bl = sum(1 for m in models if m in
               {"random_forest", "xgboost", "elasticnet", "mlp"})
    if n_bl < len(models):
        ax1.axvline(n_bl - 0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(n_bl - 0.5, color="gray", linestyle="--", alpha=0.5)

    # Mean rank correlation
    vals2 = [sel_df[sel_df["model"] == m]["mean_rank_corr"].values[0]
             for m in models]
    bars2 = ax2.bar(x, vals2, color=colors, alpha=0.8, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Spearman ρ", fontsize=11)
    ax2.set_title("Selectivity Rank Correlation", fontsize=12)
    ax2.set_ylim(-0.1, 1.0)
    for bar, v in zip(bars2, vals2):
        if not np.isnan(v):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(f"JAK Selectivity Prediction ({split.capitalize()} Split)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIGURES_DIR / f"phase6_jak_selectivity_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# 7. Easy vs Hard Target Radar
# ═══════════════════════════════════════════════════════════════════════

def plot_target_difficulty_radar(df, per_target_df, split="random"):
    """Radar chart showing factors that make each JAK member easy or hard."""
    logger.info("=" * 60)
    logger.info("STEP 7: Target difficulty radar")
    logger.info("=" * 60)

    jak = df[df["gene_symbol"].isin(JAK_GENES)]

    properties = {}
    for gene in JAK_GENES:
        sub = jak[jak["gene_symbol"] == gene]
        # Average RMSE across all models for this target
        target_metrics = per_target_df[
            (per_target_df["gene_symbol"] == gene) &
            (per_target_df["split"] == split)
        ]
        avg_rmse = target_metrics["rmse"].mean() if len(target_metrics) > 0 else np.nan

        properties[gene] = {
            "N compounds": len(sub["std_smiles"].unique()),
            "Activity range (std)": sub["pactivity"].std(),
            "% Noisy": sub["is_noisy"].mean() * 100,
            "Mean pActivity": sub["pactivity"].mean(),
            "Avg RMSE (all models)": avg_rmse,
        }

    props_df = pd.DataFrame(properties).T
    logger.info("  Target properties:\n%s", props_df.to_string())

    # Normalized radar chart
    categories = list(props_df.columns)
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for gene in JAK_GENES:
        values = props_df.loc[gene].values.tolist()
        # Normalize each property to [0, 1] for radar
        norm_values = []
        for i, v in enumerate(values):
            col_min = props_df.iloc[:, i].min()
            col_max = props_df.iloc[:, i].max()
            if col_max - col_min > 0:
                norm_values.append((v - col_min) / (col_max - col_min))
            else:
                norm_values.append(0.5)
        norm_values += norm_values[:1]
        ax.plot(angles, norm_values, "o-", color=JAK_COLORS[gene],
                linewidth=2, label=gene, markersize=6)
        ax.fill(angles, norm_values, alpha=0.1, color=JAK_COLORS[gene])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("JAK Target Properties (Normalized)", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()

    path = FIGURES_DIR / "phase6_jak_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# 8. Performance Degradation: JAK vs Dataset-Wide
# ═══════════════════════════════════════════════════════════════════════

def compare_jak_vs_global(per_target_df):
    """Compare JAK subfamily RMSE to dataset-wide RMSE for each model."""
    logger.info("=" * 60)
    logger.info("STEP 8: JAK vs dataset-wide comparison")
    logger.info("=" * 60)

    # Load global metrics
    dfs = []
    for path_name in ["phase4_summary.csv", "phase7_summary.csv"]:
        p = TABLES_DIR / path_name
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        logger.warning("  No summary files found")
        return

    global_df = pd.concat(dfs, ignore_index=True)
    global_df = global_df.drop_duplicates(subset=["model", "split"], keep="last")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, split in zip(axes, CASE_SPLITS):
        models = [m for m in ALL_MODELS if m != "elasticnet"]
        x = np.arange(len(models))
        width = 0.15

        # Dataset-wide RMSE
        global_rmse = []
        for m in models:
            row = global_df[(global_df["model"] == m) &
                            (global_df["split"] == split)]
            global_rmse.append(row["test_rmse"].values[0] if len(row) > 0 else np.nan)
        ax.bar(x - 2.5 * width, global_rmse, width, label="Full dataset",
               color="gray", alpha=0.6)

        # Per-JAK RMSE
        for j, gene in enumerate(JAK_GENES):
            rmses = []
            for m in models:
                sub = per_target_df[
                    (per_target_df["model"] == m) &
                    (per_target_df["split"] == split) &
                    (per_target_df["gene_symbol"] == gene)
                ]
                rmses.append(sub["rmse"].values[0] if len(sub) > 0 else np.nan)
            ax.bar(x + (j - 1.5) * width, rmses, width, label=gene,
                   color=JAK_COLORS[gene], alpha=0.8)

        labels = [MODEL_DISPLAY.get(m, m) for m in models]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.set_title(f"{split.capitalize()} Split", fontsize=12)
        if split == "random":
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("JAK Members vs Dataset-Wide RMSE", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = FIGURES_DIR / "phase6_jak_vs_global.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path.name)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Run all Phase 6 analyses."""
    logger.info("=" * 70)
    logger.info("  PHASE 6: JAK FAMILY CASE STUDY")
    logger.info("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df, splits = load_data()

    # 1. Per-target model comparison
    per_target_df = run_per_target_comparison(df, splits)

    # 2. Activity distributions
    plot_activity_distributions(df)

    # 3. Compound overlap
    plot_compound_overlap(df)

    # 4. Activity cliffs
    cliffs_df = detect_activity_cliffs(df, splits, split="random")

    # 5. Worst predictions
    worst_df = analyze_worst_predictions(df, splits, split="random")

    # 6. Selectivity analysis
    for split in CASE_SPLITS:
        analyze_selectivity(df, splits, split=split)

    # 7. Target difficulty radar
    plot_target_difficulty_radar(df, per_target_df, split="random")

    # 8. JAK vs dataset-wide
    compare_jak_vs_global(per_target_df)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  Phase 6 complete!")
    logger.info("  Tables: %s/phase6_jak_*", TABLES_DIR)
    logger.info("  Figures: %s/phase6_jak_*", FIGURES_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
