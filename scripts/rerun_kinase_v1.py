"""Re-run the kinase benchmark with target-affinity-ml v1.0.

Plan 1 Task 13: validates that the library refactor preserved numerical
behavior across all 105 training runs (7 models x 3 splits x 5 seeds).

Uses identical seeds (42, 123, 456, 789, 1024) and identical configs
copied from the original kinase preprint v1 work. Output directory:
results/kinase_v1_revalidation/

Validation against preprint v1 happens in Task 14 via
scripts/validate_kinase_revalidation.py.

Estimated wall-clock: ~2 days at 4-way GPU parallelism.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from target_affinity_ml.training import train_and_evaluate
from target_affinity_ml.training.deep_trainer import deep_train_and_evaluate

# Models routed through the baseline trainer (config["features"]["type"] is a string)
BASELINE_MODELS = {"random_forest", "xgboost", "elasticnet", "mlp"}
# Models routed through the deep trainer (config["features"] has ligand/protein keys)
DEEP_MODELS = {"esm_fp_mlp", "gnn", "fusion"}

KINASE_REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = KINASE_REPO / "results" / "kinase_v1_revalidation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR = OUTPUT_DIR / "predictions"
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mirror the preprint v1 protocol exactly
MODELS = ["random_forest", "xgboost", "elasticnet", "mlp",
          "esm_fp_mlp", "gnn", "fusion"]
SPLITS = ["random", "scaffold", "target"]
SEEDS = [42, 123, 456, 789, 1024]

CONFIGS_DIR = KINASE_REPO / "configs"
DATASET_VERSION = "v1"


# The model registry uses long names ("random_forest", "xgboost") but
# the YAML files on disk use short names ("rf_baseline.yaml",
# "xgb_baseline.yaml") — this map keeps both consistent.
CONFIG_FILENAMES = {
    "random_forest": "rf_baseline.yaml",
    "xgboost": "xgb_baseline.yaml",
    "elasticnet": "elasticnet_baseline.yaml",
    "mlp": "mlp_baseline.yaml",
    "esm_fp_mlp": "esm_fp_mlp.yaml",
    "gnn": "gnn.yaml",
    "fusion": "fusion.yaml",
}


def config_path_for(model: str) -> Path:
    """Resolve a model's YAML config path (model registry name → on-disk filename)."""
    return CONFIGS_DIR / CONFIG_FILENAMES[model]


def _execute_one(model: str, split: str, seed: int) -> dict:
    """Dispatch to the correct trainer and return its result dict."""
    kwargs = dict(
        config_path=config_path_for(model),
        split_strategy=split,
        dataset_version=DATASET_VERSION,
        training_seed=seed,
        output_suffix=f"_revalidation_seed{seed}",
    )
    if model in DEEP_MODELS:
        return deep_train_and_evaluate(**kwargs)
    if model in BASELINE_MODELS:
        return train_and_evaluate(**kwargs)
    raise ValueError(f"Unknown model: {model!r}")


def _existing_successes(metrics_csv: Path) -> set[tuple[str, str, int]]:
    """Read prior partial results to skip already-successful (model, split, seed)
    triples. Used by --resume to avoid redoing baselines after a deep-model bug fix.
    """
    if not metrics_csv.exists():
        return set()
    df = pd.read_csv(metrics_csv)
    # Treat null/NaN error as success (the column may be empty for successful runs)
    successes = df[df["error"].isna() | (df["error"] == "")]
    return {(r["model"], r["split"], int(r["seed"])) for _, r in successes.iterrows()}


def run_kinase_v1(resume: bool = False) -> pd.DataFrame:
    """Execute all 105 runs and write per-seed metrics CSV.

    Parameters
    ----------
    resume : bool
        If True, read all_seeds_metrics.csv (if present) and skip rows that
        already succeeded. Useful after fixing a bug that affected only some
        models (e.g. deep-model dispatch bug) without redoing baselines.
    """
    metrics_csv = OUTPUT_DIR / "all_seeds_metrics.csv"
    skip_set: set[tuple[str, str, int]] = (
        _existing_successes(metrics_csv) if resume else set()
    )
    if skip_set:
        print(f"Resume mode: skipping {len(skip_set)} already-successful runs.")

    # Preserve prior successful rows so the final CSV is complete
    rows: list[dict] = []
    if resume and metrics_csv.exists():
        prev = pd.read_csv(metrics_csv)
        rows = prev[prev["error"].isna() | (prev["error"] == "")].to_dict("records")

    total_runs = len(MODELS) * len(SPLITS) * len(SEEDS)
    run_idx = 0
    overall_start = time.time()

    for model in MODELS:
        for split in SPLITS:
            for seed in SEEDS:
                run_idx += 1
                if (model, split, seed) in skip_set:
                    print(f"[{run_idx}/{total_runs}] {model} | {split} | seed={seed} ... SKIPPED (already done)")
                    continue

                start = time.time()
                print(f"[{run_idx}/{total_runs}] {model} | {split} | seed={seed} ... ",
                      end="", flush=True)
                try:
                    result = _execute_one(model, split, seed)
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"FAILED in {elapsed:.0f}s: {type(e).__name__}: {e}")
                    rows.append({
                        "model": model, "split": split, "seed": seed,
                        "test_rmse": None, "test_r2": None, "test_pearson_r": None,
                        "wallclock_seconds": elapsed,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    continue

                elapsed = time.time() - start
                # train_and_evaluate returns a flat metrics dict
                rows.append({
                    "model": model,
                    "split": split,
                    "seed": seed,
                    "test_rmse": result.get("test_rmse"),
                    "test_mae": result.get("test_mae"),
                    "test_r2": result.get("test_r2"),
                    "test_pearson_r": result.get("test_pearson_r"),
                    "test_spearman_rho": result.get("test_spearman_rho"),
                    "test_auroc": result.get("test_auroc"),
                    "wallclock_seconds": elapsed,
                    "error": None,
                })
                print(f"OK in {elapsed:.0f}s, RMSE={result.get('test_rmse'):.4f}")

                # Periodic progress save (in case the process crashes mid-run)
                if run_idx % 10 == 0:
                    pd.DataFrame(rows).to_csv(
                        OUTPUT_DIR / "all_seeds_metrics_partial.csv",
                        index=False,
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "all_seeds_metrics.csv", index=False)
    # Drop the partial file
    partial = OUTPUT_DIR / "all_seeds_metrics_partial.csv"
    if partial.exists():
        partial.unlink()

    overall_elapsed = time.time() - overall_start
    print()
    print(f"All {total_runs} runs complete in {overall_elapsed/3600:.1f}h.")
    print(f"Per-seed metrics: {OUTPUT_DIR / 'all_seeds_metrics.csv'}")
    print(f"Predictions output suffix: _revalidation_seed{{42,123,456,789,1024}}")

    n_errors = sum(1 for r in rows if r.get("error"))
    if n_errors:
        print(f"WARNING: {n_errors}/{total_runs} runs failed. See 'error' column.")
        return df

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (model, split, seed) triples that already succeeded in "
             "results/kinase_v1_revalidation/all_seeds_metrics.csv.",
    )
    args = parser.parse_args()

    df = run_kinase_v1(resume=args.resume)
    if df["error"].notna().any():
        sys.exit(1)
