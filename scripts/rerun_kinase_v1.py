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

import sys
import time
from pathlib import Path

import pandas as pd

from target_affinity_ml.training import train_and_evaluate

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


def config_path_for(model: str) -> Path:
    """Resolve config path: baselines have _baseline suffix, deep models do not."""
    if model in ("random_forest", "xgboost", "elasticnet", "mlp"):
        return CONFIGS_DIR / f"{model}_baseline.yaml"
    return CONFIGS_DIR / f"{model}.yaml"


def run_kinase_v1() -> pd.DataFrame:
    """Execute all 105 runs and write per-seed metrics CSV."""
    rows: list[dict] = []
    total_runs = len(MODELS) * len(SPLITS) * len(SEEDS)
    run_idx = 0
    overall_start = time.time()

    for model in MODELS:
        for split in SPLITS:
            for seed in SEEDS:
                run_idx += 1
                start = time.time()
                print(f"[{run_idx}/{total_runs}] {model} | {split} | seed={seed} ... ",
                      end="", flush=True)
                try:
                    result = train_and_evaluate(
                        config_path=config_path_for(model),
                        split_strategy=split,
                        dataset_version=DATASET_VERSION,
                        training_seed=seed,
                        output_suffix=f"_revalidation_seed{seed}",
                    )
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
    df = run_kinase_v1()
    if df["error"].notna().any():
        sys.exit(1)
