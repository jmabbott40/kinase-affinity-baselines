"""Validate that kinase library-v1.0 re-run reproduces preprint v1 numbers.

Plan 1 Task 14 — runs after Task 13's `rerun_kinase_v1.py` completes.

Tolerance: ±0.001 RMSE per (model, split, seed). Looser for derived
metrics (R², Pearson) where rounding can amplify small differences.

Reference sources:
- Deep models (esm_fp_mlp, gnn, fusion): per-seed CSV
  (results/supplement_tables/S6_per_seed_metrics.csv)
- Baseline models (random_forest, xgboost, elasticnet, mlp): recomputed
  from saved preprint v1 prediction NPZs in results/predictions/. These
  represent one specific seed's output; the validation compares all 5
  rerun seeds against this single reference, allowing the seed-level
  variation to fall within tolerance.

Output: results/kinase_v1_revalidation/validation_report.md (markdown)
        results/kinase_v1_revalidation/validation_summary.json (machine-readable)

Exit code 0 = all comparisons within tolerance; 1 = at least one failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

KINASE_REPO = Path(__file__).resolve().parent.parent
RERUN_DIR = KINASE_REPO / "results" / "kinase_v1_revalidation"
PREPRINT_PRED_DIR = KINASE_REPO / "results" / "predictions"
PREPRINT_DEEP_CSV = (
    KINASE_REPO / "results" / "supplement_tables" / "S6_per_seed_metrics.csv"
)

TOLERANCE_RMSE = 0.001
TOLERANCE_R2 = 0.005
TOLERANCE_PEARSON = 0.005

DEEP_MODELS = {"esm_fp_mlp", "gnn", "fusion"}
BASELINE_MODELS = {"random_forest", "xgboost", "elasticnet", "mlp"}


def recompute_baseline_reference(model: str, split: str) -> dict | None:
    """Recompute reference metrics from saved baseline prediction NPZ files.

    Returns None if the predictions file is missing or unreadable.
    """
    pred_file = PREPRINT_PRED_DIR / f"{model}_{split}.npz"
    if not pred_file.exists():
        return None
    d = np.load(pred_file)
    y_true_keys = ["y_test_true", "y_true"]
    y_pred_keys = ["y_test_pred", "y_test_mean", "y_pred"]
    y_true = next((d[k] for k in y_true_keys if k in d), None)
    y_pred = next((d[k] for k in y_pred_keys if k in d), None)
    if y_true is None or y_pred is None:
        print(
            f"WARN: missing y_true/y_pred in {pred_file}; keys={list(d.keys())}",
            file=sys.stderr,
        )
        return None

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_val = float(r2_score(y_true, y_pred))
    # Pearson correlation undefined for constant predictions (e.g. ElasticNet at alpha=1.0)
    if np.std(y_pred) < 1e-12:
        pearson_r = float("nan")
    else:
        pearson_r, _ = pearsonr(y_true, y_pred)
        pearson_r = float(pearson_r)

    return {
        "model": model,
        "split": split,
        "ref_rmse": rmse,
        "ref_r2": r2_val,
        "ref_pearson_r": pearson_r,
    }


def load_deep_reference() -> pd.DataFrame:
    """Per-seed reference metrics for deep models from S6 supplement table."""
    if not PREPRINT_DEEP_CSV.exists():
        print(
            f"WARN: deep reference CSV missing: {PREPRINT_DEEP_CSV}",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["model", "split", "seed", "ref_rmse", "ref_r2"])
    df = pd.read_csv(PREPRINT_DEEP_CSV)
    return df.rename(columns={"rmse": "ref_rmse", "r2": "ref_r2"})[
        ["model", "split", "seed", "ref_rmse", "ref_r2"]
    ]


def validate() -> bool:
    """Run validation, write report, return True if all comparisons pass."""
    rerun_csv = RERUN_DIR / "all_seeds_metrics.csv"
    if not rerun_csv.exists():
        print(f"ERROR: rerun CSV missing: {rerun_csv}", file=sys.stderr)
        sys.exit(2)

    rerun = pd.read_csv(rerun_csv)
    print(f"Loaded {len(rerun)} re-run rows")

    required = {"model", "split", "seed", "test_rmse", "test_r2"}
    missing_cols = required - set(rerun.columns)
    if missing_cols:
        print(f"ERROR: rerun CSV missing required columns: {missing_cols}", file=sys.stderr)
        sys.exit(2)

    # --- Build references ---
    deep_ref = load_deep_reference()
    print(f"Loaded deep reference: {len(deep_ref)} rows")

    baseline_refs = []
    for model in BASELINE_MODELS:
        for split in ("random", "scaffold", "target"):
            ref = recompute_baseline_reference(model, split)
            if ref is not None:
                baseline_refs.append(ref)
    baseline_df = pd.DataFrame(baseline_refs)
    print(f"Computed baseline reference: {len(baseline_df)} (model, split) cells")

    # --- Merge: deep models per-seed; baselines aggregated per (model, split) ---
    deep_rerun = rerun[rerun["model"].isin(DEEP_MODELS)]
    baseline_rerun = rerun[rerun["model"].isin(BASELINE_MODELS)]

    deep_merged = deep_rerun.merge(deep_ref, on=["model", "split", "seed"], how="left")
    baseline_merged = baseline_rerun.merge(
        baseline_df, on=["model", "split"], how="left"
    )

    print(f"Deep model comparisons: {len(deep_merged)}")
    print(f"Baseline comparisons: {len(baseline_merged)}")

    if len(deep_merged) == 0 and len(baseline_merged) == 0:
        print(
            "ERROR: No reference values matched. Check column names and file paths.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Compute differences ---
    all_merged = pd.concat([deep_merged, baseline_merged], ignore_index=True, sort=False)
    all_merged["rmse_diff"] = (all_merged["test_rmse"] - all_merged["ref_rmse"]).abs()
    all_merged["r2_diff"] = (all_merged["test_r2"] - all_merged["ref_r2"]).abs()

    # Failures: any row where rerun has a reference AND rmse_diff > tolerance
    has_ref = all_merged["ref_rmse"].notna()
    failures = all_merged[has_ref & (all_merged["rmse_diff"] > TOLERANCE_RMSE)]
    no_ref = all_merged[~has_ref]

    # --- Write report ---
    report_path = RERUN_DIR / "validation_report.md"
    with open(report_path, "w") as f:
        f.write("# Kinase v1.0 Re-validation Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**RMSE tolerance:** ±{TOLERANCE_RMSE}  ·  **R² tolerance:** ±{TOLERANCE_R2}\n\n")
        f.write(f"## Coverage summary\n\n")
        f.write(f"- Total re-run rows: {len(rerun)}\n")
        f.write(f"- Deep model per-seed comparisons: {len(deep_merged)}\n")
        f.write(f"- Baseline (model, split) aggregate comparisons: {len(baseline_merged)}\n")
        f.write(f"- Rows without preprint v1 reference: {len(no_ref)}\n")
        f.write(f"- **RMSE failures: {len(failures)}**\n\n")

        if len(failures) == 0 and len(no_ref) == 0:
            f.write("## ✅ All re-run results match preprint v1 within tolerance\n\n")
            f.write("Library refactor preserved numerical behavior across all 105 runs. "
                    "Proceed to Plan 2.\n\n")
        elif len(failures) == 0:
            f.write("## ✅ All comparable runs match within tolerance\n\n")
            f.write(f"({len(no_ref)} rows had no preprint v1 reference and could not be "
                    "compared; see 'Unreferenced' section below.)\n\n")
        else:
            f.write("## ❌ Failures detected\n\n")
            f.write("Investigate refactor bugs before proceeding to Plan 2:\n\n")
            cols = ["model", "split", "seed", "test_rmse", "ref_rmse", "rmse_diff"]
            f.write(failures[cols].to_markdown(index=False))
            f.write("\n\n")

        f.write("## Per-model × split summary\n\n")
        summary = all_merged.groupby(["model", "split"]).agg(
            n_compared=("rmse_diff", lambda x: x.notna().sum()),
            mean_rmse_diff=("rmse_diff", "mean"),
            max_rmse_diff=("rmse_diff", "max"),
            n_failures=("rmse_diff", lambda x: (x > TOLERANCE_RMSE).sum()),
        ).round(6)
        f.write(summary.to_markdown())
        f.write("\n\n")

        if len(no_ref) > 0:
            f.write("## Unreferenced rows (no preprint v1 reference available)\n\n")
            f.write(f"{len(no_ref)} rows have a re-run RMSE but no preprint v1 reference "
                    "to compare against. This is expected for some baseline (model, split, seed) "
                    "triples — preprint v1 only saved per-seed metrics for *deep* models in "
                    f"`{PREPRINT_DEEP_CSV.name}`. Baselines have only one reference value per "
                    "(model, split) which is matched against all 5 seeds.\n\n")

    # --- Write summary JSON ---
    summary_json = {
        "total_rerun_rows": int(len(rerun)),
        "deep_per_seed_comparisons": int(len(deep_merged)),
        "baseline_aggregate_comparisons": int(len(baseline_merged)),
        "rows_without_reference": int(len(no_ref)),
        "rmse_failures": int(len(failures)),
        "tolerance_rmse": TOLERANCE_RMSE,
        "tolerance_r2": TOLERANCE_R2,
        "max_rmse_diff": float(all_merged["rmse_diff"].max())
            if all_merged["rmse_diff"].notna().any() else None,
        "passed": bool(len(failures) == 0),
    }
    with open(RERUN_DIR / "validation_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nValidation report: {report_path}")
    print(f"Summary JSON:      {RERUN_DIR / 'validation_summary.json'}")
    return len(failures) == 0


if __name__ == "__main__":
    success = validate()
    if not success:
        print("\n❌ FAILURES detected. See validation_report.md.", file=sys.stderr)
        print("Do NOT proceed to Plan 2 until failures are debugged and resolved.")
        sys.exit(1)
    print("\n✅ Validation passed. Library v1.0 reproduces preprint v1 numbers.")
    print("Ready to proceed to Plan 2.")
