#!/usr/bin/env bash
# =============================================================================
# Run Phase 5 analysis on AWS
#
# Usage (run ON the AWS instance):
#   bash scripts/aws_run_phase5.sh
#
# Steps:
#   1. Re-run all 12 experiments to generate prediction .npz files
#   2. Run hyperparameter tuning for ElasticNet and XGBoost
#   3. Re-run tuned models with best hyperparameters
#   4. Run Phase 5 analysis (uncertainty, error analysis, plots)
#
# Total time: ~2-3 hours
# =============================================================================
set -euo pipefail

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate kinase-affinity

cd "$(dirname "$0")/.."

echo "============================================="
echo "  Phase 5: Evaluation & Uncertainty Analysis"
echo "  $(date)"
echo "============================================="

# --- Step 1: Re-run experiments to generate .npz prediction files ---
echo ""
echo ">>> Step 1: Running all experiments to generate prediction files..."
echo "    (Prediction .npz files are needed for uncertainty analysis)"
python -m kinase_affinity.training.trainer --all 2>&1 | tee results/phase5_experiments_log.txt

echo ""
echo ">>> Step 1 complete. Checking prediction files..."
ls -lh results/predictions/*.npz 2>/dev/null || echo "WARNING: No .npz files found!"

# --- Step 2: Hyperparameter tuning ---
echo ""
echo ">>> Step 2: Tuning ElasticNet and XGBoost hyperparameters..."
python -m kinase_affinity.training.tune --all 2>&1 | tee results/phase5_tuning_log.txt

echo ""
echo ">>> Step 2 complete. Best parameters:"
cat results/tuning/best_*.json 2>/dev/null || echo "No tuning results found"

# --- Step 3: Run Phase 5 analysis ---
echo ""
echo ">>> Step 3: Running Phase 5 analysis (uncertainty, error analysis, plots)..."
python -m kinase_affinity.evaluation.run_phase5 2>&1 | tee results/phase5_analysis_log.txt

echo ""
echo "============================================="
echo "  Phase 5 complete!"
echo "  $(date)"
echo "  "
echo "  Results saved to:"
echo "    results/tables/phase5_summary.csv"
echo "    results/tables/phase5_*_*.json"
echo "    results/tables/per_target_*.csv"
echo "    results/tables/worst_predictions_*.csv"
echo "    results/figures/*.png"
echo "    results/tuning/"
echo "  "
echo "  To pull results back to local:"
echo "    rsync -avz -e 'ssh -i KEY' ubuntu@HOST:mlproject/results/ results/"
echo "============================================="
