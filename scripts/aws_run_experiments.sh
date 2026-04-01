#!/usr/bin/env bash
# =============================================================================
# Run all baseline experiments on AWS
#
# Usage (run ON the AWS instance):
#   bash scripts/aws_run_experiments.sh
#
# This runs all 12 experiments (4 models × 3 splits) and saves a summary.
# With AWS compute, expect:
#   - RF:         ~2-5 min per split (vs ~7 min on M-series Mac)
#   - XGBoost:    ~1-3 min per split (benefits most from multi-core)
#   - ElasticNet: ~1 min per split
#   - MLP:        ~10-20 min per split (benefits from faster CPU/more RAM)
#
# Total: ~1-2 hours depending on instance type
# =============================================================================
set -euo pipefail

# Activate environment
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate kinase-affinity

cd "$(dirname "$0")/.."

echo "============================================="
echo "  Running All Baseline Experiments"
echo "  $(date)"
echo "============================================="
echo ""

# Clean up any previous partial results from saved models
# (metrics JSON files from GitHub are kept)
echo "Cleaning previous model checkpoints..."
rm -rf results/models/ results/predictions/

echo "Starting experiment pipeline..."
python -m kinase_affinity.training.trainer --all 2>&1 | tee results/experiment_log.txt

echo ""
echo "============================================="
echo "  All experiments complete!"
echo "  $(date)"
echo "  "
echo "  Results saved to:"
echo "    results/tables/phase4_summary.csv"
echo "    results/tables/*_metrics.json"
echo "    results/experiment_log.txt"
echo "============================================="
