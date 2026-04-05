#!/bin/bash
# Preprint revision experiments — all compute for addressing reviewer gaps.
#
# This script runs on AWS (GPU instance) and covers:
#   1. P0-A: 92-target clean subset benchmark (21 experiments)
#   2. P0-B: Fallback ablation — zero and mean strategies (18 experiments)
#   3. P1-A: Multi-seed training for all models (105 experiments)
#   4. P2-A: Selectivity baselines (runs on CPU, fast)
#   5. P2-B: Supplement table generation
#
# Usage:
#   bash scripts/aws_run_revision.sh           # Run everything
#   bash scripts/aws_run_revision.sh subset    # Just the 92-target subset
#   bash scripts/aws_run_revision.sh fallback  # Just the fallback ablation
#   bash scripts/aws_run_revision.sh seeds     # Just multi-seed training
#   bash scripts/aws_run_revision.sh select    # Just selectivity baselines

set -euo pipefail

# Activate conda environment
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate kinase-affinity

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

PHASE="${1:-all}"
TRAINING_SEEDS=(42 123 456 789 1024)

echo "============================================================"
echo "PREPRINT REVISION EXPERIMENTS — Phase: $PHASE"
echo "Start time: $(date)"
echo "============================================================"

# -----------------------------------------------------------------------
# P0-A: 92-target clean subset benchmark
# -----------------------------------------------------------------------
run_subset() {
    echo ""
    echo "============================================================"
    echo "P0-A: Creating 92-target ESM subset and generating splits"
    echo "============================================================"
    python -m kinase_affinity.data.subset all --dataset-version v1

    echo ""
    echo "--- Training baselines on ESM-92 subset ---"
    for config in configs/rf_baseline.yaml configs/xgb_baseline.yaml \
                  configs/elasticnet_baseline.yaml configs/mlp_baseline.yaml; do
        for split in random scaffold target; do
            echo "  > $(basename $config .yaml) / $split (esm92)"
            python -m kinase_affinity.training.trainer \
                --config "$config" --split "$split" \
                --output-suffix "_esm92" \
                --dataset-version v1 || echo "  FAILED: $config / $split"
        done
    done

    echo ""
    echo "--- Training deep models on ESM-92 subset ---"
    for config in configs/esm_fp_mlp.yaml configs/gnn.yaml configs/fusion.yaml; do
        for split in random scaffold target; do
            echo "  > $(basename $config .yaml) / $split (esm92)"
            python -m kinase_affinity.training.deep_trainer \
                --config "$config" --split "$split" \
                --output-suffix "_esm92" \
                --dataset-version v1 || echo "  FAILED: $config / $split"
        done
    done

    echo "P0-A complete: $(date)"
}

# -----------------------------------------------------------------------
# P0-B: Fallback ablation
# -----------------------------------------------------------------------
run_fallback() {
    echo ""
    echo "============================================================"
    echo "P0-B: Fallback ablation (zero + mean strategies)"
    echo "============================================================"

    for strategy in zero mean; do
        echo ""
        echo "--- Fallback strategy: $strategy ---"
        for config in configs/esm_fp_mlp.yaml configs/fusion.yaml; do
            for split in random scaffold target; do
                echo "  > $(basename $config .yaml) / $split / fallback=$strategy"
                python -m kinase_affinity.training.deep_trainer \
                    --config "$config" --split "$split" \
                    --fallback-strategy "$strategy" \
                    --output-suffix "_fb_${strategy}" \
                    --dataset-version v1 || echo "  FAILED"
            done
        done

        # GNN doesn't use embeddings, but run with suffix for completeness
        echo "  (GNN skipped — does not use protein embeddings)"
    done

    echo "P0-B complete: $(date)"
}

# -----------------------------------------------------------------------
# P1-A: Multi-seed training
# -----------------------------------------------------------------------
run_seeds() {
    echo ""
    echo "============================================================"
    echo "P1-A: Multi-seed training (${#TRAINING_SEEDS[@]} seeds)"
    echo "============================================================"

    for seed in "${TRAINING_SEEDS[@]}"; do
        echo ""
        echo "--- Seed: $seed ---"

        # Baselines
        for config in configs/rf_baseline.yaml configs/xgb_baseline.yaml \
                      configs/elasticnet_baseline.yaml configs/mlp_baseline.yaml; do
            for split in random scaffold target; do
                echo "  > $(basename $config .yaml) / $split / seed=$seed"
                python -m kinase_affinity.training.trainer \
                    --config "$config" --split "$split" \
                    --training-seed "$seed" \
                    --output-suffix "_seed${seed}" \
                    --dataset-version v1 || echo "  FAILED"
            done
        done

        # Deep models
        for config in configs/esm_fp_mlp.yaml configs/gnn.yaml configs/fusion.yaml; do
            for split in random scaffold target; do
                echo "  > $(basename $config .yaml) / $split / seed=$seed"
                python -m kinase_affinity.training.deep_trainer \
                    --config "$config" --split "$split" \
                    --training-seed "$seed" \
                    --output-suffix "_seed${seed}" \
                    --dataset-version v1 || echo "  FAILED"
            done
        done

        echo "  Seed $seed complete: $(date)"
    done

    # Aggregate results
    echo ""
    echo "--- Aggregating multi-seed results ---"
    python -m kinase_affinity.evaluation.multi_seed_analysis \
        --seeds "${TRAINING_SEEDS[@]}" || echo "Aggregation failed"

    echo "P1-A complete: $(date)"
}

# -----------------------------------------------------------------------
# P2-A: Selectivity baselines
# -----------------------------------------------------------------------
run_selectivity() {
    echo ""
    echo "============================================================"
    echo "P2-A: Selectivity baselines"
    echo "============================================================"

    python scripts/run_selectivity_baselines.py --split scaffold
    python scripts/run_selectivity_baselines.py --split random

    echo "P2-A complete: $(date)"
}

# -----------------------------------------------------------------------
# P2-B: Supplement tables
# -----------------------------------------------------------------------
run_supplement() {
    echo ""
    echo "============================================================"
    echo "P2-B: Generating supplement tables"
    echo "============================================================"

    python scripts/generate_supplement_tables.py --dataset-version v1

    echo "P2-B complete: $(date)"
}

# -----------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------
case "$PHASE" in
    subset)     run_subset ;;
    fallback)   run_fallback ;;
    seeds)      run_seeds ;;
    select)     run_selectivity ;;
    supplement) run_supplement ;;
    all)
        run_subset
        run_fallback
        run_seeds
        run_selectivity
        run_supplement
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 {all|subset|fallback|seeds|select|supplement}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "REVISION EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo "============================================================"
