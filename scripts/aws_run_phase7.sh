#!/usr/bin/env bash
# =============================================================================
# Run Phase 7: Advanced Neural Network Models on AWS
#
# Usage (run ON the AWS instance):
#   bash scripts/aws_run_phase7.sh
#
# Prerequisites:
#   - Phase 5 complete (prediction .npz files exist)
#   - PyTorch, torch-geometric, fair-esm installed
#   - GPU available (check with nvidia-smi)
#
# Steps:
#   1. Install deep learning dependencies
#   2. Fetch protein sequences (ChEMBL + UniProt API)
#   3. Compute ESM-2 embeddings (GPU, ~30 min)
#   4. Run tests
#   5. Train all 9 experiments (3 models × 3 splits)
#   6. Generate comparison analysis
#
# Total time: ~6-12 hours depending on GPU
# =============================================================================
set -euo pipefail

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate kinase-affinity

cd "$(dirname "$0")/.."

echo "============================================="
echo "  Phase 7: Advanced Neural Network Models"
echo "  $(date)"
echo "============================================="

# --- Check GPU ---
echo ""
echo ">>> Checking GPU availability..."
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'Memory:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo ">>> Checking torch-geometric..."
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')" || {
    echo "ERROR: torch-geometric not installed. Run:"
    echo "  pip install torch-geometric"
    exit 1
}

echo ""
echo ">>> Checking ESM..."
python -c "import esm; print('ESM: OK')" || {
    echo "ERROR: fair-esm not installed. Run:"
    echo "  pip install fair-esm"
    exit 1
}

# --- Step 1: Fetch protein sequences ---
echo ""
echo ">>> Step 1: Fetching protein sequences from ChEMBL/UniProt..."
if [ -f "data/processed/v1/protein_sequences.json" ]; then
    echo "    Protein sequences already cached, skipping"
else
    python -m kinase_affinity.data.protein_sequences 2>&1 | tee results/phase7_sequences_log.txt
fi

# --- Step 2: Compute ESM-2 embeddings ---
echo ""
echo ">>> Step 2: Computing ESM-2 protein embeddings..."
if [ -f "data/processed/v1/features/esm2_embeddings.npz" ]; then
    echo "    ESM-2 embeddings already cached, skipping"
else
    python -m kinase_affinity.features.protein_embeddings 2>&1 | tee results/phase7_embeddings_log.txt
fi

# --- Step 3: Run tests ---
echo ""
echo ">>> Step 3: Running deep model tests..."
python -m pytest tests/test_deep_models.py -v --tb=short 2>&1 | tail -30

# --- Step 4: Train all deep models ---
echo ""
echo ">>> Step 4: Training all deep models (3 models × 3 splits = 9 experiments)..."
python -m kinase_affinity.training.deep_trainer --all 2>&1 | tee results/phase7_training_log.txt

# --- Step 5: Summary ---
echo ""
echo "============================================="
echo "  Phase 7 complete!"
echo "  $(date)"
echo "  "
echo "  Results saved to:"
echo "    results/tables/phase7_summary.csv"
echo "    results/tables/{esm_fp_mlp,gnn,fusion}_*_metrics.json"
echo "    results/predictions/{esm_fp_mlp,gnn,fusion}_*.npz"
echo "    results/figures/"
echo "  "
echo "  To pull results back to local:"
echo "    rsync -avz -e 'ssh -i KEY' ubuntu@HOST:mlproject/results/ results/"
echo "============================================="
