#!/usr/bin/env bash
# =============================================================================
# Migrate kinase-affinity-baselines project to AWS EC2 (Ubuntu)
#
# Usage:
#   # 1. Edit the variables below with your AWS details
#   # 2. Run from your local machine (macOS):
#   bash scripts/migrate_to_aws.sh
#
# What this does:
#   - Transfers data files (140 MB) to the EC2 instance via rsync
#   - Runs remote setup: clone repo, install Miniforge, create conda env
#   - Verifies the installation by running tests
#
# Prerequisites:
#   - SSH access to your EC2 instance (key-based auth)
#   - Security group allows SSH (port 22)
# =============================================================================
set -euo pipefail

# ===================== EDIT THESE =====================
AWS_HOST="ubuntu@ec2-3-139-61-179.us-east-2.compute.amazonaws.com"      # e.g., ubuntu@54.123.45.67
SSH_KEY="/Users/joshuaabbott/downloads/jma_key.pem"         # e.g., $HOME/.ssh/kinase-project.pem
REMOTE_PROJECT_DIR="/home/ubuntu/mlproject"      # where to put the project on EC2
# ======================================================

LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================="
echo "  Kinase Affinity Baselines → AWS Migration"
echo "============================================="
echo "Local project:  $LOCAL_PROJECT_DIR"
echo "Remote host:    $AWS_HOST"
echo "Remote dir:     $REMOTE_PROJECT_DIR"
echo "SSH key:        $SSH_KEY"
echo ""

# --- Validate SSH connection ---
echo ">>> Testing SSH connection..."
ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$AWS_HOST" "echo 'SSH connection OK'" || {
    echo "ERROR: Cannot connect to $AWS_HOST. Check your IP, key, and security group."
    exit 1
}

# --- Step 1: Remote setup (clone repo, install Miniforge) ---
echo ""
echo ">>> Step 1: Setting up remote environment..."
ssh -i "$SSH_KEY" "$AWS_HOST" bash -s "$REMOTE_PROJECT_DIR" << 'REMOTE_SETUP'
set -euo pipefail
REMOTE_PROJECT_DIR="$1"

echo "--- Updating system packages ---"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl > /dev/null 2>&1

# --- Clone repo if not already present ---
if [ -d "$REMOTE_PROJECT_DIR/.git" ]; then
    echo "--- Repository already exists, pulling latest ---"
    cd "$REMOTE_PROJECT_DIR"
    git pull origin main
else
    echo "--- Cloning repository ---"
    git clone https://github.com/jmabbott40/kinase-affinity-baselines.git "$REMOTE_PROJECT_DIR"
    cd "$REMOTE_PROJECT_DIR"
fi

# --- Install Miniforge if conda not available ---
if command -v conda &> /dev/null; then
    echo "--- Conda already installed ---"
    conda --version
elif [ -f "$HOME/miniforge3/bin/conda" ]; then
    echo "--- Miniforge found at ~/miniforge3 ---"
else
    echo "--- Installing Miniforge ---"
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm /tmp/miniforge.sh
    echo "--- Miniforge installed ---"
fi

# Make conda available
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

# --- Create conda environment if it doesn't exist ---
if conda env list | grep -q "kinase-affinity"; then
    echo "--- Conda environment 'kinase-affinity' already exists ---"
else
    echo "--- Creating conda environment from environment.yml ---"
    cd "$REMOTE_PROJECT_DIR"
    conda env create -f environment.yml
    echo "--- Environment created ---"
fi

# --- Install project in dev mode ---
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate kinase-affinity
cd "$REMOTE_PROJECT_DIR"
pip install -e . --quiet 2>/dev/null || echo "(pip install -e . skipped — no setup.py/pyproject.toml entry point needed)"

echo ""
echo "--- Remote setup complete ---"
REMOTE_SETUP

# --- Step 2: Transfer data files ---
echo ""
echo ">>> Step 2: Transferring data files (140 MB)..."

# Create remote data directories
ssh -i "$SSH_KEY" "$AWS_HOST" "mkdir -p $REMOTE_PROJECT_DIR/data/raw $REMOTE_PROJECT_DIR/data/processed/v1/features $REMOTE_PROJECT_DIR/data/processed/v1/splits"

# rsync data files (excludes .DS_Store, .gitkeep, and saved models)
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude='.DS_Store' \
    --exclude='.gitkeep' \
    "$LOCAL_PROJECT_DIR/data/raw/" \
    "$AWS_HOST:$REMOTE_PROJECT_DIR/data/raw/"

rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude='.DS_Store' \
    --exclude='.gitkeep' \
    "$LOCAL_PROJECT_DIR/data/processed/" \
    "$AWS_HOST:$REMOTE_PROJECT_DIR/data/processed/"

echo "--- Data transfer complete ---"

# --- Step 3: Verify installation ---
echo ""
echo ">>> Step 3: Verifying installation..."
ssh -i "$SSH_KEY" "$AWS_HOST" bash -s "$REMOTE_PROJECT_DIR" << 'REMOTE_VERIFY'
set -euo pipefail
REMOTE_PROJECT_DIR="$1"
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate kinase-affinity

cd "$REMOTE_PROJECT_DIR"

echo "--- Python version ---"
python --version

echo "--- Key packages ---"
python -c "
import rdkit; print(f'RDKit:        {rdkit.__version__}')
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import xgboost; print(f'XGBoost:      {xgboost.__version__}')
import pandas; print(f'pandas:       {pandas.__version__}')
import numpy; print(f'NumPy:        {numpy.__version__}')
"

echo "--- Verifying data files ---"
python -c "
import pandas as pd
import numpy as np
import json

df = pd.read_parquet('data/processed/v1/curated_activities.parquet')
print(f'Curated activities: {len(df):,} records, {df[\"std_smiles\"].nunique():,} compounds')

fp = np.load('data/processed/v1/features/morgan_fp.npz')
print(f'Morgan FP:          {fp[\"fingerprints\"].shape}')

desc = np.load('data/processed/v1/features/rdkit_descriptors.npz')
print(f'RDKit descriptors:  {desc[\"descriptors\"].shape}')

for split in ['random', 'scaffold', 'target']:
    with open(f'data/processed/v1/splits/{split}_split.json') as f:
        s = json.load(f)
    print(f'{split} split:  train={len(s[\"train\"]):,} val={len(s[\"val\"]):,} test={len(s[\"test\"]):,}')
"

echo "--- Running tests ---"
python -m pytest tests/ -v --tb=short 2>&1 | tail -20

echo ""
echo "============================================="
echo "  Migration complete!"
echo "  "
echo "  To start working on the AWS instance:"
echo "    ssh -i $SSH_KEY $AWS_HOST"
echo "    conda activate kinase-affinity"
echo "    cd $REMOTE_PROJECT_DIR"
echo "  "
echo "  To run remaining MLP experiments:"
echo "    python -m kinase_affinity.training.trainer --config configs/mlp_baseline.yaml --split random"
echo "    python -m kinase_affinity.training.trainer --config configs/mlp_baseline.yaml --split scaffold"
echo "    python -m kinase_affinity.training.trainer --config configs/mlp_baseline.yaml --split target"
echo "  "
echo "  To re-run all 12 experiments (much faster with AWS compute):"
echo "    python -m kinase_affinity.training.trainer --all"
echo "============================================="
REMOTE_VERIFY

echo ""
echo ">>> Migration finished successfully!"
