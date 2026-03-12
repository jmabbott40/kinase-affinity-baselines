# Kinase Inhibitor Affinity Prediction: Rigorous Baselines

**When do complex, structure-aware ML models outperform simple cheminformatics baselines for kinase inhibitor binding affinity prediction?**

## Motivation

Published ML models for protein–ligand affinity prediction often report impressive performance, but evaluations frequently rely on random train/test splits that allow data leakage through structural analogs. This project systematically evaluates whether protein-aware and structure-based models provide genuine improvements over simple cheminformatics baselines, using:

- **Rigorous data curation** from ChEMBL with documented inclusion criteria
- **Multiple splitting strategies** (random, scaffold, target-based) to reveal true generalization
- **Uncertainty quantification** to assess when predictions should be trusted
- **Precision-focused evaluation** relevant to real drug discovery prioritization

## Scientific Questions

1. How well can ligand-only fingerprint features predict kinase inhibitor binding affinity?
2. How much does performance degrade under scaffold and target-based splits vs. random splits?
3. Can model uncertainty estimates reliably identify unreliable predictions?
4. What chemical and biological features make some predictions systematically harder?
5. Do structure-aware models (GNNs, protein embeddings) provide genuine improvement when evaluated rigorously?

## Dataset

Protein kinase bioactivity data curated from [ChEMBL](https://www.ebi.ac.uk/chembl/):

- **Activity types**: IC50, Ki, Kd converted to pActivity (−log₁₀ M)
- **Quality filters**: exact measurements only, assay confidence ≥ 7, pChEMBL available
- **Standardization**: RDKit salt removal, charge neutralization, canonical SMILES
- **Duplicate handling**: median aggregation with noise flagging

See [`docs/data_card.md`](docs/data_card.md) for full dataset documentation.

## Models

### Baselines (Phase 1)

| Model | Features | Uncertainty |
|-------|----------|-------------|
| Random Forest | Morgan FP (2048-bit) | Tree prediction variance |
| XGBoost | Morgan FP (2048-bit) | Quantile regression |
| ElasticNet | RDKit 2D descriptors | Bootstrap CI |
| MLP | Morgan FP (2048-bit) | Ensemble variance |

### Advanced (Phase 2 — future)

- Ligand GNN (PyTorch Geometric)
- Protein embeddings (ESM-2) + ligand fingerprint fusion
- Docking-derived features + ML

## Evaluation

- **Regression**: RMSE, MAE, R², Pearson R, Spearman ρ
- **Classification**: AUROC, AUPRC, precision@k (at pActivity ≥ 6.0 threshold)
- **Splits**: random, scaffold (Murcko), target-based (kinase subfamily holdout)
- **Uncertainty**: calibration curves, selective prediction, error–uncertainty correlation

## Project Roadmap

- [x] Phase 1: Repository scaffolding and environment setup
- [ ] Phase 2: Data pipeline (ChEMBL ingestion → curated dataset)
- [ ] Phase 3: Feature engineering (Morgan FP, RDKit descriptors)
- [ ] Phase 4: Baseline models (RF, XGBoost, ElasticNet, MLP)
- [ ] Phase 5: Evaluation and uncertainty analysis
- [ ] Phase 6: Case studies and scientific report
- [ ] Phase 7: Advanced models (GNN, protein embeddings)

## Quick Start

### Prerequisites

Install [Miniforge](https://github.com/conda-forge/miniforge):

```bash
# macOS (Apple Silicon or Intel)
brew install miniforge
conda init zsh  # or bash
# restart your shell
```

### Setup

```bash
git clone https://github.com/jmabbott40/kinase-affinity-baselines.git
cd kinase-affinity-baselines

# Create and activate environment
conda env create -f environment.yml
conda activate kinase-affinity

# Install package in editable mode
pip install -e .

# Run tests
pytest tests/
```

### Usage

```bash
# Fetch raw data from ChEMBL
python -m kinase_affinity.data.fetch

# Standardize molecules and curate dataset
python -m kinase_affinity.data.curate

# Generate features
python -m kinase_affinity.features.fingerprints
python -m kinase_affinity.features.descriptors

# Train baseline models
python -m kinase_affinity.training.trainer --config configs/rf_baseline.yaml
```

## Repository Structure

```
├── configs/           # YAML configs for dataset curation and model hyperparameters
├── data/
│   ├── raw/           # Raw ChEMBL exports (gitignored)
│   └── processed/     # Versioned curated datasets (gitignored)
├── notebooks/         # Analysis and visualization notebooks
├── src/kinase_affinity/
│   ├── data/          # Ingestion, standardization, curation, splits
│   ├── features/      # Morgan FP, RDKit descriptors
│   ├── models/        # Model implementations
│   ├── training/      # Training loop and config loading
│   ├── evaluation/    # Metrics, uncertainty, error analysis
│   └── visualization/ # Plotting utilities
├── tests/             # Unit tests
├── results/           # Output tables and figures (gitignored figures)
└── docs/              # Data card and project report
```

## Acknowledgments

- Bioactivity data from [ChEMBL](https://www.ebi.ac.uk/chembl/) (Mendez et al., 2019)
- Molecular processing with [RDKit](https://www.rdkit.org/)

## License

MIT — see [LICENSE](LICENSE).
