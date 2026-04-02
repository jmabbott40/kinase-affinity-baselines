# Kinase Inhibitor Affinity Prediction: Rigorous Baselines

**When do complex, structure-aware ML models outperform simple cheminformatics baselines for kinase inhibitor binding affinity prediction?**

## Motivation

Published ML models for protein–ligand affinity prediction often report impressive performance, but evaluations frequently rely on random train/test splits that allow data leakage through structural analogs. This project systematically evaluates whether protein-aware and structure-based models provide genuine improvements over simple cheminformatics baselines, using:

- **Rigorous data curation** from ChEMBL with documented inclusion criteria
- **Multiple splitting strategies** (random, scaffold, target-based) to reveal true generalization
- **Uncertainty quantification** to assess when predictions should be trusted
- **Precision-focused evaluation** relevant to real drug discovery prioritization

## Key Findings

> **Simple baselines are remarkably hard to beat.** Random Forest with Morgan fingerprints (87s training) matches or exceeds GPU-trained graph neural networks across all evaluation settings.

| Finding | Detail |
|---------|--------|
| Random split (easy) | ESM-FP MLP wins (RMSE=0.775), 5.3% better than RF (0.818) |
| Scaffold split (realistic) | ESM-FP MLP barely edges baselines (0.897 vs 0.905), <1% improvement |
| Target split (hardest) | **Random Forest wins** (RMSE=1.067), beating all neural models |
| Uncertainty | XGBoost quantile regression best calibrated (miscal=0.003); MC-Dropout uninformative |
| JAK selectivity | **Protein-aware models shine**: 79% top-1 accuracy vs 52% for fingerprint models |

See the [full project report](docs/project_report.md) for detailed analysis.

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

| Statistic | Value |
|-----------|-------|
| Raw records | 501K |
| Curated records | 353K |
| Unique compounds | 206K |
| Unique targets | 507 |
| Murcko scaffolds | 33,808 |
| Activity types | IC50 (80%), Ki (15%), Kd (5%) |
| Active (pActivity ≥ 6.0) | 77.3% |

See [`docs/data_card.md`](docs/data_card.md) for full dataset documentation.

## Models

### Baselines

| Model | Features | Uncertainty | Train Time |
|-------|----------|-------------|------------|
| Random Forest | Morgan FP (2048-bit) | Tree prediction variance | ~87s |
| XGBoost | Morgan FP (2048-bit) | Quantile regression | ~110s |
| ElasticNet | RDKit 2D descriptors | Bootstrap CI | ~59s |
| MLP | Morgan FP (2048-bit) | Ensemble variance (3 models) | ~650s |

### Advanced Neural Networks

| Model | Ligand Representation | Protein Representation | Uncertainty | Train Time |
|-------|----------------------|----------------------|-------------|------------|
| ESM-FP MLP | Morgan FP (2048-bit) | ESM-2 embeddings (1280-dim) | MC-Dropout | ~230s |
| GIN (Graph Isomorphism Network) | Molecular graph (atom/bond features) | — | MC-Dropout | ~1400s |
| GIN + ESM-2 Fusion | GIN molecular graph | ESM-2 embeddings (1280-dim) | MC-Dropout | ~1550s |

## Results Summary

### Regression Performance (RMSE, lower is better)

| Model | Random | Scaffold | Target |
|-------|--------|----------|--------|
| **ESM-FP MLP** | **0.775** | **0.897** | 1.177 |
| Fusion | 0.793 | 0.945 | 1.138 |
| Random Forest | 0.818 | 0.919 | **1.067** |
| MLP (baseline) | 0.824 | 0.905 | 1.090 |
| GIN | 0.829 | 0.941 | 1.146 |
| XGBoost | 0.893 | 0.961 | 1.135 |
| ElasticNet | 1.274 | 1.274 | 1.267 |

### Uncertainty Quality (Miscalibration Area, lower is better)

| Model | Random | Scaffold | Target |
|-------|--------|----------|--------|
| **XGBoost** | **0.017** | **0.003** | **0.048** |
| Random Forest | 0.120 | 0.037 | 0.166 |
| ESM-FP MLP | 0.167 | 0.175 | 0.186 |
| Fusion | 0.239 | 0.259 | 0.283 |
| MLP (baseline) | 0.256 | 0.252 | 0.301 |
| GIN | 0.317 | 0.322 | 0.227 |

### JAK Family Selectivity (Case Study)

| Model | Top-1 Accuracy | Rank Correlation |
|-------|---------------|-----------------|
| **Fusion** | **79.5%** | 0.744 |
| **ESM-FP MLP** | **78.5%** | **0.781** |
| MLP (baseline) | 51.8% | N/A |
| Random Forest | 51.6% | N/A |
| GIN | 49.8% | 0.064 |

## Evaluation

- **Regression**: RMSE, MAE, R², Pearson R, Spearman ρ
- **Classification**: AUROC, AUPRC, precision@k (at pActivity ≥ 6.0 threshold)
- **Splits**: random, scaffold (Murcko), target-based (kinase subfamily holdout)
- **Uncertainty**: calibration curves, selective prediction, error–uncertainty correlation
- **Case study**: JAK family (JAK1/2/3, TYK2) selectivity and per-target analysis

## Project Roadmap

- [x] Phase 1: Repository scaffolding and environment setup
- [x] Phase 2: Data pipeline (ChEMBL ingestion → curated dataset)
- [x] Phase 3: Feature engineering (Morgan FP, RDKit descriptors)
- [x] Phase 4: Baseline models (RF, XGBoost, ElasticNet, MLP) — 12 experiments
- [x] Phase 5: Evaluation and uncertainty analysis — calibration, selective prediction, error analysis
- [x] Phase 6: Case study — JAK kinase subfamily deep dive with selectivity analysis
- [x] Phase 7: Advanced models — GIN, ESM-2 protein embeddings, GNN+ESM fusion

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
# Phase 2: Data pipeline
python -m kinase_affinity.data.fetch              # Fetch raw data from ChEMBL
python -m kinase_affinity.data.curate             # Standardize and curate dataset

# Phase 3: Feature engineering
python -m kinase_affinity.features.fingerprints   # Morgan fingerprints (2048-bit)
python -m kinase_affinity.features.descriptors    # RDKit 2D descriptors

# Phase 4: Baseline models (CPU)
python -m kinase_affinity.training.trainer --all

# Phase 5: Evaluation and uncertainty analysis
python -m kinase_affinity.evaluation.run_phase5
python -m kinase_affinity.training.tune --all     # Hyperparameter tuning

# Phase 6: JAK case study
python scripts/run_phase6_case_study.py

# Phase 7: Advanced models (GPU required)
python -m kinase_affinity.data.protein_sequences      # Fetch kinase sequences
python -m kinase_affinity.features.protein_embeddings  # Compute ESM-2 embeddings
python -m kinase_affinity.training.deep_trainer --all  # Train GIN, ESM-FP MLP, Fusion
```

## Repository Structure

```
├── configs/           # YAML configs for dataset curation and model hyperparameters
├── data/
│   ├── raw/           # Raw ChEMBL exports (gitignored)
│   └── processed/     # Versioned curated datasets (gitignored)
├── notebooks/         # Analysis and visualization notebooks (01-05)
├── scripts/           # AWS runner scripts, Phase 6 case study
├── src/kinase_affinity/
│   ├── data/          # Ingestion, standardization, curation, splits, protein sequences
│   ├── features/      # Morgan FP, RDKit descriptors, molecular graphs, ESM-2 embeddings
│   ├── models/        # RF, XGBoost, ElasticNet, MLP, GIN, ESM-FP MLP, Fusion
│   ├── training/      # Baseline trainer, deep trainer, hyperparameter tuning
│   ├── evaluation/    # Metrics, uncertainty calibration, error analysis
│   └── visualization/ # Plotting utilities (heatmaps, calibration, degradation curves)
├── tests/             # Unit tests (32+ tests across all phases)
├── results/
│   ├── predictions/   # .npz files (21 experiments: 7 models × 3 splits)
│   ├── models/        # Saved model weights (7.7 GB)
│   ├── tables/        # Metrics CSVs, per-target breakdowns, Phase 5/6/7 summaries
│   └── figures/       # 110+ plots (scatter, calibration, heatmaps, degradation, JAK)
└── docs/              # Data card and project report
```

## Acknowledgments

- Bioactivity data from [ChEMBL](https://www.ebi.ac.uk/chembl/) (Mendez et al., 2019)
- Molecular processing with [RDKit](https://www.rdkit.org/)
- Protein embeddings from [ESM-2](https://github.com/facebookresearch/esm) (Lin et al., 2023)
- Graph neural networks with [PyTorch Geometric](https://pyg.org/)

## License

MIT — see [LICENSE](LICENSE).
