> **Library extraction notice (April 2026):** The reusable model + training + evaluation code
> from this repository has been extracted into a separate library:
> [`target-affinity-ml`](https://github.com/jmabbott40/target-affinity-ml). This repo will be
> **frozen at v1.0** as the artifact accompanying the kinase preprint. Phase 1 of the multi-class
> expansion benchmarks aminergic GPCRs in
> [`gpcr-aminergic-benchmarks`](https://github.com/jmabbott40/gpcr-aminergic-benchmarks).

# Kinase Inhibitor Affinity Prediction: Rigorous Baselines

**When do complex, structure-aware ML models outperform simple cheminformatics baselines for kinase inhibitor binding affinity prediction?**

## Experimental Context and Overview

Published ML models for protein–ligand affinity prediction often report impressive performance, but evaluations frequently rely on random train/test splits that allow data leakage through structural analogs. This project systematically evaluates whether protein-aware and structure-based models provide genuine improvements over simple cheminformatics baselines, using:

- **Rigorous data curation** from ChEMBL with documented inclusion criteria
- **Multiple splitting strategies** (random, scaffold, target-based) to reveal true generalization
- **Multi-seed replication** (5 seeds, 105 training runs) with paired significance tests
- **Ablation studies** isolating the effect of ESM-2 protein embeddings (92-target clean subset, fallback strategy comparison)
- **Uncertainty quantification** to assess when predictions should be trusted
- **Precision-focused evaluation** relevant to real drug discovery prioritization

## Key Findings

> **Simple baselines are remarkably hard to beat.** Random Forest with Morgan fingerprints (87s training) matches or exceeds GPU-trained graph neural networks across all evaluation settings. Multi-seed evaluation reveals that a previously reported scaffold-split advantage for protein-aware models was a false positive.

| Finding | Detail |
|---------|--------|
| Random split (easy) | ESM-FP MLP wins (RMSE = 0.777 ± 0.002), significantly better than RF (0.819 ± 0.000, p < 0.001) |
| Scaffold split (realistic) | MLP (0.901 ± 0.003) and ESM-FP MLP (0.902 ± 0.004) are **statistically tied** (p = 0.575) |
| Target split (hardest) | **Random Forest wins** (RMSE = 1.066 ± 0.002), significantly better than all neural models (p < 0.003) |
| ESM-92 clean subset | ESM-FP MLP achieves 0.647 RMSE when all targets have real embeddings — 16.7% better than full dataset |
| Uncertainty | XGBoost quantile regression best calibrated (miscal = 0.003); MC-Dropout uninformative |
| JAK selectivity | Protein-aware models: 79% top-1 accuracy, but stronger baselines (per-target RF) reach 83% |
| Multi-seed impact | Single-seed comparisons produce false positives; deep model SD across seeds up to 0.032 RMSE |

See the [full project report](docs/project_report.md) for detailed analysis.

## Goals of Study

1. How well can ligand-only fingerprint features predict kinase inhibitor binding affinity?
2. How much does performance degrade under scaffold and target-based splits vs. random splits?
3. Can model uncertainty estimates reliably identify unreliable predictions?
4. What chemical and biological features make some predictions systematically harder?
5. Do structure-aware models (GNNs, protein embeddings) provide genuine improvement when evaluated rigorously?
6. How much does incomplete ESM-2 embedding coverage affect protein-aware model evaluation?

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
| Unique targets | 507 (full) / 92 (ESM-2 clean subset) |
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

### Regression Performance (RMSE, mean ± SD across 5 seeds)

| Model | Random | Scaffold | Target |
|-------|--------|----------|--------|
| **ESM-FP MLP** | **0.777 ± 0.002** | 0.902 ± 0.004 | 1.180 ± 0.015 |
| Fusion | 0.791 ± 0.003 | 0.939 ± 0.009 | 1.196 ± 0.031 |
| Random Forest | 0.819 ± 0.000 | 0.919 ± 0.000 | **1.066 ± 0.002** |
| MLP (baseline) | 0.827 ± 0.002 | **0.901 ± 0.003** | 1.105 ± 0.012 |
| GIN | 0.828 ± 0.002 | 0.947 ± 0.005 | 1.177 ± 0.032 |
| XGBoost | 0.894 ± 0.001 | 0.958 ± 0.002 | 1.131 ± 0.004 |
| ElasticNet | 1.274 ± 0.000 | 1.274 ± 0.000 | 1.267 ± 0.000 |

### ESM-92 Clean Subset (all targets have real ESM-2 embeddings)

| Model | Random (full → ESM-92) | Scaffold (full → ESM-92) |
|-------|------------------------|--------------------------|
| ESM-FP MLP | 0.777 → **0.647** (−16.7%) | 0.902 → **0.826** (−8.4%) |
| Fusion | 0.791 → **0.695** (−12.1%) | 0.939 → **0.858** (−8.6%) |
| Random Forest | 0.819 → 0.786 (−4.0%) | 0.919 → 0.872 (−5.1%) |

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
| Per-target RF (stronger baseline) | **83.0%** | **0.833** |
| One-hot RF (stronger baseline) | 83.5% | 0.813 |
| **Fusion** | 79.5% | 0.744 |
| **ESM-FP MLP** | 78.5% | 0.781 |
| MLP (pooled, baseline) | 51.8% | N/A |
| Random Forest (pooled, baseline) | 51.6% | N/A |
| GIN | 49.8% | 0.064 |

## Evaluation

- **Regression**: RMSE, MAE, R², Pearson R, Spearman ρ
- **Classification**: AUROC, AUPRC, precision@k (at pActivity ≥ 6.0 threshold)
- **Splits**: random, scaffold (Murcko), target-based (kinase subfamily holdout)
- **Statistical rigor**: 5-seed replication, paired t-tests, bootstrap CIs (10,000 resamples)
- **Uncertainty**: calibration curves, selective prediction, error–uncertainty correlation
- **Case study**: JAK family (JAK1/2/3, TYK2) selectivity and per-target analysis
- **Ablation**: ESM-2 fallback strategy (row0 vs zero vs mean), 92-target clean subset

## Project Overview

| Phase | Description | Experiments |
|-------|-------------|-------------|
| 1–2 | Data pipeline (ChEMBL ingestion → curated dataset) | — |
| 3 | Feature engineering (Morgan FP, RDKit descriptors) | — |
| 4 | Baseline models (RF, XGBoost, ElasticNet, MLP) | 12 |
| 5 | Evaluation and uncertainty analysis | 21 |
| 6 | Case study — JAK kinase subfamily deep dive | 7 |
| 7 | Advanced models — GIN, ESM-2, GNN+ESM Fusion | 9 |
| Rev | Multi-seed replication (5 seeds × 7 models × 3 splits) | 105 |
| Rev | ESM-92 clean subset evaluation | 21 |
| Rev | Fallback ablation (zero, mean strategies) | 12 |
| Rev | Selectivity baselines (per-target, one-hot, pairwise) | 5 |
| | **Total training runs** | **~192** |

## Supplemental Data

The `results/` directory contains all experimental outputs:

| Directory / File | Contents |
|-----------------|----------|
| `results/tables/` | Primary metrics (21 experiments), bootstrap CIs, paired tests, per-target breakdowns |
| `results/tables/multi_seed_aggregated.csv` | Mean ± SD across 5 seeds for all model/split/metric combinations |
| `results/tables/multi_seed_pairwise.csv` | Paired t-test results for key model comparisons |
| `results/tables/selectivity_baselines_*.csv` | Stronger selectivity baselines for JAK case study |
| `results/tables_esm92/` | All metrics for the 92-target clean subset experiments |
| `results/tables_fb_zero/` | Fallback ablation: zero-vector strategy results |
| `results/tables_fb_mean/` | Fallback ablation: mean-vector strategy results |
| `results/supplement_tables/` | Tables S1–S5 (target assignments, hyperparameters, compute, endpoints, ESM coverage) |
| `results/revision_log.txt` | Full log of the 19.5-hour revision experiment run |

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
# Data pipeline:
python -m kinase_affinity.data.fetch              # Fetch raw data from ChEMBL
python -m kinase_affinity.data.curate             # Standardize and curate dataset

# Feature engineering
python -m kinase_affinity.features.fingerprints   # Morgan fingerprints (2048-bit)
python -m kinase_affinity.features.descriptors    # RDKit 2D descriptors

# Baseline models (CPU)
python -m kinase_affinity.training.trainer --all

# Advanced models (GPU required)
python -m kinase_affinity.data.protein_sequences      # Fetch kinase sequences
python -m kinase_affinity.features.protein_embeddings  # Compute ESM-2 embeddings
python -m kinase_affinity.training.deep_trainer --all  # Train GIN, ESM-FP MLP, Fusion

# Multi-seed replication
for seed in 42 123 456 789 1024; do
  python -m kinase_affinity.training.trainer --all \
    --training-seed $seed --output-suffix "_seed${seed}"
  python -m kinase_affinity.training.deep_trainer --all \
    --training-seed $seed --output-suffix "_seed${seed}"
done

# Multi-seed analysis
python -m kinase_affinity.evaluation.multi_seed_analysis --seeds 42 123 456 789 1024

# ESM-92 clean subset
python -m kinase_affinity.data.subset --dataset-version v1 all

# Selectivity baselines
python scripts/run_selectivity_baselines.py

# Evaluation and uncertainty analysis
python -m kinase_affinity.evaluation.run_phase5

# JAK case study
python scripts/run_phase6_case_study.py

# Supplement tables
python scripts/generate_supplement_tables.py
```

## Repository Structure

```
├── configs/           # YAML configs for dataset curation and model hyperparameters
├── data/
│   ├── raw/           # Raw ChEMBL exports (gitignored)
│   └── processed/     # Versioned curated datasets and subsets (gitignored)
├── docs/              # Data card and project report
├── notebooks/         # Analysis and visualization notebooks (01-05)
├── scripts/           # AWS runner scripts, case study, selectivity baselines, supplement tables
├── src/kinase_affinity/
│   ├── data/          # Ingestion, standardization, curation, splits, subsets, protein sequences
│   ├── features/      # Morgan FP, RDKit descriptors, molecular graphs, ESM-2 embeddings
│   ├── models/        # RF, XGBoost, ElasticNet, MLP, GIN, ESM-FP MLP, Fusion
│   ├── training/      # Baseline trainer, deep trainer, hyperparameter tuning
│   ├── evaluation/    # Metrics, uncertainty, bootstrap, multi-seed analysis
│   └── visualization/ # Plotting utilities (heatmaps, calibration, degradation curves)
├── tests/             # Unit tests (32+ tests across all phases)
└── results/
    ├── predictions*/  # .npz files (per-seed and per-ablation)
    ├── models*/       # Saved model weights
    ├── tables*/       # Metrics (primary, ESM-92 subset, fallback ablation)
    ├── supplement_tables/ # Tables S1-S5
    └── figures/       # 110+ plots
```

## Acknowledgments

- Bioactivity data from [ChEMBL](https://www.ebi.ac.uk/chembl/) (Mendez et al., 2019)
- Molecular processing with [RDKit](https://www.rdkit.org/)
- Protein embeddings from [ESM-2](https://github.com/facebookresearch/esm) (Lin et al., 2023)
- Graph neural networks with [PyTorch Geometric](https://pyg.org/)

## License

MIT — see [LICENSE](LICENSE).
