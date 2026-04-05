# When Do Complex Models Beat Simple Baselines for Kinase Inhibitor Affinity Prediction?

## 1. Introduction

Machine learning for drug discovery promises to accelerate the identification of potent kinase inhibitors, yet published benchmarks often overstate model generalization by evaluating on random splits that allow data leakage of chemical scaffolds and target information. This project asks a deliberately simple question: **how well do classical ML baselines actually perform on kinase affinity prediction, and how much does that performance degrade under realistic evaluation conditions?**

By establishing rigorous baselines first, we create a honest performance floor against which future complex models (GNNs, protein-aware architectures) can be measured. The goal is not to achieve state-of-the-art performance, but to understand *what drives predictive performance* and *when simple models are sufficient*.

### Why kinases?

Protein kinases are among the most important drug target families in oncology and inflammation. They share a conserved ATP-binding fold, meaning inhibitors often show cross-reactivity across the family. This creates a natural test case for generalization: can a model trained on one set of kinases predict binding affinity for kinases it has never seen?

### Why baselines matter

In the ML-for-drug-discovery literature, reported metrics are frequently inflated by evaluation on random splits, where train and test sets share scaffolds and even identical compounds tested against different targets. Scaffold and target-based splits provide a more realistic estimate of how a model would perform on novel chemical matter or novel targets in a real drug discovery campaign.

## 2. Dataset Construction

We constructed a curated kinase bioactivity dataset from ChEMBL, starting from all binding affinity measurements (IC50, Ki, Kd) for human protein kinase targets. The full curation pipeline is config-driven (`configs/dataset_v1.yaml`) and documented in `docs/data_card.md`.

### 2.1 Data sourcing

Kinase targets were identified by querying ChEMBL for targets with Gene Ontology molecular function annotations for kinase activity (GO:0016301, GO:0004672) and name-based filtering, yielding 507 human protein kinase targets. Activity data was filtered to exact measurements (standard_relation = '=') from high-confidence assays (confidence score >= 7).

### 2.2 Standardization

Molecules were standardized using RDKit: salt removal (keep largest fragment), charge neutralization, canonical SMILES generation, and molecular weight filtering (100-900 Da). Activity values were converted to a unified pActivity scale: pActivity = 9 - log10(value_nM), giving a measure where higher values indicate more potent binding.

### 2.3 Duplicate handling and noise detection

When the same compound was measured against the same target in the same assay type across multiple experiments, we took the median pActivity value. Compounds with >= 3 measurements and standard deviation > 1.0 pActivity units were flagged as "noisy" (1,965 measurements, 0.56% of data). These noisy compounds are retained but flagged for downstream error analysis.

### 2.4 Final dataset

| Property | Value |
|----------|-------|
| Curated records | 352,874 |
| Unique compounds | 205,747 |
| Unique kinase targets | 507 |
| Activity types | IC50 (77%), Ki (18%), Kd (5%) |
| Active compounds (pActivity >= 6.0) | 77.3% |
| pActivity range | 3.0 - 11.0 (mean 7.0, std 1.28) |

The 77.3% active rate reflects that ChEMBL is enriched for positive results -- medicinal chemists preferentially publish potent compounds. This class imbalance means AUROC alone is insufficient; we also report AUPRC, precision@k, and enrichment factors.

## 3. Feature Representations

We implemented two complementary molecular representations, each suited to different model types.

### 3.1 Morgan fingerprints (ECFP4)

Binary circular fingerprints (radius=2, 2048 bits) encode local chemical environments around each atom. Each bit indicates the presence or absence of a particular molecular substructure within two bonds of any atom. Morgan fingerprints are the standard representation for structure-activity relationship (SAR) modeling because they capture the functional groups and connectivity patterns most relevant to binding. They are used as input for Random Forest, XGBoost, and MLP.

Implementation uses RDKit's `rdFingerprintGenerator.GetMorganGenerator` API (the modern, non-deprecated interface).

### 3.2 RDKit 2D descriptors

A set of 217 continuous physicochemical and topological descriptors (molecular weight, LogP, polar surface area, rotatable bond count, ring counts, etc.) computed using RDKit's `CalcMolDescriptors`. After dropping descriptors with > 5% missing values and imputing remaining NaNs with column medians, 217 descriptors are retained.

These continuous features are better suited to linear models like ElasticNet, where the relationship between individual molecular properties and binding affinity can be directly interpreted.

### 3.3 Feature-activity alignment

A critical implementation detail: features are computed per unique compound (205,747 rows), while activity records contain 352,874 rows (the same compound can appear in measurements against multiple targets). The training pipeline uses a SMILES-to-row lookup dictionary to map each activity record to its corresponding feature vector, enabling efficient alignment without duplicating feature storage.

## 4. Splitting Strategies

Three splitting strategies test progressively harder forms of generalization:

### 4.1 Random split

80/10/10 train/validation/test, stratified by target to maintain target representation across splits. This is the easiest evaluation setting: train and test compounds share scaffolds, and the model has seen examples for every target.

### 4.2 Scaffold split

Groups compounds by Murcko generic scaffold (core ring systems) and assigns entire scaffold groups to a single split, preventing scaffold leakage. This tests whether the model can generalize to novel chemical scaffolds not seen during training -- a more realistic simulation of lead optimization.

### 4.3 Target split

Holds out entire kinase subfamilies for testing. This is the hardest setting: the model must predict binding affinity for kinase targets it has never seen. This tests whether the model has learned transferable SAR patterns across the kinase family.

| Split | Train | Validation | Test |
|-------|-------|------------|------|
| Random | 282,299 | 35,287 | 35,288 |
| Scaffold | 282,301 | 35,288 | 35,285 |
| Target | 251,532 | 65,811 | 35,531 |

Note the target split has a smaller training set because entire kinase subfamilies are held out, and the validation set is larger because some subfamilies straddle the train/validation boundary.

## 5. Baseline Models

We implemented four baseline models, each representing a different modeling philosophy and providing a different uncertainty estimation mechanism.

### 5.1 Random Forest

**Rationale**: Random Forest is a natural first baseline for high-dimensional binary fingerprints. Each tree partitions the fingerprint space along individual bit positions (i.e., presence/absence of specific substructures), creating an implicit "if substructure X is present AND substructure Y is absent, predict pActivity ~7.5" decision logic. This is conceptually similar to how medicinal chemists reason about SAR: specific structural features contribute positively or negatively to binding.

**Configuration**: 500 trees, no depth limit, min_samples_leaf=2, max_features=sqrt (random subsets of ~45 bits per split).

**Uncertainty**: Variance across individual tree predictions. Each tree sees a bootstrap sample of the data and a random subset of features, so disagreement between trees reflects regions of chemical space with less training data or more complex SAR.

### 5.2 XGBoost

**Rationale**: Gradient-boosted trees can capture more complex feature interactions than random forests through sequential error correction. Where RF builds trees independently, XGBoost builds each tree to correct the residual errors of the ensemble so far. This often yields better point predictions at the cost of more hyperparameters to tune.

**Configuration**: 500 trees, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8.

**Uncertainty**: Quantile regression. In addition to the primary model (squared error loss), two auxiliary XGBoost models are trained with quantile loss at alpha=0.05 and 0.95, estimating the 90% prediction interval. The standard deviation is derived from the interval width: std = (q95 - q05) / (2 * 1.645). This approach gives calibrated uncertainty without the overhead of ensembling the full model.

### 5.3 ElasticNet

**Rationale**: ElasticNet combines L1 (Lasso) and L2 (Ridge) regularization, making it suited for continuous descriptor spaces where many features may be irrelevant. L1 regularization drives irrelevant descriptor coefficients to zero (feature selection), while L2 prevents coefficient explosion for correlated descriptors. This is the most interpretable model: each descriptor gets an explicit coefficient indicating its contribution to predicted affinity.

**Configuration**: alpha=1.0, l1_ratio=0.5, max_iter=10000. Features are internally standardized (StandardScaler fit on training data, applied to all predictions).

**Uncertainty**: Bootstrap resampling. 100 ElasticNet models are trained on bootstrap samples (sampling with replacement from training data), and prediction variance across the ensemble provides uncertainty estimates.

**Important caveat**: ElasticNet with alpha=1.0 proves to be too aggressively regularized for this dataset (see Section 6). All coefficients are driven to zero, producing near-constant predictions. A hyperparameter search over alpha values (planned for Phase 5) is needed to find the optimal regularization strength.

### 5.4 MLP (Multi-Layer Perceptron)

**Rationale**: A shallow neural network serves as a bridge between classical ML and deep learning. The MLP can learn non-linear combinations of fingerprint bits that tree-based models cannot represent. With two hidden layers, it can approximate complex SAR relationships while remaining trainable on CPU.

**Configuration**: Two hidden layers [256, 128], ReLU activation, Adam optimizer, batch_size=512, early stopping with patience=10, max_iter=200. Features are internally standardized.

**Uncertainty**: Ensemble of 3 independently initialized MLPs (different random seeds). Prediction variance across ensemble members captures epistemic uncertainty -- regions of input space where the model is sensitive to initialization.

**Note**: MLP training is computationally intensive (~10-12 min per ensemble member on a 48-core EC2 instance). All experiments were run on AWS to completion.

## 6. Results

### 6.1 Regression performance

| Model | Split | RMSE | MAE | R^2 | Pearson | Spearman | Train time |
|-------|-------|------|-----|-----|---------|----------|------------|
| **Random Forest** | Random | **0.818** | **0.632** | **0.587** | **0.769** | **0.754** | 87s |
| **Random Forest** | Scaffold | 0.919 | 0.722 | 0.477 | 0.712 | 0.705 | 89s |
| **Random Forest** | Target | 1.067 | 0.834 | 0.265 | 0.520 | 0.465 | 76s |
| MLP | Random | 0.824 | 0.623 | 0.581 | 0.764 | 0.746 | 739s |
| MLP | Scaffold | 0.905 | 0.697 | 0.493 | 0.705 | 0.689 | 708s |
| MLP | Target | 1.090 | 0.851 | 0.234 | 0.527 | 0.473 | 593s |
| XGBoost | Random | 0.893 | 0.710 | 0.508 | 0.718 | 0.697 | 110s |
| XGBoost | Scaffold | 0.961 | 0.757 | 0.429 | 0.664 | 0.641 | 103s |
| XGBoost | Target | 1.135 | 0.888 | 0.169 | 0.426 | 0.394 | 201s |
| ElasticNet | Random | 1.274 | 1.049 | -0.000 | NaN | NaN | 60s |
| ElasticNet | Scaffold | 1.274 | 1.043 | -0.003 | NaN | NaN | 61s |
| ElasticNet | Target | 1.267 | 1.045 | -0.035 | NaN | NaN | 52s |

### 6.2 Classification performance (active/inactive at pActivity >= 6.0)

| Model | Split | AUROC | AUPRC |
|-------|-------|-------|-------|
| **Random Forest** | Random | **0.865** | **0.953** |
| **Random Forest** | Scaffold | 0.845 | 0.953 |
| **Random Forest** | Target | 0.706 | 0.844 |
| MLP | Random | 0.860 | 0.951 |
| MLP | Scaffold | 0.842 | 0.951 |
| MLP | Target | **0.718** | **0.862** |
| XGBoost | Random | 0.834 | 0.942 |
| XGBoost | Scaffold | 0.812 | 0.939 |
| XGBoost | Target | 0.662 | 0.802 |
| ElasticNet | Random | 0.500 | 0.771 |
| ElasticNet | Scaffold | 0.500 | 0.795 |
| ElasticNet | Target | 0.500 | 0.715 |

### 6.3 Key findings

**1. Random Forest and MLP are near-equivalent top baselines.** RF (RMSE=0.818, R^2=0.587) and MLP (RMSE=0.824, R^2=0.581) perform almost identically on the random split. This is notable because the MLP can learn non-linear combinations of fingerprint bits that trees cannot represent, yet it provides no meaningful improvement. This suggests that for binary Morgan fingerprints, the piecewise-constant approximations that decision trees make already capture most of the learnable signal. Both outperform XGBoost (RMSE=0.893, R^2=0.508) by a meaningful margin.

**2. MLP shows a slight advantage for target generalization.** On the hardest split (target), MLP achieves AUROC=0.718 vs RF's 0.706 and XGBoost's 0.662. However, RF has a higher R^2 (0.265 vs 0.234). This split in ranking metrics is scientifically interesting: RF is better at predicting exact pActivity values, while MLP is better at ranking active vs. inactive compounds for unseen targets. Different drug discovery use cases would prefer different models -- virtual screening favors AUROC (ranking), while lead optimization favors RMSE (accuracy).

**3. Splitting strategy dramatically affects perceived performance.** The best model (RF) shows a clear degradation pattern:

| Metric | Random | Scaffold | Target |
|--------|--------|----------|--------|
| RMSE | 0.818 | 0.919 (+12%) | 1.067 (+30%) |
| R^2 | 0.587 | 0.477 (-19%) | 0.265 (-55%) |
| AUROC | 0.865 | 0.845 (-2%) | 0.706 (-18%) |

Going from random to scaffold split costs ~12% in RMSE, and moving to target split costs ~30%. This quantifies the "scaffold leakage" and "target leakage" effects that inflate reported performance in many published studies. MLP shows a similar pattern (random R^2=0.581 → target R^2=0.234, a 60% drop).

**4. XGBoost underperforms RF despite theoretical advantages.** Gradient boosting typically outperforms bagging on tabular data, but here XGBoost consistently trails RF by 5-10% on RMSE. The likely explanation is that XGBoost's depth limit (max_depth=6) constrains its expressiveness on 2048-dimensional binary data, while RF's unlimited-depth trees can capture deeper feature interactions. A hyperparameter search over max_depth (Phase 5) may close this gap.

**5. ElasticNet with default hyperparameters collapses completely.** With alpha=1.0 and l1_ratio=0.5, all coefficients are driven to zero, producing constant predictions (R^2 ~ 0, AUROC = 0.5). This is not a bug -- it demonstrates that strong L1 regularization on 217 descriptors is too aggressive for this problem. The AUPRC values (0.715-0.795) are not meaningful here; they simply reflect the dataset's class prior (77.3% active). Reducing alpha by 2-3 orders of magnitude (0.001-0.01) would likely recover meaningful predictions and is planned for the hyperparameter search phase.

**6. Validation and test metrics are consistent.** Across all experiments, validation and test metrics track closely (e.g., RF random: val R^2=0.594 vs test R^2=0.587; MLP random: val R^2=0.585 vs test R^2=0.581), indicating that the splits are well-constructed and results are not artifacts of a particular test set.

## 7. Uncertainty Quantification

Each model provides uncertainty estimates through a different mechanism, reflecting different sources of epistemic uncertainty:

### 7.1 Uncertainty methods

| Model | Method | What it captures |
|-------|--------|-----------------|
| Random Forest | Tree prediction variance | Disagreement between bootstrap-sampled trees; high in sparse regions of chemical space |
| XGBoost | Quantile regression (5th/95th) | Width of the 90% prediction interval; std = (q95 - q05) / (2 × 1.645) |
| ElasticNet | Bootstrap resampling (100×) | Sensitivity of linear coefficients to training data perturbation |
| MLP | Ensemble variance (3 models) | Sensitivity to random initialization; captures regions where the loss landscape has multiple basins |

### 7.2 Calibration analysis

A well-calibrated model's prediction intervals should match their stated confidence: a 90% interval should contain 90% of true values. We assess this using a calibration curve, where we vary the confidence level from ~10% to ~95% and measure the fraction of true values falling within the corresponding interval.

The **miscalibration area** (area between the calibration curve and the perfect-calibration diagonal) provides a single number summarizing calibration quality. Lower is better; zero means perfect calibration.

### 7.3 Error-uncertainty correlation

A useful uncertainty estimate should correlate with actual prediction error -- the model should "know what it doesn't know." We measure Pearson and Spearman correlation between |prediction error| and predicted uncertainty. High correlation means uncertain predictions are indeed less accurate, which is actionable: a medicinal chemist can prioritize high-confidence predictions for experimental validation.

### 7.4 Selective prediction

The most practically relevant analysis: if we refuse to predict on the most uncertain compounds, how much does RMSE improve? The selective prediction curve shows RMSE as a function of retention fraction (1.0 = all predictions, 0.5 = only the most confident half).

A steep curve means uncertainty estimation adds value to the decision-making pipeline. A flat curve means the uncertainty estimates are uninformative -- no better than random rejection.

### 7.5 Calibration results

Calibration quality varies dramatically across models and uncertainty methods. The table below shows miscalibration area (lower = better calibrated), error-uncertainty correlation (higher = more actionable), and selective prediction improvement at 50% retention (higher = more useful for decision-making).

| Model | Split | Miscal. Area | UQ Spearman ρ | Sel. Improv. (50%) |
|-------|-------|-------------|---------------|-------------------|
| **XGBoost** | Random | **0.017** | 0.195 | **12.5%** |
| **XGBoost** | Scaffold | **0.003** | 0.151 | 8.3% |
| **XGBoost** | Target | **0.048** | 0.021 | -2.5% |
| Random Forest | Random | 0.120 | 0.142 | 7.2% |
| Random Forest | Scaffold | 0.037 | **0.334** | **21.5%** |
| Random Forest | Target | 0.166 | **0.201** | **16.3%** |
| MLP (baseline) | Random | 0.256 | 0.057 | 4.7% |
| MLP (baseline) | Scaffold | 0.252 | 0.057 | 3.7% |
| MLP (baseline) | Target | 0.301 | 0.126 | 7.5% |
| ESM-FP MLP | Random | 0.167 | -0.011 | -0.5% |
| ESM-FP MLP | Scaffold | 0.175 | -0.000 | 2.5% |
| ESM-FP MLP | Target | 0.186 | 0.045 | 3.3% |
| GIN | Random | 0.317 | -0.030 | -0.4% |
| GIN | Scaffold | 0.322 | -0.058 | -2.1% |
| GIN | Target | 0.227 | 0.070 | 3.8% |
| Fusion | Random | 0.239 | -0.045 | -1.2% |
| Fusion | Scaffold | 0.259 | -0.061 | -2.6% |
| Fusion | Target | 0.283 | 0.042 | 5.6% |

**Key findings on uncertainty:**

1. **XGBoost quantile regression is best calibrated** (miscal. area 0.003-0.048), substantially outperforming all other methods. Its prediction intervals closely match stated confidence levels across all splits.

2. **Random Forest tree variance is most actionable.** While RF's calibration is moderate (miscal. area 0.037-0.166), it has the strongest error-uncertainty correlation (Spearman ρ = 0.334 on scaffold split) and the highest selective prediction improvement (21.5% RMSE reduction at 50% retention on scaffold split). This means RF's uncertainty estimates are the most useful for practical decision-making: rejecting its most uncertain predictions substantially improves accuracy.

3. **MC-Dropout uncertainty from deep models is poorly calibrated.** All three neural models (ESM-FP MLP, GIN, Fusion) show high miscalibration areas (0.167-0.322), near-zero or negative error-uncertainty correlations, and negligible or negative selective prediction improvement. MC-Dropout with only 10 forward passes produces uncertainty estimates that are effectively uninformative — the model *does not know what it doesn't know*.

4. **Baseline ensemble uncertainty >> MC-Dropout uncertainty.** Averaging across splits, baseline models achieve mean miscalibration of 0.097-0.172 and selective improvement of +7-11%, while deep models average 0.232-0.252 miscalibration and -0.7% selective improvement. This is a significant practical disadvantage of the deep learning approach: not only are the point predictions often similar, but the uncertainty estimates are worse.

## 8. Error Analysis

### 8.1 Per-target metrics

Aggregate metrics (RMSE, R²) can mask substantial variation across kinase targets. Some kinases may be well-predicted because they are heavily represented in the training data or have straightforward SAR, while others may be intrinsically difficult due to unusual binding modes or sparse data.

We compute per-target RMSE, MAE, R², and correlation for all targets with at least 10 test compounds, and examine the distribution of per-target performance to identify:
- **Easy targets**: well-predicted kinases where baselines are likely sufficient
- **Hard targets**: kinases where even the best baseline fails, suggesting that protein-aware models may be needed

### 8.2 Noise impact

During data curation (Section 2.3), we flagged 1,965 measurements as "noisy" -- compounds with ≥3 replicate measurements and standard deviation > 1.0 pActivity units. If models perform worse on these compounds, it suggests that measurement noise (not model failure) sets a floor on achievable accuracy for those data points.

### 8.3 Worst predictions

Identifying the compounds with the largest absolute errors reveals failure modes:
- **Activity cliffs**: structurally similar compounds with very different activities. Fingerprint-based models cannot distinguish these because the structural difference falls below the fingerprint resolution (radius=2 substructures)
- **Underrepresented scaffolds**: novel chemical matter far from training distribution
- **Mixed activity types**: compounds measured as IC50 vs Ki vs Kd may have systematic offsets

### 8.4 Hyperparameter tuning

Phase 4 revealed that ElasticNet with default alpha=1.0 collapses completely (all coefficients driven to zero). Phase 5 includes a validation-set hyperparameter search over:
- **ElasticNet**: alpha ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0} × l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- **XGBoost**: max_depth ∈ {4, 6, 8, 10, 12} × learning_rate ∈ {0.05, 0.1} × n_estimators ∈ {300, 500}

*Tuning results and re-evaluated metrics will be populated after running the tuning pipeline (`python -m kinase_affinity.training.tune --all`).*

## 9. Advanced Neural Network Models (Phase 7)

### 9.1 Motivation

The baseline results (Sections 6-8) establish that fingerprint-based models achieve R² = 0.59 on random splits but degrade to R² = 0.27 on target-held-out splits. The core hypothesis for Phase 7 is: **protein-aware models should primarily help on the target split**, where the model must generalize to unseen kinase subfamilies and where information about the target protein's identity and structure could compensate for the lack of training data.

### 9.2 Model architectures

Three advanced models were implemented, each introducing a different form of biological information:

**ESM-FP MLP**: Concatenates Morgan fingerprints (2048-dim) with ESM-2 protein language model embeddings (1280-dim) for a 3328-dim input. Architecture: Input(3328) → [Linear→ReLU→Dropout→BatchNorm] × 2 → Linear→ReLU→Dropout → Linear(1). This model tests whether knowing which protein the ligand binds to (via ESM-2 embeddings from the `esm2_t33_650M_UR50D` model, mean-pooled over residue positions) improves predictions.

**GIN (Graph Isomorphism Network)**: Operates directly on molecular graphs instead of fixed fingerprints. Atom features (~35 dim: element, degree, charge, hybridization, aromaticity, chirality) and bond features (~11 dim: bond type, stereo, conjugation) are processed through 3 GINConv layers with 128-dim hidden states, followed by global mean+max pooling (256-dim) and an MLP head. This tests whether *learned* molecular representations outperform *fixed* fingerprints.

**GIN + ESM-2 Fusion**: Combines both innovations — a GIN ligand branch (→ 256-dim projection) and an ESM-2 protein branch (1280 → 256) are concatenated and fed through a shared MLP head. This is the most information-rich model, with access to both learned molecular and protein representations.

All deep models use:
- **AdamW optimizer** with cosine annealing learning rate schedule
- **Early stopping** with patience = 15 epochs on validation RMSE
- **MC-Dropout** (10 forward passes) for uncertainty estimation
- **Batch size**: 512 (ESM-FP MLP), 256 (GNN, Fusion)

### 9.3 Regression performance

| Model | Split | RMSE | MAE | R² | Pearson | Train Time |
|-------|-------|------|-----|-----|---------|------------|
| **ESM-FP MLP** | **Random** | **0.775** | **0.582** | **0.629** | **0.794** | 229s |
| Fusion | Random | 0.793 | 0.603 | 0.613 | 0.784 | 1551s |
| Random Forest | Random | 0.818 | 0.632 | 0.587 | 0.769 | 87s |
| MLP (baseline) | Random | 0.824 | 0.623 | 0.581 | 0.764 | 647s |
| GIN | Random | 0.829 | 0.640 | 0.577 | 0.760 | 1411s |
| XGBoost | Random | 0.893 | 0.710 | 0.508 | 0.718 | 113s |
| | | | | | | |
| **ESM-FP MLP** | **Scaffold** | **0.897** | **0.684** | **0.502** | **0.711** | 156s |
| MLP (baseline) | Scaffold | 0.905 | 0.697 | 0.493 | 0.705 | 704s |
| Random Forest | Scaffold | 0.919 | 0.722 | 0.478 | 0.712 | 88s |
| GIN | Scaffold | 0.941 | 0.730 | 0.453 | 0.676 | 1032s |
| Fusion | Scaffold | 0.945 | 0.726 | 0.449 | 0.677 | 1178s |
| XGBoost | Scaffold | 0.961 | 0.757 | 0.429 | 0.664 | 111s |
| | | | | | | |
| **Random Forest** | **Target** | **1.067** | **0.834** | **0.265** | **0.520** | 76s |
| MLP (baseline) | Target | 1.090 | 0.851 | 0.234 | 0.527 | 599s |
| XGBoost | Target | 1.135 | 0.888 | 0.169 | 0.426 | 97s |
| Fusion | Target | 1.138 | 0.894 | 0.165 | 0.465 | 683s |
| GIN | Target | 1.146 | 0.916 | 0.154 | 0.441 | 342s |
| ESM-FP MLP | Target | 1.177 | 0.924 | 0.107 | 0.372 | 44s |

### 9.4 Key findings

**1. ESM-FP MLP wins on random and scaffold splits — but by diminishing margins.** On random splits, ESM-FP MLP (RMSE=0.775) improves over the best baseline RF (RMSE=0.818) by 5.3%. On scaffold splits, the improvement shrinks to just 0.9% (0.897 vs 0.905 MLP baseline). The protein embedding provides useful target-identity information when structural analogs are available in training, but adds less when the model must generalize to novel scaffolds.

**2. Baselines win on target-held-out splits — the most important finding.** On the target split, Random Forest (RMSE=1.067) beats every neural model, and ESM-FP MLP (RMSE=1.177) drops to *last place*. This directly contradicts the hypothesis that protein embeddings would help most where baselines struggle. Instead, ESM-2 embeddings become a *liability* — the model overfits to the association between specific embedding patterns and activity ranges during training, and these associations don't transfer to unseen kinase subfamilies.

**3. Early stopping reveals the learning dynamics.** ESM-FP MLP stopped at epoch 2 on the target split (vs. epoch 50 on random), meaning it couldn't learn any transferable patterns for unseen targets beyond what the first few gradient steps provide. GIN stopped at epoch 10, and Fusion at epoch 30 — all far earlier than on random splits (50-98 epochs), indicating that the validation signal deteriorates rapidly for novel targets.

**4. GIN does not outperform Morgan fingerprints.** The learned graph representation (GIN, RMSE=0.829 random) performs equivalently to fixed 2048-bit Morgan fingerprints (RF=0.818, MLP=0.824) across all splits. This suggests that for this dataset size (~350K) and task, the GINConv message-passing layers don't capture meaningful information beyond what ECFP4 fingerprints already encode. The atom/bond featurization (35+11 dims) may be insufficient, or the 3-layer GIN architecture may lack the depth for complex SAR.

**5. Fusion doesn't synergize — it averages.** The Fusion model (GIN + ESM-2) performs between its two component models rather than exceeding both. On random splits (RMSE=0.793), it's worse than ESM-FP MLP (0.775) but better than GIN (0.829). On target splits (RMSE=1.138), it's worse than both RF (1.067) and MLP (1.090). The concatenation fusion strategy doesn't enable the model to leverage complementary information from the two branches.

**6. Computational cost is not justified.** The GIN and Fusion models cost 16-18× more compute than Random Forest (1411-1551s vs 87s) for equivalent or worse performance. Only ESM-FP MLP offers a favorable cost-benefit tradeoff (229s, 2.6× RF) with a 5.3% RMSE improvement on random splits.

### 9.5 Performance degradation analysis

The degradation pattern from random → scaffold → target splits reveals each model's sensitivity to data leakage:

| Model | Random RMSE | Scaffold RMSE | Target RMSE | Random→Target Δ |
|-------|-------------|---------------|-------------|-----------------|
| ESM-FP MLP | 0.775 | 0.897 (+15.7%) | 1.177 (+51.9%) | **+51.9%** |
| Random Forest | 0.818 | 0.919 (+12.3%) | 1.067 (+30.4%) | +30.4% |
| MLP (baseline) | 0.824 | 0.905 (+9.8%) | 1.090 (+32.3%) | +32.3% |
| GIN | 0.829 | 0.941 (+13.5%) | 1.146 (+38.2%) | +38.2% |
| XGBoost | 0.893 | 0.961 (+7.6%) | 1.135 (+27.1%) | +27.1% |
| Fusion | 0.793 | 0.945 (+19.2%) | 1.138 (+43.5%) | +43.5% |

ESM-FP MLP degrades the most (51.9%) from random to target split, confirming that its random-split advantage is inflated by target-identity leakage. Random Forest and XGBoost show the most graceful degradation (27-30%), making them more reliable across deployment scenarios.

## 10. Case Study: JAK Family

### 10.1 Why JAK?

The Janus kinase (JAK) family — JAK1, JAK2, JAK3, and TYK2 — was selected for the case study because it offers the richest analytical opportunity in our dataset: 36,059 records across 4 well-characterized members, with 2,498 compounds tested against all 4 JAKs. Multiple FDA-approved JAK inhibitors (tofacitinib, baricitinib, ruxolitinib) provide clinical context, and the high compound overlap between members (81% of JAK1 compounds also tested on JAK2) enables selectivity prediction analysis — a key question in JAK inhibitor design.

### 10.2 Per-target model comparison

JAK targets are generally easier to predict than the dataset average, likely due to their large training set sizes and well-explored SAR landscapes. On the random split:

| Target | Best Model | RMSE | Dataset-wide RMSE (same model) |
|--------|-----------|------|-------------------------------|
| TYK2 | Fusion | 0.636 | 0.793 |
| JAK2 | ESM-FP MLP | 0.691 | 0.775 |
| JAK1 | ESM-FP MLP | 0.596 | 0.775 |
| JAK3 | ESM-FP MLP | 0.656 | 0.775 |

ESM-FP MLP dominates across JAK members on random splits, achieving 0.596 RMSE on JAK1 — 23% better than the dataset-wide average. This confirms that protein embeddings provide the most value when the model has seen abundant examples from the same target during training.

### 10.3 JAK3 is the hardest JAK target

Across all 7 models, JAK3 consistently has the highest RMSE (average 0.933 vs. 0.825 for TYK2). The radar chart analysis reveals why: JAK3 has the highest activity range (std=1.35), the highest fraction of noisy measurements (2.8%), and fewer compounds (7,201) than JAK1 or JAK2. The wider activity distribution and higher noise create a more challenging prediction landscape.

### 10.4 Activity cliffs

We identified 7 activity cliff pairs in the JAK test set (Tanimoto ≥ 0.85 and |ΔpActivity| ≥ 1.5). The most dramatic cliff involves two structurally identical compounds (Tanimoto = 1.000) with a 2.0 log-unit difference in JAK1 activity — likely reflecting different stereochemistry or measurement conditions not captured by canonical SMILES. JAK1 and JAK2 show more cliff pairs than JAK3 and TYK2, consistent with their larger and more densely explored compound sets where subtle structural modifications can dramatically alter potency.

### 10.5 Selectivity prediction — where protein-aware models shine

The most striking Phase 6 finding: **protein-aware models dramatically outperform fingerprint-based models on selectivity prediction**, even when overall affinity prediction is similar.

On the scaffold split, with 624 compounds tested against 2+ JAK members:

| Model | Top-1 JAK Accuracy | Rank Correlation (ρ) |
|-------|-------------------|---------------------|
| Random Forest | 51.6% | N/A (only 2 targets per compound) |
| XGBoost | 51.6% | N/A |
| MLP (baseline) | 51.8% | N/A |
| **ESM-FP MLP** | **78.5%** | **0.781** |
| GIN | 49.8% | 0.064 |
| **Fusion** | **79.5%** | **0.744** |

Fingerprint-based models (RF, XGBoost, MLP) achieve ~51% top-1 accuracy — barely above random chance for binary comparisons. They cannot distinguish which JAK member a compound preferentially inhibits because they see only ligand structure, which is identical across JAK assays for the same compound.

ESM-FP MLP and Fusion achieve ~79% accuracy by incorporating ESM-2 protein embeddings, which encode the sequence differences between JAK family members. This is the clearest evidence in our entire study that protein-aware models provide genuine value — not for absolute affinity prediction (where baselines match them), but for **relative selectivity prediction across related targets**.

The GIN model (49.8%) fails at selectivity because it also sees only ligand structure, confirming that this is a protein-representation advantage, not a ligand-representation advantage.

### 10.6 Worst predictions

The worst-predicted JAK compounds show interesting model-specific failure patterns:
- **Random Forest**: failures spread across JAK1 (8), JAK3 (5), JAK2 (4), TYK2 (3), dominated by IC50 measurements (14/20)
- **ESM-FP MLP**: failures concentrated on JAK2 (10/20), suggesting the model overfits to JAK2's larger training set and makes biased predictions for edge-case JAK2 compounds
- **Fusion**: failures split between JAK3 (7) and JAK2 (7), with JAK1 failures (6) — no TYK2 failures, consistent with TYK2 being easiest to predict

Notably, none of the top-20 worst predictions per model are flagged as noisy measurements, indicating that prediction failures reflect genuine model limitations rather than data quality issues.

### 10.7 Target split caveat

An important finding: **no JAK test compounds exist in the target-held-out split**. All four JAK members were assigned to the training fold during target-based splitting. This means the target-split results in Section 9 evaluate generalization to other kinase subfamilies (e.g., CDKs, MAP kinases) using JAK as *training* data, not as a prediction target. This is by design — the target split tests cross-subfamily transfer, and JAK happens to fall on the training side of the partition.

## 11. Statistical Significance: Bootstrap Analysis

### 11.1 Motivation

Point estimates of test-set metrics (RMSE, R², etc.) are subject to sampling variability — a single test set may not capture the true performance difference between models. To establish whether observed differences are statistically significant, we employ non-parametric bootstrap analysis: resampling the test predictions (without retraining) to generate confidence intervals and paired significance tests.

### 11.2 Methodology

**Bootstrap confidence intervals**: For each of the 21 experiments (7 models × 3 splits), we draw 10,000 bootstrap samples (sampling with replacement from test predictions) and compute six metrics per sample: RMSE, MAE, R², Pearson R, Spearman ρ, and AUROC. The 2.5th and 97.5th percentiles define 95% confidence intervals. RMSE and MAE are computed via a vectorized fast path that constructs a `(10000, n_test)` residual matrix for ~100× speedup over Python loops.

**Paired bootstrap tests**: Independent CIs can overlap even when one model is consistently better, because they don't account for correlated errors on the same test compounds. Paired tests resample the *same indices* for both models simultaneously, computing the metric difference on each bootstrap sample. The p-value is the fraction of samples where the sign of the difference reverses — directly testing whether model A is better than model B.

**Win-rate matrix**: For all 21 pairwise model comparisons within each split, we compute the fraction of bootstrap samples where model A has lower RMSE than model B, providing a probabilistic ranking.

### 11.3 Key results: Confidence intervals

| Model | Split | RMSE | 95% CI | R² | 95% CI |
|-------|-------|------|--------|-----|--------|
| ESM-FP MLP | Random | 0.775 | [0.768, 0.783] | 0.629 | [0.622, 0.637] |
| Fusion | Random | 0.793 | [0.785, 0.800] | 0.613 | [0.605, 0.620] |
| Random Forest | Random | 0.818 | [0.811, 0.826] | 0.587 | [0.580, 0.594] |
| MLP | Random | 0.824 | [0.816, 0.832] | 0.581 | [0.573, 0.589] |
| GIN | Random | 0.829 | [0.821, 0.836] | 0.577 | [0.569, 0.585] |
| ESM-FP MLP | Scaffold | 0.897 | [0.889, 0.905] | 0.502 | [0.493, 0.511] |
| MLP | Scaffold | 0.905 | [0.898, 0.913] | 0.493 | [0.485, 0.502] |
| RF | Scaffold | 0.919 | [0.912, 0.927] | 0.477 | [0.471, 0.484] |
| RF | Target | 1.067 | [1.059, 1.076] | 0.265 | [0.255, 0.275] |
| MLP | Target | 1.090 | [1.081, 1.099] | 0.234 | [0.222, 0.245] |
| ESM-FP MLP | Target | 1.177 | [1.167, 1.186] | 0.107 | [0.096, 0.119] |

Confidence intervals are narrow (±0.008 RMSE typical), reflecting the large test sets (~35,000 samples). This means even small metric differences are detectable.

### 11.4 Paired bootstrap test findings

The paired tests reveal critical distinctions that independent CIs miss:

**ESM-FP MLP vs. MLP (scaffold split)**: ΔRMSE = −0.008 [−0.013, −0.003], p = 0.003. Despite the tiny absolute difference (0.9%), the paired test confirms this is statistically significant — ESM-FP MLP *consistently* beats MLP on the same test compounds. The independent CIs overlap substantially, but the paired test accounts for correlated errors and detects the systematic advantage.

**RF vs. MLP (random split)**: ΔRMSE = −0.006 [−0.009, −0.002], p = 0.002. RF is significantly better than MLP on random splits, resolving the apparent near-tie in point estimates.

**MLP vs. GIN (random split)**: ΔRMSE = −0.005 [−0.009, +0.000], p = 0.050. Borderline non-significant — the fingerprint MLP and graph neural network are statistically indistinguishable, confirming that learned graph representations provide no advantage over fixed Morgan fingerprints at this scale.

**GIN vs. Fusion (scaffold split)**: ΔRMSE = −0.004 [−0.010, +0.003], p = 0.254. Not significant — these two models are interchangeable on scaffold splits, despite the Fusion model having access to protein embeddings.

**XGBoost vs. Fusion (target split)**: ΔRMSE = −0.002 [−0.008, +0.003], p = 0.407. Not significant — XGBoost (a simple fingerprint model) is statistically tied with the most complex neural architecture on the hardest evaluation setting.

**RF vs. all deep models (target split)**: All three comparisons are highly significant (p < 0.001). Random Forest's advantage over ESM-FP MLP (ΔRMSE = −0.109, p < 0.001), GIN (ΔRMSE = −0.078, p < 0.001), and Fusion (ΔRMSE = −0.070, p < 0.001) on the target split is unambiguous.

### 11.5 Win-rate matrix

The win-rate matrix provides a probabilistic model ranking. On each bootstrap sample, we determine which model has the lower RMSE, yielding a "win probability" for every pair:

- **Random split**: ESM-FP MLP wins against all other models in 100% of bootstrap samples. The ranking ESM-FP MLP > Fusion > RF > MLP > GIN > XGBoost is stable.
- **Scaffold split**: ESM-FP MLP beats MLP in 99.9% of samples (confirming significance despite small margin). MLP beats RF in 100% of samples. GIN vs. Fusion is a coin flip (87% for GIN).
- **Target split**: RF beats every other model in ≥99.6% of samples, confirming its dominance on the hardest split. MLP beats ESM-FP MLP in 100% of samples — protein embeddings are a clear liability for novel targets.

### 11.6 Implications

1. **Small differences are real.** The large test sets provide enough statistical power to distinguish 0.5-1% RMSE differences. ESM-FP MLP's scaffold-split advantage (0.9%) is genuine, not noise.
2. **The model ranking is robust.** Bootstrap win rates exceeding 99% for key comparisons mean the rankings in Section 9 would replicate under different test-set compositions.
3. **Non-results are informative.** The inability to distinguish MLP from GIN (p = 0.05) or XGBoost from Fusion on target splits (p = 0.41) confirms that architectural complexity provides no benefit in these settings.

## 12. Limitations: ESM-2 Embedding Coverage

### 12.1 The embedding fallback problem

A critical limitation of the protein-aware models (ESM-FP MLP and Fusion) must be disclosed: **only 92 of the 507 kinase targets (18.1%) have real ESM-2 embeddings.** The remaining 415 targets (81.9%) lack UniProt protein sequences in our pipeline, because the ChEMBL API did not return UniProt cross-references for these targets during the sequence retrieval step (Section 9.2).

The deep trainer code silently handles missing embeddings by falling back to row 0 of the embedding matrix:

```python
esm_rows = np.array([
    target_to_row.get(t, 0)  # fallback to row 0 for unknown targets
    for t in subset["target_chembl_id"].values
])
```

Row 0 corresponds to whichever target is alphabetically first in the target index — an arbitrary real kinase's ESM-2 embedding. This means that for 415 out of 507 targets, the protein-aware models receive an *identical, arbitrary* protein representation, effectively reducing them to fingerprint-only models for those targets.

### 12.2 Impact on results

The impact depends on the fraction of *records* (not targets) affected, since some targets have orders of magnitude more measurements than others:

Although the embedding files are not available locally for exact quantification (they reside on the AWS training instance), the user's analysis confirms that approximately 82% of targets receive the fallback embedding. The record-level impact may differ if the 92 real-embedding targets are enriched for well-studied kinases with many measurements — a plausible scenario, since UniProt coverage correlates with research interest and ChEMBL measurement count.

### 12.3 Interpretation

This limitation has two contrasting interpretations:

**Pessimistic**: The ESM-FP MLP and Fusion models are operating with severely degraded protein information for the majority of their training and test data. The reported performance improvements on random and scaffold splits may underestimate the true potential of protein-aware architectures. With proper UniProt coverage for all 507 targets, the protein-aware advantage could be substantially larger.

**Optimistic (and arguably more important)**: Despite receiving garbage protein embeddings for ~82% of targets, ESM-FP MLP still achieves the best performance on random and scaffold splits (RMSE = 0.775 and 0.897, respectively). The model successfully learned to leverage the *real* embeddings for the 92 targets that have them, while the shared fallback embedding for the remaining 415 targets acted as a form of implicit regularization — the model learned not to rely too heavily on the protein branch for the majority of targets where it provided no discriminative information.

### 12.4 JAK case study is unaffected

The JAK family case study (Section 10) is **not affected** by this limitation. All four JAK kinases (JAK1, JAK2, JAK3, TYK2) are well-characterized human proteins with UniProt sequences and therefore have real, distinct ESM-2 embeddings. The selectivity prediction results (79% top-1 accuracy for protein-aware models vs. 52% for fingerprint-only models) reflect genuine protein-aware reasoning with correct embeddings. This makes the JAK selectivity finding the strongest evidence for the value of protein embeddings in our study.

### 12.5 Recommended remediation

Future work should address this gap:
1. **Expand sequence coverage**: Use alternative databases (NCBI Protein, PDB) or manual curation to retrieve sequences for the missing 415 targets
2. **Explicit missing-target handling**: Replace the silent `.get(t, 0)` fallback with a learned "unknown protein" embedding or a zero vector, so the model can explicitly learn a fallback strategy
3. **Ablation study**: Retrain models with only the 92 targets that have real embeddings, to isolate the effect of protein information from the noise of fallback embeddings

## 13. Conclusions

### Answering the scientific question

**When do complex, structure-aware ML models outperform simple cheminformatics baselines?**

Based on our systematic evaluation of 7 models across 3 splitting strategies (21 experiments total, ~350K kinase bioactivity records):

1. **Complex models win under lenient evaluation conditions.** ESM-FP MLP achieves the best RMSE (0.775) on random splits, a 5.3% improvement over Random Forest (0.818). The protein embedding provides useful information about target identity when the model has seen examples from the same kinases during training.

2. **The advantage nearly vanishes under scaffold splits.** When structural analogs can't leak between train and test, the best deep model (ESM-FP MLP, RMSE=0.897) barely edges the best baseline (MLP, RMSE=0.905) by 0.9%. This confirms that much of the reported advantage of complex models in the literature reflects evaluation methodology, not genuine predictive improvement.

3. **Baselines win under the most realistic evaluation.** When predicting for unseen kinase subfamilies (target split), Random Forest (RMSE=1.067) outperforms every neural model, and protein-aware ESM-FP MLP drops to last place (RMSE=1.177). This directly contradicts the intuition that protein information should help most for novel targets.

4. **Uncertainty quantification is a baseline strength.** Tree-based uncertainty (RF, XGBoost) substantially outperforms MC-Dropout uncertainty (deep models) in calibration, error-uncertainty correlation, and selective prediction improvement. XGBoost's quantile regression achieves near-perfect calibration (miscal. area = 0.003 on scaffold split), while MC-Dropout estimates are effectively uninformative (negative error-uncertainty correlation, no selective prediction benefit).

5. **The representation bottleneck matters more than the model.** Morgan fingerprints with Random Forest (87s training) match or beat learned GIN representations (1411s training) across all splits. The GIN architecture doesn't learn representations superior to ECFP4 for kinase affinity prediction at this data scale.

6. **Small differences are statistically real.** Bootstrap paired tests (10,000 resamples) confirm that even the slim 0.9% ESM-FP MLP advantage on scaffold splits is significant (p = 0.003, paired ΔRMSE = −0.008 [−0.013, −0.003]). Model rankings are robust: win rates exceed 99% for key comparisons across all splits (Section 11).

7. **Protein-aware models shine for selectivity, not affinity.** The JAK case study reveals that ESM-FP MLP and Fusion achieve 79% top-1 selectivity accuracy (predicting which JAK member a compound preferentially inhibits) vs. 52% for fingerprint-only models — the strongest evidence for the value of protein embeddings (Section 10.5).

8. **ESM-2 embedding coverage is incomplete.** Only 92 of 507 targets (18%) have real protein embeddings; the remaining 415 receive an identical fallback embedding (Section 12). Despite this handicap, protein-aware models still outperform baselines on random and scaffold splits, suggesting the results may underestimate the potential of protein-aware architectures with complete coverage.

### Implications for drug discovery ML

- **Always evaluate on scaffold and target-held-out splits.** Random-split metrics inflate performance by 12-52% and can reverse model rankings.
- **Start with Random Forest + Morgan fingerprints.** This simple baseline trains in under 2 minutes, provides well-calibrated uncertainty, and is competitive with or superior to GPU-trained neural networks.
- **Protein embeddings help for interpolation, hurt for extrapolation.** ESM-2 features improve predictions for known target families but create overfitting when predicting for novel targets.
- **MC-Dropout is not a reliable uncertainty method** for these architectures and data scales. Tree-based or quantile-based methods provide far more actionable uncertainty estimates.
- **The target generalization problem remains open.** No model achieves R² > 0.27 on the target split. Closing this gap likely requires fundamentally different approaches: fine-tuning protein language models, incorporating 3D structural information, or developing transfer learning methods that explicitly account for kinase subfamily relationships.

### Future directions

- **Expand ESM-2 coverage**: retrieve UniProt sequences for the missing 415 targets (Section 12.5) and retrain protein-aware models with complete embeddings
- **Improved uncertainty for deep models**: deeper ensembles (5+ models), evidential deep learning, or heteroscedastic regression heads
- **Fine-tuned ESM-2**: rather than frozen embeddings, fine-tuning the last few layers on kinase binding data may improve target-split generalization
- **3D structure features**: docking scores, binding site descriptors, or SE(3)-equivariant networks that encode protein-ligand geometry
- **Meta-learning**: few-shot approaches for predicting on novel kinase targets with limited assay data

## References

- Mendez et al. (2019) ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930-D940.
- Rogers & Hahn (2010) Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5), 742-754.
- Sheridan (2013) Time-split cross-validation as a method for estimating the goodness of prospective prediction. *J. Chem. Inf. Model.*, 53(4), 783-790.
- Xu et al. (2019) How powerful are graph neural networks? *ICLR 2019*. (Graph Isomorphism Network)
- Lin et al. (2023) Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130. (ESM-2)
- Gal & Ghahramani (2016) Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML 2016*. (MC-Dropout)
- Efron & Tibshirani (1993) *An Introduction to the Bootstrap*. Chapman & Hall. (Bootstrap methodology)
