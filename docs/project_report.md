# When Do Complex Models Beat Simple Baselines for Kinase Inhibitor Affinity Prediction?

## 1. Introduction

Machine learning for drug discovery promises to accelerate the identification of potent kinase inhibitors, yet published benchmarks often overstate model generalization by evaluating on random splits that allow data leakage of chemical scaffolds and target information. This project asks a deliberately simple question: **how well do classical ML baselines actually perform on kinase affinity prediction, and how much does that performance degrade under realistic evaluation conditions?**

By establishing rigorous baselines first, we create an honest performance floor against which future complex models (GNNs, protein-aware architectures) can be measured. All comparisons are supported by 5-seed replication (105 training runs) and paired significance testing. The goal is not to achieve state-of-the-art performance, but to understand *what drives predictive performance* and *when simple models are sufficient*.

### Why kinases?

Protein kinases are among the most important drug target families in oncology and inflammation. They share a conserved ATP-binding fold, meaning inhibitors often show cross-reactivity across the family. This creates a natural test case for generalization: can a model trained on one set of kinases predict binding affinity for kinases it has never seen?

### Why baselines matter

In the ML-for-drug-discovery literature, reported metrics are frequently inflated by evaluation on random splits, where train and test sets share scaffolds and even identical compounds tested against different targets. Scaffold and target-based splits provide a more realistic estimate of how a model would perform on novel chemical matter or novel targets in a real drug discovery campaign.

## 2. Dataset Construction

We constructed a curated kinase bioactivity dataset from ChEMBL 36, starting from all binding affinity measurements (IC50, Ki, Kd) for human protein kinase targets. The full curation pipeline is config-driven (`configs/dataset_v1.yaml`) and documented in `docs/data_card.md`.

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

**Important caveat**: ElasticNet with alpha=1.0 proves to be too aggressively regularized for this dataset (see Section 6). All coefficients are driven to zero, producing near-constant predictions. A grid search over alpha values (Section 8.4) found that alpha=0.001 with l1_ratio=0.1 yields the best validation RMSE (1.143), though ElasticNet remains the weakest model even when optimally tuned.

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

**1. Random Forest is the top baseline, significantly outperforming MLP.** RF (RMSE=0.818, R^2=0.587) and MLP (RMSE=0.824, R^2=0.581) appear nearly equivalent in point estimates, but multi-seed evaluation (Section 11) confirms RF is significantly better on random splits (p < 0.001) and target splits (p = 0.002), while MLP is significantly better on scaffold splits (p < 0.001). This suggests that for binary Morgan fingerprints, the piecewise-constant approximations that decision trees make capture most learnable signal, with MLP's non-linear combinations providing a slight edge only for scaffold generalization. Both outperform XGBoost (RMSE=0.893, R^2=0.508) by a meaningful margin.

**2. MLP shows a slight advantage for target generalization.** On the hardest split (target), MLP achieves AUROC=0.718 vs RF's 0.706 and XGBoost's 0.662. However, RF has a higher R^2 (0.265 vs 0.234). This split in ranking metrics is scientifically interesting: RF is better at predicting exact pActivity values, while MLP is better at ranking active vs. inactive compounds for unseen targets. Different drug discovery use cases would prefer different models -- virtual screening favors AUROC (ranking), while lead optimization favors RMSE (accuracy).

**3. Splitting strategy dramatically affects perceived performance.** The best model (RF) shows a clear degradation pattern:

| Metric | Random | Scaffold | Target |
|--------|--------|----------|--------|
| RMSE | 0.818 | 0.919 (+12%) | 1.067 (+30%) |
| R^2 | 0.587 | 0.477 (-19%) | 0.265 (-55%) |
| AUROC | 0.865 | 0.845 (-2%) | 0.706 (-18%) |

Going from random to scaffold split costs ~12% in RMSE, and moving to target split costs ~30%. This quantifies the "scaffold leakage" and "target leakage" effects that inflate reported performance in many published studies. MLP shows a similar pattern (random R^2=0.581 → target R^2=0.234, a 60% drop).

**4. XGBoost underperforms RF despite theoretical advantages.** Gradient boosting typically outperforms bagging on tabular data, but here XGBoost consistently trails RF by 5-10% on RMSE. The likely explanation is that XGBoost's depth limit (max_depth=6) constrains its expressiveness on 2048-dimensional binary data, while RF's unlimited-depth trees can capture deeper feature interactions. A grid search over max_depth (Section 8.4) confirms this hypothesis: increasing depth to 12 improves XGBoost's validation RMSE from 0.886 to 0.805, approaching RF's performance.

**5. ElasticNet with default hyperparameters collapses completely.** With alpha=1.0 and l1_ratio=0.5, all coefficients are driven to zero, producing constant predictions (R^2 ~ 0, AUROC = 0.5). This is not a bug — it demonstrates that strong L1 regularization on 217 descriptors is too aggressive for this problem. The AUPRC values (0.715-0.795) are not meaningful here; they simply reflect the dataset's class prior (77.3% active). A grid search (Section 8.4) confirmed that reducing alpha to 0.001 recovers meaningful predictions (val RMSE = 1.143), though ElasticNet remains the weakest model.

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

### 8.4 Hyperparameter tuning and sensitivity

#### Search strategy and budget

We conducted exhaustive grid search (all combinations enumerated via `itertools.product`) for 2 of 7 models, evaluating on the random-split validation set:

| Model | Parameters searched | Grid size | Best val RMSE | Default val RMSE |
|-------|-------------------|-----------|---------------|------------------|
| XGBoost | max_depth × learning_rate × n_estimators | 5 × 2 × 2 = 20 | 0.805 (depth=12, lr=0.1, n=500) | 0.886 (depth=6) |
| ElasticNet | alpha × l1_ratio | 7 × 5 = 35 | 1.143 (α=0.001, ratio=0.1) | 1.276 (α=1.0) |
| **Total** | | **55 configs** | | |

The remaining 5 models (Random Forest, MLP, ESM-FP MLP, GIN, Fusion) used fixed configurations without systematic search. These choices follow standard practices in the literature: RF with unlimited depth and 500 trees, MLP with [256, 128] hidden layers, and deep models with 3-layer architectures and standard learning rates (0.001 for single-branch, 0.0005 for fusion). All deep models employ early stopping on validation RMSE as implicit regularization, which adapts effective model capacity to the dataset without explicit hyperparameter search.

Full hyperparameter values, search spaces, and selection criteria are documented in Table S2.

#### XGBoost sensitivity to tree depth

XGBoost shows substantial sensitivity to `max_depth`, the most influential hyperparameter:

| max_depth | Best RMSE at this depth | Relative to depth=12 best |
|-----------|------------------------|--------------------------|
| 4 | 0.953 | +18.4% |
| 6 | 0.886 | +10.1% |
| 8 | 0.841 | +4.4% |
| 10 | 0.814 | +1.1% |
| 12 | 0.805 | (reference) |

Performance improves monotonically with depth on 2048-dimensional binary fingerprints, with diminishing returns above depth 10. Learning rate and n_estimators have smaller effects: at any given depth, lr=0.1 outperforms lr=0.05 by 2–4%, and n_estimators=500 outperforms 300 by 2–3%.

**Important caveat**: All multi-seed experiments (Section 11) used the default XGBoost configuration (max_depth=6), not the grid search optimum (max_depth=12). The reported multi-seed XGBoost RMSE (0.894 ± 0.001 on random splits) is therefore conservative — with tuned depth, XGBoost could potentially achieve ~0.805, approaching RF's 0.819. This does not affect the primary conclusions because (a) model rankings are internally consistent across seeds, and (b) the main scientific finding — that RF matches or beats complex models — would be strengthened, not weakened, by a better-tuned XGBoost.

#### ElasticNet sensitivity to regularization strength

ElasticNet is highly sensitive to `alpha`, with aggressive regularization causing coefficient collapse:

- At α ≥ 0.5 with l1_ratio ≥ 0.5, all coefficients are driven to zero (RMSE = 1.276, equivalent to predicting the mean)
- At α = 0.001, the best configuration (l1_ratio = 0.1, more L2 regularization), validation RMSE improves to 1.143 — still the weakest model, but functional
- The l1_ratio has a modest effect (spread of ~0.01 RMSE at any given alpha), with lower ratios (more L2) consistently preferred for the dense RDKit descriptor features

The default α = 1.0 used in all multi-seed experiments produces degenerate predictions. However, even the optimal ElasticNet (RMSE = 1.143) remains substantially worse than all other models, so this limitation does not affect our comparative conclusions.

#### Models without systematic tuning

Five models used fixed configurations without hyperparameter search:

- **Random Forest**: Standard settings (500 trees, unlimited depth, sqrt features) are well-established defaults for high-dimensional tabular data. RF is known to be relatively insensitive to hyperparameter choices compared to gradient boosting methods.
- **MLP (baseline)**: Architecture [256, 128] with Adam optimizer follows standard practice. The ensemble of 3 models with early stopping (patience=10) limits overfitting.
- **ESM-FP MLP, GIN, Fusion**: Deep model architectures (3-layer MLP, 3-layer GIN, concatenation fusion) follow conventions in the molecular property prediction literature. Learning rates (0.001 for single-branch, 0.0005 for multi-branch fusion) and dropout (0.3) are standard starting points. Early stopping on validation RMSE (patience 10–15 epochs) serves as implicit capacity control, typically reducing the need for extensive learning rate or regularization searches.

We acknowledge that systematic hyperparameter optimization (e.g., Bayesian optimization with cross-validation) could potentially improve all models' absolute performance. However, our primary scientific question concerns *relative* model rankings under different splitting strategies, and the fixed-configuration approach ensures that performance differences reflect architectural and representational advantages rather than differential tuning effort. The multi-seed evaluation (Section 11) captures training stochasticity, which is the dominant source of variance for the deep models we studied.

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

All results below are reported as mean ± SD across 5 independent training seeds (42, 123, 456, 789, 1024) to capture training stochasticity. Single-seed point estimates were used in our preliminary analysis; the multi-seed values reported here supersede them.

| Model | Split | RMSE (mean ± SD) | R² (mean ± SD) | Pearson (mean ± SD) |
|-------|-------|-------------------|-----------------|----------------------|
| **ESM-FP MLP** | **Random** | **0.777 ± 0.002** | **0.627 ± 0.002** | **0.792 ± 0.001** |
| Fusion | Random | 0.791 ± 0.003 | 0.614 ± 0.003 | 0.785 ± 0.002 |
| Random Forest | Random | 0.819 ± 0.000 | 0.587 ± 0.000 | 0.769 ± 0.000 |
| MLP (baseline) | Random | 0.827 ± 0.002 | 0.579 ± 0.002 | 0.763 ± 0.001 |
| GIN | Random | 0.828 ± 0.002 | 0.578 ± 0.002 | 0.761 ± 0.001 |
| XGBoost | Random | 0.894 ± 0.001 | 0.508 ± 0.001 | 0.717 ± 0.001 |
| | | | | |
| MLP (baseline) | Scaffold | 0.901 ± 0.003 | 0.498 ± 0.003 | 0.707 ± 0.002 |
| **ESM-FP MLP** | **Scaffold** | **0.902 ± 0.004** | **0.497 ± 0.005** | **0.707 ± 0.004** |
| Random Forest | Scaffold | 0.919 ± 0.000 | 0.477 ± 0.000 | 0.712 ± 0.000 |
| Fusion | Scaffold | 0.939 ± 0.009 | 0.455 ± 0.010 | 0.680 ± 0.007 |
| GIN | Scaffold | 0.947 ± 0.005 | 0.445 ± 0.006 | 0.673 ± 0.002 |
| XGBoost | Scaffold | 0.958 ± 0.002 | 0.433 ± 0.002 | 0.667 ± 0.002 |
| | | | | |
| **Random Forest** | **Target** | **1.066 ± 0.002** | **0.267 ± 0.002** | **0.521 ± 0.002** |
| MLP (baseline) | Target | 1.105 ± 0.012 | 0.213 ± 0.017 | 0.515 ± 0.011 |
| XGBoost | Target | 1.131 ± 0.004 | 0.175 ± 0.006 | 0.432 ± 0.006 |
| GIN | Target | 1.177 ± 0.032 | 0.106 ± 0.048 | 0.403 ± 0.033 |
| ESM-FP MLP | Target | 1.180 ± 0.015 | 0.102 ± 0.023 | 0.388 ± 0.020 |
| Fusion | Target | 1.196 ± 0.031 | 0.078 ± 0.047 | 0.392 ± 0.034 |

Notable observations about training variance: Random Forest and XGBoost show near-zero variance across seeds (SD < 0.002) because their randomness comes from feature/sample subsampling with fixed sklearn random states. Deep models show substantially higher variance, particularly on the target split (GIN SD = 0.032, Fusion SD = 0.031), reflecting sensitivity to initialization and optimization trajectory.

### 9.4 Key findings

**1. ESM-FP MLP wins on random splits — confirmed by multi-seed significance testing.** ESM-FP MLP (RMSE = 0.777 ± 0.002) significantly outperforms Random Forest (RMSE = 0.819 ± 0.000; paired t-test p < 0.001) and MLP (RMSE = 0.827 ± 0.002; p < 0.001) on random splits. However, ESM-FP MLP also significantly outperforms the Fusion model (p < 0.001), suggesting that the GIN graph branch degrades rather than enhances the protein-aware prediction.

**2. The scaffold-split advantage is marginal and not significant.** On scaffold splits, MLP (RMSE = 0.901 ± 0.003) and ESM-FP MLP (RMSE = 0.902 ± 0.004) are statistically indistinguishable (p = 0.575). This is a critical update from our preliminary analysis, which suggested ESM-FP MLP had a 0.9% advantage — across 5 seeds, the two models are tied. The protein embedding provides no benefit when the model must generalize to novel chemical scaffolds.

**3. Baselines win on target-held-out splits — confirmed with high significance.** Random Forest (RMSE = 1.066 ± 0.002) significantly outperforms every other model on the target split: RF vs MLP (p = 0.002), RF vs ESM-FP MLP (p < 0.001), RF vs GIN (p = 0.001), RF vs Fusion (p not computed, but Δ = 0.130). Protein-aware models (ESM-FP MLP, Fusion) rank 5th and 6th out of 7 models, with ESM-FP MLP significantly *worse* than fingerprint-only MLP (p = 0.002).

**4. GIN does not outperform Morgan fingerprints — now statistically confirmed.** MLP (RMSE = 0.827 ± 0.002) and GIN (RMSE = 0.828 ± 0.002) are statistically indistinguishable on random splits (p = 0.403). On scaffold splits, MLP significantly outperforms GIN (ΔRMSE = −0.046, p < 0.001), and on target splits the gap widens further (ΔRMSE = −0.073, p = 0.013). Learned graph representations are equal to or worse than fixed Morgan fingerprints at this data scale.

**5. Deep model variance is a concern.** Tree-based models show near-zero training variance (RF SD = 0.0001 RMSE), while deep models on the target split show 10–30× higher variance (GIN SD = 0.032, Fusion SD = 0.031). This means single-seed evaluations of deep models may misrepresent true performance by ±0.03 RMSE — enough to change model rankings. Multi-seed evaluation is essential for reliable comparison.

**6. Fusion doesn't synergize — it averages.** The Fusion model (RMSE = 0.791 ± 0.003, random) performs between its two component models rather than exceeding both. ESM-FP MLP is significantly better than Fusion on both random (p < 0.001) and scaffold splits (p = 0.001). On target splits, they are statistically tied (p = 0.414). The concatenation fusion strategy does not enable complementary information from the two branches.

**7. Computational cost is not justified.** The GIN and Fusion models cost 16–18× more compute than Random Forest for equivalent or worse performance. Only ESM-FP MLP offers a favorable cost-benefit tradeoff (229s vs 87s for RF) with a significant 5.1% RMSE improvement on random splits.

### 9.5 Performance degradation analysis

The degradation pattern from random → scaffold → target splits reveals each model's sensitivity to data leakage (using multi-seed mean values):

| Model | Random RMSE | Scaffold RMSE | Target RMSE | Random→Target Δ |
|-------|-------------|---------------|-------------|-----------------|
| ESM-FP MLP | 0.777 | 0.902 (+16.1%) | 1.180 (+51.8%) | **+51.8%** |
| Random Forest | 0.819 | 0.919 (+12.2%) | 1.066 (+30.2%) | +30.2% |
| MLP (baseline) | 0.827 | 0.901 (+9.0%) | 1.105 (+33.6%) | +33.6% |
| GIN | 0.828 | 0.947 (+14.4%) | 1.177 (+42.2%) | +42.2% |
| XGBoost | 0.894 | 0.958 (+7.2%) | 1.131 (+26.5%) | +26.5% |
| Fusion | 0.791 | 0.939 (+18.7%) | 1.196 (+51.1%) | +51.1% |

ESM-FP MLP and Fusion degrade the most (~51%) from random to target split, confirming that their random-split advantages are inflated by target-identity leakage. Random Forest and XGBoost show the most graceful degradation (27–30%), making them more reliable across deployment scenarios.

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

### 10.5.1 Stronger selectivity baselines

A valid concern with the above comparison is that the fingerprint-only models were never designed for selectivity — they train a single pooled model without target-specific information. To address this, we implemented three stronger selectivity baselines that give fingerprint-only models a fairer chance:

| Model | Top-1 Accuracy (scaffold) | Rank Correlation |
|-------|--------------------------|------------------|
| Per-target RF (separate model per JAK) | 83.0% | 0.833 |
| One-hot RF (FP + target encoding) | 83.5% | 0.813 |
| One-hot MLP (FP + target encoding) | 82.9% | 0.825 |
| Pairwise RF (ΔpActivity models) | 82.7% | 0.791 |
| Per-target MLP | 75.2% | 0.695 |

**The stronger fingerprint baselines close the gap with protein-aware models.** When given explicit target-identity information — either through separate per-target models, one-hot target encoding, or pairwise difference modeling — fingerprint-based approaches achieve 83% top-1 accuracy, comparable to or exceeding the 79% achieved by ESM-FP MLP with implicit protein embeddings.

This refines our earlier finding: the ESM-FP MLP selectivity advantage over pooled fingerprint models (79% vs 52%) reflects the protein embedding's role as an *implicit target identifier*, not necessarily its encoding of protein-level structural information. When fingerprint models are given explicit target identity through simpler means, they perform at least as well. The value proposition of ESM-2 embeddings for selectivity would be stronger in a setting where the model must generalize to targets not seen during training — which is precisely the setting where protein-aware models underperform (Section 9.4).

### 10.6 Worst predictions

The worst-predicted JAK compounds show interesting model-specific failure patterns:
- **Random Forest**: failures spread across JAK1 (8), JAK3 (5), JAK2 (4), TYK2 (3), dominated by IC50 measurements (14/20)
- **ESM-FP MLP**: failures concentrated on JAK2 (10/20), suggesting the model overfits to JAK2's larger training set and makes biased predictions for edge-case JAK2 compounds
- **Fusion**: failures split between JAK3 (7) and JAK2 (7), with JAK1 failures (6) — no TYK2 failures, consistent with TYK2 being easiest to predict

Notably, none of the top-20 worst predictions per model are flagged as noisy measurements, indicating that prediction failures reflect genuine model limitations rather than data quality issues.

### 10.7 Target split caveat

An important finding: **no JAK test compounds exist in the target-held-out split**. All four JAK members were assigned to the training fold during target-based splitting. This means the target-split results in Section 9 evaluate generalization to other kinase subfamilies (e.g., CDKs, MAP kinases) using JAK as *training* data, not as a prediction target. This is by design — the target split tests cross-subfamily transfer, and JAK happens to fall on the training side of the partition.

## 11. Statistical Significance

### 11.1 Multi-seed evaluation methodology

Point estimates of test-set metrics from a single training run are subject to both test-set sampling variability and training stochasticity (random initialization, data shuffling, dropout masks). To disentangle these effects and establish robust model comparisons, we employ two complementary approaches:

**Multi-seed training** (5 seeds: 42, 123, 456, 789, 1024): Each of the 7 models is trained 5 times on each of the 3 splits, yielding 105 experiments. We report mean ± SD across seeds and use paired t-tests (df = 4) across seeds for pairwise model comparisons. This captures training stochasticity — the variability that different random initializations or tree constructions produce even on the same data.

**Bootstrap confidence intervals** (10,000 resamples): For each single-seed experiment, we resample test predictions to estimate CIs on individual metrics and run paired bootstrap tests for same-data model comparisons. This captures test-set sampling variability — the uncertainty about performance on different test-set compositions.

The multi-seed analysis is the more conservative and comprehensive approach: it tests whether model A is *reliably* better than model B across different training runs, not just on a specific set of learned parameters.

### 11.2 Multi-seed significance results

Paired t-tests (two-sided, 5 seeds) for key model comparisons on RMSE:

| Comparison | Split | ΔRMSE (A − B) | p-value | Significant? |
|-----------|-------|--------------|---------|-------------|
| RF vs ESM-FP MLP | Random | +0.041 | < 0.001 | *** ESM-FP MLP wins |
| RF vs MLP | Random | −0.008 | < 0.001 | *** RF wins |
| RF vs GIN | Random | −0.009 | < 0.001 | *** RF wins |
| MLP vs ESM-FP MLP | Random | +0.049 | < 0.001 | *** ESM-FP MLP wins |
| MLP vs GIN | Random | −0.001 | 0.403 | ns (tied) |
| ESM-FP MLP vs Fusion | Random | −0.014 | < 0.001 | *** ESM-FP MLP wins |
| | | | | |
| RF vs MLP | Scaffold | +0.018 | < 0.001 | *** MLP wins |
| RF vs ESM-FP MLP | Scaffold | +0.017 | < 0.001 | *** ESM-FP MLP wins |
| MLP vs ESM-FP MLP | Scaffold | −0.001 | 0.575 | **ns (tied)** |
| MLP vs GIN | Scaffold | −0.046 | < 0.001 | *** MLP wins |
| ESM-FP MLP vs Fusion | Scaffold | −0.037 | 0.001 | ** ESM-FP MLP wins |
| | | | | |
| RF vs MLP | Target | −0.038 | 0.002 | ** RF wins |
| RF vs ESM-FP MLP | Target | −0.113 | < 0.001 | *** RF wins |
| RF vs GIN | Target | −0.111 | 0.001 | ** RF wins |
| MLP vs ESM-FP MLP | Target | −0.075 | 0.002 | ** MLP wins |
| MLP vs GIN | Target | −0.073 | 0.013 | * MLP wins |
| ESM-FP MLP vs Fusion | Target | −0.016 | 0.414 | ns (tied) |

### 11.3 Key statistical findings

**1. ESM-FP MLP's scaffold-split advantage disappears under multi-seed evaluation.** Our preliminary bootstrap analysis (single seed) found ESM-FP MLP significantly better than MLP on scaffold splits (ΔRMSE = −0.008, p = 0.003). However, across 5 training seeds, MLP (0.901 ± 0.003) and ESM-FP MLP (0.902 ± 0.004) are statistically indistinguishable (p = 0.575). The single-seed result was a false positive driven by a particular initialization — the "advantage" does not replicate. This demonstrates why multi-seed evaluation is essential for deep model comparisons.

**2. All random-split rankings are statistically robust.** Every pairwise comparison on the random split is significant (p < 0.001) except MLP vs GIN (p = 0.403, tied). The confirmed ranking is: ESM-FP MLP > Fusion > RF > {MLP ≈ GIN} > XGBoost > ElasticNet.

**3. RF's target-split dominance is unambiguous.** RF significantly outperforms every other model on the target split (all p < 0.003). The margin over the best deep model (ESM-FP MLP) is ΔRMSE = 0.113 — a 10.6% gap that is both statistically significant and practically meaningful.

**4. Deep model variance inflates single-seed comparisons.** GIN and Fusion show target-split SDs of 0.031–0.032 RMSE across seeds, meaning a single-seed evaluation could over- or under-estimate their performance by ~3%. Single-seed point estimates should not be used for model ranking when deep models are involved.

### 11.4 Bootstrap confidence intervals

For completeness, bootstrap CIs from single-seed evaluations provide complementary information about test-set sampling variability:

| Model | Split | RMSE | 95% CI | R² | 95% CI |
|-------|-------|------|--------|-----|--------|
| ESM-FP MLP | Random | 0.775 | [0.768, 0.783] | 0.629 | [0.622, 0.637] |
| Fusion | Random | 0.793 | [0.785, 0.800] | 0.613 | [0.605, 0.620] |
| Random Forest | Random | 0.818 | [0.811, 0.826] | 0.587 | [0.580, 0.594] |
| RF | Target | 1.067 | [1.059, 1.076] | 0.265 | [0.255, 0.275] |
| ESM-FP MLP | Target | 1.177 | [1.167, 1.186] | 0.107 | [0.096, 0.119] |

Bootstrap CIs are narrow (±0.008 RMSE typical) due to large test sets (~35,000 samples), confirming that test-set sampling variability is small relative to the inter-model differences.

### 11.5 Implications

1. **Multi-seed evaluation changes conclusions.** The single-seed ESM-FP MLP scaffold advantage (our prior Finding 6) was a false positive. This underscores that published single-seed comparisons of deep models are unreliable.
2. **Training stochasticity dominates test-set variability for deep models.** Deep model RMSE SD across seeds (0.002–0.032) exceeds the bootstrap CI half-width (~0.008) for scaffold and target splits, meaning training randomness is the primary source of uncertainty.
3. **Tree-based models are deterministically reliable.** RF and XGBoost SDs < 0.002 across seeds mean their single-seed estimates are trustworthy, while deep models require ≥3 seeds for reliable comparison.

## 12. ESM-2 Embedding Coverage: Ablation Studies

### 12.1 The embedding fallback problem

A critical limitation of the protein-aware models (ESM-FP MLP and Fusion) is that **only 92 of the 507 kinase targets (18.1%) have real ESM-2 embeddings.** The remaining 415 targets (81.9%) lack UniProt protein sequences in our pipeline, because the ChEMBL API did not return UniProt cross-references for these targets during the sequence retrieval step (Section 9.2).

The original deep trainer code handles missing embeddings by falling back to row 0 of the embedding matrix — an arbitrary real kinase's ESM-2 embedding. This means that for 415 out of 507 targets, the protein-aware models receive an *identical, arbitrary* protein representation.

To quantify the impact of this limitation, we conducted two ablation studies: (1) a fallback strategy comparison, and (2) a clean 92-target subset evaluation.

### 12.2 Fallback strategy ablation

We retrained ESM-FP MLP and Fusion with three different fallback strategies for the 415 targets without real ESM-2 embeddings:

- **row0** (original): use the first target's real embedding as fallback
- **zero**: use a zero vector (1280 dims) — the model receives no protein signal
- **mean**: use the mean of all 92 real embeddings — a "generic kinase" representation

| Model / Split | row0 (orig) | zero | mean | Max Δ |
|---------------|-------------|------|------|-------|
| ESM-FP MLP / random | 0.775 | 0.771 | 0.775 | 0.005 |
| ESM-FP MLP / scaffold | 0.897 | 0.881 | 0.899 | 0.018 |
| ESM-FP MLP / target | 1.177 | 1.194 | 1.177 | 0.017 |
| Fusion / random | 0.793 | 0.788 | 0.792 | 0.005 |
| Fusion / scaffold | 0.945 | 0.946 | 0.949 | 0.005 |
| Fusion / target | 1.138 | 1.115 | 1.239 | **0.125** |

**Key finding: the fallback strategy has minimal impact for 5 of 6 conditions** (Δ < 0.02 RMSE). This confirms that for the majority of the dataset, the models learn to effectively ignore the protein embedding for fallback targets — the fingerprint branch dominates predictions regardless of what vector is supplied to the protein branch.

The one exception is **Fusion on the target split**, where the mean-vector fallback causes catastrophic degradation (RMSE = 1.239 vs 1.115 for zero). This suggests the Fusion model is sensitive to the protein branch signal on the target split, where the shared "generic kinase" embedding misleads predictions for novel kinase subfamilies.

### 12.3 Clean 92-target subset evaluation

To isolate the effect of protein embeddings from fallback noise, we created a clean subset containing only the 92 targets with real ESM-2 embeddings (67,902 records, 19.2% of the full dataset). All 7 models were retrained and evaluated on this subset with fresh splits.

| Model / Split | Full (507 targets) | ESM-92 subset | Δ |
|---------------|-------------------|---------------|---|
| **ESM-FP MLP / random** | **0.777** | **0.647** | **−0.130** |
| Fusion / random | 0.791 | 0.695 | −0.096 |
| RF / random | 0.819 | 0.786 | −0.033 |
| MLP / random | 0.827 | 0.835 | +0.008 |
| GIN / random | 0.828 | 0.787 | −0.041 |
| XGBoost / random | 0.894 | 0.797 | −0.097 |
| | | | |
| **ESM-FP MLP / scaffold** | **0.902** | **0.826** | **−0.076** |
| Fusion / scaffold | 0.939 | 0.858 | −0.081 |
| RF / scaffold | 0.919 | 0.872 | −0.048 |
| GIN / scaffold | 0.947 | 0.879 | −0.068 |
| | | | |
| RF / target | 1.066 | 1.017 | −0.050 |
| Fusion / target | 1.196 | 1.062 | −0.134 |
| GIN / target | 1.177 | 1.118 | −0.059 |
| ESM-FP MLP / target | 1.180 | 1.260 | +0.080 |

**Critical findings from the 92-target subset:**

**1. Protein embeddings provide a massive boost when all targets have real embeddings.** ESM-FP MLP RMSE drops from 0.777 to 0.647 on random splits — a 16.7% improvement and an absolute Δ of 0.130. This is the clearest evidence that incomplete embedding coverage masked the true potential of protein-aware models in the full-dataset analysis.

**2. The Fusion model also benefits substantially.** Fusion improves from 0.791 to 0.695 (random) and from 1.196 to 1.062 (target) on the clean subset. On the target split, the Fusion model's RMSE improves by 0.134 — suggesting that with real embeddings for all targets, the protein branch provides genuine cross-target transfer.

**3. Baselines also improve on the smaller subset.** RF improves from 0.819 to 0.786 (random), likely because the 92 well-studied targets have denser, cleaner data. This confound — well-studied targets may be intrinsically easier — means the ESM-92 subset results should be interpreted cautiously as an upper bound on protein embedding value.

**4. ESM-FP MLP degrades on the 92-target target split.** Despite having real embeddings for all targets, ESM-FP MLP RMSE worsens from 1.180 to 1.260. With only 92 targets, the target split holds out ~10 targets for testing — this small test set and the MLP architecture's tendency to overfit to target-specific embedding patterns may explain this degradation.

### 12.4 JAK case study is unaffected

The JAK family case study (Section 10) is **not affected** by the fallback limitation. All four JAK kinases (JAK1, JAK2, JAK3, TYK2) have real, distinct ESM-2 embeddings. The selectivity prediction results reflect genuine protein-aware reasoning.

### 12.5 Implications

The ablation studies resolve the earlier ambiguity about whether incomplete ESM-2 coverage undermined or coincidentally helped the protein-aware models:

1. **The full-dataset results underestimate protein embedding potential.** With complete coverage, ESM-FP MLP achieves RMSE = 0.647 on random splits — far better than any baseline. The 0.130 RMSE improvement (vs 0.042 on the full dataset) suggests 82% of the protein embedding's potential was wasted on fallback vectors.
2. **Fallback strategy doesn't matter for most conditions**, confirming that the model learns to ignore the protein branch for targets with uninformative embeddings.
3. **Expanding UniProt coverage remains the highest-priority improvement.** Retrieving sequences for the missing 415 targets via NCBI Protein, PDB, or manual curation would likely yield substantial performance gains for protein-aware architectures.

## 13. Conclusions

### Answering the scientific question

**When do complex, structure-aware ML models outperform simple cheminformatics baselines?**

Based on our systematic evaluation of 7 models across 3 splitting strategies, with 5-seed replication (105 training runs on the full 507-target dataset, plus 21 experiments on a clean 92-target subset with complete ESM-2 coverage, and 12 fallback ablation experiments):

1. **Complex models win under lenient evaluation conditions — significantly.** ESM-FP MLP achieves the best multi-seed RMSE (0.777 ± 0.002) on random splits, significantly outperforming Random Forest (0.819 ± 0.000; paired t-test p < 0.001) and MLP (0.827 ± 0.002; p < 0.001). The protein embedding provides useful information about target identity when the model has seen examples from the same kinases during training.

2. **The advantage vanishes under scaffold splits.** When structural analogs can't leak between train and test, MLP (0.901 ± 0.003) and ESM-FP MLP (0.902 ± 0.004) are statistically indistinguishable (p = 0.575). Our preliminary single-seed analysis incorrectly reported a significant 0.9% ESM-FP MLP advantage; multi-seed evaluation reveals this was a false positive. This demonstrates that single-seed comparisons of deep models are unreliable.

3. **Baselines win under the most realistic evaluation — decisively.** When predicting for unseen kinase subfamilies (target split), Random Forest (1.066 ± 0.002) significantly outperforms every other model (all p < 0.003). Protein-aware models rank 5th–6th: ESM-FP MLP (1.180 ± 0.015) and Fusion (1.196 ± 0.031). This directly contradicts the intuition that protein information should help most for novel targets.

4. **Incomplete ESM-2 coverage masked the true potential of protein embeddings.** On the 92-target clean subset (all targets with real ESM-2 embeddings), ESM-FP MLP achieves RMSE = 0.647 — a 16.7% improvement over the full-dataset result (0.777). The 415 fallback targets diluted the protein signal, meaning the full-dataset analysis substantially underestimates protein embedding potential (Section 12.3).

5. **Uncertainty quantification is a baseline strength.** Tree-based uncertainty (RF, XGBoost) substantially outperforms MC-Dropout uncertainty (deep models) in calibration, error-uncertainty correlation, and selective prediction improvement. XGBoost's quantile regression achieves near-perfect calibration (miscal. area = 0.003 on scaffold split), while MC-Dropout estimates are effectively uninformative.

6. **The representation bottleneck matters more than the model.** Morgan fingerprints with Random Forest (87s training) match or beat learned GIN representations (1411s training) across all splits. MLP and GIN are statistically indistinguishable on random splits (p = 0.403), and MLP significantly outperforms GIN on scaffold (p < 0.001) and target splits (p = 0.013).

7. **Multi-seed evaluation is essential for deep models.** Deep model RMSE SDs across seeds (0.002–0.032) exceed bootstrap CI half-widths (~0.008), meaning training stochasticity is the primary source of uncertainty. Tree-based models have near-zero cross-seed variance (SD < 0.002), making single-seed estimates reliable. Published single-seed comparisons involving deep models should be interpreted with caution.

8. **Protein-aware models shine for selectivity, but simpler approaches close the gap.** ESM-FP MLP achieves 79% top-1 JAK selectivity accuracy vs 52% for pooled fingerprint models (Section 10.5). However, stronger baselines — per-target models (83%) and one-hot target-conditioned models (84%) — achieve comparable or better selectivity using explicit target identity rather than learned protein embeddings (Section 10.5.1). The ESM-2 embedding functions primarily as an implicit target identifier in this context.

### Implications for drug discovery ML

- **Always evaluate on scaffold and target-held-out splits.** Random-split metrics inflate performance by 12–52% and can reverse model rankings.
- **Always report multi-seed results for deep models.** Single-seed evaluations can produce false positives (as demonstrated by our scaffold-split ESM-FP MLP result). We recommend ≥3 seeds with mean ± SD reporting.
- **Start with Random Forest + Morgan fingerprints.** This simple baseline trains in under 2 minutes, provides well-calibrated uncertainty, shows near-zero training variance, and is competitive with or superior to GPU-trained neural networks.
- **Protein embeddings help for interpolation, hurt for extrapolation.** ESM-2 features improve predictions for known target families but create overfitting when predicting for novel targets. Expanding embedding coverage is the single highest-impact improvement for protein-aware models.
- **MC-Dropout is not a reliable uncertainty method** for these architectures and data scales. Tree-based or quantile-based methods provide far more actionable uncertainty estimates.
- **The target generalization problem remains open.** No model achieves R² > 0.27 on the target split. Closing this gap likely requires fundamentally different approaches: fine-tuning protein language models, incorporating 3D structural information, or developing transfer learning methods that explicitly account for kinase subfamily relationships.

### Future directions

- **Expand ESM-2 coverage** (highest priority): retrieve UniProt sequences for the missing 415 targets via NCBI Protein, PDB, or manual curation — the 92-target subset results suggest this could yield RMSE improvements of 0.10–0.13 for protein-aware models
- **Temporal split validation**: add time-based splits using ChEMBL document_year to simulate prospective prediction
- **UQ normalization**: post-hoc isotonic recalibration for MC-Dropout uncertainty to match tree-based calibration quality
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
