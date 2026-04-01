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

*Results pending completion of all experiments. Will include calibration plots, error-uncertainty correlation, and selective prediction analysis.*

## 8. Error Analysis

*Planned for Phase 5. Will analyze failure modes including activity cliffs, rare scaffolds, noisy measurements, and activity type mixing.*

## 9. Case Study: [Target Family TBD]

*Planned for Phase 6. Deep dive on a specific kinase subfamily with compound-level analysis of good and bad predictions.*

## 10. Conclusions and Next Steps

### What we have established

Phase 4 provides a rigorous baseline performance floor for kinase affinity prediction across all 12 experiments (4 models x 3 splits):

- **Best baselines**: RF and MLP are near-equivalent with Morgan fingerprints (RMSE ~0.82, R^2 ~0.58 on random split), suggesting that the fingerprint representation — not the model — is the performance bottleneck
- **Splitting matters enormously**: random-split metrics overestimate real-world performance by 30-55% depending on the metric. Target-split R^2 drops to 0.23-0.27
- **Neural networks don't help (yet)**: MLP provides no improvement over RF for fingerprint features, indicating that non-linear feature combinations are not the limiting factor. The benefit of neural architectures may emerge only with richer input representations (protein embeddings, 3D structure)
- **Simple linear models are insufficient**: ElasticNet with default regularization cannot capture kinase SAR from 2D descriptors alone
- **Cross-target transfer is partial**: fingerprint-based models capture some generalizable SAR (likely ATP-site interactions), but large gaps remain for novel kinase targets — the core motivation for protein-aware models

### Future phases

- **Phase 5** (Evaluation): hyperparameter optimization, uncertainty calibration analysis, error analysis by activity type and target family
- **Phase 6** (Case studies): deep-dive on a specific kinase subfamily
- **Phase 7** (Advanced models): GNN, ESM-2 protein embeddings, and fusion models -- evaluated using the same framework to enable fair comparison against these baselines

## References

- Mendez et al. (2019) ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930-D940.
- Rogers & Hahn (2010) Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5), 742-754.
- Sheridan (2013) Time-split cross-validation as a method for estimating the goodness of prospective prediction. *J. Chem. Inf. Model.*, 53(4), 783-790.
