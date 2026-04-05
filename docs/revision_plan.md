# Preprint Revision Plan: Addressing Reviewer Gaps

**Date:** 2026-04-05
**Status:** Planning
**Estimated compute:** ~48-72 GPU-hours on AWS (g5.xlarge or similar)

---

## Overview

The reviewer identifies 7 gaps, of which **3 are structural blockers** (protein embedding coverage, CI scope, single partition) and **4 are strengthening items** (selectivity baselines, reproducibility details, temporal split, UQ normalization). This plan addresses all 7, prioritized by impact on the paper's credibility.

### Priority Tiers

| Tier | Gaps Addressed | Impact | Effort |
|------|---------------|--------|--------|
| **P0: Must-do** | #1 (92-target subset + fallback ablation) | Fixes the central methodological flaw | High |
| **P1: Critical** | #2, #3 (multi-seed, multi-partition) | Establishes benchmark robustness | High |
| **P2: Important** | #4 (selectivity baselines), #5 (supplement tables) | Strengthens claims, self-containment | Medium |
| **P3: Aspirational** | #6 (temporal split), #7 (UQ normalization) | Nice-to-have for completeness | Medium-Low |

---

## P0: Clean Protein-Aware Benchmark (Gap #1)

### The Problem
Only 92/507 targets have real ESM-2 embeddings. The other 415 use `target_to_row.get(t, 0)` — the embedding of whichever kinase sorts first alphabetically. This is **not** uninformative; it's a structured, biologically meaningful vector shared across 415 unrelated kinases. The protein-aware vs. ligand-only comparison is therefore confounded.

### Solution: Three-pronged approach

#### P0-A: 92-Target Clean Subset Benchmark

**What:** Filter the curated dataset to only the 92 targets with real ESM-2 embeddings. Generate new splits. Retrain all 7 models. Report as a companion analysis (or potentially as the primary protein-aware benchmark).

**Implementation:**

1. **Create subset dataset** (`src/kinase_affinity/data/subset.py`)
   ```python
   def create_embedding_subset(dataset_version="v1"):
       """Filter curated_activities to targets with real ESM-2 embeddings."""
       # Load target_index.json to get the 92 real-embedding targets
       # Filter curated_activities.parquet to only those targets
       # Save as curated_activities_esm92.parquet
       # Log: n_records, n_compounds, n_targets, activity distribution
   ```

2. **Generate splits for the 92-target subset**
   - Random, scaffold, and target splits on the filtered data
   - Use same split parameters (0.8/0.1/0.1, seed=42)
   - Save to `data/processed/v1/splits_esm92/`
   - For target split: reassign kinase group holdout using only the 92 targets' family structure

3. **Retrain all 7 models on 92-target subset**
   - 7 models × 3 splits = 21 experiments
   - Save predictions to `results/predictions_esm92/`
   - Save metrics to `results/tables/esm92_summary.csv`

4. **Run bootstrap analysis on 92-target results**
   - CIs + paired tests comparing protein-aware vs ligand-only
   - This is the *clean* comparison: every protein-aware prediction uses a real embedding

5. **Compare 507-target vs 92-target results**
   - Side-by-side table: does the protein-aware advantage grow when embeddings are real?
   - If ESM-FP MLP improves more on the 92-target subset, it proves the fallback was hurting
   - If performance is similar, it suggests protein embeddings add less than hoped

**Files to create/modify:**
- `src/kinase_affinity/data/subset.py` (NEW)
- `scripts/aws_run_esm92.sh` (NEW) — AWS script for the 92-target benchmark
- `configs/dataset_v1_esm92.yaml` (NEW) — config pointing to subset data

**Compute:** ~6-12 hours (same 21 experiments but on smaller dataset)

#### P0-B: Zero-Vector Fallback Ablation

**What:** Replace the `.get(t, 0)` fallback with a zero vector for missing targets, retrain deep models, and compare to both the row-0 fallback and the 92-target subset.

**Implementation:**

1. **Modify deep_trainer.py** to support configurable fallback strategies:
   ```python
   FALLBACK_STRATEGIES = {
       "row0": lambda esm_matrix, _: esm_matrix[0],     # Current behavior
       "zero": lambda esm_matrix, _: np.zeros(esm_matrix.shape[1]),  # Neutral
       "mean": lambda esm_matrix, _: esm_matrix.mean(axis=0),  # Average kinase
   }
   ```

2. **Add `fallback_strategy` parameter** to `_build_esm_fp_loaders()` and `_build_fusion_loaders()`
   ```python
   def _build_esm_fp_loaders(df, split_indices, dataset_version, batch_size,
                              fallback_strategy="row0"):
       ...
       fallback_vec = FALLBACK_STRATEGIES[fallback_strategy](esm_matrix, target_to_row)
       esm_rows = np.array([
           target_to_row.get(t, -1)  # -1 signals missing
           for t in subset["target_chembl_id"].values
       ])
       X_esm = np.where(
           esm_rows[:, None] >= 0,
           esm_matrix[np.maximum(esm_rows, 0)],
           fallback_vec[None, :]
       )
   ```

3. **Add config option** in `esm_fp_mlp.yaml`, `fusion.yaml`:
   ```yaml
   features:
     protein_fallback: "zero"  # or "row0", "mean"
   ```

4. **Retrain 3 deep models × 3 splits × 2 fallback strategies = 18 experiments**
   - Zero-vector: tests neutral missing-target handling
   - Mean-vector: tests whether average kinase is a reasonable prior

5. **Report ablation table:**
   | Fallback | ESM-FP MLP Random | ESM-FP MLP Scaffold | ESM-FP MLP Target |
   |----------|-------------------|--------------------|--------------------|
   | Row 0 (current) | 0.775 | 0.897 | 1.177 |
   | Zero vector | ? | ? | ? |
   | Mean vector | ? | ? | ? |
   | 92-target only | ? | ? | ? |

**Files to modify:**
- `src/kinase_affinity/training/deep_trainer.py` — add fallback parameter
- `configs/esm_fp_mlp.yaml`, `configs/fusion.yaml` — add fallback config
- `scripts/aws_run_fallback_ablation.sh` (NEW)

**Compute:** ~4-8 hours (18 experiments, deep models only)

#### P0-C: Fix Wording

**What:** Replace "uninformative embedding" with accurate description throughout manuscript.

**Suggested replacements:**
- "uninformative" → "shared surrogate embedding (the ESM-2 representation of an arbitrary kinase)"
- "default embedding" → "fallback embedding corresponding to [specific target at row 0]"
- Add footnote identifying which target's embedding is at row 0

**Files to modify:**
- `docs/project_report.md` — Section 12
- Preprint draft (external .docx)

---

## P1: Benchmark Robustness (Gaps #2, #3)

### The Problem
Current CIs reflect only test-set sampling variability for a single trained model on a single partition. This doesn't capture training stochasticity, hyperparameter sensitivity, or partition sensitivity.

### Solution: Multi-seed and multi-partition experiments

#### P1-A: Multi-Seed Training (Gap #2)

**What:** Retrain all neural models with 5 different random seeds. For tree-based models (RF, XGBoost), training is deterministic given the seed, so vary the `random_state` parameter.

**Implementation:**

1. **Add seed parameterization to deep_trainer.py:**
   ```python
   def deep_train_and_evaluate(config_path, split_strategy="random",
                                dataset_version="v1", training_seed=42):
       torch.manual_seed(training_seed)
       np.random.seed(training_seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(training_seed)
       ...
   ```

2. **Add seed parameterization to trainer.py** (baselines):
   ```python
   def train_and_evaluate(config_path, split_strategy="random",
                           dataset_version="v1", training_seed=42):
       # Override random_state in model config
       config["hyperparameters"]["random_state"] = training_seed
       ...
   ```

3. **Training seeds:** `[42, 123, 456, 789, 1024]`

4. **Run 7 models × 3 splits × 5 seeds = 105 experiments**
   - Each saves predictions to `results/predictions/seed{s}/{model}_{split}.npz`

5. **Aggregate results:**
   - Report mean ± SD across seeds for each model-split combination
   - Compute pairwise significance using the seed-averaged metrics
   - If SD across seeds is small relative to between-model differences, the current single-seed results are robust

6. **Create `src/kinase_affinity/evaluation/multi_seed_analysis.py`:**
   ```python
   def aggregate_seed_results(seeds, models, splits):
       """Load predictions from multiple seeds, compute mean ± SD metrics."""
       ...
       return pd.DataFrame with columns:
           model, split, metric, mean, std, seeds
   ```

**Files to create/modify:**
- `src/kinase_affinity/training/deep_trainer.py` — add `training_seed` param
- `src/kinase_affinity/training/trainer.py` — add `training_seed` param
- `src/kinase_affinity/evaluation/multi_seed_analysis.py` (NEW)
- `scripts/aws_run_multi_seed.sh` (NEW)

**Compute:** ~36-48 hours (105 experiments; parallelize across seeds)

#### P1-B: Multi-Partition Robustness (Gap #3)

**What:** Generate 5 random splits and 3 scaffold splits with different seeds. Target split is harder to vary meaningfully (there are only a few natural family groupings), but we can try 2-3 alternative family assignments.

**Implementation:**

1. **Generate alternative splits:**
   ```python
   # Random: seeds [42, 123, 456, 789, 1024]
   # Scaffold: seeds [42, 123, 456]
   # Target: try 2 alternative family groupings
   #   - default: holdout by kinase_group
   #   - alt1: holdout by kinase_family (more granular)
   #   - alt2: random target-level holdout (no family structure)
   ```

2. **Save splits to `data/processed/v1/splits_multi/`:**
   ```
   splits_multi/
   ├── random_seed42.json
   ├── random_seed123.json
   ├── ...
   ├── scaffold_seed42.json
   ├── scaffold_seed123.json
   ├── ...
   └── target_alt1.json
   ```

3. **Run key models on alternative splits:**
   - Not all 105 combinations — focus on RF, MLP, ESM-FP MLP (the 3 most important models)
   - 3 models × 5 random seeds × 1 training seed = 15 random-split experiments
   - 3 models × 3 scaffold seeds × 1 training seed = 9 scaffold experiments
   - 3 models × 3 target variants × 1 training seed = 9 target experiments
   - Total: ~33 additional experiments

4. **Analyze split sensitivity:**
   - How much does RMSE vary across splits? (SD over split seeds)
   - Do model rankings change across partitions?
   - If rankings are stable, the single-partition results are defensible

**Files to create/modify:**
- `src/kinase_affinity/data/splits.py` — add batch split generation
- `src/kinase_affinity/evaluation/multi_split_analysis.py` (NEW)
- `scripts/aws_run_multi_split.sh` (NEW)

**Compute:** ~12-18 hours (33 experiments, subset of models)

#### P1-C: Hierarchical Uncertainty Reporting

**What:** Combine seed-level and split-level variability into a single uncertainty estimate.

**Implementation:**
```python
def hierarchical_uncertainty(seed_results, split_results, bootstrap_results):
    """Combine three sources of uncertainty.

    Total variance ≈ var_split + var_seed + var_bootstrap

    Report as: metric ± sqrt(var_split + var_seed + var_bootstrap)
    """
```

This gives a single honest CI that captures test-sample, training, and partition variability.

**Files to create:**
- `src/kinase_affinity/evaluation/hierarchical_ci.py` (NEW)

**Compute:** Negligible (analysis only)

---

## P2: Strengthening Items (Gaps #4, #5)

### P2-A: Stronger Selectivity Baselines (Gap #4)

**What:** The JAK selectivity comparison is asymmetric — ligand-only pooled models *can't* encode target identity. Add three stronger non-protein baselines.

**Implementation:**

1. **Per-target ligand-only models (strongest non-protein baseline):**
   ```python
   # Train separate RF/MLP for each JAK member
   # At prediction time, predict with all 4 models
   # Compare predictions across targets for each compound

   def train_per_target_models(jak_data, model_class="random_forest"):
       """Train one model per JAK target."""
       models = {}
       for target in ["JAK1", "JAK2", "JAK3", "TYK2"]:
           subset = jak_data[jak_data["gene_symbol"] == target]
           model = train_rf_on_subset(subset)
           models[target] = model
       return models

   def predict_selectivity_per_target(models, test_compounds):
       """Predict pActivity for each compound with each target's model."""
       for compound in test_compounds:
           preds = {t: models[t].predict(compound) for t in models}
           best_target = max(preds, key=preds.get)
           # Compare to true best target
   ```

2. **One-hot target-conditioned baseline:**
   ```python
   # Concatenate Morgan FP (2048) + one-hot target ID (4 for JAK)
   # Train single pooled model with target identity as an explicit feature

   def build_onehot_features(fp_matrix, target_ids, jak_targets):
       """Concatenate fingerprints with one-hot target encoding."""
       onehot = np.zeros((len(target_ids), len(jak_targets)))
       for i, t in enumerate(target_ids):
           j = jak_targets.index(t)
           onehot[i, j] = 1
       return np.concatenate([fp_matrix, onehot], axis=1)
   ```

3. **Pairwise ΔpActivity model:**
   ```python
   # For each compound tested on JAK pairs (A, B):
   #   input = [FP, target_A_indicator, target_B_indicator]
   #   output = pActivity_A - pActivity_B
   # Directly predicts selectivity without absolute affinity

   def build_pairwise_data(jak_data, pairs):
       """Create pairwise selectivity training data."""
       for compound in shared_compounds:
           for t1, t2 in itertools.combinations(jak_targets, 2):
               delta = jak_data.loc[(compound, t1), "pactivity"] - \
                       jak_data.loc[(compound, t2), "pactivity"]
               yield (fp, t1_idx, t2_idx, delta)
   ```

4. **Updated selectivity comparison table:**
   | Model | Type | Top-1 Accuracy | Rank ρ |
   |-------|------|---------------|--------|
   | RF (pooled, ligand-only) | Fingerprint | 51.6% | N/A |
   | RF (per-target) | Fingerprint | ? | ? |
   | RF + one-hot target | Fingerprint + target ID | ? | ? |
   | ΔpActivity RF | Fingerprint + pairwise | ? | ? |
   | ESM-FP MLP | Fingerprint + ESM-2 | 78.5% | 0.781 |
   | Fusion | Graph + ESM-2 | 79.5% | 0.744 |

**Files to create/modify:**
- `scripts/run_phase6_case_study.py` — add selectivity baselines section
- OR create `scripts/run_selectivity_baselines.py` (NEW) for cleaner separation

**Compute:** ~1-2 hours (JAK subset is small)

### P2-B: Supplementary Tables (Gap #5)

**What:** Move critical reproducibility details from the code repository into the manuscript supplement.

**Tables to create:**

1. **Table S1: Target-Family Assignment for Target Split**
   ```
   | Kinase Family | N Targets | N Records | Split Assignment |
   |--------------|-----------|-----------|-----------------|
   | TK (Tyrosine) | 203 | 180,000 | Train |
   | CMGC | 89 | 52,000 | Train |
   | AGC | 63 | 31,000 | Validation |
   | TKL | 42 | 18,000 | Test |
   | ... | | | |
   ```

2. **Table S2: Hyperparameter Search Spaces and Selected Values**
   ```
   | Model | Parameter | Search Space | Selected Value | Selection Criterion |
   |-------|-----------|-------------|---------------|-------------------|
   | ElasticNet | alpha | [0.001, ..., 1.0] | 0.01 | Val RMSE |
   | XGBoost | max_depth | [4, 6, 8, 10, 12] | 8 | Val RMSE |
   | ... | | | | |
   ```

3. **Table S3: Training Budget and Compute**
   ```
   | Model | Train Time (s) | Hardware | Epochs (best) | Early Stop? |
   |-------|---------------|----------|---------------|------------|
   | RF | 87 | CPU (48-core) | N/A | No |
   | ESM-FP MLP | 229 | 1× A10G | 50 | Yes (patience=15) |
   | ... | | | | |
   ```

4. **Table S4: Dataset Composition by Endpoint Type and Split**
   ```
   | Split | Fold | N IC50 | N Ki | N Kd | % IC50 | % Active |
   |-------|------|--------|------|------|--------|----------|
   | Random | Train | X | X | X | X% | X% |
   | Random | Test | X | X | X | X% | X% |
   | ... | | | | | | |
   ```

5. **Table S5: ESM-2 Embedding Coverage by Target Family**
   ```
   | Family | Total Targets | With Embedding | Without | Coverage |
   |--------|--------------|----------------|---------|----------|
   | TK | 203 | 45 | 158 | 22.2% |
   | ... | | | | |
   ```

**Implementation:**
- Create `scripts/generate_supplement_tables.py` (NEW)
- Reads from curated data, split files, tuning results, training logs
- Outputs CSV and LaTeX-formatted tables

**Compute:** Negligible (analysis only)

### P2-C: Endpoint-Stratified Analysis (Checklist item 2.3)

**What:** Run IC50-only subset analysis to show that mixed-endpoint effects don't drive conclusions.

**Implementation:**
```python
def create_endpoint_subset(endpoint="IC50"):
    """Filter curated data to a single endpoint type."""
    df = pd.read_parquet("data/processed/v1/curated_activities.parquet")
    subset = df[df["standard_type"] == endpoint]
    # IC50 is ~77% of data, so this is a large subset
    subset.to_parquet(f"data/processed/v1/curated_activities_{endpoint.lower()}.parquet")
```

Train RF and ESM-FP MLP on IC50-only subset for random + scaffold splits (4 experiments).
Compare to full-dataset results. If rankings are preserved, mixed endpoints aren't driving conclusions.

**Compute:** ~2-3 hours

---

## P3: Aspirational Items (Gaps #6, #7)

### P3-A: Temporal Split (Gap #6)

**What:** Add a time-based split using ChEMBL deposition dates.

**Challenge:** No temporal metadata exists in the curated dataset. Requires re-querying ChEMBL.

**Implementation:**

1. **Modify `src/kinase_affinity/data/fetch.py`** to extract temporal metadata:
   ```python
   # Add to ChEMBL query:
   # - document_year (publication year from the source document)
   # - first_approval (for approved drugs)
   # These are available via the activity API but not currently retained

   # Alternative: use assay_chembl_id to look up assay metadata
   # which includes document_journal, document_year
   ```

2. **Add temporal split to `splits.py`:**
   ```python
   def temporal_split(df, year_col="document_year",
                      train_cutoff=2018, test_cutoff=2020):
       """Time-based split: train on older, test on newer data."""
       train = df[df[year_col] <= train_cutoff].index.values
       val = df[(df[year_col] > train_cutoff) &
                (df[year_col] <= test_cutoff)].index.values
       test = df[df[year_col] > test_cutoff].index.values
       return {"train": train, "val": val, "test": test}
   ```

3. **Run key models on temporal split:**
   - RF, MLP, ESM-FP MLP on temporal split
   - 3 experiments

**Feasibility assessment:** Medium-high effort due to data re-fetching. Could be a separate follow-up study if time is limited.

**Compute:** ~1 hour for re-fetching, ~3 hours for experiments

### P3-B: UQ Normalization (Gap #7)

**What:** Post-hoc calibrate all uncertainty methods to the same nominal coverage, then re-compare.

**Implementation:**
```python
from sklearn.isotonic import IsotonicRegression

def recalibrate_uncertainties(y_true, y_pred, y_std, method="isotonic"):
    """Post-hoc recalibration of prediction intervals."""
    # Compute normalized residuals: z = |y_true - y_pred| / y_std
    z = np.abs(y_true - y_pred) / np.maximum(y_std, 1e-6)
    # Fit isotonic regression: expected coverage → observed coverage
    ...
    # Return recalibrated uncertainty
```

Apply recalibration on validation set, evaluate on test set. Report both raw and recalibrated miscalibration areas.

**Compute:** Negligible (analysis only)

---

## Execution Plan

### Phase 1: Infrastructure (Days 1-2, local)

| Task | Time | Dependencies |
|------|------|-------------|
| Create `subset.py` for 92-target filtering | 2h | None |
| Add fallback strategies to `deep_trainer.py` | 2h | None |
| Add `training_seed` parameter to both trainers | 2h | None |
| Generate multi-seed splits | 1h | None |
| Create AWS run scripts | 2h | Above modules |
| Create supplement table generation script | 3h | None |
| Add selectivity baselines to Phase 6 script | 3h | None |

### Phase 2: Compute (Days 3-7, AWS)

| Batch | Experiments | Est. Time | Priority |
|-------|------------|-----------|----------|
| **Batch A:** 92-target benchmark | 21 experiments | 6-12h | P0 |
| **Batch B:** Fallback ablation | 18 experiments | 4-8h | P0 |
| **Batch C:** Multi-seed (5 seeds × 21) | 105 experiments | 36-48h | P1 |
| **Batch D:** Multi-partition | 33 experiments | 12-18h | P1 |
| **Batch E:** Selectivity baselines | ~6 experiments | 1-2h | P2 |
| **Batch F:** IC50-only subset | 4 experiments | 2-3h | P2 |

**Parallelization strategy:**
- Batches A + B can run simultaneously (different data subsets)
- Batch C can be parallelized across seeds (5 parallel jobs)
- Batch E + F are small and fast

### Phase 3: Analysis (Days 8-9, local)

| Task | Time |
|------|------|
| Aggregate multi-seed results, compute SD | 2h |
| Aggregate multi-partition results | 2h |
| Compute hierarchical CIs | 2h |
| Generate supplement tables | 2h |
| Compare 92-target vs 507-target results | 2h |
| Compare fallback strategies | 1h |
| Analyze selectivity baselines | 1h |
| Run bootstrap on new results | 3h |

### Phase 4: Writing (Days 10-12)

| Task | Time |
|------|------|
| Rewrite Section 12 (ESM-2 coverage) with ablation results | 3h |
| Add multi-seed/multi-partition results to Section 11 | 2h |
| Update Section 10 (JAK selectivity) with new baselines | 2h |
| Create supplementary methods section | 4h |
| Update conclusions with new findings | 2h |
| Revise abstract with updated claims | 1h |

---

## Expected Outcomes

### Best Case
- 92-target subset shows ESM-FP MLP advantage *increases* (e.g., 10% improvement over RF instead of 5.3%) → proves embeddings help when they're real
- Multi-seed SD is small relative to between-model differences → current rankings are robust
- Per-target selectivity models reach ~65% but ESM-FP MLP still wins at 79% → protein embeddings provide genuine value beyond target identity

### Worst Case
- 92-target subset shows ESM-FP MLP advantage *disappears* → embeddings don't help at all
- Multi-seed SD is large → some model rankings are unstable
- Per-target or one-hot baseline matches ESM-FP MLP selectivity → ESM-2 not needed

### Either Way
- The paper becomes methodologically much stronger
- Claims become precisely scoped to what the data supports
- Reviewers can't dismiss results on methodological grounds

---

## Decision Points

After Phase 2 results are in, we need to decide:

1. **Does the 92-target subset become the primary benchmark?**
   - If the 92-target protein-aware results are substantially different from 507-target results → yes, 92-target should be primary
   - If they're similar → report both, with 507-target as main and 92-target as validation

2. **How to report multi-seed variability?**
   - If SD < 0.01 RMSE → brief mention that results are stable across seeds
   - If SD > 0.01 RMSE → full mean ± SD reporting, adjust claims accordingly

3. **What happens to the selectivity narrative?**
   - If per-target models reach 70%+ → tone down ESM advantage, emphasize that *any* target encoding helps
   - If per-target models stay at ~55% → ESM-2 sequence information genuinely outperforms simple target identity

---

## Files to Create (Summary)

| File | Purpose |
|------|---------|
| `src/kinase_affinity/data/subset.py` | Create 92-target clean subset |
| `src/kinase_affinity/evaluation/multi_seed_analysis.py` | Aggregate across training seeds |
| `src/kinase_affinity/evaluation/multi_split_analysis.py` | Aggregate across split partitions |
| `src/kinase_affinity/evaluation/hierarchical_ci.py` | Combined uncertainty estimates |
| `scripts/aws_run_esm92.sh` | AWS script for 92-target benchmark |
| `scripts/aws_run_fallback_ablation.sh` | AWS script for fallback ablation |
| `scripts/aws_run_multi_seed.sh` | AWS script for multi-seed training |
| `scripts/aws_run_multi_split.sh` | AWS script for multi-partition |
| `scripts/run_selectivity_baselines.py` | Stronger selectivity comparisons |
| `scripts/generate_supplement_tables.py` | Supplement table generation |
| `configs/dataset_v1_esm92.yaml` | Config for 92-target subset |

## Files to Modify (Summary)

| File | Change |
|------|--------|
| `src/kinase_affinity/training/deep_trainer.py` | Add `training_seed`, `fallback_strategy` |
| `src/kinase_affinity/training/trainer.py` | Add `training_seed` |
| `src/kinase_affinity/data/splits.py` | Add batch split generation |
| `configs/esm_fp_mlp.yaml` | Add `protein_fallback` option |
| `configs/fusion.yaml` | Add `protein_fallback` option |
| `docs/project_report.md` | Update Sections 10, 11, 12, 13 |
