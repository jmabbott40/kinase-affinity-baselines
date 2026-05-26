# IMM1 Macrocyclic Peptide Molecular Glue — Baseline ML Benchmark

**Project:** `imm1-glue-baselines` (private repo at https://github.com/jmabbott40/imm1-glue-baselines)
**Author:** Joshua Abbott
**Date:** 2026-05-26
**Status:** Spec — pending review

## Confidentiality

This spec describes methodology only. It contains no compound SMILES, no individual KD values, and no compound identifiers beyond the generic `RAP-XXXX` series-name pattern. The raw dataset (`IMM1_SPR_Data.csv`, n≈288 rows) is **confidential** and stays local to the analyst's machine — never committed to any repo, never uploaded to cloud storage, never sent through public services. The implementation repo is private. See the data-handling policy in Section 1.

## 1. Motivation and Scope

### Question

How do standard ML baselines (Random Forest, XGBoost, MLP) perform when predicting binding affinity for a small (~288-compound) library of macrocyclic-peptide molecular glues targeting a single protein, given SPR-derived pKD measurements?

### Why this is worth doing

The existing kinase benchmarking work (`target-affinity-ml`, `kinase-affinity-baselines`, `gpcr-aminergic-benchmarks`) characterized model performance on **large public chemogenomic datasets** spanning many targets and tens to hundreds of thousands of compounds. This project addresses a different regime that is much closer to real drug-discovery program data:

- **Small n** (~288 vs ~200k).
- **Single target** (no target-stratified splits possible).
- **Macrocyclic peptide chemistry** rather than drug-like small molecules.
- **Molecular glue modality** — stabilizing a protein–protein interaction rather than blocking a binding pocket. SAR landscapes tend to be steeper and more idiosyncratic than for typical inhibitors.
- **SPR-style left-censoring** at the assay's KD floor.

The question is empirical and not assumed to have a positive answer — "the baselines all fail" or "they're indistinguishable from predicting the mean" are valid findings.

### Out of scope

- Protein-side features (ESM-2 embeddings, structure-aware features). Only one target; protein info adds no per-row signal.
- GNN baselines, generative or active-learning loops, ensemble/stacking, Bayesian hyperparameter optimization, censored-aware loss functions (Tobit, AFT).
- Cross-target or cross-program generalization tests.
- Public release of weights, predictions, or data.

## 2. Dataset

### Source

Internal SPR campaign on a single target (referred to as IMM1 throughout). File supplied as `IMM1_SPR_Data.csv` with columns `[Compound Name, SMILES, pKD]`.

### Composition (high-level only)

- ~288 rows; ~277 unique compounds after de-duplication (9 compound names with 2-3 replicates).
- pKD dynamic range ~4.0 to ~10.0; mean ≈ 6.2, std ≈ 1.6.
- ~57 rows (20%) at pKD = 4.0, confirmed to represent **left-censoring at the assay floor** (compounds with KD too weak to fit, recorded as the assay's lower limit).
- Some replicate groups contain mixed binder + censored measurements (e.g., 7.5 and 4.0 in the same group), indicating either real measurement noise or batch-to-batch variation.

### Curation rules (applied in `data/curate.py`)

1. **Replicate aggregation:** for each unique `Compound Name`, take the mean pKD across replicate rows.
2. **Noisy flag:** mark a compound `is_noisy=True` if the per-compound pKD std > 1.0. Inclusion-wise, noisy compounds remain in the dataset; the flag is for downstream error analysis only.
3. **Censoring flag:** mark `is_censored=True` if `pkd_mean ≤ 4.0 + 1e-6`.
4. **SMILES canonicalization:** via RDKit `MolToSmiles(MolFromSmiles(...))`. Compounds whose SMILES cannot be parsed are dropped, with their compound IDs logged to the curation report.
5. **Single row per compound** after curation.

A sanitized curation report (`results/curation_report.md` — counts only, no IDs/SMILES/pKDs) is committed to the repo and gates downstream work.

## 3. Modeling Decisions

### Featurization

- Morgan fingerprints, **radius = 3, n_bits = 4096**.
- Rationale: macrocyclic peptides are large (often 1000+ Da, 70+ heavy atoms); ECFP4 (radius=2) under-captures ring-spanning context. ECFP6 + 4096 bits is a literature-standard choice for peptide QSAR and limits bit collisions vs the 2048-bit default.
- Implementation reuses `target_affinity_ml.features.fingerprints.smiles_to_morgan_fp`.

### Censoring handling

- **Primary:** treat pKD = 4.0 as an exact value. All 277 curated compounds go into training and evaluation.
- **Sensitivity (parallel re-run):** repeat the full benchmark with the ~57 censored compounds removed before splitting. The sensitivity result is a **secondary deliverable, not a primary metric**.
- This avoids the engineering cost of censoring-aware losses (Tobit/AFT), which would require per-model machinery and break apples-to-apples comparison across RF/XGB/MLP.

### Models

Three baselines, all from `target_affinity_ml.models`:
- **Random Forest** — uncertainty = std across tree predictions.
- **XGBoost** — uncertainty via the existing residual-quantile estimator (audit confirms or gap-fills).
- **MLP** — MC dropout, 30 forward passes.

All expose a uniform interface: `fit(X, y)`, `predict(X) → ŷ`, `predict_with_uncertainty(X) → (ŷ, σ̂)`.

### Hyperparameter grids

| Model | Grid size | Key axes |
|---|---|---|
| RF | 81 | `n_estimators` × `max_depth` × `min_samples_leaf` × `max_features` |
| XGB | 108 | `n_estimators` × `max_depth` × `lr` × `subsample` × `colsample_bytree` |
| MLP | 24 | `hidden` × `dropout` × `lr` × `weight_decay` |

Grids are sized to keep total nested-CV fits under ~30k.

### Splits (four strategies, all evaluated)

1. **Random** — 5-fold CV stratified on pKD quartiles. Five seeds: {42, 123, 456, 789, 1000}.
2. **Murcko scaffold** — 113 unique scaffolds across 277 compounds; 67 singletons. Scaffold groups stay together; singletons randomly distributed per seed.
3. **Butina cluster** — Tanimoto on ECFP4 (radius=2, 2048-bit). Cutoff selected empirically from a pre-benchmark diagnostic sweep at {0.4, 0.5, 0.6, 0.7, 0.8} that requires ≥10 clusters of ≥5 compounds. If no cutoff qualifies, cluster split is dropped and the project documents the omission.
4. **Time / synthesis-order** — sort by `Compound Name` lex-order (RAP-XXXX format is zero-padded so lex order = numeric order = synthesis order, confirmed by the analyst). Hold out the most recent 20% as test per fold. Sanity-checked by a diagnostic that confirms monotonic ID progression.

All four strategies produce identical fold assignments across models (`results/splits/{strategy}.npy`) so within-fold model comparisons are paired.

### Evaluation protocol: nested 5×5 CV

- **Outer 5-fold CV** for evaluation. Each fold uses the strategy-specific assignment above.
- **Inner 5-fold CV** on each outer-train for hyperparameter selection.
- **Five seeds** rerun the entire procedure for variance estimation.
- Final reported metric per (model, split) cell = mean ± bootstrap 95% CI across all seeds × outer folds.

### Metrics

- **Regression:** R², RMSE, MAE, Spearman ρ, Pearson r.
- **Classification (multi-threshold):** AUROC, AUPRC, balanced accuracy, MCC, F1 — reported at thresholds pKD ≥ 6.0, ≥ 7.0, and > 4.0. Multi-threshold reporting reflects the program reality that "active" shifts as a drug-discovery program matures.
- **Uncertainty calibration:** reliability diagrams + ECE on binarized predictions; Spearman correlation between σ̂ and |residual| for regression.

## 4. Architecture

### Repo

- Private GitHub repo `imm1-glue-baselines`.
- Imports `target-affinity-ml` (existing library; pinned version) as the source of truth for shared infrastructure: FP generation, model wrappers, nested-CV utilities, multi-seed analysis, calibration plots.
- IMM1-specific code is limited to: data loader (CSV with the project schema), curation, censoring sensitivity, Butina cluster split, time split, and report-generation scripts.

### Data location and security

- Raw `IMM1_SPR_Data.csv` lives at `$IMM1_DATA_PATH` (env var, default `~/secure_data/imm1/IMM1_SPR_Data.csv`). Never committed.
- `.gitignore` blocks `data/raw/`, `data/processed/`, `results/`, `notebooks/.ipynb_checkpoints/`, and `*.csv`.
- Pre-commit hook `scripts/check_no_data_leak.sh` greps staged files for `RAP-\d{7}` and `pkd[\s=:]*\d+\.\d{4,}` patterns and refuses matching commits.

### Compute

- **Local Mac only.** Estimated wall time for the full sweep: ~5–8 hours overnight on a modern 8-core Mac with `n_jobs=-1`.
- Cloud compute (EC2) was considered and rejected: the data sensitivity (uploading confidential SMILES + KDs to S3/EC2 introduces attack surface — IAM, transit encryption, instance snapshots) outweighs the modest compute speedup at this scale.
- Practical run command: `caffeinate -i nice -n 10 python scripts/run_benchmark.py`.

### Directory layout

```
imm1-glue-baselines/
├── README.md                          # internal-only
├── CONFIDENTIAL.md                    # data-handling policy
├── pyproject.toml
├── environment.yml
├── configs/
│   ├── dataset_imm1.yaml
│   ├── splits.yaml
│   ├── rf_baseline.yaml
│   ├── xgb_baseline.yaml
│   └── mlp_baseline.yaml
├── data/
│   └── README.md                      # how to point at $IMM1_DATA_PATH
├── src/imm1_glue/
│   ├── data/{load,curate,splits}.py
│   ├── features/                      # re-export
│   ├── models/                        # re-export
│   ├── evaluation/censoring_sensitivity.py
│   └── reports/generate_tables.py
├── scripts/
│   ├── check_no_data_leak.sh
│   ├── audit_library.py
│   ├── run_diagnostics.py
│   └── run_benchmark.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_results_summary.ipynb
│   └── 03_error_analysis.ipynb
├── tests/
│   ├── test_curate.py
│   ├── test_splits.py
│   └── test_pipeline_smoke.py
└── results/                           # gitignored
    ├── splits/
    ├── splits_diag/
    ├── predictions/
    ├── tables/
    ├── figures/
    └── logs/
```

## 5. Components & Dataflow

### Dataflow

```
~/secure_data/imm1/IMM1_SPR_Data.csv  (288 rows)
    │
    ▼  data/load.py            (schema-validated DataFrame)
    │
    ▼  data/curate.py          (277 unique compounds; replicate-aggregated; flagged)
    │
    ▼  target_affinity_ml.features.fingerprints   (X: 277 × 4096)
    │
    ▼  data/splits.py          (4 strategies × 5 seeds → fold assignments)
    │
    ▼  scripts/run_benchmark.py  (nested 5×5 CV; idempotent / resumable)
    │
    ▼  evaluation/  +  reports/generate_tables.py
    │
    ▼  Publishable tables + diagnostic figures
```

### Component contracts (summary)

- `data/load.py` — strict schema validation; raises on missing env var, missing file, schema mismatch, NaN pKD.
- `data/curate.py` — pure function from loader output → curated DataFrame with the columns listed in Section 2. Invalid SMILES → log + drop.
- `data/splits.py` — produces `(n_compounds, n_seeds)` fold-ID arrays for each strategy. Random + Murcko delegate to library where present; Butina and time-split are IMM1-specific (Butina uses `rdkit.ML.Cluster.Butina`).
- `scripts/run_benchmark.py` — per `(model, split, seed, outer_fold)` writes `results/predictions/{...}.parquet`; existing files skipped → resumable. Per-call manifest in `results/run_manifest.json`.
- `evaluation/censoring_sensitivity.py` — re-orchestrates the full sweep with censored compounds removed before splitting; outputs separate tables.
- `reports/generate_tables.py` — consumes `results/predictions/` and produces CSV + Markdown tables.

### Pre-benchmark diagnostics

Three diagnostics run **before** the main sweep, committed to the repo (numbers and counts only — no IDs/SMILES/KDs):

1. **Curation report** — n in/out, n duplicates aggregated, n flagged noisy, n censored.
2. **Butina cutoff sweep** — for each cutoff in {0.4, 0.5, 0.6, 0.7, 0.8}, report cluster-count statistics; choose cutoff or drop the cluster split.
3. **Time-split sanity check** — verify lex-sort of compound IDs produces a monotonic sequence spanning the full dataset.

## 6. Error Handling

Principle: **fail loudly at boundaries; never silently downstream.**

| Failure | Where caught | Behavior |
|---|---|---|
| Missing env var / data file | `data/load.py` | Raise `FileNotFoundError` |
| Schema mismatch | `data/load.py` | Raise `ValueError` |
| Non-numeric pKD | `data/load.py` | Raise (`pd.to_numeric(errors="raise")`) |
| Invalid SMILES | `data/curate.py` | Log compound ID, drop row, continue |
| Empty replicate group after drop | `data/curate.py` | Drop compound; record in curation report |
| Butina yields no usable cutoff | `scripts/run_diagnostics.py` | Skip cluster split; record in summary |
| Time split puts all censored in one fold | `data/splits.py` | Warn; proceed |
| Inner-CV all-equal hyperparameter scores | `scripts/run_benchmark.py` | Pick alphabetically first; warn |
| Model lacks `predict_with_uncertainty` | `scripts/run_benchmark.py` | σ̂ recorded as NaN; continue; final report lists affected cells |
| Interrupted run | filesystem | Already-written parquet survives; resumable |
| Output dir unwritable | startup | Raise immediately |

Never silently swallowed: NaN/Inf in fingerprints, predictions outside `[0, 14]`, train/test compound-ID overlap (asserted in `run_benchmark.py`).

## 7. Testing Strategy

| Layer | Verifies | Implementation |
|---|---|---|
| Unit — curation | Replicate aggregation, censoring detection, noisy flagging, invalid-SMILES drop | Synthetic 10-row CSVs covering each branch |
| Unit — splits | Non-overlap of train/test compound IDs; fold size tolerance; scaffold/cluster group integrity | Synthetic 20-compound fixture |
| Unit — metrics | IMM1-specific multi-threshold classification assertions only | Crafted predictions/labels |
| Integration smoke | End-to-end run on 30-compound synthetic dataset, one model × one split × one seed × one fold | `tests/test_pipeline_smoke.py`, <30s |
| Library audit | `target_affinity_ml` exposes expected API surfaces | `scripts/audit_library.py`; produces checklist report; gates further work |
| Pre-commit hook | Refuses commits containing compound IDs or pKD-format floats | Bash hook; tested against a planted-leak fixture |

Not tested: model numerical correctness on real IMM1 data (that's what the benchmark measures); library internals (owned upstream); bitwise reproducibility across hardware.

## 8. Deliverables

| Artifact | Format | Location |
|---|---|---|
| Primary results table | CSV + MD | `results/tables/primary_metrics.{csv,md}` |
| Sensitivity (drop-censored) table | CSV + MD | `results/tables/sensitivity_metrics.{csv,md}` |
| Per-seed detailed metrics | CSV | `results/tables/per_seed_metrics.csv` |
| Multi-threshold classification table | CSV + MD | `results/tables/classification_thresholds.{csv,md}` |
| Uncertainty calibration plots | PNG | `results/figures/calibration_{model}_{split}.png` |
| Error-analysis notebook | `.ipynb` | `notebooks/03_error_analysis.ipynb` |
| Curation report (sanitized) | MD | `results/curation_report.md` |
| Split diagnostic reports | CSV + MD | `results/splits_diag/` |
| Library audit report | MD | `results/library_audit.md` |

Reproducibility: every invocation of `run_benchmark.py` writes a `run_manifest.json` capturing seeds, FP config, hyperparameter selections, and library version pins.

## 9. Implementation Phases (informational)

The full implementation plan is produced by the writing-plans skill as the next step. Rough phase ordering:

- **Phase 0** — Repo bootstrap, pre-commit hook, library audit.
- **Phase 1** — Data pipeline (`load.py`, `curate.py`) + curation report + unit tests.
- **Phase 2** — Splits + diagnostic sweep + unit tests.
- **Phase 3** — Benchmark orchestrator + smoke test + idempotency.
- **Phase 4** — Full sweep (overnight compute) + drop-censored sensitivity sweep.
- **Phase 5** — Reports, calibration plots, error-analysis notebook.
- **Phase 6** — Review and iteration.

Estimated effort: ~7–9 working days; one overnight compute run.

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Library API gap forces in-repo duplication | Med | Low | Audit first; gap-fill PRs to `target-affinity-ml` |
| Butina yields no useful cutoff | Med | Low | Drop cluster split; document |
| RAP-XXXX IDs are not actually date-ordered | Low | Low | Diagnostic catches; drop time split; document |
| MLP fails catastrophically across splits | Med | Med | Expand grid; check feature scaling; may be a genuine finding |
| Primary vs sensitivity results materially disagree | Med | Med | This is a result, not a bug — report both with discussion |
| Hyperparameter grid times out | Low | Med | Switch to random search subset |
| Hidden replicate-induced split leakage | Low | High | Aggregation precedes split; asserted in test |

## 11. Open Questions (deferred to implementation)

- Time-split validity confirmed in principle; verify monotonicity at runtime.
- Library API completeness verified by the audit script; any gap becomes a gap-fill PR upstream.
- Butina cutoff chosen empirically by diagnostic; spec does not commit to a value.
- MLP architecture defaults from the kinase work were tuned for n≈200k. If MLP underperforms systematically here, revisit the grid (narrower networks, stronger dropout).

## 12. Acceptance Criteria

Implementation phase is complete when:

1. `scripts/audit_library.py` reports all required APIs present (or gap-fill PRs merged).
2. `scripts/run_diagnostics.py` produces all three diagnostic reports.
3. `pytest tests/` passes.
4. `scripts/run_benchmark.py --config configs/dataset_imm1.yaml` completes a full sweep.
5. The primary results table is reviewed by the analyst and either accepted or flagged for re-run.

---

**Next step after spec approval:** invoke the `superpowers:writing-plans` skill to translate this spec into an ordered, step-by-step implementation plan with verification checkpoints.
