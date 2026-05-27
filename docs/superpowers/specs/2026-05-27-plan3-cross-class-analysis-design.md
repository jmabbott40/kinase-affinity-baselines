# Plan 3 — Cross-Class Analysis Design

**Date:** 2026-05-27
**Author:** Joshua Abbott (jmabbott40)
**Status:** Design — approved, pre-implementation
**Project:** Multi-class expansion of cheminformatics ML benchmark (Phase 1)
**Predecessor:** Plan 2 (GPCR data pipeline + benchmark) — complete
**Spec foundation:** [`2026-04-17-gpcr-aminergic-phase1-design.md`](2026-04-17-gpcr-aminergic-phase1-design.md), Sections 5 + 6

---

## Executive summary

Plan 3 is the **methodology and cross-class analysis phase** of the GPCR aminergic project. It computes scaffold-diversity metrics + Prabakaran-Bromberg Residue Neighborhood Significance (RNS) scores for both classes (kinase + GPCR), runs the four pre-registered hypothesis tests (H1-H4) on the combined data, generates the four main-text tables and five main-text figures, and ends at "results assembled and visualized." Manuscript drafting is **out of scope** (deferred to a separate Plan 4 if pursued).

**Estimated effort:** ~17-20 working days (~3.5 weeks). **Compute:** <1 day total on AWS (~3-5 hours wall-clock for parallelized MSA generation + RNS scoring on the 96-CPU instance). GPUs available on the AWS instance but only used if kinase reference data is missing per-run predictions and we need to re-run deep models.

**Foundational decisions (locked in during brainstorming):**

| # | Decision | Rationale |
|---|---|---|
| 1 | **Full Prabakaran-Bromberg RNS, with conservation-entropy fallback as sensitivity analysis** | Strongest scientific framing + robustness check; the sensitivity analysis tests whether the cross-class ESM-2-advantage correlation holds under both metrics |
| 2 | **End at results assembly; manuscript out of scope** | Engineering-shaped Plan 3; manuscript writing is different cognitive work, deferred |
| 3 | **Library `benchmarks/` module + GPCR repo notebooks/analyses** | Methodology (reusable) in library; application pipeline (specific to this paper) in GPCR repo. Matches spec Section 3.4 |
| 4 | **Kinase reference data committed into GPCR repo under `data/kinase_reference/`** | Simplest fully-reproducible path; resolves Plan 1 limitation L2; Zenodo deposit deferred to manuscript phase |
| 5 | **RNS-first execution structure** | Front-loads the spec's R2 risk (RNS implementation cold-start); validation gate in Week 1 enables clean pivot to fallback if needed |

---

## 1. Scope

### 1.1 In scope

- Library `target-affinity-ml` v1.2.0: new `benchmarks/` module with `scaffold_diversity.py`, `rns_scoring.py`, `hypothesis_tests.py`. Also fixes the v1.1.0 stale `__version__` constant.
- GPCR application repo extension: `notebooks/05_scaffold_diversity.ipynb`, `06_rns_analysis.ipynb`, `07_cross_class_comparison.ipynb` plus testable `src/gpcr_aminergic_benchmarks/analyses/` modules.
- Cross-class hypothesis tests: formal H1-H4 with effect sizes, bootstrap CIs, Bonferroni-corrected p-values.
- Two new methodological regressions: scaffold-diversity correlation (random→scaffold and scaffold→target degradation regressed on per-target scaffold metrics) and RNS-stratified ESM-2 analysis (per-target ESM-FP-MLP-minus-MLP advantage regressed on per-target RNS).
- Structure-source three-tier handling per spec Section 5.4: provenance annotation + pLDDT weighting + PDB-vs-AlphaFold sensitivity analysis.
- Plan 1 limitation L2 resolution: kinase reference data hosting.
- Plan 3 completion summary + Plan 4 (manuscript) handoff document.

### 1.2 Out of scope (deferred)

- Manuscript drafting (Plan 4, if pursued).
- Phase 2-N target classes (spec Section 2.3 — proteases, nuclear receptors, ion channels).
- OSF / AsPredicted pre-registration deposit (spec Section 6.5 — in-paper pre-registration sufficient for bioRxiv).
- Zenodo deposit of kinase reference data (deferred to manuscript phase).
- Library v1.1.1 patch — bundled instead into v1.2.0.

---

## 2. Architecture

### 2.1 Module split

```
target-affinity-ml/
└─ src/target_affinity_ml/benchmarks/      # NEW in v1.2.0 (was empty scaffold)
   ├─ __init__.py
   ├─ scaffold_diversity.py                # Per-target metrics + class aggregates + regressions
   ├─ rns_scoring.py                       # Prabakaran-Bromberg RNS + conservation-entropy fallback
   ├─ hypothesis_tests.py                  # H1-H4 + between-class machinery
   └─ _rns_reference_data.json             # Bundled reference values for the validation gate

gpcr-aminergic-benchmarks/
├─ data/kinase_reference/                  # NEW — Plan 1 limitation L2 resolved
│  ├─ features/morgan_fp.npz
│  ├─ features/rdkit_descriptors.npz
│  ├─ features/esm2_embeddings.npz         # if available; otherwise documented as recompute-needed
│  ├─ features/smiles_index.json
│  ├─ benchmark_v1/all_seeds_metrics.csv
│  ├─ benchmark_v1/multi_seed_aggregated.csv
│  └─ benchmark_v1/predictions_seed*/      # per-target predictions for per-target metric correlation
├─ data/structures/                        # NEW — fetched, cached, gitignored (size)
│  ├─ pdb/{uniprot}.pdb
│  └─ alphafold/{uniprot}.pdb
├─ data/msas/                              # NEW — jackhmmer outputs, gitignored
│  └─ {uniprot}.sto
├─ src/gpcr_aminergic_benchmarks/analyses/  # NEW
│  ├─ scaffold_diversity.py                # GPCR + kinase application of library benchmarks
│  ├─ rns_analysis.py                      # ditto for RNS
│  └─ cross_class.py                       # combines + runs hypothesis tests
├─ notebooks/                              # NEW
│  ├─ 05_scaffold_diversity.ipynb
│  ├─ 06_rns_analysis.ipynb
│  └─ 07_cross_class_comparison.ipynb
└─ results/
   ├─ tables/                              # 4 main-text tables
   ├─ figures/                             # 5 main-text figures
   └─ supplement/                          # per-target metrics, sensitivity analyses, structure provenance
```

**Architectural rule:** code reusable for Phase 2-N goes in the library. Code specific to kinase-vs-GPCR comparison goes in the GPCR repo. When in doubt, default to the library — it's easier to move from library to app repo than vice versa.

### 2.2 Compute placement

| Workload | Location | Hardware |
|---|---|---|
| RNS module development + unit tests | Local | macOS laptop |
| Reference protein validation gate | Local first, then AWS | CPU (small, ~5-10 proteins) |
| Structure acquisition (PDB + AlphaFold downloads) | AWS | ~5 min CPU; HTTP downloads, threaded |
| Binding-site annotation (KLIFS / GPCRdb API calls) | AWS | ~10-30 min CPU; HTTP, threaded |
| MSA generation (jackhmmer vs UniRef50) | AWS | **96 CPUs parallel, ~1-3 hours wall-clock** |
| Per-residue RNS scoring (543 targets) | AWS | ~30-60 min CPU (after MSAs cached) |
| Scaffold-diversity metrics | Local or AWS | CPU, fast |
| Hypothesis tests | Local | CPU, fast |
| Notebook rendering + figure assembly | Local | CPU |
| Completion summary | Local | — |
| **Deep-model re-runs (only if kinase predictions missing)** | AWS | **GPU; 4× A10G** |

**Rule of thumb: everything heavy runs on AWS.** GPUs are available if needed (deep-model re-runs); the 96 CPUs handle everything else parallelizably.

### 2.3 External services & rate-limit handling

| Service | What we fetch | Rate-limit handling |
|---|---|---|
| **AlphaFold DB** (`alphafold.ebi.ac.uk`) | Pre-computed structures by UniProt accession (URL pattern `files/AF-{acc}-F1-model_v4.pdb`) | Static-file CDN, no quota. 8 concurrent downloads via `ThreadPoolExecutor`; exponential backoff on 5xx; on-disk cache by accession; idempotent re-runs |
| **PDB** (RCSB) | Experimental structures | Biopython's `Bio.PDB.PDBList` with `download_pdb_files`; concurrency 4; same caching |
| **KLIFS REST API** | Kinase ATP-pocket residue annotation | One request per target; cache JSON response by ChEMBL ID; exponential backoff on 429/5xx |
| **GPCRdb REST API** | GPCR orthosteric-pocket residue annotation | Same pattern as KLIFS |
| **UniProt** | Protein sequences (already done in Plan 2 for GPCR; needs to run for kinase reference set) | Reuse Plan 2's `_fetch_sequence_fasta` defensive pattern |
| **UniRef50** (for MSA target database) | Sequence database for jackhmmer searches | Download once (~30 GB) to AWS local volume; one-time cost |

**Critical note:** AlphaFold has TWO APIs and we are NOT using the prediction service (which has daily limits). We are using the *Database* — static file hosting for 200M+ precomputed structures, no rate quotas. Every protein in our 543-target set is human and has a precomputed AlphaFold entry.

---

## 3. RNS module design

The RNS module is the centerpiece and the only piece with substantial implementation risk. See spec Section 5 for the underlying science.

### 3.1 Public API surface

| Function | Inputs | Outputs |
|---|---|---|
| `fetch_structure(uniprot_id, prefer="pdb")` | UniProt accession | `(Bio.PDB.Structure, provenance_dict)` — provenance: `source` ∈ {`PDB`, `AlphaFold`}, `pdb_id`, `pdb_resolution`, `binding_site_pLDDT_mean`, `binding_site_pLDDT_min`, `conformational_state` |
| `fetch_binding_site(target_id, class_name)` | ChEMBL target ID + class hint | List of residue indices into protein sequence. Routes to `_klifs_binding_site` (kinase, 85 residues) or `_gpcrdb_binding_site` (gpcr, ~25-40 residues) |
| `compute_msa(uniprot_id, db="uniref50", out_dir)` | UniProt accession | MSA file (Stockholm format); caches by accession to `data/msas/{uniprot}.sto` |
| `compute_per_residue_rns(structure, binding_site, msa, blosum=BLOSUM62)` | Structure + binding-site residues + MSA | `{residue_index: rns_score_in_[0,1]}` for each binding-site residue |
| `aggregate_target_rns(per_residue, provenance, use_plddt_weighting=True)` | Per-residue + provenance | Single float — mean RNS, pLDDT-weighted for AlphaFold sources per spec 5.4 Tier 2 |
| `compute_conservation_entropy(binding_site, msa)` | Binding-site + MSA | Single float — fallback metric (Shannon entropy of binding-site columns) |
| `validation_gate(reference_set="prabakaran_bromberg", tolerance=0.05)` | (reads bundled reference values) | `(passed: bool, deviations: dict, summary_csv: str)` |

### 3.2 Validation gate go/no-go

**Reference set:** 5-10 proteins from the Prabakaran-Bromberg paper with published RNS values. Bundled into `target_affinity_ml/benchmarks/_rns_reference_data.json` (~5 KB). Concrete proteins to be selected during Phase 1 from the paper's reported reference data.

**Go/no-go criterion (pre-specified):** **Spearman ρ ≥ 0.7 across reference set** between our RNS values and published values, OR **mean absolute deviation ≤ 10%** of published values. Either criterion satisfies the gate.

**On failure:**
- **First failure** → 1-2 day debugging cycle. Common causes: wrong spatial neighbor distance threshold, wrong sequence neighbor window, BLOSUM matrix variant, insertion/deletion handling at binding-site columns.
- **Repeated failure after debugging** → pivot: conservation-entropy becomes primary metric, RNS code marked experimental, plan document updated to reflect the change. Subsequent Plan 3 work proceeds on schedule.

**Rationale for picking Spearman OR MAD (not AND):** the Prabakaran-Bromberg paper may not publish absolute per-residue values for every reference protein. Spearman handles the case where only relative rankings are published; MAD handles the case where absolute values are available. Either is sufficient evidence of reproduction.

### 3.3 Structure source handling (spec 5.4)

All three tiers from the spec:

- **Tier 1 (always):** per-target provenance row written to `results/supplement/structure_provenance.csv`. Columns: `target_chembl_id`, `uniprot_id`, `class`, `structure_source`, `pdb_id`, `pdb_resolution`, `binding_site_pLDDT_mean`, `binding_site_pLDDT_min`, `conformational_state` (GPCR only — `active`/`inactive`/`unknown`).
- **Tier 2 (always):** pLDDT-weighted RNS aggregation for AlphaFold structures: `target_RNS = Σ(residue_RNS × max(0, (pLDDT-50)/50)) / Σ(max(0, (pLDDT-50)/50))`. Experimental PDB structures use uniform weights.
- **Tier 3 (sensitivity analysis):** for the subset where both PDB and AlphaFold structures exist (~25/36 aminergic, ~150-200/507 kinases), compute RNS from each, report cross-source Pearson correlation. Decision tree:
  - r > 0.85 → primary = combined + pLDDT weighting; supplementary = PDB-only
  - 0.7 ≤ r < 0.85 → primary = PDB-only (more conservative); supplementary = combined with AF-bias caveat
  - r < 0.7 → primary = PDB-only; AF-only targets excluded from RNS analyses, listed in supplement

### 3.4 Compute budget

- MSA generation: ~10-30 min CPU per target sequentially. **Parallelized across 96 cores on AWS → ~1-3 hours wall-clock.** Cached to `data/msas/{uniprot}.sto`.
- Per-residue RNS: ~seconds per target once MSA loaded.
- Structure fetching: ~5 min total (HTTPS to AlphaFold DB + PDB).
- Binding-site annotation: ~10-30 min (KLIFS + GPCRdb).

**Total wall-clock for RNS pipeline on AWS: ~3-5 hours.**

### 3.5 Implementation risks

| Risk | Mitigation |
|---|---|
| MSA depth varies per target; under-represented receptors might produce shallow MSAs | Report per-target MSA depth; flag <50 sequences as low-confidence |
| GPCRdb residue indexing inconsistencies across receptor families | Validate returned indices fall within protein sequence length; log mismatches |
| KLIFS coverage gaps (some kinases missing annotation) | Log missing targets; exclude from RNS analysis; document in supplement |
| Prabakaran-Bromberg paper ambiguity on implementation details | Validation gate forces convergence on parameters that reproduce published values; failure → pivot to fallback |
| AlphaFold DB or PDB transient unavailability | Exponential backoff, retry-with-jitter, on-disk cache for already-fetched files |

---

## 4. Scaffold-diversity module design

`target_affinity_ml/benchmarks/scaffold_diversity.py` per spec Section 5.1.

### 4.1 Per-target metrics

| Metric | Definition | Implementation |
|---|---|---|
| `n_scaffolds` | Unique Bemis-Murcko generic scaffolds | RDKit `MurckoScaffold.GetScaffoldForMol` with generic atom-typing |
| `scaffold_entropy` | Shannon entropy of scaffold-frequency distribution | `-Σ p_i log p_i` |
| `largest_cluster_fraction` | Fraction of compounds in most populous scaffold | `max(scaffold_counts) / n_compounds` |
| `mean_tanimoto` | Mean pairwise Morgan FP Tanimoto distance | Random sample of ≤500 compound pairs |
| `activity_cliff_frequency` | Pairs with Tanimoto ≥0.7 AND ΔpActivity ≥1.5 | Reuse Plan 1's JAK case study implementation |

### 4.2 Per-class aggregates

Mean, median, IQR across all targets in the class. Reported in `results/supplement/per_target_metrics.csv` and `results/tables/04_metric_correlations.csv`.

### 4.3 Statistical analyses

Two regressions across both classes pooled (spec 5.1):

1. **Random → scaffold degradation regression**
   - Y: per-target (RMSE_scaffold − RMSE_random)
   - X: per-target scaffold-diversity metrics (univariate, then joint)
   - Class as covariate; class × X interaction tests within-class vs between-class

2. **Scaffold → target degradation regression**
   - Y: per-target (RMSE_target − RMSE_scaffold)
   - X: per-target + per-class metrics
   - Same class-stratified slope test

Outputs: per-metric slope estimates with class-stratified 95% CIs, joint R², scatter plots feeding main-text Figure 3.

---

## 5. Hypothesis-tests module design

`target_affinity_ml/benchmarks/hypothesis_tests.py`. Wraps the lower-level `evaluation/{bootstrap,multi_seed_analysis}.py` into the pre-registered H1-H4 framework.

### 5.1 H1-H4 formal tests

| # | Pre-registered hypothesis | Test design | Outputs |
|---|---|---|---|
| H1 | RF competitive with deep models on scaffold/target splits | Paired t-test across 5 seeds: RF vs ESM-FP-MLP AND RF vs Fusion, per (class × split) — 12 tests total. Cohen's d for effect size. Bootstrap 10K CI for mean RMSE difference. | Row per test: model_pair, class, split, mean_diff, cohens_d, CI_low, CI_high, t_stat, p_raw, p_bonferroni, verdict ∈ {RF wins, ties, loses} |
| H2 | Random→scaffold +12-52%, scaffold→target +25-60% degradation | Per-(model × class) degradation ratios. Class × split interaction via two-way ANOVA. | 14 ratio rows per class + 1 ANOVA result. Verdict per spec 6.1: (a) within range, (b) below, (c) above |
| H3 | ESM-2 advantage vanishes on target split | Per-(class × split) compute ESM-FP-MLP RMSE − MLP RMSE. Class × split interaction test on the advantage value. | 6 advantage values + 1 interaction-test result. Verdict: (a) same pattern, (b) ESM-2 still helps GPCR target, (c) ESM-2 never helps GPCRs |
| H4 | Single-seed scaffold tests flip with multi-seed | For each (model-pair × split), count rates where sign(single_seed_diff) ≠ sign(multi_seed_mean_diff). | Per-class false-positive rates + between-class difference + bootstrap CI. Verdict: (a) similar, (b) lower, (c) higher |

**Multiple-testing correction:** Bonferroni primary (family ~20 tests, α=0.05 → threshold 0.0025). Per-target exploratory regressions use FDR (Benjamini-Hochberg) reported separately.

### 5.2 Between-class machinery

Three new test types, all in `hypothesis_tests.py`:

1. **Class × split interaction tests:** `statsmodels.formula.api.ols("rmse ~ C(class) * C(split) + C(model)", data).fit()` — interaction F-stat.
2. **Class-stratified slope tests:** regress within each class; `z = (slope_kinase − slope_gpcr) / sqrt(SE_k² + SE_g²)`; two-sided p-value from standard normal.
3. **Bootstrap difference-of-differences:** resample within each class 10K times; compute difference-of-differences; report 95% CI. Uses existing `evaluation/bootstrap.py`.

### 5.3 RNS-stratified ESM-2 analysis (spec 5.2)

- Y: per-target (ESM-FP-MLP RMSE − MLP RMSE) — the embedding advantage; negative = ESM-2 helps
- X: per-target RNS score
- Within each class separately, then pooled
- **Per-class RNS distribution comparison:** Kolmogorov-Smirnov 2-sample test on the per-class RNS distributions; Welch t-test on per-class mean RNS
- **Sensitivity analysis:** repeat with conservation-entropy substituted for RNS; report whether slope-significance conclusion holds

### 5.4 Outputs

| Output | Format | Consumer |
|---|---|---|
| `results/tables/01_dataset_summary.csv` | Table 1 (datasets side-by-side) | `07_cross_class_comparison.ipynb` |
| `results/tables/02_headline_rmse.csv` | Table 2 (7 models × 3 splits × 2 classes, mean ± SD) | `07_cross_class_comparison.ipynb` |
| `results/tables/03_hypothesis_outcomes.csv` | Table 3 (H1-H4 with effect sizes, p-values, verdicts) | `07_cross_class_comparison.ipynb` |
| `results/tables/04_metric_correlations.csv` | Table 4 (RNS + scaffold-diversity correlations) | `06_rns_analysis.ipynb` + `05_scaffold_diversity.ipynb` |
| `results/supplement/per_target_metrics.csv` | All per-target metrics | All notebooks |
| `results/supplement/structure_provenance.csv` | RNS structure-source provenance per spec 5.4 Tier 1 | `06_rns_analysis.ipynb` |
| `results/supplement/sensitivity_analyses.csv` | PDB-only RNS, conservation-entropy fallback, alternative corrections | `06_rns_analysis.ipynb` |

### 5.5 Main-text figures (per spec 6.3)

| Figure | Content | Notebook |
|---|---|---|
| 1 | Benchmark design overview (adapted from kinase preprint to cover both classes) | `07` |
| 2 | Headline replication: performance degradation across splits, side-by-side panels per class | `07` |
| 3 | Scaffold-diversity vs degradation: scatter colored by class, regression lines | `05` |
| 4 | RNS-stratified ESM-2 advantage: scatter with structure-source markers | `06` |
| 5 | Cross-class summary: radar/grouped-bar visualization of H1-H4 outcomes | `07` |

---

## 6. Sequencing

**Approach A — RNS-first** (locked in during brainstorming).

```
PHASE 1 — RNS validation gate (Week 1, ~5 working days)
  ├─ T-3.1   Library benchmarks/__init__ scaffolding + reference data file
  ├─ T-3.2   fetch_structure() — PDB + AlphaFold DB acquisition with caching
  ├─ T-3.3   fetch_binding_site() — KLIFS + GPCRdb adapters
  ├─ T-3.4   compute_msa() — jackhmmer wrapper, single-target test
  ├─ T-3.5   compute_per_residue_rns() — Prabakaran-Bromberg algorithm
  ├─ T-3.6   validation_gate() — runs on 5-10 published reference proteins
  └─ ★ GO/NO-GO DECISION ★

PHASE 2 — Parallel build-out (Weeks 2-3, ~7-8 working days)
  Three tracks dispatchable in parallel via subagents:

  Track 2A — Kinase reference data hosting (~1-2 days)
    ├─ T-3.7   Identify + verify kinase reference files (features, predictions, metrics)
    ├─ T-3.8   Add data/kinase_reference/ to GPCR repo with .gitignore exception
    ├─ T-3.9   Sync from mlproject + AWS → commit + push
    └─ T-3.10  Update GPCR repo README + data card with kinase-data attribution

  Track 2B — Scaffold-diversity module (~2-3 days)
    ├─ T-3.11  Library benchmarks/scaffold_diversity.py + unit tests
    ├─ T-3.12  Per-target metric computation for kinase + GPCR
    └─ T-3.13  Regression machinery (random→scaffold, scaffold→target)

  Track 2C — RNS full pipeline + Hypothesis tests (~4-5 days)
    ├─ T-3.14  Run full RNS pipeline on AWS — 543 targets parallelized
    │          (96 CPUs, ~3-5 hours wall-clock)
    ├─ T-3.15  Library benchmarks/hypothesis_tests.py + unit tests
    ├─ T-3.16  Structure-source tier 3 sensitivity (PDB vs AlphaFold correlation)
    └─ T-3.17  ★ Structure-source decision tree branch ★

PHASE 3 — Application notebooks + release (Week 3-4, ~5 working days)
  ├─ T-3.18  GPCR repo notebook 05_scaffold_diversity.ipynb
  ├─ T-3.19  GPCR repo notebook 06_rns_analysis.ipynb
  ├─ T-3.20  GPCR repo notebook 07_cross_class_comparison.ipynb
  ├─ T-3.21  Generate 4 main-text tables + 5 main-text figures
  ├─ T-3.22  Library v1.2.0 release (CHANGELOG + tag + push); fixes stale __version__
  ├─ T-3.23  Plan 3 completion summary + Plan 4 handoff doc
  └─ T-3.24  GPCR repo v1.1.0 tag (final analysis state); supports later Zenodo deposit
```

**Total: 24 tasks, ~17-20 working days, AWS compute well under 1 day total.**

---

## 7. Pre-specified branch points

| Branch | Condition | Action |
|---|---|---|
| **RNS validation gate** | Spearman ρ ≥ 0.7 OR mean abs. deviation ≤ 10% | Continue full RNS path |
| **RNS validation gate** | Both criteria miss after debugging cycle | Pivot to conservation-entropy primary; RNS marked experimental; plan updated |
| **PDB-vs-AF correlation** (T-3.17) | r > 0.85 | Primary analysis = combined RNS with pLDDT weighting |
| **PDB-vs-AF correlation** | 0.7 ≤ r < 0.85 | Primary = PDB-only; combined goes to supplement with caveat |
| **PDB-vs-AF correlation** | r < 0.7 | Primary = PDB-only; AF-only targets excluded, listed in supplement |
| **MSA depth per target** | < 50 sequences | Flag as low-confidence; included with caveat |
| **KLIFS coverage** | Targets missing annotation | Logged + excluded from RNS analysis (kinase) |
| **Kinase reference data completeness** (T-3.7) | Per-run predictions missing | Trigger Plan 1.5 supplementary work: rerun kinase benchmark to extract predictions |

---

## 8. Risks

Delta from spec Section 7:

| ID | Risk | Plan 3 mitigation |
|---|---|---|
| R2 (spec) | RNS implementation exceeds budget | Validation gate front-loaded (Phase 1); fallback pre-specified |
| R4 (spec) | AlphaFold systematic bias in cross-class RNS | Tier 1+2+3 strategy fully implemented (T-3.16 + 3.17) |
| R7 (spec) | Multiple-testing concerns | Bonferroni primary, FDR for exploratory regressions, family-size disclosed in supplement |
| R-3a (new) | UniRef50 unavailable on AWS | Download to AWS local volume up-front (~30 GB one-time) |
| R-3b (new) | KLIFS/GPCRdb API rate limits or downtime | Cache responses locally; if API down, use most recent cached copy with provenance noted |
| R-3c (new) | Library v1.2.0 release breaks GPCR repo | Pin GPCR repo to v1.2.0 explicitly in `pyproject.toml`; CI test that GPCR notebooks still import after upgrade |
| R-3d (new) | Kinase reference predictions missing from Plan 1 outputs | T-3.7 verifies before downstream work begins; if missing, triggers Plan 1.5 supplementary kinase re-run on AWS GPUs |

## 9. Stop conditions

In addition to spec 7.4:

1. (spec) RNS validation fails repeatedly across simpler fallback metric too → **halts Plan 3 entirely**
2. **(new)** Kinase reference data missing critical artifacts → triggers Plan 1.5 supplementary work to re-run the kinase benchmark with full prediction outputs
3. **(new)** Cross-class hypothesis tests all reject the kinase findings (H1-H4 all show GPCRs *diverge* in the *opposite* direction) → not a halt per se, but warrants discussion-section reframe and potentially additional analyses before paper submission

---

## 10. Pre-registration

The Plan 3 plan document itself (`docs/superpowers/plans/2026-05-27-plan3-cross-class-analysis.md`, to be generated next) is the pre-registration artifact. It pre-specifies:

- H1-H4 tests, effect-size measures, multiple-testing corrections
- Verdict templates per spec 6.1 (a/b/c outcomes for each hypothesis)
- Go/no-go criterion for RNS validation gate (Section 3.2)
- Structure-source decision tree (Section 3.3 / 7)
- Stop conditions (Section 9)

Per spec 6.5, no OSF/AsPredicted deposit — version-controlled plan in git is sufficient for bioRxiv preprint.

---

## 11. Estimated effort + compute

| Phase | Working days | AWS wall-clock |
|---|---|---|
| Phase 1 (validation gate) | ~5 | <1 hour |
| Phase 2 (parallel build-out) | ~7-8 | ~3-5 hours (RNS pipeline) |
| Phase 3 (notebooks + release) | ~5 | minimal |
| **Total** | **~17-20** | **~4-6 hours** |

AWS instance can be stopped between phases to control cost — only Phase 2 has substantial compute load.

---

## 12. Decision history (brainstorming summary)

| Q | Decision | Rationale |
|---|---|---|
| Q1 | Full Prabakaran-Bromberg RNS + conservation-entropy fallback as sensitivity | Strongest mechanistic claim + robustness check |
| Q2 | End at results assembly; manuscript out of scope | Engineering-shaped Plan 3; writing is different cognitive work |
| Q3 | Library benchmarks/ + GPCR repo notebooks | Matches spec 3.4; library stays reusable for Phase 2-N |
| Q4 | Kinase reference data committed into GPCR repo | Simplest fully-reproducible path; Zenodo deferred to manuscript phase |
| Q5 | RNS-first execution structure | Front-loads spec R2 risk; clean pivot if validation gate fails |
| User clarifications | AWS for everything heavy (96 CPU + 4 A10G GPU available); AlphaFold uses the DB API (no rate limits), not the prediction service | Plan compute placement section reflects this |

---

**Status:** Design approved by user during brainstorming dialogue. Ready for `superpowers:writing-plans` to generate the executable Plan 3.
