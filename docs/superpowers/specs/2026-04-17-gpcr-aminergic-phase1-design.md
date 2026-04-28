# GPCR Aminergic Phase 1 Benchmark — Design Document

**Date:** 2026-04-17
**Author:** Joshua Abbott (jmabbott40)
**Status:** Design — pre-implementation
**Project:** Multi-class expansion of cheminformatics ML benchmark (Phase 1 of N)
**Predecessor:** [`kinase-affinity-baselines`](https://github.com/jmabbott40/kinase-affinity-baselines) (preprint v1, frozen)

---

## Executive summary

Apply the seven-model benchmark framework from the kinase preprint to aminergic Class A G-protein-coupled receptors (GPCRs), testing whether four core findings — Random Forest competitiveness, scaffold-to-target performance degradation magnitude, ESM-2 functioning as an implicit target identifier, and single-seed false-positive risk — generalize beyond kinases. Add two methodological layers (scaffold-diversity correlation analysis and Residue Neighborhood Significance (RNS)-stratified ESM-2 analysis) to provide mechanistic explanation for any class-dependent differences. Deliver as a standalone preprint, a reusable Python library (`target-affinity-ml`), and a class-specific application repository (`gpcr-aminergic-benchmarks`) — establishing infrastructure for future Phase 2-N target classes.

**Phase 1 timeline:** ~8-9 weeks (parallelized across 4 GPUs and 96 CPUs).
**Compute budget:** ~330 GPU-hours, absorbed within engineering and analysis weeks.
**Deliverables:** preprint manuscript, two new GitHub repositories, multi-class infrastructure for Phase 2-N.

---

## 1. Scientific question and contribution

### 1.1 Question

Do the four core findings from the kinase benchmark — Random Forest competitiveness with deep models, scaffold-to-target performance degradation magnitude, ESM-2 functioning primarily as an implicit target identifier, and single-seed false-positive risk in scaffold-split comparisons — generalize to a structurally and pharmacologically distinct target class (aminergic Class A GPCRs)?

### 1.2 Why this matters

The kinase preprint establishes findings that *could* be artifacts of kinase-specific structure-activity relationships (SAR): congeneric series, conserved hinge motifs, solvent-exposed catalytic clefts. Aminergic GPCRs are the strongest second test case because they offer:

- **Dense ChEMBL coverage** for individual receptors (5-HT2A, D2, β2 each have tens of thousands of records)
- **Well-defined subfamilies** (dopamine, serotonin, adrenergic, histamine, muscarinic) for target-held-out splits
- **Fundamentally different binding-site architecture** — transmembrane helical bundle, allosteric pockets, broader chemical diversity than ATP-competitive kinase inhibitors

If findings replicate, that is strong evidence the original kinase results capture real properties of cheminformatics ML rather than kinase quirks. If findings diverge, the *direction* of divergence is informative — and the new methodological layers (RNS, scaffold diversity) provide mechanistic axes to explain it.

### 1.3 Differentiation from a pure replication

Two new methodological layers ensure this paper stands on its own scientifically beyond replication:

1. **Scaffold-diversity correlation analysis** — quantify scaffold concentration metrics (Bemis-Murcko entropy, largest-cluster fraction, intra-class Tanimoto distribution) and correlate them with the random→scaffold→target performance gap *across both classes combined*. Tests the hypothesis that scaffold-split degradation is a function of scaffold concentration, not target-class identity.

2. **RNS-stratified ESM-2 analysis** — compute per-target Prabakaran-Bromberg Residue Neighborhood Significance scores and correlate RNS with the ESM-FP-MLP-vs-MLP improvement per target, separately within each class. Tests whether protein embeddings help most where they encode structurally informative neighborhoods, and whether GPCRs' more variable binding-site residues yield systematically different RNS distributions than kinases.

Together, these provide the "why" mechanism that pure replication lacks.

---

## 2. Phase 1 deliverables and timeline

### 2.1 Three deliverables

1. **Preprint manuscript** — bioRxiv preprint, journal target *Journal of Chemical Information and Modeling* or *Bioinformatics*. ~12-15 main-text pages plus supplement.

2. **`target-affinity-ml` library (v1.0)** — Python package extracted from the kinase repo. Class-agnostic, semantically versioned, pip-installable. Contains all model implementations, training framework, evaluation metrics, UQ methods, splitting strategies, and the new scaffold-diversity and RNS modules.

3. **`gpcr-aminergic-benchmarks` repository** — Application repository depending on `target-affinity-ml==1.0.0`. Contains aminergic-specific data ingestion, target selection logic, audit-gate notebook, scaffold-diversity and RNS analyses, and figure-generation code.

### 2.2 In scope for Phase 1

- 7-model × 3-split × 5-seed full benchmark on aminergic Class A
- Re-run of identical protocol on kinase data with library v1.0 (validation + apples-to-apples comparison)
- Scaffold-diversity metrics for both classes
- RNS scores for both classes with structure-source handling
- Cross-class comparison tables and figures
- Pre-registered hypothesis tests on the four kinase findings with effect sizes and CIs

### 2.3 Explicitly out of scope (deferred to later phases)

- Aminergic selectivity case study (Phase 2 — direct analog to JAK)
- Functional-assay (EC50) inclusion analysis (Phase 2 — methodological side study)
- Proteases, nuclear hormone receptors, ion channels (Phase 2-N — additional class applications)
- Meta-model for architecture selection (Phase 3+ — needs ≥4 classes for statistical power)
- Cross-class transfer learning experiments (Phase 3+)
- Pan-target benchmarking synthesis paper (Phase 4+)

### 2.4 Parallelized timeline (4 GPUs, 96 CPUs)

| Weeks | Track A (Engineering) | Track B (Data + Analysis) | Track C (Manuscript) |
|---|---|---|---|
| 1-2 | Library extraction, refactoring, CI tests | GPCR data audit, target list, curation criteria | — |
| 3 | Kinase re-run on library v1.0 (~2 days compute) | GPCR data curation finalized; feature engineering kicks off | Manuscript outline + Methods drafting |
| 4 | RNS pipeline implementation + validation | GPCR ESM-2 embeddings + Morgan/RDKit features | Methods refinement |
| 5 | GPCR benchmark runs (~2 days compute) | RNS scores computed for both classes | Results scaffolding (figures defined) |
| 6 | — | Scaffold-diversity metrics + cross-class correlation analysis | Results drafting (with placeholder figures) |
| 7 | — | Final results assembly + figure generation | Discussion + supplementary materials |
| 8-9 | — | Pre-submission validation (extra seeds if needed) | Manuscript polish + bioRxiv submission |

**Total: ~8-9 weeks under good conditions, ~10-12 weeks under realistic conditions with one issue surfacing.**

### 2.5 Compute breakdown

- Kinase re-run (105 runs, ~160 GPU-hr): ~2 days wall-clock at 4-way GPU parallelism
- ESM-2 embedding for ~50 GPCR targets: ~30 minutes batched across 4 GPUs
- GPCR benchmark (105 runs, ~160 GPU-hr): ~2 days wall-clock
- RNS structural computations: CPU-bound, ~hours on 96 cores
- **Total compute: ~5 wall-clock days, easily absorbed within Track A weeks**

---

## 3. Repository architecture

### 3.1 Three-artifact structure

```
                    ┌─ kinase-affinity-baselines (frozen v1.0)
target-affinity-ml ─┤
                    └─ gpcr-aminergic-benchmarks (Phase 1)
                       └─ (Phase 2-N: additional class repos)
```

The library is the **durable artifact** citable across multiple preprints. Each application repo is a **specific scientific contribution** with its own preprint and Zenodo deposit.

### 3.2 `target-affinity-ml` library structure

```
target-affinity-ml/
├── pyproject.toml               # Versioned pip-installable package
├── src/target_affinity_ml/
│   ├── data/                    # Class-agnostic data pipeline
│   │   ├── chembl_fetcher.py
│   │   ├── standardize.py
│   │   ├── curate.py
│   │   └── splits.py
│   ├── features/                # All feature pipelines
│   │   ├── fingerprints.py
│   │   ├── descriptors.py
│   │   ├── protein_embeddings.py
│   │   └── molecular_graphs.py
│   ├── models/                  # All 7 model implementations
│   │   ├── rf_model.py, xgb_model.py, ...
│   │   ├── esm_fp_mlp_model.py
│   │   ├── gnn_model.py, fusion_model.py
│   │   └── deep_base.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── deep_trainer.py
│   ├── evaluation/              # Metrics, UQ, bootstrap, multi-seed
│   ├── visualization/
│   └── benchmarks/              # NEW: cross-class methodology
│       ├── scaffold_diversity.py
│       └── rns_scoring.py
├── tests/                       # Unit + integration tests
└── configs/                     # Default model hyperparameters
```

**Versioning policy:** Semantic versioning. `v1.0.0` for Phase 1 papers. Patch versions for non-result-changing bug fixes. Minor versions for new features. **Major version bumps for any change that produces different numerical results.**

### 3.3 `kinase-affinity-baselines` (frozen at v1.0)

**Action:** Tag a `v1.0` release matching the published preprint exactly. Do not modify the published codebase. Add a single section to README pointing to the new framework:

> "This repository accompanies the kinase-only preprint [citation]. Ongoing multi-class work uses the modular framework [`target-affinity-ml`](link). Phase 1 of the multi-class expansion benchmarks aminergic GPCRs in [`gpcr-aminergic-benchmarks`](link)."

### 3.4 `gpcr-aminergic-benchmarks` application repo

```
gpcr-aminergic-benchmarks/
├── pyproject.toml               # Depends on target-affinity-ml==1.0.0
├── README.md                    # Links to library + sister kinase repo
├── configs/
│   └── dataset_aminergic_v1.yaml
├── data/
│   └── processed/v1/
├── src/gpcr_aminergic_benchmarks/
│   ├── data_audit.py            # Phase 1 audit gate
│   ├── target_selection.py      # Aminergic family inclusion logic
│   └── analyses/
│       ├── cross_class_comparison.py
│       └── rns_stratification.py
├── notebooks/
│   ├── 00_data_audit.ipynb      # Audit gate (runs Day 1)
│   ├── 01_aminergic_curation.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_uncertainty.ipynb
│   ├── 05_scaffold_diversity.ipynb
│   ├── 06_rns_analysis.ipynb
│   └── 07_cross_class_comparison.ipynb
└── results/
    ├── tables/, figures/
    ├── predictions/, models/
    └── supplement_tables/
```

**Critical design rule:** Anything that *could* be reused for Phase 2-N goes in the library. Anything truly aminergic-specific stays in the application repo. When in doubt, default to the library (easier to move code from library to app repo than vice versa).

### 3.5 Linking strategy

- Library README has a "Used by:" section listing both `kinase-affinity-baselines` (frozen) and `gpcr-aminergic-benchmarks` (Phase 1)
- Both application repos cite the library in their READMEs
- All three repos owned by the same GitHub user (`jmabbott40`)
- Zenodo deposits follow the same three-artifact pattern: existing kinase deposit (immutable) + new library DOI + new GPCR deposit

---

## 4. Data acquisition strategy

### 4.1 Aminergic Class A target list (~35-40 human receptors)

| Family | Receptors | Notes |
|---|---|---|
| Dopamine | D1, D2, D3, D4, D5 (5 targets) | All five subtypes |
| Serotonin | 5-HT1A-F, 5-HT2A-C, 5-HT4, 5-HT5A, 5-HT6, 5-HT7 (~12 targets) | Excludes 5-HT3 (ionotropic, not GPCR) |
| Adrenergic | α1A/B/D, α2A/B/C, β1, β2, β3 (9 targets) | All adrenoceptors |
| Histamine | H1, H2, H3, H4 (4 targets) | All four histamine receptors |
| Muscarinic | M1, M2, M3, M4, M5 (5 targets) | All five muscarinic receptors |
| TAAR (optional) | TAAR1 if data sufficient | Include only if ≥1000 binding records |

### 4.2 Inclusion criteria (parallel to kinase pipeline)

- `target_type = "SINGLE PROTEIN"`
- `organism = "Homo sapiens"`
- `target_chembl_id` matching curated aminergic Class A list
- `standard_relation = "="` (no inequalities)
- `standard_units = "nM"` with pChEMBL value present
- `assay_confidence_score >= 7`
- `standard_type ∈ {"IC50", "Ki", "Kd"}` (no EC50, no functional assays)

### 4.3 Curation pipeline (reused from kinase, identical)

- Salt removal, charge neutralization, canonical SMILES (RDKit)
- Molecular weight 100-900 Da, ≤100 heavy atoms
- pActivity range [3.0, 12.0]
- Median aggregation per (canonical_smiles, target_chembl_id, activity_type)
- Noise flag: ≥3 measurements with std > 1.0 pActivity → "noisy"
- Active/inactive label at pActivity ≥ 6.0

### 4.4 Audit gate (Step 0 — runs before any infrastructure work)

The audit produces a report answering:

1. Per-target binding-record counts (IC50/Ki/Kd)
2. Compounds-per-target distribution
3. Family coverage (≥2 targets per family at threshold)
4. Scaffold diversity (Bemis-Murcko scaffold count per target)

**Pass thresholds:**

- ≥80% of targets have ≥1000 binding records → **proceed with binding-only**
- 60-80% pass → **proceed but flag in paper as a limitation**
- <60% pass → **pivot to EC50 inclusion with `assay_type` flagging, with sub-analysis on binding-only subset**

### 4.5 Subfamily structure for target-held-out splits

Five aminergic families form natural subfamily groups for target splits:

- **Primary protocol:** single target split (matches kinase paper for direct comparison)
- **Optional supplement:** leave-one-family-out (5 splits) if data density permits

Decision deferred to post-audit.

### 4.6 Implementation

Data audit lives in `gpcr-aminergic-benchmarks/notebooks/00_data_audit.ipynb` and runs **Day 1 of Week 1** before library refactoring even starts. The audit gates the rest of the project.

---

## 5. Methodological additions

### 5.1 Scaffold-diversity metrics (`benchmarks/scaffold_diversity.py`)

**Per-target metrics:**

| Metric | Definition | Implementation |
|---|---|---|
| `n_scaffolds` | Unique Bemis-Murcko generic scaffolds | RDKit `MurckoScaffold.GetScaffoldForMol` with generic atom-typing |
| `scaffold_entropy` | Shannon entropy of scaffold-frequency distribution | `-Σ p_i log p_i` |
| `largest_cluster_fraction` | Fraction in most populous scaffold | `max(scaffold_counts) / n_compounds` |
| `mean_tanimoto` | Mean pairwise Morgan-FP Tanimoto | Random sample of ≤500 compound pairs |
| `activity_cliff_frequency` | Pairs with Tanimoto ≥0.7 + ΔpActivity ≥1.5 | Reuse JAK case study implementation |

**Per-class aggregates:** mean, median, IQR across all targets in the class.

**Statistical analysis:**

Two regressions, computed across both classes pooled:

1. **Random → scaffold degradation regression**
   - Y: per-target RMSE_scaffold − RMSE_random
   - X: per-target scaffold-diversity metrics (univariate, then joint)
   - Class as covariate: tests whether the relationship holds within each class or only between classes
   - Hypothesis: Higher scaffold concentration predicts larger leakage gap

2. **Scaffold → target degradation regression**
   - Y: per-target RMSE_target − RMSE_scaffold
   - X: per-target metrics + per-class metrics (scaffold diversity within target's family)
   - Hypothesis: Targets in less scaffold-diverse families show larger target-split degradation

**Outputs:** supplementary table (per-target metrics × class), main-text figure (scatter + regression lines), explicit statistical statement in results.

### 5.2 RNS-stratified ESM-2 analysis (`benchmarks/rns_scoring.py`)

**RNS framework (Prabakaran & Bromberg):** Residue Neighborhood Significance scores how predictive a residue's spatial+sequence neighborhood is of functional importance. Higher RNS = more informative neighborhood = embedding methods like ESM-2 should capture more useful signal.

**Pipeline:**

1. **Structure acquisition per target**
   - Primary source: PDB (experimental structures preferred)
   - Fallback: AlphaFold DB (computational, available for nearly all human proteins)
   - For aminergic GPCRs: ~25 of 35 have experimental structures; rest from AlphaFold
   - For kinases: KLIFS database provides curated structures + binding-site annotations

2. **Binding-site residue annotation**
   - Aminergic GPCRs: orthosteric pocket residues annotated via GPCRdb (well-curated for these receptors); typically ~25-40 residues per receptor in the TM bundle
   - Kinases: ATP-binding-pocket residues from KLIFS (85-residue pocket definition)

3. **Per-residue RNS computation**
   - For each binding-site residue:
     - Identify spatial neighbors (Cα distance ≤8 Å) and sequence neighbors (±5 positions)
     - Compute Shannon entropy of local sequence environment across a homolog set
     - Weight by evolutionary conservation (BLOSUM-scored)
   - Output: per-residue RNS in [0, 1]
   - Per-target aggregate: mean RNS over binding-site residues

4. **Validation against published method**
   - Compute RNS on 5-10 well-studied proteins from the Prabakaran-Bromberg paper
   - Verify scores reproduce published values within ±5% tolerance
   - **Validation gate:** if implementation diverges significantly, debug before proceeding

**Statistical analysis:**

Per-target regression, separately within each class then pooled:

- Y: ESM-FP MLP RMSE − MLP RMSE (the embedding advantage; negative = ESM-2 helps)
- X: per-target RNS score
- Hypothesis: Higher RNS predicts larger embedding advantage (more negative Y)

**Critical secondary analysis:** Compare per-class RNS distributions. If GPCRs have systematically higher RNS than kinases, that mechanistically explains "do protein embeddings help GPCRs more?"

### 5.3 Risk mitigation: RNS implementation effort

**Tallest pole risk.** Three sub-decisions to manage:

1. **Use existing structural-biology tooling** — Biopython for structure I/O, DSSP for secondary structure, FreeSASA for accessibility
2. **Validate early (Week 4 task)** — RNS reproduces published values *before* running on all 542 targets
3. **Defined fallback metric** — if RNS proves intractable, replace with simpler "binding-site residue conservation" metric (entropy of binding-site columns in homolog MSA). Loses the published-framework citation but preserves the analytical structure.

### 5.4 Structure source handling

**Problem:** Mixing experimental PDB and AlphaFold structures introduces a confound — RNS values from AlphaFold may be systematically biased relative to experimental ones, especially for side-chain-sensitive computations.

**Three-tier mitigation:**

**Tier 1: Per-target structure provenance annotation**

Captured in supplementary table:

- `structure_source`: `"PDB"` | `"AlphaFold"` | `"hybrid"`
- `pdb_id` (if available), `pdb_resolution` (Å)
- `binding_site_pLDDT_mean`, `binding_site_pLDDT_min` (AlphaFold)
- `conformational_state`: `"active"` | `"inactive"` | `"unknown"` (GPCRs)

**Tier 2: pLDDT-weighted RNS for AlphaFold structures**

```
target_RNS = Σ (residue_RNS × pLDDT_normalized) / Σ pLDDT_normalized
```

Where `pLDDT_normalized = max(0, (pLDDT - 50) / 50)`. Residues below pLDDT 50 contribute essentially nothing; residues at 90+ contribute fully. Experimental PDB structures use uniform weighting.

**Tier 3: PDB-vs-AlphaFold sensitivity analysis**

For ~25/35 aminergic GPCRs and ~150-200/507 kinases with both structure types available:

- Compute correlation between PDB-RNS and AlphaFold-RNS across paired targets
- Decision tree:

```
If overall PDB-AF correlation > 0.85:
    Primary analysis = combined (PDB + AF) with pLDDT weighting
    Supplementary = PDB-only sanity check

If 0.7 ≤ correlation < 0.85:
    Primary analysis = PDB-only (more conservative)
    Supplementary = combined (with explicit AF-bias caveat)

If correlation < 0.7:
    Primary = PDB-only
    AF-only targets excluded from RNS analyses, listed in supplement
```

**Why this matters for cross-class comparison:** If aminergic GPCRs disproportionately rely on AlphaFold (because experimental membrane-protein structures are scarcer) and kinases mostly use PDB, then class-level RNS differences could be artifacts of structure source. The Tier 3 sensitivity analysis directly checks for this.

**Reporting:** Methods section describes structure provenance, pLDDT weighting, decision criterion. Discussion acknowledges AlphaFold side-chain accuracy and conformational-state limitations.

---

## 6. Cross-class comparison structure

### 6.1 Pre-registered hypothesis tests

Each kinase finding becomes a pre-registered hypothesis with explicit replication outcomes. Pre-registration in the methods section commits to the analyses *before* GPCR results are computed.

| # | Kinase finding | Replication test | Possible outcomes |
|---|---|---|---|
| H1 | RF competitive with deep models on scaffold/target splits | Paired t-test across 5 seeds: RF vs ESM-FP-MLP and RF vs Fusion, per split | (a) Replicates (RF tied/wins); (b) Diverges (deep models win); (c) Reverses (RF loses) |
| H2 | Random→scaffold RMSE +12-52%, scaffold→target +25-60% | Compute per-model degradation ratios; compare distributions to kinase | (a) Within range; (b) Below range; (c) Above range |
| H3 | ESM-2 advantage vanishes on target split | (ESM-FP-MLP RMSE − MLP RMSE) per split, per class; class × split interaction test | (a) Same pattern; (b) ESM-2 still helps on GPCR target split; (c) ESM-2 never helps on GPCRs |
| H4 | Single-seed scaffold-split tests flip with multi-seed | Count "false positive" comparisons across model-pair × split combinations | (a) Similar rate; (b) Lower rate; (c) Higher rate |

The paper commits to reporting actual outcomes regardless of direction.

### 6.2 Statistical machinery

**Within-class tests (established protocol):**

- Paired t-tests across 5 seeds for model-pair comparisons within a single split
- Bootstrap 10,000-resample CIs for absolute RMSE
- Multi-seed paired tests matching kinase paper

**Between-class tests (new methodology):**

1. **Class × split interaction tests** (for H2): two-way ANOVA with class and split as factors; test the interaction term
2. **Class-stratified slope tests** (for per-target metric regressions): regress within each class; z-test on slope difference
3. **Bootstrap difference-of-differences** (for headline findings): resample within each class 10,000 times; compute the difference of differences; report 95% CI

**Critical guardrail:** All cross-class comparisons use the **same library version** and **exact same model hyperparameters**. Any GPCR-specific deviation is documented and justified.

### 6.3 Tables and figures

**Main-text tables (4):**

| Table | Content |
|---|---|
| 1 | Dataset summary side-by-side: targets, compounds, records, scaffolds, activity types |
| 2 | Headline RMSE: 7 models × 3 splits × 2 classes (mean ± SD across 5 seeds) |
| 3 | Hypothesis test outcomes: H1-H4 with effect sizes, p-values, verdicts |
| 4 | RNS and scaffold-diversity correlations: per-class slopes, pooled, structure-source stratified |

**Main-text figures (5):**

| Fig | Content |
|---|---|
| 1 | Benchmark design overview (adapted from kinase preprint to cover both classes) |
| 2 | Headline replication: performance degradation across splits, side-by-side panels per class |
| 3 | Scaffold-diversity vs degradation: scatter colored by class, regression lines |
| 4 | RNS-stratified ESM-2 advantage: scatter with structure-source markers |
| 5 | Cross-class summary: radar/grouped-bar visualization of H1-H4 outcomes |

**Supplementary materials:**

- All per-target metrics (scaffold metrics, RNS, structure provenance, per-model RMSE)
- Sensitivity analyses (PDB-only, EC50-included if applicable)
- Per-model calibration and selective-prediction curves for both classes
- Per-seed metrics for all 210 training runs (105 kinase + 105 GPCR)

### 6.4 Reporting discipline

Each finding has all three reporting templates pre-written. The paper accommodates per-model, per-split granularity for nuanced "partial replication" outcomes.

### 6.5 Pre-registration commitment

Hypothesis tests in Table 3 are pre-registered in the introduction *before* any GPCR results are presented. The paper is structured to be unembarrassed by whichever outcome the data produces. **Optional:** post analysis plan to OSF or AsPredicted before running GPCR experiments. Decision deferred — in-paper pre-registration is sufficient for bioRxiv preprint.

---

## 7. Risk register and mitigations

### 7.1 High-priority risks

**R1: Library refactor breaks reproducibility of kinase preprint v1 numbers**

- *Probability:* Medium
- *Impact:* High (suspect cross-class comparison; Phase 1 delayed)
- *Mitigation:* Re-run with exact same seeds; tolerance threshold ±0.001 RMSE per model × split; debug to root cause if exceeded; CI tests in library lock kinase output values

**R2: RNS implementation cold-start exceeds 3-week budget**

- *Probability:* Medium-high
- *Impact:* Medium (weakens differentiating science)
- *Mitigation:* Validation gate Week 4; defined fallback (binding-site residue conservation entropy); RNS parallelized with other Phase 1 work

**R3: Aminergic data audit fails Option A thresholds**

- *Probability:* Low-medium
- *Impact:* Medium (triggers EC50 inclusion pivot)
- *Mitigation:* Audit runs Day 1; pivot decision pre-defined; if pivot to EC50, binding-only remains primary sub-analysis

### 7.2 Medium-priority risks

**R4: AlphaFold systematic bias in cross-class RNS**

- *Probability:* Medium
- *Impact:* Medium (could invalidate "GPCRs have higher RNS" headline)
- *Mitigation:* Three-tier strategy from Section 5.4 (provenance annotation + pLDDT weighting + sensitivity analysis)

**R5: Compute resources change mid-project**

- *Probability:* Low-medium
- *Impact:* Medium (timeline slips proportionally)
- *Mitigation:* Run kinase re-run early (Week 3); checkpoint-resumable benchmark configs; documented baseline compute requirements

**R6: Pre-registered outcomes don't match any of three templates**

- *Probability:* Medium
- *Impact:* Low (paper handles nuanced outcomes)
- *Mitigation:* Templates are starting points, not exhaustive; per-model, per-split granularity accommodates partial replication

**R7: Multiple-testing concerns across many comparisons**

- *Probability:* Medium
- *Impact:* Low-medium (reviewer pushback)
- *Mitigation:* Pre-register primary tests (4 H-hypotheses, ~20 individual tests); apply Bonferroni or Holm correction; per-target regressions framed as exploratory; FDR control reported

### 7.3 Low-priority risks

**R8: Reviewer rejects "aminergic Class A only" as too narrow**

- *Probability:* Low
- *Impact:* Low (Phase 2 is the response)
- *Mitigation:* Discussion explicitly justifies scope as cleanest second-class test

**R9: ChEMBL API changes or data version drift**

- *Probability:* Low
- *Impact:* Low-medium (version-mismatch issues)
- *Mitigation:* Pin both classes to same ChEMBL version (ChEMBL 36 or current at start); document in dataset cards; preference for stability over upgrade mid-project

**R10: Library extraction over-abstracts**

- *Probability:* Low-medium
- *Impact:* Low (maintenance burden)
- *Mitigation:* YAGNI principle; only abstract code that needs to vary between classes; refactor when second class needs it, not preemptively

### 7.4 Stop conditions

The Phase 1 design is paused/reconsidered if any of these occur:

1. Refactor produces non-tolerance kinase numbers for >1 model × split combination after debugging
2. Aminergic audit shows <30 viable targets (insufficient even with EC50 inclusion)
3. RNS validation fails repeatedly across simpler fallback metric too
4. GPU access drops below 1 GPU sustained

---

## Appendix A: Decision history

This design emerged from structured brainstorming. Key decisions made:

| Q | Decision | Rationale |
|---|---|---|
| Q1 | Phased approach (Phase 1 GPCR standalone, infrastructure for Phase 2-N) | De-risks engineering; landable preprint in 3 months; modular extension |
| Q2 | Aminergic Class A only | Cleanest second-class test; defensible scope; avoids data-heterogeneity confounds |
| Q3 | Two repos with library extraction | Library is durable artifact; preserves frozen kinase repo; cleanest Phase 2-N onboarding |
| Q4 | Replication + scaffold-diversity correlation | Differentiation beyond pure replication; framework scales to Phase 2-N |
| Q5 | Full kinase re-run with library v1.0 | Apples-to-apples cross-class comparison; defensible against version-drift critique |
| Q6 | Add RNS-stratified ESM-2 analysis | Mechanistic explanation; per-target stratification gives statistical power |
| Q7 | Binding-only data (IC50/Ki/Kd) with audit gate | Clean cross-class comparison; audit prevents pivot surprise |
| Timeline | 8-9 weeks parallelized across 4 GPUs | Compute is not bottleneck; engineering and analysis time dominate |

---

## Appendix B: Glossary

- **RNS:** Residue Neighborhood Significance (Prabakaran-Bromberg framework)
- **GPCR:** G-protein-coupled receptor
- **ESM-2:** Evolutionary Scale Model v2 (protein language model)
- **Murcko scaffold:** Bemis-Murcko ring-system decomposition of a molecular structure
- **pLDDT:** Per-residue confidence score from AlphaFold (0-100)
- **KLIFS:** Kinase-Ligand Interaction Fingerprint database
- **GPCRdb:** Curated database of GPCR sequences, structures, ligands

---

## Appendix C: References

- Original kinase preprint: [citation to preprint v1]
- Prabakaran & Bromberg (RNS framework): [citation]
- ESM-2 (Lin et al. 2023): "Evolutionary-scale prediction of atomic-level protein structure"
- AlphaFold (Jumper et al. 2021): "Highly accurate protein structure prediction with AlphaFold"
- ChEMBL (Mendez et al. 2019): "ChEMBL: towards direct deposition of bioassay data"
- KLIFS (van Linden et al. 2014): "KLIFS: A knowledge-based structural database to navigate kinase-ligand interaction space"
- GPCRdb (Pándy-Szekeres et al. 2023): "GPCRdb in 2023: state-specific structure models using AlphaFold2 and new ligand resources"

---

**Status:** Design complete. Ready for spec review loop and transition to writing-plans skill.
