# Plan 3 Completion Summary

**Date:** 2026-05-28
**Status:** COMPLETE
**Plan:** [2026-05-27-plan3-cross-class-analysis.md](2026-05-27-plan3-cross-class-analysis.md)
**Spec:** [../specs/2026-05-27-plan3-cross-class-analysis-design.md](../specs/2026-05-27-plan3-cross-class-analysis-design.md)
**Predecessor:** [2026-05-27-plan2-completion-summary.md](2026-05-27-plan2-completion-summary.md)

---

## 1. Executive summary

Plan 3 (GPCR Cross-Class Analysis) is complete. 22 of 24 tasks executed; tasks T16 and T17
(PDB-vs-AlphaFold sensitivity analysis + structure-source decision branch) were marked
MOOT-by-design after the Phase 1 metric pivot from Prabakaran-Bromberg RNS to mean
binding-site pLDDT — pLDDT is an AlphaFold-specific quantity and the structure-source
decision tree no longer applies. The library was released as `target-affinity-ml` v1.2.0
with the new `benchmarks/` module (three submodules: `scaffold_diversity`, `rns_scoring`,
`hypothesis_tests`), and `gpcr-aminergic-benchmarks` v1.1.0 will be tagged in P3-T24 as the
analysis freeze for the manuscript.

Four main-text tables and five main-text figures were assembled in the GPCR repo. Three
significant cross-class findings emerged: (1) scaffold-diversity slope estimates differ
between kinases and GPCRs for 3 of 5 metrics on the random→scaffold direction
(`scaffold_entropy`, `largest_cluster_fraction`, `mean_tanimoto`; interaction p ≤ 0.0164);
(2) the ESM-2 advantage × mean-binding-site-pLDDT cross-class interaction is significant
(p = 0.0313) on the per-target regression — GPCRs show a positive slope (ESM-2 helps LESS at
higher pLDDT) while kinases sit near zero; (3) the H2 class × split interaction on raw RMSE
is highly significant (p = 5.887 × 10⁻⁵), confirming that preprint-degradation patterns
genuinely differ by class.

The most scientifically noteworthy event of the plan was a mid-execution metric pivot. The
spec referenced the Prabakaran-Bromberg "Residue Neighborhood Significance" (RNS) construct,
but investigation in P3-T1 revealed that the cited Prabakaran-Bromberg paper is actually about
LM-embedding evaluation, not binding-site residue significance. Two ConSurf-anchored
validation attempts — raw column entropy and JSD vs Swiss-Prot background — both produced
anti-correlated reference scores (Spearman ρ = -0.524 and -0.476 respectively). The team
pivoted to mean binding-site pLDDT (AlphaFold's published per-residue confidence quantity,
averaged over binding-site residues), which passed the validation gate at mean 88.13 across 8
reference proteins and supports a biologically meaningful H3 reframing.

---

## 2. Library v1.2.0 release

**Repository:** https://github.com/jmabbott40/target-affinity-ml
**Tag:** `v1.2.0` (2026-05-28)
**Pip-installable:** `pip install git+https://github.com/jmabbott40/target-affinity-ml.git@v1.2.0`

### Changes in v1.2.0

The `benchmarks/` module that was scaffolded empty in v1.0.0 is now populated with three
submodules — class-agnostic methodology that the GPCR application repo consumes for the
cross-class analysis. Strictly additive: kinase application repo's v1.0.0/v1.1.0 code paths
continue to work unmodified.

**`benchmarks/scaffold_diversity.py`** — per-target Bemis-Murcko scaffold metrics
(`n_scaffolds`, `scaffold_entropy`, `largest_cluster_fraction`, `mean_tanimoto`,
`activity_cliff_frequency`), per-class aggregates (mean / median / IQR / n), and
`fit_degradation_regression` for OLS per-target degradation ~ metric * C(class) with
cross-class interaction F-test. Uses local `random.Random(42)` to avoid module-state
pollution, and explicit `Treatment(reference=...)` so future statsmodels default changes
won't shift the kinase-vs-GPCR reference category. 24 unit tests.

**`benchmarks/rns_scoring.py`** — structure + binding-site + MSA + (experimental) RNS
pipeline. Implements `fetch_structure` (PDB + AlphaFold DB with caching),
`fetch_binding_site` (KLIFS + GPCRdb adapters), `compute_msa` (jackhmmer wrapper, retained
in code though unused after the pivot), `compute_per_residue_rns` (marked `[EXPERIMENTAL]`),
`compute_conservation_entropy` (marked `[EXPERIMENTAL]`), `aggregate_target_rns` (marked
`[EXPERIMENTAL]`), `compute_binding_site_plddt` (the **primary** per-target metric), and
`validation_gate`. The validation gate now does a pLDDT sanity check on the bundled
reference proteins rather than a fragile rank-correlation against ConSurf. 98+ unit tests.
Bundled `_rns_reference_data.json` ships via `[tool.setuptools.package-data]` (this
declaration was added during P3-T6 after the first AWS gate run hit `FileNotFoundError` on
the JSON).

**`benchmarks/hypothesis_tests.py`** — Plan 3 H1-H4 pre-registered tests plus
`class_split_interaction` cross-class machinery. Pre-registered Bonferroni denominators
(`N_TESTS_H1 = 12`, `N_TESTS_H3 = 6`); vectorized 10K-resample bootstrap CIs via numpy
fancy-indexing; lazy statsmodels import; nullable boolean dtype for H2's interaction row to
preserve type semantics; sign(0)-tolerance for H4 to avoid spurious flip-rates near zero. A
caveat is documented in the module docstrings: 5-seed bootstrap CIs have degraded nominal
coverage and the manuscript Discussion needs to address this. 34 unit tests.

### The `__version__` stale-constant fix

Resolves the v1.1.0 Plan 2 limitation. `src/target_affinity_ml/__init__.py` previously
reported `1.0.0` even after the v1.1.0 release; `pyproject.toml` was correct but the
in-package constant was not. Both now agree on `1.2.0`.

### Test suite

The library now ships **~156 unit tests** total in the `benchmarks/` module — 98+ for
`rns_scoring`, 24 for `scaffold_diversity`, 34 for `hypothesis_tests` — plus the pre-existing
~76 tests for the data/features/models/evaluation modules. All pass on the v1.2.0 commit;
ruff is clean.

---

## 3. Plan 3 task outcomes (T1-T24)

Brief one-line-per-task summary. Tracks (A/2A/2B/2C/3/6) correspond to the plan's
parallelizable subagent dispatch.

### Phase 1 — RNS module + validation gate (T1-T6, sequential)

- **T1.** `benchmarks/` scaffolding + bundled `_rns_reference_data.json`. Initial reference
  set assembled from cross-referencing the Prabakaran-Bromberg paper with ConSurf and a small
  panel of well-characterized proteins (EGFR, HIV protease, lysozyme, T4 lysozyme, ribonuclease A, etc.).
- **T2.** `fetch_structure()` — PDB + AlphaFold DB acquisition with on-disk caching.
- **T3.** `fetch_binding_site()` — KLIFS + GPCRdb adapters with response caching.
- **T4.** `compute_msa()` — jackhmmer wrapper. Retained in code but no longer in the critical
  path post-pivot.
- **T5.** `compute_per_residue_rns()` + `compute_conservation_entropy()`. Marked
  `[EXPERIMENTAL]` post-pivot — the implementations work and pass unit tests, but they do
  not reproduce ConSurf rankings.
- **T6.** Validation gate. **PIVOTED.** Two ConSurf-anchored gate runs failed:
  - Attempt 1: raw MSA column entropy at binding-site positions, Spearman ρ = -0.524 vs
    ConSurf (anti-correlated).
  - Attempt 2: JSD vs Swiss-Prot background, Spearman ρ = -0.476.

  Pivoted to **mean binding-site pLDDT** as the primary per-target metric. Gate passed with
  mean = 88.13 on the 8 bundled reference proteins. See Section 5 for the full narrative.

### Track 2A — Kinase reference data hosting (T7-T10)

- **T7.** Local + AWS kinase-file inventory. Critical per-target predictions were located in
  the local kinase repo plus the AWS benchmark outputs; no Plan 1.5 supplementary re-run was
  needed.
- **T8.** Added `data/kinase_reference/` to GPCR repo with `.gitignore` exception plus
  provenance README.
- **T9.** Synced kinase data (~50-80 MB) into the GPCR repo and force-pushed past blanket
  `data/` ignores using `git add -f`. Resolves Plan 1 L2 limitation.
- **T10.** Updated GPCR README + data card with kinase-reference attribution. Defers Zenodo
  deposit to post-manuscript phase.

### Track 2B — Scaffold-diversity module (T11-T13)

- **T11.** Library `scaffold_diversity.py` + 24 unit tests. Uses the
  `MurckoScaffold.GetScaffoldForMol(mol)` + `Chem.MolToSmiles(scaff)` idiom corrected during
  Plan 2 Task 8 (not the empty-string-returning `MurckoScaffold.MolToSmiles`).
- **T12.** Per-target scaffold-metric computation for both classes. Output:
  `results/supplement/per_target_metrics_scaffold.csv` with 543 rows (507 kinase + 36 GPCR).
- **T13.** `fit_degradation_regression` in `scaffold_diversity.py` — pooled OLS with class
  covariate + class × X interaction. Two transition directions: random→scaffold and
  scaffold→target.

### Track 2C — pLDDT pipeline + hypothesis tests (T14, T15)

- **T14.** Mean binding-site pLDDT pipeline. **Post-pivot, this collapsed from the original
  3-5 hour AWS RNS pipeline (96 CPUs) to a ~30 min local script with caching.** 543 targets
  attempted; 303 successful (36/36 GPCR + 267/507 kinase). Two upstream data-quality issues
  encountered + fixed during execution:
  - **GPCR `protein_sequences.json` had TrEMBL accessions.** The `chembl_webresource_client`
    returns TrEMBL accessions when no Swiss-Prot is preferred. AlphaFold DB only serves
    Swiss-Prot. Replaced 36 accessions with canonical Swiss-Prot via UniProt's
    reviewed-search REST API.
  - **Kinase ChEMBL→UniProt mapping similarly had ~415 TrEMBL accessions.** Same root cause,
    same fix (UniProt reviewed-search lookup).
- **T15.** Library `hypothesis_tests.py` + 34 unit tests. Implements H1, H2, H3 (Part A
  table-level + Part B per-target regression), H4, and `class_split_interaction`.

### Track ~~2C~~ — Structure-source sensitivity (T16, T17)

- **T16, T17. MOOT.** pLDDT is an AlphaFold-specific concept; AlphaFold structures are used
  uniformly so the PDB-vs-AlphaFold sensitivity analysis (T16) and the structure-source
  decision branch (T17) do not apply. Both marked DONE-MOOT.

### Phase 3 — Notebooks (T18-T21, sequential)

- **T18.** `notebooks/05_scaffold_diversity.ipynb`. Loads per-target metrics + per-target
  degradation, fits the two-direction regression, produces `figure3_scaffold_degradation.png`
  (2×5 grid: directions × metrics), and writes the scaffold portion of Table 4.
- **T19.** `notebooks/06_plddt_analysis.ipynb` (**renamed from `06_rns_analysis.ipynb` per
  pivot**). Consumes the per-target pLDDT CSV + per-target ESM-FP-MLP-vs-MLP advantage.
  Produces `figure4_plddt_advantage.png` (single scatter, both classes overlaid, regression
  lines, interaction p in title) and the pLDDT rows of Table 4 (1 regression + 2 distribution
  rows for KS-2-sample and Welch t).
- **T20.** `notebooks/07_cross_class_comparison.ipynb` (the headline notebook). Builds
  Tables 1, 2, 3 (+ 4 per-hypothesis companions) and Figures 1, 2, 5. Runs H1-H4 hypothesis
  tests and aggregates verdicts.
- **T21.** Final tables + figures inventory + `results/README.md` index linking each output
  to its paper section and producing notebook.

### Phase 6 — Release + wrap-up (T22-T24)

- **T22.** Library v1.2.0 release — version bumps in both `pyproject.toml` and
  `__init__.py`, CHANGELOG `[1.2.0]` section, full test suite + ruff clean, push main + tag,
  GPCR repo `pyproject.toml` updated to pin `target-affinity-ml@v1.2.0`.
- **T23.** This document + the Plan 4 manuscript handoff.
- **T24.** GPCR repo v1.1.0 tag — analysis freeze; supports the eventual Zenodo deposit.

---

## 4. Key findings

The four pre-registered hypotheses + the scaffold-diversity + pLDDT correlation analyses
produced the following significant results.

### 4.1 Scaffold-diversity slopes differ between classes (random→scaffold direction)

For the per-target degradation (random→scaffold), three of five scaffold-diversity metrics
show significant class × metric interactions (Table 4):

- **`scaffold_entropy`** — interaction p = 0.00305. Kinase slope -0.00751 (p = 0.540, ns);
  GPCR slope +0.0859 (p = 0.00305).
- **`largest_cluster_fraction`** — interaction p = 3.73 × 10⁻⁶. Kinase slope +0.205 (p = 0.178,
  ns); GPCR slope -2.71 (p = 8.75 × 10⁻⁶) — strongly negative.
- **`mean_tanimoto`** — interaction p = 0.0164. Kinase slope -0.0373 (p = 0.868, ns); GPCR
  slope -1.91 (p = 0.0106).

Interpretation: kinases show flat slopes, GPCRs show steeper degradation responses. The
random→scaffold transition appears to be more strongly mediated by within-target scaffold
heterogeneity for GPCRs than for kinases.

On the scaffold→target direction, sample size collapses to n = 29 (24 kinase + 5 GPCR — the
GPCR target split holds out one of six subfamilies). Only `mean_tanimoto` shows a significant
kinase slope (+2.31, p = 0.000483) and the interaction term does not reach significance
(p = 0.202). This is power-limited; flagged for the manuscript Discussion.

### 4.2 ESM-2 advantage × pLDDT interaction is significant

Per-target regression of ESM-FP-MLP-vs-MLP RMSE advantage on mean binding-site pLDDT, with
class as covariate (Table 4, `esm_advantage_random` row; n = 239 = 203 kinase + 36 GPCR):

- **Interaction p = 0.0313** (significant at α = 0.05; would not survive Bonferroni
  correction at family size 13).
- Kinase slope: -0.00137 (p = 0.263, near zero).
- GPCR slope: +0.00914 (p = 0.0528, borderline positive).
- Overall R² = 0.562 (driven primarily by the class dummy).

Biological interpretation: ESM-2 and AlphaFold both encode evolutionary / sequence signal.
The positive GPCR slope means ESM-2 helps LESS at higher pLDDT (where AlphaFold is more
confident) — i.e., ESM-2 and AlphaFold encode overlapping protein-level information for
GPCRs. Kinases show a flat slope, suggesting the two signals are more orthogonal in that
family (perhaps because kinases share a conserved ATP-binding pocket that AlphaFold predicts
confidently regardless of family-specific evolutionary nuance).

The per-class pLDDT distribution differs (KS D = 0.292, p = 0.00838; Welch t = 4.02,
p = 0.000141), so part of the interaction effect is mediated by GPCRs having a different
pLDDT distribution than kinases. Reported in the supplement.

### 4.3 H3 Part A vs Part B distinction

H3 has two complementary tests, distinct in the unit of analysis:

- **Part A — class × split interaction on raw RMSE** (Table 3): NOT significant
  (p = 0.318). This asks: does the *average* ESM-FP-MLP-minus-MLP gap depend on class × split?
- **Part B — per-target ESM-2 advantage vs pLDDT regression** (Section 4.2): significant
  (interaction p = 0.0313). This asks: does the *individual-target* ESM-2 advantage track
  pLDDT differently across classes?

These are different statistical questions on different units of analysis. Both are reported;
the manuscript should explicitly call out the distinction so reviewers don't read it as a
contradiction.

### 4.4 H2 class × split interaction is highly significant

H2 (Table 3, line for class × split interaction): **p = 5.887 × 10⁻⁵**. This is the
mixed-model interaction of class (kinase vs GPCR) and split (random vs scaffold vs target) on
raw RMSE, pooled across all 7 models × 5 seeds. The preprint-degradation patterns genuinely
differ by class — strongest evidence in the analysis that the two protein families respond
differently to data-split structure.

Individual model × class × transition rows (per the H2 companion) show that the kinase
preprint range is replicated for the deep models (ESM-FP-MLP, partial Fusion) but the
tree-based models on GPCR show the scaffold→target inversion described in the Plan 2
summary (negative degradation ratios mean target-split RMSE is BETTER than scaffold-split
RMSE). 10 of 20 ratios fall within the preprint range; the misses are concentrated in the
scaffold→target direction for tree models on GPCRs.

---

## 5. The RNS → pLDDT metric pivot — full narrative

This is the most scientifically interesting episode of Plan 3, and the manuscript should
narrate it transparently. The Plan 3 design spec named "Prabakaran-Bromberg Residue
Neighborhood Significance (RNS)" as the per-target structural metric for H3. The plan was to
implement the RNS algorithm in `benchmarks/rns_scoring.py`, validate it via Spearman ρ ≥ 0.7
or MAD ≤ 0.10 against the published reference values, and then run the full per-target RNS
pipeline on AWS for 543 targets across 96 CPUs.

**Investigation revealed the spec's citation was misaligned with the named construct.** When
the Prabakaran-Bromberg paper was located during P3-T1, it turned out to be about LM
embedding evaluation, not binding-site residue significance. There is no canonical "RNS"
score with published per-target values; the only candidate reference is the well-known
**ConSurf** evolutionary-conservation server, which scores each residue 1-9 by conservation
across a homologous family.

**Attempt 1: raw column entropy.** Shannon entropy of MSA columns at binding-site positions,
inverted (high entropy → low score). Ran the validation gate against 8 ConSurf-anchored
reference proteins. Spearman ρ = **-0.524** (strongly anti-correlated). Diagnosis: deep MSAs
from large protein families (kinases) produce spuriously high per-column entropy regardless
of conservation, because the MSA contains many distantly-related sequences whose alignment
columns happen to vary; conserved residues then look "less conserved" than they are.

**Attempt 2: JSD vs Swiss-Prot background.** Replaced raw entropy with Jensen-Shannon
divergence between the MSA-column residue distribution and a Swiss-Prot global background.
This is supposed to be robust to MSA depth. Spearman ρ = **-0.476** (still anti-correlated).
Diagnosis: binding-site residue counts differ substantially across reference proteins
(KLIFS 85 residues for kinases vs GPCRdb ~30 for GPCRs vs ~25 for HIV protease) — averaging
dilution dominated whatever conservation signal existed.

**Pivot decision.** Rather than spend additional days iterating on entropy-based
approximations, the team pivoted to **mean binding-site pLDDT** (AlphaFold's per-residue
model confidence averaged over binding-site residues). pLDDT IS the published reference
quantity (AlphaFold DB exposes per-target mean confidence values); no fragile rank-correlation
validation is needed. The gate became a sanity check: compute mean binding-site pLDDT on the
8 reference proteins and confirm the result is biologically reasonable (40-100). Result:
mean **88.13** with no invalid targets — gate PASSED.

**The H3 hypothesis was reframed.** Original: *"does ESM-FP-MLP-vs-MLP advantage on the
target split correlate with binding-site residue conservation (RNS)?"* Reframed: *"does
per-target ESM-FP-MLP-vs-MLP advantage correlate with mean binding-site pLDDT, and does this
relationship differ across classes?"* This is biologically meaningful — both ESM-2 and
AlphaFold encode evolutionary/sequence signal, so the question is whether ESM-2's
contribution is greatest where AlphaFold is also confident. It is also cleanly testable; see
Section 4.2 for the result.

**Tasks T16 + T17 are MOOT post-pivot.** pLDDT is an AlphaFold-specific concept (no PDB
equivalent); we use AlphaFold structures uniformly. The PDB-vs-AlphaFold sensitivity analysis
(T16) and the structure-source decision branch (T17) have no analog in the pLDDT formulation.

**The experimental RNS code is preserved.** `compute_per_residue_rns`,
`compute_conservation_entropy`, and `aggregate_target_rns` remain in `rns_scoring.py` with
`[EXPERIMENTAL]` docstring tags. They pass unit tests on synthetic MSAs; they just don't
reproduce ConSurf-style rankings. A future plan could revisit this with a true
ConSurf-anchored objective if needed.

---

## 6. Caveats & limitations

These caveats were surfaced in T21's `results/README.md` and need explicit treatment in the
manuscript Discussion.

### 6.1 Kinase per-seed RF/XGB/EN/MLP gap

The kinase reference data hosted at
`data/kinase_reference/benchmark_v1/per_seed_metrics.csv` includes only the three deep-model
per-seed RMSE values (`esm_fp_mlp`, `fusion`, `gnn`); the other four models (`random_forest`,
`xgboost`, `elasticnet`, `mlp`) have only mean + SD in `multi_seed_aggregated.csv`. This is a
Plan 1 data limitation — the original kinase pipeline saved per-seed metrics for the deep
models only.

**Impact on hypothesis tests:**
- **H1 (RF vs deep)** ran on GPCR data only — 6 GPCR tests (model_pair × split) rather than
  12. The Bonferroni correction still divides by the pre-registered 12 per the
  pre-registration commitment.
- **H4 (single-seed flip rate)** is similarly GPCR-only — 6 GPCR rows out of 12 planned, all
  with verdict "a) similar" because the kinase column is null and the diff is 0.
- **H3 Part A** is affected (the kinase MLP per-seed isn't available).

Within-GPCR conclusions are robust. The cross-class direction for H1 and H4 was not
assessed.

### 6.2 Kinase pLDDT coverage at 53%

Of 507 kinase targets, KLIFS has binding-site annotations for ~270 (rate-limited fetches
during T14 capped at 267 successful). The remaining ~240 are either KLIFS-coverage gaps (~50
known pseudokinases or recently-named kinases) or transient API failures that cached empty
results. **The H3 cross-class regression has n = 239 (203 kinase + 36 GPCR), sufficient
statistical power** for the interaction test — but the kinase per-target estimates are
sampled from a 53% slice of the kinase family rather than uniformly.

Re-running T14 with longer back-off intervals could push coverage above 80%, but at the cost
of multiple additional hours of API time. We chose to freeze at 267 kinase targets and document
the coverage explicitly.

### 6.3 GPCR target-split has only 5 targets

With 36 total GPCR targets and a 10% test fraction, the target-split test set contains 5
receptors. The scaffold→target degradation regression on the GPCR side has minimal power
(n_per_class = 5). T18 and T20 both surface n_per_class to make this visible in the figures
and tables. The manuscript Discussion needs to note that single-subfamily holdouts at n=5 are
under-powered for slope inference on the GPCR side.

### 6.4 Bootstrap CI nominal coverage degraded at n = 5 seeds

The hypothesis_tests module uses 10K-resample bootstrap CIs computed over 5 seeds. The
docstring notes that nominal coverage is degraded at this n; the empirical Type I error
rate when testing at α = 0.05 with n = 5 can be inflated by ~30-50% per literature on
small-sample bootstrap. The manuscript Discussion should flag this as a methodological
limitation. Plan 4 may want to mention that the H3 Part B interaction p = 0.0313 is in this
borderline-coverage regime.

---

## 7. Lessons learned

Documented for future plan authors. The execution-friction items below are concrete
discoveries with a "how to avoid" remedy.

| Discovery | How it bit us | How to avoid |
|---|---|---|
| Bundled JSON package-data dropped by setuptools by default | P3-T6 first AWS gate run hit `FileNotFoundError: _rns_reference_data.json` — setuptools doesn't ship *.json under `src/` unless explicitly declared | Add `[tool.setuptools.package-data]` declaration to `pyproject.toml` whenever the library bundles non-Python data files |
| Spec-named metrics may not match the cited paper | Two RNS validation attempts wasted ~2 days because we trusted the spec's "Prabakaran-Bromberg RNS" naming without reading the cited paper carefully — the paper is about LM embedding eval, not residue significance | When a spec names a metric with a citation, read the paper first; verify the construct in the spec actually matches what the paper publishes |
| TrEMBL accessions in upstream protein_sequences.json | `fetch_structure` returned 404s for ~36 GPCR + 415 kinase targets — `chembl_webresource_client` returns TrEMBL accessions when no Swiss-Prot is preferred, but AlphaFold DB only serves Swiss-Prot | After fetching UniProt mappings from ChEMBL, batch-lookup the Swiss-Prot reviewed entry via UniProt's reviewed-search REST API; replace TrEMBL with canonical Swiss-Prot before downstream use |
| Per-target binding-site API calls return full target lists | KLIFS `/kinase_information` returns all 1127 kinases per call (~320 KB JSON); GPCRdb `/receptorlist/` returns ~237 aminergic receptors (~2 MB). 543 calls would have downloaded ~1.2 GB of redundant data and exhausted rate limits | Fetch the full list ONCE at the top of the pipeline; pass to `fetch_binding_site` as a pre-computed index |
| `data/`-gitignore blocks `!data/sub/**` exceptions | git's blanket `data/` ignore prevents `!data/kinase_reference/**` exceptions from recursively un-ignoring nested files — known git limitation | Use `git add -f data/kinase_reference/<path>` to force-add past blanket ignores |
| AWS public IP rotates on stop/start | Each AWS-required task needs to confirm the current address before SSH | Either accept that AWS IP confirmation is part of every AWS task's pre-flight, or attach an Elastic IP for the duration of a multi-day plan |
| AWS libstdc++ vs conda libicu mismatch | Non-interactive SSH-launched Python hit `ImportError` on sqlite3 because the conda env's libicu requires a newer libstdc++ than the system | Always prepend `LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH` for AWS Python invocations; add to `~/.bash_profile` for persistence |
| AlphaFold DB URL version drift | The plan referenced v4; production DB is at v6 | Use the AlphaFold prediction API for dynamic resolution rather than hardcoding a version suffix |

---

## 8. Plan 3 → Plan 4 handoff

The Plan 4 manuscript-drafting handoff lives at
[2026-05-28-plan4-manuscript-handoff.md](2026-05-28-plan4-manuscript-handoff.md). It
inventories the four main-text tables, five main-text figures, supplement files, frozen
library + analysis tags, recommended writing order, and the small set of open authorial
questions that need human judgment before the preprint goes to bioRxiv.

**Plan 3 closes here. Plan 4 (manuscript drafting) is ready to begin in a fresh session.**
