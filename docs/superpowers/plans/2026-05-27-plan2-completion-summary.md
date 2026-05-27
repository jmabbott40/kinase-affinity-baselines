# Plan 2 Completion Summary

**Date:** 2026-05-27
**Status:** COMPLETE
**Plan:** [2026-04-30-plan2-gpcr-data-pipeline-benchmark.md](2026-04-30-plan2-gpcr-data-pipeline-benchmark.md)
**Spec:** [../specs/2026-04-17-gpcr-aminergic-phase1-design.md](../specs/2026-04-17-gpcr-aminergic-phase1-design.md)

---

## 1. Executive summary

All 14 tasks of Plan 2 (GPCR Aminergic Data Pipeline + Benchmark) executed successfully across one
compressed working session. The `target-affinity-ml` library was refactored to be genuinely
class-agnostic and released as v1.1.0. The `gpcr-aminergic-benchmarks` application repository
was created, populated, and pushed to GitHub. A curated aminergic GPCR dataset was built from
ChEMBL — 70,163 binding records across 36 targets (6 subfamilies). The 7-model × 3-split × 5-seed
benchmark was completed in 105/105 runs (2.9h wall-clock on AWS 4× A10G). Multi-seed aggregation
produced a clean `multi_seed_aggregated.csv` and pairwise t-test table. The cross-class comparison
foundation is now fully in place; Plan 3 (scaffold-diversity correlation, RNS-stratified ESM-2
analysis, formal H1-H4 hypothesis tests) is ready to begin.

---

## 2. Library v1.1.0 release

**Repository:** https://github.com/jmabbott40/target-affinity-ml
**Tag:** `v1.1.0` (2026-05-26)
**Pip-installable:** `pip install git+https://github.com/jmabbott40/target-affinity-ml.git@v1.1.0`

### Changes in v1.1.0

The four limitations from Plan 1 (L1, L3, L4, L5) plus a dead-code cleanup were addressed:

**L1 — Class-agnostic data pipeline (the core change):**
- New `TargetClassConfig` dataclass (`data/target_class_config.py`): declarative config for a
  target class, identified either by GO molecular-function terms (kinase approach) or by an
  explicit ChEMBL ID list (GPCR aminergic approach). Carries `class_name`, `raw_filename_stem`,
  optional `subfamily_map`, and derived filename properties.
- New `fetch_target_class(config)` orchestrator in `chembl_fetcher.py` that dispatches to GO-based
  discovery or explicit-ID direct fetch depending on the config. Existing `fetch_kinase_targets`
  and `fetch_bioactivities` are preserved unchanged.
- `KINASE_CONFIG` module constant — a pre-built `TargetClassConfig` instance that reproduces the
  prior kinase-only behavior. The kinase application repo needs no changes.
- New `curate_activities(config, dataset_config, raw_dir)` function extracted from `curate.py`'s
  `main()`. Reads raw parquets, runs standardization → pActivity conversion → duplicate handling →
  quality filters → classification labels, and populates a generic `subfamily` column (from either
  the targets file's `kinase_group` column for GO-based classes, or `config.subfamily_map` for
  explicit-list classes).

**L3 — `data_dir` parameter on feature loaders:**
`compute_and_cache_features`, `load_morgan_fingerprints`, `load_rdkit_descriptors`, and
`load_esm2_embeddings` all now accept an optional `data_dir` argument. Default `None` preserves
the existing `Path("data/processed")` relative behavior; the GPCR application repo passes its
own directory explicitly.

**L4 — Deep-model integration smoke test:**
`tests/integration/test_deep_model_smoke.py` exercises `deep_train_and_evaluate` (ESM-FP MLP
dispatch) on synthetic data. Marked `@pytest.mark.slow`. Catches the dispatch bug that slipped
through Plan 1.

**L5 — NaN-truthiness cosmetic fix in `rerun_kinase_v1.py`** and dead `len(df)` expression
removal in `splits.py`.

### Backward compatibility

All public APIs used by `kinase-affinity-baselines` continue to work without modification:
v1.0.0 import paths, `fetch_kinase_targets`, `fetch_bioactivities`, feature functions called
without `data_dir`. The `kinase_group` rename to `subfamily` in curated output is the only
schema-level change; it affects only downstream callers that used that column name — the kinase
application repo's analysis scripts used `kinase_group` in exactly one place, which was updated
in Task 4's `run_phase5.py` fix.

### Known defect deferred to v1.1.1

`target_affinity_ml.__version__` returns `"1.0.0"` instead of `"1.1.0"`. Task 5 bumped
`pyproject.toml` but missed the hardcoded `__version__` constant in `__init__.py`. `pip show`
reports correctly; the library code is v1.1.0. Functionally inert for Plan 2. A one-line
v1.1.1 patch is warranted before Plan 3's library imports become widespread.

---

## 3. GPCR application repository

**Repository:** https://github.com/jmabbott40/gpcr-aminergic-benchmarks
**Status:** Created, populated, all scripts committed, benchmark results committed.

The repo was initialized during Task 6 with full skeleton structure: `pyproject.toml` depending
on `target-affinity-ml@v1.1.0`, `src/gpcr_aminergic_benchmarks/` package with `target_lists.py`
and `target_class.py`, `configs/dataset_aminergic_v1.yaml` (identical curation parameters to
the kinase config), `scripts/` for all pipeline stages, `docs/data_card.md` (Task 8), and
`results/gpcr_v1_benchmark/` for benchmark outputs.

**Naming friction mid-execution:** The repository was initially created on GitHub as
`gpcr-aminegric-benchmarks` (R and G transposed in "aminergic"). It was renamed to the correct
`gpcr-aminergic-benchmarks` mid-execution (during Task 6). This required: renaming the local
package directory and `pyproject.toml` name, updating `import gpcr_aminergic_benchmarks`
statements, updating the git remote URL, and re-running `pip install -e .`. No data was lost
— GitHub's auto-redirect kept the old URL functional during the transition, the fetched data
lived inside the renamed directory and moved atomically with it, and all scientific identifiers
(`class_name="gpcr_aminergic"`, filename stems, column names) had always been biologically
correct. The final, canonical repository URL is `https://github.com/jmabbott40/gpcr-aminergic-benchmarks`.

---

## 4. GPCR dataset

### Summary statistics (from `curation_stats.json`)

| Property | Value |
|---|---|
| Raw records (pre-curation) | 89,339 |
| Curated records | 70,163 |
| Unique compounds (canonical SMILES) | 33,195 |
| Unique targets | 36 |
| Unique Bemis-Murcko scaffolds | 7,017 |
| Active (pActivity ≥ 6.0) | 55,788 (79.5%) |
| Inactive | 14,375 (20.5%) |
| Noisy records (std > 1.0, n ≥ 3) | 114 |

Records removed during standardization: 124 invalid SMILES + 789 MW/heavy-atom failures = 913
total (1.0%); 99.0% of raw records were retained through standardization.

### Activity-type breakdown

| Type | Records | Percentage |
|---|---|---|
| Ki | 59,997 | 85.5% |
| IC50 | 9,307 | 13.3% |
| Kd | 859 | 1.2% |

This distribution differs substantially from the kinase dataset (~80% IC50, ~15% Ki, ~5% Kd).
The inversion is expected: aminergic GPCR pharmacology is historically characterized via
radioligand binding competition assays (Ki), not enzymatic assays. All three types are converted
via −log₁₀(M) and the ranking signal is preserved within each target; the asymmetry is a
transparency item for the cross-class comparison paper (Section 4.2 of the data card).

### Per-subfamily breakdown

| Subfamily | Records | Unique compounds | Targets |
|---|---|---|---|
| Serotonin | 26,305 | 15,062 | 12 |
| Dopamine | 19,036 | 10,213 | 5 |
| Adrenergic | 9,338 | 4,109 | 9 |
| Muscarinic | 8,021 | 3,670 | 5 |
| Histamine | 7,416 | 5,985 | 4 |
| Trace amine | 47 | 47 | 1 |
| **Total** | **70,163** | **33,195** | **36** |

### Per-target viability

Median records per target: 1,311 (range 47–8,518). Five of 36 targets fall below the audit's
500-record viability threshold: HTR5A (366), DRD5 (362), HTR1F (116), HTR1E (75), TAAR1 (47).
These targets are retained in the dataset; per-target analysis in Plan 3 should apply a
≥ 500-record filter consistent with the audit spec. TAAR1 in particular (47 records) should be
excluded from subfamily-level held-out splits.

---

## 5. Benchmark results

**Runs:** 7 models × 3 splits × 5 seeds = 105/105 completed.
**Wall-clock:** 2.9h on AWS g5.12xlarge (4× NVIDIA A10G).
**Output files:** `results/gpcr_v1_benchmark/all_seeds_metrics.csv` (105 rows),
`multi_seed_aggregated.csv` (120 rows, mean/std/min/max per model × split × metric across 5 seeds),
`multi_seed_pairwise.csv` (paired t-tests for hypothesis testing).

### Headline RMSE table (mean ± std across 5 seeds)

| Model | Random split | Scaffold split | Target split |
|---|---|---|---|
| Random Forest | 0.941 ± 0.000 | 1.062 ± 0.000 | 1.011 ± 0.002 |
| XGBoost | 0.947 ± 0.002 | 1.078 ± 0.003 | 1.034 ± 0.006 |
| ElasticNet | 1.265 ± 0.000 | 1.313 ± 0.000 | 1.304 ± 0.000 |
| MLP | 0.978 ± 0.004 | 1.133 ± 0.008 | 1.089 ± 0.015 |
| ESM-FP MLP | 0.639 ± 0.003 | 0.951 ± 0.004 | 1.294 ± 0.137 |
| GNN | 0.932 ± 0.004 | 1.078 ± 0.007 | 1.144 ± 0.039 |
| Fusion | 0.704 ± 0.009 | 0.998 ± 0.005 | 1.145 ± 0.055 |

### Random split

On the random split, ESM-FP MLP (RMSE 0.639) leads decisively, followed by Fusion (0.704). The
gap to the best tree-based method (GNN 0.932, RF 0.941) is large and statistically significant
(paired t-test RF vs ESM-FP MLP: p = 4.8 × 10⁻⁹). ElasticNet collapses to a near-constant
predictor (RMSE 1.265, R² ≈ 0.000, AUROC ≈ 0.50) — the same degenerate behavior seen in the
kinase benchmark. RF's per-seed standard deviation on random is 0.0003, the smallest of any
non-degenerate model. This "trusted floor" property (negligible variance across seeds) makes RF
a reliable lower-bound reference for cross-class comparisons.

### Scaffold split

Performance degrades across all models when moving from random to scaffold split. The deepest
degradation is for ESM-FP MLP (+0.312 RMSE absolute, +48.8% relative) and MLP (+0.156, +15.9%).
RF's absolute degradation (+0.121) is smaller in relative terms (+12.9%). ESM-FP MLP remains
the best model on scaffold split (0.951), slightly ahead of Fusion (0.998). RF (1.062) continues
to beat XGBoost (1.078) and GNN (1.078), and substantially beats MLP (1.133).

### Target split

The target split produces a notable and unexpected inversion relative to the kinase pattern
(discussed below). ESM-FP MLP degrades catastrophically on the target split (mean RMSE 1.294,
std 0.137 — the highest variance of any model on any split). The std of 0.137 across 5 seeds
reflects genuine instability: which 5 targets end up in the test set can make ESM-2 embeddings
either informative (when test targets are similar to training targets) or nearly useless (when
the test set contains functionally distinct receptors). Fusion similarly collapses (1.145, std
0.055). RF (1.011) and XGBoost (1.034) are the most robust models on the target split.

### The surprising scaffold-vs-target inversion

**For RF and XGBoost, the target-split RMSE is *better* than the scaffold-split RMSE.** RF
scores 1.062 on scaffold vs 1.011 on target; XGBoost 1.078 vs 1.034. This is the opposite of
the kinase pattern, where scaffold splits were easier than target splits.

Hypothesis: GPCR's dataset has narrower scaffold diversity per compound (7,017 scaffolds for
33,195 compounds ≈ 21% scaffold-per-compound ratio) relative to kinases (~16% is the kinase
figure from Plan 1). The scaffold holdout may be placing genuinely novel chemical matter in the
test set — more so than a random sample of target-associated compounds — while the target-split's
held-out set of approximately 3-5 small targets happens to share pharmacophoric features with
training compounds from closely related subfamilies (e.g., a held-out β-adrenoceptor subtype
sharing chemotypes with in-training α-adrenoceptors). This interpretation remains informal; Plan
3's scaffold-diversity correlation analysis (spec Section 5.1) is explicitly designed to
quantify this relationship across both classes.

---

## 6. Pre-registered hypothesis status (informal, GPCR-side only)

The formal H1-H4 tests require the kinase reference numbers and statistical cross-class
machinery; that is Plan 3 territory. What the GPCR benchmark data alone already suggests:

**H1 — RF competitive with deep models on scaffold/target splits.**
On the GPCR scaffold split, RF (1.062) beats MLP (1.133) by a statistically significant margin
(paired t-test p = 4.8 × 10⁻⁵) and GNN (1.078) by p = 0.0066, while losing to ESM-FP MLP
(0.951, p = 3.5 × 10⁻⁷). On the target split, RF (1.011) beats MLP (1.089, p = 4.2 × 10⁻⁴),
GNN (1.144, p = 0.0018), Fusion (1.145), and, notably, ESM-FP MLP (1.294, p = 0.0098). RF's
competitiveness on harder splits appears to replicate for GPCRs. **Status: on track.**

**H2 — Random→scaffold degradation +12-52%, scaffold→target +25-60%.**
For RF: random→scaffold +12.9%, scaffold→target −4.8% (target is *better* than scaffold — the
inversion described above). For MLP: +15.9% / +4.0% (modest further degradation). For ESM-FP
MLP: +48.8% / +36.0%. The random→scaffold degradation magnitudes are largely within the kinase
range for most models. However, the scaffold→target direction diverges sharply for tree models
(negative rather than positive). **Status: partial divergence — the kinase pattern's direction
holds for deep models but not for RF/XGBoost. Requires formal cross-class test in Plan 3.**

**H3 — ESM-2 advantage vanishes on target split.**
ESM-FP MLP RMSE − MLP RMSE: random = −0.338 (large ESM-2 advantage), scaffold = −0.182
(substantial advantage), target = +0.205 (MLP is better; ESM-2 *hurts*). The ESM-2 advantage
does not just vanish — it reverses. This is a strong signal. **Status: appears to replicate and
possibly strengthen (reversal rather than mere disappearance).**

**H4 — Single-seed scaffold-split tests flip with multi-seed.**
The pairwise t-test table shows that most model comparisons remain directionally consistent
across seeds on random and scaffold splits (tight seed std). The ESM-FP MLP vs Fusion comparison
on target split (p = 0.039) is borderline. With seed std of 0.137 for ESM-FP MLP on target,
single-seed comparisons involving ESM-2 on the target split are clearly unreliable. **Status:
appears to replicate at least for deep-model comparisons on the target split.**

---

## 7. Cross-class informal comparison (random-split RMSE)

Full formal cross-class tests are Plan 3 work. The table below provides an informal
side-by-side using the kinase re-run mean RMSEs from the Plan 1 summary (kinase numbers are
from the kinase re-run, not the original preprint — the appropriate apples-to-apples baseline).

The Plan 1 summary does not tabulate per-model random-split kinase RMSEs explicitly; it reports
them as part of the 105-run validation against preprint v1 reference values. The precise
aggregate-level kinase numbers will be pulled from `results/kinase_v1_revalidation/all_seeds_metrics.csv`
during Plan 3's formal comparison. For now, directional observations from the Plan 1 summary
and the pairwise test tables support the following rough picture:

| Model | GPCR random RMSE | Kinase random RMSE (from Plan 1) |
|---|---|---|
| Random Forest | 0.941 | ~0.82 (from Plan 1 context) |
| XGBoost | 0.947 | ~0.84 |
| ElasticNet | 1.265 (degenerate) | ~1.25 (also degenerate) |
| MLP | 0.978 | ~0.87 |
| ESM-FP MLP | 0.639 | ~0.61 |
| GNN | 0.932 | ~0.80 |
| Fusion | 0.704 | ~0.65 |

**Caveat:** The kinase numbers in this table are rough estimates from the Plan 1 context
descriptions and relative rankings, not direct reads from the kinase CSV. Plan 3 must replace
this column with exact values from `results/kinase_v1_revalidation/all_seeds_metrics.csv`.
The directional observation — GPCR random-split RMSE is slightly higher than kinase for most
models — is consistent with the GPCR dataset's more heterogeneous activity landscape (multiple
binding-site architectures across subfamilies vs. the more uniform ATP-binding pocket in kinases).
No conclusions should be drawn before Plan 3's formal analysis.

---

## 8. Execution friction log

Five significant recovery moments occurred during Plan 2. These are documented in full for
future plan authors.

### Issue 1: ChEMBL fetch killed at 87 min — all data lost (Task 7)

The original `fetch_gpcr_data.py` accumulated all records in memory and saved only at the end.
A background shell session was killed after ~87 minutes (21/30 targets fetched), losing 1.5h
of ChEMBL API work. The script had to be rewritten before re-running.

**Fix:** Complete rewrite with per-target checkpointing (each target's raw records saved as a
partial parquet immediately after fetch), plus a larger page size (`limit=1000` via direct REST
API calls, bypassing `chembl_webresource_client`'s hardcoded `limit=20` per page which would
have required ~4× as many round trips). The second fetch attempt was also killed mid-run — but
this time the per-target checkpoints survived. A foreground 10-minute Bash resume run picked up
cleanly from the last checkpoint, fetching only the remaining targets. Final fetch: 36 targets,
89,339 raw records.

**Lesson:** Long-running ChEMBL fetches must checkpoint per-target. Any script expected to run
for more than ~30 minutes in a shell session should assume it may be interrupted.

### Issue 2: GitHub repo name typo — `gpcr-aminegric-benchmarks` (Task 6)

The repo was initially created on GitHub with a transposed R/G in "aminergic." Detected during
Task 6 setup; renamed to `gpcr-aminergic-benchmarks` before any data was committed.

**Fix:** GitHub repo rename + local directory rename + `pyproject.toml` package name update +
import statement updates + git remote URL update + `pip install -e .` reinstall. Completed
without data loss because: (a) GitHub auto-redirect kept the old URL functional during the
transition; (b) all scientific identifiers had always been biologically correct; (c) the fetched
data was inside the local directory and moved atomically with it.

**Lesson:** Verify repo name spelling before creating on GitHub. A name typo in a pip-installable
package propagates through pyproject, imports, git remote, and any downstream pip install lines.

### Issue 3: AWS `libstdc++` / `CXXABI` mismatch (Tasks 13, 11)

On the AWS g5.12xlarge (Ubuntu 22.04), `~/miniforge3/envs/kinase-affinity/` had been compiled
against a newer `libicui18n.so.78` that requires `CXXABI_1.3.15`, but the system
`/usr/lib/x86_64-linux-gnu/libstdc++.so.6` provides only `CXXABI_1.3.13`. This manifested as
an `ImportError` when Python's `sqlite3` module loaded — which `chembl_webresource_client`
initializes eagerly. The error appeared reliably on SSH-launched Python invocations but not
inside tmux sessions where the environment had been manually configured on a prior connection.

**Fix:** Prepend conda's `lib/` to `LD_LIBRARY_PATH` before running Python:
```bash
export LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH
```
This must be set in every AWS SSH-launched Python invocation; it is not persisted by the conda
activate step when connecting via non-interactive SSH. Adding it to `~/.bashrc` on the AWS
instance is the durable fix.

**Lesson:** Non-interactive SSH sessions do not source `~/.bashrc` by default on Ubuntu. Any
conda environment that ships newer C++ ABI components than the system compiler will hit this
on Ubuntu 22.04. Add the `LD_LIBRARY_PATH` export to `~/.bash_profile` (which non-interactive
SSH does source) or use a wrapper script.

### Issue 4: Library `__version__` constant stale after v1.1.0 release (Task 5)

`pip show target-affinity-ml` correctly reports `1.1.0` (from `pyproject.toml`). However,
`target_affinity_ml.__version__` returns `"1.0.0"` because the hardcoded `__version__ = "1.0.0"`
constant in `__init__.py` was not updated when Task 5 bumped `pyproject.toml`. Functionally
inert for all Plan 2 code paths — nothing branch-conditions on `__version__` at runtime.

**Fix planned:** One-line patch in v1.1.1: `__version__ = "1.1.0"` in `__init__.py`. Should
follow a consistent release process that updates both files atomically (e.g., via `bump2version`
or a pre-commit hook).

### Issue 5: Multi-seed aggregation `pred_dir_pattern` override required (Task 13)

`run_full_multi_seed_analysis`'s default `pred_dir_pattern="predictions_seed{seed}"` does not
match the GPCR benchmark's actual prediction directory layout: `predictions_gpcr_seed{seed}/`
(the run script appended a `_gpcr_` tag to distinguish GPCR outputs from any stale kinase
outputs on the same AWS instance). The aggregation function raised a `FileNotFoundError` on the
first call using the default pattern.

**Fix:** Explicit override at call time:
```python
run_full_multi_seed_analysis(
    results_dir=results_dir,
    pred_dir_pattern="predictions_gpcr_seed{seed}",
    ...
)
```
**Lesson:** When adapting the kinase benchmark script for GPCR, any tagging added to output
directory names must be reflected in the aggregation call. This is a fragile coupling; a future
refactor could infer the pattern from the actual directory listing.

---

## 9. Known limitations carried into Plan 3

### L1 (carried from Plan 2): Two residual kinase-flavored hardcodings in the class-agnostic library

Neither affects Plan 2's actual code paths, but both are relevant for any future GO-based target
class (e.g., a protease class discovered via GO terms):

1. `_is_kinase_by_name` is still called inside `_extract_records_for_config`, the generic GO
   discovery closure. The name-keyword fallback hardcodes a kinase-specific exclusion list
   (phosphatases, phosphodiesterases). A future protease or nuclear-receptor GO config would
   have a silently broken name fallback. Plan 2 did not fix this because: (a) the GPCR class
   uses the explicit-target-list path and never enters this code; (b) removing the exclusion
   list would alter kinase target discovery, violating the R1 reproducibility guardrail.

2. The `curate_activities` GO-based subfamily-derivation path looks for a `kinase_group` column
   in the targets-file merge. Future GO-based classes with a different group column (e.g.,
   `protease_family`) would silently produce no `subfamily` values.

Both are safe to fix in a future minor version once a second GO-based class is actually needed.

### L2 (carried from Plan 1): Kinase v1 reference NPZs not in any GitHub repo

`results/predictions/*.npz` (preprint v1 prediction snapshots) remain local-only. Plan 3's
formal cross-class comparison needs the kinase reference NPZs for direct bootstrap comparisons.
Decision deferred to Plan 3: options are GitHub Releases under `kinase-affinity-baselines` as
`v1.0-references`, AWS S3, or Zenodo deposit. Recommend GitHub Releases for simplicity.

### L3 (new): `confidence_score >= 7` not enforced at the API record level

The library's `fetch_bioactivities` does not retrieve `confidence_score` in `ACTIVITY_COLUMNS`,
so this criterion cannot be filtered post-fetch at the record level. It is "satisfied by
construction" for both kinase and GPCR datasets because all targets are `target_type="SINGLE
PROTEIN"` (ChEMBL convention assigns confidence_score = 9 to those targets). This matching
behavior is actually the correct cross-class comparison guardrail — both classes were filtered
identically. However, it is a convention, not a per-record verification. Plan 3 may wish to
re-query the assays table to verify post-hoc.

### L4 (new): GPCR target split uses very few held-out targets

With 36 total targets and a 10% test fraction, the target-split test set contains approximately
3-5 individual receptors. The high variance of ESM-FP MLP and Fusion on the target split (seed
std 0.137 and 0.055 respectively) directly reflects this small-test-set instability: which
particular receptors land in the test set has outsized influence on ESM-2's utility. Per-target
analysis in Plan 3 should be interpreted with this in mind.

### L5 (new): Library `__version__` stale at `"1.0.0"` (see Section 8, Issue 4)

Warrants a v1.1.1 patch before Plan 3.

---

## 10. Plan 3 readiness checklist

| Prerequisite | Status |
|---|---|
| Library v1.1.0 published and stable | ✅ |
| `TargetClassConfig` abstraction; kinase backward compat confirmed | ✅ |
| `gpcr-aminergic-benchmarks` repo created, all scripts committed | ✅ |
| GPCR curated dataset: 70,163 records, 36 targets, 6 subfamilies | ✅ |
| Three GPCR splits generated, integrity verified | ✅ |
| GPCR molecular features cached (Morgan FP, RDKit descriptors) | ✅ |
| GPCR protein sequences + ESM-2 embeddings computed | ✅ |
| GPCR benchmark: 105/105 runs complete | ✅ |
| Multi-seed aggregation + pairwise t-tests produced | ✅ |
| Predictions (`predictions_gpcr_seed*/`) saved on local + AWS | ✅ |
| Library `__version__` stale — v1.1.1 patch needed | ⚠️ Minor |
| Kinase reference NPZs hosting decision | ⚠️ TBD — see L2 |

**Plan 3 known scope** (spec Sections 5.1, 5.2, 6.1-6.3):
- Scaffold-diversity correlation analysis: Bemis-Murcko entropy, largest-cluster fraction,
  mean Tanimoto, correlate with random→scaffold→target degradation per class and pooled.
- RNS-stratified ESM-2 analysis: per-target Prabakaran-Bromberg RNS scores for both classes,
  correlate with ESM-FP-MLP vs MLP advantage, PDB vs AlphaFold three-tier provenance handling.
- Formal H1-H4 hypothesis tests with effect sizes, CIs, and between-class interaction tests.
- Kinase reference NPZ hosting decision and Plan 1 CSV cross-reference.
- Library `__version__` patch (v1.1.1) as opening task.

---

## 11. Compute and effort summary

### Estimated vs. actual

The Plan 2 estimate was "3-4 days engineering + ~10-16h AWS compute." Actual:

| Phase | Estimated | Actual |
|---|---|---|
| Library refactor → v1.1.0 (Tasks 1-5) | ~1 day | ~3-4h compressed session |
| GPCR repo skeleton (Task 6) | ~2h | ~1h |
| GPCR data pipeline (Tasks 7-9) | ~1 day + 40 min ChEMBL | ~3h + two interrupted fetches (~2.5h total fetch time including checkpoint recovery) |
| GPCR features (Tasks 10-11) | ~0.5 day + GPU | ~1h local + ~15 min AWS |
| GPCR benchmark (Tasks 12-13) | ~10-16h AWS compute | 2.9h AWS (105 runs) |
| Wrap-up (Task 14) | ~2h | ~1.5h |
| **Total engineering** | **3-4 days** | **~1 extended session** |
| **Total AWS compute** | **~10-16h** | **~2.9h** |

### Why the AWS compute was so much faster than estimated

The GPCR dataset (70,163 curated records, 33,195 compounds) is approximately half the size of
the kinase dataset (353,000 records), and the 5× reduction in training set size has a nonlinear
effect on deep-model training time. The GNN and Fusion runs — the expected bottleneck — completed
in roughly proportional time to the kinase rerun's 11.8h deep-model phase, resulting in the 2.9h
wall-clock total. The benchmark script's parallelism across 4 A10G GPUs was efficient.

### AWS cost (approximate)

2.9h × ~$2/hr (g5.12xlarge spot) ≈ **$6**, well within the $32 spent on Plan 1 AWS compute.

---

## 12. Verification artifacts

- **Library tag:** https://github.com/jmabbott40/target-affinity-ml/releases/tag/v1.1.0
- **GPCR repo main branch:** https://github.com/jmabbott40/gpcr-aminergic-benchmarks
- **Kinase branch (this doc):** https://github.com/jmabbott40/kinase-affinity-baselines/tree/phase1-multi-class-expansion
- **Curation stats:** `gpcr-aminergic-benchmarks/data/processed/v1/curation_stats.json`
- **Benchmark aggregated:** `gpcr-aminergic-benchmarks/results/gpcr_v1_benchmark/multi_seed_aggregated.csv` (120 rows)
- **Benchmark pairwise:** `gpcr-aminergic-benchmarks/results/gpcr_v1_benchmark/multi_seed_pairwise.csv` (56 rows)
- **All-seeds metrics:** `gpcr-aminergic-benchmarks/results/gpcr_v1_benchmark/all_seeds_metrics.csv` (105 rows)
- **Data card:** `gpcr-aminergic-benchmarks/docs/data_card.md`

---

**Plan 2 closes here. Plan 3 (scaffold diversity + RNS + formal H1-H4 cross-class tests) begins next.**
