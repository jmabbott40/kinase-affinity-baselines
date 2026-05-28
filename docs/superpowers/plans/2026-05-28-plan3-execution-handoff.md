# Plan 3 Execution Handoff — Track 2B onward

**Purpose:** Everything a fresh Claude Code session needs to resume Plan 3 (GPCR Cross-Class Analysis) from the **Track 2B starting line** (P3-T11 onward). Phase 1 (Tasks 1-6) is complete with a pivoted metric. Track 2A (Tasks 7-10) is complete. 14 tasks remain.

This document is the clean-context bridge to a new session — read it first.

---

## START HERE — paste this into a fresh session

```
Resume Plan 3 from Track 2B with subagent-driven development.

Handoff context: /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-05-28-plan3-execution-handoff.md
Plan: /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-05-27-plan3-cross-class-analysis.md
Spec: /Users/joshuaabbott/mlproject/docs/superpowers/specs/2026-05-27-plan3-cross-class-analysis-design.md

Read the handoff context FIRST (it documents the Phase 1 metric pivot and
modifications from the original plan), run the pre-flight check, then invoke
superpowers:subagent-driven-development to execute the remaining 14 tasks
(P3-T11 through P3-T24).
```

That's it. The new session reads this file, runs the pre-flight script below, and proceeds.

---

## Pre-flight environment check

Save this as `/tmp/plan3_resume_preflight.sh` and execute. It confirms the environment matches the handoff state.

```bash
#!/usr/bin/env bash
# Plan 3 Track 2B resume pre-flight
set -u
echo "=========================================="
echo "  Plan 3 Resume Pre-Flight"
echo "=========================================="
PASS=0; FAIL=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ✅ $1"; PASS=$((PASS+1)); else echo "  ❌ $1"; FAIL=$((FAIL+1)); fi; }

echo
echo "--- Repositories ---"
check "kinase repo exists"       "test -d /Users/joshuaabbott/mlproject/.git"
check "library repo exists"      "test -d /Users/joshuaabbott/target-affinity-ml/.git"
check "GPCR repo exists"         "test -d /Users/joshuaabbott/gpcr-aminergic-benchmarks/.git"
check "kinase on phase1 branch"  "cd /Users/joshuaabbott/mlproject && git branch --show-current | grep -q phase1-multi-class-expansion"
check "library on main"          "cd /Users/joshuaabbott/target-affinity-ml && git branch --show-current | grep -q main"
check "GPCR on main"             "cd /Users/joshuaabbott/gpcr-aminergic-benchmarks && git branch --show-current | grep -q main"

echo
echo "--- Library state ---"
LIB_HEAD=$(cd /Users/joshuaabbott/target-affinity-ml && git rev-parse HEAD)
check "library local has 10+ Plan-3 commits since v1.1.0" \
  "test \$(cd /Users/joshuaabbott/target-affinity-ml && git rev-list --count v1.1.0..main) -ge 10"
check "library plan3-development branch on origin"        \
  "git ls-remote https://github.com/jmabbott40/target-affinity-ml.git plan3-development | grep -q refs/heads/plan3-development"
check "library main still at v1.1.0 (Task 22 will tag v1.2.0)" \
  "test \"\$(git ls-remote https://github.com/jmabbott40/target-affinity-ml.git refs/tags/v1.1.0 | awk '{print \$1}')\" != \"\""

echo
echo "--- Phase 1 + Track 2A artifacts on disk ---"
check "RNS module + benchmarks/ scaffold"                  "test -f /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/benchmarks/rns_scoring.py"
check "Bundled RNS reference data JSON"                    "test -f /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/benchmarks/_rns_reference_data.json"
check "Validation gate CLI script"                         "test -f /Users/joshuaabbott/target-affinity-ml/scripts/run_rns_validation_gate.py"
check "Gate pass result archived locally"                  "test -f /Users/joshuaabbott/target-affinity-ml/results/validation_gate_pass_plddt.csv"
check "Kinase reference data in GPCR repo"                 "test -d /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/benchmark_v1/per_target"
check "Per-target metric CSVs (H3 critical artifact)"     "test \$(ls /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/benchmark_v1/per_target/per_target_*.csv | wc -l) -ge 18"

echo
echo "--- Python environment (use 3.11 conda env) ---"
CONDA_PY=/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
check "kinase-affinity conda env exists"          "test -x \$CONDA_PY"
check "Python is 3.11"                            "\$CONDA_PY --version 2>&1 | grep -q 'Python 3.11'"
check "library importable"                        "\$CONDA_PY -c 'from target_affinity_ml.benchmarks.rns_scoring import compute_binding_site_plddt'"
check "biopython available"                       "\$CONDA_PY -c 'import Bio'"
check "statsmodels available"                     "\$CONDA_PY -c 'import statsmodels'"
check "freesasa available"                        "\$CONDA_PY -c 'import freesasa' || pip show freesasa >/dev/null 2>&1"

echo
echo "--- AWS (per user: kept running for other work) ---"
AWS_KEY=/Users/joshuaabbott/Downloads/jma_key.pem
# IMPORTANT: AWS public IP changes on each stop/start; current address at handoff is below.
# If unreachable, the user has restarted and assigned a new IP — ask them for the new address.
AWS_HOST=ubuntu@ec2-3-144-163-211.us-east-2.compute.amazonaws.com
check "AWS SSH key present"                        "test -f \$AWS_KEY"
if ssh -i \$AWS_KEY -o ConnectTimeout=10 \$AWS_HOST 'echo ok' >/dev/null 2>&1; then
    echo "  ✅ AWS instance reachable at current handoff address"
    PASS=\$((PASS+1))
else
    echo "  ⚠️  AWS instance NOT reachable at handoff address"
    echo "     User said AWS is up for other work — ask user for the current public IP."
fi

echo
echo "=========================================="
echo "  Pre-flight: \$PASS passed, \$FAIL failed"
echo "=========================================="
if [ \$FAIL -gt 0 ]; then
    echo "  Resolve failures before executing Track 2B."
    exit 1
fi
echo "  Environment ready. Resume from P3-T11."
```

---

## Project state at handoff (2026-05-28)

### What's complete

**Phase 1 — RNS module + validation gate (Tasks 1-6):** complete.

- Built the `benchmarks/` module: `rns_scoring.py` (565+ lines), `_rns_reference_data.json`, full unit-test coverage (~98 tests passing).
- Functions implemented: `fetch_structure` (PDB + AlphaFold DB with caching), `fetch_binding_site` (KLIFS + GPCRdb adapters), `compute_msa` (jackhmmer wrapper, kept though no longer used), `compute_per_residue_rns` (Prabakaran-Bromberg-style algorithm, marked EXPERIMENTAL), `compute_conservation_entropy` (fallback, marked EXPERIMENTAL), `aggregate_target_rns` (marked EXPERIMENTAL), `compute_binding_site_plddt` (the new primary metric), `validation_gate`.
- **Validation gate PASSED** with the pivoted **mean binding-site pLDDT** metric: mean = 88.13 across 8 reference proteins, no invalid targets.

**Track 2A — Kinase reference data hosting (Tasks 7-10):** complete.

- `gpcr-aminergic-benchmarks/data/kinase_reference/` populated with:
  - `features/morgan_fp.npz` (13 MB)
  - `features/esm2_embeddings.npz`
  - `features/smiles_index.json` (12 MB)
  - `features/target_index.json`
  - `curated_activities.parquet` (13 MB; 353K kinase records)
  - `splits/{random,scaffold,target}_split.json` (3 × 2.6 MB)
  - `benchmark_v1/multi_seed_aggregated.csv`
  - `benchmark_v1/per_seed_metrics.csv`
  - `benchmark_v1/per_target/per_target_<model>_<split>.csv` × 21 files (per-model × per-split per-target metrics — **the H3 cross-class correlation reads these**)
- README + data card updated documenting provenance + Zenodo deferral.

### What's NOT complete (14 tasks remaining)

| Track | Tasks | Status |
|---|---|---|
| **2B Scaffold-diversity** | T11, T12, T13 | pending — local-only work |
| **2C RNS pipeline (modified)** | T14, T15 | pending — T14 simplified per pLDDT pivot |
| **~~2C Structure-source~~** | ~~T16, T17~~ | **MOOT** per pLDDT pivot (no PDB-vs-AF needed) |
| **Phase 3 Notebooks** | T18, T19, T20, T21 | pending — sequential |
| **Phase 6 Release** | T22, T23, T24 | pending — sequential |

### Critical discoveries during Phase 1 execution

These shape the remaining work and should be encoded into any new session's mental model:

#### 1. Metric pivot: pLDDT, not RNS

The original spec referenced "Prabakaran-Bromberg RNS." Investigation during P3-T1 revealed the actual Prabakaran-Bromberg 2026 paper is about **LM embedding evaluation**, not binding-site residue significance. We attempted two approximations:

1. **Raw column entropy** of MSA columns at binding-site positions → gate failed, Spearman ρ = −0.524 (anti-correlated with ConSurf reference). Diagnosis: deep MSAs from large protein families (kinases) produced spuriously high per-column entropy regardless of conservation.

2. **JSD vs Swiss-Prot background** → gate failed, ρ = −0.476. Diagnosis: binding-site definition size differed across reference proteins (KLIFS 85 residues for kinases vs GPCRdb ~30 for GPCRs vs ~25 for HIV protease) — averaging dilution dominated whatever conservation signal existed.

3. **Mean binding-site pLDDT** (AlphaFold's per-residue confidence averaged over binding-site residues) → gate PASSED, mean = 88.13. pLDDT IS the published reference quantity (AlphaFold DB exposes per-target mean confidence), no fragile rank-correlation validation needed.

The H3 hypothesis is reframed: *"does per-target ESM-FP-MLP-vs-MLP advantage correlate with mean binding-site pLDDT?"* This is biologically meaningful (ESM-2 and AlphaFold both encode evolutionary/sequence signal at protein level; the question is whether ESM-2's contribution is greatest where AlphaFold also has high structural confidence) and cleanly testable.

The experimental RNS code (`compute_per_residue_rns`, `compute_conservation_entropy`, `aggregate_target_rns`) is preserved in `rns_scoring.py` with `EXPERIMENTAL:` docstring annotations. It works (passing unit tests) — it just doesn't reproduce ConSurf.

#### 2. Tasks 14, 16, 17 dramatically modified

**T14** was: "Run full RNS pipeline on AWS — 543 targets, ~3-5 hours wall-clock on 96 CPUs." With the pLDDT pivot, **T14 collapses to ~5 min local work**:

```python
# For each of 543 targets:
#   1. Build aminergic config (T7) for GPCR side; build kinase config the same way
#   2. fetch_structure(uniprot, cache_dir, prefer="alphafold")
#   3. fetch_binding_site(chembl_id, class_name="kinase" or "gpcr_aminergic", cache_dir)
#   4. mean_plddt = compute_binding_site_plddt(structure, binding_site, provenance)
#   5. Save row to results/per_target_plddt.csv
```

No MSAs needed; no jackhmmer; no UniRef50 or Swiss-Prot. The AlphaFold structures are downloaded by `fetch_structure` (a few seconds each, cached); the binding sites by `fetch_binding_site` (cached). Total wall-clock dominated by binding-site annotation API calls (~10-30 min for 543 targets across KLIFS + GPCRdb due to per-target rate limiting).

**T16** (PDB-vs-AlphaFold sensitivity analysis) — **MOOT**. pLDDT is an AlphaFold-specific concept (no PDB equivalent); we use AlphaFold structures uniformly. The structure-source decision tree from spec 5.4 doesn't apply.

**T17** (structure-source decision branch) — **MOOT** for the same reason.

These two tasks should be marked DONE-MOOT in the new session's plan execution; the completion summary (T23) notes the pivot reasoning.

#### 3. Performance optimization for T14's binding-site annotation

`fetch_binding_site` was implemented in P3-T3 with per-target API calls. KLIFS's `/kinase_information` returns all 1127 kinases per call (~320 KB JSON); GPCRdb's `/receptorlist/` similarly returns all 237+ aminergic receptors per call (~2 MB). For 543-target T14 pipeline execution, **fetch these once at the top of the script and pass to `fetch_binding_site` as a pre-computed index** rather than hammering the APIs 543 times.

#### 4. RDKit descriptors deferred from in-git copy

`data/kinase_reference/features/rdkit_descriptors.npz` is **80 MB** — exceeds GitHub's recommended file-size threshold. The file is local-only at `/Users/joshuaabbott/mlproject/data/processed/v1/features/rdkit_descriptors.npz`. Plan 3's analyses (T15 hypothesis tests, T18-T20 notebooks) use the per-target metric CSVs not raw feature arrays, so this exclusion does not block downstream work. Document this in the manuscript's data-availability statement; include in eventual Zenodo deposit.

#### 5. Library `__version__` constant is stale at 1.0.0

Functionally inert (no consumer reads it) but cosmetic — pip reports v1.1.0 correctly; the in-`__init__.py` constant reads "1.0.0". **Task 22's v1.2.0 release fixes this** by bumping both `pyproject.toml` and `__init__.py` `__version__` in the same commit.

---

## Critical context for the resuming session

### Repos & branches

| Repo | Local path | Branch | Remote | Latest commit |
|---|---|---|---|---|
| Library | `/Users/joshuaabbott/target-affinity-ml` | `main` | github.com/jmabbott40/target-affinity-ml | `5b22478` (P3-T6 pLDDT pivot); plan3-development branch on origin at same SHA. **main on origin still at v1.1.0** — Task 22 fast-forwards + tags v1.2.0 |
| Kinase app | `/Users/joshuaabbott/mlproject` | `phase1-multi-class-expansion` | github.com/jmabbott40/kinase-affinity-baselines | `a3eba41` (P3 plan + spec docs) |
| GPCR app | `/Users/joshuaabbott/gpcr-aminergic-benchmarks` | `main` | github.com/jmabbott40/gpcr-aminergic-benchmarks | `59af044` (T10 README + data card updates) |

### Python environment

- **Local:** `/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python` (Python 3.11.15). NEVER use base miniforge.
- **AWS** (currently up at `ec2-3-144-163-211.us-east-2.compute.amazonaws.com`): `~/miniforge3/envs/kinase-affinity/bin/python`. **Must prepend** `LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH` before every Python invocation (libstdc++ version mismatch with system).
- AWS-installed library state: from `git+...@plan3-development` (force-reinstall to refresh after local commits). When new library code lands, `pip install --upgrade --force-reinstall --no-deps git+...@plan3-development` on AWS picks it up.

### Git conventions

- Always use `git -c commit.gpgsign=false commit`. GPG signing prompts otherwise block non-interactive commits.
- Co-Author trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- Library work happens on local `main` but pushes to `plan3-development` branch on origin until Task 22 release. **Do not push library main to origin** until Task 22.
- GPCR repo pushes to its `origin/main` directly (no dev branch — analysis-application repo, work-in-progress is the work).
- Kinase repo pushes to `phase1-multi-class-expansion` (not main, per Plan 2 handoff convention).

### GitHub large-push fix

`git push` may fail with HTTP 400 for packs >~50 MB. Apply this once per repo:

```bash
cd <repo>
git config http.postBuffer 524288000   # 500 MB
```

Already applied to `gpcr-aminergic-benchmarks` during P3-T9. May need application to other repos if a large push fails.

### .gitignore force-add pattern

Once a directory is `data/`-ignored, `!data/subdir/**` exceptions don't recursively unignore files inside (known git limitation). Use `git add -f data/kinase_reference/<path>` to force-add past blanket ignores. This was needed for P3-T9 and may be needed for similar patterns in T18-T20 if those notebooks generate outputs under `results/` that hit `*.npz`/`*.parquet` blanket-ignores.

---

## Remaining task breakdown

### Track 2B — Scaffold-diversity module (Tasks 11-13)

All local. No AWS needed.

**P3-T11 — `target-affinity-ml/src/target_affinity_ml/benchmarks/scaffold_diversity.py` + unit tests.** Per-target metrics: `n_scaffolds` (Bemis-Murcko), `scaffold_entropy` (Shannon), `largest_cluster_fraction`, `mean_tanimoto` (Morgan FP), `activity_cliff_frequency`. Plus `compute_class_aggregates` for per-class mean/median/IQR. Implementation template in the plan doc. Use `MurckoScaffold.GetScaffoldForMol(mol)` + `Chem.MolToSmiles(...)` (the Plan 2 Task 8 corrected idiom — not `MurckoScaffold.MolToSmiles` which returns empty strings).

**P3-T12 — Compute scaffold metrics for both classes.** Create `gpcr-aminergic-benchmarks/src/gpcr_aminergic_benchmarks/analyses/scaffold_diversity.py` + `scripts/compute_scaffold_metrics.py`. Reads:
- GPCR curated: `data/processed/v1/curated_activities.parquet` (70,163 rows)
- Kinase curated: `data/kinase_reference/curated_activities.parquet` (353K rows)

Outputs `results/supplement/per_target_metrics_scaffold.csv` with ~543 rows (~507 kinase + 36 GPCR).

**P3-T13 — `fit_degradation_regression` in the library's scaffold_diversity.py.** Two regressions: random→scaffold and scaffold→target, pooled across both classes, class as covariate with class × X interaction. Uses `statsmodels.formula.api.ols`.

### Track 2C — pLDDT pipeline + hypothesis tests (Tasks 14, 15; ~~16, 17~~ MOOT)

**P3-T14 — Compute mean binding-site pLDDT for 543 targets.** Local script in GPCR repo. Pseudocode:

```python
# scripts/compute_per_target_plddt.py
from pathlib import Path
import json
import pandas as pd
from target_affinity_ml.benchmarks.rns_scoring import (
    fetch_structure, fetch_binding_site, compute_binding_site_plddt,
)

# Pre-cache the KLIFS + GPCRdb full lists ONCE (see discovery #3)
# Then loop:
rows = []
for chembl_id, uniprot_id, class_name in target_index:
    structure, prov = fetch_structure(uniprot_id, cache_dir, prefer="alphafold")
    binding_site = fetch_binding_site(chembl_id, class_name, cache_dir, uniprot_id=uniprot_id)
    if not binding_site:
        rows.append({"chembl_id": chembl_id, "class": class_name, "plddt": float("nan"), "n_binding_residues": 0, "error": "no binding site"})
        continue
    mean_plddt = compute_binding_site_plddt(structure, binding_site, prov)
    rows.append({
        "chembl_id": chembl_id, "class": class_name,
        "uniprot": uniprot_id, "structure_source": prov["source"],
        "n_binding_residues": len(binding_site),
        "mean_binding_site_plddt": mean_plddt,
        "error": None,
    })
pd.DataFrame(rows).to_csv("data/processed/v1/per_target_plddt.csv", index=False)
```

Expected runtime: 10-30 min wall-clock (binding-site API rate limits dominate). Output ~543 rows. **No AWS, no MSAs.**

**P3-T15 — `target-affinity-ml/src/target_affinity_ml/benchmarks/hypothesis_tests.py` + unit tests.** Implements H1-H4 + between-class machinery per spec Section 5 / design Section 5. The H3 test specifically uses `data/processed/v1/per_target_plddt.csv` (GPCR side) + `data/kinase_reference/benchmark_v1/per_target/per_target_*.csv` (kinase per-target metrics) and the per-target ESM-FP-MLP-vs-MLP advantage values.

**P3-T16, P3-T17 — MOOT.** Mark DONE-MOOT in TaskList. The completion summary (T23) explains why.

### Phase 3 — Notebooks (Tasks 18-21)

Sequential — each notebook builds on previous outputs.

**P3-T18 — `notebooks/05_scaffold_diversity.ipynb`** consumes T12's CSV + per-target metrics from both classes, calls `fit_degradation_regression`, produces Figure 3 (scatter + regression lines) + Table 4 partial.

**P3-T19 — `notebooks/06_plddt_analysis.ipynb`** (**rename from `06_rns_analysis.ipynb`** per pivot). Consumes T14's per-target pLDDT CSV + ESM-FP-MLP and MLP per-target metrics. Produces Figure 4 (per-target ESM-2-advantage vs pLDDT scatter, colored by class) + Table 4 RNS rows (now "pLDDT rows"). Per-class distribution comparison via KS + Welch.

**P3-T20 — `notebooks/07_cross_class_comparison.ipynb`** (the headline notebook). Tables 1, 2, 3 + Figures 1, 2, 5. Runs H1-H4 hypothesis tests, aggregates verdicts.

**P3-T21 — Final tables + figures assembly.** Inventory check; `results/README.md` index linking each table/figure to paper section + producing notebook.

### Phase 6 — Release + wrap-up (Tasks 22-24)

**P3-T22 — Library v1.2.0 release.** Bump `pyproject.toml` version `1.1.0` → `1.2.0`. **Also bump `src/target_affinity_ml/__init__.py`'s `__version__ = "1.0.0"` → `"1.2.0"`** (fixes the Plan 2 stale-version bug). Update CHANGELOG. Run full test suite + ruff. Commit, tag v1.2.0, push `main` + tag.

```bash
cd /Users/joshuaabbott/target-affinity-ml
git push origin main:main           # fast-forwards origin main from v1.1.0 to v1.2.0
git push origin v1.2.0              # tag
git push origin :plan3-development  # delete the dev branch (optional cleanup)
```

After v1.2.0 lands, update GPCR repo's `pyproject.toml` dependency from `@plan3-development` to `@v1.2.0`. Commit + push GPCR repo.

**P3-T23 — Plan 3 completion summary + Plan 4 (manuscript) handoff.** New file at `kinase-affinity-baselines/docs/superpowers/plans/2026-XX-XX-plan3-completion-summary.md`. Mirror the Plan 2 summary structure. **Document the metric pivot transparently** (from RNS to pLDDT, including the two failed validation gates and why) so the manuscript can be written from this record. Document T16/T17 as MOOT-by-design.

Plus `2026-XX-XX-plan4-manuscript-handoff.md` — brief: what's ready for manuscript writing (Tables 1-4, Figures 1-5, `results/README.md` index), what's not, recommended starting points.

**P3-T24 — GPCR repo v1.1.0 tag.** Bump `gpcr-aminergic-benchmarks` `pyproject.toml` to `1.1.0`; CHANGELOG; commit + tag + push. Supports later Zenodo deposit.

---

## Lessons learned (avoid repeating)

| Discovery | How it bit us | How the new session avoids it |
|---|---|---|
| Bundled JSON data files dropped by pip install | P3-T6 first gate run on AWS hit FileNotFoundError on `_rns_reference_data.json` | `pyproject.toml` now has `[tool.setuptools.package-data]` declaration; new data files in `benchmarks/` are bundled automatically |
| ConSurf-style "RNS" doesn't match what entropy-based metrics measure | Two gate failures (ρ = −0.5 each) before pivoting to pLDDT | Use `compute_binding_site_plddt` as the per-target metric; the experimental RNS functions remain in code but are NOT the H3 metric |
| Binding-site annotation API per-call overhead | KLIFS/GPCRdb return full lists per call (~320 KB / ~2 MB); 543 calls = many MB redundant + rate-limit pressure | T14's pipeline must fetch the full lists ONCE and cache locally before iterating targets |
| GitHub HTTP 400 on mid-sized pushes | T9 push of ~46 MB failed with HTTP 400 + sideband disconnect | `git config http.postBuffer 524288000` + push in 2-3 smaller commits (small first) |
| gitignore `data/` blocks `!data/sub/**` exceptions | T9 couldn't `git add` the kinase reference data through normal flow | Use `git add -f` to force-add past blanket ignores |
| Library `__version__` stale | Already documented; pip says 1.1.0 but `__init__.py` constant says 1.0.0 | Task 22 fixes both in the same release commit |
| AWS public IP rotates on stop/start | Each AWS-required task needs to confirm address | User has kept AWS running for other work; if it stops between sessions, get the new IP from user |
| AlphaFold DB URL version drift | Plan said v4, DB is at v6 | `fetch_structure` resolves dynamically via the prediction API; version-agnostic |
| AWS libstdc++ vs conda libicu | Python's sqlite3 import fails | Always prepend `LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH` for AWS Python invocations |

---

## Estimated effort

With the pivot's simplifications:

| Track | Tasks | Engineering | AWS |
|---|---|---|---|
| 2B (scaffold-diversity) | T11-T13 | ~2-3 days | none |
| 2C (pLDDT pipeline + hypothesis tests) | T14, T15 | ~2-3 days | ~30 min for T14 binding-site API calls (could be local) |
| Phase 3 (notebooks) | T18-T21 | ~3-5 days | none |
| Phase 6 (release + wrap-up) | T22-T24 | ~1 day | none |
| **Total remaining** | **14 tasks (T16/T17 MOOT)** | **~8-12 working days** | **< 1 hour** |

Significantly less than the original spec's ~17-20 day estimate, owing entirely to the pLDDT pivot's collapse of T14's RNS-pipeline scope.

---

## When Plan 3 completes

Task 23 produces the Plan 3 completion summary + a Plan 4 (manuscript-drafting) handoff. After that, Plan 4 — if pursued — would be the bioRxiv preprint writeup. Plan 4 is its own creative-work phase that benefits from a fresh session and possibly different cognitive context (writing prose vs. shipping code) — not subagent-driven-development territory.
