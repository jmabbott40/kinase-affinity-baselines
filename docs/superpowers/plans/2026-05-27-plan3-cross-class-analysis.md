# Plan 3: GPCR Cross-Class Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute scaffold-diversity metrics + Prabakaran-Bromberg Residue Neighborhood Significance (RNS) scores for both kinase and GPCR classes, run the four pre-registered hypothesis tests (H1-H4), generate the four main-text tables and five main-text figures, and end at "results assembled and visualized." Manuscript drafting is out of scope.

**Architecture:** Parts A-B build the library's `benchmarks/` module (scaffold_diversity, rns_scoring, hypothesis_tests) — class-agnostic methodology. Parts C-D apply that methodology to host the kinase reference data + run the full cross-class pipeline on AWS. Part E builds the GPCR-repo notebooks that consume the methodology to produce paper-ready tables and figures. Part F releases library v1.2.0 and writes the Plan 3 completion summary. **RNS-first execution structure** (Approach A from design spec) front-loads the only significant risk: the RNS validation gate runs in Part A before downstream work commits to RNS as the primary metric.

**Tech Stack:** Python 3.11, `target-affinity-ml@v1.2.0` (release this plan produces), Biopython, DSSP, FreeSASA, jackhmmer (HMMER package), statsmodels, scipy.stats, RDKit, matplotlib, AWS g5.12xlarge (4× A10G GPUs, 96 CPUs — Plan 3 mostly uses CPUs).

---

## Spec & predecessor references

- **Plan 3 design spec:** `docs/superpowers/specs/2026-05-27-plan3-cross-class-analysis-design.md` (Sections 3-5 cover module designs; Section 6 the sequencing; Section 7 the branch points; Section 9 the stop conditions)
- **Plan 1 design / methodology source:** `docs/superpowers/specs/2026-04-17-gpcr-aminergic-phase1-design.md` (Sections 5 + 6 — the scientific methodology this plan implements)
- **Plan 2 completion summary** (predecessor state): `docs/superpowers/plans/2026-05-27-plan2-completion-summary.md`
- **Plan 2 plan** (for structural mirror): `docs/superpowers/plans/2026-04-30-plan2-gpcr-data-pipeline-benchmark.md`
- **GPCR benchmark CSV** (Plan 3 input): `gpcr-aminergic-benchmarks/results/gpcr_v1_benchmark/multi_seed_aggregated.csv`

---

## Limitations from Plan 1 / Plan 2 addressed here

| ID | Limitation | Addressed by |
|----|-----------|--------------|
| L2 (Plan 1) | Kinase reference NPZs not on GitHub | Tasks 7-10 — committed into GPCR repo under `data/kinase_reference/` |
| v1.1.0 stale `__version__` (Plan 2) | `target_affinity_ml.__version__` returns `"1.0.0"` after v1.1.0 install | Task 22 — bumped + fixed in v1.2.0 release |
| Spec R2 (RNS implementation cold-start) | Risk of multi-week RNS implementation slip | Tasks 1-6 — validation gate runs Week 1; clean pivot to conservation-entropy fallback if gate fails |
| Spec R4 (AlphaFold systematic bias) | Mixing PDB + AlphaFold structures may bias cross-class RNS | Tasks 14, 16, 17 — Tier 1+2+3 strategy from spec 5.4 |
| Spec R7 (multiple-testing concerns) | Many hypothesis tests inflate family-wise error rate | Task 15 — Bonferroni primary, FDR exploratory, family size disclosed |

---

## File structure

### `target-affinity-ml` library — Part A + C + D modifications (→ v1.2.0)

```
target-affinity-ml/
├── src/target_affinity_ml/benchmarks/
│   ├── __init__.py                            # MODIFY: expose public functions
│   ├── README.md                              # MODIFY: document new modules
│   ├── _rns_reference_data.json               # NEW: bundled Prabakaran-Bromberg reference values
│   ├── scaffold_diversity.py                  # NEW: per-target metrics + regressions (Task 11-13)
│   ├── rns_scoring.py                         # NEW: full RNS pipeline (Tasks 1-6, 14, 16)
│   └── hypothesis_tests.py                    # NEW: H1-H4 + between-class machinery (Task 15)
├── src/target_affinity_ml/__init__.py         # MODIFY: bump __version__ to "1.2.0" (Task 22)
├── tests/unit/
│   ├── test_scaffold_diversity.py             # NEW (Task 11)
│   ├── test_rns_scoring.py                    # NEW (Tasks 2-6)
│   └── test_hypothesis_tests.py               # NEW (Task 15)
├── CHANGELOG.md                               # MODIFY: 1.2.0 section (Task 22)
└── pyproject.toml                             # MODIFY: version → 1.2.0 + new optional deps (biopython, statsmodels) (Task 22)
```

### `gpcr-aminergic-benchmarks` — Part B + E + F additions

```
gpcr-aminergic-benchmarks/
├── data/kinase_reference/                     # NEW: hosted kinase data (Tasks 7-10)
│   ├── features/morgan_fp.npz
│   ├── features/rdkit_descriptors.npz
│   ├── features/esm2_embeddings.npz           # if available, else documented
│   ├── features/smiles_index.json
│   ├── features/target_index.json
│   ├── curated_activities.parquet
│   ├── splits/{random,scaffold,target}_split.json
│   ├── benchmark_v1/all_seeds_metrics.csv
│   ├── benchmark_v1/multi_seed_aggregated.csv
│   ├── benchmark_v1/predictions_seed*/        # per-target predictions
│   └── README.md                              # provenance + DOI placeholders
├── data/structures/                           # gitignored — fetched, cached
│   ├── pdb/{uniprot}.pdb
│   └── alphafold/{uniprot}.pdb
├── data/msas/                                 # gitignored — jackhmmer outputs
│   └── {uniprot}.sto
├── src/gpcr_aminergic_benchmarks/analyses/    # NEW
│   ├── __init__.py
│   ├── scaffold_diversity.py                  # Task 12: per-target metrics for kinase + GPCR
│   ├── rns_analysis.py                        # Task 14: GPCR+kinase RNS pipeline
│   └── cross_class.py                         # Task 15: combined + hypothesis tests
├── notebooks/
│   ├── 05_scaffold_diversity.ipynb            # NEW (Task 18)
│   ├── 06_rns_analysis.ipynb                  # NEW (Task 19)
│   └── 07_cross_class_comparison.ipynb        # NEW (Task 20)
├── results/
│   ├── tables/                                # 4 main-text tables (Task 21)
│   ├── figures/                               # 5 main-text figures (Task 21)
│   └── supplement/                            # per-target metrics, sensitivity analyses, structure provenance
├── docs/data_card.md                          # MODIFY: append kinase-reference section (Task 10)
└── CHANGELOG.md                               # MODIFY: 1.1.0 section (Task 24)
```

### `kinase-affinity-baselines` (mlproject) — Part F final addition

```
kinase-affinity-baselines/
└── docs/superpowers/plans/
    └── 2026-XX-XX-plan3-completion-summary.md  # NEW (Task 23) + Plan 4 handoff doc
```

---

# PART A — RNS module + validation gate (Phase 1: RNS-first)

The whole point of Approach A: implement RNS, run the validation gate, learn EARLY whether it works before downstream code commits to it. Tasks 1-6 are sequential.

## Task 1: Library `benchmarks/` scaffolding + bundled reference data

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/__init__.py`
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/README.md`
- Create: `target-affinity-ml/src/target_affinity_ml/benchmarks/_rns_reference_data.json`
- Create: `target-affinity-ml/tests/unit/test_rns_scoring.py` (stubs only)

**Context:** The `benchmarks/` module was scaffolded empty in spec Section 3.2 but never populated. This task creates the empty module-level stubs for `rns_scoring.py` (Task 2-6 fill them in), bundles the published Prabakaran-Bromberg reference data for the validation gate, and writes a placeholder test file.

- [ ] **Step 1: Look up Prabakaran-Bromberg reference proteins**

Read the Prabakaran-Bromberg paper to identify 5-10 reference proteins with published per-target RNS values (or per-residue RNS values that can be aggregated). Extract: protein name, UniProt accession, PDB ID, reported RNS values. If the paper publishes only per-residue values, compute the mean over the published binding-site residues to get per-target values.

If exact published values aren't available per protein, fall back to a **relative ranking** — the reference set is then the ranked order of RNS values across proteins, and the validation gate criterion uses Spearman ρ rather than absolute MAD.

- [ ] **Step 2: Create `_rns_reference_data.json`**

Format:
```json
{
  "source": "Prabakaran & Bromberg, <year>, <journal>",
  "doi": "...",
  "reference_proteins": [
    {
      "name": "<protein_name>",
      "uniprot": "<accession>",
      "pdb_id": "<pdb_id>",
      "binding_site_residues": [<list of 1-indexed residue numbers from the PDB chain>],
      "published_target_rns": <float in [0, 1]>,
      "published_per_residue_rns": null,
      "notes": "..."
    }
  ]
}
```

The `binding_site_residues` list is what the validation gate uses to know which residues to score. The `published_target_rns` is the reference value to compare against.

- [ ] **Step 3: Update `benchmarks/__init__.py`**

```python
"""Cross-class methodology modules for the target-affinity-ml benchmark suite.

Modules
-------
scaffold_diversity : per-target + per-class scaffold metrics + correlation regressions
rns_scoring        : Prabakaran-Bromberg RNS scoring with conservation-entropy fallback
hypothesis_tests   : Pre-registered H1-H4 hypothesis tests + between-class machinery
"""
from target_affinity_ml.benchmarks.scaffold_diversity import (
    compute_scaffold_metrics,
    compute_class_aggregates,
    fit_degradation_regression,
)
from target_affinity_ml.benchmarks.rns_scoring import (
    fetch_structure,
    fetch_binding_site,
    compute_msa,
    compute_per_residue_rns,
    aggregate_target_rns,
    compute_conservation_entropy,
    validation_gate,
)
from target_affinity_ml.benchmarks.hypothesis_tests import (
    h1_rf_vs_deep,
    h2_split_degradation,
    h3_esm_target_advantage,
    h4_single_seed_flip_rate,
    class_split_interaction,
)
__all__ = [...]  # explicit re-export list — avoids the Plan 1 wildcard-drop trap
```

These imports will fail initially (modules empty). That's expected — Tasks 2-6 + 11-15 populate them.

- [ ] **Step 4: Update `benchmarks/README.md`**

Brief module-level description of what each `.py` is for. Note that the `_rns_reference_data.json` is the bundled validation reference set.

- [ ] **Step 5: Create test stub `tests/unit/test_rns_scoring.py`**

```python
"""Tests for the RNS scoring module. Populated incrementally by Tasks 2-6."""
import pytest

@pytest.fixture
def reference_data():
    import json
    from pathlib import Path
    path = Path(__file__).parent.parent.parent / "src/target_affinity_ml/benchmarks/_rns_reference_data.json"
    with open(path) as fh:
        return json.load(fh)


def test_reference_data_loads(reference_data):
    assert "reference_proteins" in reference_data
    assert len(reference_data["reference_proteins"]) >= 5
    for p in reference_data["reference_proteins"]:
        assert "uniprot" in p
        assert "binding_site_residues" in p
```

- [ ] **Step 6: Run the stub test**

Run: `cd target-affinity-ml && /opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python -m pytest tests/unit/test_rns_scoring.py::test_reference_data_loads -v`
Expected: PASS (this confirms the JSON loads).

- [ ] **Step 7: Commit**

```bash
cd /Users/joshuaabbott/target-affinity-ml
git add src/target_affinity_ml/benchmarks/__init__.py \
        src/target_affinity_ml/benchmarks/README.md \
        src/target_affinity_ml/benchmarks/_rns_reference_data.json \
        tests/unit/test_rns_scoring.py
git -c commit.gpgsign=false commit -m "Scaffold benchmarks/ module + bundled RNS reference data (Task 1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `fetch_structure()` — PDB + AlphaFold DB acquisition

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/rns_scoring.py`
- Modify: `target-affinity-ml/tests/unit/test_rns_scoring.py`
- Modify: `target-affinity-ml/pyproject.toml` (add `biopython>=1.81` to optional `[benchmarks]` extra)

**Context:** First building block of the RNS pipeline. Fetches a protein structure given a UniProt accession. Prefers PDB experimental structures; falls back to AlphaFold DB.

**Important — AlphaFold:** the AlphaFold DB at `https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb` is static-file hosting (NOT the prediction service — no quotas). Use polite concurrency (max 8 parallel downloads), exponential backoff on 5xx.

- [ ] **Step 1: Add `biopython` to optional deps**

In `pyproject.toml`'s `[project.optional-dependencies]`, add:
```toml
benchmarks = ["biopython>=1.81", "statsmodels>=0.14", "matplotlib>=3.7", "freesasa>=2.2"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_rns_scoring.py
def test_fetch_structure_returns_structure_and_provenance(tmp_path):
    from target_affinity_ml.benchmarks.rns_scoring import fetch_structure
    structure, provenance = fetch_structure("P00533", cache_dir=tmp_path, prefer="alphafold")
    # P00533 is human EGFR — known to have AlphaFold structure
    assert structure is not None
    assert provenance["source"] == "AlphaFold"
    assert provenance["uniprot_id"] == "P00533"
    assert "binding_site_pLDDT_mean" in provenance  # will be None pre-binding-site computation
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_rns_scoring.py::test_fetch_structure_returns_structure_and_provenance -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_structure'`

- [ ] **Step 4: Implement `fetch_structure`**

In `src/target_affinity_ml/benchmarks/rns_scoring.py`:

```python
"""Prabakaran-Bromberg Residue Neighborhood Significance (RNS) scoring.

See Plan 3 design spec Section 3 for the pipeline data flow and Section 7
for the structure-source decision tree.
"""
from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import requests
from Bio.PDB import PDBParser, MMCIFParser

logger = logging.getLogger(__name__)

ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb"
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_structure(
    uniprot_id: str,
    cache_dir: Path,
    prefer: Literal["pdb", "alphafold"] = "pdb",
    pdb_id: str | None = None,
    max_retries: int = 4,
) -> tuple[Any, dict]:
    """Fetch a protein structure (PDB experimental preferred, AlphaFold fallback).

    Returns
    -------
    (structure, provenance_dict): Biopython structure object + provenance metadata.
        provenance includes: source, uniprot_id, pdb_id (if PDB), pdb_resolution,
        binding_site_pLDDT_mean (None — populated downstream), conformational_state.

    The function caches downloads to cache_dir. Subsequent calls with the same
    inputs are no-ops (return cached structure).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # PDB path (only if prefer="pdb" AND pdb_id provided)
    if prefer == "pdb" and pdb_id:
        pdb_path = cache_dir / "pdb" / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            _download_with_backoff(PDB_URL.format(pdb_id=pdb_id.lower()), pdb_path, max_retries)
        if pdb_path.exists():
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, pdb_path)
            return structure, {
                "source": "PDB",
                "uniprot_id": uniprot_id,
                "pdb_id": pdb_id,
                "pdb_resolution": _extract_resolution(structure),
                "binding_site_pLDDT_mean": None,
                "binding_site_pLDDT_min": None,
                "conformational_state": "unknown",
            }

    # AlphaFold fallback (always works for human proteins in the DB)
    af_path = cache_dir / "alphafold" / f"{uniprot_id}.pdb"
    if not af_path.exists():
        _download_with_backoff(ALPHAFOLD_URL.format(accession=uniprot_id), af_path, max_retries)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(uniprot_id, af_path)
    return structure, {
        "source": "AlphaFold",
        "uniprot_id": uniprot_id,
        "pdb_id": None,
        "pdb_resolution": None,
        "binding_site_pLDDT_mean": None,
        "binding_site_pLDDT_min": None,
        "conformational_state": "unknown",
    }


def _download_with_backoff(url: str, path: Path, max_retries: int) -> None:
    """Download with exponential backoff on 5xx / network errors. Idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                path.write_bytes(r.content)
                return
            if 500 <= r.status_code < 600 or r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()  # 4xx other than 429 — give up
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url} after {max_retries} retries")


def _extract_resolution(structure) -> float | None:
    """Extract resolution from PDB header. None if not present (AlphaFold)."""
    return structure.header.get("resolution")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_rns_scoring.py::test_fetch_structure_returns_structure_and_provenance -v`
Expected: PASS. (Requires network — make sure your dev env has internet.)

- [ ] **Step 6: Add a batched-download helper test**

```python
def test_fetch_structure_caches_correctly(tmp_path):
    """Re-fetching the same accession returns cached file without network."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_structure
    _ = fetch_structure("P00533", cache_dir=tmp_path, prefer="alphafold")
    mtime_first = (tmp_path / "alphafold" / "P00533.pdb").stat().st_mtime
    _ = fetch_structure("P00533", cache_dir=tmp_path, prefer="alphafold")
    mtime_second = (tmp_path / "alphafold" / "P00533.pdb").stat().st_mtime
    assert mtime_first == mtime_second  # no re-download
```

Run + verify PASS.

- [ ] **Step 7: Commit**

```bash
git add src/target_affinity_ml/benchmarks/rns_scoring.py \
        tests/unit/test_rns_scoring.py \
        pyproject.toml
git -c commit.gpgsign=false commit -m "Add fetch_structure() — PDB + AlphaFold DB acquisition with caching (Task 2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `fetch_binding_site()` — KLIFS + GPCRdb adapters

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/rns_scoring.py`
- Modify: `target-affinity-ml/tests/unit/test_rns_scoring.py`

**Context:** Routes binding-site annotation by class — KLIFS for kinases (returns the 85-residue ATP-pocket definition), GPCRdb for aminergic GPCRs (returns ~25-40 orthosteric-pocket residues). Both are REST APIs; cache responses.

- [ ] **Step 1: Verify KLIFS API endpoints**

Look up the KLIFS REST API documentation. The endpoint for binding-site residues by ChEMBL ID or UniProt ID is something like `https://klifs.net/api_v2/kinase_information` followed by `/binding_pocket`. Confirm the exact path + JSON response schema.

- [ ] **Step 2: Verify GPCRdb API endpoints**

Look up the GPCRdb REST API. For a given UniProt ID, the orthosteric binding-pocket residues are reachable via something like `https://gpcrdb.org/services/structure/<protein>/binding_residues`. Confirm path + schema.

- [ ] **Step 3: Write failing tests**

```python
def test_fetch_binding_site_kinase(tmp_path):
    """Kinase binding-site routes to KLIFS and returns ~85 residues."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_binding_site
    residues = fetch_binding_site("CHEMBL203", class_name="kinase", cache_dir=tmp_path)
    # CHEMBL203 is EGFR (kinase)
    assert isinstance(residues, list)
    assert all(isinstance(r, int) for r in residues)
    assert 60 <= len(residues) <= 100  # KLIFS canonical pocket is 85; allow drift

def test_fetch_binding_site_gpcr(tmp_path):
    """GPCR binding-site routes to GPCRdb and returns ~25-40 residues."""
    from target_affinity_ml.benchmarks.rns_scoring import fetch_binding_site
    residues = fetch_binding_site("CHEMBL217", class_name="gpcr_aminergic", cache_dir=tmp_path)
    # CHEMBL217 is DRD2
    assert 15 <= len(residues) <= 50  # GPCRdb orthosteric pocket is typically ~25-40
```

- [ ] **Step 4: Run tests → fail (function missing)**

- [ ] **Step 5: Implement `fetch_binding_site`** with `_klifs_binding_site` and `_gpcrdb_binding_site` helpers. Use response caching to `cache_dir / "binding_sites" / "{class}_{chembl_id}.json"`.

Handle the case where the API returns no residues (target not in KLIFS / GPCRdb): return empty list and log a warning. Downstream code (Task 5+) handles empty binding sites by skipping that target with a note in supplement.

- [ ] **Step 6: Run tests → pass**

- [ ] **Step 7: Add a robustness test**

```python
def test_fetch_binding_site_missing_target_returns_empty(tmp_path):
    """A target not in KLIFS returns empty list (and logs a warning) — does NOT raise."""
    residues = fetch_binding_site("CHEMBL_INVALID", class_name="kinase", cache_dir=tmp_path)
    assert residues == []
```

- [ ] **Step 8: Commit**

```bash
git add src/target_affinity_ml/benchmarks/rns_scoring.py tests/unit/test_rns_scoring.py
git -c commit.gpgsign=false commit -m "Add fetch_binding_site() — KLIFS + GPCRdb adapters with caching (Task 3)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `compute_msa()` — jackhmmer wrapper

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/rns_scoring.py`
- Modify: `target-affinity-ml/tests/unit/test_rns_scoring.py`

**Context:** Generates a per-target multiple sequence alignment from a UniProt accession by running `jackhmmer` against UniRef50. Caches MSAs to `cache_dir / msas / {uniprot}.sto`. This is the slowest single step in RNS (~10-30 min/target sequentially) — parallelized by Task 14 across 96 CPUs.

**Local dev consideration:** UniRef50 is ~30 GB. Don't expect this to run on a laptop. For TDD locally, use a tiny synthetic database fixture; for the real run (Task 14), AWS has the database.

- [ ] **Step 1: Verify jackhmmer availability on AWS**

SSH to AWS, check `jackhmmer --version` and `which jackhmmer`. If missing, install via `sudo apt install hmmer` (it's in standard Ubuntu repos). UniRef50 download location: `~/databases/uniref50.fasta` (~30 GB). If missing, `wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz` + gunzip.

- [ ] **Step 2: Write the failing test (uses synthetic fixture)**

```python
def test_compute_msa_invokes_jackhmmer(tmp_path, monkeypatch):
    """compute_msa shells out to jackhmmer with the expected args."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_msa
    calls = []
    def fake_run(args, **kw):
        calls.append(args)
        # write a minimal Stockholm-format file so the cache check passes
        out_path = Path(args[args.index("-A") + 1])
        out_path.write_text("# STOCKHOLM 1.0\nseq1 MGKLA\n//\n")
        class R: returncode = 0
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    # Need to provide an input sequence; the function fetches from UniProt internally
    monkeypatch.setattr(
        "target_affinity_ml.benchmarks.rns_scoring._fetch_uniprot_fasta",
        lambda uid: f">sp|{uid}|TEST\nMGKLA\n"
    )
    out = compute_msa("P00533", db_path=Path("/fake/uniref50.fasta"), out_dir=tmp_path)
    assert out.exists()
    assert any("jackhmmer" in str(a) for a in calls[0])
```

- [ ] **Step 3: Run → fails (no function)**

- [ ] **Step 4: Implement `compute_msa`**

```python
def compute_msa(
    uniprot_id: str,
    db_path: Path,
    out_dir: Path,
    n_iter: int = 3,
    n_cpu: int = 4,
) -> Path:
    """Run jackhmmer (uniprot_id sequence against db_path) and cache MSA.

    Returns
    -------
    Path to the Stockholm-format MSA file at out_dir / "{uniprot_id}.sto".
    Idempotent — if the file already exists, returns immediately.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msa_path = out_dir / f"{uniprot_id}.sto"
    if msa_path.exists() and msa_path.stat().st_size > 0:
        return msa_path
    fasta = _fetch_uniprot_fasta(uniprot_id)
    query_path = out_dir / f"{uniprot_id}_query.fasta"
    query_path.write_text(fasta)
    import subprocess
    args = [
        "jackhmmer",
        "-N", str(n_iter),
        "--cpu", str(n_cpu),
        "-A", str(msa_path),
        str(query_path),
        str(db_path),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"jackhmmer failed for {uniprot_id}: {result.stderr}")
    return msa_path


def _fetch_uniprot_fasta(uniprot_id: str) -> str:
    """Fetch UniProt FASTA via REST. Cached upstream by caller if needed."""
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", timeout=30)
    r.raise_for_status()
    return r.text
```

- [ ] **Step 5: Run test → passes**

- [ ] **Step 6: Add a real-jackhmmer integration test (mark `@pytest.mark.slow`)**

```python
@pytest.mark.slow
def test_compute_msa_real_jackhmmer(tmp_path):
    """Integration test — runs real jackhmmer against UniRef50 (~10-30 min)."""
    db = Path("~/databases/uniref50.fasta").expanduser()
    if not db.exists():
        pytest.skip("UniRef50 not available in this env (run on AWS)")
    msa = compute_msa("P00533", db_path=db, out_dir=tmp_path, n_iter=1, n_cpu=4)
    assert msa.exists()
    assert msa.stat().st_size > 1000  # nontrivial alignment
```

Skip locally; runs as part of Task 14 on AWS.

- [ ] **Step 7: Commit**

```bash
git add src/target_affinity_ml/benchmarks/rns_scoring.py tests/unit/test_rns_scoring.py
git -c commit.gpgsign=false commit -m "Add compute_msa() — jackhmmer wrapper with caching (Task 4)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `compute_per_residue_rns()` — the Prabakaran-Bromberg algorithm

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/rns_scoring.py`
- Modify: `target-affinity-ml/tests/unit/test_rns_scoring.py`

**Context:** The core RNS computation. For each binding-site residue:
1. Identify *spatial* neighbors (Cα distance ≤ 8 Å) and *sequence* neighbors (±5 positions)
2. Compute Shannon entropy of the local sequence environment across the MSA
3. Weight by BLOSUM62 evolutionary conservation
4. Output: per-residue RNS in [0, 1]

This is where the published-paper ambiguity bites — exact parameter choices (neighbor radius, sequence-window size, BLOSUM scheme, indel handling) may need iteration via the validation gate (Task 6).

- [ ] **Step 1: Read the Prabakaran-Bromberg paper carefully**

Extract the exact algorithm. Note any parameters where the paper allows multiple interpretations — those become validation-gate iteration knobs.

- [ ] **Step 2: Write the failing test**

```python
def test_compute_per_residue_rns_returns_normalized_scores(tmp_path):
    """RNS scores are floats in [0, 1] for each binding-site residue."""
    from target_affinity_ml.benchmarks.rns_scoring import compute_per_residue_rns, fetch_structure
    structure, _ = fetch_structure("P00533", cache_dir=tmp_path, prefer="alphafold")
    # Use a small synthetic MSA — 5 sequences, 100 residues, mostly conserved
    msa_path = tmp_path / "test.sto"
    msa_path.write_text(_build_synthetic_msa(n_seqs=5, length=100))
    binding_site = [25, 30, 50, 75]
    scores = compute_per_residue_rns(structure, binding_site, msa_path)
    assert set(scores.keys()) == set(binding_site)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
```

(The `_build_synthetic_msa` helper is a test fixture you write; produces a valid Stockholm-format MSA with controlled conservation patterns.)

- [ ] **Step 3: Run → fails**

- [ ] **Step 4: Implement `compute_per_residue_rns`**

Use:
- `Bio.PDB.NeighborSearch` for spatial neighbors (Cα coordinates)
- `Bio.AlignIO.read(path, "stockholm")` for MSA parsing
- A BLOSUM62 matrix from `Bio.Align.substitution_matrices`
- Shannon entropy from `scipy.stats.entropy`

```python
def compute_per_residue_rns(
    structure: Any,
    binding_site: list[int],
    msa_path: Path,
    neighbor_radius_angstrom: float = 8.0,
    sequence_window: int = 5,
    blosum_name: str = "BLOSUM62",
) -> dict[int, float]:
    """Compute per-residue RNS scores for binding-site residues.

    For each binding-site residue, identifies spatial + sequence neighbors,
    computes Shannon entropy of the local sequence environment across the MSA,
    weights by BLOSUM evolutionary conservation. Returns dict {residue_idx: RNS}.
    """
    # ... full implementation per the Prabakaran-Bromberg paper
```

The implementation is ~80-100 lines. Document any deviations from the paper inline.

- [ ] **Step 5: Run test → passes**

- [ ] **Step 6: Add `compute_conservation_entropy` (fallback metric)**

```python
def compute_conservation_entropy(
    binding_site: list[int],
    msa_path: Path,
) -> float:
    """Simpler fallback metric: Shannon entropy of binding-site MSA columns.

    Used in parallel with RNS for the sensitivity analysis (does the cross-class
    correlation hold under both metrics?), AND as the pivot metric if RNS
    validation gate (Task 6) fails.
    """
    # ... ~30 lines
```

Add a test that exercises it on the same synthetic MSA.

- [ ] **Step 7: Commit**

```bash
git add src/target_affinity_ml/benchmarks/rns_scoring.py tests/unit/test_rns_scoring.py
git -c commit.gpgsign=false commit -m "Add compute_per_residue_rns() + compute_conservation_entropy() — RNS core algorithm (Task 5)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `validation_gate()` — RNS GO/NO-GO decision ⚠️ critical branch point

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/rns_scoring.py`
- Modify: `target-affinity-ml/tests/unit/test_rns_scoring.py`
- Create: `target-affinity-ml/scripts/run_rns_validation_gate.py`

**Context:** This is the Approach A go/no-go. Run the full pipeline on the 5-10 reference proteins from Task 1's bundled JSON. Compare our RNS values to the published ones. **Pre-specified criterion: Spearman ρ ≥ 0.7 OR mean absolute deviation ≤ 10%.**

⚠️ **Branch point per design spec Section 7:**
- **GATE PASSES** → continue to Part B/C/D with full RNS as primary metric
- **GATE FAILS** → pivot: mark RNS module experimental; conservation-entropy becomes primary metric for Tasks 14-20; update plan with marginal-effort downstream changes (since the fallback metric returns the same `dict[chembl_id, float]` shape, downstream code is mostly unaffected)

- [ ] **Step 1: Implement `aggregate_target_rns`**

```python
def aggregate_target_rns(
    per_residue: dict[int, float],
    provenance: dict,
    structure: Any,
    use_plddt_weighting: bool = True,
) -> float:
    """Aggregate per-residue RNS to per-target. For AlphaFold structures, apply
    pLDDT weighting (spec 5.4 Tier 2): residues below pLDDT 50 contribute nothing,
    residues at 90+ contribute fully. PDB structures use uniform weights.
    """
    if not per_residue:
        return float("nan")
    if provenance["source"] == "AlphaFold" and use_plddt_weighting:
        weights = {r: max(0.0, (_get_residue_plddt(structure, r) - 50) / 50) for r in per_residue}
        total_w = sum(weights.values())
        if total_w == 0:
            return float("nan")
        return sum(per_residue[r] * weights[r] for r in per_residue) / total_w
    return sum(per_residue.values()) / len(per_residue)


def _get_residue_plddt(structure, residue_idx: int) -> float:
    """Extract pLDDT (in AlphaFold PDBs, stored in the B-factor field)."""
    for chain in structure[0]:
        for res in chain:
            if res.id[1] == residue_idx:
                return res["CA"].get_bfactor() if "CA" in res else 50.0
    return 50.0
```

- [ ] **Step 2: Implement `validation_gate`**

```python
def validation_gate(
    cache_dir: Path,
    db_path: Path,
    reference_set: str = "prabakaran_bromberg",
    spearman_threshold: float = 0.7,
    mad_threshold: float = 0.10,
) -> tuple[bool, dict, Path]:
    """Run the full RNS pipeline on bundled reference proteins; compare to published.

    Returns
    -------
    (passed, deviations_dict, summary_csv_path)

    passed = True iff Spearman rho >= spearman_threshold OR MAD <= mad_threshold.
    summary_csv has one row per reference protein with our_rns, published_rns, abs_dev.
    """
    import json
    ref_path = Path(__file__).parent / "_rns_reference_data.json"
    with open(ref_path) as fh:
        ref = json.load(fh)
    our_values = []
    pub_values = []
    rows = []
    for protein in ref["reference_proteins"]:
        try:
            structure, prov = fetch_structure(
                protein["uniprot"], cache_dir=cache_dir, pdb_id=protein.get("pdb_id")
            )
            msa = compute_msa(protein["uniprot"], db_path, cache_dir / "msas")
            per_res = compute_per_residue_rns(structure, protein["binding_site_residues"], msa)
            target_rns = aggregate_target_rns(per_res, prov, structure)
            our_values.append(target_rns)
            pub_values.append(protein["published_target_rns"])
            rows.append({
                "name": protein["name"],
                "our_rns": target_rns,
                "published_rns": protein["published_target_rns"],
                "abs_dev": abs(target_rns - protein["published_target_rns"]),
            })
        except Exception as exc:
            logger.error("validation_gate failed for %s: %s", protein["name"], exc)
            rows.append({"name": protein["name"], "error": str(exc)})

    # Compute gate criteria
    valid = [(o, p) for o, p in zip(our_values, pub_values) if o == o]  # filter NaN
    if len(valid) < 5:
        return False, {"error": f"Only {len(valid)} reference proteins succeeded"}, Path()
    our, pub = zip(*valid)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(our, pub)
    mad = sum(abs(o - p) for o, p in valid) / len(valid)
    passed = (rho >= spearman_threshold) or (mad <= mad_threshold)
    # Write summary CSV
    import pandas as pd
    csv_path = cache_dir / "validation_gate_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return passed, {"spearman_rho": rho, "mad": mad}, csv_path
```

- [ ] **Step 3: Write a unit test that mocks fetch_structure/compute_msa**

The unit test mocks the network calls but exercises the gate logic. Tests both the pass case and the fail case.

- [ ] **Step 4: Create `scripts/run_rns_validation_gate.py`**

```python
#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""CLI driver for the RNS validation gate.

Usage
-----
    # Local (small reference set, no UniRef50 needed if using mock MSAs)
    python scripts/run_rns_validation_gate.py --cache-dir /tmp/rns_validation

    # AWS (with real UniRef50)
    LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH \\
        ~/miniforge3/envs/kinase-affinity/bin/python scripts/run_rns_validation_gate.py \\
        --cache-dir ~/rns_validation --db ~/databases/uniref50.fasta
"""
import argparse
import json
import sys
from pathlib import Path

from target_affinity_ml.benchmarks.rns_scoring import validation_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("/home/ubuntu/databases/uniref50.fasta"))
    args = parser.parse_args()
    passed, deviations, csv_path = validation_gate(cache_dir=args.cache_dir, db_path=args.db)
    print("=" * 60)
    print(f"VALIDATION GATE: {'PASS' if passed else 'FAIL'}")
    print(f"  spearman_rho = {deviations.get('spearman_rho', 'n/a'):.3f}")
    print(f"  mad          = {deviations.get('mad', 'n/a'):.3f}")
    print(f"  summary csv  = {csv_path}")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the gate on AWS**

Sync the library, kinase repo, GPCR repo to AWS as needed. Run the validation gate script on AWS where UniRef50 is available.

```bash
ssh -i $AWS_KEY $AWS_HOST
cd ~/target-affinity-ml
git pull origin main  # bring down all the Task 1-5 commits
LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH \
    ~/miniforge3/envs/kinase-affinity/bin/python scripts/run_rns_validation_gate.py \
    --cache-dir ~/rns_validation
```

⚠️ **Branch on result:**

- [ ] **Step 6a (GATE PASSES):**
  Continue to Part B. Commit the gate CSV under `target-affinity-ml/results/rns_validation_summary.csv` as an artifact. Update Plan 3 plan with the actual gate outcome.

- [ ] **Step 6b (GATE FAILS):**
  Iterate on the algorithm parameters (Step 4 has knobs: neighbor radius, sequence window, BLOSUM choice). 1-2 day debugging budget. If still failing after debugging:
  - Mark `compute_per_residue_rns` as experimental in the docstring
  - Update Tasks 14-20 to use `compute_conservation_entropy` as the primary metric — most code paths are minimally affected since both functions return per-target floats
  - Update the Plan 3 plan document to reflect the pivot
  - Commit the pivot decision with a clear rationale

- [ ] **Step 7: Commit (regardless of branch)**

```bash
git add src/target_affinity_ml/benchmarks/rns_scoring.py \
        tests/unit/test_rns_scoring.py \
        scripts/run_rns_validation_gate.py
git -c commit.gpgsign=false commit -m "Add validation_gate() + aggregate_target_rns() + CLI runner (Task 6)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If branch 6b chosen: a second commit documenting the pivot in `CHANGELOG.md`.

---

# PART B — Kinase reference data hosting (Track 2A)

Three tasks resolving Plan 1 limitation L2. Runs in parallel with Tracks 2B + 2C via subagent-driven-development.

## Task 7: Inventory + verify kinase reference files

**Files:**
- Create: `gpcr-aminergic-benchmarks/scripts/_kinase_reference_inventory.py` (utility script — not committed)

- [ ] **Step 1: Inventory local + AWS kinase files**

```bash
# Local
ls -lR /Users/joshuaabbott/mlproject/data/processed/v1/features/
ls -lR /Users/joshuaabbott/mlproject/results/ 2>/dev/null
# AWS
ssh -i $AWS_KEY $AWS_HOST 'ls -lR ~/mlproject/data/processed/v1/ ~/mlproject/results/ 2>&1 | head -100'
```

Catalog: features (morgan_fp, rdkit_descriptors, esm2_embeddings if available, smiles_index, target_index), splits (random/scaffold/target JSONs), curated parquet, benchmark CSVs (all_seeds_metrics, multi_seed_aggregated), per-run predictions npz files.

- [ ] **Step 2: Identify gaps**

Cross-reference what Plan 3 needs (per the design spec Section 2.1) vs what exists. **Critical gap if discovered: per-target predictions for the H3 RNS-correlation analysis** — Plan 1 may have only saved aggregate metrics, not per-target predictions.

- [ ] **Step 3: If per-target predictions are missing → trigger Plan 1.5**

Per stop condition #2 in the design spec: missing critical artifacts triggers a supplementary kinase benchmark re-run on AWS GPUs. The Plan 1.5 supplement is a one-off short doc + ~4-hour AWS GPU re-run that produces the missing artifacts. If triggered, document and pause Plan 3 Task 7 until 1.5 completes.

- [ ] **Step 4: Document the inventory**

Brief report: what's present, what's missing, what's been triggered as Plan 1.5 supplementary work. No commit yet (Task 8 commits the actual data).

---

## Task 8: Add `data/kinase_reference/` to GPCR repo with .gitignore exception

**Files:**
- Modify: `gpcr-aminergic-benchmarks/.gitignore`
- Create: `gpcr-aminergic-benchmarks/data/kinase_reference/README.md`

- [ ] **Step 1: Add .gitignore exception**

In `gpcr-aminergic-benchmarks/.gitignore`, the current `data/` line excludes everything. Add an exception:
```
data/
!data/kinase_reference/
!data/kinase_reference/**
```

- [ ] **Step 2: Create the README.md**

`data/kinase_reference/README.md`:
```markdown
# Kinase reference data

This directory mirrors the kinase Plan 1 outputs that Plan 3's cross-class
analysis needs. Provenance:

- Source repo: https://github.com/jmabbott40/kinase-affinity-baselines
- Source data version: v1 (kinase preprint v1)
- Source commit: <commit SHA from kinase repo at time of copy>
- Library version that produced this data: target-affinity-ml v1.0.0

This data is committed here (despite gitignore-by-default of data/)
to resolve Plan 1 limitation L2. Eventual Zenodo deposit will replace
this in-git copy with a versioned DOI.

Files:
  features/                  - Morgan FP + RDKit descriptors + ESM-2 + indexes
  curated_activities.parquet - 353K curated kinase records (~206K compounds)
  splits/                    - random / scaffold / target index JSONs
  benchmark_v1/              - all_seeds_metrics.csv + multi_seed_aggregated.csv
                              + per-seed prediction npz files (for per-target analysis)
```

- [ ] **Step 3: Commit (data not yet present — just the gitignore + README)**

```bash
cd /Users/joshuaabbott/gpcr-aminergic-benchmarks
git add .gitignore data/kinase_reference/README.md
git -c commit.gpgsign=false commit -m "Add data/kinase_reference/ gitignore exception + provenance README (Task 8)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Sync kinase data → commit + push

**Files:**
- Add: many under `gpcr-aminergic-benchmarks/data/kinase_reference/`

- [ ] **Step 1: Copy from local mlproject**

```bash
cp -r /Users/joshuaabbott/mlproject/data/processed/v1/features /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/
cp -r /Users/joshuaabbott/mlproject/data/processed/v1/splits /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/  # if present
cp /Users/joshuaabbott/mlproject/data/processed/v1/curated_activities.parquet /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/  # if present
```

- [ ] **Step 2: Sync from AWS for items not present locally**

```bash
scp -i $AWS_KEY -r $AWS_HOST:~/mlproject/results/kinase_v1_benchmark/ /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/benchmark_v1/
scp -i $AWS_KEY -r $AWS_HOST:~/mlproject/results/predictions_seed*/ /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/benchmark_v1/predictions/
```

- [ ] **Step 3: Verify size + content**

```bash
du -sh /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/
find /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/ -type f | head -20
```

Expected ~50-80 MB total. If much larger, check for accidentally-included extras (model checkpoints, big logs).

- [ ] **Step 4: Commit + push**

```bash
cd /Users/joshuaabbott/gpcr-aminergic-benchmarks
git add data/kinase_reference/
git status  # confirm only kinase_reference/ items staged
git -c commit.gpgsign=false commit -m "Add kinase reference data resolving Plan 1 L2 limitation (Task 9)

Sourced from kinase-affinity-baselines v1 (commit <SHA>) and the
AWS kinase benchmark results. Eventual Zenodo deposit will replace
this in-git copy with a versioned DOI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

## Task 10: Update GPCR repo README + data card with kinase-data attribution

**Files:**
- Modify: `gpcr-aminergic-benchmarks/README.md`
- Modify: `gpcr-aminergic-benchmarks/docs/data_card.md`

- [ ] **Step 1: README addition**

Add a section "Cross-class data" explaining that the repo now hosts both the GPCR primary data AND the kinase reference data (under `data/kinase_reference/`) for cross-class comparison.

- [ ] **Step 2: data_card.md addition**

Append a "Kinase reference data" section with the same provenance info as `data/kinase_reference/README.md`. Cross-link to the kinase repo + preprint.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/data_card.md
git -c commit.gpgsign=false commit -m "Document kinase reference data in README + data card (Task 10)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# PART C — Scaffold-diversity module (Track 2B)

Three tasks implementing `benchmarks/scaffold_diversity.py`. Parallel with Tracks 2A + 2C.

## Task 11: Library `scaffold_diversity.py` + unit tests

**Files:**
- Create: `target-affinity-ml/src/target_affinity_ml/benchmarks/scaffold_diversity.py`
- Create: `target-affinity-ml/tests/unit/test_scaffold_diversity.py`

**Context:** Per-target metrics per spec Section 5.1 / design Section 4.1. RDKit's `MurckoScaffold.GetScaffoldForMol` + `Chem.MolToSmiles` for scaffold extraction (the corrected pattern from Plan 2 Task 8).

- [ ] **Step 1: TDD — write failing tests**

```python
# tests/unit/test_scaffold_diversity.py
import pytest
import pandas as pd

@pytest.fixture
def synthetic_target_df():
    return pd.DataFrame({
        "canonical_smiles": ["c1ccccc1", "CCO", "c1ccccc1C", "CCO", "c1ccccc1CC"],  # 3 distinct scaffolds (benzene, ethanol, benzene-derivatives)
        "target_chembl_id": ["CHEMBL1"] * 5,
        "pactivity": [6.0, 5.0, 7.0, 5.5, 6.5],
    })


def test_compute_scaffold_metrics_returns_expected_columns(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics
    metrics = compute_scaffold_metrics(synthetic_target_df, target_col="target_chembl_id", smiles_col="canonical_smiles")
    assert isinstance(metrics, pd.DataFrame)
    expected_cols = {"target_chembl_id", "n_compounds", "n_scaffolds", "scaffold_entropy", "largest_cluster_fraction", "mean_tanimoto"}
    assert expected_cols.issubset(metrics.columns)


def test_n_scaffolds_correct(synthetic_target_df):
    from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics
    m = compute_scaffold_metrics(synthetic_target_df, "target_chembl_id", "canonical_smiles")
    # Aspirin-free synthetic — 3 distinct generic scaffolds (benzene base, ethanol, but generics may collapse)
    assert m.iloc[0]["n_scaffolds"] >= 2


def test_largest_cluster_fraction_in_bounds(synthetic_target_df):
    m = compute_scaffold_metrics(synthetic_target_df, "target_chembl_id", "canonical_smiles")
    assert 0.0 < m.iloc[0]["largest_cluster_fraction"] <= 1.0
```

- [ ] **Step 2: Run → fails**

- [ ] **Step 3: Implement `compute_scaffold_metrics`**

```python
def compute_scaffold_metrics(
    df: pd.DataFrame,
    target_col: str = "target_chembl_id",
    smiles_col: str = "canonical_smiles",
    pairwise_sample_size: int = 500,
    activity_col: str | None = "pactivity",
) -> pd.DataFrame:
    """Per-target scaffold-diversity metrics.

    Returns one row per target with columns:
      target_chembl_id, n_compounds, n_scaffolds, scaffold_entropy,
      largest_cluster_fraction, mean_tanimoto, activity_cliff_frequency
    """
    rows = []
    for tid, sub in df.groupby(target_col):
        scaffolds = _bemis_murcko_scaffolds(sub[smiles_col])
        from collections import Counter
        counts = Counter(scaffolds)
        n_comp = len(sub)
        n_scaff = len(counts)
        entropy = _shannon_entropy(list(counts.values()))
        lcf = max(counts.values()) / n_comp if n_comp else float("nan")
        mt = _mean_tanimoto(sub[smiles_col].tolist(), sample_size=pairwise_sample_size)
        acf = _activity_cliff_frequency(sub, smiles_col, activity_col) if activity_col else None
        rows.append({
            "target_chembl_id": tid,
            "n_compounds": n_comp,
            "n_scaffolds": n_scaff,
            "scaffold_entropy": entropy,
            "largest_cluster_fraction": lcf,
            "mean_tanimoto": mt,
            "activity_cliff_frequency": acf,
        })
    return pd.DataFrame(rows)


def _bemis_murcko_scaffolds(smiles: pd.Series) -> list[str]:
    """Compute Bemis-Murcko generic scaffolds. Uses the corrected idiom from Plan 2."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append("INVALID")
            continue
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        out.append(Chem.MolToSmiles(scaff) if scaff is not None else "NO_SCAFFOLD")
    return out


def _shannon_entropy(counts: list[int]) -> float:
    """Shannon entropy in nats."""
    import math
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


def _mean_tanimoto(smiles: list[str], sample_size: int = 500) -> float:
    """Mean pairwise Morgan FP Tanimoto over a random sample of pairs."""
    import random
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    if len(fps) < 2:
        return float("nan")
    pairs = []
    if len(fps) * (len(fps) - 1) // 2 <= sample_size:
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                pairs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    else:
        random.seed(42)
        for _ in range(sample_size):
            i, j = random.sample(range(len(fps)), 2)
            pairs.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return sum(pairs) / len(pairs) if pairs else float("nan")


def _activity_cliff_frequency(...):
    """Pairs with Tanimoto >= 0.7 AND |delta_pActivity| >= 1.5. Sample pairs."""
    # ~30 lines
```

- [ ] **Step 4: Run tests → pass**

- [ ] **Step 5: Commit**

```bash
git add src/target_affinity_ml/benchmarks/scaffold_diversity.py tests/unit/test_scaffold_diversity.py
git -c commit.gpgsign=false commit -m "Add scaffold_diversity.py — per-target Bemis-Murcko metrics (Task 11)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Per-target scaffold-metric computation for kinase + GPCR

**Files:**
- Create: `gpcr-aminergic-benchmarks/src/gpcr_aminergic_benchmarks/analyses/scaffold_diversity.py`
- Create: `gpcr-aminergic-benchmarks/scripts/compute_scaffold_metrics.py`

**Context:** Library function applied to actual data. Produces `results/supplement/per_target_metrics_scaffold.csv` for both classes.

- [ ] **Step 1: Write the analysis module**

```python
# src/gpcr_aminergic_benchmarks/analyses/scaffold_diversity.py
from pathlib import Path
import pandas as pd
from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics, compute_class_aggregates


def compute_both_classes(
    gpcr_curated: Path,
    kinase_curated: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Compute per-target scaffold metrics for both classes; concatenate; save."""
    gpcr = pd.read_parquet(gpcr_curated)
    gpcr_m = compute_scaffold_metrics(gpcr).assign(class_name="gpcr_aminergic")
    kin = pd.read_parquet(kinase_curated)
    kin_m = compute_scaffold_metrics(kin).assign(class_name="kinase")
    out = pd.concat([gpcr_m, kin_m], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
```

- [ ] **Step 2: Write the CLI driver script**

```python
# scripts/compute_scaffold_metrics.py — calls compute_both_classes with the real paths
```

- [ ] **Step 3: Run it**

```bash
cd /Users/joshuaabbott/gpcr-aminergic-benchmarks
/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python scripts/compute_scaffold_metrics.py
```

Expected output: `results/supplement/per_target_metrics_scaffold.csv` with ~543 rows (~36 GPCR + ~507 kinase).

- [ ] **Step 4: Commit**

```bash
git add src/gpcr_aminergic_benchmarks/analyses/scaffold_diversity.py \
        scripts/compute_scaffold_metrics.py \
        results/supplement/per_target_metrics_scaffold.csv
git -c commit.gpgsign=false commit -m "Compute per-target scaffold metrics for kinase + GPCR (Task 12)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Scaffold-diversity regression machinery

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/benchmarks/scaffold_diversity.py`
- Modify: `target-affinity-ml/tests/unit/test_scaffold_diversity.py`

**Context:** Implements `fit_degradation_regression` (per spec 5.1 / design 4.3). Two regressions: random→scaffold and scaffold→target, across both classes pooled, with class × X interaction.

- [ ] **Step 1: TDD — failing test with synthetic data**

A test that produces synthetic per-target metrics + degradation values, fits the regression, asserts the slope sign matches expectation.

- [ ] **Step 2: Implement `fit_degradation_regression`**

Uses `statsmodels.formula.api.ols`. Returns a structured dict with slope estimates (per-metric, per-class), CIs, R², interaction-term p-value.

- [ ] **Step 3: Run → pass**

- [ ] **Step 4: Commit**

```bash
git add src/target_affinity_ml/benchmarks/scaffold_diversity.py tests/unit/test_scaffold_diversity.py
git -c commit.gpgsign=false commit -m "Add fit_degradation_regression — class-stratified slope tests (Task 13)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# PART D — RNS full pipeline + hypothesis tests (Track 2C)

## Task 14: Run full RNS pipeline on AWS — 543 targets parallelized across 96 CPUs

**Files:**
- Create: `gpcr-aminergic-benchmarks/scripts/run_rns_pipeline.py`
- Output: `data/processed/v1/per_target_rns.csv`, structure cache, MSA cache

**Context:** This is the long compute. Per design spec Section 3.5: ~3-5 hours wall-clock on AWS for 543 targets across 96 CPUs (MSA generation dominates).

- [ ] **Step 1: Write the pipeline script**

```python
#!/usr/bin/env python
"""Run the full RNS pipeline for kinase + GPCR targets in parallel."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

import pandas as pd
from target_affinity_ml.benchmarks.rns_scoring import (
    fetch_structure, fetch_binding_site, compute_msa,
    compute_per_residue_rns, aggregate_target_rns, compute_conservation_entropy,
)


def process_one_target(args):
    chembl_id, uniprot_id, class_name, cache_dir, db_path = args
    try:
        structure, prov = fetch_structure(uniprot_id, cache_dir / "structures")
        binding_site = fetch_binding_site(chembl_id, class_name, cache_dir / "binding_sites")
        if not binding_site:
            return {"chembl_id": chembl_id, "rns": float("nan"), "entropy": float("nan"), "error": "no binding site"}
        msa = compute_msa(uniprot_id, db_path, cache_dir / "msas")
        per_res = compute_per_residue_rns(structure, binding_site, msa)
        rns = aggregate_target_rns(per_res, prov, structure)
        entropy = compute_conservation_entropy(binding_site, msa)
        return {"chembl_id": chembl_id, "class": class_name, "uniprot": uniprot_id,
                "rns": rns, "entropy": entropy, **prov, "error": None}
    except Exception as e:
        return {"chembl_id": chembl_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpcr-mapping", required=True)  # resolved_target_ids.json from Plan 2
    parser.add_argument("--kinase-mapping", required=True)  # kinase ChEMBL→UniProt
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("/home/ubuntu/databases/uniref50.fasta"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-workers", type=int, default=96)
    args = parser.parse_args()

    work_items = _assemble_work_items(args.gpcr_mapping, args.kinase_mapping, args.cache_dir, args.db)
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = [ex.submit(process_one_target, w) for w in work_items]
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            results.append(r)
            print(f"[{i + 1}/{len(futures)}] {r['chembl_id']}: rns={r.get('rns', 'err')}")
    pd.DataFrame(results).to_csv(args.out, index=False)
```

- [ ] **Step 2: Sync to AWS and run**

```bash
ssh -i $AWS_KEY $AWS_HOST 'cd ~/gpcr-aminergic-benchmarks && git pull'
ssh -i $AWS_KEY $AWS_HOST 'cd ~/gpcr-aminergic-benchmarks && \
    LD_LIBRARY_PATH=~/miniforge3/envs/kinase-affinity/lib:$LD_LIBRARY_PATH \
    ~/miniforge3/envs/kinase-affinity/bin/python scripts/run_rns_pipeline.py \
      --gpcr-mapping data/processed/v1/resolved_target_ids.json \
      --kinase-mapping data/kinase_reference/resolved_target_ids.json \
      --cache-dir ~/rns_cache \
      --out data/processed/v1/per_target_rns.csv'
```

Use `nohup` if you want to disconnect; ~3-5 hours.

- [ ] **Step 3: Sync results back; commit**

```bash
scp -i $AWS_KEY $AWS_HOST:~/gpcr-aminergic-benchmarks/data/processed/v1/per_target_rns.csv /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/processed/v1/
cd /Users/joshuaabbott/gpcr-aminergic-benchmarks
git add data/processed/v1/per_target_rns.csv
# Note: per_target_rns.csv is a key analysis output — committed despite data/ gitignore
# Add an exception in .gitignore: !data/processed/v1/per_target_rns.csv
git add .gitignore
git -c commit.gpgsign=false commit -m "Add per-target RNS scores for 543 kinase+GPCR targets (Task 14)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

## Task 15: Library `hypothesis_tests.py` + unit tests

**Files:**
- Create: `target-affinity-ml/src/target_affinity_ml/benchmarks/hypothesis_tests.py`
- Create: `target-affinity-ml/tests/unit/test_hypothesis_tests.py`

**Context:** Implements H1-H4 + between-class machinery per design Section 5. Wraps existing `evaluation/bootstrap.py` and `evaluation/multi_seed_analysis.py`.

- [ ] **Step 1: TDD — failing tests for each of H1-H4**

```python
def test_h1_rf_vs_deep_returns_dataframe():
    from target_affinity_ml.benchmarks.hypothesis_tests import h1_rf_vs_deep
    fake_per_seed = pd.DataFrame({
        "model": ["random_forest"] * 5 + ["esm_fp_mlp"] * 5,
        "split": ["random"] * 10,
        "class": ["kinase"] * 10,
        "seed": [42, 123, 456, 789, 1024] * 2,
        "rmse": [0.9, 0.91, 0.89, 0.92, 0.90, 1.5, 1.4, 1.6, 1.45, 1.55],
    })
    result = h1_rf_vs_deep(fake_per_seed)
    assert isinstance(result, pd.DataFrame)
    assert {"model_pair", "class", "split", "cohens_d", "p_raw", "p_bonferroni", "verdict"}.issubset(result.columns)
```

Repeat for H2, H3, H4.

- [ ] **Step 2: Implement each Hi function**

Per the design Section 5.1 test design. Each is ~30-60 lines.

- [ ] **Step 3: Implement `class_split_interaction`** for the between-class tests.

- [ ] **Step 4: Run → all pass**

- [ ] **Step 5: Commit**

```bash
git add src/target_affinity_ml/benchmarks/hypothesis_tests.py tests/unit/test_hypothesis_tests.py
git -c commit.gpgsign=false commit -m "Add hypothesis_tests.py — H1-H4 + between-class machinery (Task 15)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Structure-source Tier 3 sensitivity analysis (PDB vs AlphaFold)

**Files:**
- Modify: `gpcr-aminergic-benchmarks/scripts/run_rns_pipeline.py` (extend with PDB-vs-AF dual computation)
- Create: `gpcr-aminergic-benchmarks/data/processed/v1/per_target_rns_pdb_vs_af.csv`

**Context:** Per design Section 3.3 Tier 3 / spec 5.4. For targets with both PDB and AlphaFold structures, compute RNS from each source separately. Report Pearson correlation across paired targets.

- [ ] **Step 1: Add a `--dual-source` flag** to the pipeline script that runs `compute_per_residue_rns` once with PDB-only, once with AlphaFold-only for targets that have both.

- [ ] **Step 2: Run on AWS** with `--dual-source`.

- [ ] **Step 3: Sync `per_target_rns_pdb_vs_af.csv` back; commit.**

---

## Task 17: Structure-source decision-tree branch ⚠️

**Files:**
- Create: `gpcr-aminergic-benchmarks/results/supplement/structure_source_decision.md` (records the branch taken)

⚠️ **Branch point per design spec Section 7:**

- [ ] **Step 1: Compute the Pearson correlation** between PDB-RNS and AlphaFold-RNS across paired targets from Task 16's output.

- [ ] **Step 2: Branch on r**
  - **r > 0.85** → primary analysis = combined RNS with pLDDT weighting (default behavior of `aggregate_target_rns`)
  - **0.7 ≤ r < 0.85** → primary = PDB-only; supplement = combined with caveat. Update `notebooks/06_rns_analysis.ipynb` to filter PDB-only for the headline correlation; report combined in supplement.
  - **r < 0.7** → primary = PDB-only; AF-only targets excluded from RNS analyses, listed in supplement.

- [ ] **Step 3: Document the branch in `structure_source_decision.md`** with the observed correlation, the chosen branch, and the date.

- [ ] **Step 4: Commit**

```bash
git add results/supplement/structure_source_decision.md
git -c commit.gpgsign=false commit -m "Record structure-source decision branch per design spec 5.4 (Task 17)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# PART E — Application notebooks (Phase 3)

Three sequential notebook tasks. Each notebook is thin — orchestrates `analyses/` + library functions, renders figures, writes tables.

## Task 18: Notebook `05_scaffold_diversity.ipynb`

**Files:**
- Create: `gpcr-aminergic-benchmarks/notebooks/05_scaffold_diversity.ipynb`
- Create: `gpcr-aminergic-benchmarks/results/figures/figure3_scaffold_degradation.png`
- Create: `gpcr-aminergic-benchmarks/results/tables/04_metric_correlations_scaffold.csv` (partial — RNS portion added in Task 19)

- [ ] **Step 1: Notebook structure**

Cells:
1. Imports (numpy, pandas, matplotlib, the library's benchmarks functions, the GPCR repo's analyses module)
2. Load per_target_metrics_scaffold.csv + per-target degradation values (joined from multi_seed_aggregated.csv from both classes)
3. Call `fit_degradation_regression` for random→scaffold + scaffold→target
4. Render Figure 3: scatter of per-target degradation vs per-target scaffold-diversity metric, colored by class, with regression lines
5. Write Table 4 partial (scaffold portion)
6. Display key statistics + interpretation

- [ ] **Step 2: Run the notebook end-to-end**

`jupyter nbconvert --execute notebooks/05_scaffold_diversity.ipynb`

- [ ] **Step 3: Commit (notebook + figure + table)**

---

## Task 19: Notebook `06_rns_analysis.ipynb`

**Files:**
- Create: `gpcr-aminergic-benchmarks/notebooks/06_rns_analysis.ipynb`
- Create: `gpcr-aminergic-benchmarks/results/figures/figure4_rns_advantage.png`
- Modify: `gpcr-aminergic-benchmarks/results/tables/04_metric_correlations_scaffold.csv` → rename to `04_metric_correlations.csv` (append RNS portion)
- Create: `gpcr-aminergic-benchmarks/results/supplement/structure_provenance.csv`
- Create: `gpcr-aminergic-benchmarks/results/supplement/rns_pdb_vs_alphafold.csv`
- Create: `gpcr-aminergic-benchmarks/results/supplement/conservation_entropy_sensitivity.csv`

- [ ] **Step 1: Notebook structure**

Cells:
1. Imports + load per_target_rns.csv + per-target ESM-FP-MLP-minus-MLP advantage
2. Fit per-class regression: per-target advantage ~ per-target RNS. Plot Figure 4.
3. Per-class RNS distribution comparison (KS + Welch).
4. **Sensitivity analysis** — repeat with conservation-entropy substituted for RNS; write sensitivity CSV.
5. Honor the Task 17 structure-source branch.
6. Render Table 4 RNS rows.
7. Render supplement tables.

- [ ] **Step 2: Run end-to-end + commit.**

---

## Task 20: Notebook `07_cross_class_comparison.ipynb` (the headline notebook)

**Files:**
- Create: `gpcr-aminergic-benchmarks/notebooks/07_cross_class_comparison.ipynb`
- Create: `gpcr-aminergic-benchmarks/results/tables/01_dataset_summary.csv`
- Create: `gpcr-aminergic-benchmarks/results/tables/02_headline_rmse.csv`
- Create: `gpcr-aminergic-benchmarks/results/tables/03_hypothesis_outcomes.csv`
- Create: `gpcr-aminergic-benchmarks/results/figures/figure1_design_overview.png` (adapted from kinase preprint)
- Create: `gpcr-aminergic-benchmarks/results/figures/figure2_headline_replication.png`
- Create: `gpcr-aminergic-benchmarks/results/figures/figure5_hypothesis_summary.png`

- [ ] **Step 1: Build Tables 1, 2, 3**

Load both classes' data + benchmark results. Call `h1_rf_vs_deep`, `h2_split_degradation`, `h3_esm_target_advantage`, `h4_single_seed_flip_rate`. Aggregate to Table 3 with verdicts.

- [ ] **Step 2: Build Figures 1, 2, 5**

Figure 1 is a design schematic (probably reuses an asset from the kinase preprint adapted to cover both classes). Figure 2 is side-by-side bar charts of per-model RMSE × split for each class. Figure 5 is a radar/grouped-bar visualization of H1-H4 outcomes.

- [ ] **Step 3: Run end-to-end + commit.**

---

## Task 21: Final figures + tables assembly

**Files:**
- Verify: all tables in `results/tables/` exist and match design spec Section 5.4
- Verify: all figures in `results/figures/` exist
- Create: `gpcr-aminergic-benchmarks/results/README.md` (index of outputs)

- [ ] **Step 1: Inventory check**

Verify the 4 tables + 5 figures from the design spec are all present and well-formed.

- [ ] **Step 2: Write `results/README.md`**

Index linking each table/figure to the corresponding paper section + the notebook that produced it. This becomes navigation for the manuscript-drafting phase.

- [ ] **Step 3: Commit**

---

# PART F — Library release + wrap-up

## Task 22: Library v1.2.0 release (CHANGELOG + tag + push); fixes stale `__version__`

**Files:**
- Modify: `target-affinity-ml/pyproject.toml` (version → 1.2.0)
- Modify: `target-affinity-ml/src/target_affinity_ml/__init__.py` (`__version__ = "1.2.0"`)
- Modify: `target-affinity-ml/CHANGELOG.md`

- [ ] **Step 1: Bump versions in both places** (fixes the v1.1.0 stale-`__version__` bug).

- [ ] **Step 2: Add CHANGELOG `[1.2.0]` section** documenting the new `benchmarks/` module + the stale-`__version__` fix.

- [ ] **Step 3: Run full test suite + ruff**

```bash
cd /Users/joshuaabbott/target-affinity-ml
/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python -m pytest tests/ -v
/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python -m ruff check src/ tests/
```

Must pass + clean.

- [ ] **Step 4: Commit, tag v1.2.0, push main + tag**

```bash
git add pyproject.toml src/target_affinity_ml/__init__.py CHANGELOG.md
git -c commit.gpgsign=false commit -m "Release v1.2.0: benchmarks/ module + __version__ fix"
git tag -a v1.2.0 -m "target-affinity-ml v1.2.0 — cross-class benchmarks module

Adds scaffold_diversity, rns_scoring, hypothesis_tests modules under
benchmarks/. Implements Prabakaran-Bromberg RNS with conservation-entropy
fallback, formal H1-H4 hypothesis tests, scaffold-diversity correlation
regressions. Also fixes the v1.1.0 stale __version__ constant."
git push origin main
git push origin v1.2.0
```

- [ ] **Step 5: Pin GPCR repo to v1.2.0**

In `gpcr-aminergic-benchmarks/pyproject.toml`, bump dependency to `target-affinity-ml @ git+...@v1.2.0`. Run `pip install -e . --no-deps` to refresh. Commit.

---

## Task 23: Plan 3 completion summary + Plan 4 handoff doc

**Files:**
- Create: `kinase-affinity-baselines/docs/superpowers/plans/2026-XX-XX-plan3-completion-summary.md`
- Create: `kinase-affinity-baselines/docs/superpowers/plans/2026-XX-XX-plan4-manuscript-handoff.md`

- [ ] **Step 1: Plan 3 completion summary**

Mirror the Plan 2 completion summary structure. Cover:
- Library v1.2.0 release
- Kinase reference data hosting (Plan 1 L2 resolved)
- RNS validation gate outcome
- Per-target RNS pipeline outcome (success rate, structure-source decision)
- Scaffold-diversity findings
- H1-H4 outcomes — for each, the pre-registered prediction and the observed result
- Execution-friction log (any issues encountered)
- Plan 4 (manuscript) readiness

- [ ] **Step 2: Plan 4 handoff doc**

Brief: what's ready for manuscript drafting, what's not, recommended starting points (Tables 1-4, Figures 1-5, the results/README.md inventory).

- [ ] **Step 3: Commit + push (kinase repo, phase1 branch)**

---

## Task 24: GPCR repo v1.1.0 tag (final analysis state)

**Files:**
- Modify: `gpcr-aminergic-benchmarks/pyproject.toml` (version → 1.1.0)
- Modify: `gpcr-aminergic-benchmarks/CHANGELOG.md`

- [ ] **Step 1: Bump version + CHANGELOG**

GPCR repo v1.1.0 represents "Plan 3 complete; analysis frozen for paper." Eventual Zenodo deposit pins this tag.

- [ ] **Step 2: Commit + tag + push**

```bash
git add pyproject.toml CHANGELOG.md
git -c commit.gpgsign=false commit -m "Release v1.1.0: Plan 3 cross-class analysis complete"
git tag -a v1.1.0 -m "gpcr-aminergic-benchmarks v1.1.0 — Plan 3 complete"
git push origin main
git push origin v1.1.0
```

---

## Plan 3 verification checklist

- [ ] Library v1.2.0 tagged + pushed
- [ ] RNS validation gate result documented (passed OR fallback chosen)
- [ ] Per-target RNS computed for 543 targets (success rate documented)
- [ ] Structure-source decision branch (Task 17) documented in `structure_source_decision.md`
- [ ] Kinase reference data committed to GPCR repo (L2 resolved)
- [ ] 4 main-text tables in `results/tables/`
- [ ] 5 main-text figures in `results/figures/`
- [ ] Supplement tables (per-target metrics, structure provenance, sensitivity analyses) committed
- [ ] H1-H4 outcomes with effect sizes + Bonferroni-corrected p-values
- [ ] Scaffold-diversity correlation regression results
- [ ] RNS-stratified ESM-2 analysis results (PRIMARY metric + sensitivity analysis with conservation-entropy)
- [ ] Plan 3 completion summary committed (kinase repo, phase1 branch)
- [ ] Plan 4 (manuscript) handoff doc committed
- [ ] GPCR repo v1.1.0 tagged + pushed

## Estimated effort

| Part | Tasks | Engineering days | AWS wall-clock |
|------|-------|---|---|
| A (RNS validation gate) | 1-6 | ~5 | <1 hour |
| B (Kinase data hosting) | 7-10 | ~1-2 | minimal |
| C (Scaffold-diversity module) | 11-13 | ~2-3 | minimal |
| D (RNS full pipeline + hypothesis tests) | 14-17 | ~4-5 | **~3-5 hours** |
| E (Notebooks + tables/figures) | 18-21 | ~3-5 | minimal |
| F (Release + wrap-up) | 22-24 | ~1 | minimal |
| **Total** | **24 tasks** | **~17-20 days** | **~4-6 hours** |

## Notes for plan execution

- **Tasks 1-6 (RNS validation gate)** MUST complete before Tasks 14-17 — gate result determines whether full RNS or conservation-entropy is the primary metric downstream
- **Parts B, C, D are parallelizable** — they have no inter-dependencies. Subagent-driven dev can dispatch them in parallel via three subagents
- **Tasks 18, 19, 20 are sequential** — each notebook builds on the previous notebooks' outputs (07 consumes results from 05 + 06)
- **Library v1.2.0 (Task 22) is the gate for tagging GPCR v1.1.0 (Task 24)** — GPCR pin must reference v1.2.0, which must exist first
- **AWS instance management:** Phase 2 needs AWS for ~half a day (Task 14's 3-5 hour RNS pipeline). Other phases use local compute. Stop the AWS instance between phases to control cost (~$5.67/hr running, ~$0.01/hr stopped)
- **Backward compatibility:** v1.2.0 is strictly additive. Kinase repo's existing v1.0.0/v1.1.0 code paths must continue to work. The library's existing test suite (~76 unit tests) must still pass with no modifications
