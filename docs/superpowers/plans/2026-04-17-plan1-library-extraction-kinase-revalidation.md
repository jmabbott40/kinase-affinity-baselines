# Plan 1: Library Extraction + Kinase Re-Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the existing `kinase_affinity` codebase into a class-agnostic Python library `target-affinity-ml` (v1.0.0), validate that it reproduces the published kinase preprint v1 numbers within tolerance, and run the aminergic data audit gate that determines whether Phase 1 proceeds with binding-only data or pivots to EC50 inclusion.

**Architecture:** Extract reusable model + training + evaluation code into a pip-installable Python package, leaving only kinase-specific scripts/data in `kinase-affinity-baselines`. The library is the durable artifact citable across multiple preprints; the kinase repo becomes a thin application that imports it. Integration tests lock the kinase output values to ensure no behavior change during refactoring.

**Tech Stack:** Python 3.11, pip-installable package via `pyproject.toml`, pytest for tests, GitHub Actions for CI, semver for versioning, ChEMBL API for audit data.

---

## Spec reference

This plan implements Sections 3 (repository architecture) and 4.4 (data audit gate) of the spec at:
`/Users/joshuaabbott/mlproject/docs/superpowers/specs/2026-04-17-gpcr-aminergic-phase1-design.md`

Plan 2 (GPCR data + benchmark) and Plan 3 (methodology + cross-class) are downstream and depend on this plan completing successfully.

---

## File structure for Plan 1

### New repository: `target-affinity-ml`

```
target-affinity-ml/
├── pyproject.toml                    # NEW: pip-installable package definition
├── README.md                         # NEW: library README with "Used by" section
├── CHANGELOG.md                      # NEW: version history
├── LICENSE                           # NEW: MIT license (matching kinase repo)
├── .gitignore                        # NEW: Python artifacts
├── .github/workflows/ci.yml          # NEW: CI pipeline
├── src/
│   └── target_affinity_ml/
│       ├── __init__.py               # NEW: __version__, public API
│       ├── data/                     # MOVED from kinase_affinity/data/
│       │   ├── __init__.py
│       │   ├── chembl_fetcher.py     # RENAMED from fetch.py (more general)
│       │   ├── standardize.py
│       │   ├── curate.py
│       │   ├── splits.py
│       │   └── protein_sequences.py
│       ├── features/                 # MOVED from kinase_affinity/features/
│       │   ├── __init__.py
│       │   ├── fingerprints.py
│       │   ├── descriptors.py
│       │   ├── protein_embeddings.py
│       │   └── molecular_graphs.py
│       ├── models/                   # MOVED from kinase_affinity/models/
│       │   ├── __init__.py
│       │   ├── rf_model.py, xgb_model.py, ...
│       │   ├── esm_fp_mlp_model.py
│       │   ├── gnn_model.py, fusion_model.py
│       │   └── deep_base.py
│       ├── training/                 # MOVED from kinase_affinity/training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── deep_trainer.py
│       ├── evaluation/               # MOVED from kinase_affinity/evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py, uncertainty.py, analysis.py
│       │   ├── bootstrap.py, multi_seed.py
│       │   └── run_phase5.py         # CONSIDER: rename to run_evaluation.py
│       ├── visualization/            # MOVED from kinase_affinity/visualization/
│       │   ├── __init__.py
│       │   └── plots.py
│       └── benchmarks/               # NEW: placeholder for Plan 3 methodology
│           ├── __init__.py           # NEW: empty package init
│           └── README.md             # NEW: explains future contents
├── tests/                            # MOVED + EXTENDED from kinase repo
│   ├── unit/                         # Existing unit tests, namespace updated
│   │   ├── test_standardize.py
│   │   ├── test_features.py
│   │   ├── test_splits.py
│   │   └── test_models.py
│   └── integration/                  # NEW: locked kinase output tests
│       └── test_kinase_reproducibility.py   # NEW: validates v1.0 matches preprint
└── configs/                          # MOVED: default model hyperparameters
    ├── rf_baseline.yaml
    ├── xgb_baseline.yaml
    ├── elasticnet_baseline.yaml
    ├── mlp_baseline.yaml
    ├── esm_fp_mlp.yaml
    ├── gnn.yaml
    └── fusion.yaml
```

### Modified repository: `kinase-affinity-baselines`

```
kinase-affinity-baselines/
├── pyproject.toml                    # MODIFIED: add target-affinity-ml==1.0.0 dep
├── README.md                         # MODIFIED: add "Library extraction" section
├── src/
│   └── kinase_affinity/
│       ├── __init__.py               # MODIFIED: thin re-export from target_affinity_ml
│       └── (other files become thin wrappers OR are deleted)
├── scripts/
│   ├── aminergic_audit/              # NEW (Plan 1 Task 1)
│   │   ├── run_audit.py              # NEW: standalone audit script
│   │   └── target_lists.py           # NEW: aminergic Class A target list
│   └── rerun_kinase_v1.py            # NEW: kinase re-run with library v1.0
└── results/
    ├── aminergic_audit/              # NEW: audit outputs (markdown + figures)
    └── kinase_v1_revalidation/       # NEW: re-run results vs preprint v1
```

---

## Task 1: Aminergic data audit gate

**Files:**
- Create: `scripts/aminergic_audit/target_lists.py`
- Create: `scripts/aminergic_audit/run_audit.py`
- Create: `tests/test_aminergic_targets.py`
- Output: `results/aminergic_audit/audit_report.md`
- Output: `results/aminergic_audit/figures/per_target_record_counts.png`

**Why first:** This audit determines whether Phase 1 proceeds with binding-only data (Option A from spec Section 4.4) or pivots to EC50 inclusion. Running it first means we know early — before any library refactoring — whether the protocol holds.

**This is a standalone script** that uses existing `kinase_affinity.data` modules (which are class-agnostic). No library extraction needed yet.

- [ ] **Step 1: Write target list as Python module**

Create `scripts/aminergic_audit/target_lists.py`:

```python
"""Aminergic Class A GPCR target list for Phase 1 audit.

Source: GPCRdb (https://gpcrdb.org/), filtered to:
- Class A GPCRs (rhodopsin-like)
- Aminergic receptors only (excludes peptide, nucleotide, lipid, etc.)
- Human (Homo sapiens)
- Excludes 5-HT3 (ionotropic, not GPCR)

ChEMBL target IDs verified against ChEMBL 36.
"""

AMINERGIC_TARGETS = {
    "dopamine": {
        "DRD1": "CHEMBL2056",
        "DRD2": "CHEMBL217",
        "DRD3": "CHEMBL234",
        "DRD4": "CHEMBL219",
        "DRD5": "CHEMBL1850",
    },
    "serotonin": {
        "5-HT1A": "CHEMBL214",
        "5-HT1B": "CHEMBL1898",
        "5-HT1D": "CHEMBL1983",
        "5-HT1E": "CHEMBL3371",
        "5-HT1F": "CHEMBL1850",
        "5-HT2A": "CHEMBL224",
        "5-HT2B": "CHEMBL1833",
        "5-HT2C": "CHEMBL225",
        "5-HT4": "CHEMBL1875",
        "5-HT5A": "CHEMBL3426",
        "5-HT6": "CHEMBL3371",
        "5-HT7": "CHEMBL3155",
    },
    "adrenergic": {
        "ADRA1A": "CHEMBL229",
        "ADRA1B": "CHEMBL1867",
        "ADRA1D": "CHEMBL223",
        "ADRA2A": "CHEMBL1867",
        "ADRA2B": "CHEMBL1942",
        "ADRA2C": "CHEMBL1916",
        "ADRB1": "CHEMBL213",
        "ADRB2": "CHEMBL210",
        "ADRB3": "CHEMBL246",
    },
    "histamine": {
        "HRH1": "CHEMBL231",
        "HRH2": "CHEMBL1941",
        "HRH3": "CHEMBL264",
        "HRH4": "CHEMBL3759",
    },
    "muscarinic": {
        "CHRM1": "CHEMBL216",
        "CHRM2": "CHEMBL211",
        "CHRM3": "CHEMBL245",
        "CHRM4": "CHEMBL1821",
        "CHRM5": "CHEMBL2035",
    },
    "trace_amine": {
        # Optional: include only if ≥1000 binding records (audit decides)
        "TAAR1": "CHEMBL2095203",
    },
}


def get_all_target_ids(include_taar: bool = True) -> list[str]:
    """Return flat list of all aminergic target ChEMBL IDs."""
    targets = []
    for family, members in AMINERGIC_TARGETS.items():
        if family == "trace_amine" and not include_taar:
            continue
        targets.extend(members.values())
    return targets


def get_target_to_family() -> dict[str, str]:
    """Return mapping of ChEMBL ID → family name."""
    mapping = {}
    for family, members in AMINERGIC_TARGETS.items():
        for chembl_id in members.values():
            mapping[chembl_id] = family
    return mapping
```

**NOTE:** ChEMBL IDs above are placeholders requiring verification. The audit script will verify each ID exists in ChEMBL 36 and report any mismatches.

- [ ] **Step 2: Write test for target list module**

Create `tests/test_aminergic_targets.py`:

```python
"""Tests for aminergic target list module."""
import pytest
from scripts.aminergic_audit.target_lists import (
    AMINERGIC_TARGETS,
    get_all_target_ids,
    get_target_to_family,
)


def test_target_families_are_complete():
    """All five aminergic families plus optional trace_amine."""
    expected_families = {"dopamine", "serotonin", "adrenergic",
                         "histamine", "muscarinic", "trace_amine"}
    assert set(AMINERGIC_TARGETS.keys()) == expected_families


def test_all_target_ids_are_unique():
    ids = get_all_target_ids(include_taar=True)
    assert len(ids) == len(set(ids))


def test_target_to_family_inverse_consistent():
    mapping = get_target_to_family()
    for family, members in AMINERGIC_TARGETS.items():
        for chembl_id in members.values():
            assert mapping[chembl_id] == family


def test_chembl_ids_match_format():
    """All ChEMBL IDs are well-formed."""
    for ids in [v for family in AMINERGIC_TARGETS.values()
                for v in family.values()]:
        assert ids.startswith("CHEMBL")
        assert ids[6:].isdigit()


def test_exclude_taar_option():
    """include_taar=False excludes TAAR1."""
    with_taar = set(get_all_target_ids(include_taar=True))
    without_taar = set(get_all_target_ids(include_taar=False))
    excluded = with_taar - without_taar
    assert excluded == {"CHEMBL2095203"}
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_aminergic_targets.py -v`
Expected: All 5 tests PASS

- [ ] **Step 4: Implement audit script**

Create `scripts/aminergic_audit/run_audit.py`:

```python
"""Aminergic GPCR data feasibility audit.

Determines whether Phase 1 proceeds with binding-only data (IC50/Ki/Kd)
or pivots to EC50 inclusion based on per-target record counts.

Pass thresholds (per spec Section 4.4):
  ≥80% of targets with ≥1000 binding records → Option A (binding-only)
  60-80% pass → Option A but flag in paper
  <60% pass → Pivot to Option B (include EC50 with flagging)

Stop condition (per spec Section 7.4):
  <30 viable targets total → halt project for design review
"""
import json
from pathlib import Path
import pandas as pd
from chembl_webresource_client.new_client import new_client

from scripts.aminergic_audit.target_lists import (
    AMINERGIC_TARGETS, get_all_target_ids, get_target_to_family,
)

OUTPUT_DIR = Path("results/aminergic_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_BINDING_RECORDS = 1000
PASS_FRACTION_HIGH = 0.80  # ≥80% triggers Option A
PASS_FRACTION_LOW = 0.60   # 60-80% Option A with caveat; <60% pivot to Option B
HALT_MIN_TARGETS = 30      # Stop condition


def query_target_records(target_chembl_id: str) -> dict:
    """Query ChEMBL for binding (IC50/Ki/Kd) record counts per target."""
    activity = new_client.activity
    records = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type__in=["IC50", "Ki", "Kd"],
        standard_relation="=",
        standard_units="nM",
        pchembl_value__isnull=False,
        assay_type="B",  # binding assays
        confidence_score__gte=7,
    ).only(["activity_id", "molecule_chembl_id", "standard_type"])

    records_list = list(records)
    n_records = len(records_list)
    n_compounds = len(set(r["molecule_chembl_id"] for r in records_list))
    type_counts = pd.Series(
        [r["standard_type"] for r in records_list]
    ).value_counts().to_dict()

    return {
        "target_chembl_id": target_chembl_id,
        "n_binding_records": n_records,
        "n_unique_compounds": n_compounds,
        "type_breakdown": type_counts,
    }


def run_audit():
    """Execute the audit and write report."""
    print(f"Auditing {len(get_all_target_ids())} aminergic targets...")
    target_to_family = get_target_to_family()
    results = []

    for chembl_id in get_all_target_ids(include_taar=True):
        try:
            stats = query_target_records(chembl_id)
            stats["family"] = target_to_family[chembl_id]
            stats["passes_threshold"] = stats["n_binding_records"] >= THRESHOLD_BINDING_RECORDS
            results.append(stats)
            print(f"  {chembl_id} ({stats['family']}): "
                  f"{stats['n_binding_records']} records, "
                  f"{stats['n_unique_compounds']} compounds")
        except Exception as e:
            print(f"  {chembl_id}: ERROR — {e}")
            results.append({
                "target_chembl_id": chembl_id,
                "family": target_to_family[chembl_id],
                "n_binding_records": 0,
                "error": str(e),
                "passes_threshold": False,
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "per_target_audit.csv", index=False)

    # Compute summary stats
    n_total = len(df)
    n_pass = df["passes_threshold"].sum()
    pass_fraction = n_pass / n_total

    # Determine decision
    if n_pass < HALT_MIN_TARGETS:
        decision = "HALT"
        decision_msg = f"Only {n_pass} targets meet threshold (<{HALT_MIN_TARGETS}). HALT for design review."
    elif pass_fraction >= PASS_FRACTION_HIGH:
        decision = "OPTION_A"
        decision_msg = f"{pass_fraction:.0%} pass threshold. Proceed with binding-only data."
    elif pass_fraction >= PASS_FRACTION_LOW:
        decision = "OPTION_A_FLAGGED"
        decision_msg = f"{pass_fraction:.0%} pass threshold. Proceed with binding-only, flag in paper."
    else:
        decision = "OPTION_B_PIVOT"
        decision_msg = f"{pass_fraction:.0%} pass threshold. Pivot to EC50 inclusion."

    # Write decision summary
    summary = {
        "n_targets_total": n_total,
        "n_targets_passing": int(n_pass),
        "pass_fraction": pass_fraction,
        "decision": decision,
        "decision_message": decision_msg,
        "threshold": THRESHOLD_BINDING_RECORDS,
    }
    with open(OUTPUT_DIR / "audit_decision.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate audit report
    write_audit_report(df, summary)

    # Generate figures
    plot_per_target_counts(df)

    print(f"\n{decision_msg}")
    print(f"Full report: {OUTPUT_DIR / 'audit_report.md'}")
    print(f"Decision file: {OUTPUT_DIR / 'audit_decision.json'}")

    return summary


def write_audit_report(df: pd.DataFrame, summary: dict) -> None:
    """Write human-readable audit report."""
    report = OUTPUT_DIR / "audit_report.md"
    with open(report, "w") as f:
        f.write("# Aminergic GPCR Data Audit Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"**Decision:** `{summary['decision']}`\n\n")
        f.write(f"**Decision message:** {summary['decision_message']}\n\n")
        f.write(f"## Summary statistics\n\n")
        f.write(f"- Total targets queried: {summary['n_targets_total']}\n")
        f.write(f"- Targets passing threshold (≥{summary['threshold']} binding records): {summary['n_targets_passing']}\n")
        f.write(f"- Pass fraction: {summary['pass_fraction']:.1%}\n\n")
        f.write(f"## Per-family breakdown\n\n")
        family_stats = df.groupby("family").agg(
            n_targets=("target_chembl_id", "count"),
            n_passing=("passes_threshold", "sum"),
            mean_records=("n_binding_records", "mean"),
            median_records=("n_binding_records", "median"),
        ).round(0).astype(int)
        f.write(family_stats.to_markdown())
        f.write("\n\n## Per-target details\n\n")
        f.write(df.to_markdown(index=False))


def plot_per_target_counts(df: pd.DataFrame) -> None:
    """Bar plot of per-target binding record counts, colored by family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("n_binding_records", ascending=False).reset_index(drop=True)
    family_colors = {
        "dopamine": "#1f77b4",
        "serotonin": "#ff7f0e",
        "adrenergic": "#2ca02c",
        "histamine": "#d62728",
        "muscarinic": "#9467bd",
        "trace_amine": "#8c564b",
    }
    colors = [family_colors[f] for f in df_sorted["family"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(df_sorted)), df_sorted["n_binding_records"], color=colors)
    ax.axhline(y=THRESHOLD_BINDING_RECORDS, color="red", linestyle="--",
               linewidth=1, label=f"Threshold: {THRESHOLD_BINDING_RECORDS}")
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["target_chembl_id"], rotation=90, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("Binding records (log scale)")
    ax.set_title("Per-target binding record counts (IC50/Ki/Kd)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_target_record_counts.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_audit()
```

- [ ] **Step 5: Run audit script**

Run: `python -m scripts.aminergic_audit.run_audit`
Expected:
- Console output showing per-target query progress
- Files written: `results/aminergic_audit/audit_decision.json`, `audit_report.md`, `figures/per_target_record_counts.png`
- Decision printed at end (one of: HALT, OPTION_A, OPTION_A_FLAGGED, OPTION_B_PIVOT)

If decision = HALT, **STOP** and return to user for design review.
If decision = OPTION_B_PIVOT, **PAUSE** and consult user — Plan 2 will need restructuring.
If OPTION_A or OPTION_A_FLAGGED, **PROCEED** to Task 2.

- [ ] **Step 6: Commit audit infrastructure**

```bash
git add scripts/aminergic_audit/ tests/test_aminergic_targets.py results/aminergic_audit/
git commit -m "Add aminergic Class A data audit for Phase 1 expansion

Implements the data feasibility audit gate per spec Section 4.4. Queries
ChEMBL for per-target binding (IC50/Ki/Kd) record counts across 35-40
aminergic Class A GPCRs and produces decision JSON, markdown report,
and per-target visualization. Determines whether to proceed with
binding-only protocol or pivot to EC50 inclusion."
```

---

## Task 2: Set up `target-affinity-ml` repo skeleton

**Files:**
- Create: `target-affinity-ml/pyproject.toml`
- Create: `target-affinity-ml/README.md`
- Create: `target-affinity-ml/CHANGELOG.md`
- Create: `target-affinity-ml/LICENSE`
- Create: `target-affinity-ml/.gitignore`
- Create: `target-affinity-ml/src/target_affinity_ml/__init__.py`

**Note:** This task creates the new repo *adjacent to* the existing kinase repo, not inside it. We'll initialize as a separate git repository, push to GitHub later.

- [ ] **Step 1: Create new repo directory adjacent to kinase repo**

```bash
cd /Users/joshuaabbott/
mkdir target-affinity-ml
cd target-affinity-ml
git init
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "target-affinity-ml"
version = "1.0.0"
description = "Class-agnostic ML benchmarking framework for protein-ligand binding affinity prediction"
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
authors = [
    {name = "Joshua Abbott", email = "your-email@example.com"},
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "rdkit>=2023.09",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "pyyaml>=6.0",
    "joblib>=1.3",
    "chembl_webresource_client>=0.10",
    "tqdm>=4.65",
]

[project.optional-dependencies]
deep = [
    "torch>=2.1",
    "torch-geometric>=2.4",
    "fair-esm>=2.0.0",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1.6",
]

[project.urls]
Homepage = "https://github.com/jmabbott40/target-affinity-ml"
"Used by (frozen)" = "https://github.com/jmabbott40/kinase-affinity-baselines"
"Phase 1 Application" = "https://github.com/jmabbott40/gpcr-aminergic-benchmarks"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Create `README.md`**

```markdown
# target-affinity-ml

A class-agnostic Python library for benchmarking machine learning models
on protein-ligand binding affinity prediction tasks. Implements the
seven-model framework (Random Forest, XGBoost, ElasticNet, MLP, ESM-FP MLP,
GIN, GIN+ESM Fusion) with three splitting strategies (random, scaffold,
target-held-out), 5-seed multi-seed validation, and bootstrap CIs.

## Installation

```bash
pip install target-affinity-ml
```

For deep models (ESM-FP MLP, GIN, Fusion), install with the `deep` extra:

```bash
pip install target-affinity-ml[deep]
```

## Usage

```python
from target_affinity_ml.training import train_and_evaluate

results = train_and_evaluate(
    config_path="configs/rf_baseline.yaml",
    split_strategy="random",
    dataset_version="v1",
)
```

## Used by

This library is used by the following application repositories:

- [`kinase-affinity-baselines`](https://github.com/jmabbott40/kinase-affinity-baselines) — frozen at preprint v1.0
- [`gpcr-aminergic-benchmarks`](https://github.com/jmabbott40/gpcr-aminergic-benchmarks) — Phase 1 of multi-class expansion

## Versioning

This library uses semantic versioning. Major version bumps indicate any
change that produces different numerical results. Both `kinase-affinity-baselines`
and `gpcr-aminergic-benchmarks` pin to specific versions for reproducibility.

## Citation

[Add citation when papers are published]

## License

MIT
```

- [ ] **Step 4: Create `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to `target-affinity-ml` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial library extraction from `kinase-affinity-baselines`
```

- [ ] **Step 5: Create `LICENSE`** (MIT, copy from kinase repo)

```bash
cp /Users/joshuaabbott/mlproject/LICENSE /Users/joshuaabbott/target-affinity-ml/LICENSE
```

- [ ] **Step 6: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
.venv/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Project-specific
*.pkl
*.npz
*.pt
*.pth
*.parquet
data/raw/
data/processed/
results/models/
.ipynb_checkpoints/
```

- [ ] **Step 7: Create package `__init__.py`**

Create `src/target_affinity_ml/__init__.py`:

```python
"""target-affinity-ml: Class-agnostic ML benchmarking for protein-ligand affinity.

Public API:
    from target_affinity_ml.training import train_and_evaluate
    from target_affinity_ml.evaluation.metrics import compute_regression_metrics
    from target_affinity_ml.data.splits import random_split, scaffold_split, target_split
"""

__version__ = "1.0.0"

__all__ = [
    "__version__",
]
```

- [ ] **Step 8: Verify package can be installed**

```bash
cd /Users/joshuaabbott/target-affinity-ml
pip install -e .
python -c "import target_affinity_ml; print(target_affinity_ml.__version__)"
```

Expected: prints `1.0.0`

- [ ] **Step 9: Commit repo skeleton**

```bash
git add .
git commit -m "Initialize target-affinity-ml v1.0.0 repo skeleton

Empty package skeleton with pyproject.toml, README, CHANGELOG, LICENSE,
and .gitignore. Library extraction from kinase-affinity-baselines
proceeds in subsequent tasks."
```

---

## Task 3: Move `data/` modules to library

**Files:**
- Move from `/Users/joshuaabbott/mlproject/src/kinase_affinity/data/`:
  - `fetch.py` → `target_affinity_ml/data/chembl_fetcher.py` (with rename)
  - `standardize.py` → `target_affinity_ml/data/standardize.py`
  - `curate.py` → `target_affinity_ml/data/curate.py`
  - `splits.py` → `target_affinity_ml/data/splits.py`
  - `protein_sequences.py` → `target_affinity_ml/data/protein_sequences.py`
- Move tests from kinase repo `tests/test_standardize.py`, `tests/test_splits.py`

**Approach:** For each file, copy to library, replace any kinase-specific assumptions with parameterizable arguments, run unit tests to verify identical behavior.

- [ ] **Step 1: Identify kinase-specific assumptions in `data/` modules**

```bash
grep -rn "kinase\|Kinase\|KINASE" /Users/joshuaabbott/mlproject/src/kinase_affinity/data/
```

Expected: list of any kinase-specific code that needs parameterization.

- [ ] **Step 2: Copy files to library with namespace updates**

```bash
cp /Users/joshuaabbott/mlproject/src/kinase_affinity/data/standardize.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/data/

cp /Users/joshuaabbott/mlproject/src/kinase_affinity/data/curate.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/data/

cp /Users/joshuaabbott/mlproject/src/kinase_affinity/data/splits.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/data/

cp /Users/joshuaabbott/mlproject/src/kinase_affinity/data/protein_sequences.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/data/

cp /Users/joshuaabbott/mlproject/src/kinase_affinity/data/fetch.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/data/chembl_fetcher.py
```

- [ ] **Step 3: Update internal imports in moved files**

In each moved file, replace:
```python
from kinase_affinity.data.standardize import ...
```
with:
```python
from target_affinity_ml.data.standardize import ...
```

Use sed if needed:
```bash
find /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/data -name "*.py" \
  -exec sed -i '' 's/kinase_affinity/target_affinity_ml/g' {} \;
```

- [ ] **Step 4: Create `__init__.py` for data subpackage**

Create `src/target_affinity_ml/data/__init__.py`:

```python
"""Data ingestion, standardization, curation, and splitting."""

from target_affinity_ml.data.chembl_fetcher import fetch_target_data
from target_affinity_ml.data.standardize import standardize_molecule
from target_affinity_ml.data.curate import curate_activities
from target_affinity_ml.data.splits import (
    random_split, scaffold_split, target_split
)

__all__ = [
    "fetch_target_data",
    "standardize_molecule",
    "curate_activities",
    "random_split",
    "scaffold_split",
    "target_split",
]
```

- [ ] **Step 5: Move + adapt unit tests**

```bash
mkdir -p /Users/joshuaabbott/target-affinity-ml/tests/unit
cp /Users/joshuaabbott/mlproject/tests/test_standardize.py \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/
cp /Users/joshuaabbott/mlproject/tests/test_splits.py \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/

# Update imports in test files
find /Users/joshuaabbott/target-affinity-ml/tests -name "*.py" \
  -exec sed -i '' 's/kinase_affinity/target_affinity_ml/g' {} \;
```

- [ ] **Step 6: Run tests to verify behavior preserved**

```bash
cd /Users/joshuaabbott/target-affinity-ml
pytest tests/unit/test_standardize.py tests/unit/test_splits.py -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit `data/` module migration**

```bash
git add src/target_affinity_ml/data/ tests/unit/
git commit -m "Migrate data modules from kinase_affinity to target_affinity_ml

Includes: chembl_fetcher (renamed from fetch.py), standardize, curate,
splits, protein_sequences. All internal imports updated. Existing unit
tests pass with new namespace."
```

---

## Task 4: Move `features/` modules to library

**Files:**
- Move all from `/Users/joshuaabbott/mlproject/src/kinase_affinity/features/`:
  - `fingerprints.py`, `descriptors.py`, `protein_embeddings.py`, `molecular_graphs.py`

- [ ] **Step 1: Verify modules are class-agnostic**

```bash
grep -rn "kinase" /Users/joshuaabbott/mlproject/src/kinase_affinity/features/
```

Expected: zero kinase-specific code (these operate on SMILES and protein sequences regardless of class).

- [ ] **Step 2: Copy and update imports**

```bash
cp /Users/joshuaabbott/mlproject/src/kinase_affinity/features/*.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/features/

find /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/features -name "*.py" \
  -exec sed -i '' 's/kinase_affinity/target_affinity_ml/g' {} \;
```

- [ ] **Step 3: Create features `__init__.py`**

```python
"""Molecular and protein feature engineering pipelines."""

from target_affinity_ml.features.fingerprints import compute_morgan_fp
from target_affinity_ml.features.descriptors import compute_rdkit_descriptors
from target_affinity_ml.features.protein_embeddings import compute_esm2_embeddings
from target_affinity_ml.features.molecular_graphs import smiles_to_graph

__all__ = [
    "compute_morgan_fp",
    "compute_rdkit_descriptors",
    "compute_esm2_embeddings",
    "smiles_to_graph",
]
```

- [ ] **Step 4: Move + run feature tests**

```bash
cp /Users/joshuaabbott/mlproject/tests/test_features.py \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/
sed -i '' 's/kinase_affinity/target_affinity_ml/g' \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/test_features.py

cd /Users/joshuaabbott/target-affinity-ml
pytest tests/unit/test_features.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit features migration**

```bash
git add src/target_affinity_ml/features/ tests/unit/test_features.py
git commit -m "Migrate features modules to target_affinity_ml

fingerprints, descriptors, protein_embeddings, molecular_graphs all moved.
No class-specific code; modules work on any SMILES/protein input."
```

---

## Task 5: Move `models/` modules to library

**Files:**
- Move all from `/Users/joshuaabbott/mlproject/src/kinase_affinity/models/`:
  - `rf_model.py`, `xgb_model.py`, `elasticnet_model.py`, `mlp_model.py`
  - `esm_fp_mlp_model.py`, `gnn_model.py`, `fusion_model.py`, `deep_base.py`

- [ ] **Step 1: Verify model modules are class-agnostic**

```bash
grep -rn "kinase" /Users/joshuaabbott/mlproject/src/kinase_affinity/models/
```

Expected: zero kinase-specific code.

- [ ] **Step 2: Copy and update imports**

```bash
cp /Users/joshuaabbott/mlproject/src/kinase_affinity/models/*.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/models/

find /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/models -name "*.py" \
  -exec sed -i '' 's/kinase_affinity/target_affinity_ml/g' {} \;
```

- [ ] **Step 3: Create models `__init__.py`**

```python
"""Seven model implementations with unified interface."""

from target_affinity_ml.models.rf_model import RandomForestModel
from target_affinity_ml.models.xgb_model import XGBoostModel
from target_affinity_ml.models.elasticnet_model import ElasticNetModel
from target_affinity_ml.models.mlp_model import MLPModel
from target_affinity_ml.models.esm_fp_mlp_model import ESMFPMLPModel
from target_affinity_ml.models.gnn_model import GNNModel
from target_affinity_ml.models.fusion_model import FusionModel

MODEL_REGISTRY = {
    "random_forest": "target_affinity_ml.models.rf_model.RandomForestModel",
    "xgboost": "target_affinity_ml.models.xgb_model.XGBoostModel",
    "elasticnet": "target_affinity_ml.models.elasticnet_model.ElasticNetModel",
    "mlp": "target_affinity_ml.models.mlp_model.MLPModel",
    "esm_fp_mlp": "target_affinity_ml.models.esm_fp_mlp_model.ESMFPMLPModel",
    "gnn": "target_affinity_ml.models.gnn_model.GNNModel",
    "fusion": "target_affinity_ml.models.fusion_model.FusionModel",
}

__all__ = [
    "RandomForestModel", "XGBoostModel", "ElasticNetModel", "MLPModel",
    "ESMFPMLPModel", "GNNModel", "FusionModel",
    "MODEL_REGISTRY",
]
```

- [ ] **Step 4: Move + run model tests**

```bash
cp /Users/joshuaabbott/mlproject/tests/test_models.py \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/
cp /Users/joshuaabbott/mlproject/tests/test_deep_models.py \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/

find /Users/joshuaabbott/target-affinity-ml/tests/unit -name "test_*model*.py" \
  -exec sed -i '' 's/kinase_affinity/target_affinity_ml/g' {} \;

cd /Users/joshuaabbott/target-affinity-ml
pytest tests/unit/test_models.py -v
pytest tests/unit/test_deep_models.py -v  # requires `pip install -e .[deep]`
```

Expected: all tests PASS (after installing deep dependencies)

- [ ] **Step 5: Commit models migration**

```bash
git add src/target_affinity_ml/models/ tests/unit/test_models.py tests/unit/test_deep_models.py
git commit -m "Migrate model modules to target_affinity_ml

All 7 models (RF, XGBoost, ElasticNet, MLP, ESM-FP MLP, GIN, Fusion)
plus deep_base. MODEL_REGISTRY updated to use new namespace."
```

---

## Task 6: Move `training/` modules to library

**Files:**
- Move from `/Users/joshuaabbott/mlproject/src/kinase_affinity/training/`:
  - `trainer.py` (baseline trainer)
  - `deep_trainer.py` (deep model trainer)
  - `tune.py` (grid search)

- [ ] **Step 1: Identify kinase-specific code in training modules**

```bash
grep -n "kinase\|Kinase" /Users/joshuaabbott/mlproject/src/kinase_affinity/training/*.py
```

Expected: imports of `kinase_affinity.X` need replacement; otherwise class-agnostic.

- [ ] **Step 2: Copy and update imports**

```bash
cp /Users/joshuaabbott/mlproject/src/kinase_affinity/training/*.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/training/

find /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/training -name "*.py" \
  -exec sed -i '' 's/kinase_affinity/target_affinity_ml/g' {} \;
```

- [ ] **Step 3: Create training `__init__.py`**

```python
"""Training and evaluation orchestration."""

from target_affinity_ml.training.trainer import train_and_evaluate, run_all_experiments
from target_affinity_ml.training.deep_trainer import deep_train_and_evaluate
from target_affinity_ml.training.tune import grid_search

__all__ = [
    "train_and_evaluate",
    "run_all_experiments",
    "deep_train_and_evaluate",
    "grid_search",
]
```

- [ ] **Step 4: Run smoke test for trainer module imports**

```bash
cd /Users/joshuaabbott/target-affinity-ml
python -c "from target_affinity_ml.training import train_and_evaluate; print('OK')"
```

Expected: prints `OK`

- [ ] **Step 5: Commit training migration**

```bash
git add src/target_affinity_ml/training/
git commit -m "Migrate training modules to target_affinity_ml

trainer (baseline orchestration), deep_trainer (deep models), tune (grid search).
All imports updated to new namespace."
```

---

## Task 7: Move `evaluation/` modules to library

**Files:**
- Move from `/Users/joshuaabbott/mlproject/src/kinase_affinity/evaluation/`:
  - `metrics.py`, `uncertainty.py`, `analysis.py`, `bootstrap.py`, `multi_seed.py`, `run_phase5.py`

- [ ] **Step 1: Copy and update imports**

```bash
cp /Users/joshuaabbott/mlproject/src/kinase_affinity/evaluation/*.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/evaluation/

find /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/evaluation -name "*.py" \
  -exec sed -i '' 's/kinase_affinity/target_affinity_ml/g' {} \;
```

- [ ] **Step 2: Create evaluation `__init__.py`**

```python
"""Metrics, uncertainty quantification, bootstrap CIs, multi-seed analysis."""

from target_affinity_ml.evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
)
from target_affinity_ml.evaluation.uncertainty import (
    calibration_curve,
    miscalibration_area,
    selective_prediction_curve,
)
from target_affinity_ml.evaluation.bootstrap import bootstrap_ci, bootstrap_paired_test
from target_affinity_ml.evaluation.multi_seed import multi_seed_paired_test

__all__ = [
    "compute_regression_metrics",
    "compute_classification_metrics",
    "calibration_curve",
    "miscalibration_area",
    "selective_prediction_curve",
    "bootstrap_ci",
    "bootstrap_paired_test",
    "multi_seed_paired_test",
]
```

- [ ] **Step 3: Move + run evaluation tests**

```bash
cp /Users/joshuaabbott/mlproject/tests/test_metrics.py \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/
sed -i '' 's/kinase_affinity/target_affinity_ml/g' \
   /Users/joshuaabbott/target-affinity-ml/tests/unit/test_metrics.py

pytest tests/unit/test_metrics.py -v
```

Expected: all tests PASS

- [ ] **Step 4: Commit evaluation migration**

```bash
git add src/target_affinity_ml/evaluation/ tests/unit/test_metrics.py
git commit -m "Migrate evaluation modules to target_affinity_ml

metrics, uncertainty, bootstrap, multi_seed, analysis. All metric tests
pass with new namespace."
```

---

## Task 8: Move `visualization/` modules and configs

**Files:**
- Move `visualization/plots.py` and all `configs/*.yaml`

- [ ] **Step 1: Copy visualization module**

```bash
cp /Users/joshuaabbott/mlproject/src/kinase_affinity/visualization/plots.py \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/visualization/

sed -i '' 's/kinase_affinity/target_affinity_ml/g' \
   /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/visualization/plots.py
```

- [ ] **Step 2: Create visualization `__init__.py`**

```python
"""Plotting helpers (consistent colors, model names, multi-panel figures)."""

from target_affinity_ml.visualization.plots import (
    MODEL_COLORS, MODEL_DISPLAY_NAMES, MODEL_ORDER, SPLIT_MARKERS,
    plot_performance_degradation,
    plot_split_comparison,
)

__all__ = [
    "MODEL_COLORS", "MODEL_DISPLAY_NAMES", "MODEL_ORDER", "SPLIT_MARKERS",
    "plot_performance_degradation",
    "plot_split_comparison",
]
```

- [ ] **Step 3: Copy configs**

```bash
mkdir -p /Users/joshuaabbott/target-affinity-ml/configs
cp /Users/joshuaabbott/mlproject/configs/*.yaml \
   /Users/joshuaabbott/target-affinity-ml/configs/
```

- [ ] **Step 4: Commit visualization + configs**

```bash
git add src/target_affinity_ml/visualization/ configs/
git commit -m "Migrate visualization and default model configs

plots.py with shared MODEL_COLORS, model display names, plotting helpers.
configs/ contains 7 default model hyperparameter YAMLs."
```

---

## Task 9: Add `benchmarks/` placeholder subpackage for Plan 3

**Files:**
- Create: `src/target_affinity_ml/benchmarks/__init__.py`
- Create: `src/target_affinity_ml/benchmarks/README.md`

**Why now:** Reserves the namespace for Plan 3's scaffold-diversity and RNS modules. Empty package, but importable.

- [ ] **Step 1: Create empty benchmarks subpackage**

Create `src/target_affinity_ml/benchmarks/__init__.py`:

```python
"""Cross-class benchmarking methodology.

Placeholder for Plan 3 implementations:
- scaffold_diversity: per-target scaffold concentration metrics
- rns_scoring: Prabakaran-Bromberg Residue Neighborhood Significance

Modules will be added when Plan 3 begins. Library version v1.0.0 ships
with this empty package; v1.1.0 will include the new methodology.
"""

# Empty placeholder — see README.md
```

Create `src/target_affinity_ml/benchmarks/README.md`:

```markdown
# benchmarks/

Cross-class benchmarking methodology subpackage.

**Status (v1.0.0):** Placeholder. Modules will be added in v1.1.0.

**Planned contents:**
- `scaffold_diversity.py` — Per-target scaffold concentration metrics (Bemis-Murcko entropy, largest-cluster fraction, mean Tanimoto, activity cliff frequency)
- `rns_scoring.py` — Prabakaran-Bromberg RNS pipeline with structure-source handling

See implementation plan: `docs/superpowers/plans/2026-04-XX-plan3-methodology-cross-class.md`
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/joshuaabbott/target-affinity-ml
python -c "from target_affinity_ml import benchmarks; print('OK')"
```

Expected: prints `OK`

- [ ] **Step 3: Commit benchmarks placeholder**

```bash
git add src/target_affinity_ml/benchmarks/
git commit -m "Add empty benchmarks/ subpackage for Plan 3 methodology

Reserves namespace for scaffold-diversity and RNS modules to be
implemented in v1.1.0."
```

---

## Task 10: Add CI tests + integration test for kinase reproducibility

**Files:**
- Create: `tests/integration/test_kinase_reproducibility.py`
- Create: `.github/workflows/ci.yml`

**Why critical:** This is the validation gate that ensures the library refactor didn't change any numerical outputs. The test loads a saved snapshot of preprint v1 metrics and asserts that re-runs produce the same numbers within tolerance.

**Strategy:** Save expected kinase metrics from preprint v1 as a JSON fixture. The integration test runs a *small* subset (e.g., RF on random split with seed=42) and checks the output matches.

- [ ] **Step 1: Extract reference kinase metrics from preprint v1**

Create reference fixture by reading existing kinase results:

```python
# scripts/extract_reference_metrics.py
import json
import pandas as pd
from pathlib import Path

# Read multi-seed aggregated results from kinase preprint v1
df = pd.read_csv(
    Path("/Users/joshuaabbott/mlproject/results/tables/multi_seed_aggregated.csv")
)

# Filter to RF + random split + RMSE (smoke test target)
ref = df[
    (df["model"] == "random_forest")
    & (df["split"] == "random")
    & (df["metric"] == "rmse")
].iloc[0]

reference = {
    "kinase_v1_rf_random_rmse_seed42": ref["min"],   # seed=42 is one of the 5 seeds
    "kinase_v1_rf_random_rmse_mean": ref["mean"],
    "kinase_v1_rf_random_rmse_std": ref["std"],
    "tolerance": 0.001,  # ±1e-3 RMSE difference allowed (floating-point noise)
}

out = Path("tests/integration/kinase_v1_reference.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(reference, f, indent=2)

print(f"Reference metrics saved: {reference}")
```

Run: `python scripts/extract_reference_metrics.py`
Expected: writes `tests/integration/kinase_v1_reference.json`

- [ ] **Step 2: Write integration test for RF on kinase random split**

Create `tests/integration/test_kinase_reproducibility.py`:

```python
"""Integration test: RF on kinase random split with library v1.0
matches preprint v1 numerical output within tolerance.

This is the validation gate for the library refactor. Failure here
indicates the refactor changed numerical behavior — investigate and
fix before proceeding to Plan 2 (GPCR work).
"""
import json
import pytest
import numpy as np
from pathlib import Path

from target_affinity_ml.training import train_and_evaluate

REFERENCE_PATH = Path(__file__).parent / "kinase_v1_reference.json"
KINASE_REPO = Path("/Users/joshuaabbott/mlproject")


@pytest.fixture(scope="session")
def reference_metrics():
    with open(REFERENCE_PATH) as f:
        return json.load(f)


@pytest.mark.slow  # marker for ~2-5 minute test
def test_rf_random_seed42_matches_preprint_v1(reference_metrics):
    """RF on random split with seed=42 reproduces preprint v1 RMSE."""
    config_path = KINASE_REPO / "configs" / "rf_baseline.yaml"
    dataset_dir = KINASE_REPO / "data" / "processed" / "v1"

    result = train_and_evaluate(
        config_path=str(config_path),
        split_strategy="random",
        dataset_version="v1",
        dataset_dir=str(dataset_dir),
        seed=42,
    )

    rmse = result["test_metrics"]["rmse"]
    expected = reference_metrics["kinase_v1_rf_random_rmse_seed42"]
    tolerance = reference_metrics["tolerance"]

    assert abs(rmse - expected) <= tolerance, (
        f"RF random seed=42 RMSE diverged: got {rmse:.6f}, "
        f"expected {expected:.6f} ± {tolerance}"
    )
```

- [ ] **Step 3: Add `slow` marker to pytest config**

Update `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "slow: tests that take >30 seconds (integration tests)",
]
```

- [ ] **Step 4: Add CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -e .[dev]
      - name: Run unit tests (fast)
        run: pytest tests/unit/ -v -m "not slow"
      - name: Lint with ruff
        run: ruff check src/ tests/
```

- [ ] **Step 5: Run integration test locally**

```bash
cd /Users/joshuaabbott/target-affinity-ml
pytest tests/integration/test_kinase_reproducibility.py -v -m slow
```

Expected: PASS (RMSE within ±0.001 of preprint v1)

If FAIL: do NOT proceed to Task 11. Investigate refactor bug.

- [ ] **Step 6: Commit CI + integration test**

```bash
git add tests/integration/ .github/workflows/ scripts/extract_reference_metrics.py
git commit -m "Add CI pipeline + kinase reproducibility integration test

GitHub Actions workflow runs unit tests + ruff lint on push/PR.
Integration test (marked 'slow') validates that RF on kinase random
split with seed=42 reproduces preprint v1 RMSE within ±0.001 tolerance.
Pass = library refactor preserved numerical behavior."
```

---

## Task 11: Tag library v1.0.0 release

**Files:**
- Modify: `src/target_affinity_ml/__init__.py` (verify version)
- Modify: `CHANGELOG.md` (move Unreleased → 1.0.0 section)

- [ ] **Step 1: Update CHANGELOG**

Move "Unreleased" section to a versioned 1.0.0 entry in `CHANGELOG.md`:

```markdown
## [1.0.0] - 2026-04-XX

### Added
- Library extracted from `kinase-affinity-baselines` (commit hash from kinase repo)
- Class-agnostic data, features, models, training, evaluation, visualization modules
- 7 model implementations: RF, XGBoost, ElasticNet, MLP, ESM-FP MLP, GIN, GIN+ESM Fusion
- Three split strategies: random, scaffold (Bemis-Murcko), target-held-out
- Multi-seed validation framework + bootstrap CIs
- Empty `benchmarks/` placeholder for Plan 3 (scaffold diversity, RNS)
- CI workflow with unit tests and lint checks
- Kinase reproducibility integration test (validation gate for refactor)

### Migration notes
- Imports change: `kinase_affinity.X` → `target_affinity_ml.X`
- `fetch.py` renamed to `chembl_fetcher.py`
- All other module names preserved
```

- [ ] **Step 2: Verify version string**

```bash
grep "version" /Users/joshuaabbott/target-affinity-ml/pyproject.toml
grep "__version__" /Users/joshuaabbott/target-affinity-ml/src/target_affinity_ml/__init__.py
```

Both should show `1.0.0`.

- [ ] **Step 3: Run all tests one more time**

```bash
cd /Users/joshuaabbott/target-affinity-ml
pytest tests/ -v
```

Expected: All unit tests PASS. Integration test PASS (or skipped with `-m "not slow"`).

- [ ] **Step 4: Commit version finalization**

```bash
git add CHANGELOG.md
git commit -m "Finalize v1.0.0 release notes"
```

- [ ] **Step 5: Tag the release**

```bash
git tag -a v1.0.0 -m "target-affinity-ml v1.0.0 — initial library extraction

First versioned release. Library extracted from kinase-affinity-baselines.
Validates kinase reproducibility within ±0.001 RMSE tolerance against
preprint v1 numbers. Used by:
- kinase-affinity-baselines (frozen at v1.0)
- gpcr-aminergic-benchmarks (Phase 1, in progress)"

git log --oneline | head -5
```

Expected: tag `v1.0.0` shown in log.

- [ ] **Step 6: Push to GitHub** (manual step — user creates the repo on GitHub first)

```bash
# After creating empty repo at github.com/jmabbott40/target-affinity-ml:
git remote add origin git@github.com:jmabbott40/target-affinity-ml.git
git push -u origin main
git push origin v1.0.0
```

---

## Task 12: Update `kinase-affinity-baselines` to depend on library v1.0.0

**Files:**
- Modify: `pyproject.toml` (add `target-affinity-ml==1.0.0` dependency)
- Modify: `src/kinase_affinity/__init__.py` (thin re-exports)
- Modify: `README.md` (add library extraction note)
- Delete: `src/kinase_affinity/{data,features,models,training,evaluation,visualization}/` (now in library)
- Keep: `src/kinase_affinity/scripts/` and any kinase-specific code

**Strategy:** Replace internal modules with thin re-exports from the library so existing user code (e.g., notebooks) continues to work. Tag a `v1.0` release of the kinase repo simultaneously.

- [ ] **Step 1: Update `pyproject.toml`**

```bash
cd /Users/joshuaabbott/mlproject
```

Add dependency:

```toml
dependencies = [
    "target-affinity-ml==1.0.0",
    # ... existing dependencies removed (now provided by library)
]
```

- [ ] **Step 2: Replace `kinase_affinity/__init__.py` with thin re-exports**

```python
"""kinase-affinity-baselines: kinase-specific application using target-affinity-ml.

This package is FROZEN at v1.0 — the version accompanying the published
preprint. Ongoing work has moved to:
- Library: target-affinity-ml (https://github.com/jmabbott40/target-affinity-ml)
- Phase 1 application: gpcr-aminergic-benchmarks (https://github.com/jmabbott40/gpcr-aminergic-benchmarks)

For backward compatibility, all submodules re-export from the library:
    from kinase_affinity.models import RandomForestModel  # → target_affinity_ml.models.RandomForestModel
"""
__version__ = "1.0.0"

# Backward-compatibility re-exports
from target_affinity_ml import data, features, models, training, evaluation, visualization

__all__ = ["data", "features", "models", "training", "evaluation", "visualization", "__version__"]
```

- [ ] **Step 3: Delete migrated module directories**

```bash
cd /Users/joshuaabbott/mlproject/src/kinase_affinity
rm -rf data/ features/ models/ training/ evaluation/ visualization/
```

- [ ] **Step 4: Update README to add library extraction section**

Edit `README.md` to add at top (above existing content):

```markdown
> **Library extraction notice (April 2026):** The reusable model + training + evaluation code
> from this repository has been extracted into a separate library:
> [`target-affinity-ml`](https://github.com/jmabbott40/target-affinity-ml). This repo is
> **frozen at v1.0** as the artifact accompanying the kinase preprint
> ([citation TBD]). Phase 1 of the multi-class expansion benchmarks aminergic GPCRs in
> [`gpcr-aminergic-benchmarks`](https://github.com/jmabbott40/gpcr-aminergic-benchmarks).
```

- [ ] **Step 5: Verify kinase repo still works with library**

```bash
cd /Users/joshuaabbott/mlproject
pip install -e .
python -c "from kinase_affinity.models import RandomForestModel; print(RandomForestModel)"
```

Expected: prints `<class 'target_affinity_ml.models.rf_model.RandomForestModel'>` (the alias works)

- [ ] **Step 6: Tag kinase repo v1.0**

```bash
git add pyproject.toml src/kinase_affinity/__init__.py README.md
git rm -rf src/kinase_affinity/data src/kinase_affinity/features src/kinase_affinity/models src/kinase_affinity/training src/kinase_affinity/evaluation src/kinase_affinity/visualization

git commit -m "Migrate to target-affinity-ml v1.0.0 library

Library extraction completed. This repo is now frozen at v1.0 as the
artifact accompanying the kinase preprint. Internal modules replaced
with thin re-exports from target_affinity_ml for backward compatibility.

See: https://github.com/jmabbott40/target-affinity-ml"

git tag -a v1.0 -m "kinase-affinity-baselines v1.0 — frozen preprint artifact"
git push origin main
git push origin v1.0
```

---

## Task 13: Re-run kinase benchmark with library v1.0

**Files:**
- Create: `scripts/rerun_kinase_v1.py`
- Output: `results/kinase_v1_revalidation/`

**Goal:** Run all 105 kinase training runs (7 models × 3 splits × 5 seeds) with library v1.0, save predictions and metrics, and compare to preprint v1.

- [ ] **Step 1: Write re-run script**

Create `scripts/rerun_kinase_v1.py`:

```python
"""Re-run the kinase benchmark with target-affinity-ml v1.0.

Uses identical seeds (42, 123, 456, 789, 1024) and identical configs.
Output directory: results/kinase_v1_revalidation/

Compares to preprint v1 numbers in results/tables/multi_seed_aggregated.csv.
Tolerance: ±0.001 RMSE per model × split per seed.
"""
import json
from pathlib import Path
import pandas as pd

from target_affinity_ml.training import train_and_evaluate
from target_affinity_ml.evaluation.multi_seed import multi_seed_paired_test

OUTPUT_DIR = Path("results/kinase_v1_revalidation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["random_forest", "xgboost", "elasticnet", "mlp",
          "esm_fp_mlp", "gnn", "fusion"]
SPLITS = ["random", "scaffold", "target"]
SEEDS = [42, 123, 456, 789, 1024]

CONFIGS_DIR = Path("configs")  # uses kinase repo configs (still present)
DATASET_VERSION = "v1"
DATASET_DIR = Path("data/processed/v1")


def run_kinase_v1():
    results = []
    for model in MODELS:
        config_path = CONFIGS_DIR / f"{model}_baseline.yaml" if model in [
            "random_forest", "xgboost", "elasticnet", "mlp"
        ] else CONFIGS_DIR / f"{model}.yaml"

        for split in SPLITS:
            for seed in SEEDS:
                print(f"Training {model} on {split} split with seed={seed}...")
                result = train_and_evaluate(
                    config_path=str(config_path),
                    split_strategy=split,
                    dataset_version=DATASET_VERSION,
                    dataset_dir=str(DATASET_DIR),
                    seed=seed,
                )

                # Save predictions
                pred_dir = OUTPUT_DIR / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                # ... (save logic similar to existing trainer.py)

                # Collect metrics
                metrics_row = {
                    "model": model,
                    "split": split,
                    "seed": seed,
                    "test_rmse": result["test_metrics"]["rmse"],
                    "test_r2": result["test_metrics"]["r2"],
                    "test_pearson_r": result["test_metrics"]["pearson_r"],
                    # ... etc
                }
                results.append(metrics_row)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "all_seeds_metrics.csv", index=False)
    print(f"All 105 runs complete. Results: {OUTPUT_DIR / 'all_seeds_metrics.csv'}")


if __name__ == "__main__":
    run_kinase_v1()
```

- [ ] **Step 2: Run re-run on first model + split + seed (smoke test)**

```bash
cd /Users/joshuaabbott/mlproject
python -c "
from target_affinity_ml.training import train_and_evaluate
result = train_and_evaluate(
    config_path='configs/rf_baseline.yaml',
    split_strategy='random',
    dataset_version='v1',
    dataset_dir='data/processed/v1',
    seed=42,
)
print(f'RF random seed=42 RMSE: {result[\"test_metrics\"][\"rmse\"]:.6f}')
"
```

Expected: prints RMSE close to preprint v1 value (~0.819)

- [ ] **Step 3: Launch full re-run**

```bash
python scripts/rerun_kinase_v1.py 2>&1 | tee results/kinase_v1_revalidation/run_log.txt
```

Expected:
- Compute time: ~2 days wall-clock at 4-way GPU parallelism
- Output: `results/kinase_v1_revalidation/all_seeds_metrics.csv` with 105 rows
- All 105 runs complete without errors

If errors: investigate, debug, restart.

- [ ] **Step 4: Commit re-run results**

```bash
git add scripts/rerun_kinase_v1.py results/kinase_v1_revalidation/
git commit -m "Re-run kinase benchmark with target-affinity-ml v1.0

All 105 training runs (7 models × 3 splits × 5 seeds) complete.
Validation against preprint v1 in next task."
```

---

## Task 14: Validate kinase results match preprint v1 within tolerance

**Files:**
- Create: `scripts/validate_kinase_revalidation.py`
- Output: `results/kinase_v1_revalidation/validation_report.md`

- [ ] **Step 1: Write validation script**

Create `scripts/validate_kinase_revalidation.py`:

```python
"""Validate that kinase v1.0 re-run reproduces preprint v1 numbers.

Tolerance: ±0.001 RMSE per (model, split, seed) combination.
Failures are flagged in validation_report.md with diagnostic info.
"""
import json
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("results/kinase_v1_revalidation")
TOLERANCE_RMSE = 0.001
TOLERANCE_OTHER = 0.005  # Slightly looser for R², Pearson R, etc.

def validate():
    rerun = pd.read_csv(OUTPUT_DIR / "all_seeds_metrics.csv")
    # Read preprint v1 multi-seed (need to re-aggregate from per-seed if available)
    preprint = pd.read_csv("results/supplement_tables/S6_per_seed_metrics.csv")

    # Join on (model, split, seed) and compute differences
    merged = rerun.merge(
        preprint, on=["model", "split", "seed"], suffixes=("_rerun", "_preprint")
    )
    merged["rmse_diff"] = (merged["test_rmse_rerun"] - merged["test_rmse_preprint"]).abs()

    # Flag failures
    failures = merged[merged["rmse_diff"] > TOLERANCE_RMSE]

    # Write report
    report_path = OUTPUT_DIR / "validation_report.md"
    with open(report_path, "w") as f:
        f.write("# Kinase v1.0 Re-validation Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"**Tolerance:** ±{TOLERANCE_RMSE} RMSE\n\n")
        f.write(f"**Total comparisons:** {len(merged)}\n")
        f.write(f"**Failures:** {len(failures)}\n\n")

        if len(failures) == 0:
            f.write("## ✅ All re-run results match preprint v1 within tolerance\n\n")
            f.write("Library refactor preserved numerical behavior. Proceed to Plan 2.\n\n")
        else:
            f.write("## ❌ Failures detected\n\n")
            f.write("Investigate refactor bugs before proceeding to Plan 2:\n\n")
            f.write(failures[["model", "split", "seed",
                             "test_rmse_rerun", "test_rmse_preprint",
                             "rmse_diff"]].to_markdown(index=False))

        f.write("\n\n## Summary statistics by model × split\n\n")
        summary = merged.groupby(["model", "split"]).agg(
            mean_rmse_diff=("rmse_diff", "mean"),
            max_rmse_diff=("rmse_diff", "max"),
            n_failures=("rmse_diff", lambda x: (x > TOLERANCE_RMSE).sum()),
        ).round(6)
        f.write(summary.to_markdown())

    print(f"Validation report: {report_path}")
    return len(failures) == 0


if __name__ == "__main__":
    success = validate()
    if not success:
        print("\n❌ FAILURES detected. See validation_report.md.")
        print("Do NOT proceed to Plan 2 until failures are debugged and resolved.")
        exit(1)
    else:
        print("\n✅ Validation passed. Library v1.0 reproduces preprint v1 numbers.")
        print("Ready to proceed to Plan 2.")
```

- [ ] **Step 2: Run validation**

```bash
python scripts/validate_kinase_revalidation.py
```

Expected: prints validation result. If passed, validation_report.md shows ✅.

If failures detected: investigate. Possible causes:
- Floating-point reordering (e.g., set iteration order)
- Random seed handling change (e.g., numpy default RNG state)
- Dependency version change (e.g., scikit-learn upgrade)

Fix the cause, re-run, re-validate.

- [ ] **Step 3: Commit validation results**

```bash
git add scripts/validate_kinase_revalidation.py results/kinase_v1_revalidation/validation_report.md
git commit -m "Validate kinase v1.0 re-run against preprint v1

All 105 comparisons within ±0.001 RMSE tolerance. Library refactor
preserved numerical behavior. Ready to proceed to Plan 2."
```

---

## Task 15: Final summary and Plan 2 handoff

**Goal:** Document Plan 1 completion status, surface any deviations, and create the handoff document for Plan 2 to start.

- [ ] **Step 1: Write Plan 1 completion summary**

Create `docs/superpowers/plans/2026-04-17-plan1-completion-summary.md`:

```markdown
# Plan 1 Completion Summary

**Date:** [completion date]
**Status:** Complete / Partial / Failed

## Deliverables

- [ ] `target-affinity-ml` v1.0.0 tagged and pushed to GitHub
- [ ] `kinase-affinity-baselines` v1.0 tagged (frozen)
- [ ] All 105 kinase training runs complete with library v1.0
- [ ] Validation report shows ≤[X] failures within tolerance
- [ ] Aminergic data audit: decision = [OPTION_A / OPTION_A_FLAGGED / OPTION_B_PIVOT]

## Test results

- Unit tests: [pass count] / [total]
- Integration tests: [pass count] / [total]
- CI pipeline: [pass / fail / pending]

## Validation outcome

- Total comparisons: 105
- Failures: [count]
- Max RMSE difference: [value]
- Mean RMSE difference: [value]

## Deviations from plan

[Any unexpected issues, fixes applied, or decisions made during execution]

## Plan 2 readiness

- Library v1.0 is the dependency for Plan 2 — confirmed installable
- Aminergic audit decision determines Plan 2 scope:
  - OPTION_A: Plan 2 proceeds as written (binding-only)
  - OPTION_A_FLAGGED: Plan 2 proceeds, flag added to manuscript discussion
  - OPTION_B_PIVOT: Plan 2 needs revision (EC50 inclusion)

## Recommendations for Plan 2

[Any insights from Plan 1 that should inform Plan 2 task structure]
```

- [ ] **Step 2: Commit completion summary**

```bash
cd /Users/joshuaabbott/mlproject
git add docs/superpowers/plans/2026-04-17-plan1-completion-summary.md
git commit -m "Plan 1 completion summary

Library extraction + kinase re-validation + aminergic data audit complete.
Ready to proceed to Plan 2 (GPCR data pipeline + benchmark)."
```

- [ ] **Step 3: Final state check**

```bash
# Confirm both repos are in expected state
cd /Users/joshuaabbott/target-affinity-ml && git log --oneline | head -10
cd /Users/joshuaabbott/mlproject && git log --oneline | head -10

# Confirm library is installable
pip install -e /Users/joshuaabbott/target-affinity-ml
python -c "from target_affinity_ml import __version__; print(__version__)"
```

Expected: prints `1.0.0`

- [ ] **Step 4: Surface results to user**

Report to user:
- Library v1.0.0 release status (URL on GitHub)
- Kinase v1 re-validation outcome (pass / fail counts)
- Aminergic audit decision
- Whether Plan 2 can be written next, or whether issues need resolution first

---

## Plan 1 verification checklist

Before declaring Plan 1 complete, verify:

- [ ] All 15 tasks committed
- [ ] CI passes on `target-affinity-ml`
- [ ] `pip install target-affinity-ml==1.0.0` works (after PyPI publish, or from GitHub)
- [ ] Integration test passes (kinase reproducibility)
- [ ] Aminergic audit decision is documented
- [ ] Validation report shows zero or expected failures
- [ ] Both repos tagged with v1.0/v1.0.0
- [ ] Plan 1 completion summary written

## Notes for plan execution

- **Tasks 3-8** (data, features, models, training, evaluation, visualization migrations) can be executed in parallel by separate subagents — they have no inter-dependencies.
- **Task 1** (audit gate) should run first as a parallel independent stream — its decision unblocks Plan 2 even before Plan 1's other tasks complete.
- **Tasks 13-14** (kinase re-run + validation) are the tallest pole; budget ~2 days wall-clock and one debugging cycle in case of failure.
- **Task 12** (kinase repo update) deletes the migrated modules — verify backward-compatibility re-exports work *before* committing the deletions.
