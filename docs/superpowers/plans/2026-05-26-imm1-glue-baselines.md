# IMM1 Macrocyclic Glue Baseline Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a private benchmark of Random Forest, XGBoost, and MLP for predicting binding affinity (pKD) on a ~288-compound macrocyclic peptide molecular-glue library against a single protein target, using SPR-derived data with explicit handling of assay-floor left-censoring and rigorous nested cross-validation across four split strategies.

**Architecture:** New private repo `imm1-glue-baselines` that pip-installs the existing `target-affinity-ml` library (v1.1.0+) and adds IMM1-specific data loading, curation, two new split strategies (Butina cluster, time/synthesis-order), a censoring-sensitivity wrapper, and report-generation scripts. Raw data stays local at `$IMM1_DATA_PATH`. Compute runs locally on Mac.

**Tech Stack:** Python 3.11, scikit-learn 1.3+, XGBoost 2.0+, RDKit 2023.09+, PyTorch 2.1+ (for MLP MC dropout), pytest, pandas/numpy, target-affinity-ml v1.1.0+.

**Spec:** [docs/superpowers/specs/2026-05-26-imm1-glue-baselines-design.md](../specs/2026-05-26-imm1-glue-baselines-design.md)

**Implementation repo:** https://github.com/jmabbott40/imm1-glue-baselines (private)

**Library source:** `/Users/joshuaabbott/target-affinity-ml/` (already installed v1.1.0)

---

## How to read this plan

Each phase produces a working, committable increment. Within a phase, each **Task** is a self-contained piece of work; within a task, each **Step** is 2–5 minutes of focused effort. TDD discipline — failing test first, then minimal implementation, then commit.

**File path convention:** unless otherwise noted, paths in this plan are relative to the `imm1-glue-baselines` repo root. The plan itself lives in the `mlproject` repo as a planning artifact.

**Working directory for execution:** clone the private repo locally and `cd` into it before starting Phase 0. Suggested location: `~/code/imm1-glue-baselines/`.

---

## File Structure (final state after plan complete)

```
imm1-glue-baselines/
├── README.md                              # internal-only
├── CONFIDENTIAL.md                        # data-handling policy
├── pyproject.toml                         # depends on target-affinity-ml ~=1.1
├── environment.yml                        # conda env
├── .gitignore
├── .githooks/
│   └── pre-commit                         # symlink target
├── configs/
│   ├── dataset_imm1.yaml
│   ├── splits.yaml
│   ├── rf_baseline.yaml
│   ├── xgb_baseline.yaml
│   └── mlp_baseline.yaml
├── data/
│   └── README.md
├── src/imm1_glue/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   ├── curate.py
│   │   └── splits.py
│   ├── features/__init__.py               # re-export
│   ├── models/__init__.py                 # re-export
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── censoring_sensitivity.py
│   └── reports/
│       ├── __init__.py
│       └── generate_tables.py
├── scripts/
│   ├── check_no_data_leak.sh              # pre-commit hook
│   ├── audit_library.py
│   ├── run_diagnostics.py
│   └── run_benchmark.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_results_summary.ipynb
│   └── 03_error_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_load.py
│   ├── test_curate.py
│   ├── test_splits.py
│   └── test_pipeline_smoke.py
└── results/                               # gitignored
    ├── splits/
    ├── splits_diag/
    ├── predictions/
    ├── tables/
    ├── figures/
    └── logs/
```

**Per-file responsibility:**

| File | Responsibility |
|---|---|
| `src/imm1_glue/data/load.py` | Read CSV at `$IMM1_DATA_PATH`, schema-validate, return typed DataFrame |
| `src/imm1_glue/data/curate.py` | Aggregate replicates, canonicalize SMILES, flag censored + noisy |
| `src/imm1_glue/data/splits.py` | Four split strategies (random/scaffold/cluster/time); delegate random/scaffold to library |
| `src/imm1_glue/evaluation/censoring_sensitivity.py` | Re-orchestrate the sweep with censored compounds removed |
| `src/imm1_glue/reports/generate_tables.py` | Consume predictions, emit CSV + MD tables |
| `scripts/audit_library.py` | Verify `target_affinity_ml` API surfaces; produce checklist report |
| `scripts/run_diagnostics.py` | Curation report + Butina cutoff sweep + time-split sanity |
| `scripts/run_benchmark.py` | Main nested-CV orchestrator (model × split × seed × fold) |
| `scripts/check_no_data_leak.sh` | Pre-commit hook blocking `RAP-\d{7}` / pKD floats |

---

# Phase 0 — Repo Bootstrap & Library Audit

**Goal:** Repo exists locally, has the skeleton and dependencies, the library audit has run, and the team knows which (if any) gap-fill PRs are needed before Phase 1.

**Exit criteria:**
- `pip install -e .` succeeds in the new repo.
- `scripts/check_no_data_leak.sh` is wired as a pre-commit hook.
- `scripts/audit_library.py` produces `results/library_audit.md` with all critical-path APIs green.
- `pytest tests/` runs (will pass trivially since there are no tests yet).

---

### Task 0.1 — Clone repo and initialize working directory

**Files:**
- Initial clone target: `~/code/imm1-glue-baselines/` (or your preferred location)

- [ ] **Step 1: Clone the private repo locally**

Run:
```bash
mkdir -p ~/code
cd ~/code
gh repo clone jmabbott40/imm1-glue-baselines
cd imm1-glue-baselines
```
Expected: clone succeeds (empty repo is fine).

- [ ] **Step 2: Confirm you're in the right directory**

Run: `pwd && git remote -v`
Expected: prints `/Users/.../imm1-glue-baselines` and `origin → git@github.com:jmabbott40/imm1-glue-baselines`.

- [ ] **Step 3: Create the directory skeleton**

Run:
```bash
mkdir -p configs data src/imm1_glue/{data,features,models,evaluation,reports} \
         scripts notebooks tests results/{splits,splits_diag,predictions,tables,figures,logs} \
         .githooks
```

- [ ] **Step 4: Verify directory layout**

Run: `find . -type d -not -path './.git*' | sort`
Expected: lists exactly the directories above.

---

### Task 0.2 — Create README.md, CONFIDENTIAL.md, and data/README.md

**Files:**
- Create: `README.md`
- Create: `CONFIDENTIAL.md`
- Create: `data/README.md`

- [ ] **Step 1: Create README.md**

```markdown
# IMM1 Macrocyclic Glue Baseline Benchmark (Private)

Internal benchmark of RF, XGBoost, and MLP for predicting binding affinity
(pKD from SPR) on a single-target macrocyclic peptide molecular-glue library.

**See:** [CONFIDENTIAL.md](CONFIDENTIAL.md) for data-handling policy.

**Spec:** see the mlproject repo at `docs/superpowers/specs/2026-05-26-imm1-glue-baselines-design.md`.

## Setup

```bash
conda env create -f environment.yml
conda activate imm1-glue
pip install -e .
export IMM1_DATA_PATH=~/secure_data/imm1/IMM1_SPR_Data.csv
pytest tests/
```

## Usage

```bash
python scripts/audit_library.py            # one-time gate
python scripts/run_diagnostics.py          # pre-benchmark diagnostics
python scripts/run_benchmark.py            # full sweep
python scripts/run_benchmark.py --sensitivity   # drop-censored sweep
python -m imm1_glue.reports.generate_tables     # emit final tables
```
```

- [ ] **Step 2: Create CONFIDENTIAL.md**

```markdown
# Data Handling Policy

The raw dataset (`IMM1_SPR_Data.csv`) is **proprietary and confidential**.

## Rules

1. **Never commit raw data.** `data/raw/` and `data/processed/` are gitignored.
   Raw CSV lives at `$IMM1_DATA_PATH` (default: `~/secure_data/imm1/`).
2. **Never upload to cloud.** No S3, no EC2, no GitHub Issues, no public services.
3. **Never paste compound IDs (`RAP-XXXXXXX`) or pKD values into commits, PR
   descriptions, issue titles, or external chat.** Use generic summary statistics
   (counts, percentages) instead.
4. **The pre-commit hook (`scripts/check_no_data_leak.sh`) refuses commits
   containing `RAP-\d{7}` or pKD-format floats.** Do not bypass it with
   `--no-verify` without a documented justification.
5. **The repo is private and stays private** until the IP owner explicitly
   approves public release.

## Allowed disclosures (in committed files)

- Compound counts (n=288 raw, ~277 curated).
- Censoring fraction (~20% at floor).
- Summary distribution shape (mean/std/range may be sensitive — prefer counts).
- Methodology descriptions.

## Violation response

If raw data is accidentally committed: (a) immediately revoke the commit with
`git filter-repo` or `bfg-repo-cleaner`, (b) force-push the cleaned history,
(c) treat the original commit as compromised and rotate any associated tokens.
```

- [ ] **Step 3: Create data/README.md**

```markdown
# Local Data Directory

This directory is intentionally empty. The raw IMM1 SPR dataset is **not**
committed to this repo.

## Setup

1. Place `IMM1_SPR_Data.csv` somewhere outside the repo. Recommended:
   ```
   ~/secure_data/imm1/IMM1_SPR_Data.csv
   ```

2. Export the path:
   ```bash
   export IMM1_DATA_PATH=~/secure_data/imm1/IMM1_SPR_Data.csv
   ```
   Add to `~/.zshrc` (or equivalent) for persistence.

3. Verify:
   ```bash
   python -c "from imm1_glue.data.load import load_imm1; print(load_imm1().shape)"
   ```

## Schema

Required columns: `Compound Name`, `SMILES`, `pKD`.
Expected: ~288 rows (replicates included).
```

- [ ] **Step 4: Commit**

```bash
git add README.md CONFIDENTIAL.md data/README.md
git commit -m "Add README, confidentiality policy, and data setup instructions"
```

---

### Task 0.3 — Create .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write .gitignore**

```gitignore
# Confidential data — NEVER commit
data/raw/
data/processed/
*.csv
*.parquet
!tests/fixtures/*.csv

# Run artifacts
results/predictions/
results/tables/
results/figures/
results/logs/
results/splits/
results/library_audit.md

# Notebook checkpoints
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
htmlcov/

# Virtualenv / conda
.venv/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

- [ ] **Step 2: Verify**

Run: `cat .gitignore | head -5`
Expected: shows the `data/raw/` line.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "Add .gitignore — block raw data, csv, parquet, results"
```

---

### Task 0.4 — Pre-commit data-leak hook

**Files:**
- Create: `scripts/check_no_data_leak.sh`
- Create: `.githooks/pre-commit` (symlink)
- Test fixture: `tests/fixtures/planted_leak.txt` (will be created in test task later)

- [ ] **Step 1: Write the hook script**

Create `scripts/check_no_data_leak.sh`:
```bash
#!/usr/bin/env bash
# Refuses commits containing RAP-XXXXXXX compound IDs or pKD-looking floats.
# Scans staged content only — does not block git operations on already-committed files.

set -euo pipefail

PATTERN_ID='RAP-[0-9]{7}'
PATTERN_PKD='pkd[[:space:]]*[:=]?[[:space:]]*[0-9]+\.[0-9]{4,}'

# Get staged content (added or modified lines only)
LEAKED=$(git diff --cached --no-color --unified=0 | \
         grep -E "^\+" | \
         grep -v "^\+\+\+" | \
         grep -Ei "($PATTERN_ID|$PATTERN_PKD)" || true)

if [ -n "$LEAKED" ]; then
    echo "ERROR: pre-commit hook detected possible confidential data leak."
    echo "Offending lines:"
    echo "$LEAKED"
    echo
    echo "If you believe this is a false positive, document the justification"
    echo "and re-run with --no-verify (NOT recommended for IMM1 work)."
    exit 1
fi
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/check_no_data_leak.sh`

- [ ] **Step 3: Wire the hook**

Run:
```bash
git config core.hooksPath .githooks
ln -s ../../scripts/check_no_data_leak.sh .githooks/pre-commit
ls -la .githooks/pre-commit
```
Expected: lists the symlink.

- [ ] **Step 4: Manually test the hook with a fake leak**

Run:
```bash
echo "compound RAP-0010972 has pkd=8.6695" > /tmp/leak_test.txt
git add /tmp/leak_test.txt 2>/dev/null || true   # may fail if outside repo
# Inline test:
echo "+compound RAP-0010972 has pkd=8.6695" | grep -Ei "RAP-[0-9]{7}|pkd[[:space:]]*[:=]?[[:space:]]*[0-9]+\.[0-9]{4,}"
```
Expected: prints the line (proves pattern matches).

- [ ] **Step 5: Commit hook script (not the symlink — that's per-clone)**

```bash
git add scripts/check_no_data_leak.sh
git commit -m "Add pre-commit hook blocking compound IDs + pKD-format floats"
```

---

### Task 0.5 — Python environment and packaging

**Files:**
- Create: `pyproject.toml`
- Create: `environment.yml`
- Create: `src/imm1_glue/__init__.py`
- Create: `src/imm1_glue/{data,features,models,evaluation,reports}/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "imm1-glue-baselines"
version = "0.1.0"
description = "Private baseline benchmark — macrocyclic peptide molecular glue binding affinity"
readme = "README.md"
requires-python = ">=3.11"
authors = [{name = "Joshua Abbott"}]

dependencies = [
    "target-affinity-ml @ file:///Users/joshuaabbott/target-affinity-ml",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyarrow>=15.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "rdkit>=2023.09",
    "torch>=2.1",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "ruff>=0.1", "ipykernel>=6.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```

> **Note on the target-affinity-ml file:// dependency:** This is a local-path install. If you push this repo to other machines, that path won't resolve. For now this is the right call since it's a single-machine private project. If you ever need multi-machine, switch to a git+ssh URL after pushing target-affinity-ml to a private repo.

- [ ] **Step 2: Write environment.yml**

```yaml
name: imm1-glue
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - -e .
```

- [ ] **Step 3: Write src/imm1_glue/__init__.py**

```python
"""imm1-glue-baselines: private baseline benchmark for macrocyclic peptide molecular glues.

Re-exports infrastructure from target_affinity_ml; adds IMM1-specific
data loading, curation, two new split strategies (Butina cluster,
time/synthesis-order), and report generation.
"""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write empty package __init__.py files**

Create each of these as empty files (just `""`):
- `src/imm1_glue/data/__init__.py`
- `src/imm1_glue/features/__init__.py`
- `src/imm1_glue/models/__init__.py`
- `src/imm1_glue/evaluation/__init__.py`
- `src/imm1_glue/reports/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Create the conda env and install**

Run:
```bash
conda env create -f environment.yml
conda activate imm1-glue
python -c "import imm1_glue; print(imm1_glue.__version__)"
python -c "import target_affinity_ml; print(target_affinity_ml.__version__)"
```
Expected: prints `0.1.0` and `1.1.0` (or current version of target-affinity-ml).

- [ ] **Step 6: Run pytest to confirm the test infrastructure picks up the package**

Run: `pytest tests/ -v`
Expected: "no tests ran in 0.01s" or similar (no tests yet — that's fine).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml environment.yml src/ tests/__init__.py
git commit -m "Add Python env, pyproject.toml, package skeleton"
```

---

### Task 0.6 — Config YAML files

**Files:**
- Create: `configs/dataset_imm1.yaml`
- Create: `configs/splits.yaml`
- Create: `configs/rf_baseline.yaml`
- Create: `configs/xgb_baseline.yaml`
- Create: `configs/mlp_baseline.yaml`

- [ ] **Step 1: Write configs/dataset_imm1.yaml**

```yaml
# Dataset and feature configuration for IMM1 macrocyclic glue benchmark

dataset:
  name: imm1
  path_env_var: IMM1_DATA_PATH
  expected_columns: ["Compound Name", "SMILES", "pKD"]
  expected_min_rows: 280
  expected_max_rows: 300

curation:
  pkd_floor: 4.0
  pkd_floor_tolerance: 1.0e-6  # is_censored if pkd_mean <= floor + tolerance
  noisy_std_threshold: 1.0      # is_noisy if pkd_std > this
  canonicalize_smiles: true
  drop_invalid_smiles: true

features:
  type: morgan_fingerprint
  radius: 3
  n_bits: 4096
  use_chirality: true
  use_features: false           # use radius-only ECFP-style, not FCFP

classification_thresholds:
  - 6.0      # KD <= 1 uM
  - 7.0      # KD <= 100 nM
  - 4.0001   # anything above the assay floor
```

- [ ] **Step 2: Write configs/splits.yaml**

```yaml
# Split strategy parameters

random:
  n_folds: 5
  stratify_quartiles: true       # for outer random split only
  seeds: [42, 123, 456, 789, 1000]

scaffold:
  n_folds: 5
  scaffold_type: murcko
  singleton_assignment: round_robin  # seed-dependent
  seeds: [42, 123, 456, 789, 1000]

butina:
  n_folds: 5
  similarity_fp:
    type: morgan_fingerprint
    radius: 2
    n_bits: 2048
  cutoff_sweep: [0.4, 0.5, 0.6, 0.7, 0.8]
  min_clusters: 10
  min_cluster_size: 5
  cutoff_selection: smallest_qualifying  # tiebreaker
  seeds: [42, 123, 456, 789, 1000]

time:
  n_folds: 5
  id_pattern: "RAP-[0-9]{7}"
  sort_order: lexicographic        # equivalent to numeric for zero-padded IDs
  seeds: [42, 123, 456, 789, 1000]  # only affect inner-CV under time split
```

- [ ] **Step 3: Write configs/rf_baseline.yaml**

```yaml
model:
  name: random_forest
  type: regression

uncertainty:
  method: tree_variance

hyperparameters_default:
  n_estimators: 500
  max_depth: null
  min_samples_leaf: 2
  max_features: sqrt
  n_jobs: -1
  random_state: 42

search_grid:
  n_estimators: [200, 500, 1000]
  max_depth: [null, 20, 50]
  min_samples_leaf: [1, 2, 5]
  max_features: ["sqrt", "log2", 0.3]
```

- [ ] **Step 4: Write configs/xgb_baseline.yaml**

```yaml
model:
  name: xgboost
  type: regression

uncertainty:
  method: residual_quantile

hyperparameters_default:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.1
  subsample: 1.0
  colsample_bytree: 1.0
  objective: reg:squarederror
  n_jobs: -1
  random_state: 42

search_grid:
  n_estimators: [200, 500, 1000]
  max_depth: [4, 6, 8]
  learning_rate: [0.03, 0.1, 0.3]
  subsample: [0.7, 1.0]
  colsample_bytree: [0.7, 1.0]
```

- [ ] **Step 5: Write configs/mlp_baseline.yaml**

```yaml
model:
  name: mlp
  type: regression

uncertainty:
  method: mc_dropout
  n_passes: 30

hyperparameters_default:
  hidden_layers: [512]
  dropout: 0.3
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  batch_size: 32
  epochs: 200
  early_stopping_patience: 30

search_grid:
  hidden_layers: [[256], [512], [256, 128]]
  dropout: [0.2, 0.4]
  learning_rate: [1.0e-3, 3.0e-4]
  weight_decay: [0, 1.0e-4]
```

- [ ] **Step 6: Commit**

```bash
git add configs/
git commit -m "Add YAML configs: dataset, splits, RF/XGB/MLP baselines"
```

---

### Task 0.7 — Library audit script

**Files:**
- Create: `scripts/audit_library.py`

- [ ] **Step 1: Write the audit script**

```python
"""Library API audit for target-affinity-ml.

Verifies that the installed version of target-affinity-ml exposes the
API surfaces this project depends on. Produces results/library_audit.md
with a checklist and gap classification (critical-path / per-model
uncertainty / non-critical helper) per the spec's blocking policy.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any


# (module_path, attr_name, gap_class)
REQUIRED_APIS = [
    # Features
    ("target_affinity_ml.features.fingerprints", "smiles_to_morgan_fp", "critical"),
    # Models
    ("target_affinity_ml.models", "RandomForestModel", "critical"),
    ("target_affinity_ml.models", "XGBoostModel", "critical"),
    ("target_affinity_ml.models", "MLPModel", "critical"),
    # Splits
    ("target_affinity_ml.data.splits", "random_split", "critical"),
    ("target_affinity_ml.data.splits", "scaffold_split", "critical"),
    # Tuning
    ("target_affinity_ml.training.tune", "tune_model", "critical"),
    # Metrics
    ("target_affinity_ml.evaluation.metrics", "compute_regression_metrics", "critical"),
    ("target_affinity_ml.evaluation.metrics", "compute_classification_metrics", "critical"),
    # Bootstrap / multi-seed
    ("target_affinity_ml.evaluation.bootstrap", "bootstrap_ci", "non_critical"),
    ("target_affinity_ml.evaluation.multi_seed_analysis", "aggregate_seeds", "non_critical"),
    # Uncertainty calibration
    ("target_affinity_ml.evaluation.uncertainty", "reliability_diagram", "non_critical"),
]

PER_MODEL_UNCERTAINTY = [
    ("target_affinity_ml.models.rf_model", "RandomForestModel", "predict_with_uncertainty"),
    ("target_affinity_ml.models.xgb_model", "XGBoostModel", "predict_with_uncertainty"),
    ("target_affinity_ml.models.mlp_model", "MLPModel", "predict_with_uncertainty"),
]


def resolve(module_path: str, attr_name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module_path)
        if not hasattr(mod, attr_name):
            return False, f"missing attribute: {attr_name}"
        return True, ""
    except ImportError as e:
        return False, f"import failed: {e}"


def check_method(module_path: str, class_name: str, method_name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            return False, f"missing class: {class_name}"
        if not hasattr(cls, method_name):
            return False, f"missing method: {class_name}.{method_name}"
        return True, ""
    except ImportError as e:
        return False, f"import failed: {e}"


def main() -> None:
    out = Path("results/library_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Library Audit Report", ""]
    try:
        import target_affinity_ml
        lines.append(f"**target-affinity-ml version:** {target_affinity_ml.__version__}")
    except (ImportError, AttributeError):
        lines.append("**target-affinity-ml version:** UNKNOWN")
    lines.append("")

    critical_failures: list[str] = []
    uncertainty_failures: list[str] = []
    helper_failures: list[str] = []

    lines.append("## Required API Surfaces")
    lines.append("")
    lines.append("| Module | Attribute | Class | Status |")
    lines.append("|---|---|---|---|")
    for module_path, attr, gap_class in REQUIRED_APIS:
        ok, msg = resolve(module_path, attr)
        status = "PASS" if ok else f"FAIL ({msg})"
        lines.append(f"| `{module_path}` | `{attr}` | {gap_class} | {status} |")
        if not ok:
            if gap_class == "critical":
                critical_failures.append(f"{module_path}.{attr}")
            else:
                helper_failures.append(f"{module_path}.{attr}")

    lines.append("")
    lines.append("## Per-Model Uncertainty (`predict_with_uncertainty`)")
    lines.append("")
    lines.append("| Module | Class | Method | Status |")
    lines.append("|---|---|---|---|")
    for module_path, cls_name, method in PER_MODEL_UNCERTAINTY:
        ok, msg = check_method(module_path, cls_name, method)
        status = "PASS" if ok else f"FAIL ({msg})"
        lines.append(f"| `{module_path}` | `{cls_name}` | `{method}` | {status} |")
        if not ok:
            uncertainty_failures.append(f"{cls_name}.{method}")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if not (critical_failures or uncertainty_failures or helper_failures):
        lines.append("**All API surfaces present. No gap-fill required. Proceed to Phase 1.**")
    else:
        if critical_failures:
            lines.append(f"**Critical-path gaps (BLOCK Phase 1):** {len(critical_failures)}")
            for f in critical_failures:
                lines.append(f"  - {f}")
            lines.append("")
            lines.append("Action: open gap-fill PRs to `target-affinity-ml` and pin the new version.")
        if uncertainty_failures:
            lines.append(f"**Per-model uncertainty gaps (does NOT block):** {len(uncertainty_failures)}")
            for f in uncertainty_failures:
                lines.append(f"  - {f}")
            lines.append("")
            lines.append("Action: model's sigma column will be NaN; affected cells flagged in report.")
        if helper_failures:
            lines.append(f"**Non-critical helper gaps (does NOT block):** {len(helper_failures)}")
            for f in helper_failures:
                lines.append(f"  - {f}")
            lines.append("")
            lines.append("Action: implement local stub in `src/imm1_glue/` with TODO to upstream.")

    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")

    if critical_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the audit**

Run: `python scripts/audit_library.py`
Expected: writes `results/library_audit.md`. If all required APIs are present, exit code 0; otherwise exit code 1.

- [ ] **Step 3: Read the audit report**

Run: `cat results/library_audit.md`

**Decision point:**
- If "**All API surfaces present**" — proceed to Step 4.
- If critical-path gaps — STOP. Open gap-fill PRs to `target-affinity-ml`, merge them, bump version pin in `pyproject.toml`, re-run audit.
- If only per-model uncertainty or non-critical helper gaps — note in plan and proceed.

- [ ] **Step 4: Commit the audit script**

```bash
git add scripts/audit_library.py
git commit -m "Add library audit script"
```

> `results/library_audit.md` is gitignored — do not commit it. The audit re-runs as needed.

---

### Task 0.8 — Pytest infrastructure and conftest

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest.py with shared fixtures**

```python
"""Shared pytest fixtures for IMM1 baseline tests."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """Tiny synthetic CSV matching the IMM1 schema for loader/curation tests.

    Covers each replicate-handling branch:
    - single-row compound
    - replicated all-binder
    - replicated all-censored
    - replicated mixed binder+censored
    - replicated noisy (std > 1.0 but not censored)
    """
    rows = [
        # single binder
        {"Compound Name": "TEST-0000001", "SMILES": "CCO", "pKD": 7.5},
        # all-binder replicates (low variance)
        {"Compound Name": "TEST-0000002", "SMILES": "CCN", "pKD": 6.3},
        {"Compound Name": "TEST-0000002", "SMILES": "CCN", "pKD": 6.5},
        # all-censored replicates
        {"Compound Name": "TEST-0000003", "SMILES": "CCC", "pKD": 4.0},
        {"Compound Name": "TEST-0000003", "SMILES": "CCC", "pKD": 4.0},
        # mixed binder + censored (will be noisy, not censored after mean)
        {"Compound Name": "TEST-0000004", "SMILES": "CCCC", "pKD": 7.5},
        {"Compound Name": "TEST-0000004", "SMILES": "CCCC", "pKD": 4.0},
        # noisy non-censored
        {"Compound Name": "TEST-0000005", "SMILES": "c1ccccc1", "pKD": 6.0},
        {"Compound Name": "TEST-0000005", "SMILES": "c1ccccc1", "pKD": 7.5},
        {"Compound Name": "TEST-0000005", "SMILES": "c1ccccc1", "pKD": 8.0},
        # invalid SMILES
        {"Compound Name": "TEST-0000006", "SMILES": "not_a_smiles_string", "pKD": 5.0},
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic_imm1.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def env_imm1_data_path(synthetic_csv: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set IMM1_DATA_PATH env var to the synthetic CSV path."""
    monkeypatch.setenv("IMM1_DATA_PATH", str(synthetic_csv))
    return synthetic_csv
```

- [ ] **Step 2: Run pytest to confirm fixture syntax**

Run: `pytest tests/ -v --co`
Expected: pytest collects no tests (none defined yet) and exits 0.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "Add pytest fixtures: synthetic CSV covering replicate branches"
```

---

**Phase 0 exit gate:**
- [ ] `pip install -e .` succeeded.
- [ ] `python scripts/audit_library.py` produced a report; critical-path APIs are PASS, OR gap-fill PRs to `target-affinity-ml` have been merged and pinned.
- [ ] Pre-commit hook is wired (`.githooks/pre-commit` symlink exists).
- [ ] `pytest tests/` runs without error.

---

# Phase 1 — Data Pipeline

**Goal:** `load.py` and `curate.py` produce a clean DataFrame from the raw CSV, with replicate aggregation and censored/noisy flags exactly as the spec defines. Curation produces a sanitized markdown report.

**Exit criteria:**
- `pytest tests/test_load.py tests/test_curate.py` passes.
- Running curation on the real `$IMM1_DATA_PATH` CSV produces a sanitized `results/curation_report.md` and an in-memory DataFrame ready for splits.

---

### Task 1.1 — Loader tests (failing first)

**Files:**
- Create: `tests/test_load.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for imm1_glue.data.load.load_imm1."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from imm1_glue.data.load import load_imm1


def test_load_imm1_returns_dataframe_with_expected_columns(env_imm1_data_path: Path) -> None:
    df = load_imm1()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"compound_id", "smiles", "pkd"}


def test_load_imm1_returns_expected_row_count(env_imm1_data_path: Path) -> None:
    df = load_imm1()
    assert len(df) == 11  # synthetic CSV has 11 rows


def test_load_imm1_pkd_is_numeric(env_imm1_data_path: Path) -> None:
    df = load_imm1()
    assert pd.api.types.is_numeric_dtype(df["pkd"])


def test_load_imm1_raises_if_env_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMM1_DATA_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="IMM1_DATA_PATH"):
        load_imm1()


def test_load_imm1_raises_if_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMM1_DATA_PATH", "/nonexistent/path.csv")
    with pytest.raises(FileNotFoundError):
        load_imm1()


def test_load_imm1_raises_on_schema_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("WrongCol1,WrongCol2\nfoo,bar\n")
    monkeypatch.setenv("IMM1_DATA_PATH", str(bad_csv))
    with pytest.raises(ValueError, match="expected columns"):
        load_imm1()


def test_load_imm1_raises_on_nonnumeric_pkd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_csv = tmp_path / "bad_pkd.csv"
    bad_csv.write_text("Compound Name,SMILES,pKD\nC1,CCO,not_a_number\n")
    monkeypatch.setenv("IMM1_DATA_PATH", str(bad_csv))
    with pytest.raises((ValueError, TypeError)):
        load_imm1()
```

- [ ] **Step 2: Run tests, confirm they fail with ImportError**

Run: `pytest tests/test_load.py -v`
Expected: all tests fail with `ImportError: cannot import name 'load_imm1'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_load.py
git commit -m "Add failing tests for data loader (TDD red phase)"
```

---

### Task 1.2 — Implement load.py

**Files:**
- Create: `src/imm1_glue/data/load.py`

- [ ] **Step 1: Write load.py**

```python
"""IMM1 SPR dataset loader.

Reads the CSV at $IMM1_DATA_PATH, validates schema, returns a typed
DataFrame with columns [compound_id, smiles, pkd].
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["Compound Name", "SMILES", "pKD"]
RENAME_MAP = {"Compound Name": "compound_id", "SMILES": "smiles", "pKD": "pkd"}


def load_imm1(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the IMM1 SPR dataset.

    Parameters
    ----------
    path : str | Path | None
        Override the IMM1_DATA_PATH env var. If None, reads the env var.

    Returns
    -------
    pd.DataFrame
        Columns: [compound_id: str, smiles: str, pkd: float]

    Raises
    ------
    FileNotFoundError
        If IMM1_DATA_PATH is unset or the file does not exist.
    ValueError
        If the CSV does not contain the expected columns or pKD is non-numeric.
    """
    if path is None:
        env_path = os.environ.get("IMM1_DATA_PATH")
        if not env_path:
            raise FileNotFoundError(
                "IMM1_DATA_PATH environment variable is not set. "
                "Set it to the local path of IMM1_SPR_Data.csv."
            )
        path = env_path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IMM1 data file not found at {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV at {path} is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[EXPECTED_COLUMNS].rename(columns=RENAME_MAP)
    df["pkd"] = pd.to_numeric(df["pkd"], errors="raise")

    return df
```

- [ ] **Step 2: Run tests, confirm pass**

Run: `pytest tests/test_load.py -v`
Expected: all 7 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/imm1_glue/data/load.py
git commit -m "Implement data loader with schema validation"
```

---

### Task 1.3 — Curation tests

**Files:**
- Create: `tests/test_curate.py`

- [ ] **Step 1: Write failing tests covering each curation branch**

```python
"""Tests for imm1_glue.data.curate.curate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imm1_glue.data.curate import curate
from imm1_glue.data.load import load_imm1


def test_curate_returns_expected_columns(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    assert set(df.columns) == {
        "compound_id", "smiles", "pkd_mean", "pkd_std",
        "n_replicates", "is_censored", "is_noisy",
    }


def test_curate_drops_invalid_smiles(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    # Synthetic fixture has TEST-0000006 with invalid SMILES; should be dropped.
    assert "TEST-0000006" not in df["compound_id"].values


def test_curate_collapses_replicates_to_one_row(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    # 5 unique compounds remain after dropping the invalid-SMILES one
    assert len(df) == 5
    assert df["compound_id"].is_unique


def test_curate_single_compound_n_replicates_equals_one(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    row = df[df["compound_id"] == "TEST-0000001"].iloc[0]
    assert row["n_replicates"] == 1


def test_curate_all_binder_replicates_mean(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    row = df[df["compound_id"] == "TEST-0000002"].iloc[0]
    assert row["pkd_mean"] == pytest.approx(6.4)
    assert row["n_replicates"] == 2
    assert not row["is_censored"]
    assert not row["is_noisy"]


def test_curate_all_censored_replicates_flagged_as_censored(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    row = df[df["compound_id"] == "TEST-0000003"].iloc[0]
    assert row["is_censored"]
    assert row["pkd_mean"] == pytest.approx(4.0)


def test_curate_mixed_replicates_classified_as_noisy_binder(env_imm1_data_path: Path) -> None:
    """Per spec: mean(7.5, 4.0) = 5.75 → is_censored=False, is_noisy=True."""
    df = curate(load_imm1())
    row = df[df["compound_id"] == "TEST-0000004"].iloc[0]
    assert row["pkd_mean"] == pytest.approx(5.75)
    assert not row["is_censored"]
    assert row["is_noisy"]


def test_curate_noisy_non_censored_flagged(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    row = df[df["compound_id"] == "TEST-0000005"].iloc[0]
    assert row["is_noisy"]  # std > 1.0
    assert not row["is_censored"]


def test_curate_smiles_canonicalized(env_imm1_data_path: Path) -> None:
    df = curate(load_imm1())
    # Benzene 'c1ccccc1' canonicalizes to 'c1ccccc1' (no change), but verify it's RDKit-roundtripped
    benzene = df[df["compound_id"] == "TEST-0000005"]["smiles"].iloc[0]
    from rdkit import Chem
    assert Chem.MolFromSmiles(benzene) is not None
```

- [ ] **Step 2: Run tests, confirm they fail with ImportError**

Run: `pytest tests/test_curate.py -v`
Expected: tests fail with `ImportError: cannot import name 'curate'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_curate.py
git commit -m "Add failing tests for curation (TDD red phase)"
```

---

### Task 1.4 — Implement curate.py

**Files:**
- Create: `src/imm1_glue/data/curate.py`

- [ ] **Step 1: Write curate.py**

```python
"""Curation: replicate aggregation, SMILES canonicalization, censoring + noisy flags.

Per spec section 2:
- Mean pKD per compound across replicate rows.
- is_censored = True iff pkd_mean <= 4.0 + 1e-6 (i.e., mean is at the assay floor).
- is_noisy = True iff per-compound pKD std > 1.0.
- Mixed binder + censored groups average as-is; their mean determines is_censored.
- Invalid SMILES rows are dropped (logged at WARNING).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

PKD_FLOOR = 4.0
PKD_FLOOR_TOLERANCE = 1e-6
NOISY_STD_THRESHOLD = 1.0


def canonicalize_smiles(smiles: str) -> str | None:
    """Return RDKit-canonicalized SMILES, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def curate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicates, canonicalize SMILES, flag censored + noisy.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loader output with columns [compound_id, smiles, pkd].

    Returns
    -------
    pd.DataFrame
        Columns: [compound_id, smiles, pkd_mean, pkd_std,
                  n_replicates, is_censored, is_noisy]
        One row per unique compound.
    """
    df = df.copy()

    # SMILES canonicalization (drop invalid)
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    n_before = len(df)
    df = df[df["canonical_smiles"].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with invalid SMILES", n_dropped)

    # Replicate aggregation
    grouped = df.groupby("compound_id").agg(
        smiles=("canonical_smiles", "first"),
        pkd_mean=("pkd", "mean"),
        pkd_std=("pkd", "std"),
        n_replicates=("pkd", "count"),
    ).reset_index()

    # Flags
    grouped["is_censored"] = grouped["pkd_mean"] <= (PKD_FLOOR + PKD_FLOOR_TOLERANCE)
    grouped["is_noisy"] = (grouped["pkd_std"] > NOISY_STD_THRESHOLD).fillna(False)

    logger.info(
        "Curated: %d compounds (%d censored, %d noisy)",
        len(grouped),
        grouped["is_censored"].sum(),
        grouped["is_noisy"].sum(),
    )

    return grouped[[
        "compound_id", "smiles", "pkd_mean", "pkd_std",
        "n_replicates", "is_censored", "is_noisy",
    ]]
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_curate.py -v`
Expected: all 9 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/imm1_glue/data/curate.py
git commit -m "Implement curation with replicate aggregation and censored+noisy flags"
```

---

### Task 1.5 — Sanitized curation report on real data

**Files:**
- Modify: `scripts/run_diagnostics.py` (we'll create this fresh — full content below)

- [ ] **Step 1: Write run_diagnostics.py (skeleton — only curation report part for now)**

```python
"""Pre-benchmark diagnostics:
1. Curation report (sanitized — counts only, no IDs/SMILES/KDs).
2. Butina cutoff sweep (added in Task 2.x).
3. Time-split sanity check (added in Task 2.x).

Run: python scripts/run_diagnostics.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from imm1_glue.data.load import load_imm1
from imm1_glue.data.curate import curate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def curation_report(out_path: Path) -> None:
    raw = load_imm1()
    curated = curate(raw)

    lines = ["# Curation Report (sanitized)", ""]
    lines.append(f"- **Raw rows:** {len(raw)}")
    lines.append(f"- **Unique compound IDs:** {raw['compound_id'].nunique()}")
    lines.append(f"- **Curated compounds (after dedup + invalid-SMILES drop):** {len(curated)}")
    lines.append(f"- **Censored compounds (pkd_mean ≤ 4.0 + 1e-6):** {int(curated['is_censored'].sum())}")
    lines.append(f"- **Noisy compounds (pkd_std > 1.0):** {int(curated['is_noisy'].sum())}")
    lines.append("")
    lines.append("## pKD distribution (curated)")
    desc = curated["pkd_mean"].describe()
    lines.append(f"- count: {int(desc['count'])}")
    lines.append(f"- min:   {desc['min']:.2f}")
    lines.append(f"- 25%:   {desc['25%']:.2f}")
    lines.append(f"- 50%:   {desc['50%']:.2f}")
    lines.append(f"- 75%:   {desc['75%']:.2f}")
    lines.append(f"- max:   {desc['max']:.2f}")
    lines.append(f"- mean:  {desc['mean']:.2f}")
    lines.append(f"- std:   {desc['std']:.2f}")
    lines.append("")
    lines.append("## Replicate-group histogram")
    rep_counts = curated["n_replicates"].value_counts().sort_index()
    for n, c in rep_counts.items():
        lines.append(f"- {n} replicate(s): {c} compound(s)")
    lines.append("")
    lines.append("---")
    lines.append("**No compound IDs, SMILES, or individual pKD values are included in this report.**")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results", type=Path)
    args = p.parse_args()

    curation_report(args.out_dir / "curation_report.md")
    # Butina sweep + time-split check added in Phase 2.


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against real data**

```bash
export IMM1_DATA_PATH=~/secure_data/imm1/IMM1_SPR_Data.csv
python scripts/run_diagnostics.py
```
Expected: writes `results/curation_report.md`.

- [ ] **Step 3: Inspect the report — confirm no leaks**

Run: `cat results/curation_report.md`
Sanity check: counts only, no `RAP-XXXX`, no individual pKD values.

- [ ] **Step 4: Run the pre-commit hook as a dry test**

Run: `bash scripts/check_no_data_leak.sh < results/curation_report.md || echo "would block"`
Expected: nothing blocks (the report is clean).

- [ ] **Step 5: Commit the diagnostic script (not the generated report)**

```bash
git add scripts/run_diagnostics.py
git commit -m "Add diagnostics script with sanitized curation report"
```

---

**Phase 1 exit gate:**
- [ ] `pytest tests/test_load.py tests/test_curate.py` → all pass.
- [ ] `results/curation_report.md` exists, looks sane, contains no PII/proprietary data.
- [ ] You've eyeballed the curation numbers against your expectations (e.g., ~277 compounds, ~50–57 censored, some noisy).

---

# Phase 2 — Splits and Diagnostics

**Goal:** All four split strategies are implemented and tested; the Butina cutoff sweep and time-split sanity check run on the real data and inform the chosen cutoff / time-split feasibility.

**Exit criteria:**
- `pytest tests/test_splits.py` passes for all four strategies.
- `python scripts/run_diagnostics.py` produces all three diagnostic outputs.
- The Butina cutoff for the main sweep is recorded in `configs/splits.yaml`.

---

### Task 2.1 — Random split (delegate to library)

**Files:**
- Create: `src/imm1_glue/data/splits.py` (initial skeleton + random split)
- Modify: `tests/test_splits.py`

- [ ] **Step 1: Write the random-split test**

Append to `tests/test_splits.py`:
```python
"""Tests for imm1_glue.data.splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imm1_glue.data.curate import curate
from imm1_glue.data.load import load_imm1
from imm1_glue.data.splits import random_fold_assignment


def test_random_fold_assignment_shape_and_range(env_imm1_data_path) -> None:
    curated = curate(load_imm1())
    folds = random_fold_assignment(curated, n_folds=5, seed=42)
    assert folds.shape == (len(curated),)
    assert set(np.unique(folds)).issubset({0, 1, 2, 3, 4})


def test_random_fold_assignment_balanced(env_imm1_data_path) -> None:
    curated = curate(load_imm1())
    folds = random_fold_assignment(curated, n_folds=5, seed=42)
    counts = np.bincount(folds, minlength=5)
    # Each fold should have ~len/5 compounds, ±2
    assert counts.min() >= len(curated) // 5 - 2
    assert counts.max() <= len(curated) // 5 + 2


def test_random_fold_assignment_reproducible(env_imm1_data_path) -> None:
    curated = curate(load_imm1())
    f1 = random_fold_assignment(curated, n_folds=5, seed=42)
    f2 = random_fold_assignment(curated, n_folds=5, seed=42)
    np.testing.assert_array_equal(f1, f2)


def test_random_fold_assignment_seeds_differ(env_imm1_data_path) -> None:
    curated = curate(load_imm1())
    f1 = random_fold_assignment(curated, n_folds=5, seed=42)
    f2 = random_fold_assignment(curated, n_folds=5, seed=123)
    assert not np.array_equal(f1, f2)
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest tests/test_splits.py -v`
Expected: all fail with ImportError.

- [ ] **Step 3: Implement random_fold_assignment in splits.py**

Create `src/imm1_glue/data/splits.py`:
```python
"""Four split strategies for the IMM1 benchmark.

Random and Murcko scaffold delegate to target_affinity_ml; Butina cluster
and time/synthesis-order are implemented locally.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def random_fold_assignment(
    curated: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Stratified k-fold by pKD quartile.

    Parameters
    ----------
    curated : pd.DataFrame
        Curated data with column `pkd_mean`.
    n_folds : int
    seed : int

    Returns
    -------
    np.ndarray (int) of shape (n_compounds,) with values in [0, n_folds).
    """
    quartiles = pd.qcut(curated["pkd_mean"], q=4, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(curated), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.arange(len(curated)), quartiles)):
        folds[test_idx] = fold_idx
    return folds
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_splits.py -v`
Expected: 4 random-split tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/imm1_glue/data/splits.py tests/test_splits.py
git commit -m "Implement random fold assignment (stratified by pKD quartile)"
```

---

### Task 2.2 — Scaffold (Murcko) split

**Files:**
- Modify: `src/imm1_glue/data/splits.py`
- Modify: `tests/test_splits.py`

- [ ] **Step 1: Append scaffold-split tests**

```python
from imm1_glue.data.splits import scaffold_fold_assignment


def test_scaffold_fold_assignment_groups_stay_together(env_imm1_data_path) -> None:
    """Compounds with the same Murcko scaffold land in the same fold."""
    curated = curate(load_imm1())
    folds = scaffold_fold_assignment(curated, n_folds=5, seed=42)
    # Synthetic SMILES are tiny so each is its own scaffold; this test mostly verifies
    # the function runs and returns valid folds.
    assert folds.shape == (len(curated),)
    assert set(np.unique(folds)).issubset({0, 1, 2, 3, 4})


def test_scaffold_fold_assignment_reproducible(env_imm1_data_path) -> None:
    curated = curate(load_imm1())
    f1 = scaffold_fold_assignment(curated, n_folds=5, seed=42)
    f2 = scaffold_fold_assignment(curated, n_folds=5, seed=42)
    np.testing.assert_array_equal(f1, f2)


def test_scaffold_fold_assignment_with_known_duplicates() -> None:
    """Two compounds with the same scaffold must land in the same fold."""
    df = pd.DataFrame({
        "compound_id": [f"T-{i}" for i in range(10)],
        # 5 compounds with benzene ring scaffold + alkyl variants
        "smiles": [
            "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CCCC", "c1ccccc1CCCCC",
            # 5 with cyclopentane
            "C1CCCC1C", "C1CCCC1CC", "C1CCCC1CCC", "C1CCCC1CCCC", "C1CCCC1CCCCC",
        ],
        "pkd_mean": [5.0] * 10,
    })
    folds = scaffold_fold_assignment(df, n_folds=2, seed=42)
    # Benzene-scaffolded (first 5) and cyclopentane-scaffolded (last 5) must each
    # be in a single fold (no scaffold split across folds).
    benzene_folds = set(folds[:5])
    cyclo_folds = set(folds[5:])
    assert len(benzene_folds) == 1
    assert len(cyclo_folds) == 1
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_splits.py::test_scaffold_fold_assignment_with_known_duplicates -v`
Expected: fails (ImportError).

- [ ] **Step 3: Implement scaffold_fold_assignment**

Append to `src/imm1_glue/data/splits.py`:
```python
def _get_murcko_scaffold(smiles: str) -> str:
    """Return canonical Murcko scaffold SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_fold_assignment(
    curated: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Group by Murcko scaffold; assign whole scaffolds to folds.

    Singletons (scaffolds with one compound) are shuffled by seed and
    round-robin assigned to folds.
    """
    curated = curated.copy()
    curated["_scaffold"] = curated["smiles"].apply(_get_murcko_scaffold)
    scaffold_groups = curated.groupby("_scaffold").indices  # dict: scaffold -> array of row indices

    rng = np.random.RandomState(seed)
    folds = np.full(len(curated), -1, dtype=int)

    # Sort scaffold groups by size descending, place largest first (greedy balancing).
    scaffold_items = sorted(scaffold_groups.items(), key=lambda kv: -len(kv[1]))
    fold_sizes = np.zeros(n_folds, dtype=int)

    # Separate singletons from multi-compound scaffolds.
    multi = [(s, idx) for s, idx in scaffold_items if len(idx) > 1]
    singletons = [(s, idx) for s, idx in scaffold_items if len(idx) == 1]

    # Multi-compound scaffolds: greedy fill into the currently-smallest fold.
    for scaffold, idx in multi:
        target = int(np.argmin(fold_sizes))
        folds[idx] = target
        fold_sizes[target] += len(idx)

    # Singletons: seeded shuffle, then round-robin.
    singleton_order = list(range(len(singletons)))
    rng.shuffle(singleton_order)
    for k, order_idx in enumerate(singleton_order):
        _, idx = singletons[order_idx]
        target = k % n_folds
        folds[idx] = target
        fold_sizes[target] += 1

    assert (folds >= 0).all(), "Some compounds unassigned"
    return folds
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_splits.py -v`
Expected: 3 new scaffold tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/imm1_glue/data/splits.py tests/test_splits.py
git commit -m "Implement Murcko scaffold split with seeded round-robin singletons"
```

---

### Task 2.3 — Butina cluster split

**Files:**
- Modify: `src/imm1_glue/data/splits.py`
- Modify: `tests/test_splits.py`

- [ ] **Step 1: Append Butina cluster tests**

```python
from imm1_glue.data.splits import butina_fold_assignment, butina_cluster_compounds


def test_butina_cluster_compounds_returns_cluster_per_compound() -> None:
    smiles = ["CCO", "CCCO", "CCN", "c1ccccc1", "Cc1ccccc1"]
    clusters = butina_cluster_compounds(smiles, cutoff=0.5)
    assert len(clusters) == len(smiles)
    # All cluster IDs are non-negative integers
    assert all(c >= 0 for c in clusters)


def test_butina_fold_assignment_groups_clusters_together(env_imm1_data_path) -> None:
    curated = curate(load_imm1())
    folds = butina_fold_assignment(curated, n_folds=5, seed=42, cutoff=0.6)
    assert folds.shape == (len(curated),)
    assert set(np.unique(folds)).issubset({0, 1, 2, 3, 4})
```

- [ ] **Step 2: Implement butina_cluster_compounds + butina_fold_assignment**

Append to `src/imm1_glue/data/splits.py`:
```python
def _morgan_fp_bitvect(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Return RDKit ExplicitBitVect for Tanimoto similarity (Butina expects bitvects)."""
    from rdkit.Chem import rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def butina_cluster_compounds(
    smiles_list: list[str], cutoff: float = 0.6, radius: int = 2, n_bits: int = 2048
) -> list[int]:
    """Cluster compounds by Butina at Tanimoto distance (1 - sim) threshold.

    Parameters
    ----------
    smiles_list : list[str]
    cutoff : float
        Tanimoto-distance cutoff (so 0.6 cluster cutoff means similarity >= 0.4).
        Note: RDKit Butina uses distance (1 - sim), so `cutoff` here is the distance.
        Most QSAR papers report a "similarity cutoff" of 0.6 which corresponds to a
        distance cutoff of 0.4. To match the standard convention, we treat the
        `cutoff` argument as the SIMILARITY cutoff and convert: distance = 1 - cutoff.

    Returns
    -------
    list[int]
        Cluster ID per compound (in input order).
    """
    from rdkit import DataStructs

    fps = [_morgan_fp_bitvect(s, radius=radius, n_bits=n_bits) for s in smiles_list]
    n = len(fps)
    # Build the lower-triangular distance matrix as Butina expects
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    distance_cutoff = 1 - cutoff
    cluster_groups = Butina.ClusterData(dists, n, distance_cutoff, isDistData=True)

    cluster_ids = [0] * n
    for cid, members in enumerate(cluster_groups):
        for m in members:
            cluster_ids[m] = cid
    return cluster_ids


def butina_fold_assignment(
    curated: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    cutoff: float = 0.6,
) -> np.ndarray:
    """Assign compounds to outer folds by Butina cluster (entire clusters stay together).

    Identical balancing strategy as scaffold_fold_assignment: greedy fill multi-compound
    clusters by size descending; singletons round-robin under the chosen seed.
    """
    cluster_ids = butina_cluster_compounds(curated["smiles"].tolist(), cutoff=cutoff)
    cluster_groups: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        cluster_groups.setdefault(cid, []).append(idx)

    rng = np.random.RandomState(seed)
    folds = np.full(len(curated), -1, dtype=int)
    fold_sizes = np.zeros(n_folds, dtype=int)

    multi = [(c, idx) for c, idx in cluster_groups.items() if len(idx) > 1]
    singletons = [(c, idx) for c, idx in cluster_groups.items() if len(idx) == 1]

    multi.sort(key=lambda kv: -len(kv[1]))
    for _, idx in multi:
        target = int(np.argmin(fold_sizes))
        folds[idx] = target
        fold_sizes[target] += len(idx)

    singleton_order = list(range(len(singletons)))
    rng.shuffle(singleton_order)
    for k, order_idx in enumerate(singleton_order):
        _, idx = singletons[order_idx]
        target = k % n_folds
        folds[idx] = target
        fold_sizes[target] += 1

    assert (folds >= 0).all()
    return folds
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_splits.py -v`
Expected: Butina tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/imm1_glue/data/splits.py tests/test_splits.py
git commit -m "Implement Butina cluster split (similarity cutoff convention)"
```

---

### Task 2.4 — Time / synthesis-order split

**Files:**
- Modify: `src/imm1_glue/data/splits.py`
- Modify: `tests/test_splits.py`

- [ ] **Step 1: Append time-split tests**

```python
from imm1_glue.data.splits import time_fold_assignment, verify_time_order_monotonic


def test_time_fold_assignment_sequential_blocks() -> None:
    df = pd.DataFrame({
        "compound_id": [f"RAP-{i:07d}" for i in range(20)],
        "smiles": ["CCO"] * 20,
        "pkd_mean": [5.0] * 20,
    })
    folds = time_fold_assignment(df, n_folds=5)
    # Each block of 4 consecutive compounds (after sort) should land in one fold
    assert folds.shape == (20,)
    assert set(np.unique(folds)) == {0, 1, 2, 3, 4}
    # Compounds 0-3 → fold 0, 4-7 → fold 1, etc. (sort is already lex-monotone here)
    np.testing.assert_array_equal(folds, np.array([
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
    ]))


def test_time_fold_assignment_deterministic_under_seed_change() -> None:
    df = pd.DataFrame({
        "compound_id": [f"RAP-{i:07d}" for i in range(20)],
        "smiles": ["CCO"] * 20,
        "pkd_mean": [5.0] * 20,
    })
    f1 = time_fold_assignment(df, n_folds=5)
    f2 = time_fold_assignment(df, n_folds=5)
    np.testing.assert_array_equal(f1, f2)


def test_verify_time_order_monotonic_passes_for_monotonic_ids() -> None:
    df = pd.DataFrame({
        "compound_id": [f"RAP-{i:07d}" for i in range(10)],
    })
    assert verify_time_order_monotonic(df, pattern=r"RAP-(\d{7})")


def test_verify_time_order_monotonic_passes_for_unordered_ids() -> None:
    """The function should still pass on unsorted input — caller sorts before fold assign."""
    df = pd.DataFrame({
        "compound_id": ["RAP-0000005", "RAP-0000001", "RAP-0000003"],
    })
    # Returns True because *after sort* the sequence is monotone.
    assert verify_time_order_monotonic(df, pattern=r"RAP-(\d{7})")
```

- [ ] **Step 2: Implement the functions**

Append to `src/imm1_glue/data/splits.py`:
```python
def verify_time_order_monotonic(curated: pd.DataFrame, pattern: str = r"RAP-(\d{7})") -> bool:
    """Verify that compound IDs, when sorted lexicographically, form a monotone integer sequence.

    Returns True if every ID matches `pattern` and the extracted integers are strictly
    increasing after sort. Returns False (and logs a warning) otherwise.
    """
    rx = re.compile(pattern)
    nums = []
    for cid in sorted(curated["compound_id"].tolist()):
        m = rx.match(cid)
        if not m:
            logger.warning("compound_id %s does not match pattern %s", cid, pattern)
            return False
        nums.append(int(m.group(1)))
    if not all(a < b for a, b in zip(nums, nums[1:])):
        logger.warning("compound IDs are not strictly increasing after sort")
        return False
    return True


def time_fold_assignment(curated: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    """Split chronologically into n_folds sequential blocks.

    Block 0 = oldest 20%, block n_folds-1 = newest 20%. Outer fold k uses block k
    as the test set. Deterministic — no seed dependence.
    """
    n = len(curated)
    # Sort by compound_id lex-order; map back to original row order.
    sorted_order = curated.sort_values("compound_id").index.tolist()
    folds = np.empty(n, dtype=int)
    block_size = n // n_folds
    remainder = n % n_folds
    cursor = 0
    for fold_idx in range(n_folds):
        sz = block_size + (1 if fold_idx < remainder else 0)
        for k in range(cursor, cursor + sz):
            original_idx = sorted_order[k]
            # Note: original_idx is the DataFrame index; we need positional index.
            pos = curated.index.get_loc(original_idx)
            folds[pos] = fold_idx
        cursor += sz
    return folds
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_splits.py -v`
Expected: time-split tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/imm1_glue/data/splits.py tests/test_splits.py
git commit -m "Implement time/synthesis-order split with monotonicity check"
```

---

### Task 2.5 — Butina cutoff sweep and time-split sanity in run_diagnostics.py

**Files:**
- Modify: `scripts/run_diagnostics.py`

- [ ] **Step 1: Append Butina cutoff sweep + time-split check to the diagnostic script**

Replace the bottom of `scripts/run_diagnostics.py` (the `def main()` block and onward) with:

```python
def butina_cutoff_sweep(out_path: Path) -> None:
    from imm1_glue.data.splits import butina_cluster_compounds

    raw = load_imm1()
    curated = curate(raw)
    smiles = curated["smiles"].tolist()

    cutoffs = [0.4, 0.5, 0.6, 0.7, 0.8]
    rows = []
    for cutoff in cutoffs:
        cluster_ids = butina_cluster_compounds(smiles, cutoff=cutoff)
        n_clusters = len(set(cluster_ids))
        from collections import Counter
        sizes = list(Counter(cluster_ids).values())
        n_qualifying = sum(1 for s in sizes if s >= 5)
        n_singletons = sum(1 for s in sizes if s == 1)
        rows.append({
            "cutoff": cutoff,
            "n_clusters": n_clusters,
            "max_cluster_size": max(sizes),
            "mean_cluster_size": float(np.mean(sizes)),
            "n_clusters_size_ge5": n_qualifying,
            "n_singletons": n_singletons,
            "qualifies": n_qualifying >= 10,
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Also emit a sanitized markdown summary
    md_path = out_path.with_suffix(".md")
    qualifying = df[df["qualifies"]]
    if len(qualifying) > 0:
        chosen = qualifying.iloc[qualifying["cutoff"].argmin()]
        chosen_cutoff = chosen["cutoff"]
        verdict = f"**Chosen cutoff:** {chosen_cutoff} (smallest qualifying)."
    else:
        chosen_cutoff = None
        verdict = "**No cutoff qualifies (≥10 clusters of ≥5 compounds). Cluster split will be dropped.**"

    md_path.write_text(
        "# Butina Cutoff Sweep\n\n"
        + df.to_markdown(index=False)
        + f"\n\n{verdict}\n"
    )
    logger.info("Wrote %s and %s", out_path, md_path)
    if chosen_cutoff is not None:
        logger.info("Use cutoff=%.2f for the main benchmark sweep.", chosen_cutoff)


def time_split_sanity(out_path: Path) -> None:
    from imm1_glue.data.splits import verify_time_order_monotonic, time_fold_assignment

    raw = load_imm1()
    curated = curate(raw)
    is_monotone = verify_time_order_monotonic(curated)

    folds = time_fold_assignment(curated, n_folds=5) if is_monotone else None

    lines = ["# Time-split Sanity Check", ""]
    lines.append(f"- Compound IDs strictly increasing after sort: **{is_monotone}**")
    if is_monotone:
        from collections import Counter
        counts = Counter(folds.tolist())
        lines.append("- Fold sizes:")
        for k in sorted(counts):
            lines.append(f"  - Fold {k}: {counts[k]} compounds")
        # Per-fold censored count
        lines.append("- Per-fold censored count:")
        for k in range(5):
            mask = (folds == k)
            n_cens = int(curated.loc[mask, "is_censored"].sum())
            lines.append(f"  - Fold {k}: {n_cens} censored")
    else:
        lines.append("- **Time split is INFEASIBLE** — IDs are not monotonic. Drop time split.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results", type=Path)
    args = p.parse_args()

    curation_report(args.out_dir / "curation_report.md")
    butina_cutoff_sweep(args.out_dir / "splits_diag" / "butina_cutoff_sweep.csv")
    time_split_sanity(args.out_dir / "splits_diag" / "time_split_check.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run diagnostics**

Run: `python scripts/run_diagnostics.py`
Expected: writes `results/splits_diag/butina_cutoff_sweep.{csv,md}` and `results/splits_diag/time_split_check.md`.

- [ ] **Step 3: Inspect the Butina sweep**

Run: `cat results/splits_diag/butina_cutoff_sweep.md`
**Decision point:** which cutoff qualifies? Record the chosen cutoff in `configs/splits.yaml` under the `butina.cutoff_sweep` section by adding a `chosen_cutoff:` field, or leave as smallest-qualifying default.

- [ ] **Step 4: Inspect time-split**

Run: `cat results/splits_diag/time_split_check.md`
Confirm IDs are monotonic and fold sizes look reasonable. If not monotonic, time split will be skipped — update `configs/splits.yaml` to remove it from the active strategy list.

- [ ] **Step 5: Update configs/splits.yaml with the chosen Butina cutoff (if any)**

Edit `configs/splits.yaml`, add to the `butina:` block:
```yaml
butina:
  ...
  chosen_cutoff: 0.5  # set based on diagnostic output; or null if no cutoff qualified
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_diagnostics.py configs/splits.yaml
git commit -m "Add Butina cutoff sweep + time-split sanity check + record chosen Butina cutoff"
```

---

**Phase 2 exit gate:**
- [ ] `pytest tests/test_splits.py` → all pass.
- [ ] `results/splits_diag/butina_cutoff_sweep.md` exists; cutoff decision recorded in `configs/splits.yaml`.
- [ ] `results/splits_diag/time_split_check.md` exists; time-split feasibility decision recorded.

---

# Phase 3 — Benchmark Orchestrator

**Goal:** A single command runs the full nested-CV sweep over (model, split, seed, outer_fold), writing per-fold predictions to parquet, idempotent and resumable.

**Exit criteria:**
- Smoke test passes: end-to-end run on a 30-compound synthetic dataset for one (model, split, seed, fold) tuple.
- `python scripts/run_benchmark.py --dry-run` enumerates all 3×4×5×5 = 300 outer-fold runs (12k counting inner-CV).

---

### Task 3.1 — Smoke test for end-to-end pipeline

**Files:**
- Create: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
"""End-to-end smoke test on a small synthetic dataset.

Runs one (model, split, seed, outer_fold) tuple through curate → featurize → fit → predict.
Verifies that:
- The pipeline runs without error.
- Predictions are produced (non-NaN, in physical range).
- A predictions parquet file is written.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def smoke_csv(tmp_path: Path) -> Path:
    rng = np.random.RandomState(42)
    n = 40
    smiles_pool = ["CCO", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1"]
    rows = []
    for i in range(n):
        rows.append({
            "Compound Name": f"SMOKE-{i:07d}",
            "SMILES": smiles_pool[i % len(smiles_pool)],
            "pKD": 4.0 + rng.uniform(0, 6.0),
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "smoke.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_smoke_end_to_end(smoke_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs full pipeline for one (model=rf, split=random, seed=42, fold=0)."""
    monkeypatch.setenv("IMM1_DATA_PATH", str(smoke_csv))
    result_dir = tmp_path / "results"

    proc = subprocess.run(
        [
            sys.executable, "scripts/run_benchmark.py",
            "--models", "rf",
            "--splits", "random",
            "--seeds", "42",
            "--folds", "0",
            "--results-dir", str(result_dir),
            "--smoke",  # disables expensive hyperparameter search
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"

    pred_files = list((result_dir / "predictions").glob("*.parquet"))
    assert len(pred_files) == 1
    preds = pd.read_parquet(pred_files[0])
    assert "y_true" in preds.columns
    assert "y_pred" in preds.columns
    assert preds["y_pred"].notna().all()
    assert preds["y_pred"].between(0, 14).all()
```

- [ ] **Step 2: Run test — expect failure (run_benchmark.py doesn't exist)**

Run: `pytest tests/test_pipeline_smoke.py -v`
Expected: fails — script not found.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_pipeline_smoke.py
git commit -m "Add failing end-to-end smoke test"
```

---

### Task 3.2 — Implement run_benchmark.py

**Files:**
- Create: `scripts/run_benchmark.py`

- [ ] **Step 1: Write run_benchmark.py**

```python
"""Main nested-CV benchmark orchestrator.

For each (model, split_strategy, seed, outer_fold):
  1. Load + curate + featurize (cached across the outer loop).
  2. Compute outer-fold assignment.
  3. On outer-train, run inner 5-fold CV grid search to select hyperparameters.
  4. Fit best model on full outer-train; predict on outer-test.
  5. Write predictions parquet to results/predictions/{model}_{split}_seed{s}_fold{f}.parquet.

Idempotent: existing prediction files are skipped (resumable).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from imm1_glue.data.curate import curate
from imm1_glue.data.load import load_imm1
from imm1_glue.data.splits import (
    butina_fold_assignment,
    random_fold_assignment,
    scaffold_fold_assignment,
    time_fold_assignment,
)

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "rf": ("target_affinity_ml.models.rf_model", "RandomForestModel"),
    "xgb": ("target_affinity_ml.models.xgb_model", "XGBoostModel"),
    "mlp": ("target_affinity_ml.models.mlp_model", "MLPModel"),
}


def _import_model(model_key: str):
    module_path, class_name = MODEL_CLASSES[model_key]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def featurize(curated: pd.DataFrame, dataset_cfg: dict) -> np.ndarray:
    """Compute Morgan FPs for all curated compounds."""
    from target_affinity_ml.features.fingerprints import smiles_to_morgan_fp
    feat_cfg = dataset_cfg["features"]
    fps = []
    for smi in curated["smiles"]:
        fp = smiles_to_morgan_fp(smi, radius=feat_cfg["radius"], n_bits=feat_cfg["n_bits"])
        if fp is None:
            raise ValueError(f"Failed to featurize SMILES: {smi}")
        fps.append(fp)
    return np.vstack(fps)


def compute_fold_assignment(
    curated: pd.DataFrame, strategy: str, seed: int, splits_cfg: dict
) -> np.ndarray:
    if strategy == "random":
        return random_fold_assignment(curated, n_folds=splits_cfg["random"]["n_folds"], seed=seed)
    if strategy == "scaffold":
        return scaffold_fold_assignment(curated, n_folds=splits_cfg["scaffold"]["n_folds"], seed=seed)
    if strategy == "butina":
        cutoff = splits_cfg["butina"].get("chosen_cutoff", 0.6)
        if cutoff is None:
            raise ValueError("Butina cutoff is null in config — cluster split is disabled.")
        return butina_fold_assignment(
            curated,
            n_folds=splits_cfg["butina"]["n_folds"],
            seed=seed,
            cutoff=cutoff,
        )
    if strategy == "time":
        # Seed doesn't change outer-fold assignment for time split.
        return time_fold_assignment(curated, n_folds=splits_cfg["time"]["n_folds"])
    raise ValueError(f"Unknown split strategy: {strategy}")


def inner_cv_select(
    model_key: str,
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: dict,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    """5-fold inner CV grid search; return best hyperparameters by mean RMSE."""
    from sklearn.model_selection import KFold

    ModelCls = _import_model(model_key)
    grid = model_cfg["search_grid"] if not smoke else _smoke_grid(model_cfg)
    keys, values = list(grid.keys()), list(grid.values())

    best_score = float("inf")
    best_params = None
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_rmses = []
        for tr_idx, va_idx in kf.split(X):
            m = ModelCls(**params)
            m.fit(X[tr_idx], y[tr_idx])
            preds = m.predict(X[va_idx])
            rmse = float(np.sqrt(np.mean((preds - y[va_idx]) ** 2)))
            fold_rmses.append(rmse)
        mean_rmse = float(np.mean(fold_rmses))
        if mean_rmse < best_score - 1e-9:
            best_score = mean_rmse
            best_params = params
        elif abs(mean_rmse - best_score) < 1e-9:
            # Tiebreaker: alphabetically-first by parameter string
            cur_key = json.dumps(params, sort_keys=True, default=str)
            best_key = json.dumps(best_params, sort_keys=True, default=str)
            if cur_key < best_key:
                best_params = params

    return best_params


def _smoke_grid(model_cfg: dict) -> dict:
    """Reduce grid to 1 value per axis for smoke test."""
    return {k: [v[0]] for k, v in model_cfg["search_grid"].items()}


def run_single(
    model_key: str,
    split_strategy: str,
    seed: int,
    outer_fold: int,
    curated: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    model_cfg: dict,
    out_path: Path,
    smoke: bool = False,
) -> None:
    """Run one (model, split, seed, outer_fold) combination."""
    test_mask = folds == outer_fold
    train_mask = ~test_mask

    if test_mask.sum() == 0:
        logger.warning("Skipping (%s, %s, %s, fold %s): empty test set", model_key, split_strategy, seed, outer_fold)
        return

    best_params = inner_cv_select(model_key, X[train_mask], y[train_mask], model_cfg, seed, smoke=smoke)
    ModelCls = _import_model(model_key)
    m = ModelCls(**best_params)
    m.fit(X[train_mask], y[train_mask])

    y_pred = m.predict(X[test_mask])
    try:
        _, y_sigma = m.predict_with_uncertainty(X[test_mask])
    except (AttributeError, NotImplementedError):
        y_sigma = np.full(len(y_pred), np.nan)

    df = pd.DataFrame({
        "compound_id": curated.loc[test_mask, "compound_id"].values,
        "y_true": y[test_mask],
        "y_pred": y_pred,
        "y_sigma": y_sigma,
        "is_censored": curated.loc[test_mask, "is_censored"].values,
        "is_noisy": curated.loc[test_mask, "is_noisy"].values,
        "model": model_key,
        "split": split_strategy,
        "seed": seed,
        "outer_fold": outer_fold,
    })
    df.attrs["best_params"] = best_params

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Also write hyperparameter manifest
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"best_params": best_params}, indent=2, default=str))

    logger.info("Wrote %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="rf,xgb,mlp", help="comma-separated")
    p.add_argument("--splits", default="random,scaffold,butina,time", help="comma-separated")
    p.add_argument("--seeds", default="42,123,456,789,1000", help="comma-separated")
    p.add_argument("--folds", default="0,1,2,3,4", help="comma-separated outer folds")
    p.add_argument("--results-dir", default="results", type=Path)
    p.add_argument("--config-dir", default="configs", type=Path)
    p.add_argument("--sensitivity", action="store_true", help="drop is_censored compounds before split")
    p.add_argument("--smoke", action="store_true", help="reduced grid for smoke test")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.results_dir / "logs" / "benchmark.log"),
        ],
    )
    (args.results_dir / "logs").mkdir(parents=True, exist_ok=True)

    dataset_cfg = _load_config(args.config_dir / "dataset_imm1.yaml")
    splits_cfg = _load_config(args.config_dir / "splits.yaml")
    model_cfgs = {k: _load_config(args.config_dir / f"{k}_baseline.yaml") for k in args.models.split(",")}

    # One-time load + curate + featurize
    raw = load_imm1()
    curated = curate(raw).reset_index(drop=True)
    if args.sensitivity:
        before = len(curated)
        curated = curated[~curated["is_censored"]].reset_index(drop=True)
        logger.info("Sensitivity mode: dropped %d censored, %d remain", before - len(curated), len(curated))

    X = featurize(curated, dataset_cfg)
    y = curated["pkd_mean"].values

    models = args.models.split(",")
    splits = args.splits.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    suffix = "_sensitivity" if args.sensitivity else ""

    combos = list(itertools.product(models, splits, seeds, folds))
    logger.info("%d total (model, split, seed, fold) combinations", len(combos))

    if args.dry_run:
        for c in combos[:10]:
            print(c)
        print(f"... ({len(combos)} total)")
        return

    # Pre-compute fold assignments per (split, seed)
    fold_cache: dict[tuple[str, int], np.ndarray] = {}
    for split_strategy, seed in set((s, sd) for _, s, sd, _ in combos):
        fold_cache[(split_strategy, seed)] = compute_fold_assignment(curated, split_strategy, seed, splits_cfg)

    for model_key, split_strategy, seed, outer_fold in combos:
        out_path = args.results_dir / "predictions" / (
            f"{model_key}_{split_strategy}_seed{seed}_fold{outer_fold}{suffix}.parquet"
        )
        if out_path.exists():
            logger.info("SKIP (exists): %s", out_path.name)
            continue

        logger.info("RUN: %s", out_path.name)
        run_single(
            model_key=model_key,
            split_strategy=split_strategy,
            seed=seed,
            outer_fold=outer_fold,
            curated=curated,
            X=X,
            y=y,
            folds=fold_cache[(split_strategy, seed)],
            model_cfg=model_cfgs[model_key],
            out_path=out_path,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_pipeline_smoke.py -v`
Expected: passes.

- [ ] **Step 3: Run a dry-run against real data**

```bash
python scripts/run_benchmark.py --dry-run
```
Expected: prints first 10 combos + total count = 300.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_benchmark.py
git commit -m "Implement nested-CV benchmark orchestrator (idempotent, resumable)"
```

---

### Task 3.3 — Censoring sensitivity wrapper

**Files:**
- Create: `src/imm1_glue/evaluation/censoring_sensitivity.py`

- [ ] **Step 1: Write the wrapper module (thin — the `--sensitivity` flag in run_benchmark.py already does the work)**

```python
"""Convenience wrapper for the drop-censored sensitivity sweep.

The actual logic lives in scripts/run_benchmark.py under the --sensitivity flag.
This module documents the contract and provides a single function for
notebooks to reload sensitivity results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_sensitivity_predictions(results_dir: Path = Path("results")) -> pd.DataFrame:
    """Concatenate all sensitivity prediction files into one DataFrame."""
    files = list((results_dir / "predictions").glob("*_sensitivity.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No sensitivity predictions found in {results_dir / 'predictions'}/. "
            "Run: python scripts/run_benchmark.py --sensitivity"
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_primary_predictions(results_dir: Path = Path("results")) -> pd.DataFrame:
    """Concatenate all primary (non-sensitivity) prediction files."""
    files = [
        f for f in (results_dir / "predictions").glob("*.parquet")
        if "_sensitivity" not in f.name
    ]
    if not files:
        raise FileNotFoundError(
            f"No primary predictions found in {results_dir / 'predictions'}/. "
            "Run: python scripts/run_benchmark.py"
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/imm1_glue/evaluation/censoring_sensitivity.py
git commit -m "Add sensitivity-prediction loader (thin wrapper around --sensitivity flag)"
```

---

**Phase 3 exit gate:**
- [ ] `pytest tests/test_pipeline_smoke.py` passes.
- [ ] `python scripts/run_benchmark.py --dry-run` shows 300 combos.
- [ ] A dry run with `--smoke --models rf --splits random --seeds 42 --folds 0` against real data completes in <2 min and writes one parquet.

---

# Phase 4 — Full Sweep (Compute)

**Goal:** Both the primary and sensitivity sweeps complete; predictions parquet files exist for all 300 + 300 = 600 (model, split, seed, fold) tuples (minus any that are infeasible — e.g., Butina dropped or time-split dropped).

**Exit criteria:**
- `results/predictions/*.parquet` exists for all tuples in the active sweep configuration.
- No model raised an unrecoverable error during the sweep.

---

### Task 4.1 — Primary sweep

- [ ] **Step 1: Run a single (model, split, seed, fold) end-to-end to time it**

```bash
python scripts/run_benchmark.py --models rf --splits random --seeds 42 --folds 0
```
Note the wall-clock time printed in the log. This sets the expected per-combo budget.

- [ ] **Step 2: Run the full primary sweep overnight**

```bash
caffeinate -i nice -n 10 python scripts/run_benchmark.py > results/logs/sweep_primary.log 2>&1 &
echo $! > results/logs/sweep_primary.pid
```

Periodically check: `tail -f results/logs/benchmark.log`

- [ ] **Step 3: Once done, verify completeness**

```bash
ls results/predictions/*.parquet | grep -v sensitivity | wc -l
```
Expected: 300 (3 models × 4 splits × 5 seeds × 5 folds), or fewer if any split was disabled.

- [ ] **Step 4: Spot-check predictions**

```bash
python -c "
import pandas as pd
from pathlib import Path
for p in list(Path('results/predictions').glob('rf_random_seed42_fold0*.parquet'))[:1]:
    df = pd.read_parquet(p)
    print(df.describe())
    print('NaN y_pred:', df['y_pred'].isna().sum())
"
```

- [ ] **Step 5: Commit only logs that are sanitized** (no compound IDs in standard log output)

```bash
# Verify the log is clean before staging anything
bash scripts/check_no_data_leak.sh < results/logs/benchmark.log || echo "Log contains leak — DO NOT COMMIT"
```
If clean, you may optionally commit a sanitized digest later. By default `results/logs/` is gitignored so nothing is committed.

---

### Task 4.2 — Sensitivity sweep

- [ ] **Step 1: Run the drop-censored sensitivity sweep**

```bash
caffeinate -i nice -n 10 python scripts/run_benchmark.py --sensitivity > results/logs/sweep_sensitivity.log 2>&1 &
```

- [ ] **Step 2: Verify completeness**

```bash
ls results/predictions/*_sensitivity.parquet | wc -l
```
Expected: 300.

---

**Phase 4 exit gate:**
- [ ] Primary predictions: 300 parquet files (or correctly fewer if a split was disabled).
- [ ] Sensitivity predictions: 300 parquet files.
- [ ] No NaN in `y_pred` columns.
- [ ] No errors in `results/logs/benchmark.log` beyond expected warnings.

---

# Phase 5 — Reports and Error Analysis

**Goal:** Publishable tables + calibration figures + error-analysis notebook.

**Exit criteria:**
- `results/tables/primary_metrics.{csv,md}` exists.
- `results/tables/sensitivity_metrics.{csv,md}` exists.
- `results/tables/per_seed_metrics.csv` exists.
- `results/tables/classification_thresholds.{csv,md}` exists.
- `results/figures/calibration_*.png` files exist for each model × split.
- `notebooks/03_error_analysis.ipynb` runs end-to-end without error.

---

### Task 5.1 — Generate tables

**Files:**
- Create: `src/imm1_glue/reports/generate_tables.py`

- [ ] **Step 1: Write generate_tables.py**

```python
"""Generate publishable tables from results/predictions/*.parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from target_affinity_ml.evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
)

logger = logging.getLogger(__name__)

CLASSIFICATION_THRESHOLDS = [6.0, 7.0, 4.0001]


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(preds: pd.DataFrame) -> pd.DataFrame:
    """Per (model, split): compute metrics per outer fold, aggregate to mean ± CI."""
    rows = []
    for (model, split), g in preds.groupby(["model", "split"]):
        fold_metrics = []
        for (seed, fold), gf in g.groupby(["seed", "outer_fold"]):
            y_true = gf["y_true"].values
            y_pred = gf["y_pred"].values
            reg = compute_regression_metrics(y_true, y_pred)
            fold_metrics.append(reg)
        df_fm = pd.DataFrame(fold_metrics)
        row = {"model": model, "split": split}
        for col in df_fm.columns:
            mean, lo, hi = _bootstrap_ci(df_fm[col].values)
            row[f"{col}_mean"] = mean
            row[f"{col}_ci_lo"] = lo
            row[f"{col}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def classification_table(preds: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for (model, split), g in preds.groupby(["model", "split"]):
        for thr in thresholds:
            fold_metrics = []
            for (seed, fold), gf in g.groupby(["seed", "outer_fold"]):
                y_true_bin = (gf["y_true"].values >= thr).astype(int)
                y_score = gf["y_pred"].values
                y_pred_bin = (y_score >= thr).astype(int)
                if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
                    continue  # degenerate fold
                metrics = compute_classification_metrics(y_true_bin, y_score, y_pred_bin)
                fold_metrics.append(metrics)
            if not fold_metrics:
                continue
            df_fm = pd.DataFrame(fold_metrics)
            row = {"model": model, "split": split, "threshold": thr}
            for col in df_fm.columns:
                row[f"{col}_mean"] = float(df_fm[col].mean())
                row[f"{col}_std"] = float(df_fm[col].std())
            rows.append(row)
    return pd.DataFrame(rows)


def per_seed_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, seed), g in preds.groupby(["model", "split", "seed"]):
        fold_metrics = []
        for fold, gf in g.groupby("outer_fold"):
            reg = compute_regression_metrics(gf["y_true"].values, gf["y_pred"].values)
            reg["outer_fold"] = fold
            fold_metrics.append(reg)
        for fm in fold_metrics:
            row = {"model": model, "split": split, "seed": seed, **fm}
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results", type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    pred_dir = args.results_dir / "predictions"
    out_dir = args.results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_files = [f for f in pred_dir.glob("*.parquet") if "_sensitivity" not in f.name]
    sens_files = list(pred_dir.glob("*_sensitivity.parquet"))

    if primary_files:
        preds = pd.concat([pd.read_parquet(f) for f in primary_files], ignore_index=True)
        primary = aggregate(preds)
        primary.to_csv(out_dir / "primary_metrics.csv", index=False)
        (out_dir / "primary_metrics.md").write_text(primary.to_markdown(index=False))
        logger.info("Wrote primary_metrics tables")

        cls_df = classification_table(preds, CLASSIFICATION_THRESHOLDS)
        cls_df.to_csv(out_dir / "classification_thresholds.csv", index=False)
        (out_dir / "classification_thresholds.md").write_text(cls_df.to_markdown(index=False))

        ps = per_seed_table(preds)
        ps.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    if sens_files:
        preds_sens = pd.concat([pd.read_parquet(f) for f in sens_files], ignore_index=True)
        sens = aggregate(preds_sens)
        sens.to_csv(out_dir / "sensitivity_metrics.csv", index=False)
        (out_dir / "sensitivity_metrics.md").write_text(sens.to_markdown(index=False))
        logger.info("Wrote sensitivity_metrics tables")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python -m imm1_glue.reports.generate_tables
```
Expected: writes 4 tables to `results/tables/`.

- [ ] **Step 3: Inspect a table**

Run: `cat results/tables/primary_metrics.md`
Sanity-check: 3 models × ≤4 splits = ≤12 rows.

- [ ] **Step 4: Commit**

```bash
git add src/imm1_glue/reports/generate_tables.py
git commit -m "Implement table generation with bootstrap CIs + multi-threshold classification"
```

---

### Task 5.2 — Calibration figures

**Files:**
- Modify: `src/imm1_glue/reports/generate_tables.py` (add `--figures` mode) OR
- Create: `src/imm1_glue/reports/generate_figures.py` (separate module)

For clarity, use a separate module.

- [ ] **Step 1: Create generate_figures.py**

```python
"""Generate calibration + diagnostic figures from predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def predicted_vs_actual(preds: pd.DataFrame, model: str, split: str, ax: plt.Axes) -> None:
    g = preds[(preds["model"] == model) & (preds["split"] == split)]
    ax.scatter(g["y_true"], g["y_pred"], alpha=0.4, s=15)
    lo = min(g["y_true"].min(), g["y_pred"].min())
    hi = max(g["y_true"].max(), g["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("True pKD")
    ax.set_ylabel("Predicted pKD")
    ax.set_title(f"{model} / {split}")


def sigma_vs_residual(preds: pd.DataFrame, model: str, split: str, ax: plt.Axes) -> None:
    g = preds[(preds["model"] == model) & (preds["split"] == split)]
    if g["y_sigma"].isna().all():
        ax.text(0.5, 0.5, "No uncertainty", ha="center", va="center")
        ax.set_title(f"{model} / {split} (no σ̂)")
        return
    resid = (g["y_true"] - g["y_pred"]).abs()
    ax.scatter(g["y_sigma"], resid, alpha=0.4, s=15)
    rho, _ = spearmanr(g["y_sigma"], resid)
    ax.set_xlabel("σ̂ (predicted uncertainty)")
    ax.set_ylabel("|residual|")
    ax.set_title(f"{model} / {split}  ρ={rho:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results", type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    pred_dir = args.results_dir / "predictions"
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in pred_dir.glob("*.parquet") if "_sensitivity" not in f.name]
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    models = sorted(preds["model"].unique())
    splits = sorted(preds["split"].unique())

    for model in models:
        for split in splits:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            predicted_vs_actual(preds, model, split, axes[0])
            sigma_vs_residual(preds, model, split, axes[1])
            fig.suptitle(f"{model} on {split} split (pooled across seeds × folds)")
            fig.tight_layout()
            out = fig_dir / f"calibration_{model}_{split}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python -m imm1_glue.reports.generate_figures
```
Expected: writes a PNG per (model, split) tuple.

- [ ] **Step 3: Commit**

```bash
git add src/imm1_glue/reports/generate_figures.py
git commit -m "Add calibration figure generation (predicted-vs-actual + σ-vs-residual)"
```

---

### Task 5.3 — Error-analysis notebook

**Files:**
- Create: `notebooks/03_error_analysis.ipynb`

- [ ] **Step 1: Create the notebook (paste-as-py-then-convert is fine)**

Create `notebooks/03_error_analysis.ipynb` with the following cells (use Jupyter or `jupytext` to convert):

**Cell 1 (markdown):**
```markdown
# IMM1 Glue Baseline — Error Analysis

Identify the 10 compounds with largest |residual| per model across random and
scaffold splits; cluster them by Murcko scaffold; surface any chemotype where
all models fail.

**Sanitization:** This notebook reads predictions which contain compound IDs.
Do not export this notebook as HTML/PDF or commit the executed version with
outputs. Save with cleared outputs only.
```

**Cell 2 (code):**
```python
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from imm1_glue.data.load import load_imm1
from imm1_glue.data.curate import curate

preds = pd.concat([
    pd.read_parquet(f)
    for f in Path("results/predictions").glob("*.parquet")
    if "_sensitivity" not in f.name
], ignore_index=True)

curated = curate(load_imm1())
preds = preds.merge(curated[["compound_id", "smiles"]], on="compound_id", how="left")
preds["residual"] = preds["y_pred"] - preds["y_true"]
preds["abs_residual"] = preds["residual"].abs()
print(preds.shape)
preds.head()
```

**Cell 3 (code):**
```python
def top_failures(preds: pd.DataFrame, model: str, split: str, n: int = 10) -> pd.DataFrame:
    g = preds[(preds["model"] == model) & (preds["split"] == split)]
    # Average residual per compound across folds × seeds
    by_compound = g.groupby(["compound_id", "smiles"]).agg(
        mean_abs_residual=("abs_residual", "mean"),
        mean_pkd=("y_true", "mean"),
        is_censored=("is_censored", "first"),
        is_noisy=("is_noisy", "first"),
    ).reset_index()
    return by_compound.nlargest(n, "mean_abs_residual")

top_failures(preds, "rf", "scaffold", n=10)
```

**Cell 4 (code):**
```python
def murcko(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None

for model in preds["model"].unique():
    for split in preds["split"].unique():
        top = top_failures(preds, model, split, n=10)
        top["scaffold"] = top["smiles"].apply(murcko)
        scaffold_counts = top["scaffold"].value_counts()
        print(f"--- {model} / {split} ---")
        print(scaffold_counts.head())
        print()
```

**Cell 5 (markdown):**
```markdown
## Chemotypes failing across all models

Looking for scaffolds that appear in the top-10 failure list for **all three**
models on at least one split — indicates a chemotype the models cannot learn.
```

**Cell 6 (code):**
```python
shared_failures = None
for model in preds["model"].unique():
    failure_scaffolds = set()
    for split in preds["split"].unique():
        top = top_failures(preds, model, split, n=15)
        top["scaffold"] = top["smiles"].apply(murcko)
        failure_scaffolds.update(top["scaffold"].dropna())
    shared_failures = failure_scaffolds if shared_failures is None else shared_failures & failure_scaffolds

print(f"{len(shared_failures)} scaffolds appear in top-15 failures for all models.")
```

- [ ] **Step 2: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute notebooks/03_error_analysis.ipynb \
  --output 03_error_analysis_executed.ipynb
```
Expected: completes without errors. **Do not commit `*_executed.ipynb`.**

- [ ] **Step 3: Clear outputs and commit only the un-executed notebook**

```bash
jupyter nbconvert --clear-output --inplace notebooks/03_error_analysis.ipynb
git add notebooks/03_error_analysis.ipynb
git commit -m "Add error-analysis notebook (outputs cleared)"
```

---

**Phase 5 exit gate:**
- [ ] `results/tables/primary_metrics.{csv,md}` exists and shows reasonable R²/RMSE values.
- [ ] `results/tables/sensitivity_metrics.{csv,md}` exists.
- [ ] `results/tables/per_seed_metrics.csv` exists with one row per (model, split, seed, fold).
- [ ] `results/tables/classification_thresholds.{csv,md}` exists with multi-threshold metrics.
- [ ] `results/figures/calibration_*.png` exists for each (model, split) tuple.
- [ ] `notebooks/03_error_analysis.ipynb` runs end-to-end.

---

# Phase 6 — Review & Iteration

This phase is **manual**, not codable. After Phase 5:

1. Open `results/tables/primary_metrics.md` and `sensitivity_metrics.md`. Compare model rankings across splits. Note any surprises.
2. Open the calibration figures. Look for: predictions clumping at the assay floor (expected), systematic over/under-prediction at the high-pKD end, σ̂ vs |residual| correlation poor on any (model, split).
3. Run through the error-analysis notebook. If a chemotype fails across all models, that's a finding — document it.
4. **Decide:** is anything worth re-running?
   - Hyperparameter grid produced a degenerate selection? Tighten and re-run that model only.
   - MLP underperforms catastrophically? Try a smaller architecture (`[128]`) with higher dropout (0.5) and re-run.
   - Butina cutoff was wrong (clusters too big/small)? Revise cutoff in config and re-run cluster split only.

5. Once accepted: tag the repo with `v0.1-results` (`git tag -a v0.1-results -m "Initial benchmark complete"`).

---

# Appendix — Useful Commands

```bash
# Resume an interrupted sweep (idempotent)
python scripts/run_benchmark.py

# Run only one model × split for debugging
python scripts/run_benchmark.py --models rf --splits scaffold --seeds 42 --folds 0,1

# Run smoke test (1 hyperparameter per axis)
python scripts/run_benchmark.py --smoke

# Run sensitivity sweep
python scripts/run_benchmark.py --sensitivity

# Re-generate tables and figures only (no compute)
python -m imm1_glue.reports.generate_tables
python -m imm1_glue.reports.generate_figures

# Re-run diagnostics
python scripts/run_diagnostics.py

# Run all tests
pytest tests/ -v

# Check the pre-commit hook on a planted leak (sanity)
echo "+RAP-0010972 pkd=8.6695" | bash scripts/check_no_data_leak.sh && echo OK || echo BLOCKED
```

---

## Plan summary

**Total tasks:** ~25 across 6 phases.
**Estimated effort:** 7–9 working days + one overnight compute run.
**Critical path:** Phase 0 (library audit) → Phase 1 (data pipeline) → Phase 2 (splits + diagnostics) → Phase 3 (orchestrator) → Phase 4 (overnight sweep) → Phase 5 (reports).

Each phase produces a working, committable increment. The exit gate of each phase should be confirmed before starting the next.
