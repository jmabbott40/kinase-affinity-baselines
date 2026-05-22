# Plan 2: GPCR Aminergic Data Pipeline + Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the `target-affinity-ml` library to be genuinely class-agnostic (v1.1.0), then build the aminergic GPCR data pipeline and run the full 7-model × 3-split × 5-seed benchmark — producing the GPCR half of the cross-class comparison.

**Architecture:** Part A refactors the library's `chembl_fetcher.py` and `curate.py` to accept a `TargetClassConfig` describing how to identify and curate a target class, replacing hardcoded kinase logic. Parts B-F build the `gpcr-aminergic-benchmarks` application repo: data ingestion, curation, splitting, feature engineering (Morgan FP, RDKit descriptors, ESM-2 embeddings), and the 105-run benchmark on AWS.

**Tech Stack:** Python 3.11, `target-affinity-ml` library, ChEMBL API, RDKit, scikit-learn, XGBoost, PyTorch + torch-geometric + fair-esm, AWS g5.12xlarge (4× A10G).

---

## Spec & predecessor references

- **Spec:** `docs/superpowers/specs/2026-04-17-gpcr-aminergic-phase1-design.md` (Sections 4-6)
- **Plan 1 completion summary:** `docs/superpowers/plans/2026-04-30-plan1-completion-summary.md` — see limitations L1-L5
- **Audit decision:** OPTION_A — binding-only protocol, 30/36 aminergic targets viable at ≥500 records
- **Aminergic target list:** `scripts/aminergic_audit/target_lists.py` (kinase repo) — gene-symbol-based, ready to move to the GPCR application repo

Plan 3 (methodology: scaffold diversity + RNS + cross-class comparison) is downstream and depends on this plan completing.

---

## Limitations from Plan 1 addressed here

| ID | Limitation | Addressed by |
|----|-----------|--------------|
| L1 | `chembl_fetcher.py` + `curate.py` have hardcoded kinase logic | Tasks 1-2 |
| L3 | Feature loaders use relative `data/processed` paths | Task 3 |
| L4 | Integration test only covered RF, not deep models | Task 4 |
| L5 | Validation script's NaN-truthiness cosmetic bug | Task 4 |
| L2 | Reference NPZs not on GitHub | Deferred to Plan 3 (cross-class comparison is where kinase references are needed) |

---

## File structure

### `target-affinity-ml` library — Part A modifications (→ v1.1.0)

```
target-affinity-ml/
├── src/target_affinity_ml/
│   ├── data/
│   │   ├── target_class_config.py   # NEW: TargetClassConfig dataclass
│   │   ├── chembl_fetcher.py        # MODIFY: parameterize by TargetClassConfig
│   │   ├── curate.py                # MODIFY: remove hardcoded paths/columns
│   │   └── splits.py                # MODIFY: generalize docstrings; verify target split
│   └── features/
│       └── __init__.py              # MODIFY: add data_dir param to loaders (L3)
├── tests/
│   ├── unit/
│   │   ├── test_target_class_config.py   # NEW
│   │   └── test_chembl_fetcher.py        # NEW (mocked API)
│   └── integration/
│       └── test_deep_model_smoke.py      # NEW (L4)
├── CHANGELOG.md                     # MODIFY: 1.1.0 section
└── pyproject.toml                   # MODIFY: version → 1.1.0
```

### `gpcr-aminergic-benchmarks` — new application repo (Parts B-F)

```
gpcr-aminergic-benchmarks/
├── pyproject.toml                   # depends on target-affinity-ml==1.1.0
├── README.md
├── LICENSE, .gitignore, CHANGELOG.md
├── configs/
│   ├── dataset_aminergic_v1.yaml    # curation parameters
│   └── (7 model configs copied from library defaults)
├── src/gpcr_aminergic_benchmarks/
│   ├── __init__.py
│   ├── target_lists.py              # moved from kinase repo scripts/aminergic_audit/
│   └── target_class.py              # the aminergic TargetClassConfig instance
├── scripts/
│   ├── fetch_gpcr_data.py           # Task 7
│   ├── curate_gpcr_data.py          # Task 8
│   ├── build_gpcr_splits.py         # Task 9
│   ├── build_gpcr_features.py       # Task 10
│   ├── build_gpcr_esm.py            # Task 11
│   └── run_gpcr_benchmark.py        # Task 12-13
├── data/processed/v1/               # gitignored — populated by pipeline
├── results/                         # benchmark outputs
└── notebooks/                       # (Plan 3 will add analysis notebooks)
```

---

# PART A — Library class-agnostic refactor (→ v1.1.0)

## Task 1: Add `TargetClassConfig` + refactor `chembl_fetcher.py`

**Files:**
- Create: `target-affinity-ml/src/target_affinity_ml/data/target_class_config.py`
- Create: `target-affinity-ml/tests/unit/test_target_class_config.py`
- Modify: `target-affinity-ml/src/target_affinity_ml/data/chembl_fetcher.py`
- Create: `target-affinity-ml/tests/unit/test_chembl_fetcher.py`

**Context:** `chembl_fetcher.py` currently hardcodes `KINASE_GO_TERMS`, `_classify_kinase`, `_is_kinase_by_name`. The refactor introduces a `TargetClassConfig` dataclass that captures everything class-specific, so the fetcher works for kinases, GPCRs, or any future class.

- [ ] **Step 1: Write the failing test for `TargetClassConfig`**

Create `tests/unit/test_target_class_config.py`:

```python
"""Tests for the TargetClassConfig abstraction."""
import pytest
from target_affinity_ml.data.target_class_config import TargetClassConfig


def test_minimal_config_construction():
    cfg = TargetClassConfig(
        class_name="kinase",
        go_terms={"GO:0016301"},
        name_keywords=["kinase"],
        raw_filename_stem="chembl_kinase",
    )
    assert cfg.class_name == "kinase"
    assert "GO:0016301" in cfg.go_terms
    assert cfg.raw_activities_filename == "chembl_kinase_activities.parquet"
    assert cfg.raw_targets_filename == "chembl_kinase_targets.parquet"


def test_config_with_explicit_target_ids():
    """A class can be defined by an explicit ChEMBL ID list (GPCR aminergic case)."""
    cfg = TargetClassConfig(
        class_name="gpcr_aminergic",
        explicit_target_ids=["CHEMBL217", "CHEMBL224"],
        raw_filename_stem="chembl_gpcr_aminergic",
    )
    assert cfg.explicit_target_ids == ["CHEMBL217", "CHEMBL224"]
    assert cfg.uses_explicit_target_list is True


def test_config_go_based_is_not_explicit():
    cfg = TargetClassConfig(
        class_name="kinase",
        go_terms={"GO:0016301"},
        raw_filename_stem="chembl_kinase",
    )
    assert cfg.uses_explicit_target_list is False


def test_config_requires_identification_method():
    """Must provide either go_terms or explicit_target_ids."""
    with pytest.raises(ValueError, match="go_terms.*or.*explicit_target_ids"):
        TargetClassConfig(class_name="empty", raw_filename_stem="x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd target-affinity-ml && python -m pytest tests/unit/test_target_class_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'target_affinity_ml.data.target_class_config'`

- [ ] **Step 3: Implement `TargetClassConfig`**

Create `src/target_affinity_ml/data/target_class_config.py`:

```python
"""TargetClassConfig: declares how to identify and curate a protein target class.

This abstraction replaces hardcoded kinase logic so the data pipeline works for
any target class (kinases, GPCRs, proteases, etc.). A class is identified either
by GO molecular-function terms (the kinase approach) or by an explicit list of
ChEMBL target IDs (the GPCR aminergic approach, where the 30 targets are
hand-curated).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TargetClassConfig:
    """Declarative configuration for a protein target class.

    Parameters
    ----------
    class_name : str
        Short identifier, e.g. "kinase" or "gpcr_aminergic".
    raw_filename_stem : str
        Stem for raw data files; e.g. "chembl_kinase" yields
        "chembl_kinase_activities.parquet" and "chembl_kinase_targets.parquet".
    go_terms : set[str]
        GO molecular-function terms identifying the class. Used when
        explicit_target_ids is not provided.
    name_keywords : list[str]
        Keywords that (case-insensitive) appear in target names of this class.
        Used as a secondary filter alongside GO terms.
    explicit_target_ids : list[str] | None
        If provided, the class is defined by exactly these ChEMBL target IDs.
        Takes precedence over go_terms.
    subfamily_map : dict[str, str]
        Optional mapping of target_chembl_id → subfamily name, used for the
        target-held-out split. For kinases this is the kinase group; for
        aminergic GPCRs it is the receptor family (dopamine, serotonin, etc.).
    """

    class_name: str
    raw_filename_stem: str
    go_terms: set[str] = field(default_factory=set)
    name_keywords: list[str] = field(default_factory=list)
    explicit_target_ids: list[str] | None = None
    subfamily_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.go_terms and not self.explicit_target_ids:
            raise ValueError(
                "TargetClassConfig requires either go_terms or "
                "explicit_target_ids to identify the class."
            )

    @property
    def uses_explicit_target_list(self) -> bool:
        """True if the class is defined by an explicit ChEMBL ID list."""
        return self.explicit_target_ids is not None

    @property
    def raw_activities_filename(self) -> str:
        return f"{self.raw_filename_stem}_activities.parquet"

    @property
    def raw_targets_filename(self) -> str:
        return f"{self.raw_filename_stem}_targets.parquet"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_target_class_config.py -v`
Expected: PASS — all 4 tests

- [ ] **Step 5: Refactor `chembl_fetcher.py` to accept `TargetClassConfig`**

Modify `src/target_affinity_ml/data/chembl_fetcher.py`:
- Keep the existing `KINASE_GO_TERMS` set but move it into a module-level `KINASE_CONFIG = TargetClassConfig(...)` instance for backward compatibility.
- Change the main fetch entry point to accept a `config: TargetClassConfig` parameter.
- When `config.uses_explicit_target_list` is True, skip GO-term discovery and fetch activities directly for the listed target IDs.
- When `config.go_terms` is set, use the existing GO-based discovery path.
- Replace `_classify_kinase` with a generic `_classify_subfamily(target_id, config)` that looks up `config.subfamily_map`.
- Rename `fetch_kinase_targets` → `fetch_target_class` (keep a thin `fetch_kinase_targets` wrapper that calls `fetch_target_class(KINASE_CONFIG)` for backward compatibility).

**Important:** Preserve the existing kinase code path exactly — the kinase repo still depends on it. The backward-compat wrapper must produce identical output.

- [ ] **Step 6: Write mocked test for `chembl_fetcher.py`**

Create `tests/unit/test_chembl_fetcher.py` with tests that mock the ChEMBL client and verify:
- `fetch_target_class` with an explicit-ID config queries activities for exactly those IDs
- `fetch_target_class` with a GO-term config runs the discovery path
- The `fetch_kinase_targets` backward-compat wrapper delegates to `fetch_target_class(KINASE_CONFIG)`

Use `unittest.mock.patch` on `chembl_webresource_client.new_client`.

- [ ] **Step 7: Run all data tests**

Run: `python -m pytest tests/unit/test_target_class_config.py tests/unit/test_chembl_fetcher.py -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
cd target-affinity-ml
git add src/target_affinity_ml/data/target_class_config.py \
        src/target_affinity_ml/data/chembl_fetcher.py \
        tests/unit/test_target_class_config.py \
        tests/unit/test_chembl_fetcher.py
git -c commit.gpgsign=false commit -m "Add TargetClassConfig; refactor chembl_fetcher to be class-agnostic"
```

---

## Task 2: Refactor `curate.py` to remove hardcoded kinase logic

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/data/curate.py`
- Create: `target-affinity-ml/tests/unit/test_curate.py`

**Context:** `curate.py` hardcodes `chembl_kinase_activities.parquet`, `chembl_kinase_targets.parquet`, and a `kinase_group` column merge. The refactor makes `curate_activities` accept a `TargetClassConfig` so filenames and the subfamily column are derived from config.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_curate.py` — test that `curate_activities` accepts a `TargetClassConfig`, reads from `config.raw_activities_filename`, and produces a curated DataFrame with a generic `subfamily` column (not `kinase_group`). Use a small synthetic raw-activities DataFrame fixture.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_curate.py -v`
Expected: FAIL

- [ ] **Step 3: Refactor `curate.py`**

- `curate_activities(config: TargetClassConfig, ...)` — derive raw paths from `config.raw_activities_filename` / `config.raw_targets_filename`.
- Rename the merged `kinase_group` column to a generic `subfamily`.
- Keep all the curation logic (median aggregation, noise flag, pActivity range, active label) unchanged — those are class-agnostic.
- Keep a backward-compat `curate_kinase_activities()` wrapper that calls `curate_activities(KINASE_CONFIG)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_curate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/target_affinity_ml/data/curate.py tests/unit/test_curate.py
git -c commit.gpgsign=false commit -m "Refactor curate.py: derive paths/columns from TargetClassConfig"
```

---

## Task 3: Add `data_dir` parameter to feature loaders (L3)

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/features/__init__.py`
- Modify: `target-affinity-ml/tests/unit/test_features.py`

**Context:** `load_morgan_fingerprints`, `load_rdkit_descriptors`, `load_esm2_embeddings` use a global `PROCESSED_DIR = Path("data/processed")` relative to cwd. The GPCR application repo needs to point them at its own data directory without `os.chdir`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_features.py` a test that calls `load_morgan_fingerprints(version="v1", data_dir=<tmp_path>)` and verifies it reads from the supplied directory, not cwd.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_features.py -k data_dir -v`
Expected: FAIL — `data_dir` is an unexpected keyword argument

- [ ] **Step 3: Add `data_dir` parameter**

Modify all three loaders in `features/__init__.py`:
```python
def load_morgan_fingerprints(version="v1", data_dir=None):
    base = Path(data_dir) if data_dir is not None else Path("data/processed")
    features_dir = base / version / "features"
    ...
```
Same pattern for `load_rdkit_descriptors` and `load_esm2_embeddings`. Default `None` preserves the existing relative-path behavior (kinase repo unaffected).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_features.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/target_affinity_ml/features/__init__.py tests/unit/test_features.py
git -c commit.gpgsign=false commit -m "Add data_dir parameter to feature loaders (L3)"
```

---

## Task 4: Deep-model integration test (L4) + validation cosmetic fix (L5)

**Files:**
- Create: `target-affinity-ml/tests/integration/test_deep_model_smoke.py`
- Modify: `kinase-affinity-baselines/scripts/rerun_kinase_v1.py` (fix L5 NaN-truthiness bug)

**Context:** L4 — the Task 10 integration test only covered RF; the deep-trainer dispatch bug slipped through. L5 — the rerun script's "60/105 failed" false warning from NaN truthiness.

- [ ] **Step 1: Write the deep-model smoke test**

Create `tests/integration/test_deep_model_smoke.py`. Mark `@pytest.mark.slow`. Skip if torch unavailable. The test calls `deep_train_and_evaluate` on ESM-FP MLP for a small synthetic or sampled dataset and asserts it returns a metrics dict with `test_rmse`. This exercises the deep-trainer code path that Plan 1 missed.

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/integration/test_deep_model_smoke.py -v -m slow`
Expected: PASS or SKIP (if torch not installed locally)

- [ ] **Step 3: Fix the L5 cosmetic bug in `rerun_kinase_v1.py`**

In `kinase-affinity-baselines/scripts/rerun_kinase_v1.py`, the final-summary error count uses `if r.get("error")` which treats NaN as truthy. Change to:
```python
n_errors = sum(
    1 for r in rows
    if r.get("error") is not None and str(r.get("error")).strip() not in ("", "nan")
)
```

- [ ] **Step 4: Commit (two repos)**

```bash
cd target-affinity-ml
git add tests/integration/test_deep_model_smoke.py
git -c commit.gpgsign=false commit -m "Add deep-model integration smoke test (L4)"

cd ../mlproject
git add scripts/rerun_kinase_v1.py
git -c commit.gpgsign=false commit -m "Fix NaN-truthiness false failure count in rerun script (L5)"
git push origin phase1-multi-class-expansion
```

---

## Task 5: Tag library v1.1.0

**Files:**
- Modify: `target-affinity-ml/pyproject.toml` (version → 1.1.0)
- Modify: `target-affinity-ml/CHANGELOG.md`

- [ ] **Step 1: Update version + CHANGELOG**

Bump `version = "1.1.0"` in pyproject.toml. Add a `[1.1.0]` CHANGELOG section documenting: TargetClassConfig abstraction, class-agnostic chembl_fetcher + curate, data_dir param on loaders, deep-model integration test.

- [ ] **Step 2: Run the full test suite**

Run: `cd target-affinity-ml && python -m pytest tests/ -v --ignore=tests/integration`
Expected: all unit tests pass (previous 46 + new tests from Tasks 1-4)

- [ ] **Step 3: Run ruff**

Run: `ruff check src/ tests/`
Expected: All checks passed

- [ ] **Step 4: Commit, tag, push**

```bash
cd target-affinity-ml
git add pyproject.toml CHANGELOG.md
git -c commit.gpgsign=false commit -m "Release v1.1.0: class-agnostic data pipeline"
git tag -a v1.1.0 -m "target-affinity-ml v1.1.0 — class-agnostic data pipeline

TargetClassConfig abstraction enables kinases, GPCRs, and future target
classes to share the data ingestion + curation pipeline. Backward
compatible: kinase code paths preserved via KINASE_CONFIG wrappers."
git push origin main
git push origin v1.1.0
```

---

# PART B — GPCR application repository

## Task 6: Create `gpcr-aminergic-benchmarks` repo skeleton

**Files:** new repo at `/Users/joshuaabbott/gpcr-aminergic-benchmarks/`

**Prerequisite:** User creates an empty GitHub repo `https://github.com/jmabbott40/gpcr-aminergic-benchmarks`. Pause and confirm before the push step if it doesn't exist.

- [ ] **Step 1: Clone/init the repo**

```bash
cd /Users/joshuaabbott
git clone https://github.com/jmabbott40/gpcr-aminergic-benchmarks.git || \
  (mkdir gpcr-aminergic-benchmarks && cd gpcr-aminergic-benchmarks && git init)
```

- [ ] **Step 2: Create `pyproject.toml`**

Package `gpcr-aminergic-benchmarks`, Python ≥3.11, depends on
`target-affinity-ml @ git+https://github.com/jmabbott40/target-affinity-ml.git@v1.1.0`.
Include `dev` (pytest, ruff) and `deep` (torch, torch-geometric, fair-esm) extras. `src/` layout, ruff + pytest config (mirror the library's pyproject).

- [ ] **Step 3: Create README.md, LICENSE (MIT), .gitignore, CHANGELOG.md**

README links to the library and to `kinase-affinity-baselines`. `.gitignore` excludes `data/`, `results/models/`, `*.npz`, `*.parquet`, `*.pt`.

- [ ] **Step 4: Create the package skeleton**

```
src/gpcr_aminergic_benchmarks/__init__.py   # __version__ = "1.0.0"
```

- [ ] **Step 5: Move the aminergic target list into the repo**

Copy `kinase-affinity-baselines/scripts/aminergic_audit/target_lists.py` →
`gpcr-aminergic-benchmarks/src/gpcr_aminergic_benchmarks/target_lists.py`.
Update imports (`scripts.aminergic_audit.target_lists` → `gpcr_aminergic_benchmarks.target_lists`).

- [ ] **Step 6: Create the aminergic `TargetClassConfig` instance**

Create `src/gpcr_aminergic_benchmarks/target_class.py`:
```python
"""The aminergic GPCR TargetClassConfig instance."""
from target_affinity_ml.data.target_class_config import TargetClassConfig
from gpcr_aminergic_benchmarks.target_lists import (
    get_all_gene_symbols, get_gene_to_family,
)

# Built at runtime after ChEMBL ID resolution (Task 7 fills explicit_target_ids
# and subfamily_map from the resolved gene→ChEMBL mapping).
def build_aminergic_config(resolved_ids: dict[str, str]) -> TargetClassConfig:
    """Construct the aminergic config from resolved gene→ChEMBL IDs."""
    gene_to_family = get_gene_to_family()
    subfamily_map = {
        chembl_id: gene_to_family[gene]
        for gene, chembl_id in resolved_ids.items()
    }
    return TargetClassConfig(
        class_name="gpcr_aminergic",
        raw_filename_stem="chembl_gpcr_aminergic",
        explicit_target_ids=list(resolved_ids.values()),
        subfamily_map=subfamily_map,
    )
```

- [ ] **Step 7: Install + verify**

```bash
cd gpcr-aminergic-benchmarks
pip install -e .
python -c "import gpcr_aminergic_benchmarks; print(gpcr_aminergic_benchmarks.__version__)"
```
Expected: prints `1.0.0`

- [ ] **Step 8: Commit + push**

```bash
git add -A
git -c commit.gpgsign=false commit -m "Initialize gpcr-aminergic-benchmarks repo skeleton"
git branch -M main
git push -u origin main
```

---

# PART C — GPCR data pipeline

## Task 7: GPCR data ingestion

**Files:**
- Create: `gpcr-aminergic-benchmarks/scripts/fetch_gpcr_data.py`
- Output: `data/raw/chembl_gpcr_aminergic_{activities,targets}.parquet`

- [ ] **Step 1: Write `fetch_gpcr_data.py`**

The script:
1. Resolves aminergic gene symbols → ChEMBL IDs via `target_lists.resolve_chembl_ids()`
2. Builds the aminergic `TargetClassConfig` via `build_aminergic_config()`
3. Calls `target_affinity_ml.data.chembl_fetcher.fetch_target_class(config)` to fetch IC50/Ki/Kd binding activities for the 30 aminergic targets (the same inclusion criteria the audit used: `assay_type="B"`, `confidence_score>=7`, `standard_relation="="`, `standard_units="nM"`, pChEMBL present)
4. Saves raw activities + targets parquet files

- [ ] **Step 2: Run the fetch (live ChEMBL API, ~20-40 min)**

Run: `python scripts/fetch_gpcr_data.py`
Expected: `data/raw/chembl_gpcr_aminergic_activities.parquet` written; console reports record count (~50-90K binding records expected for 30 aminergic targets based on the audit's per-target counts).

If the fetch fails or is rate-limited, surface as DONE_WITH_CONCERNS — do not fake data.

- [ ] **Step 3: Commit the script (not the data — data/ is gitignored)**

```bash
git add scripts/fetch_gpcr_data.py
git -c commit.gpgsign=false commit -m "Add GPCR data ingestion script"
```

---

## Task 8: GPCR curation + dataset card

**Files:**
- Create: `gpcr-aminergic-benchmarks/scripts/curate_gpcr_data.py`
- Create: `gpcr-aminergic-benchmarks/configs/dataset_aminergic_v1.yaml`
- Create: `gpcr-aminergic-benchmarks/docs/data_card.md`
- Output: `data/processed/v1/curated_activities.parquet`, `curation_stats.json`

- [ ] **Step 1: Create `configs/dataset_aminergic_v1.yaml`**

Mirror the kinase `dataset_v1.yaml` curation parameters: pActivity range [3.0, 12.0], MW 100-900, ≤100 heavy atoms, median aggregation, noise flag (std>1.0, n≥3), active threshold pActivity≥6.0.

- [ ] **Step 2: Write `curate_gpcr_data.py`**

Calls `target_affinity_ml.data.curate.curate_activities(aminergic_config, config_yaml)`. Produces curated parquet + curation_stats.json.

- [ ] **Step 3: Run curation**

Run: `python scripts/curate_gpcr_data.py`
Expected: `data/processed/v1/curated_activities.parquet` written; curation_stats.json reports compounds, targets (~30), records retained.

- [ ] **Step 4: Write `docs/data_card.md`**

Document the aminergic dataset: source (ChEMBL), 30 targets across 5 families, inclusion criteria, curation decisions, known limitations. Mirror the kinase `data_card.md` structure.

- [ ] **Step 5: Commit**

```bash
git add scripts/curate_gpcr_data.py configs/dataset_aminergic_v1.yaml docs/data_card.md
git -c commit.gpgsign=false commit -m "Add GPCR curation pipeline + dataset card"
```

---

## Task 9: GPCR splits (random, scaffold, target)

**Files:**
- Create: `gpcr-aminergic-benchmarks/scripts/build_gpcr_splits.py`
- Output: `data/processed/v1/splits/{random,scaffold,target}_split.json`

**Context:** The library's `splits.py` provides `random_split`, `scaffold_split`, `target_split`. The target split holds out entire targets/subfamilies. For aminergic GPCRs, the target split should hold out entire **receptor families** (dopamine, serotonin, etc.) using `config.subfamily_map` — verify the library's `target_split` accepts a subfamily grouping rather than hardcoding kinase logic.

- [ ] **Step 1: Inspect the library `target_split` signature**

Run: `python -c "import inspect; from target_affinity_ml.data.splits import target_split; print(inspect.signature(target_split))"`
If `target_split` requires a kinase-specific grouping argument, note it — Step 2 adapts accordingly.

- [ ] **Step 2: Write `build_gpcr_splits.py`**

Generate all three splits (seed=42), saving index JSONs. For the target split, pass the aminergic family grouping (from `config.subfamily_map`) so entire families are held out. 80/10/10 for random; Murcko scaffold groups for scaffold.

- [ ] **Step 3: Run + verify split integrity**

Run: `python scripts/build_gpcr_splits.py`
Verify: no index overlap between train/val/test; scaffold split has no scaffold leakage; target split holds out complete families.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_gpcr_splits.py
git -c commit.gpgsign=false commit -m "Add GPCR train/val/test split generation"
```

---

# PART D — GPCR feature engineering

## Task 10: GPCR molecular features (Morgan FP + RDKit descriptors)

**Files:**
- Create: `gpcr-aminergic-benchmarks/scripts/build_gpcr_features.py`
- Output: `data/processed/v1/features/{morgan_fp,rdkit_descriptors}.npz`, `smiles_index.json`

- [ ] **Step 1: Write `build_gpcr_features.py`**

Calls `target_affinity_ml.features.compute_and_cache_features` (the restored loader from Plan 1) on the curated GPCR dataset. Produces Morgan FP (2048-bit, radius 2) + RDKit 2D descriptors, keyed by SMILES order.

- [ ] **Step 2: Run feature generation**

Run: `python scripts/build_gpcr_features.py`
Expected: `morgan_fp.npz`, `rdkit_descriptors.npz`, `smiles_index.json` written. Console reports compound count.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_gpcr_features.py
git -c commit.gpgsign=false commit -m "Add GPCR molecular feature generation"
```

---

## Task 11: GPCR protein sequences + ESM-2 embeddings

**Files:**
- Create: `gpcr-aminergic-benchmarks/scripts/build_gpcr_esm.py`
- Output: `data/processed/v1/protein_sequences.json`, `features/esm2_embeddings.npz`, `features/target_index.json`

**Context:** Deep models (ESM-FP MLP, Fusion) need ESM-2 embeddings. The library's `protein_sequences.py` fetches UniProt sequences; `protein_embeddings.py` computes ESM-2. Aminergic GPCRs are membrane proteins — the ESM-2 step is identical mechanically (mean-pool over residues), though the spec Section 5.4 flags that membrane proteins are harder for pLMs (relevant to Plan 3's RNS analysis, not here).

- [ ] **Step 1: Write `build_gpcr_esm.py`**

1. Fetch UniProt sequences for the 30 aminergic targets via `target_affinity_ml.data.protein_sequences` (resolve ChEMBL target → UniProt accession → sequence).
2. Compute ESM-2 (`esm2_t33_650M_UR50D`) mean-pooled embeddings via `target_affinity_ml.features.protein_embeddings`.
3. Save `protein_sequences.json`, `esm2_embeddings.npz`, `target_index.json`.

- [ ] **Step 2: Run on a GPU machine (the AWS instance)**

ESM-2 inference needs a GPU. Run on AWS:
```bash
# On AWS, after syncing the GPCR repo + curated data:
python scripts/build_gpcr_esm.py
```
Expected: ~30 sequences embedded in a few minutes on one A10G. `esm2_embeddings.npz` shape ≈ (30, 1280).

- [ ] **Step 3: Commit**

```bash
git add scripts/build_gpcr_esm.py
git -c commit.gpgsign=false commit -m "Add GPCR protein sequence + ESM-2 embedding pipeline"
```

---

# PART E — GPCR benchmark

## Task 12: GPCR benchmark configs + run script

**Files:**
- Create: `gpcr-aminergic-benchmarks/configs/` — 7 model configs
- Create: `gpcr-aminergic-benchmarks/scripts/run_gpcr_benchmark.py`

- [ ] **Step 1: Copy the 7 model configs**

Copy the library's default configs (rf_baseline.yaml, xgb_baseline.yaml, elasticnet_baseline.yaml, mlp_baseline.yaml, esm_fp_mlp.yaml, gnn.yaml, fusion.yaml) into `gpcr-aminergic-benchmarks/configs/`. These define the same hyperparameters used for kinases — critical for a fair cross-class comparison (spec Section 6.2 guardrail).

- [ ] **Step 2: Write `run_gpcr_benchmark.py`**

Adapt `kinase-affinity-baselines/scripts/rerun_kinase_v1.py`:
- Same 7 models × 3 splits × 5 seeds = 105 runs
- Same deep/baseline trainer dispatch (the fix from Plan 1)
- Same `--resume` flag
- Point `data_dir` at the GPCR repo's `data/processed/v1/` (using the L3 `data_dir` parameter)
- Output to `results/gpcr_v1_benchmark/all_seeds_metrics.csv`

- [ ] **Step 3: Smoke test one run locally**

Run RF random seed=42 to confirm the pipeline works end-to-end before committing to the full benchmark.

- [ ] **Step 4: Commit**

```bash
git add configs/ scripts/run_gpcr_benchmark.py
git -c commit.gpgsign=false commit -m "Add GPCR benchmark configs + run script"
git push origin main
```

---

## Task 13: Run the 105-run GPCR benchmark on AWS + aggregate

**Files:**
- Output: `results/gpcr_v1_benchmark/all_seeds_metrics.csv`
- Output: `results/gpcr_v1_benchmark/multi_seed_aggregated.csv`

- [ ] **Step 1: Sync GPCR repo + data to AWS**

Clone `gpcr-aminergic-benchmarks` on the AWS instance, `pip install -e .[deep]`, and ensure `data/processed/v1/` is populated (run Tasks 7-11's scripts on AWS, or scp the processed data up).

- [ ] **Step 2: Launch the benchmark**

```bash
nohup python scripts/run_gpcr_benchmark.py > gpcr_rerun.log 2>&1 &
```
Expected runtime: ~10-16 hours (aminergic dataset is smaller than kinase but the 45 deep runs dominate). The AWS instance has 4× A10G.

- [ ] **Step 3: Monitor to completion**

Verify 105/105 runs succeed. Watch for the deep-model dispatch working correctly (the Plan 1 bug is already fixed in the adapted script).

- [ ] **Step 4: Aggregate multi-seed results**

Run the library's multi-seed analysis (`target_affinity_ml.evaluation.multi_seed_analysis`) to produce `multi_seed_aggregated.csv` (mean ± SD across 5 seeds per model × split).

- [ ] **Step 5: Commit results**

```bash
git add results/gpcr_v1_benchmark/all_seeds_metrics.csv \
        results/gpcr_v1_benchmark/multi_seed_aggregated.csv
git -c commit.gpgsign=false commit -m "Add GPCR 105-run benchmark results"
git push origin main
```

---

# PART F — Wrap-up

## Task 14: Plan 2 completion summary + Plan 3 handoff

**Files:**
- Create: `kinase-affinity-baselines/docs/superpowers/plans/2026-XX-XX-plan2-completion-summary.md`

- [ ] **Step 1: Write the completion summary**

Document: library v1.1.0 release, GPCR dataset stats (targets, compounds, records, scaffolds, activity-type breakdown), benchmark outcome (per-model RMSE/R² across 3 splits), any issues encountered, and Plan 3 readiness. Include a sanity comparison of GPCR per-model RMSEs against the kinase rerun (informal — full cross-class comparison is Plan 3).

- [ ] **Step 2: Commit + push**

```bash
cd kinase-affinity-baselines
git add docs/superpowers/plans/2026-XX-XX-plan2-completion-summary.md
git -c commit.gpgsign=false commit -m "Plan 2 completion summary"
git push origin phase1-multi-class-expansion
```

---

## Plan 2 verification checklist

- [ ] Library v1.1.0 tagged + pushed; CI green
- [ ] `TargetClassConfig` abstraction; kinase code paths still work (backward compat)
- [ ] `gpcr-aminergic-benchmarks` repo created + pushed
- [ ] GPCR curated dataset built (~30 targets, binding-only)
- [ ] Three GPCR splits generated, integrity verified
- [ ] GPCR features cached (Morgan FP, RDKit descriptors, ESM-2 embeddings)
- [ ] GPCR benchmark: 105/105 runs complete
- [ ] Multi-seed aggregation produced
- [ ] Plan 2 completion summary written

## Estimated effort

| Part | Tasks | Time |
|------|-------|------|
| A (library refactor → v1.1.0) | 1-5 | ~1 day engineering |
| B (GPCR repo skeleton) | 6 | ~2 hours |
| C (GPCR data pipeline) | 7-9 | ~1 day (incl. ~40 min ChEMBL fetch) |
| D (GPCR features) | 10-11 | ~0.5 day (+ GPU time for ESM-2) |
| E (GPCR benchmark) | 12-13 | ~10-16h AWS compute |
| F (wrap-up) | 14 | ~2 hours |
| **Total** | **14 tasks** | **~3-4 days + compute** |

## Notes for plan execution

- **Tasks 1-5 (library refactor)** must complete and v1.1.0 must be tagged before Task 6+, because the GPCR repo pins to `target-affinity-ml==1.1.0`.
- **Backward compatibility is critical**: the kinase repo depends on the library. Every refactor task keeps a `KINASE_CONFIG`-based wrapper. Run the kinase integration test after Task 5 to confirm nothing broke.
- **Tasks 7-11 (GPCR data + features)** are sequential — each consumes the previous output.
- **Task 13** is the long compute; budget ~10-16h on AWS and one debugging cycle.
- The AWS instance from Plan 1 can be reused if still running.
