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

**Current structure (verified):** `chembl_fetcher.py` has TWO separate functions plus an orchestrator:
- `fetch_kinase_targets()` — discovers kinase targets via GO terms, returns target metadata
- `fetch_bioactivities(...)` — fetches IC50/Ki/Kd activities for a set of targets
- `main()` — CLI orchestrator that calls both and saves parquet files

The refactor:
- Keep the existing `KINASE_GO_TERMS` set; add a module-level `KINASE_CONFIG = TargetClassConfig(class_name="kinase", go_terms=KINASE_GO_TERMS, name_keywords=["kinase"], raw_filename_stem="chembl_kinase")` instance.
- **Add a NEW orchestrator function** `fetch_target_class(config: TargetClassConfig) -> tuple[pd.DataFrame, pd.DataFrame]` that returns `(activities_df, targets_df)`. Internally:
  - If `config.uses_explicit_target_list`: skip GO discovery; use `config.explicit_target_ids` directly as the target set.
  - Else (`config.go_terms` set): run the existing GO-based discovery (the current `fetch_kinase_targets` logic, generalized to use `config.go_terms` instead of the module constant).
  - Then call `fetch_bioactivities` for the resolved target set.
- Replace `_classify_kinase` with a generic `_classify_subfamily(target_id, config)` that looks up `config.subfamily_map` (returns `"unknown"` if not present).
- Keep `fetch_kinase_targets()` and `fetch_bioactivities()` working unchanged — they stay as the underlying functions. **Do not add any separate `fetch_kinase_data` wrapper.** The only new public symbols are `fetch_target_class` (the orchestrator) and `KINASE_CONFIG` (the constant). The existing `main()` CLI may optionally be rewired to call `fetch_target_class(KINASE_CONFIG)` internally, but its external behavior must stay identical.

**Important:** Preserve the existing kinase code paths exactly — the kinase repo depends on `fetch_kinase_targets` and `fetch_bioactivities`. Do not change their signatures; only ADD the new `fetch_target_class` orchestrator and the `KINASE_CONFIG` constant. Task 7's GPCR script will call `fetch_target_class(aminergic_config)`.

- [ ] **Step 6: Write mocked test for `chembl_fetcher.py`**

Create `tests/unit/test_chembl_fetcher.py` with tests that mock the ChEMBL client and verify:
- `fetch_target_class` with an explicit-ID config queries activities for exactly those IDs (no GO discovery)
- `fetch_target_class` with a GO-term config runs the GO discovery path
- `fetch_kinase_targets` and `fetch_bioactivities` still work unchanged (regression guard — they keep their original signatures)

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

## Task 2: Extract a class-agnostic `curate_activities` function from `curate.py`

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/data/curate.py`
- Create: `target-affinity-ml/tests/unit/test_curate.py`

**Context (verified):** `curate.py` has NO reusable `curate_activities` function — the entire pipeline lives inside `main()` (curate.py:257), which: (1) hardcodes `chembl_kinase_activities.parquet` / `chembl_kinase_targets.parquet` paths, (2) hardcodes the `kinase_group` column in the targets merge (curate.py:283-288), (3) wraps everything in an argparse CLI. The *step* functions it calls — `standardize_dataframe`, `convert_to_pactivity`, `handle_duplicates`, `apply_quality_filters`, `add_classification_labels` — are already class-agnostic.

**This task EXTRACTS** a new reusable `curate_activities(config, dataset_config)` function out of `main()`, then rewires `main()` to call it.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_curate.py`. Test that the NEW `curate_activities(config: TargetClassConfig, dataset_config: dict, raw_dir: Path)` function:
- reads raw activities from `raw_dir / config.raw_activities_filename`
- merges target metadata from `raw_dir / config.raw_targets_filename` if present, producing a generic `subfamily` column (not `kinase_group`)
- returns a curated DataFrame
Use a small synthetic raw-activities parquet fixture written to a `tmp_path`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_curate.py -v`
Expected: FAIL — `cannot import name 'curate_activities'`

- [ ] **Step 3: Extract `curate_activities` from `main()`**

In `curate.py`:
- Add `def curate_activities(config: TargetClassConfig, dataset_config: dict, raw_dir: Path = Path("data/raw")) -> pd.DataFrame:` containing the body of `main()` Steps 1-6 (load raw → standardize → pActivity → duplicates → quality filters → classification labels). Returns the curated DataFrame. `raw_dir` defaults to `Path("data/raw")` (the conventional location); callers may override.
  - Raw paths derive from `raw_dir / config.raw_activities_filename` and `raw_dir / config.raw_targets_filename`.
  - **The `subfamily` column is always populated inside `curate_activities`, by one of two paths:**
    1. **GO-based classes** (kinase): the targets-file merge selects `["target_chembl_id", "pref_name", "gene_symbol"]`; if `"kinase_group"` exists in `targets_df.columns`, rename it to `subfamily`.
    2. **Explicit-target-list classes** (GPCR aminergic, `config.uses_explicit_target_list` is True): populate `subfamily` by mapping each row's `target_chembl_id` through `config.subfamily_map`.
  This way **both classes get a `subfamily` column from this one function** — no caller has to attach it afterward.
- Rewire `main()` to: parse args → load `dataset_config` YAML → call `curate_activities(KINASE_CONFIG, dataset_config)` → run the split step → save outputs. `main()` keeps the CLI and the kinase defaults.
- Keep all step-function logic (median aggregation, noise flag, pActivity range, active label) byte-identical — class-agnostic already.

- [ ] **Step 4: Run test + verify kinase `main()` still works**

Run: `python -m pytest tests/unit/test_curate.py -v`
Expected: PASS

Verify `main()` is unbroken — its argparse + kinase-default behavior must be unchanged (the kinase repo's `python -m kinase_affinity.data.curate` path still works via the re-export shim).

- [ ] **Step 5: Commit**

```bash
git add src/target_affinity_ml/data/curate.py tests/unit/test_curate.py
git -c commit.gpgsign=false commit -m "Extract class-agnostic curate_activities function from curate.main()"
```

---

## Task 3: Add `data_dir` parameter to feature loaders AND `compute_and_cache_features` (L3)

**Files:**
- Modify: `target-affinity-ml/src/target_affinity_ml/features/__init__.py`
- Modify: `target-affinity-ml/tests/unit/test_features.py`

**Context:** `load_morgan_fingerprints`, `load_rdkit_descriptors`, `load_esm2_embeddings`, **and `compute_and_cache_features`** all use a module-global `PROCESSED_DIR = Path("data/processed")` relative to cwd. The GPCR application repo needs to point them at its own data directory without `os.chdir`. All four functions need the `data_dir` parameter — Task 10 calls `compute_and_cache_features` and would hit the same relative-path problem otherwise.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_features.py` tests that:
- `load_morgan_fingerprints(version="v1", data_dir=<tmp_path>)` reads from the supplied directory, not cwd
- `compute_and_cache_features(config_path=..., data_dir=<tmp_path>)` writes outputs under the supplied directory

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_features.py -k data_dir -v`
Expected: FAIL — `data_dir` is an unexpected keyword argument

- [ ] **Step 3: Add `data_dir` parameter to all four functions**

Modify `features/__init__.py`. For the three loaders:
```python
def load_morgan_fingerprints(version="v1", data_dir=None):
    base = Path(data_dir) if data_dir is not None else Path("data/processed")
    features_dir = base / version / "features"
    ...
```
Same for `load_rdkit_descriptors`, `load_esm2_embeddings`. For `compute_and_cache_features`, add `data_dir=None` and replace the module-global `PROCESSED_DIR` usage with `Path(data_dir) if data_dir else PROCESSED_DIR`. Default `None` preserves existing relative-path behavior (kinase repo unaffected).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_features.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/target_affinity_ml/features/__init__.py tests/unit/test_features.py
git -c commit.gpgsign=false commit -m "Add data_dir parameter to feature loaders + compute_and_cache_features (L3)"
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

- [ ] **Step 4: Fix the dead-code statement in `splits.py`**

The reviewer noted `splits.py:287` has a dead `n = len(df)` assignment whose result is never used (it was flagged by ruff F841 during Plan 1 but the assignment itself remains). In `target-affinity-ml/src/target_affinity_ml/data/splits.py`, locate the unused `n = len(df)` near the target-split logging block and either remove it or inline it into the `logger.info` call that needs the count. Run `ruff check src/target_affinity_ml/data/splits.py` to confirm clean.

- [ ] **Step 5: Commit (two repos)**

```bash
cd target-affinity-ml
git add tests/integration/test_deep_model_smoke.py src/target_affinity_ml/data/splits.py
git -c commit.gpgsign=false commit -m "Add deep-model integration smoke test (L4); remove dead code in splits.py"

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
2. **Persists the resolved mapping** to `data/processed/v1/resolved_target_ids.json` (`{gene_symbol: chembl_id}`) so Task 8's curation script and Task 11's ESM script can rebuild the aminergic `TargetClassConfig` without re-querying ChEMBL.
3. Builds the aminergic `TargetClassConfig` via `build_aminergic_config(resolved_ids)`
4. Calls `target_affinity_ml.data.chembl_fetcher.fetch_target_class(config)` to fetch IC50/Ki/Kd binding activities for the 30 aminergic targets (the same inclusion criteria the audit used: `assay_type="B"`, `confidence_score>=7`, `standard_relation="="`, `standard_units="nM"`, pChEMBL present)
5. Saves raw activities + targets parquet files to `data/raw/` (using `config.raw_activities_filename` / `config.raw_targets_filename`)

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

The script:
1. Loads `configs/dataset_aminergic_v1.yaml` as `dataset_config`.
2. Builds the aminergic `TargetClassConfig` (via `build_aminergic_config` — needs the resolved gene→ChEMBL mapping from Task 7's fetch step; persist that mapping in Task 7 so this script can reload it).
3. Calls `target_affinity_ml.data.curate.curate_activities(config=aminergic_config, dataset_config=dataset_config, raw_dir=Path("data/raw"))` — the function extracted in Task 2. Because `aminergic_config.uses_explicit_target_list` is True, `curate_activities` itself populates the `subfamily` column from `config.subfamily_map` (per Task 2 Step 3) — the GPCR script does NOT need to attach it separately.
4. Writes `data/processed/v1/curated_activities.parquet` + `curation_stats.json`.

- [ ] **Step 3: Run curation**

Run: `python scripts/curate_gpcr_data.py`
Expected: `data/processed/v1/curated_activities.parquet` written; curation_stats.json reports compounds, targets (~30), records retained. Every row should have a non-null `subfamily` (one of: dopamine, serotonin, adrenergic, histamine, muscarinic).

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

**Context (verified):** The library's `splits.py` provides `random_split`, `scaffold_split`, `target_split`. **`target_split` (splits.py:231) holds out entire individual targets by `target_col` — it does NOT take a subfamily-grouping argument.** This is already class-agnostic: it holds out targets regardless of class.

**Design decision — GPCR target split uses individual-target holdout, identical to the kinase protocol.** The kinase preprint's `target_split` used per-target holdout (that is what the code does). For a faithful cross-class comparison (spec Section 6.2 guardrail: "exact same model hyperparameters... any deviation documented"), the GPCR target split MUST use the identical `target_split` function — per-target holdout. The spec Section 4.5's "hold out entire families" language describes an *optional supplement*; the **primary** target split matches the kinase code exactly.

The `config.subfamily_map` (receptor family per target) is **metadata for Plan 3's per-family analysis**, not used by the Plan 2 target split. Plan 3 may add a leave-one-family-out supplement; Plan 2 does not.

- [ ] **Step 1: Confirm the kinase target split behavior**

Run: `python -c "import inspect; from target_affinity_ml.data.splits import target_split; print(inspect.signature(target_split))"`
Confirm it splits by individual target (parameter `target_col`, no group argument). This is the function the GPCR target split will use unchanged.

- [ ] **Step 2: Write `build_gpcr_splits.py`**

Generate all three splits with `seed=42`, saving index JSONs to `data/processed/v1/splits/`:
- `random_split` — 80/10/10, stratified by target
- `scaffold_split` — Murcko scaffold groups, no scaffold leakage
- `target_split` — individual-target holdout via the library's `target_split(df, target_col="target_chembl_id", seed=42)` — **the exact same call the kinase pipeline uses**

Note: with ~30 aminergic targets, the target split's test set is ~3 targets (10%). This is small but is the faithful analog of the kinase protocol. Document this in the script's docstring and in the Task 14 completion summary.

- [ ] **Step 3: Run + verify split integrity**

Run: `python scripts/build_gpcr_splits.py`
Verify: no index overlap between train/val/test; scaffold split has no scaffold leakage (no scaffold appears in two splits); target split's test targets do not appear in train/val.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_gpcr_splits.py
git -c commit.gpgsign=false commit -m "Add GPCR train/val/test split generation (per-target holdout, matching kinase protocol)"
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

1. Fetch UniProt sequences for the 30 aminergic targets. `target_affinity_ml.data.protein_sequences` exposes `fetch_uniprot_accessions`, `fetch_sequences_from_uniprot`, and the orchestrator `build_protein_sequence_cache` — call `build_protein_sequence_cache` with the aminergic target IDs to produce `protein_sequences.json`.
2. Compute ESM-2 (`esm2_t33_650M_UR50D`) mean-pooled embeddings via `target_affinity_ml.features.protein_embeddings.compute_esm2_embeddings`.
3. Save `protein_sequences.json`, `esm2_embeddings.npz`, `target_index.json` under `data/processed/v1/`.

**Verify the function signatures first** with `python -c "import inspect; from target_affinity_ml.data.protein_sequences import build_protein_sequence_cache; print(inspect.signature(build_protein_sequence_cache))"` — adapt the call if the parameters differ from this description.

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

Run the library's multi-seed analysis function `target_affinity_ml.evaluation.multi_seed_analysis.run_full_multi_seed_analysis` to produce `multi_seed_aggregated.csv` (mean ± SD across 5 seeds per model × split). Verify its signature first with `inspect.signature` — it may expect a specific input directory layout; point it at `results/gpcr_v1_benchmark/`.

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
