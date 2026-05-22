# Plan 2 Execution Handoff

**Purpose:** Everything a fresh Claude Code session needs to execute Plan 2 (GPCR Aminergic Data Pipeline + Benchmark) with subagent-driven development. Plan 2 is written, reviewed, approved, and pushed. This document is the clean-context bridge.

---

## START HERE — paste this into a fresh session

```
Execute Plan 2 with subagent-driven development.

Plan: /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-04-30-plan2-gpcr-data-pipeline-benchmark.md
Spec: /Users/joshuaabbott/mlproject/docs/superpowers/specs/2026-04-17-gpcr-aminergic-phase1-design.md
Handoff context: /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-04-30-plan2-execution-handoff.md

Read the handoff context first, run the pre-flight check, then invoke the
superpowers:subagent-driven-development skill to execute the 14 tasks.
```

That's it. The new session reads this file, runs the pre-flight script below, and proceeds.

---

## Pre-flight environment check

Before dispatching any subagent, run this to confirm the environment. Save it as `/tmp/plan2_preflight.sh` and execute it.

```bash
#!/usr/bin/env bash
# Plan 2 pre-flight environment check
set -u
echo "=========================================="
echo "  Plan 2 Pre-Flight Check"
echo "=========================================="
PASS=0; FAIL=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ✅ $1"; PASS=$((PASS+1)); else echo "  ❌ $1"; FAIL=$((FAIL+1)); fi; }

echo
echo "--- Local repositories ---"
check "kinase repo exists"        "test -d /Users/joshuaabbott/mlproject/.git"
check "library repo exists"       "test -d /Users/joshuaabbott/target-affinity-ml/.git"
check "kinase on phase1 branch"   "cd /Users/joshuaabbott/mlproject && git branch --show-current | grep -q phase1-multi-class-expansion"
check "library on main branch"    "cd /Users/joshuaabbott/target-affinity-ml && git branch --show-current | grep -q main"

echo
echo "--- Plan & spec documents ---"
check "Plan 2 document present"   "test -f /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-04-30-plan2-gpcr-data-pipeline-benchmark.md"
check "Spec document present"     "test -f /Users/joshuaabbott/mlproject/docs/superpowers/specs/2026-04-17-gpcr-aminergic-phase1-design.md"
check "Plan 1 summary present"    "test -f /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-04-30-plan1-completion-summary.md"

echo
echo "--- Python environment (use the 3.11 conda env, NOT base) ---"
CONDA_PY=/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
check "kinase-affinity conda env exists" "test -x $CONDA_PY"
check "conda env is Python 3.11"  "$CONDA_PY --version 2>&1 | grep -q 'Python 3.11'"
check "target_affinity_ml importable" "$CONDA_PY -c 'import target_affinity_ml; print(target_affinity_ml.__version__)'"

echo
echo "--- Library v1.0.0 state ---"
check "library v1.0.0 tag exists" "cd /Users/joshuaabbott/target-affinity-ml && git tag -l | grep -q v1.0.0"
check "library working tree clean" "cd /Users/joshuaabbott/target-affinity-ml && test -z \"\$(git status --porcelain)\""

echo
echo "--- AWS instance (Task 11 ESM + Task 13 benchmark need GPUs) ---"
AWS_KEY=/Users/joshuaabbott/Downloads/jma_key.pem
AWS_HOST=ubuntu@ec2-3-17-4-165.us-east-2.compute.amazonaws.com
check "AWS SSH key present"        "test -f $AWS_KEY"
if ssh -i $AWS_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $AWS_HOST 'echo ok' >/dev/null 2>&1; then
    echo "  ✅ AWS instance reachable"
    PASS=$((PASS+1))
    ssh -i $AWS_KEY -o ConnectTimeout=10 $AWS_HOST 'nvidia-smi -L 2>/dev/null | head -1' 2>/dev/null | sed 's/^/     /'
else
    echo "  ⚠️  AWS instance NOT reachable — may be stopped. Restart it before Tasks 11/13."
    echo "     (Tasks 1-10, 12, 14 are local/CPU work and don't need AWS.)"
fi

echo
echo "=========================================="
echo "  Pre-flight: $PASS passed, $FAIL failed"
echo "=========================================="
if [ $FAIL -gt 0 ]; then
    echo "  Resolve failures before executing Plan 2."
    exit 1
fi
echo "  Environment ready. Proceed with subagent-driven-development."
```

---

## Project state at handoff (2026-04-30)

### What's done — Plan 1 (complete)

- `target-affinity-ml` **v1.0.0** published: https://github.com/jmabbott40/target-affinity-ml
- `kinase-affinity-baselines` branch `phase1-multi-class-expansion` pushed: https://github.com/jmabbott40/kinase-affinity-baselines/tree/phase1-multi-class-expansion
- Kinase benchmark re-validated (105/105 runs; Tier-A deterministic models reproduce bit-exactly)
- Aminergic data audit cleared: **OPTION_A** — 30/36 targets viable at ≥500 binding records
- Full details: `docs/superpowers/plans/2026-04-30-plan1-completion-summary.md`

### What Plan 2 does

14 tasks, 6 parts. Builds `target-affinity-ml` v1.1.0 (class-agnostic refactor) then the `gpcr-aminergic-benchmarks` application repo with the full GPCR benchmark. See the plan document for task-by-task detail.

---

## Critical context for the executing session

### Repositories & paths

| Repo | Local path | Branch | Remote |
|---|---|---|---|
| Kinase (application) | `/Users/joshuaabbott/mlproject` | `phase1-multi-class-expansion` | `kinase-affinity-baselines` |
| Library | `/Users/joshuaabbott/target-affinity-ml` | `main` | `target-affinity-ml` |
| GPCR (application) | `/Users/joshuaabbott/gpcr-aminergic-benchmarks` | — created in Task 6 | `gpcr-aminergic-benchmarks` |

### USER ACTION REQUIRED before Task 6

The executing session must **pause at Task 6** and ask the user to create an empty GitHub repo `https://github.com/jmabbott40/gpcr-aminergic-benchmarks` before the push step. (Same pattern as Plan 1 Task 2.)

### Python environment — IMPORTANT

There are **two** Python environments locally. Use the **right one**:

- ❌ **Base miniforge** (`/opt/homebrew/Caskroom/miniforge/base/bin/python`, Python 3.13) — has a **broken xgboost** (missing libomp). Do NOT use for anything that imports xgboost or runs the trainer.
- ✅ **`kinase-affinity` conda env** (`/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python`, Python 3.11) — this is the working env. Use it for all tests, imports, and runs.

When dispatching subagents, tell them explicitly to use the 3.11 conda env path for any Python execution.

### AWS instance

- SSH: `ssh -i "/Users/joshuaabbott/Downloads/jma_key.pem" ubuntu@ec2-3-17-4-165.us-east-2.compute.amazonaws.com`
- 4× NVIDIA A10G GPUs, miniforge installed, `kinase-affinity` conda env (Python 3.11) present
- Git remote on AWS uses **HTTPS**, not SSH
- The instance **may have been stopped** since Plan 1 — the pre-flight check reports reachability. If stopped, the user must restart it before Tasks 11 (ESM-2 embeddings, needs GPU) and 13 (benchmark, needs GPU). Tasks 1-10, 12, 14 are local/CPU and don't need AWS.
- Old data lives at `~/mlproject/data/processed/v1/` on the instance (kinase dataset, ESM embeddings) — useful reference, but Plan 2 builds fresh GPCR data.

### Git commit conventions

- Use `git -c commit.gpgsign=false commit ...` — GPG signing prompts otherwise block non-interactive commits.
- Commit trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- Never `git push` to `main` of the kinase repo — work stays on `phase1-multi-class-expansion`. The library repo pushes to its `main`.
- Don't commit `data/` or large binaries — both repos gitignore them.

### Backward-compatibility rule (Part A is delicate)

The kinase repo depends on the library. **Every Part A refactor task (1-5) must preserve kinase code paths.** The pattern: keep existing functions working, ADD new class-agnostic ones, provide `KINASE_CONFIG`-based wrappers/defaults. After Task 5, the integration test `tests/integration/test_kinase_reproducibility.py` must still pass — that's the regression guard.

### Subagent-driven workflow reminders

- Fresh subagent per task; pass the FULL task text (don't make subagents read the plan file).
- Two-stage review after each task: spec-compliance reviewer first, then code-quality reviewer (`superpowers:code-reviewer`).
- Use the 3.11 conda env in all subagent instructions.
- Tasks 1-5 are sequential (library refactor → v1.1.0 tag). Tasks 7-11 are sequential (each consumes the previous output). Task 6 (repo skeleton) gates Tasks 7+.
- Plan 1's lesson: the integration test only covered RF and missed a deep-model bug. Task 4 adds a deep-model smoke test — make sure it actually runs.

---

## Lessons from Plan 1 execution (avoid repeating)

| Plan 1 issue | How it bit us | Prevention in Plan 2 |
|---|---|---|
| Wildcard re-export shim dropped 4 loader functions | Trainer broke; caught only at integration-test time | Task 2/3 explicitly preserve + test every function |
| pyarrow was a transitive dep, not declared | Fresh AWS install failed | Already fixed in library v1.0.0 deps |
| Rerun script called baseline trainer for deep models | 45 wasted runs (~2 GPU-hours) | Plan 2's run script reuses the fixed dispatch |
| Reference data not on GitHub | Validation couldn't find files on AWS | Plan 2 doesn't need kinase references (that's Plan 3) |
| Integration test only covered RF | Deep-trainer bug slipped through | Task 4 adds deep-model smoke test |
| Config filename mismatch (rf_baseline vs random_forest_baseline) | All 105 runs failed instantly | Task 12 copies configs explicitly |

---

## Estimated effort

~3-4 days of engineering + ~10-16h AWS compute (Task 13). Tasks 1-12 + 14 are local/CPU; only Tasks 11 and 13 need AWS GPUs.

---

## When Plan 2 completes

Task 14 produces a Plan 2 completion summary. After that, Plan 3 (scaffold-diversity metrics + RNS-stratified ESM-2 analysis + cross-class comparison) is the final plan — it will need a fresh handoff of its own, and it's where the kinase reference data (Plan 1 limitation L2) must be permanently hosted.
