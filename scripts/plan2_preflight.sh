#!/usr/bin/env bash
# Plan 2 pre-flight environment check.
# Run this before executing Plan 2 with subagent-driven development.
# Usage: bash scripts/plan2_preflight.sh
set -u
echo "=========================================="
echo "  Plan 2 Pre-Flight Check"
echo "=========================================="
PASS=0; FAIL=0
check() {
    if eval "$2" >/dev/null 2>&1; then
        echo "  PASS  $1"; PASS=$((PASS+1))
    else
        echo "  FAIL  $1"; FAIL=$((FAIL+1))
    fi
}

echo
echo "--- Local repositories ---"
check "kinase repo exists"        "test -d /Users/joshuaabbott/mlproject/.git"
check "library repo exists"       "test -d /Users/joshuaabbott/target-affinity-ml/.git"
check "kinase on phase1 branch"   "cd /Users/joshuaabbott/mlproject && git branch --show-current | grep -q phase1-multi-class-expansion"
check "library on main branch"    "cd /Users/joshuaabbott/target-affinity-ml && git branch --show-current | grep -q main"

echo
echo "--- Plan & spec documents ---"
DOCS=/Users/joshuaabbott/mlproject/docs/superpowers
check "Plan 2 document present"   "test -f $DOCS/plans/2026-04-30-plan2-gpcr-data-pipeline-benchmark.md"
check "Spec document present"     "test -f $DOCS/specs/2026-04-17-gpcr-aminergic-phase1-design.md"
check "Plan 1 summary present"    "test -f $DOCS/plans/2026-04-30-plan1-completion-summary.md"
check "Handoff doc present"       "test -f $DOCS/plans/2026-04-30-plan2-execution-handoff.md"

echo
echo "--- Python environment (use the 3.11 conda env, NOT base) ---"
CONDA_PY=/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
check "kinase-affinity conda env exists"  "test -x $CONDA_PY"
check "conda env is Python 3.11"          "$CONDA_PY --version 2>&1 | grep -q 'Python 3.11'"
check "target_affinity_ml importable"     "$CONDA_PY -c 'import target_affinity_ml'"

echo
echo "--- Library v1.0.0 state ---"
check "library v1.0.0 tag exists"   "cd /Users/joshuaabbott/target-affinity-ml && git tag -l | grep -q v1.0.0"
check "library working tree clean"  "cd /Users/joshuaabbott/target-affinity-ml && test -z \"\$(git status --porcelain)\""

echo
echo "--- AWS instance (Task 11 ESM + Task 13 benchmark need GPUs) ---"
AWS_KEY=/Users/joshuaabbott/Downloads/jma_key.pem
AWS_HOST=ubuntu@ec2-3-17-4-165.us-east-2.compute.amazonaws.com
check "AWS SSH key present" "test -f $AWS_KEY"
if ssh -i "$AWS_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$AWS_HOST" 'echo ok' >/dev/null 2>&1; then
    echo "  PASS  AWS instance reachable"
    PASS=$((PASS+1))
    ssh -i "$AWS_KEY" -o ConnectTimeout=10 "$AWS_HOST" 'nvidia-smi -L 2>/dev/null | head -1' 2>/dev/null | sed 's/^/        /'
else
    echo "  WARN  AWS instance NOT reachable - may be stopped."
    echo "        Restart it before Tasks 11/13. Tasks 1-10, 12, 14 are local/CPU."
fi

echo
echo "=========================================="
echo "  Pre-flight: $PASS passed, $FAIL failed"
echo "=========================================="
if [ "$FAIL" -gt 0 ]; then
    echo "  Resolve failures before executing Plan 2."
    exit 1
fi
echo "  Environment ready. Proceed with subagent-driven-development."
