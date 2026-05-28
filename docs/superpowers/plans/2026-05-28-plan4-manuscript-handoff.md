# Plan 4 Manuscript Handoff

**Date:** 2026-05-28
**Status:** Ready for manuscript drafting
**Predecessor:** [2026-05-28-plan3-completion-summary.md](2026-05-28-plan3-completion-summary.md)

This handoff inventories what's frozen-and-ready for the bioRxiv preprint, what still needs
to be done, what the writer should NOT touch while drafting, and a small set of authorial
judgment calls.

---

## START HERE — paste this into a fresh session

```
Begin Plan 4: draft the bioRxiv manuscript for the cross-class affinity benchmark.

Handoff context:    /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-05-28-plan4-manuscript-handoff.md
Completion summary: /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-05-28-plan3-completion-summary.md
Plan 3 plan:        /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-05-27-plan3-cross-class-analysis.md
Spec:               /Users/joshuaabbott/mlproject/docs/superpowers/specs/2026-05-27-plan3-cross-class-analysis-design.md
Results inventory:  /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/README.md

Read the handoff FIRST — it has the curated context (frozen outputs, what NOT to
touch, open authorial questions). Then read the completion summary's §4 (Key findings)
and §6 (Caveats). Then run the pre-flight check (save the bash script in the handoff
doc to /tmp/plan4_kickoff_preflight.sh and execute it). Once pre-flight passes:

1. Switch working directory to /Users/joshuaabbott/gpcr-aminergic-benchmarks
   (NOT the kinase repo — manuscript drafts and references live with the
   results that produced them).
2. Surface the 5 open authorial questions from §"Open questions" of the handoff
   doc. Some are decisions only Joshua can make (title framing, RNS-pivot
   narrative voice); others can wait until the relevant draft section is
   underway.
3. Use the anthropic-skills:docx skill for the manuscript output. Create a
   manuscript/ subdirectory in the GPCR repo for drafts and revisions.
4. Reverse-write order: Methods → Results → Discussion → Introduction. Don't
   try to draft the whole paper in one session — expect 6-10 working days
   across multiple sessions.

Plan 4 is NOT subagent-driven-development. It's iterative prose work with
human-in-the-loop review. Do not invoke superpowers:subagent-driven-development;
do not split this into 24 small tasks for parallel subagents. One writer per
section, sequential, with revisions before moving on.
```

That's it. The fresh session reads this file, runs the pre-flight script below,
and proceeds to authorial-questions discussion before drafting.

---

## Pre-flight environment check

Save this as `/tmp/plan4_kickoff_preflight.sh` and execute. It confirms the
analysis state is frozen and ready to consume as manuscript inputs.

```bash
#!/usr/bin/env bash
# Plan 4 kickoff pre-flight — verifies Plan 3 outputs are intact and pinned.
set -u
echo "=========================================="
echo "  Plan 4 Kickoff Pre-Flight"
echo "=========================================="
PASS=0; FAIL=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ✅ $1"; PASS=$((PASS+1)); else echo "  ❌ $1"; FAIL=$((FAIL+1)); fi; }

echo
echo "--- Repos & frozen tags (locally + on origin) ---"
check "library repo exists"                  "test -d /Users/joshuaabbott/target-affinity-ml/.git"
check "GPCR repo exists"                     "test -d /Users/joshuaabbott/gpcr-aminergic-benchmarks/.git"
check "kinase repo exists"                   "test -d /Users/joshuaabbott/mlproject/.git"
check "library v1.2.0 tag (local)"           "test -n \"\$(cd /Users/joshuaabbott/target-affinity-ml && git tag -l v1.2.0)\""
check "library v1.2.0 tag (origin)"          "git ls-remote --tags https://github.com/jmabbott40/target-affinity-ml.git v1.2.0 | grep -q refs/tags/v1.2.0"
check "GPCR v1.1.0 tag (local)"              "test -n \"\$(cd /Users/joshuaabbott/gpcr-aminergic-benchmarks && git tag -l v1.1.0)\""
check "GPCR v1.1.0 tag (origin)"             "git ls-remote --tags https://github.com/jmabbott40/gpcr-aminergic-benchmarks.git v1.1.0 | grep -q refs/tags/v1.1.0"
check "GPCR pyproject pins library to v1.2.0" "grep -q 'target-affinity-ml.*@v1.2.0' /Users/joshuaabbott/gpcr-aminergic-benchmarks/pyproject.toml"
check "library plan3-development branch deleted from origin" "! git ls-remote --heads https://github.com/jmabbott40/target-affinity-ml.git plan3-development | grep -q refs/heads/plan3-development"

echo
echo "--- Plan 3 handoff docs ---"
check "plan3 completion summary"             "test -f /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-05-28-plan3-completion-summary.md"
check "plan4 manuscript handoff (this file)" "test -f /Users/joshuaabbott/mlproject/docs/superpowers/plans/2026-05-28-plan4-manuscript-handoff.md"

echo
echo "--- Main-text tables (4) ---"
check "Table 1 — dataset summary"            "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/tables/01_dataset_summary.csv"
check "Table 2 — headline RMSE"              "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/tables/02_headline_rmse.csv"
check "Table 3 — hypothesis outcomes rollup" "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/tables/03_hypothesis_outcomes.csv"
check "Table 4 — metric correlations (unified)" "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/tables/04_metric_correlations.csv"

echo
echo "--- Hypothesis companion tables (4) ---"
for h in h1 h2 h3 h4; do
    check "Table 3 — $h companion"           "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/tables/03_hypothesis_outcomes_${h}.csv"
done

echo
echo "--- Main-text figures (5) ---"
for f in figure1_design_overview figure2_headline_replication figure3_scaffold_degradation figure4_plddt_advantage figure5_hypothesis_summary; do
    check "Figure — $f"                       "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/figures/${f}.png"
done

echo
echo "--- Supplementary outputs ---"
check "per_target_metrics_scaffold.csv"      "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/supplement/per_target_metrics_scaffold.csv"
check "structure_provenance.csv"             "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/supplement/structure_provenance.csv"
check "results/README.md (T21 inventory)"    "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/results/README.md"

echo
echo "--- Source notebooks (for figure regeneration only) ---"
for nb in 05_scaffold_diversity 06_plddt_analysis 07_cross_class_comparison; do
    check "notebook — $nb.ipynb"             "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/notebooks/${nb}.ipynb"
done

echo
echo "--- Per-target analysis inputs (drives manuscript supplementary, do not modify) ---"
check "per_target_plddt.csv (T14 output)"    "test -f /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/processed/v1/per_target_plddt.csv"
check "GPCR per-target benchmark CSVs (21)" "test \$(ls /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/processed/v1/per_target/per_target_*.csv 2>/dev/null | wc -l) -eq 21"
check "kinase per-target CSVs (21)"          "test \$(ls /Users/joshuaabbott/gpcr-aminergic-benchmarks/data/kinase_reference/benchmark_v1/per_target/per_target_*.csv 2>/dev/null | wc -l) -ge 18"

echo
echo "--- Python environment + library v1.2.0 ---"
CONDA_PY=/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
check "kinase-affinity conda env exists"     "test -x \$CONDA_PY"
check "Python is 3.11"                       "\$CONDA_PY --version 2>&1 | grep -q 'Python 3.11'"
check "library importable (v1.2.0)"          "test \"\$(\$CONDA_PY -c 'import target_affinity_ml; print(target_affinity_ml.__version__)' 2>/dev/null)\" = '1.2.0'"
check "benchmarks submodule importable"      "\$CONDA_PY -c 'from target_affinity_ml.benchmarks import compute_scaffold_metrics, h1_rf_vs_deep, compute_binding_site_plddt'"

echo
echo "--- Manuscript drafting prereqs ---"
check "python-docx available (for anthropic-skills:docx)" "\$CONDA_PY -c 'import docx' 2>/dev/null || pip show python-docx >/dev/null 2>&1"
check "pandoc available (optional, for md→docx if used)" "command -v pandoc"

echo
echo "=========================================="
echo "  Pre-flight: \$PASS passed, \$FAIL failed"
echo "=========================================="
if [ \$FAIL -gt 0 ]; then
    echo "  Investigate failures before drafting."
    echo "  - Missing tags  → check git ls-remote on the relevant repo and re-tag if needed"
    echo "  - Missing tables/figures → these are frozen at v1.1.0; should not be missing"
    echo "  - Missing library v1.2.0 → pip install --upgrade git+https://github.com/jmabbott40/target-affinity-ml.git@v1.2.0"
    echo "  - Missing python-docx     → pip install python-docx"
    exit 1
fi
echo "  Analysis state intact. Ready to draft manuscript."
echo "  Working directory should be /Users/joshuaabbott/gpcr-aminergic-benchmarks"
```

---

## What's ready

### Main-text tables (4)

| Path (relative to `gpcr-aminergic-benchmarks/`) | Paper section | Rows |
|---|---|---|
| `results/tables/01_dataset_summary.csv` | §Methods – Datasets | 2 (kinase vs GPCR) |
| `results/tables/02_headline_rmse.csv` | §Results – Cross-class replication | 42 (7 models × 3 splits × 2 classes), with `rmse_str` formatted column |
| `results/tables/03_hypothesis_outcomes.csv` | §Results – Hypothesis tests | 4 (one per H1–H4) with rollup verdict counts |
| `results/tables/04_metric_correlations.csv` | §Results – Cross-class scaffold + pLDDT | 13 (10 scaffold = 5 metrics × 2 directions; 1 pLDDT regression; 2 pLDDT distribution: KS + Welch) |

### Hypothesis companion tables (4)

| Path | Use |
|---|---|
| `results/tables/03_hypothesis_outcomes_h1.csv` | H1 per-test details (model_pair × class × split) — Bonferroni-corrected p, verdict. GPCR-only rows (6 of 12 planned). |
| `results/tables/03_hypothesis_outcomes_h2.csv` | H2 per-test details (model × class × transition) — ratio, in-range bool; final row holds class × split interaction p = 5.887 × 10⁻⁵. |
| `results/tables/03_hypothesis_outcomes_h3.csv` | H3 advantage values (class × split) — mean ESM-2 advantage, Bonferroni-corrected p. |
| `results/tables/03_hypothesis_outcomes_h4.csv` | H4 per-pair flip rates (model_pair × split) — all verdicts "a) similar" (GPCR-only). |

### Main-text figures (5)

| Path | Paper section | Notes |
|---|---|---|
| `results/figures/figure1_design_overview.png` | §Methods – Benchmark design | Schematic: classes → splits → models pipeline. Working figure; T-21 left polish for manuscript phase. |
| `results/figures/figure2_headline_replication.png` | §Results – Cross-class replication | 2-panel grouped-bar: per-model RMSE × split, per class. |
| `results/figures/figure3_scaffold_degradation.png` | §Results – Scaffold-diversity correlation | 2×5 grid (directions × metrics) scatter + per-class regression lines. |
| `results/figures/figure4_plddt_advantage.png` | §Results – pLDDT-stratified ESM-2 advantage | Single scatter, both classes overlaid, regression lines, interaction p in title. |
| `results/figures/figure5_hypothesis_summary.png` | §Discussion – Hypothesis outcome summary | Grouped-bar count of significant vs non-significant tests per hypothesis. |

### Supplementary tables

| Path | Use |
|---|---|
| `results/supplement/per_target_metrics_scaffold.csv` | 543 rows (36 GPCR + 507 kinase) — per-target scaffold diversity. Input to T18; also a paper supplementary table. |
| `results/supplement/structure_provenance.csv` | 543 rows — AlphaFold source + binding-site residue count for each target. NaN pLDDT rows carry an `error` column indicating the failure reason (e.g., "no binding site"). |
| `results/tables/04_metric_correlations_scaffold.csv` | Superseded by unified `04_metric_correlations.csv`; kept for provenance. |

### Results index

`results/README.md` in the GPCR repo indexes every table/figure to its paper section + the
producing notebook. Use this as navigation while drafting.

### Code (frozen at v1.2.0 / v1.1.0)

| Repo | Tag | URL |
|---|---|---|
| `target-affinity-ml` | `v1.2.0` (2026-05-28) | https://github.com/jmabbott40/target-affinity-ml/releases/tag/v1.2.0 |
| `gpcr-aminergic-benchmarks` | `v1.1.0` (to be tagged in P3-T24) | https://github.com/jmabbott40/gpcr-aminergic-benchmarks |
| `kinase-affinity-baselines` | `phase1-multi-class-expansion` branch | https://github.com/jmabbott40/kinase-affinity-baselines/tree/phase1-multi-class-expansion |

The library v1.2.0 is pinned in `gpcr-aminergic-benchmarks/pyproject.toml`; once GPCR v1.1.0
is tagged the analysis state is fully reproducible from those two refs.

---

## Recommended writing order

Reverse-write the manuscript — easiest sequence after results are frozen:

1. **Methods → Results → Discussion → Introduction.** Methods sections use Table 1 (datasets)
   + Figure 1 (design overview).
2. **Results section order:**
   - H1-H4 verdicts from Table 3 + per-hypothesis companions (one paragraph per hypothesis).
   - Scaffold-diversity from Table 4 (scaffold rows) + Figure 3.
   - pLDDT analysis from Table 4 (pLDDT rows) + Figure 4.
   - Cross-class headline from Figure 2 + Figure 5.
3. **Discussion** must address the four caveats from §6 of the completion summary:
   - kinase per-seed RF/XGB/EN/MLP gap (H1, H4, H3 Part A affected)
   - kinase pLDDT coverage at 53%
   - GPCR target-split n = 5 power limit
   - bootstrap CI nominal-coverage degradation at n = 5 seeds
4. **Introduction** last — once the result emphasis is firm.

---

## What's NOT ready

- **bioRxiv submission format / typesetting.** The figures are working-PNG quality. Margins,
  font choices, panel letters, R²/p annotations per panel — all noted in the T18+T19+T20
  code reviews as deferred to manuscript polish.
- **Colorblind palettes.** Figures 2-5 currently use the matplotlib default. The manuscript
  phase should switch to a colorblind-safe palette (e.g., `cmocean` or `viridis`-aligned).
- **Pre-registration deposit.** Per design spec §6.5, in-paper pre-registration via the
  Methods section is sufficient for bioRxiv. An OSF or AsPredicted registration was not
  performed and is not in scope for the preprint draft; the manuscript should clarify this.
- **Data-availability statement language.** RDKit descriptors (~80 MB) are local-only at
  `/Users/joshuaabbott/mlproject/data/processed/v1/features/rdkit_descriptors.npz`; the
  manuscript Data Availability statement needs to either flag this or include in the eventual
  Zenodo deposit.

---

## What to NOT change while writing

- **Do NOT re-run T14 hoping for better kinase coverage.** KLIFS is rate-limited; the
  267/507 kinase pLDDT count is what we have. The H3 cross-class regression has n = 239,
  which is adequate; further effort produces marginal returns.
- **Do NOT recompute the H1-H4 numbers.** They are frozen at Plan 3 / library v1.2.0. Any
  parameter sweep risks unintentional p-hacking; the pre-registration commits us to the
  numbers in Table 3 and the companion files.
- **Do NOT merge T19 (Part B) and T20 (Part A) of H3.** They ask different statistical
  questions on different units of analysis. Both should be reported with the distinction
  called out explicitly in the Results text.
- **Do NOT rewrite the experimental RNS code.** It is preserved in `rns_scoring.py` with
  `[EXPERIMENTAL]` tags for archival reasons. The manuscript should reference the pivot in
  Methods/Discussion as described below; the code stays as-is.

---

## Open questions for the writer

These need authorial judgment:

1. **How to frame the RNS → pLDDT pivot.** Three options:
   - Methodological narrative in Methods — transparent paragraph describing the two failed
     gate attempts (entropy + JSD) with ρ values, followed by the pLDDT pivot rationale.
     Most honest but draws reviewer attention.
   - Quiet substitution — describe pLDDT as the per-target structural-confidence metric in
     Methods without dwelling on the pivot, then mention the abandoned alternatives in a
     footnote or supplement.
   - Hybrid — one-sentence-in-Methods + one-paragraph-in-supplement.
   The completion summary §5 contains all the material either path would draw from.

2. **H3 Part A vs Part B presentation.** Should the distinction get its own Results
   paragraph or a footnote? The Part A test is NOT significant (p = 0.318) while Part B IS
   (p = 0.0313). A footnote risks the reader seeing a contradiction; a paragraph spends
   manuscript real estate on a methodological clarification.

3. **Figure 5 placement.** Figure 5 (hypothesis summary grouped-bar) is somewhat redundant
   with Table 3. Main-text or supplement? Vote in favor of main-text only if the
   visual gestalt of "X / Y hypotheses significant" adds something Table 3's text can't.

4. **Caveats section voice.** The four caveats in completion summary §6 are honest
   self-criticism. Does the manuscript Discussion address them in one limitations paragraph,
   or thread them through the relevant results subsections as in-context disclaimers? The
   former is more readable; the latter is more rigorous.

5. **Title framing.** The plan tagline was "When do complex ML models outperform simple
   cheminformatics baselines for kinase inhibitor affinity prediction?" The cross-class
   finding extends this. Possible reframings:
   - "Cross-class benchmarking reveals split-dependent ESM-2 advantages in kinase and GPCR
     affinity prediction"
   - "Class-specific scaffold diversity, not structural confidence, predicts ESM-2 utility in
     bioactivity modeling"
   - "Pre-registered cross-class comparison of ML methods for kinase and GPCR ligand affinity"

---

## Plan 4 scope estimate

Estimated ~6-10 working days for a bioRxiv preprint draft, depending on writer pace. **Plan 4
is NOT a subagent-driven-development task** — it is creative-prose work best done in a fresh
session with manuscript-writing tools (Word docx skill, citation manager, reference fetch via
the PubMed MCP), possibly with intermediate review-and-revise rounds rather than dispatchable
subagents.

When Plan 4 begins, the writer should:

1. Read this handoff + the completion summary (the two predecessor docs).
2. Spin up a fresh session in `gpcr-aminergic-benchmarks` (not the kinase repo) since that's
   where the tables, figures, and `results/README.md` index live.
3. Set up a `manuscript/` directory in the GPCR repo with a single docx output (the
   `anthropic-skills:docx` skill is the right tool).
4. Reverse-write per the order above. Don't touch tables/figures except for caption polish.

**Plan 3 closes here. Plan 4 (manuscript drafting) is ready to begin in a fresh session.**
