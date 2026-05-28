# Plan 4 Manuscript Handoff

**Date:** 2026-05-28
**Status:** Ready for manuscript drafting
**Predecessor:** [2026-05-28-plan3-completion-summary.md](2026-05-28-plan3-completion-summary.md)

This handoff inventories what's frozen-and-ready for the bioRxiv preprint, what still needs
to be done, what the writer should NOT touch while drafting, and a small set of authorial
judgment calls.

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
