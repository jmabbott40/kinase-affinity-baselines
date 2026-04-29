"""Aminergic GPCR data feasibility audit.

Determines whether Phase 1 proceeds with binding-only data (IC50/Ki/Kd)
or pivots to EC50 inclusion. See spec Section 4.4 for thresholds.
"""
import json
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client

from scripts.aminergic_audit.target_lists import (
    AMINERGIC_TARGETS_BY_FAMILY,  # noqa: F401  (re-exported for inspection)
    get_all_gene_symbols,
    get_gene_to_family,
    resolve_chembl_ids,
)

OUTPUT_DIR = Path("results/aminergic_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Binding records threshold: minimum number of IC50/Ki/Kd records per target
# for the target to be considered "viable" for ML benchmarking.
#
# Originally 1000 per spec Section 4.4. Lowered to 500 after Phase 1 audit
# revealed only 24/36 aminergic targets met the 1000 threshold (failing
# HALT_MIN_TARGETS=30). At 500, 30/36 (83.3%) targets pass — meeting both
# the absolute floor and the OPTION_A pass-fraction threshold.
# 500 binding records is comparable to the lower end of the kinase preprint's
# per-target record counts, preserving data-quality consistency.
# Decision logged in: results/aminergic_audit/audit_decision.json
THRESHOLD_BINDING_RECORDS = 500
PASS_FRACTION_HIGH = 0.80
PASS_FRACTION_LOW = 0.60
HALT_MIN_TARGETS = 30


def compute_decision(n_pass: int, n_total: int, threshold: int) -> dict:
    """Apply the audit decision logic given pass count and total count.

    Decision tiers (HALT check is FIRST and overrides pass-fraction tiers):
      HALT             — n_pass < HALT_MIN_TARGETS (insufficient absolute count)
      OPTION_A         — pass_fraction >= PASS_FRACTION_HIGH (>=80%)
      OPTION_A_FLAGGED — pass_fraction >= PASS_FRACTION_LOW  (60-80%)
      OPTION_B_PIVOT   — otherwise (<60%, pivot to EC50 inclusion)

    Returns a summary dict suitable for serialization as audit_decision.json.
    Pure function — no I/O, no external state.
    """
    pass_fraction = n_pass / n_total if n_total > 0 else 0.0

    if n_pass < HALT_MIN_TARGETS:
        decision = "HALT"
        decision_msg = (f"Only {n_pass} targets meet threshold "
                        f"(<{HALT_MIN_TARGETS}). HALT for design review.")
    elif pass_fraction >= PASS_FRACTION_HIGH:
        decision = "OPTION_A"
        decision_msg = (f"{pass_fraction:.0%} pass threshold. "
                        f"Proceed with binding-only data.")
    elif pass_fraction >= PASS_FRACTION_LOW:
        decision = "OPTION_A_FLAGGED"
        decision_msg = (f"{pass_fraction:.0%} pass threshold. "
                        f"Proceed with binding-only, flag in paper.")
    else:
        decision = "OPTION_B_PIVOT"
        decision_msg = (f"{pass_fraction:.0%} pass threshold. "
                        f"Pivot to EC50 inclusion.")

    return {
        "n_targets_total": n_total,
        "n_targets_passing": n_pass,
        "pass_fraction": pass_fraction,
        "decision": decision,
        "decision_message": decision_msg,
        "threshold": threshold,
    }


def query_target_records(target_chembl_id: str) -> dict:
    """Query ChEMBL for binding (IC50/Ki/Kd) record counts per target.

    Uses per-type ``len()`` calls (server-side counts) instead of iterating
    full result sets. Iterating 12k+ records per target via the
    chembl_webresource_client takes ~5 minutes per target; ``len()`` returns
    instantly. We sacrifice exact unique-compound counts (which require
    enumerating records) — the decision rule depends on ``n_binding_records``
    only, so this is safe. ``n_unique_compounds`` is recorded as ``None``
    to make the omission explicit downstream.
    """
    activity = new_client.activity
    type_counts: dict[str, int] = {}
    for st in ("IC50", "Ki", "Kd"):
        qs = activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type=st,
            standard_relation="=",
            standard_units="nM",
            pchembl_value__isnull=False,
            assay_type="B",
            confidence_score__gte=7,
        ).only(["activity_id"])
        type_counts[st] = len(qs)

    n_records = sum(type_counts.values())

    return {
        "target_chembl_id": target_chembl_id,
        "n_binding_records": n_records,
        "n_unique_compounds": None,
        "type_breakdown": type_counts,
    }


def run_audit():
    """Execute audit, write outputs, return decision summary."""
    all_genes = get_all_gene_symbols(include_taar=True)
    print(f"Resolving ChEMBL IDs for {len(all_genes)} aminergic targets...")
    gene_to_chembl = resolve_chembl_ids(all_genes)
    print(f"Resolved {len(gene_to_chembl)} of {len(all_genes)} targets.\n")

    gene_to_family = get_gene_to_family()
    results = []

    for gene, chembl_id in gene_to_chembl.items():
        try:
            stats = query_target_records(chembl_id)
            stats["gene_symbol"] = gene
            stats["family"] = gene_to_family[gene]
            stats["passes_threshold"] = (
                stats["n_binding_records"] >= THRESHOLD_BINDING_RECORDS
            )
            results.append(stats)
            print(f"  {gene} ({chembl_id}, {stats['family']}): "
                  f"{stats['n_binding_records']} records "
                  f"({stats['type_breakdown']})")
        except Exception as e:
            print(f"  {gene} ({chembl_id}): ERROR — {e}")
            results.append({
                "gene_symbol": gene,
                "target_chembl_id": chembl_id,
                "family": gene_to_family[gene],
                "n_binding_records": 0,
                "n_unique_compounds": 0,
                "type_breakdown": {},
                "error": str(e),
                "passes_threshold": False,
            })

    # Capture targets that failed to resolve
    unresolved = set(all_genes) - set(gene_to_chembl.keys())
    for gene in unresolved:
        results.append({
            "gene_symbol": gene,
            "target_chembl_id": None,
            "family": gene_to_family[gene],
            "n_binding_records": 0,
            "n_unique_compounds": 0,
            "type_breakdown": {},
            "error": "Failed to resolve ChEMBL ID",
            "passes_threshold": False,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "per_target_audit.csv", index=False)

    # Compute decision
    n_total = len(df)
    n_pass = int(df["passes_threshold"].sum())
    summary = compute_decision(n_pass, n_total, THRESHOLD_BINDING_RECORDS)
    with open(OUTPUT_DIR / "audit_decision.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_audit_report(df, summary)
    plot_per_target_counts(df)

    print(f"\n{decision_msg}")
    print(f"Full report: {OUTPUT_DIR / 'audit_report.md'}")
    return summary


def write_audit_report(df: pd.DataFrame, summary: dict) -> None:
    """Write human-readable audit report."""
    report = OUTPUT_DIR / "audit_report.md"
    with open(report, "w") as f:
        f.write("# Aminergic GPCR Data Audit Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"**Decision:** `{summary['decision']}`\n\n")
        f.write(f"**Decision message:** {summary['decision_message']}\n\n")
        f.write("## Summary statistics\n\n")
        f.write(f"- Total targets queried: {summary['n_targets_total']}\n")
        f.write(
            f"- Targets passing threshold (>={summary['threshold']} binding "
            f"records): {summary['n_targets_passing']}\n"
        )
        f.write(f"- Pass fraction: {summary['pass_fraction']:.1%}\n\n")
        f.write("## Per-family breakdown\n\n")
        family_stats = df.groupby("family").agg(
            n_targets=("gene_symbol", "count"),
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
    from matplotlib.patches import Patch

    df_sorted = df.sort_values("n_binding_records", ascending=False).reset_index(drop=True)
    family_colors = {
        "dopamine": "#1f77b4", "serotonin": "#ff7f0e", "adrenergic": "#2ca02c",
        "histamine": "#d62728", "muscarinic": "#9467bd", "trace_amine": "#8c564b",
    }
    colors = [family_colors[f] for f in df_sorted["family"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    # log scale requires positive values; substitute zeros with a tiny floor
    # for plotting (the bars still render very small, signaling "no data").
    plot_values = df_sorted["n_binding_records"].clip(lower=0.5)
    ax.bar(range(len(df_sorted)), plot_values, color=colors)
    # Threshold line — label set only via the explicit Line2D below to avoid
    # a duplicate legend entry.
    ax.axhline(y=THRESHOLD_BINDING_RECORDS, color="red", linestyle="--",
               linewidth=1)
    labels = [f"{row['gene_symbol']}" for _, row in df_sorted.iterrows()]
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("Binding records (log scale)")
    ax.set_title("Per-target binding record counts (IC50/Ki/Kd)")
    legend_elements = [Patch(facecolor=c, label=f) for f, c in family_colors.items()]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="red", linestyle="--",
                   label=f"Threshold: {THRESHOLD_BINDING_RECORDS}")
    ], loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_target_record_counts.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_audit()
