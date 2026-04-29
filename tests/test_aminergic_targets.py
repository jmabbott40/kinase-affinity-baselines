"""Tests for aminergic target list module and audit decision logic."""
import re

import pytest

from scripts.aminergic_audit.run_audit import (
    HALT_MIN_TARGETS,
    PASS_FRACTION_HIGH,
    PASS_FRACTION_LOW,
    compute_decision,
)
from scripts.aminergic_audit.target_lists import (
    AMINERGIC_TARGETS_BY_FAMILY,
    get_all_gene_symbols,
    get_gene_to_family,
)


def test_target_families_are_complete():
    expected = {"dopamine", "serotonin", "adrenergic",
                "histamine", "muscarinic", "trace_amine"}
    assert set(AMINERGIC_TARGETS_BY_FAMILY.keys()) == expected


def test_all_gene_symbols_are_unique():
    genes = get_all_gene_symbols(include_taar=True)
    assert len(genes) == len(set(genes)), \
        f"Duplicates: {[g for g in set(genes) if genes.count(g) > 1]}"


def test_gene_to_family_inverse_consistent():
    mapping = get_gene_to_family()
    for family, members in AMINERGIC_TARGETS_BY_FAMILY.items():
        for gene in members:
            assert mapping[gene] == family


def test_gene_symbols_match_hgnc_format():
    pattern = re.compile(r"^[A-Z0-9]+$")
    for genes in [v for v in AMINERGIC_TARGETS_BY_FAMILY.values()]:
        for gene in genes:
            assert pattern.match(gene), f"Invalid: {gene}"


def test_exclude_taar_option():
    with_taar = set(get_all_gene_symbols(include_taar=True))
    without_taar = set(get_all_gene_symbols(include_taar=False))
    assert with_taar - without_taar == {"TAAR1"}


def test_expected_target_count():
    n_without_taar = len(get_all_gene_symbols(include_taar=False))
    n_with_taar = len(get_all_gene_symbols(include_taar=True))
    assert 30 <= n_without_taar <= 40
    assert n_with_taar == n_without_taar + 1


# ──────────────────────────────────────────────────────────────────────
# Decision logic tests — pin down boundary behavior
# These tests guard the gating decision that determines whether the rest
# of Phase 1 proceeds. The HALT-first ordering is critical: an absolute
# count below HALT_MIN_TARGETS overrides any pass-fraction tier.
# ──────────────────────────────────────────────────────────────────────


def test_halt_takes_priority_over_high_pass_fraction():
    """Even at 100% pass fraction, n_pass < HALT_MIN_TARGETS triggers HALT."""
    # 25 / 25 = 100% pass fraction, but absolute count is below the floor
    result = compute_decision(n_pass=25, n_total=25, threshold=500)
    assert result["decision"] == "HALT"


def test_option_a_at_threshold_boundary():
    """n_pass exactly equals HALT_MIN_TARGETS, pass_fraction at HIGH boundary."""
    # 30 / 36 ≈ 83.3% — actual Phase 1 audit outcome
    result = compute_decision(n_pass=30, n_total=36, threshold=500)
    assert result["decision"] == "OPTION_A"
    assert result["n_targets_passing"] == 30
    assert abs(result["pass_fraction"] - 30 / 36) < 1e-9


@pytest.mark.parametrize("n_pass, n_total, expected", [
    # HALT region: absolute count below floor regardless of fraction
    (29, 36, "HALT"),                # one short of HALT_MIN
    (10, 36, "HALT"),                # well below
    (29, 30, "HALT"),                # high fraction but absolute below floor
    # OPTION_A region: ≥ HALT_MIN AND ≥ PASS_FRACTION_HIGH (80%)
    (30, 36, "OPTION_A"),            # 83.3% — actual audit decision
    (40, 50, "OPTION_A"),            # 80.0% — exact lower edge
    (45, 50, "OPTION_A"),            # 90%
    # OPTION_A_FLAGGED region: ≥ HALT_MIN AND in [60%, 80%)
    (30, 50, "OPTION_A_FLAGGED"),    # 60.0% — exact lower edge
    (35, 50, "OPTION_A_FLAGGED"),    # 70%
    (39, 50, "OPTION_A_FLAGGED"),    # 78%
    # OPTION_B_PIVOT region: ≥ HALT_MIN AND < 60%
    (30, 60, "OPTION_B_PIVOT"),      # 50%
    (30, 100, "OPTION_B_PIVOT"),     # 30%
])
def test_decision_logic_boundaries(n_pass, n_total, expected):
    """Verify decision tier ordering across boundary cases."""
    result = compute_decision(n_pass=n_pass, n_total=n_total, threshold=500)
    assert result["decision"] == expected


def test_decision_summary_schema():
    """Decision summary contains all keys consumed by audit_decision.json."""
    result = compute_decision(n_pass=30, n_total=36, threshold=500)
    expected_keys = {
        "n_targets_total", "n_targets_passing", "pass_fraction",
        "decision", "decision_message", "threshold",
    }
    assert set(result.keys()) == expected_keys


def test_zero_total_does_not_divide_by_zero():
    """Edge case: n_total=0 should still produce a valid decision (HALT)."""
    result = compute_decision(n_pass=0, n_total=0, threshold=500)
    assert result["decision"] == "HALT"
    assert result["pass_fraction"] == 0.0


def test_threshold_constants_match_spec():
    """The constants used at runtime match spec Section 4.4 thresholds."""
    assert HALT_MIN_TARGETS == 30
    assert PASS_FRACTION_HIGH == 0.80
    assert PASS_FRACTION_LOW == 0.60
