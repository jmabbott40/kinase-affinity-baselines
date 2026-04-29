"""Tests for aminergic target list module."""
import re

import pytest  # noqa: F401

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
