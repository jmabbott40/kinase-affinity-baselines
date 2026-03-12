"""Molecule standardization using RDKit.

Standardization pipeline applied to each molecule:
    1. Parse SMILES → RDKit Mol (discard unparseable)
    2. Remove salts (keep largest fragment)
    3. Neutralize charges
    4. Generate canonical SMILES
    5. Filter by molecular weight and heavy atom count

Each step logs how many molecules are removed, enabling transparent
data curation reporting in the data card.

Usage:
    from kinase_affinity.data.standardize import standardize_smiles
    canonical, is_valid = standardize_smiles("CC(=O)Oc1ccccc1C(=O)O.[Na]")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def standardize_smiles(
    smiles: str,
    mw_min: float = 100.0,
    mw_max: float = 900.0,
    max_heavy_atoms: int = 100,
) -> tuple[str | None, bool]:
    """Standardize a SMILES string.

    Parameters
    ----------
    smiles : str
        Input SMILES string (may contain salts, charges, etc.).
    mw_min : float
        Minimum molecular weight in Da.
    mw_max : float
        Maximum molecular weight in Da.
    max_heavy_atoms : int
        Maximum number of heavy (non-hydrogen) atoms.

    Returns
    -------
    tuple[str | None, bool]
        (canonical_smiles, is_valid). If invalid, returns (None, False).
    """
    raise NotImplementedError("TODO: Implement RDKit standardization pipeline")


def standardize_dataframe(df, config: dict):
    """Apply standardization to a DataFrame of molecules.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'canonical_smiles' column.
    config : dict
        Standardization parameters from dataset config.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with added 'std_smiles' column.
        Invalid molecules are removed.
    dict
        Statistics: counts of molecules removed at each step.
    """
    raise NotImplementedError("TODO: Implement batch standardization")
