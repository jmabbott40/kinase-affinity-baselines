"""Morgan fingerprint generation using RDKit.

Morgan (circular) fingerprints encode the local chemical environment
around each atom up to a given radius. They are the standard baseline
representation for molecular property prediction.

Key parameters:
    - radius: number of bonds from each atom to consider
      (radius=2 ≈ ECFP4, radius=3 ≈ ECFP6)
    - n_bits: fingerprint length (2048 is standard)

The fingerprint is a bit vector where each bit indicates the presence
or absence of a particular substructural feature.

Usage:
    from kinase_affinity.features.fingerprints import smiles_to_morgan_fp
    fp = smiles_to_morgan_fp("c1ccccc1", radius=2, n_bits=2048)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string.
    radius : int
        Fingerprint radius (2 = ECFP4, 3 = ECFP6).
    n_bits : int
        Length of the bit vector.

    Returns
    -------
    np.ndarray or None
        Binary fingerprint vector of shape (n_bits,), or None if
        the SMILES cannot be parsed.
    """
    raise NotImplementedError("TODO: Implement Morgan FP generation")


def compute_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute Morgan fingerprints for a list of SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        List of canonical SMILES strings.
    radius : int
        Fingerprint radius.
    n_bits : int
        Fingerprint length.

    Returns
    -------
    np.ndarray
        Fingerprint matrix of shape (n_molecules, n_bits).
    """
    raise NotImplementedError("TODO: Implement batch FP computation")
