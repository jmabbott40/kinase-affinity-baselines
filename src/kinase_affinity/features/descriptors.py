"""RDKit 2D molecular descriptor calculation.

RDKit provides ~200 computed molecular descriptors covering:
    - Constitutional: MW, heavy atom count, ring count
    - Topological: Wiener index, Balaban J, etc.
    - Electronic: partial charges, TPSA
    - Physicochemical: LogP, HBA, HBD, rotatable bonds
    - Complexity: BertzCT, fragment counts

Unlike fingerprints, descriptors are continuous real-valued features
that benefit from feature scaling (StandardScaler). They provide
complementary information to fingerprint-based models.

Usage:
    from kinase_affinity.features.descriptors import compute_descriptors
    desc_matrix, desc_names = compute_descriptors(smiles_list)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def smiles_to_descriptors(smiles: str) -> dict[str, float] | None:
    """Compute all RDKit 2D descriptors for a single molecule.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string.

    Returns
    -------
    dict[str, float] or None
        Dictionary of descriptor_name → value, or None if SMILES is invalid.
    """
    raise NotImplementedError("TODO: Implement descriptor computation")


def compute_descriptors(
    smiles_list: list[str],
    drop_missing_threshold: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    """Compute RDKit descriptors for a list of SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        List of canonical SMILES strings.
    drop_missing_threshold : float
        Drop descriptors with more than this fraction of NaN values.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (descriptor_matrix, descriptor_names).
        Matrix shape: (n_molecules, n_descriptors).
    """
    raise NotImplementedError("TODO: Implement batch descriptor computation")
