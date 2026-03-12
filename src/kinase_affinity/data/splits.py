"""Train/validation/test splitting strategies.

Three splitting strategies, each revealing different aspects of model
generalization:

    1. Random split — baseline performance (with target stratification)
    2. Scaffold split — tests generalization to novel chemical scaffolds
    3. Target split — tests generalization to unseen kinase subfamilies

Scaffold splitting uses Murcko generic scaffolds (ring systems only)
to ensure that structurally similar compounds don't leak between splits.
This is the recommended evaluation for drug discovery applications.

Target splitting holds out entire kinase subfamilies, testing whether
models learn generalizable protein-ligand interaction patterns or just
memorize target-specific SAR.

Usage:
    from kinase_affinity.data.splits import create_splits
    splits = create_splits(df, strategy="scaffold", config=config)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def random_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    stratify_col: str | None = None,
) -> dict[str, np.ndarray]:
    """Random train/val/test split with optional stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split.
    train_frac, val_frac, test_frac : float
        Split proportions (must sum to 1.0).
    seed : int
        Random seed for reproducibility.
    stratify_col : str, optional
        Column to stratify by (e.g., 'target_chembl_id').

    Returns
    -------
    dict[str, np.ndarray]
        {'train': indices, 'val': indices, 'test': indices}
    """
    raise NotImplementedError("TODO: Implement random split")


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "std_smiles",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Scaffold-based split using Murcko generic scaffolds.

    Groups molecules by their Murcko generic scaffold (ring system
    skeleton). Entire scaffold groups are assigned to train, val, or
    test to prevent structural leakage.

    Scaffolds are sorted by size (largest first) and greedily assigned
    to splits to approximate the target proportions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a SMILES column.
    smiles_col : str
        Column containing standardized SMILES.
    train_frac, val_frac, test_frac : float
        Target split proportions.
    seed : int
        Random seed for tie-breaking.

    Returns
    -------
    dict[str, np.ndarray]
        {'train': indices, 'val': indices, 'test': indices}
    """
    raise NotImplementedError("TODO: Implement scaffold split")


def target_split(
    df: pd.DataFrame,
    target_col: str = "target_chembl_id",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Target-based split: hold out entire kinase subfamilies.

    Assigns all measurements for held-out targets to the test set,
    testing whether models generalize to unseen protein targets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain target identifier column.
    target_col : str
        Column with target identifiers.
    seed : int
        Random seed for selecting holdout targets.

    Returns
    -------
    dict[str, np.ndarray]
        {'train': indices, 'val': indices, 'test': indices}
    """
    raise NotImplementedError("TODO: Implement target-based split")


def create_splits(
    df: pd.DataFrame,
    strategy: str,
    config: dict,
) -> dict[str, np.ndarray]:
    """Create train/val/test split using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Full curated dataset.
    strategy : str
        One of 'random', 'scaffold', 'target'.
    config : dict
        Split configuration from dataset YAML.

    Returns
    -------
    dict[str, np.ndarray]
        Split indices.
    """
    split_config = config["splits"][strategy]
    if strategy == "random":
        return random_split(df, **{k: v for k, v in split_config.items() if k != "stratify_by_target"})
    elif strategy == "scaffold":
        return scaffold_split(df, **{k: v for k, v in split_config.items() if k != "scaffold_type"})
    elif strategy == "target":
        return target_split(df, seed=split_config.get("seed", 42))
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def save_splits(splits: dict[str, np.ndarray], output_path: Path) -> None:
    """Save split indices to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.tolist() for k, v in splits.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f)
    logger.info("Saved splits to %s", output_path)


def load_splits(path: Path) -> dict[str, np.ndarray]:
    """Load split indices from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}
