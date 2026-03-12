"""Unified training loop for all baseline models.

Config-driven training pipeline:
    1. Load dataset and split indices
    2. Generate or load cached features
    3. Instantiate model from config
    4. Optionally tune hyperparameters (grid search or Optuna)
    5. Train on training set
    6. Evaluate on validation and test sets
    7. Save model, predictions, and metrics

Usage:
    python -m kinase_affinity.training.trainer --config configs/rf_baseline.yaml --split scaffold
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "random_forest": "kinase_affinity.models.rf_model.RandomForestModel",
    "xgboost": "kinase_affinity.models.xgb_model.XGBoostModel",
    "elasticnet": "kinase_affinity.models.elasticnet_model.ElasticNetModel",
    "mlp": "kinase_affinity.models.mlp_model.MLPModel",
}


def load_model_config(config_path: Path) -> dict:
    """Load model configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_class(model_name: str):
    """Resolve model class from registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}")
    raise NotImplementedError("TODO: Implement dynamic model import")


def train_and_evaluate(
    config_path: Path,
    split_strategy: str = "random",
    dataset_version: str = "v1",
) -> dict:
    """Run the full train/evaluate pipeline for one model + split combination.

    Parameters
    ----------
    config_path : Path
        Path to model config YAML.
    split_strategy : str
        One of 'random', 'scaffold', 'target'.
    dataset_version : str
        Dataset version (subdirectory of data/processed/).

    Returns
    -------
    dict
        Evaluation metrics on the test set.
    """
    raise NotImplementedError("TODO: Implement training pipeline")


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train a baseline model")
    parser.add_argument("--config", type=Path, required=True, help="Model config YAML")
    parser.add_argument(
        "--split",
        choices=["random", "scaffold", "target"],
        default="random",
        help="Split strategy",
    )
    parser.add_argument("--dataset-version", default="v1", help="Dataset version")
    args = parser.parse_args()

    results = train_and_evaluate(args.config, args.split, args.dataset_version)
    logger.info("Results: %s", results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
