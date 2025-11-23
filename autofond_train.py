"""
Simplified training script for AutoFOND development
No W&B, simple CSV logging, epoch-based training
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import yaml

from src.train.fit_simple import fit_simple
from src.utils.misc import config_logging


def main():
    parser = argparse.ArgumentParser(description="Train AutoFOND model")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default="experiments",
                       help="Directory for experiment outputs")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name (default: auto-generated)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create experiment directory
    exp_name = args.name or f"{config['algo']}_{config['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    config_logging()
    logging.info(f"Starting experiment: {exp_name}")
    logging.info(f"Output directory: {exp_dir}")

    # Save config to experiment directory
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create CSV logger
    from src.utils.simple_logger import CSVLogger
    logger = CSVLogger(exp_dir / "metrics.csv")

    # Run training
    fit_simple(
        exp_dir=str(exp_dir),
        logger=logger,
        seed=config.get("seed", 0),
        trial_seed=config.get("trial_seed", 0),
        hparams_seed=config.get("hparams_seed", 0),
        algorithm_name=config["algo"],
        dataset_name=config["dataset"],
        data_dir=config["data_dir"],
        num_workers=config.get("num_workers", 4),
        test_envs=[config["test_set_id"]],
        overlap_type=config.get("overlap", "none"),
        holdout_fraction=config.get("holdout_fraction", 0.2),
        n_epochs=config.get("n_epochs", 100),  # Changed to epochs!
        checkpoint_freq=config.get("checkpoint_freq", 5),  # In epochs
        model_checkpoint=config.get("model_checkpoint", {
            "metric": "val/acc",
            "maximize": True
        }),
        num_domain_linked_classes=config.get("num_domain_linked_classes"),
        num_classes=config.get("num_classes"),
        # AutoAugmentation specific
        auto_augment=config.get("auto_augment", False),
        augment_search_epochs=config.get("augment_search_epochs", 10),
    )

    logging.info(f"Experiment complete. Results saved to {exp_dir}")
    print(f"\n✓ Experiment complete!")
    print(f"  Results: {exp_dir / 'metrics.csv'}")
    print(f"  Best model: {exp_dir / 'best_model.ckpt'}")
    print(f"  Config: {exp_dir / 'config.json'}")


if __name__ == "__main__":
    main()
