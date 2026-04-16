#!/usr/bin/env python3
"""Step 4: SFT training on synthesized reasoning path data."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crps.training import CRPSTrainer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: SFT training on synthesized reasoning paths."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to synthesized training data JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override training output directory from config.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    training_config = config["training"].copy()

    # Inject model path into training config
    training_config["model_name_or_path"] = config["model"]["name_or_path"]

    if args.output_dir:
        training_config["output_dir"] = args.output_dir

    logger.info(f"Training config: {training_config}")
    logger.info(f"Data path: {args.data_path}")

    trainer = CRPSTrainer(config=training_config)

    logger.info("Starting training...")
    trainer.train(data_path=args.data_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
