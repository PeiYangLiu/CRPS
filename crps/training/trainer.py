"""SFT Trainer for the CRPS framework.

Wraps HuggingFace ``Trainer`` with DeepSpeed support to fine-tune
causal language models on CRPS-synthesized mathematical reasoning data.

Usage:
    python -m crps.training.trainer \
        --config configs/default.yaml \
        --data_path data/synthesized/crps_30k.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from crps.training.dataset import CRPSDataset, DataCollator

logger = logging.getLogger(__name__)


class CRPSTrainer:
    """Supervised fine-tuning trainer for CRPS.

    Reads training hyper-parameters from a configuration dict (typically
    the ``training`` section of the project YAML) and orchestrates model
    loading, dataset creation, and HuggingFace ``Trainer`` execution
    with optional DeepSpeed integration.

    Args:
        config: Dict with keys matching the ``training`` section of
            ``configs/default.yaml``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, data_path: str) -> None:
        """Run the full SFT training loop.

        Args:
            data_path: Path to a JSONL file with ``{"problem", "solution"}``
                entries produced by the CRPS synthesis pipeline.
        """
        model_name: str = self.config.get("model_name_or_path", "deepseek-ai/deepseek-math-7b-base")

        logger.info("Setting up tokenizer from %s", model_name)
        tokenizer = self._setup_tokenizer(model_name)

        logger.info("Setting up model from %s", model_name)
        model = self._setup_model(model_name)

        # Resize embeddings if new tokens were added (e.g. pad token)
        model.resize_token_embeddings(len(tokenizer))

        max_seq_length: int = self.config.get("max_seq_length", 2048)
        logger.info("Loading dataset from %s (max_seq_length=%d)", data_path, max_seq_length)
        dataset = CRPSDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )

        training_args = self._build_training_arguments()
        collator = DataCollator(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )

        logger.info("Starting training …")
        trainer.train()

        # Save final checkpoint
        output_dir = Path(training_args.output_dir) / "final"
        logger.info("Saving final model to %s", output_dir)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load a causal LM and enable gradient checkpointing.

        Args:
            model_name: HuggingFace model identifier or local path.

        Returns:
            An ``AutoModelForCausalLM`` instance ready for training.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        logger.info(
            "Model loaded: %s (%.1fM params, gradient checkpointing ON)",
            model_name,
            sum(p.numel() for p in model.parameters()) / 1e6,
        )
        return model

    def _setup_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load and configure the tokenizer.

        Ensures a padding token is set (falls back to ``eos_token``)
        and forces right-padding for causal LM training.

        Args:
            model_name: HuggingFace model identifier or local path.

        Returns:
            A configured ``AutoTokenizer``.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token (%s)", tokenizer.eos_token)

        tokenizer.padding_side = "right"
        return tokenizer

    def _build_training_arguments(self) -> TrainingArguments:
        """Construct ``TrainingArguments`` from the config dict.

        Returns:
            A ``TrainingArguments`` instance with DeepSpeed, bf16, cosine
            scheduler, and other CRPS defaults applied.
        """
        cfg = self.config
        output_dir: str = cfg.get("output_dir", "./outputs")

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.get("num_epochs", 3),
            per_device_train_batch_size=cfg.get("per_device_batch_size", 2),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            learning_rate=cfg.get("learning_rate", 2e-5),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            weight_decay=cfg.get("weight_decay", 0.0),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            bf16=cfg.get("bf16", True),
            deepspeed=cfg.get("deepspeed", None),
            logging_steps=cfg.get("logging_steps", 10),
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            save_steps=cfg.get("save_steps", 500),
            save_total_limit=cfg.get("save_total_limit", 3),
            seed=cfg.get("seed", 42),
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.95,
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            disable_tqdm=False,
        )


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------


def main() -> None:
    """Load config from YAML and run SFT training.

    Example::

        python -m crps.training.trainer \\
            --config configs/default.yaml \\
            --data_path data/synthesized/crps_30k.jsonl
    """
    parser = argparse.ArgumentParser(
        description="CRPS SFT Training",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g. configs/default.yaml).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Override model name/path from config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config, "r", encoding="utf-8") as fh:
        full_config: dict[str, Any] = yaml.safe_load(fh)

    training_config: dict[str, Any] = full_config.get("training", {})

    # Inject model name from the top-level model section if not overridden
    if args.model_name_or_path:
        training_config["model_name_or_path"] = args.model_name_or_path
    elif "model_name_or_path" not in training_config:
        model_section = full_config.get("model", {})
        training_config["model_name_or_path"] = model_section.get(
            "name_or_path", "deepseek-ai/deepseek-math-7b-base"
        )

    if args.output_dir:
        training_config["output_dir"] = args.output_dir

    trainer = CRPSTrainer(config=training_config)
    trainer.train(data_path=args.data_path)


if __name__ == "__main__":
    main()
