"""SFT dataset and data collation for the CRPS training pipeline.

Loads JSONL data with {"problem": ..., "solution": ...} entries and
tokenizes them for causal-LM supervised fine-tuning, masking prompt
tokens so that loss is computed only on the solution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CRPSDataset(Dataset):
    """Dataset for CRPS supervised fine-tuning.

    Each sample is a (problem, solution) pair loaded from a JSONL file.
    During tokenization the prompt portion is masked (labels = -100) so
    that the training loss is computed exclusively on solution tokens.

    Args:
        data_path: Path to a JSONL file where each line contains
            ``{"problem": str, "solution": str}``.
        tokenizer: A HuggingFace tokenizer instance.
        max_seq_length: Maximum token length; sequences are truncated.
    """

    PROMPT_TEMPLATE = "Problem: {problem}\n\nSolution: {solution}"

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_seq_length: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._load_data(data_path)
        logger.info("Loaded %d samples from %s", len(self.data), data_path)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_data(data_path: str) -> list[dict[str, str]]:
        """Read JSONL file and return list of dicts."""
        samples: list[dict[str, str]] = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line %d: %s", line_no, exc)
                    continue
                if "problem" not in entry or "solution" not in entry:
                    logger.warning(
                        "Skipping line %d: missing 'problem' or 'solution' key",
                        line_no,
                    )
                    continue
                samples.append(entry)
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize a single sample with prompt masking.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, and ``labels``
            tensors.  Prompt tokens have ``labels`` set to -100.
        """
        sample = self.data[idx]
        problem: str = sample["problem"]
        solution: str = sample["solution"]

        # ----- Build prompt and full text -----
        # Use DeepSeek official prompt format
        user_content = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            prompt_messages = [
                {"role": "user", "content": user_content},
            ]
            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": solution},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
        else:
            prompt_text = f"{user_content}\n"
            full_text = f"{user_content}\n{solution}"

        # ----- Tokenize -----
        prompt_enc = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length - 1,  # leave room for EOS
        )

        # Append EOS token so model learns when to stop
        eos_id = self.tokenizer.eos_token_id
        full_ids = full_enc["input_ids"] + [eos_id]
        full_mask = full_enc["attention_mask"] + [1]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.tensor(full_mask, dtype=torch.long)

        # Mask prompt tokens in labels
        labels = input_ids.clone()
        prompt_length = min(len(prompt_enc["input_ids"]), len(labels))
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollator:
    """Pads a batch of tokenized samples to uniform length.

    Handles both left- and right-padding depending on the tokenizer's
    ``padding_side`` configuration.

    Args:
        tokenizer: HuggingFace tokenizer (used for ``pad_token_id`` and
            ``padding_side``).
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id: int = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        self.padding_side: str = getattr(tokenizer, "padding_side", "right")

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate and pad a list of feature dicts.

        Args:
            features: Each element has ``input_ids``, ``attention_mask``,
                and ``labels`` tensors (variable length).

        Returns:
            Batch dict with padded tensors.
        """
        max_len = max(f["input_ids"].size(0) for f in features)

        input_ids_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        for feat in features:
            seq_len = feat["input_ids"].size(0)
            pad_len = max_len - seq_len

            if pad_len == 0:
                input_ids_list.append(feat["input_ids"])
                attention_mask_list.append(feat["attention_mask"])
                labels_list.append(feat["labels"])
                continue

            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            pad_attn = torch.zeros(pad_len, dtype=torch.long)
            pad_labels = torch.full((pad_len,), -100, dtype=torch.long)

            if self.padding_side == "left":
                input_ids_list.append(torch.cat([pad_ids, feat["input_ids"]]))
                attention_mask_list.append(torch.cat([pad_attn, feat["attention_mask"]]))
                labels_list.append(torch.cat([pad_labels, feat["labels"]]))
            else:
                input_ids_list.append(torch.cat([feat["input_ids"], pad_ids]))
                attention_mask_list.append(torch.cat([feat["attention_mask"], pad_attn]))
                labels_list.append(torch.cat([feat["labels"], pad_labels]))

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
