"""Post-hoc verification filter for synthesized reasoning paths.

Checks synthesized answers against ground truth using mathematical
equivalence and produces filtered training datasets.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Optional

from crps.synthesis.synthesizer import SynthesizedPath
from crps.utils.math_verify import MathVerifier

logger = logging.getLogger(__name__)


class SynthesisVerifier:
    """Verify synthesized paths against ground-truth answers.

    Args:
        math_verifier: A :class:`MathVerifier` instance for SymPy-based
            equivalence checking.  If ``None``, a default instance is
            created.
    """

    def __init__(self, math_verifier: Optional[MathVerifier] = None) -> None:
        self.math_verifier = math_verifier or MathVerifier()

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def verify(
        self,
        path: SynthesizedPath,
        ground_truth: str,
    ) -> SynthesizedPath:
        """Verify a single synthesized path against *ground_truth*.

        Updates ``path.is_verified`` in-place and returns the path.

        Args:
            path: The :class:`SynthesizedPath` to verify.
            ground_truth: The reference / gold answer string.

        Returns:
            The same :class:`SynthesizedPath` with ``is_verified``
            updated.
        """
        if path.final_answer is None:
            path.is_verified = False
        else:
            path.is_verified = self.math_verifier.verify(
                path.final_answer, ground_truth
            )
        return path

    def filter_verified(
        self,
        paths: list[SynthesizedPath],
        ground_truths: dict[str, str],
    ) -> list[SynthesizedPath]:
        """Verify all paths and return only those that are correct.

        Args:
            paths: List of :class:`SynthesizedPath` instances.
            ground_truths: Mapping from problem statement to the
                expected answer string.

        Returns:
            A list containing only the verified-correct paths.
        """
        verified: list[SynthesizedPath] = []
        num_no_gt = 0

        for path in paths:
            gt = ground_truths.get(path.problem)
            if gt is None:
                num_no_gt += 1
                continue
            self.verify(path, gt)
            if path.is_verified:
                verified.append(path)

        num_total = len(paths)
        num_verified = len(verified)
        num_failed = num_total - num_verified - num_no_gt

        logger.info(
            "Verification stats: %d total, %d passed (%.1f%%), "
            "%d failed, %d missing ground truth",
            num_total,
            num_verified,
            (num_verified / num_total * 100) if num_total else 0.0,
            num_failed,
            num_no_gt,
        )

        return verified

    @staticmethod
    def build_dataset(
        verified_paths: list[SynthesizedPath],
        target_size: int,
        output_path: str,
    ) -> None:
        """Build a JSONL training dataset from verified paths.

        Uniformly samples *target_size* examples (with replacement if
        the pool is smaller) and writes them as newline-delimited JSON.

        Each line has the format::

            {"problem": "...", "solution": "..."}

        where ``solution`` is the concatenation of the step strings.

        Args:
            verified_paths: List of verified :class:`SynthesizedPath`
                instances.
            target_size: Number of examples to include in the dataset.
            output_path: File path for the output JSONL file.
        """
        if not verified_paths:
            logger.warning("No verified paths available; writing empty dataset.")
            with open(output_path, "w", encoding="utf-8") as f:
                pass
            return

        if len(verified_paths) >= target_size:
            sampled = random.sample(verified_paths, target_size)
        else:
            sampled = random.choices(verified_paths, k=target_size)
            logger.info(
                "Pool size (%d) < target_size (%d); sampling with replacement.",
                len(verified_paths),
                target_size,
            )

        with open(output_path, "w", encoding="utf-8") as f:
            for path in sampled:
                solution = "\n".join(
                    f"Step {i}: {step}"
                    for i, step in enumerate(path.steps, start=1)
                )
                record = {"problem": path.problem, "solution": solution}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Dataset written to %s: %d examples from %d verified paths",
            output_path,
            target_size,
            len(verified_paths),
        )
