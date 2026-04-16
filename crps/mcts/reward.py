"""Reward functions for MCTS trajectory evaluation."""

from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class VerifierInterface(Protocol):
    """Minimal interface expected from an answer verifier."""

    def verify(self, predicted: str, ground_truth: str) -> bool:
        """Return ``True`` if *predicted* matches *ground_truth*."""
        ...


class MathRewardFunction:
    """Binary reward for mathematical reasoning trajectories.

    Extracts the predicted answer from the trajectory text and delegates
    equivalence checking to an external *verifier*.

    Args:
        verifier: An object implementing
            :class:`VerifierInterface` — its ``verify`` method is
            called with the extracted prediction and the ground-truth
            answer.

    Example::

        reward_fn = MathRewardFunction(verifier=my_verifier)
        score = reward_fn(trajectory_text, ground_truth="42")
    """

    def __init__(self, verifier: VerifierInterface) -> None:
        self.verifier = verifier

    def __call__(self, trajectory_text: str, ground_truth: str) -> float:
        """Score a completed trajectory.

        Args:
            trajectory_text: The full chain-of-thought text produced
                during simulation.
            ground_truth: The reference answer to compare against.

        Returns:
            ``1.0`` if the extracted answer matches *ground_truth*
            according to the verifier, ``0.0`` otherwise.
        """
        predicted = self.extract_answer(trajectory_text)
        if predicted is None:
            return 0.0
        return 1.0 if self.verifier.verify(predicted, ground_truth) else 0.0

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        r"""Extract the final answer from *text*.

        Extraction strategy (in priority order):

        1. Content inside ``\boxed{...}`` (last occurrence wins).
           Handles nested braces.
        2. The last standalone numerical value (int or float, possibly
           negative) found in the text.

        Args:
            text: The trajectory / completion text.

        Returns:
            The extracted answer string, or ``None`` if nothing could
            be extracted.
        """
        # --- Strategy 1: \boxed{...} ---
        boxed = _extract_last_boxed(text)
        if boxed is not None:
            return boxed.strip()

        # --- Strategy 2: last number ---
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_last_boxed(text: str) -> Optional[str]:
    r"""Return the content of the last ``\boxed{...}`` in *text*.

    Handles one level of nested braces so that expressions like
    ``\boxed{\frac{1}{2}}`` are captured correctly.
    """
    # Find all \boxed occurrences and take the last one
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()  # position right after the opening '{'
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        return None

    return text[start : pos - 1]
