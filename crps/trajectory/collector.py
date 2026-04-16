"""Extract and stratify trajectories from MCTS search results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from crps.mcts.node import MCTSTree, Node


class _AnswerExtractor(Protocol):
    """Structural type for answer extractors."""

    def extract(self, text: str) -> Optional[str]: ...


class _Verifier(Protocol):
    """Structural type for answer verifiers."""

    def verify(self, predicted: str, ground_truth: str) -> bool: ...


@dataclass
class Trajectory:
    """A single complete reasoning trajectory extracted from an MCTS tree.

    Attributes:
        steps: Reasoning steps from root to terminal node.
        is_correct: Whether the final answer matches the ground truth.
        reward: Terminal reward (0.0 or 1.0).
        visit_count: Visit count at the terminal node.
        length: Number of reasoning steps.
        max_q_value: Maximum Q-value among nodes on this path.
        final_answer: Extracted final answer, or ``None``.
    """

    steps: list[str]
    is_correct: bool
    reward: float
    visit_count: int
    length: int
    max_q_value: float
    final_answer: str | None


def _max_q_along_path(terminal_node: Node) -> float:
    """Walk from *terminal_node* to the root and return the max Q-value."""
    max_q = float("-inf")
    node: Node | None = terminal_node
    while node is not None:
        if node.visit_count > 0:
            max_q = max(max_q, node.q_value)
        node = node.parent
    return max_q if max_q != float("-inf") else 0.0


class TrajectoryCollector:
    """Collects and stratifies trajectories from MCTS search trees."""

    def collect(
        self,
        tree: MCTSTree,
        ground_truth: str,
        answer_extractor: Any,
        verifier: Any,
    ) -> list[Trajectory]:
        """Extract all complete trajectories from an MCTS tree.

        For every terminal node the method builds a :class:`Trajectory`,
        extracts the final answer from the last reasoning step, and
        verifies correctness against *ground_truth*.

        Args:
            tree: The MCTS tree to extract trajectories from.
            ground_truth: The reference answer for the problem.
            answer_extractor: Object with an ``extract(text) -> str | None``
                method (e.g. :class:`~crps.utils.math_verify.AnswerExtractor`).
            verifier: Object with a ``verify(predicted, ground_truth) -> bool``
                method (e.g. :class:`~crps.utils.math_verify.MathVerifier`).

        Returns:
            List of :class:`Trajectory` objects, one per terminal node.
        """
        raw_trajectories = tree.get_all_trajectories()
        trajectories: list[Trajectory] = []

        for steps, terminal_node in raw_trajectories:
            # Build the full text from the trajectory steps for answer extraction
            full_text = "\n".join(steps) if steps else ""
            final_answer = answer_extractor.extract(full_text)

            is_correct = (
                verifier.verify(final_answer, ground_truth)
                if final_answer is not None
                else False
            )

            reward = terminal_node.reward if terminal_node.reward is not None else 0.0

            trajectories.append(
                Trajectory(
                    steps=steps,
                    is_correct=is_correct,
                    reward=reward,
                    visit_count=terminal_node.visit_count,
                    length=len(steps),
                    max_q_value=_max_q_along_path(terminal_node),
                    final_answer=final_answer,
                )
            )

        return trajectories

    def stratify(
        self, trajectories: list[Trajectory]
    ) -> dict[str, list[Trajectory]]:
        """Stratify trajectories into contrastive groups.

        Groups:
            - ``"gold_positive"``: Correct trajectories (candidates for
              the positive anchor).
            - ``"hard_negative"``: Incorrect trajectories.
            - ``"soft_negative"``: Correct but sub-optimal trajectories
              (longer than the shortest correct trajectory).

        Args:
            trajectories: List of trajectories to stratify.

        Returns:
            Dict mapping group name to list of trajectories.
        """
        correct = [t for t in trajectories if t.is_correct]
        incorrect = [t for t in trajectories if not t.is_correct]

        if correct:
            min_length = min(t.length for t in correct)
            gold_positive = [t for t in correct if t.length == min_length]
            soft_negative = [t for t in correct if t.length > min_length]
        else:
            gold_positive = []
            soft_negative = []

        return {
            "gold_positive": gold_positive,
            "hard_negative": incorrect,
            "soft_negative": soft_negative,
        }

    def select_positive_anchor(
        self, correct_trajectories: list[Trajectory]
    ) -> Trajectory:
        """Select the best correct trajectory as the positive anchor τ+.

        Selection criteria (in priority order):
            1. Minimal number of steps (shortest trajectory).
            2. Highest ``max_q_value`` among ties.

        Args:
            correct_trajectories: Non-empty list of correct trajectories.

        Returns:
            The selected positive anchor trajectory.

        Raises:
            ValueError: If *correct_trajectories* is empty.
        """
        if not correct_trajectories:
            raise ValueError("Cannot select anchor from empty trajectory list.")

        return min(
            correct_trajectories,
            key=lambda t: (t.length, -t.max_q_value),
        )
