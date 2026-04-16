"""Distribution-aware contrastive pair sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from crps.trajectory.collector import Trajectory


@dataclass
class ContrastivePair:
    """A single contrastive training pair.

    Attributes:
        problem: The problem statement.
        positive: The positive anchor trajectory (τ+).
        negative: The negative trajectory (τ-).
        contrast_type: ``"hard"`` when the negative is incorrect, or
            ``"soft"`` when the negative is correct but sub-optimal.
    """

    problem: str
    positive: Trajectory
    negative: Trajectory
    contrast_type: str  # "hard" or "soft"


class ContrastivePairSampler:
    """Sample contrastive pairs from stratified trajectories.

    The sampler produces a configurable mix of *hard* negatives (incorrect
    trajectories, sampled proportionally to their MCTS visit count) and
    *soft* negatives (correct but longer trajectories, sampled uniformly).

    Args:
        k: Number of contrastive pairs to sample per problem.
        hard_ratio: Target proportion of hard negatives (default 0.7).
    """

    def __init__(self, k: int = 10, hard_ratio: float = 0.7) -> None:
        self.k = k
        self.hard_ratio = hard_ratio

    def sample(
        self,
        problem: str,
        stratified: dict[str, list[Trajectory]],
        positive_anchor: Trajectory,
    ) -> list[ContrastivePair]:
        """Sample K contrastive pairs for a single problem.

        Hard negatives are drawn from the ``"hard_negative"`` group with
        probability proportional to their visit count ``N(τ)``.  Soft
        negatives are drawn uniformly from the ``"soft_negative"`` group.

        If one pool is empty or exhausted, the other pool is used to
        fill the remaining quota.

        Args:
            problem: The problem statement.
            stratified: Output of
                :meth:`~crps.trajectory.collector.TrajectoryCollector.stratify`.
            positive_anchor: The positive anchor trajectory (τ+).

        Returns:
            List of up to *k* :class:`ContrastivePair` objects.
        """
        hard_negatives = stratified.get("hard_negative", [])
        soft_negatives = stratified.get("soft_negative", [])

        total_available = len(hard_negatives) + len(soft_negatives)
        if total_available == 0:
            return []

        n_hard_target = int(round(self.k * self.hard_ratio))
        n_soft_target = self.k - n_hard_target

        # Clamp to available pool sizes
        n_hard = min(n_hard_target, len(hard_negatives)) if hard_negatives else 0
        n_soft = min(n_soft_target, len(soft_negatives)) if soft_negatives else 0

        # Fill shortfall from the other pool
        hard_shortfall = n_hard_target - n_hard
        soft_shortfall = n_soft_target - n_soft

        n_soft = min(n_soft + hard_shortfall, len(soft_negatives))
        n_hard = min(n_hard + soft_shortfall, len(hard_negatives))

        # Cap total at k
        if n_hard + n_soft > self.k:
            n_soft = self.k - n_hard

        pairs: list[ContrastivePair] = []
        used_hard: set[int] = set()
        used_soft: set[int] = set()

        for _ in range(n_hard):
            neg, neg_idx = self._sample_hard_negative(hard_negatives, used_hard)
            if neg is None:
                break
            used_hard.add(neg_idx)
            pairs.append(
                ContrastivePair(
                    problem=problem,
                    positive=positive_anchor,
                    negative=neg,
                    contrast_type="hard",
                )
            )

        for _ in range(n_soft):
            neg, neg_idx = self._sample_soft_negative(soft_negatives, used_soft)
            if neg is None:
                break
            used_soft.add(neg_idx)
            pairs.append(
                ContrastivePair(
                    problem=problem,
                    positive=positive_anchor,
                    negative=neg,
                    contrast_type="soft",
                )
            )

        return pairs

    @staticmethod
    def _sample_hard_negative(
        hard_negatives: list[Trajectory],
        used: set[int],
    ) -> tuple[Trajectory | None, int]:
        """Sample a hard negative proportional to visit count, without replacement."""
        available = [(i, t) for i, t in enumerate(hard_negatives) if i not in used]
        if not available:
            return None, -1

        indices, trajs = zip(*available)
        visit_counts = np.array([t.visit_count for t in trajs], dtype=np.float64)

        total = visit_counts.sum()
        if total == 0:
            weights = np.ones_like(visit_counts) / len(visit_counts)
        else:
            weights = visit_counts / total

        pick = int(np.random.choice(len(available), p=weights))
        return trajs[pick], indices[pick]

    @staticmethod
    def _sample_soft_negative(
        soft_negatives: list[Trajectory],
        used: set[int],
    ) -> tuple[Trajectory | None, int]:
        """Uniformly sample a soft negative, without replacement."""
        available = [(i, t) for i, t in enumerate(soft_negatives)
                     if i not in used]
        if not available:
            return None, -1
        pick = random.randrange(len(available))
        return available[pick][1], available[pick][0]
