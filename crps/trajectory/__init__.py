"""Trajectory collection and contrastive pair sampling for CRPS.

This package processes MCTS search trees to extract reasoning
trajectories, stratify them by quality, and sample contrastive
pairs for downstream synthesis.
"""

from crps.trajectory.collector import Trajectory, TrajectoryCollector
from crps.trajectory.sampler import ContrastivePair, ContrastivePairSampler

__all__ = [
    "Trajectory",
    "TrajectoryCollector",
    "ContrastivePair",
    "ContrastivePairSampler",
]