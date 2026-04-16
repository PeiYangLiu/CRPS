"""MCTS (Monte Carlo Tree Search) module for chain-of-thought reasoning.

This package provides the core search infrastructure used by CRPS to
explore reasoning trajectories and collect contrastive training pairs.

Public API
----------
.. autosummary::

    Node
    MCTSTree
    MCTSSearch
    MathRewardFunction
"""

from crps.mcts.node import MCTSTree, Node
from crps.mcts.reward import MathRewardFunction
from crps.mcts.search import MCTSSearch

__all__ = [
    "Node",
    "MCTSTree",
    "MCTSSearch",
    "MathRewardFunction",
]