"""MCTS Node and Tree data structures for chain-of-thought reasoning."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


class Node:
    """A node in the Monte Carlo Tree Search tree.

    Each node represents a partial reasoning state — a sequence of
    chain-of-thought steps taken so far.  Leaf nodes may be terminal
    (a complete solution) or awaiting expansion.

    Attributes:
        state: Ordered list of reasoning step strings accumulated
            from the root to this node.
        parent: Parent node, or ``None`` for the root.
        children: Child nodes produced by expansion.
        visit_count: Number of times this node has been visited (N).
        total_value: Accumulated reward across all visits (W).
        reward: Terminal reward (``None`` while non-terminal).
        is_terminal: Whether this node represents a completed solution.
        depth: Distance from the root (root has depth 0).
        action: The reasoning step that was appended to the parent's
            state to arrive at this node.  ``None`` for the root.
    """

    def __init__(
        self,
        state: List[str],
        parent: Optional["Node"] = None,
        action: Optional[str] = None,
        is_terminal: bool = False,
        reward: Optional[float] = None,
    ) -> None:
        self.state: List[str] = list(state)
        self.parent: Optional["Node"] = parent
        self.children: List["Node"] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.reward: Optional[float] = reward
        self.is_terminal: bool = is_terminal
        self.action: Optional[str] = action
        self.depth: int = parent.depth + 1 if parent is not None else 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        """Average reward (W / N).  Returns 0.0 for unvisited nodes."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_fully_expanded(self) -> bool:
        """True if this node has been expanded (has at least one child)
        or is terminal."""
        return self.is_terminal or len(self.children) > 0

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    def add_child(self, action: str, state: List[str], **kwargs) -> "Node":
        """Create a child node and attach it to this node.

        Args:
            action: The reasoning step that leads to the new child.
            state: The full reasoning state at the new child.
            **kwargs: Extra keyword arguments forwarded to the
                :class:`Node` constructor (e.g. ``is_terminal``,
                ``reward``).

        Returns:
            The newly created child :class:`Node`.
        """
        child = Node(state=state, parent=self, action=action, **kwargs)
        self.children.append(child)
        return child

    def best_child(self, c_puct: float = 1.4) -> "Node":
        """Select the child with the highest UCT value.

        Uses the Upper Confidence Bound for Trees formula::

            UCT(s) = Q(s) + c_puct * sqrt(ln(N_parent) / N(s))

        Un-visited children receive ``float('inf')`` to guarantee they
        are explored at least once.

        Args:
            c_puct: Exploration constant.

        Returns:
            The child :class:`Node` with the highest UCT score.

        Raises:
            ValueError: If the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select best child from a node with no children.")

        log_parent = math.log(self.visit_count) if self.visit_count > 0 else 0.0

        def _uct(child: "Node") -> float:
            if child.visit_count == 0:
                return float("inf")
            exploitation = child.q_value
            exploration = c_puct * math.sqrt(log_parent / child.visit_count)
            return exploitation + exploration

        return max(self.children, key=_uct)

    def backpropagate(self, reward: float) -> None:
        """Propagate a reward value up to the root.

        Increments ``visit_count`` and adds ``reward`` to
        ``total_value`` for this node and every ancestor.

        Args:
            reward: The reward obtained from a rollout through this
                node's subtree.
        """
        node: Optional["Node"] = self
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    def get_trajectory(self) -> List[str]:
        """Return the sequence of actions from the root to this node.

        Returns:
            A list of action strings (excluding the root, whose action
            is ``None``).
        """
        actions: List[str] = []
        node: Optional["Node"] = self
        while node is not None:
            if node.action is not None:
                actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Node(depth={self.depth}, N={self.visit_count}, "
            f"W={self.total_value:.3f}, Q={self.q_value:.3f}, "
            f"terminal={self.is_terminal}, children={len(self.children)})"
        )


class MCTSTree:
    """Container for a full MCTS search tree rooted at a single problem.

    Attributes:
        root: The root :class:`Node` (empty reasoning state).
        problem: The problem statement being solved.
        max_depth: Maximum reasoning depth allowed.
    """

    def __init__(
        self,
        problem: str,
        max_depth: int = 16,
    ) -> None:
        self.problem: str = problem
        self.max_depth: int = max_depth
        self.root: Node = Node(state=[])

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    def get_all_trajectories(self) -> List[Tuple[List[str], Node]]:
        """Collect every completed trajectory in the tree.

        Returns:
            A list of ``(trajectory_steps, terminal_node)`` tuples for
            every terminal node reachable from the root.
        """
        results: List[Tuple[List[str], Node]] = []
        self._collect_terminal(self.root, results)
        return results

    def get_trajectory_visit_count(self, trajectory: List[str]) -> int:
        """Return the accumulated visit count along *trajectory*.

        Walks the tree following *trajectory* (matching actions) and
        returns the visit count of the deepest matched node.  If the
        trajectory cannot be fully matched, returns the visit count at
        the last matched node.

        Args:
            trajectory: Ordered list of action strings.

        Returns:
            Visit count at the terminal node of the trajectory.
        """
        node = self.root
        for step in trajectory:
            matched = False
            for child in node.children:
                if child.action == step:
                    node = child
                    matched = True
                    break
            if not matched:
                break
        return node.visit_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_terminal(
        self,
        node: Node,
        results: List[Tuple[List[str], Node]],
    ) -> None:
        """Recursively collect terminal nodes via DFS."""
        if node.is_terminal:
            results.append((node.get_trajectory(), node))
            return
        for child in node.children:
            self._collect_terminal(child, results)

    def __repr__(self) -> str:
        trajectories = self.get_all_trajectories()
        return (
            f"MCTSTree(problem={self.problem[:50]!r}..., "
            f"max_depth={self.max_depth}, "
            f"terminal_trajectories={len(trajectories)})"
        )
