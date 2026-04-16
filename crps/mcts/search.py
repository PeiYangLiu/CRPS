"""UCT-based Monte Carlo Tree Search for chain-of-thought reasoning.

The :class:`MCTSSearch` orchestrator drives the four canonical MCTS
phases — *selection*, *expansion*, *simulation*, and
*backpropagation* — using an external LLM for step generation and an
external reward function for terminal evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from crps.mcts.node import MCTSTree, Node
from crps.utils.segmentation import segment_steps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for dependency injection
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMInterface(Protocol):
    """Minimal interface expected from the LLM inference backend."""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Generate *n* completions for *prompt*."""
        ...


# Type alias for reward functions:  (trajectory_text, **kwargs) -> float
RewardFn = Callable[..., float]

# ---------------------------------------------------------------------------
# Default prompt templates — few-shot CoT format for base models
# ---------------------------------------------------------------------------

_FEW_SHOT_PREFIX = (
    # GSM8K-level example
    "Problem: There are 15 trees in the grove. Grove workers will plant trees today. "
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
    "Solution: There are 15 trees originally. Then there were 21 trees after some more were planted. "
    "So there must have been 21 - 15 = 6 trees planted. The answer is \\boxed{{6}}.\n\n"
    # Algebra (Level 4)
    "Problem: Consider the given functions: $f(x) = 5x^2 - \\frac{{1}}{{x}} + 3$ and $g(x) = x^2 - k$. "
    "If $f(2) - g(2) = 2$, what is the value of $k$?\n"
    "Solution: We substitute $f(2) = 5(2)^2 - \\frac{{1}}{{2}} + 3 = \\frac{{45}}{{2}}$ and "
    "$g(2) = (2)^2 - k = 4 - k$. So $f(2) - g(2) = 2$ gives us $\\frac{{45}}{{2}} - 4 + k = 2$. "
    "Solving for $k$, we find $k = \\frac{{4}}{{2}} - \\frac{{45}}{{2}} + \\frac{{8}}{{2}}$ "
    "so $\\boxed{{k = \\frac{{-33}}{{2}}}}$.\n\n"
    # Counting & Probability (Level 4)
    "Problem: Two fair 6-sided dice are rolled. What is the probability at least one of the dice shows a 1?\n"
    "Solution: There are 5 ways in which the first roll is not 1, and 5 ways in which the second roll is not 1, "
    "so there are $5 \\times 5 = 25$ ways in which neither die shows 1. Therefore there are $36 - 25 = 11$ ways "
    "in which one or both dice show 1. So the probability is $\\boxed{{\\dfrac{{11}}{{36}}}}$.\n\n"
    # Geometry (Level 3)
    "Problem: The measure of each exterior angle of a regular polygon is $30$ degrees. "
    "What is the sum of the measures of the interior angles, in degrees?\n"
    "Solution: The sum of the exterior angles of a polygon is $360^\\circ$. If each exterior angle is $30^\\circ$, "
    "then the polygon has $\\frac{{360}}{{30}} = 12$ sides. The sum of the interior angles of an $n$-sided polygon "
    "is $180(n-2)$, so for 12 sides, the sum is $180(12-2) = \\boxed{{1800}}$ degrees.\n\n"
    # Number Theory (Level 3)
    "Problem: If the seven-digit number $854n526$ is divisible by $11$, what is $n$?\n"
    "Solution: A number is divisible by $11$ if and only if the alternating sum of digits is a multiple of $11$. "
    "The sum of the 1st, 3rd, 5th, 7th digits is $8+4+5+6=23$. The sum of the 2nd, 4th, 6th digits is $5+n+2=7+n$. "
    "Thus $23-(7+n)=16-n$ must be a multiple of $11$. This is satisfied only by $n=\\boxed{{5}}$.\n\n"
    # Precalculus (Level 4)
    "Problem: A line is expressed in the form $\\begin{{pmatrix}} -2 \\\\ -5 \\end{{pmatrix}} \\cdot "
    "\\left( \\begin{{pmatrix}} x \\\\ y \\end{{pmatrix}} - \\begin{{pmatrix}} 1 \\\\ 11 \\end{{pmatrix}} \\right) = 0.$ "
    "Find the equation in the form $y = mx + b$ and enter $(m,b)$.\n"
    "Solution: Expanding, we get $(-2)(x-1) + (-5)(y-11) = 0$. Solving for $y$, we find "
    "$y = -\\frac{{2}}{{5}}x + \\frac{{57}}{{5}}$. Thus $(m,b) = \\boxed{{\\left(-\\frac{{2}}{{5}}, \\frac{{57}}{{5}}\\right)}}$.\n\n"
)

_EXPANSION_PROMPT = (
    "Problem: {problem}\n"
    "Write the next single step in solving this problem. Do NOT write the full solution, just one reasoning step.\n"
    "Solution so far:{steps}\n"
    "Next step:"
)

_SIMULATION_PROMPT = (
    _FEW_SHOT_PREFIX +
    "Problem: {problem}\n"
    "Solution:{steps}"
)


# ---------------------------------------------------------------------------
# MCTSSearch
# ---------------------------------------------------------------------------

class MCTSSearch:
    """Upper Confidence bound applied to Trees (UCT) search.

    Args:
        llm: An object exposing a ``generate(prompt, n)`` method.
        reward_fn: A callable that scores a completed trajectory.
        config: Optional dict overriding default hyper-parameters.

    Configuration keys (with defaults):

    ============== =========== ======================================
    Key            Default     Description
    ============== =========== ======================================
    c_puct         1.4         Exploration constant in UCT.
    max_depth      16          Maximum reasoning depth.
    max_actions    3           Number of candidate next-steps per
                               expansion.
    num_rollouts   10          Number of MCTS iterations.
    temperature    0.7         LLM sampling temperature (informational
                               — the LLM object is responsible for
                               using it).
    ============== =========== ======================================
    """

    def __init__(
        self,
        llm: LLMInterface,
        reward_fn: RewardFn,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        self.llm = llm
        self.reward_fn = reward_fn

        self.c_puct: float = cfg.get("c_puct", 1.4)
        self.max_depth: int = cfg.get("max_depth", 16)
        self.max_actions_per_node: int = cfg.get("max_actions_per_node", 3)
        self.num_rollouts: int = cfg.get("num_rollouts", 10)
        self.temperature: float = cfg.get("temperature", 0.7)
        self.expand_max_tokens: int = cfg.get("expand_max_tokens", 256)
        self.simulate_max_tokens: int = cfg.get("simulate_max_tokens", 2048)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, problem: str, **reward_kwargs) -> MCTSTree:
        """Run MCTS on *problem* and return the resulting search tree.

        Args:
            problem: The problem statement.
            **reward_kwargs: Extra keyword arguments forwarded to the
                reward function (e.g. ``ground_truth``).

        Returns:
            An :class:`MCTSTree` containing all explored trajectories.
        """
        tree = MCTSTree(problem=problem, max_depth=self.max_depth)

        for i in range(self.num_rollouts):
            logger.debug("Rollout %d/%d", i + 1, self.num_rollouts)

            # 1. Selection
            leaf = self._select(tree.root)

            # 2. Expansion (skip if terminal or at max depth)
            if not leaf.is_terminal and leaf.depth < self.max_depth:
                leaf = self._expand(leaf, problem)

            # 3. Simulation
            reward = self._simulate(leaf, problem, **reward_kwargs)

            # 4. Backpropagation
            self._backpropagate(leaf, reward)

        return tree

    def batch_search(
        self,
        problems: List[str],
        ground_truths: List[str],
        reward_fn_factory: Optional[Callable] = None,
    ) -> List[MCTSTree]:
        """Run MCTS on multiple problems with batched LLM calls.

        Interleaves rollouts across problems so that LLM calls from
        different trees can be batched together for higher throughput.

        Args:
            problems: List of problem statements.
            ground_truths: Corresponding ground truth answers.
            reward_fn_factory: Optional callable that takes a ground_truth
                and returns a reward function.  Defaults to using
                ``self.reward_fn`` with ``ground_truth=gt``.

        Returns:
            A list of :class:`MCTSTree`, one per problem.
        """
        n_problems = len(problems)
        trees = [MCTSTree(problem=p, max_depth=self.max_depth) for p in problems]

        for rollout_idx in range(self.num_rollouts):
            logger.debug("Batch rollout %d/%d for %d problems",
                         rollout_idx + 1, self.num_rollouts, n_problems)

            # 1. Selection for all trees
            leaves = [self._select(t.root) for t in trees]

            # 2. Batch expansion — collect prompts, call LLM once
            expand_indices = []
            expand_prompts = []
            for i, (leaf, problem) in enumerate(zip(leaves, problems)):
                if not leaf.is_terminal and leaf.depth < self.max_depth:
                    steps_text = self._format_steps(leaf.state)
                    prompt = _EXPANSION_PROMPT.format(problem=problem, steps=steps_text)
                    expand_indices.append(i)
                    expand_prompts.append(prompt)

            if expand_prompts:
                # Batch generate all expansions at once
                all_candidates = self.llm.batch_generate(
                    expand_prompts,
                    n=self.max_actions_per_node,
                    max_tokens=self.expand_max_tokens,
                )
                for idx, candidates in zip(expand_indices, all_candidates):
                    leaf = leaves[idx]
                    seen: set = set()
                    unique_actions: List[str] = []
                    for action in candidates:
                        action = action.strip()
                        if not action or action in seen:
                            continue
                        sub_steps = segment_steps(action, max_step_tokens=self.expand_max_tokens)
                        action = sub_steps[0].strip() if sub_steps else action
                        if action and action not in seen:
                            seen.add(action)
                            unique_actions.append(action)
                    if not unique_actions:
                        leaf.is_terminal = True
                        leaf.reward = 0.0
                    else:
                        for action in unique_actions:
                            new_state = leaf.state + [action]
                            is_terminal = len(new_state) >= self.max_depth
                            leaf.add_child(
                                action=action, state=new_state,
                                is_terminal=is_terminal, reward=None,
                            )
                        leaves[idx] = leaf.children[0]

            # 3. Batch simulation
            sim_indices = []
            sim_prompts = []
            for i, (leaf, problem) in enumerate(zip(leaves, problems)):
                if not (leaf.is_terminal and leaf.reward is not None):
                    steps_text = self._format_steps(leaf.state)
                    prompt = _SIMULATION_PROMPT.format(problem=problem, steps=steps_text)
                    sim_indices.append(i)
                    sim_prompts.append(prompt)

            rewards = [0.0] * n_problems
            # Pre-fill cached rewards
            for i, leaf in enumerate(leaves):
                if leaf.is_terminal and leaf.reward is not None:
                    rewards[i] = leaf.reward

            if sim_prompts:
                all_completions = self.llm.batch_generate(
                    sim_prompts, n=1, max_tokens=self.simulate_max_tokens,
                )
                for idx, completions in zip(sim_indices, all_completions):
                    leaf = leaves[idx]
                    completion = completions[0].strip() if completions else ""
                    steps_text = self._format_steps(leaf.state)
                    full_traj = steps_text + "\n" + completion if completion else steps_text
                    reward = self.reward_fn(full_traj, ground_truth=ground_truths[idx])
                    leaf.is_terminal = True
                    leaf.reward = reward
                    rewards[idx] = reward

            # 4. Backpropagation for all
            for leaf, reward in zip(leaves, rewards):
                self._backpropagate(leaf, reward)

        return trees

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: Node) -> Node:
        """Descend the tree using UCT until reaching a non-fully-expanded
        or terminal node.

        Args:
            node: The node to start selection from (usually root).

        Returns:
            A leaf :class:`Node` suitable for expansion.
        """
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child(c_puct=self.c_puct)
        return node

    def _expand(self, node: Node, problem: str) -> Node:
        """Expand *node* by generating candidate next reasoning steps.

        Uses the LLM to produce ``max_actions_per_node`` candidate
        steps.  Each unique step is added as a child.  Returns one of
        the newly created children (the first one) for simulation.

        Args:
            node: The node to expand.
            problem: The problem statement.

        Returns:
            A newly created child :class:`Node`.
        """
        steps_text = self._format_steps(node.state)
        prompt = _EXPANSION_PROMPT.format(problem=problem, steps=steps_text)

        candidates: List[str] = self.llm.generate(
            prompt, n=self.max_actions_per_node, max_tokens=self.expand_max_tokens
        )

        # Deduplicate while preserving order.
        # Apply step segmentation: if the LLM produced multi-step output
        # in a single response, keep only the first semantic step.
        seen: set = set()
        unique_actions: List[str] = []
        for action in candidates:
            action = action.strip()
            if not action or action in seen:
                continue
            # Segment and take the first semantic reasoning act
            sub_steps = segment_steps(action, max_step_tokens=self.expand_max_tokens)
            action = sub_steps[0].strip() if sub_steps else action
            if action and action not in seen:
                seen.add(action)
                unique_actions.append(action)

        if not unique_actions:
            # Fallback: mark node terminal with zero reward
            node.is_terminal = True
            node.reward = 0.0
            return node

        for action in unique_actions:
            new_state = node.state + [action]
            is_terminal = len(new_state) >= self.max_depth
            node.add_child(
                action=action,
                state=new_state,
                is_terminal=is_terminal,
                reward=None,
            )

        return node.children[0]

    def _simulate(
        self,
        node: Node,
        problem: str,
        **reward_kwargs,
    ) -> float:
        """Run a simulation (rollout) from *node* to a terminal state.

        If the node is already terminal (either by depth or by having a
        pre-assigned reward) the existing reward is returned.  Otherwise
        the LLM completes the reasoning chain and the reward function
        scores it.

        Args:
            node: The starting node for simulation.
            problem: The problem statement.
            **reward_kwargs: Forwarded to ``self.reward_fn``.

        Returns:
            A scalar reward in [0, 1].
        """
        if node.is_terminal and node.reward is not None:
            return node.reward

        steps_text = self._format_steps(node.state)
        prompt = _SIMULATION_PROMPT.format(problem=problem, steps=steps_text)

        completions: List[str] = self.llm.generate(
            prompt, n=1, max_tokens=self.simulate_max_tokens
        )
        completion = completions[0].strip() if completions else ""

        full_trajectory = steps_text + "\n" + completion if completion else steps_text
        reward: float = self.reward_fn(full_trajectory, **reward_kwargs)

        # Cache reward on the node
        node.is_terminal = True
        node.reward = reward
        return reward

    @staticmethod
    def _backpropagate(node: Node, reward: float) -> None:
        """Propagate *reward* from *node* up to the root.

        Args:
            node: The leaf node where the rollout ended.
            reward: The reward to propagate.
        """
        node.backpropagate(reward)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_steps(steps: List[str]) -> str:
        """Format reasoning steps as a continuation string for few-shot prompts."""
        if not steps:
            return " "
        return " " + " ".join(steps)
