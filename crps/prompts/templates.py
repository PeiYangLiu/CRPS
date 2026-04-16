"""All prompt templates used in the CRPS pipeline.

Templates follow the paper's Appendix (Section 9, Prompts and Instructions).
"""

from __future__ import annotations

from crps.utils.segmentation import segment_steps


class CRPSPrompts:
    """Centralised repository of every prompt template in the CRPS framework."""

    # ------------------------------------------------------------------
    # MCTS prompts
    # ------------------------------------------------------------------

    MCTS_EXPANSION_SYSTEM = (
        "You are a mathematical reasoning assistant. "
        "Given a problem and the reasoning steps so far, generate the next "
        "logical reasoning step. Be precise and show your work."
    )

    MCTS_SIMULATION_SYSTEM = (
        "You are a mathematical reasoning assistant. "
        "Given a problem and partial reasoning, complete the full solution "
        "to arrive at a final answer. Be thorough and precise."
    )

    # ------------------------------------------------------------------
    # Contrastive analysis prompts (Section 9 of the paper)
    # ------------------------------------------------------------------

    CONTRASTIVE_ANALYSIS_SYSTEM = (
        "You are an expert Mathematical Reasoning Analyst. Your task is to "
        "perform a dual-granularity contrastive analysis between two "
        "reasoning trajectories for the same mathematical problem. One "
        "trajectory arrives at the correct answer (Trajectory A) and the "
        "other arrives at an incorrect answer (Trajectory B). You must "
        "identify where and why the trajectories diverge, analyze both "
        "local step-level logic and global strategic differences, and "
        "synthesize actionable guidance."
    )

    CONTRASTIVE_ANALYSIS_USER = """\
**Problem:**
{problem}

**Trajectory A (Correct):**
{positive_trajectory}

**Trajectory B (Incorrect):**
{negative_trajectory}

Perform a dual-granularity contrastive analysis of the two trajectories above. \
Provide your analysis as a JSON object with the following structure:

{{
  "divergence_step_index": <int>,
  "local_step_critique": {{
    "trajectory_a_logic": "<description of Trajectory A's reasoning at the divergence point>",
    "trajectory_b_logic": "<description of Trajectory B's reasoning at the divergence point>",
    "critique_of_difference": "<precise explanation of why Trajectory B's step is incorrect and how Trajectory A's step is correct>"
  }},
  "global_strategic_analysis": "<high-level comparison of the overall strategies employed by each trajectory, including differences in approach, method selection, and problem decomposition>",
  "synthesized_guidance": {{
    "success_pattern": "<the key reasoning pattern or strategy from Trajectory A that led to the correct answer>",
    "failure_mode_to_avoid": "<the specific reasoning pitfall or error pattern from Trajectory B that should be avoided>"
  }}
}}

Return ONLY the JSON object, with no additional text."""

    # ------------------------------------------------------------------
    # Synthesis prompts (Section 9 of the paper)
    # ------------------------------------------------------------------

    SYNTHESIS_SYSTEM = (
        "You are an advanced Mathematical Reasoning Engine. Your task is to "
        "solve a mathematical problem by generating a step-by-step solution "
        "that is informed by prior contrastive analysis of correct and "
        "incorrect reasoning paths.\n\n"
        "KEY REQUIREMENTS:\n"
        "1. INCORPORATE contrastive insights naturally into your reasoning. "
        "At critical steps, briefly explain WHY you chose the correct approach "
        "and what common mistake to avoid. For example: 'A common mistake is "
        "to use C(6,2) here, which treats identical objects as distinct. "
        "Instead, we enumerate cases.' This is the core value of CRPS.\n"
        "2. Do NOT use meta-language like 'the critique suggests', 'following "
        "the success pattern', or 'as identified in the analysis'. Write as "
        "if YOU discovered these insights while solving.\n"
        "3. Match the target model's formatting style (bold headers, LaTeX, "
        "numbered steps)."
    )

    SYNTHESIS_USER = """\
**Problem:**
{problem}

**Contrastive Insights (weave naturally into solution, do NOT reference directly):**
- Why the correct approach works: {success_pattern}
- Common mistake to avoid: {failure_mode_to_avoid}
- Key difference at step {divergence_step_index}: {critique_of_difference}
- Strategic overview: {global_strategic_analysis}

**Target Output Style:**
{style_example}

Solve step by step. At the critical decision point, naturally explain why \
you chose your approach and what pitfall you are avoiding — as if you \
discovered this yourself. Do NOT say "the critique" or "the analysis".

Format:
Step 1: ...
Step 2: ...
...
Final Answer: \\boxed{{...}}"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def format_trajectory(steps: list[str]) -> str:
        """Format a list of reasoning steps as a numbered string.

        If any step contains embedded sub-steps (e.g. from a raw LLM
        completion), it is first segmented using the hierarchical
        protocol described in Appendix (Step Segmentation Protocol).

        Args:
            steps: List of reasoning step strings.

        Returns:
            A single string with steps numbered starting from 1.
        """
        # Re-segment: each input "step" might contain multiple semantic
        # reasoning acts (e.g. from a simulation completion).
        flat_steps: list[str] = []
        for step in steps:
            sub = segment_steps(step)
            if sub:
                flat_steps.extend(sub)
            else:
                flat_steps.append(step)
        return "\n".join(
            f"Step {i}: {s}" for i, s in enumerate(flat_steps, start=1)
        )

    @classmethod
    def mcts_expansion_prompt(cls, problem: str, current_steps: list[str]) -> str:
        """Build the prompt for MCTS node expansion.

        Asks the LLM to generate the *next* reasoning step given the problem
        and the partial solution so far.

        Args:
            problem: The mathematical problem statement.
            current_steps: Reasoning steps accumulated so far.

        Returns:
            The formatted prompt string.
        """
        steps_text = cls.format_trajectory(current_steps) if current_steps else "No steps yet."
        return (
            f"{cls.MCTS_EXPANSION_SYSTEM}\n\n"
            f"**Problem:**\n{problem}\n\n"
            f"**Reasoning so far:**\n{steps_text}\n\n"
            "Generate the next reasoning step. "
            "Provide ONLY the next step, nothing else."
        )

    @classmethod
    def mcts_simulation_prompt(cls, problem: str, current_steps: list[str]) -> str:
        """Build the prompt for MCTS rollout / simulation.

        Asks the LLM to complete the reasoning from the current state all
        the way to a final answer.

        Args:
            problem: The mathematical problem statement.
            current_steps: Reasoning steps accumulated so far.

        Returns:
            The formatted prompt string.
        """
        steps_text = cls.format_trajectory(current_steps) if current_steps else "No steps yet."
        return (
            f"{cls.MCTS_SIMULATION_SYSTEM}\n\n"
            f"**Problem:**\n{problem}\n\n"
            f"**Reasoning so far:**\n{steps_text}\n\n"
            "Complete the solution from here. Show all remaining steps and "
            "provide the final answer in the format: Final Answer: \\boxed{...}"
        )

    @classmethod
    def contrastive_analysis_prompt(
        cls,
        problem: str,
        positive_trajectory: str,
        negative_trajectory: str,
    ) -> tuple[str, str]:
        """Build the dual-granularity contrastive analysis prompt pair.

        Args:
            problem: The mathematical problem statement.
            positive_trajectory: The correct (positive) reasoning trajectory.
            negative_trajectory: The incorrect (negative) reasoning trajectory.

        Returns:
            A ``(system_prompt, user_prompt)`` tuple.
        """
        user_prompt = cls.CONTRASTIVE_ANALYSIS_USER.format(
            problem=problem,
            positive_trajectory=positive_trajectory,
            negative_trajectory=negative_trajectory,
        )
        return cls.CONTRASTIVE_ANALYSIS_SYSTEM, user_prompt

    # Default style example (DeepSeekMath style)
    DEFAULT_STYLE_EXAMPLE = (
        "Write in a clean, structured mathematical style. Use numbered steps "
        "with bold headers like '**Step 1: ...**'. Show calculations using "
        "LaTeX notation (e.g., \\(3 \\times 12 = 36\\)). Be direct and "
        "concise without meta-commentary. Example:\n"
        "Step 1: **Determine the total number of slices.**\n"
        "Kim buys 3 pizzas with 12 slices each. The total number of slices "
        "is \\(3 \\times 12 = 36\\).\n"
        "Step 2: **Calculate the cost per slice.**\n"
        "The total cost is $72 for 36 slices, so the cost per slice is "
        "\\(\\frac{72}{36} = 2\\) dollars."
    )

    @classmethod
    def synthesis_prompt(
        cls,
        problem: str,
        critique: dict,
        style_example: str | None = None,
    ) -> tuple[str, str]:
        """Build the pattern-informed path synthesis prompt pair.

        Args:
            problem: The mathematical problem statement.
            critique: Dictionary with keys from the contrastive analysis
                output (``global_strategic_analysis``,
                ``divergence_step_index``, ``local_step_critique``,
                ``synthesized_guidance``).

        Returns:
            A ``(system_prompt, user_prompt)`` tuple.
        """
        local = critique.get("local_step_critique", {})
        guidance = critique.get("synthesized_guidance", {})

        user_prompt = cls.SYNTHESIS_USER.format(
            problem=problem,
            global_strategic_analysis=critique.get("global_strategic_analysis", "N/A"),
            divergence_step_index=critique.get("divergence_step_index", "N/A"),
            critique_of_difference=local.get("critique_of_difference", "N/A"),
            success_pattern=guidance.get("success_pattern", "N/A"),
            failure_mode_to_avoid=guidance.get("failure_mode_to_avoid", "N/A"),
            style_example=style_example or cls.DEFAULT_STYLE_EXAMPLE,
        )
        return cls.SYNTHESIS_SYSTEM, user_prompt
