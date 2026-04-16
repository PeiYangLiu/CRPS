"""Pattern-informed path synthesis conditioned on contrastive critiques.

Given a :class:`ContrastiveCritique`, the synthesizer asks an LLM to
produce a new high-quality reasoning path that follows identified
success patterns and avoids known failure modes.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from crps.analysis.contrastive import ContrastiveCritique
from crps.prompts.templates import CRPSPrompts
from crps.utils.math_verify import AnswerExtractor

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------

@dataclass
class SynthesizedPath:
    """A reasoning path synthesized from contrastive critique guidance.

    Attributes:
        problem: The mathematical problem statement.
        steps: Ordered list of reasoning step strings.
        final_answer: Extracted answer, or ``None`` if extraction failed.
        critique_used: The :class:`ContrastiveCritique` that guided
            synthesis.
        raw_response: Full LLM output for debugging.
        is_verified: Whether the answer has been verified against a
            ground truth.
    """

    problem: str
    steps: list[str]
    final_answer: Optional[str]
    critique_used: ContrastiveCritique
    raw_response: str
    is_verified: bool


# -----------------------------------------------------------------------
# Synthesizer
# -----------------------------------------------------------------------

class PathSynthesizer:
    """Synthesize reasoning paths guided by contrastive critiques.

    Args:
        llm: LLM inference object exposing a
            ``generate_with_system(system, user, n=1)`` method.
        prompts_cls: Prompt templates class; defaults to
            :class:`CRPSPrompts`.
        temperature: Sampling temperature passed through to the LLM.
    """

    def __init__(self, llm, prompts_cls=None, temperature: float = 0.0) -> None:
        self.llm = llm
        self.prompts_cls = prompts_cls or CRPSPrompts
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def synthesize(self, critique: ContrastiveCritique) -> SynthesizedPath:
        """Synthesize a single reasoning path from a critique.

        Args:
            critique: A :class:`ContrastiveCritique` providing the
                problem context and analytical guidance.

        Returns:
            A :class:`SynthesizedPath` (with ``is_verified=False``).
        """
        critique_dict = {
            "global_strategic_analysis": critique.global_strategic_analysis,
            "divergence_step_index": critique.divergence_step_index,
            "local_step_critique": critique.local_step_critique,
            "synthesized_guidance": {
                "success_pattern": critique.success_pattern,
                "failure_mode_to_avoid": critique.failure_mode_to_avoid,
            },
        }

        system_prompt, user_prompt = self.prompts_cls.synthesis_prompt(
            problem=critique.problem,
            critique=critique_dict,
        )

        responses = self.llm.generate_with_system(system_prompt, user_prompt, n=1)
        raw_response = responses[0] if responses else ""

        steps = self._parse_steps(raw_response)
        final_answer = AnswerExtractor.extract(raw_response)

        return SynthesizedPath(
            problem=critique.problem,
            steps=steps,
            final_answer=final_answer,
            critique_used=critique,
            raw_response=raw_response,
            is_verified=False,
        )

    def batch_synthesize(
        self,
        critiques: list[ContrastiveCritique],
        max_workers: int = 4,
    ) -> list[SynthesizedPath]:
        """Synthesize reasoning paths for multiple critiques.

        Args:
            critiques: List of :class:`ContrastiveCritique` instances.
            max_workers: Maximum number of parallel workers.  Set to 1
                for sequential execution.

        Returns:
            A list of :class:`SynthesizedPath` instances (one per
            critique).
        """
        if max_workers <= 1:
            return [self.synthesize(c) for c in critiques]

        results: list[tuple[int, SynthesizedPath]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.synthesize, c): i
                for i, c in enumerate(critiques)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results.append((idx, future.result()))
                except Exception:
                    logger.exception("Error synthesizing path for critique %d", idx)
        results.sort(key=lambda x: x[0])
        return [path for _, path in results]

    # ------------------------------------------------------------------ #
    # Parsing                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_steps(response: str) -> list[str]:
        """Parse ``Step N: ...`` formatted text into a list of steps.

        Handles common formatting variations such as:
        * ``Step 1: ...``
        * ``Step 1. ...``
        * ``**Step 1:** ...``

        Args:
            response: Raw LLM output text.

        Returns:
            A list of step strings.  Returns a single-element list
            containing the full response if no step markers are found.
        """
        if not response:
            return []

        # Split on "Step N:" / "Step N." / "**Step N:**" patterns
        pattern = re.compile(
            r"(?:^|\n)\s*\**\s*Step\s+\d+\s*[:.]\s*\**\s*",
            re.IGNORECASE,
        )

        parts = pattern.split(response)
        # First element is text before "Step 1:" (usually empty or preamble)
        steps = [part.strip() for part in parts[1:] if part.strip()]

        if not steps:
            # No step markers found – treat the whole response as one step
            stripped = response.strip()
            return [stripped] if stripped else []

        return steps
