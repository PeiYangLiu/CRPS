"""Self-reflective dual-granularity contrastive analysis.

Compares correct and incorrect reasoning trajectories to identify
divergence points, local step-level errors, and global strategic
differences.  Produces structured :class:`ContrastiveCritique` objects
that guide downstream path synthesis.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from crps.prompts.templates import CRPSPrompts
from crps.trajectory.sampler import ContrastivePair

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------

@dataclass
class ContrastiveCritique:
    """Structured output of a dual-granularity contrastive analysis.

    Attributes:
        problem: The mathematical problem statement.
        divergence_step_index: Index of the first step where
            the two trajectories diverge.
        local_step_critique: Dict with keys ``trajectory_a_logic``,
            ``trajectory_b_logic``, and ``critique_of_difference``.
        global_strategic_analysis: High-level comparison of the
            overall strategies employed by each trajectory.
        success_pattern: The key reasoning pattern from the correct
            trajectory that led to the right answer.
        failure_mode_to_avoid: The reasoning pitfall from the
            incorrect trajectory to avoid.
        raw_response: Original LLM output for debugging.
        contrast_type: ``"hard"`` or ``"soft"``.
    """

    problem: str
    divergence_step_index: int
    local_step_critique: dict  # {trajectory_a_logic, trajectory_b_logic, critique_of_difference}
    global_strategic_analysis: str
    success_pattern: str
    failure_mode_to_avoid: str
    raw_response: str
    contrast_type: str  # "hard" or "soft"


# -----------------------------------------------------------------------
# Analyser
# -----------------------------------------------------------------------

class ContrastiveAnalyzer:
    """Run dual-granularity contrastive analysis via an LLM.

    Args:
        llm: LLM inference object exposing a
            ``generate_with_system(system, user, n=1)`` method.
        prompts_cls: Prompt templates class; defaults to
            :class:`CRPSPrompts`.
    """

    def __init__(self, llm, prompts_cls=None) -> None:
        self.llm = llm
        self.prompts_cls = prompts_cls or CRPSPrompts

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(self, pair: ContrastivePair) -> Optional[ContrastiveCritique]:
        """Analyse a single contrastive pair.

        Args:
            pair: A :class:`ContrastivePair` containing the problem,
                positive trajectory, negative trajectory, and contrast
                type.

        Returns:
            A :class:`ContrastiveCritique` on success, or ``None`` if the
            LLM response cannot be parsed.
        """
        positive_text = self.prompts_cls.format_trajectory(pair.positive.steps)
        negative_text = self.prompts_cls.format_trajectory(pair.negative.steps)

        system_prompt, user_prompt = self.prompts_cls.contrastive_analysis_prompt(
            problem=pair.problem,
            positive_trajectory=positive_text,
            negative_trajectory=negative_text,
        )

        responses = self.llm.generate_with_system(system_prompt, user_prompt, n=1)
        raw_response = responses[0] if responses else ""

        return self._parse_critique(raw_response, pair.problem, pair.contrast_type)

    def batch_analyze(
        self,
        pairs: list[ContrastivePair],
        max_workers: int = 4,
    ) -> list[ContrastiveCritique]:
        """Analyse multiple contrastive pairs, optionally in parallel.

        Args:
            pairs: List of :class:`ContrastivePair` instances.
            max_workers: Maximum number of parallel workers.  Set to 1
                for sequential execution.

        Returns:
            A list of successfully parsed :class:`ContrastiveCritique`
            objects (failures are filtered out).
        """
        results: list[Optional[ContrastiveCritique]] = []

        if max_workers <= 1:
            results = [self.analyze(p) for p in pairs]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.analyze, p): i for i, p in enumerate(pairs)}
                indexed_results: list[tuple[int, Optional[ContrastiveCritique]]] = []
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        indexed_results.append((idx, future.result()))
                    except Exception:
                        logger.exception("Error analysing pair %d", idx)
                        indexed_results.append((idx, None))
                indexed_results.sort(key=lambda x: x[0])
                results = [r for _, r in indexed_results]

        critiques = [r for r in results if r is not None]
        num_failures = len(pairs) - len(critiques)
        if num_failures > 0:
            failure_rate = num_failures / len(pairs) * 100
            logger.warning(
                "Contrastive analysis: %d/%d parse failures (%.1f%%)",
                num_failures,
                len(pairs),
                failure_rate,
            )

        return critiques

    # ------------------------------------------------------------------ #
    # Parsing                                                             #
    # ------------------------------------------------------------------ #

    def _parse_critique(
        self,
        response: str,
        problem: str,
        contrast_type: str,
    ) -> Optional[ContrastiveCritique]:
        """Parse an LLM response into a :class:`ContrastiveCritique`.

        Handles markdown code blocks, bare JSON objects, and applies
        rule-based repair for common formatting issues.

        Args:
            response: Raw LLM output text.
            problem: The problem statement (carried through to the
                critique).
            contrast_type: ``"hard"`` or ``"soft"``.

        Returns:
            A :class:`ContrastiveCritique` on success, or ``None`` if
            parsing fails.
        """
        if not response:
            logger.warning("Empty LLM response for contrastive analysis")
            return None

        json_str = self._extract_json_string(response)
        if json_str is None:
            logger.warning(
                "Could not locate JSON block in response (len=%d): %.200s…",
                len(response),
                response,
            )
            return None

        # Attempt parse, with one retry after rule-based repair
        data = self._try_parse_json(json_str)
        if data is None:
            repaired = self._repair_json(json_str)
            data = self._try_parse_json(repaired)

        if data is None:
            # Last resort: regex-based field extraction
            critique = self._regex_fallback_extract(
                response, problem, contrast_type, response,
            )
            if critique is not None:
                logger.debug(
                    "Recovered critique via regex fallback (len=%d)",
                    len(response),
                )
                return critique
            logger.warning(
                "JSON parse failed after repair for response (len=%d): %.200s…",
                len(response),
                response,
            )
            return None

        return self._build_critique(data, problem, contrast_type, response)

    # ------------------------------------------------------------------ #
    # JSON extraction helpers                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_json_string(text: str) -> Optional[str]:
        """Extract a JSON block from *text*.

        Checks for fenced code blocks first (```json ... ```), then
        falls back to the outermost ``{ ... }`` pair.
        """
        # Try fenced code blocks
        fence_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)```", re.DOTALL
        )
        m = fence_pattern.search(text)
        if m:
            return m.group(1).strip()

        # Fall back to outermost braces
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # Unbalanced – return from first brace to end
        return text[start:]

    @staticmethod
    def _try_parse_json(text: str) -> Optional[dict]:
        """Attempt ``json.loads``; return ``None`` on failure."""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _repair_json(text: str) -> str:
        """Apply rule-based repairs for common JSON issues.

        Repairs attempted:
        * Unescaped newlines / tabs inside string values.
        * Unescaped internal double-quotes (replace with single quotes).
        * Missing closing brace.
        * Trailing commas before closing braces/brackets.
        """
        # 1) Escape literal newlines and tabs inside JSON string values
        #    Walk through the text and replace raw newlines/tabs that are
        #    inside double-quoted strings with their escaped counterparts.
        result_chars: list[str] = []
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string
                result_chars.append(ch)
            elif in_string and ch == '\n':
                result_chars.append('\\n')
            elif in_string and ch == '\t':
                result_chars.append('\\t')
            elif in_string and ch == '\\' and i + 1 < len(text) and text[i + 1] not in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                # Escape bare backslashes (e.g. LaTeX \frac)
                result_chars.append('\\\\')
            else:
                result_chars.append(ch)
            i += 1
        repaired = ''.join(result_chars)

        # 2) Remove trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # 3) Ensure balanced braces
        open_count = repaired.count("{")
        close_count = repaired.count("}")
        if open_count > close_count:
            repaired += "}" * (open_count - close_count)

        return repaired

    @staticmethod
    def _regex_fallback_extract(text: str, problem: str, contrast_type: str, raw: str) -> Optional["ContrastiveCritique"]:
        """Last-resort extraction using regex to pull fields individually."""
        def _find_value(key: str) -> str:
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            m = re.search(pattern, text, re.DOTALL)
            return m.group(1) if m else ""

        def _find_int(key: str) -> int:
            pattern = rf'"{key}"\s*:\s*(\d+)'
            m = re.search(pattern, text)
            return int(m.group(1)) if m else 0

        divergence = _find_int("divergence_step_index")
        global_analysis = _find_value("global_strategic_analysis")
        success = _find_value("success_pattern")
        failure = _find_value("failure_mode_to_avoid")
        a_logic = _find_value("trajectory_a_logic")
        b_logic = _find_value("trajectory_b_logic")
        critique_diff = _find_value("critique_of_difference")

        if not global_analysis and not success:
            return None

        return ContrastiveCritique(
            problem=problem,
            divergence_step_index=divergence,
            local_step_critique={
                "trajectory_a_logic": a_logic,
                "trajectory_b_logic": b_logic,
                "critique_of_difference": critique_diff,
            },
            global_strategic_analysis=global_analysis,
            success_pattern=success,
            failure_mode_to_avoid=failure,
            raw_response=raw,
            contrast_type=contrast_type,
        )

    # ------------------------------------------------------------------ #
    # Building the critique object                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_critique(
        data: dict,
        problem: str,
        contrast_type: str,
        raw_response: str,
    ) -> Optional[ContrastiveCritique]:
        """Construct a :class:`ContrastiveCritique` from parsed JSON.

        Returns ``None`` if essential fields are missing.
        """
        try:
            divergence_step_index = int(data.get("divergence_step_index", -1))
        except (ValueError, TypeError):
            divergence_step_index = -1

        local_step_critique = data.get("local_step_critique", {})
        if not isinstance(local_step_critique, dict):
            local_step_critique = {}

        global_strategic_analysis = data.get("global_strategic_analysis", "")

        guidance = data.get("synthesized_guidance", {})
        if not isinstance(guidance, dict):
            guidance = {}
        success_pattern = guidance.get("success_pattern", "")
        failure_mode_to_avoid = guidance.get("failure_mode_to_avoid", "")

        # Validate essential fields
        if divergence_step_index < 0 and not global_strategic_analysis:
            logger.warning(
                "Missing essential fields in critique JSON: "
                "divergence_step_index=%s, global_strategic_analysis=%r",
                data.get("divergence_step_index"),
                global_strategic_analysis,
            )
            return None

        return ContrastiveCritique(
            problem=problem,
            divergence_step_index=max(divergence_step_index, 0),
            local_step_critique=local_step_critique,
            global_strategic_analysis=global_strategic_analysis,
            success_pattern=success_pattern,
            failure_mode_to_avoid=failure_mode_to_avoid,
            raw_response=raw_response,
            contrast_type=contrast_type,
        )
