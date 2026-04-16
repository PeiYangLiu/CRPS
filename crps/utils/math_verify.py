"""Math answer verification using SymPy and pattern-based extraction."""

from __future__ import annotations

import re
from typing import Optional


class AnswerExtractor:
    """Extract the final answer from free-form reasoning text.

    Extraction is attempted in the following order:
    1. ``\\boxed{...}`` (LaTeX, handles nested braces)
    2. ``Final Answer: ...``
    3. ``The answer is ...``
    4. Last number appearing in the text
    """

    # Pre-compiled patterns (order matters)
    _FINAL_ANSWER_RE = re.compile(
        r"[Ff]inal\s+[Aa]nswer\s*:\s*(.+?)(?:\n|$)", re.DOTALL
    )
    _THE_ANSWER_IS_RE = re.compile(
        r"[Tt]he\s+answer\s+is\s*:?\s*(.+?)(?:\.\s|\.\Z|\n|$)", re.DOTALL
    )
    _LAST_NUMBER_RE = re.compile(
        r"(-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)"
    )

    @classmethod
    def extract(cls, text: str) -> Optional[str]:
        """Extract the final answer from *text*.

        Args:
            text: The full reasoning / solution text.

        Returns:
            The extracted answer string, or ``None`` if nothing is found.
        """
        if not text:
            return None

        # 1. \boxed{...}
        boxed = cls._extract_boxed(text)
        if boxed is not None:
            return boxed.strip()

        # 2. Final Answer: ...
        m = cls._FINAL_ANSWER_RE.search(text)
        if m:
            return m.group(1).strip()

        # 3. The answer is ...
        m = cls._THE_ANSWER_IS_RE.search(text)
        if m:
            return m.group(1).strip()

        # 4. Last number in text
        numbers = cls._LAST_NUMBER_RE.findall(text)
        if numbers:
            return numbers[-1].replace(",", "")

        return None

    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        r"""Extract content from ``\boxed{...}``, handling nested braces."""
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return None
        # Walk forward from the opening brace
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            return text[start : pos - 1]
        return None


class MathVerifier:
    """Check mathematical equivalence between two answer strings."""

    @classmethod
    def verify(cls, predicted: str, ground_truth: str) -> bool:
        """Return ``True`` if *predicted* and *ground_truth* are equivalent.

        Tries normalised string comparison first, then falls back to
        SymPy symbolic comparison.

        Args:
            predicted: The model's predicted answer.
            ground_truth: The reference / gold answer.

        Returns:
            ``True`` when the answers are mathematically equivalent.
        """
        if predicted is None or ground_truth is None:
            return False

        pred_norm = cls.normalize_answer(str(predicted))
        gt_norm = cls.normalize_answer(str(ground_truth))

        if not pred_norm or not gt_norm:
            return False

        # Fast path: exact string match after normalisation
        if pred_norm == gt_norm:
            return True

        # Slow path: symbolic comparison via SymPy
        return cls._sympy_equal(pred_norm, gt_norm)

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalise an answer string for comparison.

        Handles common LaTeX artefacts, whitespace, and number formats.

        Args:
            answer: Raw answer string.

        Returns:
            A cleaned, normalised string.
        """
        if not answer:
            return ""

        s = answer.strip()

        # Strip surrounding $ signs (LaTeX inline math)
        s = s.strip("$")

        # Common LaTeX commands → plain text
        s = s.replace("\\frac", "frac")
        s = s.replace("\\dfrac", "frac")
        s = s.replace("\\tfrac", "frac")
        s = s.replace("\\left", "")
        s = s.replace("\\right", "")
        s = s.replace("\\,", "")
        s = s.replace("\\;", "")
        s = s.replace("\\!", "")
        s = s.replace("\\quad", " ")
        s = s.replace("\\text", "")
        s = s.replace("\\mathrm", "")
        s = s.replace("\\mathbf", "")

        # Remove commas used as thousands separators (but keep decimal points)
        s = re.sub(r"(\d),(\d)", r"\1\2", s)

        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()

        return s

    @staticmethod
    def _sympy_equal(expr1: str, expr2: str) -> bool:
        """Use SymPy to check symbolic equivalence of two expressions.

        Args:
            expr1: First normalised expression string.
            expr2: Second normalised expression string.

        Returns:
            ``True`` when SymPy considers the expressions equal.
        """
        try:
            from sympy import simplify, nsimplify
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
                convert_xor,
            )

            transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )

            # Replace LaTeX-style frac{a}{b} with (a)/(b) for the parser
            def _replace_frac(s: str) -> str:
                while "frac{" in s:
                    idx = s.find("frac{")
                    # Extract numerator
                    start_num = idx + len("frac{")
                    depth, pos = 1, start_num
                    while pos < len(s) and depth > 0:
                        if s[pos] == "{":
                            depth += 1
                        elif s[pos] == "}":
                            depth -= 1
                        pos += 1
                    numerator = s[start_num : pos - 1]
                    # Extract denominator
                    if pos < len(s) and s[pos] == "{":
                        start_den = pos + 1
                        depth, pos = 1, start_den
                        while pos < len(s) and depth > 0:
                            if s[pos] == "{":
                                depth += 1
                            elif s[pos] == "}":
                                depth -= 1
                            pos += 1
                        denominator = s[start_den : pos - 1]
                    else:
                        denominator = "1"
                    s = s[:idx] + f"(({numerator})/({denominator}))" + s[pos:]
                return s

            e1 = _replace_frac(expr1)
            e2 = _replace_frac(expr2)

            sym1 = parse_expr(e1, transformations=transformations)
            sym2 = parse_expr(e2, transformations=transformations)

            # Try symbolic simplification
            if simplify(sym1 - sym2) == 0:
                return True

            # Fall back to high-precision numerical comparison
            diff = abs(complex(sym1.evalf()) - complex(sym2.evalf()))
            return diff < 1e-8

        except Exception:
            # If SymPy cannot parse either expression, fall back to numeric
            try:
                return abs(float(expr1) - float(expr2)) < 1e-8
            except (ValueError, TypeError):
                return False
