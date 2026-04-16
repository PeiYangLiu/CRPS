"""Step segmentation for reasoning trajectories.

Implements the hierarchical segmentation protocol described in
Appendix (Step Segmentation Protocol).  A "step" is defined as a
*Semantic Reasoning Act* — a discrete move that transforms the state
of the problem.

Segmentation priority (highest first):
    1. **Structural delimiters** — ``\\n\\n``, ``\\n``, ``\\\\`` (LaTeX newline)
    2. **Logic connectors** — ``Therefore,``, ``Thus,``, ``Hence,``,
       ``So,``, ``Consequently,``
    3. **Explicit enumerations** — ``Step N:``, ``1.``, ``2.``, ``First,``

Constraints:
    * Never split inside LaTeX math (``$...$`` or ``\\(...\\)``).
    * Enforce a maximum of ``max_step_tokens`` tokens per step (default 256)
      to prevent run-on steps that dilute credit assignment.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Priority-2 logic connectors (sentence-initial)
# ---------------------------------------------------------------------------

_LOGIC_CONNECTORS = re.compile(
    r"(?:(?:^|\n)\s*|(?<=\. ))(?:Therefore|Thus|Hence|So|Consequently),\s",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Priority-3 explicit enumerations
# ---------------------------------------------------------------------------

_ENUMERATION = re.compile(
    r"(?:(?:^|\n)\s*|(?<=\. ))(?:Step\s+\d+[:.]\s|"  # Step 1: / Step 1.
    r"\d+[.)]\s|"                                       # 1. / 1)
    r"(?:First|Second|Third|Next|Finally),\s)",          # First, / Next,
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# LaTeX math spans (to protect from splitting)
# ---------------------------------------------------------------------------

_MATH_INLINE = re.compile(r"\$[^$]+\$")
_MATH_PAREN = re.compile(r"\\\(.*?\\\)", re.DOTALL)
_MATH_DISPLAY = re.compile(r"\\\[.*?\\\]", re.DOTALL)


def _mask_math(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace math spans with placeholders so delimiters inside are ignored."""
    replacements: list[tuple[str, str]] = []
    counter = 0

    def _replace(m: re.Match) -> str:
        nonlocal counter
        placeholder = f"\x00MATH{counter}\x00"
        replacements.append((placeholder, m.group(0)))
        counter += 1
        return placeholder

    masked = _MATH_DISPLAY.sub(_replace, text)
    masked = _MATH_PAREN.sub(_replace, masked)
    masked = _MATH_INLINE.sub(_replace, masked)
    return masked, replacements


def _unmask_math(text: str, replacements: list[tuple[str, str]]) -> str:
    """Restore math spans from placeholders."""
    for placeholder, original in replacements:
        text = text.replace(placeholder, original)
    return text


def _approx_token_count(text: str) -> int:
    """Rough token count (words + punctuation clusters ≈ tokens)."""
    return len(text.split())


def segment_steps(
    text: str,
    max_step_tokens: int = 256,
    tokenizer: Optional[object] = None,
) -> list[str]:
    """Segment a reasoning text into semantic steps.

    Applies the hierarchical segmentation protocol:

    1. Mask LaTeX math to protect equations from splitting.
    2. Split on **structural delimiters** (``\\n\\n``, then ``\\n``).
    3. For segments still too long, split on **logic connectors**.
    4. For segments still too long, split on **explicit enumerations**.
    5. Enforce ``max_step_tokens`` by hard-splitting any remaining
       oversized segments at word boundaries.
    6. Restore math and strip empty segments.

    Args:
        text: Raw reasoning text (e.g., LLM simulation output).
        max_step_tokens: Maximum approximate tokens per step.
        tokenizer: Optional HuggingFace tokenizer for exact token
            counting.  If ``None``, uses whitespace-based approximation.

    Returns:
        List of step strings, each representing one semantic reasoning act.
    """
    if not text or not text.strip():
        return []

    # Step 0: mask math
    masked, math_repls = _mask_math(text)

    # Step 1: structural split (Priority 1)
    # Always split on double-newline and single newline — these are the
    # primary structural boundaries per the segmentation protocol.
    segments = re.split(r"\n", masked)
    refined: list[str] = [s for s in segments if s.strip()]

    # Sub-split on LaTeX \\ for remaining oversized segments
    final_structural: list[str] = []
    for seg in refined:
        if _tok_len(seg, tokenizer) > max_step_tokens and "\\\\" in seg:
            final_structural.extend(
                s for s in re.split(r"\\\\", seg) if s.strip()
            )
        else:
            final_structural.append(seg)

    # Step 2: logic connector split (Priority 2)
    after_logic: list[str] = []
    for seg in final_structural:
        if _tok_len(seg, tokenizer) > max_step_tokens:
            parts = _split_keeping_delimiter(_LOGIC_CONNECTORS, seg)
            after_logic.extend(p for p in parts if p.strip())
        else:
            after_logic.append(seg)

    # Step 3: enumeration split (Priority 3)
    after_enum: list[str] = []
    for seg in after_logic:
        if _tok_len(seg, tokenizer) > max_step_tokens:
            parts = _split_keeping_delimiter(_ENUMERATION, seg)
            after_enum.extend(p for p in parts if p.strip())
        else:
            after_enum.append(seg)

    # Step 4: hard split oversized segments at word boundaries
    final: list[str] = []
    for seg in after_enum:
        if _tok_len(seg, tokenizer) > max_step_tokens:
            final.extend(_hard_split(seg, max_step_tokens, tokenizer))
        else:
            final.append(seg)

    # Step 5: unmask math and clean up
    steps = []
    for seg in final:
        restored = _unmask_math(seg.strip(), math_repls)
        if restored:
            steps.append(restored)

    return steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tok_len(text: str, tokenizer: Optional[object] = None) -> int:
    """Token length using tokenizer if available, else whitespace approx."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return _approx_token_count(text)


def _split_keeping_delimiter(pattern: re.Pattern, text: str) -> list[str]:
    """Split *text* at *pattern* matches, prepending each delimiter to
    the segment that follows it."""
    positions = [m.start() for m in pattern.finditer(text)]
    if not positions:
        return [text]

    parts: list[str] = []
    prev = 0
    for pos in positions:
        if pos > prev:
            parts.append(text[prev:pos])
        prev = pos
    parts.append(text[prev:])
    return parts


def _hard_split(
    text: str,
    max_tokens: int,
    tokenizer: Optional[object] = None,
) -> list[str]:
    """Split text at word boundaries to respect max_tokens."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        wlen = _tok_len(word, tokenizer)
        if current and current_len + wlen > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
            current_len = wlen
        else:
            current.append(word)
            current_len += wlen

    if current:
        chunks.append(" ".join(current))
    return chunks
