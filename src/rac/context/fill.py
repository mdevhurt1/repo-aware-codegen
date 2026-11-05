"""
Budget fill rule.

Append units in rank order. If a unit crosses the budget, return the
longest character prefix of the assembled text that fits.
"""

from __future__ import annotations

from typing import Any

# How far past the bisected cut to look for a longer prefix that fits. Token counts can
# drop as characters merge into one token, so the fit test is not monotonic near the cut.
_MERGE_WINDOW = 32


def _fits(text: str, n: int, tokenizer: Any) -> bool:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= n


def _longest_fitting_prefix(text: str, fitting: int, n: int, tokenizer: Any) -> str:
    """Longest prefix of `text` within n tokens, given text[:fitting] fits and text does not."""
    low, high = fitting, len(text)
    while high - low > 1:
        mid = (low + high) // 2
        if _fits(text[:mid], n, tokenizer):
            low = mid
        else:
            high = mid

    for end in range(min(low + _MERGE_WINDOW, len(text) - 1), high, -1):
        if _fits(text[:end], n, tokenizer):
            return text[:end]
    return text[:low]


def fill_to_cutoff(units: list[str], n: int, tokenizer: Any, sep: str = "\n") -> str:
    """Join ranked units while keeping the result within n model tokens.

    Count the joined text without adding special tokens, because tokens can merge across
    a join and per-unit counts do not add up. Cut the original text at character boundaries
    so decoding cannot introduce broken characters.

    The cut takes O(log L) tokenizer calls for an oversized unit of L characters. Every
    returned prefix was counted, so the result is always within budget; a token count that
    dips below n more than _MERGE_WINDOW characters past the cut would only make it shorter
    than the longest possible prefix.
    """
    if n < 0:
        raise ValueError("Token budget must be non-negative.")

    if n == 0 or not units:
        return ""

    text = ""

    for index, unit in enumerate(units):
        candidate = text + sep + unit if index > 0 else unit

        if not _fits(candidate, n, tokenizer):
            return _longest_fitting_prefix(candidate, len(text), n, tokenizer)

        text = candidate

    return text
