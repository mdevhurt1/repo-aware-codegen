"""
Budget fill rule.

Append units in rank order. If a unit crosses the budget, return the
longest character prefix of the assembled text that fits.
"""

from __future__ import annotations

from typing import Any


def fill_to_cutoff(units: list[str], n: int, tokenizer: Any, sep: str = "\n") -> str:
    """Join ranked units while keeping the result within n model tokens.

    Count separators without adding special tokens. Cut the original text
    at character boundaries so decoding cannot introduce broken characters.
    """
    if n < 0:
        raise ValueError("Token budget must be non-negative.")

    if n == 0 or not units:
        return ""

    text = ""

    for index, unit in enumerate(units):
        candidate = text + sep + unit if index > 0 else unit
        used = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])

        if used <= n:
            text = candidate
            continue

        # Search longest prefixes first. Token counts can decrease when
        # characters combine into a token, so binary search is not reliable.
        for end in range(len(candidate) - 1, 0, -1):
            prefix = candidate[:end]
            used = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])

            if used <= n:
                return prefix

        return ""

    return text