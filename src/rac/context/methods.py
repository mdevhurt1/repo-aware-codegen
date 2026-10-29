"""
Context methods. Signature for all: (example, n, tokenizer) -> str of <= n tokens.

All methods draw from the candidate snippets RepoBench ships with each example plus
the file's import statements; only selection and encoding differ.
"""

from __future__ import annotations

from typing import Any

from rac.context import bm25
from rac.context.fill import fill_to_cutoff
from rac.context.tokenize import code_tokens
from rac.types import Example


def none(example: Example, n: int, tokenizer: Any) -> str:
    """Floor: no cross-file context."""
    return ""

def bm25_long(example: Example, n: int, tokenizer: Any) -> str:
    """Rank full snippets using lexical tokens, then fill with original code."""
    if n < 0:
        raise ValueError("Token budget must be non-negative.")

    if n == 0 or not example.context:
        return ""

    units = [candidate.snippet for candidate in example.context]
    tokenized_units = [code_tokens(unit) for unit in units]
    ranked = bm25.rank(example, tokenized_units)
    ranked_units = [units[index] for index, _score in ranked]

    return fill_to_cutoff(ranked_units, n, tokenizer)


def bm25_short(example: Example, n: int, tokenizer: Any) -> str:
    """First 10 lines of each candidate ranked by BM25, filled to cutoff."""
    raise NotImplementedError("BM25 short")


def oracle(example: Example, n: int, tokenizer: Any) -> str:
    """Ceiling: gold snippet first, then the method's own rule."""
    raise NotImplementedError("oracle")
