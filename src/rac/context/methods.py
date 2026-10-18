"""
Context methods. Signature for all: (example, n, tokenizer) -> str of <= n tokens.

All methods draw from the candidate snippets RepoBench ships with each example plus
the file's import statements; only selection and encoding differ.
"""

from __future__ import annotations

from typing import Any

from rac.types import Example


def none(example: Example, n: int, tokenizer: Any) -> str:
    """Floor: no cross-file context."""
    return ""


def bm25_long(example: Example, n: int, tokenizer: Any) -> str:
    """Whole candidate snippets ranked by BM25, filled to cutoff."""
    raise NotImplementedError("BM25 long")


def bm25_short(example: Example, n: int, tokenizer: Any) -> str:
    """First 10 lines of each candidate ranked by BM25, filled to cutoff."""
    raise NotImplementedError("BM25 short")


def oracle(example: Example, n: int, tokenizer: Any) -> str:
    """Ceiling: gold snippet first, then the method's own rule."""
    raise NotImplementedError("oracle")
