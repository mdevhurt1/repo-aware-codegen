"""
Context methods. Signature for all: (example, n, tokenizer) -> str of <= n tokens.

All methods draw from the candidate snippets RepoBench ships with each example plus
the file's import statements; only selection and encoding differ.
"""

from __future__ import annotations

from typing import Any

from rac.context import bm25, fill, tokenize
from rac.types import Example


def none(example: Example, n: int, tokenizer: Any) -> str:
    """Floor: no cross-file context."""
    return ""


def bm25_long(example: Example, n: int, tokenizer: Any) -> str:
    """Whole candidate snippets ranked by BM25, filled to cutoff."""
    raise NotImplementedError("BM25 long")


def bm25_short(example: Example, n: int, tokenizer: Any) -> str:
    """First 10 lines of each candidate ranked by BM25, filled to cutoff."""

    # Helper to get first 10 lines of a snippet
    def first_10_lines(snippet: str) -> str:
        lines = snippet.splitlines()
        return "\n".join(lines[:10])

    # Generate a list of candidate snippet strings (from Example context)
    snippets: list[str] = [
        first_10_lines(snippet_obj.snippet)
        for snippet_obj in example.context
    ]

    # Tokenize each snippet string and store in another list
    snippets_tokenized = [
        tokenize.code_tokens(snippet_str)
        for snippet_str in snippets
    ]

    # Rank using BM25
    ranking = bm25.rank(example, snippets_tokenized)

    # Order snippets by rank
    ranked_idxs = [rank[0] for rank in ranking]
    ranked_snippets = [snippets[i] for i in ranked_idxs]

    # Fill to cutoff
    return fill.fill_to_cutoff(ranked_snippets, n, tokenizer, "\n")


def oracle(example: Example, n: int, tokenizer: Any) -> str:
    """Ceiling: gold snippet first, then the method's own rule."""
    # NOTE: For now, just return the gold snippet

    # Get gold snippet (from Example context and gold snippet index)
    gold_snippet: str = example.context[example.gold_snippet_index].snippet

    return gold_snippet
