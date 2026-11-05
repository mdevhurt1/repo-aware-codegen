"""
Context methods. Signature for all: (example, n, tokenizer) -> str of <= n tokens.

All methods draw from the candidate snippets RepoBench ships with each example plus
the file's import statements; only selection and encoding differ.
"""

from __future__ import annotations

from typing import Any, Literal

from rac.context import bm25
from rac.context.bm25 import rank
from rac.context.fill import fill_to_cutoff
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
    ranked = bm25.rank(example, units)
    ranked_units = [units[index] for index, _score in ranked]

    return fill_to_cutoff(ranked_units, n, tokenizer)


def bm25_short(example: Example, n: int, tokenizer: Any) -> str:
    """First 10 lines of each candidate ranked by BM25, filled to cutoff."""

    # Generate a list of candidate snippet strings (from Example context)
    # Using first 10 lines of each
    snippets: list[str] = [
        "\n".join(snippet_obj.snippet.splitlines()[:10])
        for snippet_obj in example.context
    ]

    # Rank using BM25
    ranking = rank(example, snippets)

    # Order snippets by rank
    ranked_snippets = [snippets[i] for i, _score in ranking]

    # Fill to cutoff
    return fill_to_cutoff(ranked_snippets, n, tokenizer)


def oracle(
    example: Example,
    n: int,
    tokenizer: Any,
    fallback: Literal["none", "bm25_long", "bm25_short"] = "none",
) -> str:
    """Ceiling: gold snippet first, then the method's own rule."""

    # Get the full gold snippet (from Example context and gold snippet index)
    gold_snippet: str = example.context[example.gold_snippet_index].snippet

    # Preserve existing oracle behavior by default
    if fallback == "none":
        return fill_to_cutoff([gold_snippet], n, tokenizer)

    # If different fallback rule provided, validate it
    if fallback not in ("bm25_long", "bm25_short"):
        raise ValueError(f"Unknown oracle fallback: {fallback}")

    # Get remaining snippets (all but gold snippet)
    remaining = [
        snippet.snippet
        for idx, snippet in enumerate(example.context)
        if idx != example.gold_snippet_index
    ]

    # If no additional candidates, use default behavior
    if not remaining:
        return fill_to_cutoff([gold_snippet], n, tokenizer)

    # Apply the selected fallback retrieval rule
    # If bm25_short, shorten each candidate to first 10 lines
    # Else (bm25_long), keep as-is
    if fallback == "bm25_short":
        remaining = [
            "\n".join(snippet.splitlines()[:10])
            for snippet in remaining
        ]

    # Rank the remaining candidates with BM25
    ranking = rank(example, remaining)

    # Order by rank
    ranked_remaining = [
        remaining[idx]
        for idx, _score in ranking
    ]

    # Gold always first; fill remaining budget with ranked candidates
    return fill_to_cutoff(
        [gold_snippet, *ranked_remaining],
        n,
        tokenizer,
    )