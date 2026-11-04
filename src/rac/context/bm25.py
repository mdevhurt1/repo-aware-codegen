"""BM25 ranking over an example's own candidate snippets."""

from __future__ import annotations

from math import isfinite

# Importing open-source rank_bm25 package
# Github: https://github.com/dorianbrown/rank_bm25
from rank_bm25 import BM25Okapi

from rac.context import tokenize
from rac.types import Example

QUERY_LINES = 5

def query_text(example: Example, n_lines: int = QUERY_LINES) -> str:
    """The last `n_lines` lines of in-file code before the cursor."""

    # Input validation
    if n_lines <= 0:
        raise ValueError("n_lines must be a positive integer")

    # Using 'cropped_code' from dataset Example
    # Split cropped code into lines
    lines = example.cropped_code.splitlines()
    # Subset the last n_lines, join back into str and return
    return "\n".join(lines[-n_lines:])

def rank(example: Example, units: list[str]) -> list[tuple[int, float]]:
    """(unit index, BM25 score) sorted best first. Scores are also pass 2 features."""
    # NOTE: Using the open-source 'rank_bm25' Python library to implement this.

    # Input validation for snippets (units)
    # Don't attempt to rank empty units
    if not units:
        raise ValueError("units must not be empty")

    # First query the last N lines of in-file code; this will be used as our BM25 query
    last_lines: str = query_text(example)
    # Tokenize the query using our code-aware tokenization
    query_tokens: list[str] = tokenize.code_tokens(last_lines)

    # Verify that query token list is not empty
    # Don't attempt to rank empty tokenized query
    if not query_tokens:
        raise ValueError("Query tokenization failed: received empty code token list")

    # Tokenize the given candidate snippets
    # They will be used as our documents (corpus)
    snippet_tokens: list[list[str]] = [
        tokenize.code_tokens(snippet) for snippet in units
    ]

    # Verify that there are no empty token lists in corpus (to avoid bm25 failures)
    # Don't attempt to rank with empty tokenized snippet
    if not all(snippet_tokens):
        empty_indices = [i for i, tokens in enumerate(snippet_tokens) if not tokens]
        raise ValueError(
            f"Snippet tokenization failed: found empty token lists at indices {empty_indices}"
        )

    corpus = snippet_tokens

    # Create BM25 model and index all snippets as documents
    bm25 = BM25Okapi(corpus)

    # Get score of each document (snippet) against the same query (code)
    scores = bm25.get_scores(query_tokens)

    # Verify that number of received scores match number of candidate snippets
    if len(scores) != len(units):
        raise ValueError(f"Expected {len(units)} BM25 scores, received {len(scores)}")

    # Verify that received scores are finite
    if not all(isfinite(score) for score in scores):
        invalid_indices = [i for i, score in enumerate(scores) if not isfinite(score)]
        raise ValueError(
            f"Found non-finite BM25 scores at indices {invalid_indices}"
        )

    # Associate scores with their original snippet indexes
    results = [
        (idx, float(score))
        for idx, score in enumerate(scores)
    ]

    # Sort results by descending BM25 score
    results.sort(key=lambda x: x[1], reverse=True)

    return results
