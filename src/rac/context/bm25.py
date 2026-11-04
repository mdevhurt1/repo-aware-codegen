"""BM25 ranking over an example's own candidate snippets."""

from __future__ import annotations

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

    # First query the last N lines of in-file code; this will be used as our BM25 query
    last_lines: str = query_text(example)
    # Tokenize the query using our code-aware tokenization
    # A query with no tokens (e.g. punctuation only) is fine: every snippet scores 0.0
    query_tokens: list[str] = tokenize.code_tokens(last_lines)

    # Tokenize the given candidate snippets
    # They will be used as our documents (corpus)
    # RepoBench ships some empty snippets; an empty document is fine and scores 0.0
    corpus: list[list[str]] = [
        tokenize.code_tokens(snippet) for snippet in units
    ]

    # rank_bm25 divides by zero when no snippet has any token (or units is empty)
    # Nothing can match, so score every snippet 0.0 and keep dataset order
    if not any(corpus):
        return [(idx, 0.0) for idx in range(len(units))]

    # Create BM25 model and index all snippets as documents
    bm25 = BM25Okapi(corpus)

    # Get score of each document (snippet) against the same query (code)
    scores = bm25.get_scores(query_tokens)

    # Associate scores with their original snippet indexes
    results = [
        (idx, float(score))
        for idx, score in enumerate(scores)
    ]

    # Sort results by descending BM25 score; ties keep dataset order (stable sort)
    results.sort(key=lambda x: x[1], reverse=True)

    return results
