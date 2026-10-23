"""BM25 ranking over an example's own candidate snippets."""

from __future__ import annotations

from rac.types import Example
from rac.context import tokenize

# Importing open-source BM25 package
# PyPi Source: https://pypi.org/project/BM25/
# GitHub Repo: https://github.com/xhluca/bm25s/tree/main
import bm25s

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

def rank(example: Example, units: list[list[str]]) -> list[tuple[int, float]]:
    """(unit index, BM25 score) sorted best first. Scores are also pass 2 features."""
    # NOTE: Trying out the open-source 'BM25' Python library to implement this.

    # Handle empty units by returning empty ranking
    if not units:
        return []

    # First query the last N lines of in-file code; this will be used as our BM25 query
    last_lines: str = query_text(example)
    # Tokenize the query using our code-aware tokenization
    query_tokens: list[str] = tokenize.code_tokens(last_lines)

    # Handle empty queries by returning 0 score for each snippet
    if not query_tokens:
        return [(idx, 0.0) for idx in range(len(units))]

    # Assume candidate snippets are already tokenized
    # They will be used as our documents (corpus)
    corpus = units

    # Create BM25 model and index all snippets as documents
    retriever = bm25s.BM25()
    retriever.index(corpus)

    # Get score of each document (snippet) against the same query (code)
    scores = retriever.get_scores(query_tokens)

    # Associate scores with their original snippet indexes
    results = [
        (idx, float(score))
        for idx, score in enumerate(scores)
    ]

    # Sort results by descending BM25 score
    results.sort(key=lambda x: x[1], reverse=True)

    return results
