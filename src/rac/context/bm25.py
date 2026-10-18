"""BM25 ranking over an example's own candidate snippets."""

from __future__ import annotations

from rac.types import Example

QUERY_LINES = 5

def query_text(example: Example, n_lines: int = QUERY_LINES) -> str:
    """The last `n_lines` lines of in-file code before the cursor."""
    raise NotImplementedError("retrieval query")

def rank(example: Example, units: list[str]) -> list[tuple[int, float]]:
    """(unit index, BM25 score) sorted best first. Scores are also pass 2 features."""
    raise NotImplementedError("BM25 ranking")
