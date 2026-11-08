"""Features available to the Pass 2 selector before generation."""

from __future__ import annotations

from rac.context import bm25
from rac.context.tokenize import code_tokens
from rac.types import Example


def extract(example: Example) -> dict[str, int | float | str]:
    """Summarize repository imports, candidate code, and retrieval scores.

    The import count uses nonblank lines in RepoBench's repository import field.
    Lengths are Unicode character counts. Identifier overlap counts distinct
    lexical terms shared by the BM25 query and candidate identifiers. The
    score gap is zero when fewer than two candidates can be compared.
    """
    candidates = [candidate.snippet for candidate in example.context]
    query_terms = set(code_tokens(bm25.query_text(example)))
    candidate_terms = {
        term
        for candidate in example.context
        for term in code_tokens(candidate.identifier)
    }
    ranking = bm25.rank(example, candidates) if candidates else []
    top_score = ranking[0][1] if ranking else 0.0
    score_gap = top_score - ranking[1][1] if len(ranking) >= 2 else 0.0

    return {
        "import_count": sum(bool(line.strip()) for line in example.import_statement.splitlines()),
        "candidate_count": len(candidates),
        "candidate_total_chars": sum(map(len, candidates)),
        "identifier_overlap": len(query_terms & candidate_terms),
        "bm25_top_score": top_score,
        "bm25_score_gap": score_gap,
        "prefix_chars": len(example.cropped_code),
        "level": example.level,
    }
