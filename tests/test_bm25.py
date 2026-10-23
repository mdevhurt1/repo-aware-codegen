"""Tests for rac.context.bm25 using real RepoBench Example dataclasses.

Run: python -m pytest test_bm25.py -v -s
"""
import math
from pathlib import Path

import pytest

from rac.context import bm25, tokenize
from rac.types import Example

FIXTURE = Path(__file__).parent / "fixtures" / "repobench_tiny.jsonl"

@pytest.fixture
def example(examples: list[Example]) -> Example:
    return examples[0]


def test_query_text_extracts_last_lines(example):
    assert bm25.query_text(example, n_lines=2) == (
        '    def total(self, code: str = "") -> float:\n'
        '        subtotal = sum(i.price for i in self.items)'
    )
    assert bm25.query_text(example, n_lines=1) == (
        '        subtotal = sum(i.price for i in self.items)'
    )
    assert bm25.query_text(example, n_lines=100) == example.cropped_code.rstrip("\n")


@pytest.mark.parametrize("n_lines", [0, -1, -5])
def test_query_text_rejects_nonpositive_lines(example, n_lines):
    with pytest.raises(ValueError, match="positive"):
        bm25.query_text(example, n_lines=n_lines)


def test_query_text_preserves_internal_blank_lines(example):
    assert "\n\n" in bm25.query_text(example, n_lines=10)


def test_rank_indexes_candidates_and_sorts_scores(example, monkeypatch):
    units = [["items"], ["subtotal", "sum", "items"], ["sum", "items"]]
    query_tokens = ["subtotal", "sum", "items"]
    observed = {"indexed": [], "queries": []}

    def fake_code_tokens(text):
        assert text == bm25.query_text(example)
        return query_tokens

    class FakeBM25:
        def index(self, corpus):
            observed["indexed"].append(corpus)

        def get_scores(self, query):
            observed["queries"].append(query)
            return [1.0, 3.0, 2.0]

    monkeypatch.setattr(bm25.tokenize, "code_tokens", fake_code_tokens)
    monkeypatch.setattr(bm25.bm25s, "BM25", FakeBM25)
    assert bm25.rank(example, units) == [(1, 3.0), (2, 2.0), (0, 1.0)]
    assert observed["indexed"] == [units]
    assert observed["queries"] == [query_tokens]


def test_rank_empty_candidates(example, monkeypatch):
    monkeypatch.setattr(bm25.bm25s, "BM25", lambda: pytest.fail("BM25 should not be constructed"))
    assert bm25.rank(example, []) == []


def test_rank_empty_query(example, monkeypatch):
    monkeypatch.setattr(bm25.tokenize, "code_tokens", lambda text: [])
    monkeypatch.setattr(bm25.bm25s, "BM25", lambda: pytest.fail("BM25 should not be constructed"))
    assert bm25.rank(example, [["a"], ["b"]]) == [(0, 0.0), (1, 0.0)]


def test_query_text_empty_code(example):
    from dataclasses import replace
    assert bm25.query_text(replace(example, cropped_code="")) == ""


def test_rank_score_ties_preserve_original_order(example, monkeypatch):
    class FakeBM25:
        def index(self, corpus):
            assert corpus == [["first"], ["second"], ["third"]]
        def get_scores(self, query):
            return [2, 3, 3]
    monkeypatch.setattr(bm25.bm25s, "BM25", FakeBM25)
    assert bm25.rank(example, [["first"], ["second"], ["third"]]) == [
        (1, 3.0), (2, 3.0), (0, 2.0)
    ]


def test_rank_uses_real_tokenizer_on_dataset(example, monkeypatch):
    expected_query = tokenize.code_tokens(bm25.query_text(example))
    units = [tokenize.code_tokens(s.snippet) for s in example.context]
    seen = {}
    class FakeBM25:
        def index(self, corpus):
            seen["corpus"] = corpus
        def get_scores(self, query):
            seen["query"] = query
            return [float(i) for i in range(len(units))]
    monkeypatch.setattr(bm25.bm25s, "BM25", FakeBM25)
    result = bm25.rank(example, units)
    assert seen == {"corpus": units, "query": expected_query}
    assert [i for i, _ in result] == list(reversed(range(len(units))))

# Integration Test
@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_real_bm25_ranks_dataset_snippets(examples, record_index):
    """Integration: real code tokenizer and bm25s; gold rank is observational."""
    ex = examples[record_index]
    units = [tokenize.code_tokens(snippet.snippet) for snippet in ex.context]
    ranked = bm25.rank(ex, units)

    assert len(ranked) == len(ex.context)
    assert sorted(idx for idx, _ in ranked) == list(range(len(ex.context)))
    assert all(isinstance(score, float) and math.isfinite(score) for _, score in ranked)
    assert all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:], strict=False))
    assert ranked == bm25.rank(ex, units)

    gold_rank = next(i for i, (idx, _) in enumerate(ranked, 1) if idx == ex.gold_snippet_index)
    print(f"\n{ex.repo_name}: gold={ex.gold_snippet_index}, "
          f"rank={gold_rank}/{len(units)}, "
          f"ordering={[(i, round(s, 4)) for i, s in ranked]}")


# Integration Test
@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_real_bm25_single_candidate(examples, record_index):
    ex = examples[record_index]
    if len(ex.context) != 1:
        pytest.skip("Example has multiple candidates")
    ranked = bm25.rank(ex, [tokenize.code_tokens(ex.context[0].snippet)])
    assert len(ranked) == 1
    assert ranked[0][0] == 0
    assert math.isfinite(ranked[0][1])