"""Tests for rac.context.bm25 using real RepoBench Example objects."""
import math
from dataclasses import replace

import pytest

from rac.context import bm25, tokenize
from rac.types import Example


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


def test_query_text_empty_code(example):
    assert bm25.query_text(replace(example, cropped_code="")) == ""


def test_rank_tokenizes_candidates_and_sorts_scores(example, monkeypatch):
    units = ["items", "subtotal_sum(items)", "sum(items)"]
    query = bm25.query_text(example)
    expected_query_tokens = ["subtotal", "sum", "items"]
    expected_corpus = [["items"], ["subtotal", "sum", "items"], ["sum", "items"]]
    observed = {"texts": [], "corpora": [], "queries": []}

    def fake_code_tokens(text):
        observed["texts"].append(text)
        if text == query:
            return expected_query_tokens
        return {unit: tokens for unit, tokens in zip(units, expected_corpus, strict=False)}[text]

    class FakeBM25Okapi:
        def __init__(self, corpus):
            observed["corpora"].append(corpus)
        def get_scores(self, query_tokens):
            observed["queries"].append(query_tokens)
            return [1.0, 3.0, 2.0]

    monkeypatch.setattr(bm25.tokenize, "code_tokens", fake_code_tokens)
    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    assert bm25.rank(example, units) == [(1, 3.0), (2, 2.0), (0, 1.0)]
    assert observed["texts"] == [query, *units]
    assert observed["corpora"] == [expected_corpus]
    assert observed["queries"] == [expected_query_tokens]


def test_rank_empty_candidates(example):
    assert bm25.rank(example, []) == []


@pytest.mark.parametrize("cropped_code", ["", '"""\n\n\n\n', "\n   !!!"])
def test_rank_empty_query_scores_zero_in_dataset_order(example, cropped_code):
    """A query with no code tokens (RepoBench has them) is not an error."""
    ex = replace(example, cropped_code=cropped_code)
    assert bm25.rank(ex, ["valid snippet", "another_valid"]) == [(0, 0.0), (1, 0.0)]


def test_rank_scores_empty_snippets_zero(example):
    """RepoBench ships '' snippets; they are valid candidates that match nothing."""
    ex = replace(example, cropped_code="result = parse_config(path)\n")
    units = ["", "def parse_config(path): return path", "# !@#", "x = 1"]
    ranked = bm25.rank(ex, units)
    assert ranked[0][0] == 1
    assert ranked[0][1] > 0
    assert ranked[1:] == [(0, 0.0), (2, 0.0), (3, 0.0)]


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ([""], [(0, 0.0)]),
        (["", "   ", "# !@#"], [(0, 0.0), (1, 0.0), (2, 0.0)]),
    ],
)
def test_rank_all_empty_snippets_score_zero_in_dataset_order(example, units, expected):
    assert bm25.rank(example, units) == expected


def test_rank_score_ties_preserve_original_order(example, monkeypatch):
    class FakeBM25Okapi:
        def __init__(self, corpus):
            assert corpus == [["first"], ["second"], ["third"]]
        def get_scores(self, query):
            return [2, 3, 3]
    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    assert bm25.rank(example, ["first", "second", "third"]) == [
        (1, 3.0), (2, 3.0), (0, 2.0)
    ]


def test_rank_uses_real_tokenizer_on_dataset(example, monkeypatch):
    expected_query = tokenize.code_tokens(bm25.query_text(example))
    units = [s.snippet for s in example.context]
    expected_corpus = [tokenize.code_tokens(unit) for unit in units]
    seen = {}

    class FakeBM25Okapi:
        def __init__(self, corpus):
            seen["corpus"] = corpus
        def get_scores(self, query):
            seen["query"] = query
            return [float(i) for i in range(len(units))]

    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    result = bm25.rank(example, units)
    assert seen == {"corpus": expected_corpus, "query": expected_query}
    assert [i for i, _ in result] == list(reversed(range(len(units))))


# Integration Test
@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_real_bm25_ranks_dataset_snippets(examples, record_index):
    """Real rank_bm25 integration; gold rank is reported, not assumed to be first."""
    ex = examples[record_index]
    units = [snippet.snippet for snippet in ex.context]
    ranked = bm25.rank(ex, units)

    assert len(ranked) == len(units)
    assert sorted(idx for idx, _ in ranked) == list(range(len(units)))
    assert all(isinstance(score, float) and math.isfinite(score) for _, score in ranked)
    assert all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:], strict=False))
    assert ranked == bm25.rank(ex, units)

    gold_rank = next(i for i, (idx, _) in enumerate(ranked, 1)
                     if idx == ex.gold_snippet_index)
    print(f"\n{ex.repo_name}: gold={ex.gold_snippet_index}, "
          f"rank={gold_rank}/{len(units)}, "
          f"ordering={[(i, round(s, 4)) for i, s in ranked]}")


# Integration Test
def test_real_bm25_ranks_matching_snippet_first(example):
    """The snippet that shares the query's identifiers outranks the ones that do not."""
    ex = replace(example, cropped_code="result = parse_config(path)\n")
    units = ["class Unrelated: pass", "def parse_config(path): return path", "x = 1"]
    ranked = bm25.rank(ex, units)
    assert [idx for idx, _ in ranked] == [1, 0, 2]
    assert ranked[0][1] > ranked[1][1]


# Integration Test
def test_real_bm25_single_candidate(examples):
    ex = examples[2]  # the fixture's only single-candidate example
    assert len(ex.context) == 1
    ranked = bm25.rank(ex, [ex.context[0].snippet])
    assert len(ranked) == 1
    assert ranked[0][0] == 0
    assert math.isfinite(ranked[0][1])


def test_rank_sorts_negative_and_zero_scores(example, monkeypatch):
    """BM25Okapi returns negative scores on small corpora; they sort below zero."""
    class FakeBM25Okapi:
        def __init__(self, corpus):
            assert corpus == [["first"], ["second"], ["third"]]

        def get_scores(self, query):
            return [-2.5, 0.0, 1.25]

    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    assert bm25.rank(example, ["first", "second", "third"]) == [
        (2, 1.25), (1, 0.0), (0, -2.5)
    ]
