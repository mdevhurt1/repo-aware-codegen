"""Tests for rac.context.bm25 using real RepoBench Example objects.

Run: python -m pytest test_bm25.py -v -s
"""
import math
from pathlib import Path
from dataclasses import replace

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
        return {unit: tokens for unit, tokens in zip(units, expected_corpus)}[text]

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


def test_rank_empty_candidates(example, monkeypatch):
    monkeypatch.setattr(bm25, "BM25Okapi", lambda corpus: pytest.fail("BM25 should not be constructed"))
    with pytest.raises(ValueError, match="units must not be empty"):
        bm25.rank(example, [])


def test_rank_empty_query(example, monkeypatch):
    monkeypatch.setattr(bm25.tokenize, "code_tokens", lambda text: [])
    monkeypatch.setattr(bm25, "BM25Okapi", lambda corpus: pytest.fail("BM25 should not be constructed"))
    with pytest.raises(ValueError, match="Query tokenization failed"):
        bm25.rank(example, ["a", "b"])


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
    assert all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:]))
    assert ranked == bm25.rank(ex, units)

    gold_rank = next(i for i, (idx, _) in enumerate(ranked, 1)
                     if idx == ex.gold_snippet_index)
    print(f"\n{ex.repo_name}: gold={ex.gold_snippet_index}, "
          f"rank={gold_rank}/{len(units)}, "
          f"ordering={[(i, round(s, 4)) for i, s in ranked]}")


# Integration Test
@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_real_bm25_single_candidate(examples, record_index):
    ex = examples[record_index]
    if len(ex.context) != 1:
        pytest.skip("Example has multiple candidates")
    ranked = bm25.rank(ex, [ex.context[0].snippet])
    assert len(ranked) == 1
    assert ranked[0][0] == 0
    assert math.isfinite(ranked[0][1])


@pytest.mark.parametrize(
    ("units", "invalid_indices"),
    [
        (["valid_identifier", ""], [1]),
        (["", "valid_identifier"], [0]),
        (["", "valid_identifier", "   ", "!!!"], [0, 2, 3]),
    ],
)
def test_rank_rejects_empty_tokenized_snippets(example, monkeypatch, units, invalid_indices):
    """Reject blank/punctuation-only snippets and report all invalid positions."""
    monkeypatch.setattr(
        bm25, "BM25Okapi",
        lambda corpus: pytest.fail("BM25 must not be constructed for invalid snippets"),
    )
    with pytest.raises(ValueError, match="Snippet tokenization failed") as exc_info:
        bm25.rank(example, units)
    assert f"indices {invalid_indices}" in str(exc_info.value)


def test_rank_validates_query_before_tokenizing_snippets(example, monkeypatch):
    observed = []

    def fake_code_tokens(text):
        observed.append(text)
        return []

    monkeypatch.setattr(bm25.tokenize, "code_tokens", fake_code_tokens)
    monkeypatch.setattr(
        bm25, "BM25Okapi",
        lambda corpus: pytest.fail("BM25 must not be constructed for an empty query"),
    )
    with pytest.raises(ValueError, match="Query tokenization failed"):
        bm25.rank(example, ["valid", "also_valid"])
    assert observed == [bm25.query_text(example)]


def test_rank_rejects_empty_query_from_real_tokenizer(example, monkeypatch):
    monkeypatch.setattr(
        bm25, "BM25Okapi",
        lambda corpus: pytest.fail("BM25 must not be constructed for an empty query"),
    )
    with pytest.raises(ValueError, match="Query tokenization failed"):
        bm25.rank(replace(example, cropped_code="\n   !!!"), ["valid snippet"])


def test_rank_rejects_empty_snippets_from_real_tokenizer(example, monkeypatch):
    monkeypatch.setattr(
        bm25, "BM25Okapi",
        lambda corpus: pytest.fail("BM25 must not be constructed for invalid snippets"),
    )
    with pytest.raises(ValueError, match=r"indices \[1, 3\]"):
        bm25.rank(example, ["valid", "", "another_valid", "# !@#"])

@pytest.mark.parametrize(
    ("scores", "invalid_indices"),
    [
        ([float("nan"), 1.0, 2.0], [0]),
        ([1.0, float("inf"), 2.0], [1]),
        ([1.0, 2.0, float("-inf")], [2]),
        ([float("nan"), float("inf"), float("-inf")], [0, 1, 2]),
    ],
)
def test_rank_rejects_nonfinite_scores(example, monkeypatch, scores, invalid_indices):
    """All NaN/+inf/-inf scores must be rejected with their candidate indices."""
    observed = {}

    class FakeBM25Okapi:
        def __init__(self, corpus):
            observed["corpus"] = corpus

        def get_scores(self, query):
            observed["query"] = query
            return scores

    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    units = ["first", "second", "third"]
    with pytest.raises(ValueError, match="Found non-finite BM25 scores") as exc:
        bm25.rank(example, units)
    assert f"indices {invalid_indices}" in str(exc.value)
    assert observed["corpus"] == [tokenize.code_tokens(unit) for unit in units]
    assert observed["query"] == tokenize.code_tokens(bm25.query_text(example))


def test_rank_accepts_finite_negative_and_zero_scores(example, monkeypatch):
    """Negative BM25 scores are valid; validation must reject only non-finite values."""
    class FakeBM25Okapi:
        def __init__(self, corpus):
            assert corpus == [["first"], ["second"], ["third"]]

        def get_scores(self, query):
            return [-2.5, 0.0, 1.25]

    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    assert bm25.rank(example, ["first", "second", "third"]) == [
        (2, 1.25), (1, 0.0), (0, -2.5)
    ]


@pytest.mark.parametrize(
    ("scores", "expected_message"),
    [
        ([1.0, 2.0], "Expected 3 BM25 scores, received 2"),
        ([1.0, 2.0, 3.0, 4.0], "Expected 3 BM25 scores, received 4"),
        ([], "Expected 3 BM25 scores, received 0"),
    ],
)
def test_rank_rejects_incorrect_score_count(example, monkeypatch, scores, expected_message):
    """A score must be returned for every candidate, with no extras."""
    class FakeBM25Okapi:
        def __init__(self, corpus):
            assert corpus == [["first"], ["second"], ["third"]]

        def get_scores(self, query):
            return scores

    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    with pytest.raises(ValueError, match=expected_message):
        bm25.rank(example, ["first", "second", "third"])


def test_rank_checks_score_count_before_finiteness(example, monkeypatch):
    """A malformed score vector is rejected for its length first."""
    class FakeBM25Okapi:
        def __init__(self, corpus):
            pass

        def get_scores(self, query):
            return [float("nan"), 1.0]  # Expected three scores

    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    with pytest.raises(ValueError, match="Expected 3 BM25 scores, received 2"):
        bm25.rank(example, ["first", "second", "third"])


def test_rank_accepts_correct_score_count(example, monkeypatch):
    """A complete finite score vector is converted and sorted normally."""
    class FakeBM25Okapi:
        def __init__(self, corpus):
            assert len(corpus) == 3

        def get_scores(self, query):
            return [0, 4, 2]

    monkeypatch.setattr(bm25, "BM25Okapi", FakeBM25Okapi)
    assert bm25.rank(example, ["first", "second", "third"]) == [
        (1, 4.0), (2, 2.0), (0, 0.0)
    ]
