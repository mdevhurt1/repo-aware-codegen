"""Tests for gold-snippet based context assembly with optional BM25-ranked fallback."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from rac.context import methods
from rac.types import Snippet

FIXTURE = Path(__file__).parent / "fixtures" / "repobench_tiny.jsonl"

@pytest.fixture
def example(examples):
    return examples[0]


def _with_snippets(example, texts, gold_index=0):
    return replace(
        example,
        context=[Snippet(identifier=f"id{i}", path=f"module{i}.py", snippet=text)
                 for i, text in enumerate(texts)],
        gold_snippet_index=gold_index,
    )


class CharacterTokenizer:
    """Deterministic tokenizer: one character counts as one model token."""

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(text)}


def test_oracle_selects_nonfirst_gold_snippet(example):
    ex = _with_snippets(example, ["first", "second", "gold", "last"], gold_index=2)
    assert methods.oracle(ex, 100, CharacterTokenizer()) == "gold"


@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_oracle_real_dataset_gold_snippet(examples, record_index):
    ex = examples[record_index]
    gold = ex.context[ex.gold_snippet_index].snippet
    tokenizer = CharacterTokenizer()
    result = methods.oracle(ex, len(gold) + 10, tokenizer)
    assert result == gold
    assert len(tokenizer(result, add_special_tokens=False)["input_ids"]) <= len(gold) + 10


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(0, ""), (1, "a"), (3, "abc"), (4, "abcd"), (6, "abcdef"), (10, "abcdef")],
)
def test_oracle_real_fill_token_budget(example, budget, expected):
    ex = _with_snippets(example, ["other", "abcdef"], gold_index=1)
    result = methods.oracle(ex, budget, CharacterTokenizer())
    assert result == expected
    assert len(result) <= budget


def test_oracle_negative_budget_raises(example):
    with pytest.raises(ValueError, match="Token budget must be non-negative"):
        methods.oracle(example, -1, CharacterTokenizer())


def test_oracle_does_not_rank(example, monkeypatch):
    rank_mock = Mock(side_effect=AssertionError("oracle should not rank"))
    monkeypatch.setattr(methods, "rank", rank_mock)
    gold = example.context[example.gold_snippet_index].snippet
    assert methods.oracle(example, len(gold) + 1, CharacterTokenizer()) == gold
    rank_mock.assert_not_called()


def test_oracle_delegates_to_fill_with_only_gold(example, monkeypatch):
    fill_mock = Mock(return_value="filled")
    monkeypatch.setattr(methods, "fill_to_cutoff", fill_mock)
    tokenizer = object()
    assert methods.oracle(example, 42, tokenizer) == "filled"
    fill_mock.assert_called_once_with(
        [example.context[example.gold_snippet_index].snippet], 42, tokenizer
    )


def test_oracle_invalid_gold_index_raises(example):
    ex = replace(example, gold_snippet_index=len(example.context))
    with pytest.raises(IndexError):
        methods.oracle(ex, 100, CharacterTokenizer())


def test_oracle_does_not_mutate_context(example):
    original = tuple(example.context)
    methods.oracle(example, 5, CharacterTokenizer())
    assert tuple(example.context) == original


# Oracle fallback strategies (unit + integration tests)

# Existing gold-only behavior

def test_oracle_default_and_explicit_none_match(example, monkeypatch):
    rank_mock = Mock(side_effect=AssertionError("BM25 must not run"))
    monkeypatch.setattr(methods, "rank", rank_mock)
    tok = CharacterTokenizer()
    gold = example.context[example.gold_snippet_index].snippet
    assert methods.oracle(example, len(gold) + 20, tok) == gold
    assert methods.oracle(example, len(gold) + 20, tok, fallback="none") == gold
    rank_mock.assert_not_called()


@pytest.mark.parametrize("fallback", ["unknown", "BM25_long", "", None])
def test_oracle_invalid_fallback_raises(example, fallback, monkeypatch):
    rank_mock = Mock(side_effect=AssertionError("BM25 must not run"))
    monkeypatch.setattr(methods, "rank", rank_mock)
    with pytest.raises(ValueError, match="Unknown oracle fallback"):
        methods.oracle(example, 100, CharacterTokenizer(), fallback=fallback)
    rank_mock.assert_not_called()


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_single_candidate_skips_rank(example, fallback, monkeypatch):
    ex = _with_snippets(example, ["only gold"], gold_index=0)
    rank_mock = Mock(side_effect=AssertionError("BM25 must not run"))
    monkeypatch.setattr(methods, "rank", rank_mock)
    assert methods.oracle(ex, 100, CharacterTokenizer(), fallback=fallback) == "only gold"
    rank_mock.assert_not_called()


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_gold_first_and_ranking_order(example, fallback, monkeypatch):
    ex = _with_snippets(example, ["low", "GOLD", "high", "middle"], gold_index=1)
    rank_mock = Mock(return_value=[(1, 3.0), (2, 2.0), (0, 1.0)])
    monkeypatch.setattr(methods, "rank", rank_mock)
    tok = CharacterTokenizer()
    result = methods.oracle(ex, 100, tok, fallback=fallback)
    assert result == "GOLD\nhigh\nmiddle\nlow"
    rank_mock.assert_called_once_with(ex, ["low", "high", "middle"])
    assert ex.context[1].snippet == "GOLD"


def test_oracle_long_passes_complete_candidates_to_rank(example, monkeypatch):
    long_candidate = "\n".join(f"line{i}" for i in range(12))
    ex = _with_snippets(example, [long_candidate, "GOLD", "other"], gold_index=1)
    rank_mock = Mock(return_value=[(0, 2.0), (1, 1.0)])
    monkeypatch.setattr(methods, "rank", rank_mock)
    result = methods.oracle(ex, 1000, CharacterTokenizer(), fallback="bm25_long")
    rank_mock.assert_called_once_with(ex, [long_candidate, "other"])
    assert result == f"GOLD\n{long_candidate}\nother"
    assert "line11" in result


def test_oracle_short_truncates_only_fallback_candidates(example, monkeypatch):
    long_candidate = "\n".join(f"line{i}" for i in range(12))
    long_gold = "\n".join(f"gold{i}" for i in range(12))
    ex = _with_snippets(example, [long_candidate, long_gold, "other"], gold_index=1)
    rank_mock = Mock(return_value=[(0, 2.0), (1, 1.0)])
    monkeypatch.setattr(methods, "rank", rank_mock)
    result = methods.oracle(ex, 1000, CharacterTokenizer(), fallback="bm25_short")
    short_candidate = "\n".join(f"line{i}" for i in range(10))
    rank_mock.assert_called_once_with(ex, [short_candidate, "other"])
    assert result == f"{long_gold}\n{short_candidate}\nother"
    assert "gold11" in result
    assert "line10" not in result


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_excludes_gold_by_index_not_by_text(example, fallback, monkeypatch):
    ex = _with_snippets(example, ["same", "same", "different"], gold_index=1)
    rank_mock = Mock(return_value=[(0, 2.0), (1, 1.0)])
    monkeypatch.setattr(methods, "rank", rank_mock)
    assert methods.oracle(ex, 
                          100, 
                          CharacterTokenizer(), 
                          fallback=fallback) == "same\nsame\ndifferent"
    rank_mock.assert_called_once_with(ex, ["same", "different"])


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_propagates_rank_error(example, fallback, monkeypatch):
    monkeypatch.setattr(methods, 
                        "rank", 
                        Mock(side_effect=ValueError("invalid snippet tokens")))
    with pytest.raises(ValueError, match="invalid snippet tokens"):
        methods.oracle(example, 100, CharacterTokenizer(), fallback=fallback)


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_delegates_gold_and_ranked_candidates_to_fill(
    example, 
    fallback, 
    monkeypatch):
    ex = _with_snippets(example, ["A", "GOLD", "B"], gold_index=1)
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(1, 2.0), (0, 1.0)]))
    fill_mock = Mock(return_value="filled")
    monkeypatch.setattr(methods, "fill_to_cutoff", fill_mock)
    tokenizer = object()
    assert methods.oracle(ex, 15, tokenizer, fallback=fallback) == "filled"
    fill_mock.assert_called_once_with(["GOLD", "B", "A"], 15, tokenizer)


# Integration: actual fill_to_cutoff, controlled ranking

@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
@pytest.mark.parametrize("budget, expected", [
    (0, ""), (1, "G"), (4, "GOLD"), (5, "GOLD\n"),
    (7, "GOLD\nAB"), (10, "GOLD\nAB\nCD"), (100, "GOLD\nAB\nCD")
])
def test_oracle_fallback_real_fill_respects_budget(
    example, 
    monkeypatch, 
    fallback, 
    budget, 
    expected):
    ex = _with_snippets(example, ["AB", "GOLD", "CD"], gold_index=1)
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(0, 2.0), (1, 1.0)]))
    tok = CharacterTokenizer()
    result = methods.oracle(ex, budget, tok, fallback=fallback)
    assert result == expected
    assert len(tok(result, add_special_tokens=False)["input_ids"]) <= budget


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_fallback_negative_budget_raises(example, monkeypatch, fallback):
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(0, 1.0)]))
    ex = _with_snippets(example, ["GOLD", "other"], gold_index=0)
    with pytest.raises(ValueError, match="Token budget must be non-negative"):
        methods.oracle(ex, -1, CharacterTokenizer(), fallback=fallback)


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_oracle_fallback_real_dataset_with_fill(examples, 
                                                monkeypatch, 
                                                fallback, 
                                                record_index):
    ex = examples[record_index]
    remaining = [
        s.snippet 
        for i, s in enumerate(ex.context) 
        if i != ex.gold_snippet_index
        ]
    if fallback == "bm25_short":
        remaining = ["\n".join(s.splitlines()[:10]) for s in remaining]
    rank_mock = Mock(
        return_value=[
            (i, float(len(remaining) - i)) 
            for i in range(len(remaining))
            ]
            )
    monkeypatch.setattr(methods, "rank", rank_mock)
    tok = CharacterTokenizer()
    gold = ex.context[ex.gold_snippet_index].snippet
    budget = len(gold) + 5000
    expected = "\n".join([gold, *remaining]) if remaining else gold
    result = methods.oracle(ex, budget, tok, fallback=fallback)
    assert result == expected
    assert result.startswith(gold)
    assert len(result) <= budget
    if remaining:
        rank_mock.assert_called_once_with(ex, remaining)
    else:
        rank_mock.assert_not_called()


@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_fallback_does_not_mutate_example(example, fallback, monkeypatch):
    ex = _with_snippets(example, ["A", "GOLD", "B"], gold_index=1)
    original = tuple(ex.context)
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(0, 2.0), (1, 1.0)]))
    methods.oracle(ex, 100, CharacterTokenizer(), fallback=fallback)
    assert tuple(ex.context) == original


# Optional full pipeline: real BM25Okapi, code tokenizer and fill_to_cutoff.
@pytest.mark.parametrize("fallback", ["bm25_long", "bm25_short"])
def test_oracle_fallback_real_bm25_end_to_end(example, fallback):
    pytest.importorskip("rank_bm25")
    ex = _with_snippets(example, [
        "def apply_discount(total, code):\n    return total",
        "class Item:\n    price = 0",
        "def total(items):\n    return sum(item.price for item in items)",
    ], gold_index=1)
    tok = CharacterTokenizer()
    result = methods.oracle(ex, 1000, tok, fallback=fallback)
    assert result.startswith(ex.context[1].snippet)
    assert result.count(ex.context[1].snippet) == 1
    assert "apply_discount" in result
    assert "def total" in result
    assert len(result) <= 1000
