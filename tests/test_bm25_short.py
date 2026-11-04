"""Tests for short-snippet BM25 context assembly."""

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


def test_bm25_short_truncates_each_snippet_to_first_ten_lines(example, monkeypatch):
    texts = ["\n".join(f"line_{i}" for i in range(12)), "one\ntwo\nthree"]
    ex = _with_snippets(example, texts)
    expected = ["\n".join(f"line_{i}" for i in range(10)), "one\ntwo\nthree"]
    rank_mock = Mock(return_value=[(0, 2.0), (1, 1.0)])
    fill_mock = Mock(return_value="assembled context")
    monkeypatch.setattr(methods, "rank", rank_mock)
    monkeypatch.setattr(methods, "fill_to_cutoff", fill_mock)
    tokenizer = object()

    assert methods.bm25_short(ex, 40, tokenizer) == "assembled context"
    rank_mock.assert_called_once_with(ex, expected)
    fill_mock.assert_called_once_with(expected, 40, tokenizer)


def test_bm25_short_respects_bm25_order_not_score_values(example, monkeypatch):
    ex = _with_snippets(example, ["alpha", "beta", "gamma"])
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(2, 9.0), (0, 4.0), (1, 1.0)]))
    fill_mock = Mock(return_value="filled")
    monkeypatch.setattr(methods, "fill_to_cutoff", fill_mock)
    tokenizer = object()

    assert methods.bm25_short(ex, 12, tokenizer) == "filled"
    fill_mock.assert_called_once_with(["gamma", "alpha", "beta"], 12, tokenizer)


def test_bm25_short_preserves_original_snippets(example, monkeypatch):
    ex = _with_snippets(example, ["\n".join(f"line{i}" for i in range(11)), "second"])
    original = [s.snippet for s in ex.context]
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(1, 1.0), (0, 0.0)]))
    monkeypatch.setattr(methods, "fill_to_cutoff", Mock(return_value="ok"))

    methods.bm25_short(ex, 10, object())
    assert [s.snippet for s in ex.context] == original


@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_bm25_short_uses_real_dataset_candidates(examples, record_index, monkeypatch):
    ex = examples[record_index]
    expected = ["\n".join(s.snippet.splitlines()[:10]) for s in ex.context]
    order = list(reversed(range(len(expected))))
    rank_mock = Mock(return_value=[(i, float(len(expected) - pos))
                                   for pos, i in enumerate(order)])
    fill_mock = Mock(return_value="dataset context")
    monkeypatch.setattr(methods, "rank", rank_mock)
    monkeypatch.setattr(methods, "fill_to_cutoff", fill_mock)
    tokenizer = object()

    assert methods.bm25_short(ex, 50, tokenizer) == "dataset context"
    rank_mock.assert_called_once_with(ex, expected)
    fill_mock.assert_called_once_with([expected[i] for i in order], 50, tokenizer)


def test_bm25_short_propagates_rank_validation_error(example, monkeypatch):
    ex = _with_snippets(example, [])
    monkeypatch.setattr(methods, "rank", Mock(side_effect=ValueError("units must not be empty")))
    fill_mock = Mock()
    monkeypatch.setattr(methods, "fill_to_cutoff", fill_mock)

    with pytest.raises(ValueError, match="units must not be empty"):
        methods.bm25_short(ex, 10, object())
    fill_mock.assert_not_called()


def test_bm25_short_propagates_fill_error(example, monkeypatch):
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(0, 1.0), (1, 0.0)]))
    monkeypatch.setattr(methods, "fill_to_cutoff", Mock(side_effect=ValueError("invalid budget")))

    with pytest.raises(ValueError, match="invalid budget"):
        methods.bm25_short(example, -1, object())


# Integration Testing with real fill_to_cutoff()
class CharacterTokenizer:
    """Deterministic test tokenizer: one character = one model token."""

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(text)}


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (0, ""),
        (2, "ga"),
        (5, "gamma"),
        (6, "gamma\n"),
        (9, "gamma\nbet"),
        (10, "gamma\nbeta"),
        (16, "gamma\nbeta\nalpha"),
        (100, "gamma\nbeta\nalpha"),
    ],
)
def test_bm25_short_real_fill_respects_budget_and_rank(example, monkeypatch, budget, expected):
    ex = _with_snippets(example, ["alpha", "beta", "gamma"])
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(2, 3.0), (1, 2.0), (0, 1.0)]))
    result = methods.bm25_short(ex, budget, CharacterTokenizer())
    assert result == expected
    assert len(result) <= budget


def test_bm25_short_real_fill_truncates_at_ten_lines(example, monkeypatch):
    first = "\n".join(f"line{i}" for i in range(12))
    ex = _with_snippets(example, [first, "other"])
    rank_mock = Mock(return_value=[(0, 2.0), (1, 1.0)])
    monkeypatch.setattr(methods, "rank", rank_mock)
    expected_first = "\n".join(f"line{i}" for i in range(10))
    result = methods.bm25_short(ex, 1000, CharacterTokenizer())
    rank_mock.assert_called_once_with(ex, [expected_first, "other"])
    assert result == expected_first + "\nother"
    assert "line10" not in result and "line11" not in result


@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_bm25_short_real_fill_on_dataset(examples, record_index, monkeypatch):
    ex = examples[record_index]
    short = ["\n".join(s.snippet.splitlines()[:10]) for s in ex.context]
    order = list(reversed(range(len(short))))
    monkeypatch.setattr(methods, 
                        "rank", 
                        Mock(return_value=[
                            (i, float(len(short)-p)) for p, i in enumerate(order)
                            ]))
    budget = 40
    result = methods.bm25_short(ex, budget, CharacterTokenizer())
    expected_full = "\n".join(short[i] for i in order)
    assert result == expected_full[:budget]
    assert len(result) <= budget


def test_bm25_short_real_fill_rejects_negative_budget(example, monkeypatch):
    ex = _with_snippets(example, ["alpha"])
    monkeypatch.setattr(methods, "rank", Mock(return_value=[(0, 1.0)]))
    with pytest.raises(ValueError, match="Token budget must be non-negative"):
        methods.bm25_short(ex, -1, CharacterTokenizer())
