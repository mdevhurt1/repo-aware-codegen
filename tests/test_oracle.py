"""Tests for gold-snippet based context assembly."""

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
