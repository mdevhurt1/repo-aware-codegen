"""Tests for full-snippet BM25 context assembly."""

from dataclasses import replace

import pytest

from rac.context import build_context, methods
from rac.types import Snippet


class CharacterTokenizer:
    """Reversible tokenizer with one token per character."""

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(chr(token_id) for token_id in token_ids)


def make_example(example):
    return replace(
        example,
        context=[
            Snippet(identifier="first", path="first.py", snippet="AAAA"),
            Snippet(identifier="second", path="second.py", snippet="BBBB"),
        ],
    )


def test_full_snippets_sent_to_ranker(examples, monkeypatch):
    long_snippet = "\n".join(f"line_{index}" for index in range(15))
    example = replace(
        examples[0],
        context=[
            Snippet(identifier="long", path="long.py", snippet=long_snippet),
        ],
    )

    def fake_rank(received_example, units):
        assert received_example is example
        assert units == [long_snippet]
        return [(0, 1.0)]

    monkeypatch.setattr(methods.bm25, "rank", fake_rank)

    result = methods.bm25_long(example, 1000, CharacterTokenizer())

    assert result == long_snippet


def test_ranked_order_preserved(examples, monkeypatch):
    example = make_example(examples[0])
    monkeypatch.setattr(
        methods.bm25,
        "rank",
        lambda example, units: [(1, 2.0), (0, 1.0)],
    )

    result = methods.bm25_long(example, 100, CharacterTokenizer())

    assert result == "BBBB\nAAAA"


def test_budget_enforced_through_dispatcher(examples, monkeypatch):
    example = make_example(examples[0])
    monkeypatch.setattr(
        methods.bm25,
        "rank",
        lambda example, units: [(1, 2.0), (0, 1.0)],
    )
    tokenizer = CharacterTokenizer()

    result = build_context(example, "bm25_long", 7, tokenizer)

    assert result == "BBBB\nAA"
    assert len(tokenizer(result, add_special_tokens=False)["input_ids"]) <= 7


@pytest.mark.parametrize("case", ["zero_budget", "empty_context"])
def test_empty_result_skips_ranking(examples, monkeypatch, case):
    example = make_example(examples[0])
    budget = 100

    if case == "zero_budget":
        budget = 0
    else:
        example = replace(example, context=[])

    def unexpected_rank(example, units):
        pytest.fail("Ranking should not run for an empty context or zero budget.")

    monkeypatch.setattr(methods.bm25, "rank", unexpected_rank)

    assert methods.bm25_long(example, budget, CharacterTokenizer()) == ""


def test_negative_budget_rejected(examples):
    with pytest.raises(ValueError, match="non-negative"):
        methods.bm25_long(examples[0], -1, CharacterTokenizer())


def test_gold_label_does_not_override_ranking(examples, monkeypatch):
    example = make_example(examples[0])
    monkeypatch.setattr(
        methods.bm25,
        "rank",
        lambda example, units: [(1, 2.0), (0, 1.0)],
    )
    tokenizer = CharacterTokenizer()

    for gold_index in (0, 1):
        changed_example = replace(
            example,
            gold_snippet_index=gold_index,
            next_line="DO_NOT_INCLUDE_THIS_REFERENCE_ANSWER",
        )

        result = methods.bm25_long(changed_example, 100, tokenizer)

        assert result == "BBBB\nAAAA"