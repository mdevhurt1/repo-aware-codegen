"""Contract tests for the interfaces."""

import dataclasses

import pytest

from rac import context
from rac.context import BudgetExceeded, build_context
from rac.types import ResultRow


def test_fixture_parses_into_examples(examples):
    assert len(examples) == 3
    for ex in examples:
        assert 0 <= ex.gold_snippet_index < len(ex.context)
        assert ex.next_line.strip()


def test_milestone1_methods_registered():
    assert {"none", "bm25_long", "bm25_short", "oracle"} <= set(context.METHODS)


def test_none_returns_empty(examples, tokenizer):
    assert build_context(examples[0], "none", 512, tokenizer) == ""


def test_unknown_method_rejected(examples, tokenizer):
    with pytest.raises(ValueError):
        build_context(examples[0], "no_such_method", 512, tokenizer)


def test_budget_enforced(examples, tokenizer, monkeypatch):
    monkeypatch.setitem(context.METHODS, "too_long", lambda ex, n, tok: "word " * (n + 1))
    with pytest.raises(BudgetExceeded):
        build_context(examples[0], "too_long", 8, tokenizer)


def test_result_row_fields():
    names = {f.name for f in dataclasses.fields(ResultRow)}
    assert {"example_id", "method", "budget", "prediction", "em", "es",
            "gen_seconds", "context_tokens"} <= names
