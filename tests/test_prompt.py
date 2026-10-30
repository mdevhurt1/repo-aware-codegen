"""Tests for prompt assembly: cross-file block, then imports, then in-file code."""

import dataclasses

import pytest

from rac.prompt import build_prompt


class WordTokenizer:
    """Test tokenizer: one token per whitespace-separated word, with a model window."""

    def __init__(self, model_max_length=10_000):
        self.model_max_length = model_max_length

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


@pytest.fixture
def example(examples):
    return dataclasses.replace(
        examples[0],
        import_statement="from shop.models import Item",
        cropped_code="def total(items):\n    ",
    )


def test_order_is_context_then_imports_then_code(example):
    prompt = build_prompt(example, "class Item: ...", WordTokenizer())

    assert prompt == (
        "class Item: ...\n\n"
        "from shop.models import Item\n\n"
        "def total(items):\n    "
    )


def test_empty_context_adds_no_leading_separator(example):
    prompt = build_prompt(example, "", WordTokenizer())

    assert prompt == "from shop.models import Item\n\ndef total(items):\n    "


def test_empty_imports_add_no_separator(example):
    example = dataclasses.replace(example, import_statement="")

    assert build_prompt(example, "", WordTokenizer()) == "def total(items):\n    "
    assert build_prompt(example, "ctx", WordTokenizer()) == "ctx\n\ndef total(items):\n    "


def test_cropped_code_kept_verbatim_at_the_cursor(example):
    code = "    x = 1\n\n\t# trailing  \n"
    example = dataclasses.replace(example, cropped_code=code)

    assert build_prompt(example, "ctx", WordTokenizer()).endswith("\n\n" + code)


def test_context_kept_verbatim(example):
    context = "  # path: shop/models.py\nclass Item:\n    pass\n"

    assert build_prompt(example, context, WordTokenizer()).startswith(context + "\n\n")


def test_fixture_examples_build(examples):
    for ex in examples:
        prompt = build_prompt(ex, "", WordTokenizer())
        assert ex.import_statement in prompt
        assert prompt.endswith(ex.cropped_code)


def test_prompt_plus_generation_must_fit_the_window(example):
    context = "a b c d e f"
    words = len(build_prompt(example, context, WordTokenizer()).split())
    window = WordTokenizer(model_max_length=words + 4)

    assert build_prompt(example, context, window, max_new_tokens=4)
    with pytest.raises(ValueError):
        build_prompt(example, context, window, max_new_tokens=5)


def test_default_reserves_room_for_generation(example):
    # Prompt alone fits, but not with the default 64 new tokens on top.
    with pytest.raises(ValueError):
        build_prompt(example, "", WordTokenizer(model_max_length=20))
