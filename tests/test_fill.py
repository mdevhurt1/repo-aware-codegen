"""Tests for filling ranked snippets within a model-token budget."""

import pytest

from rac.context.fill import fill_to_cutoff


class CharacterTokenizer:
    """Reversible test tokenizer: each character is one token."""

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


@pytest.mark.parametrize(
    ("units", "budget", "sep", "expected"),
    [
        ([], 10, "\n", ""),
        (["abc"], 0, "\n", ""),
        (["abc"], 10, "\n", "abc"),
        (["abc"], 3, "\n", "abc"),
        (["abcdef"], 3, "\n", "abc"),
        (["abc", "def"], 7, "\n", "abc\ndef"),
        (["abc", "def"], 5, "\n", "abc\nd"),
        (["abc", "def"], 4, "\n", "abc\n"),
        (["abc", "def", "x"], 5, "\n", "abc\nd"),
        (["ab", "cd"], 5, "::", "ab::c"),
        (["ab", "cd"], 3, "", "abc"),
        (["", "abc"], 4, "\n", "\nabc"),
        ([""], 5, "\n", ""),
    ],
)
def test_fill_to_cutoff(units, budget, sep, expected):
    tokenizer = CharacterTokenizer()

    result = fill_to_cutoff(units, budget, tokenizer, sep)

    assert result == expected
    assert len(tokenizer(result, add_special_tokens=False)["input_ids"]) <= budget


def test_negative_budget_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        fill_to_cutoff(["abc"], -1, CharacterTokenizer())


def test_preserves_code_whitespace_when_it_fits():
    text = "def total():\n    return 42\n"

    assert fill_to_cutoff([text], 100, CharacterTokenizer()) == text


def test_rechecks_budget_after_decoding():
    class ExpandingTokenizer(CharacterTokenizer):
        """Simulate decoding that increases the re-encoded token count."""

        def decode(
            self,
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            return super().decode(token_ids) + "!"

    tokenizer = ExpandingTokenizer()

    result = fill_to_cutoff(["abcdef"], 3, tokenizer)

    assert result == "ab!"
    assert len(tokenizer(result, add_special_tokens=False)["input_ids"]) <= 3