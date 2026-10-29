"""Tests for filling ranked snippets within a model-token budget."""

import pytest

from rac.context.fill import fill_to_cutoff


class CharacterTokenizer:
    """Test tokenizer: each character counts as one token."""

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


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
    assert sep.join(units).startswith(result)
    assert len(tokenizer(result, add_special_tokens=False)["input_ids"]) <= budget


def test_negative_budget_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        fill_to_cutoff(["abc"], -1, CharacterTokenizer())


def test_preserves_code_whitespace_when_it_fits():
    text = "def total():\n    return 42\n"

    assert fill_to_cutoff([text], 100, CharacterTokenizer()) == text


def test_unicode_cut_preserves_original_characters():
    class Utf8Tokenizer:
        """Each UTF-8 byte counts as one token; decoding is forbidden."""

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": list(text.encode("utf-8"))}

        def decode(self, *args, **kwargs):
            pytest.fail("Budget filling must not decode partial token sequences.")

    tokenizer = Utf8Tokenizer()
    original = "# 日本語コメント\ndef f():\n    return 'ü'\n"

    # "# " takes two bytes and each Japanese character takes three.
    result = fill_to_cutoff([original], 6, tokenizer)

    assert result == "# 日"
    assert original.startswith(result)
    assert "\ufffd" not in result
    assert len(tokenizer(result, add_special_tokens=False)["input_ids"]) <= 6


def test_longest_prefix_with_nonmonotonic_token_counts():
    class NonmonotonicTokenizer:
        """A longer prefix can have fewer tokens due to token merging."""

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            counts = {
                "": 0,
                "a": 1,
                "ab": 2,
                "abc": 1,
                "abcd": 3,
            }
            return {"input_ids": list(range(counts[text]))}

    result = fill_to_cutoff(["abcd"], 1, NonmonotonicTokenizer())

    assert result == "abc"