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


class CountingTokenizer(CharacterTokenizer):
    """Character tokenizer that records how many characters it was asked to tokenize."""

    def __init__(self):
        self.calls = 0
        self.chars = 0

    def __call__(self, text, add_special_tokens=True):
        self.calls += 1
        self.chars += len(text)
        return super().__call__(text, add_special_tokens)


def test_oversized_unit_needs_few_tokenizer_calls():
    """A cut must not retokenize once per character of the oversized unit."""
    tokenizer = CountingTokenizer()
    budget = 500

    result = fill_to_cutoff(["x" * 200_000], budget, tokenizer)

    assert result == "x" * budget
    assert tokenizer.calls < 100


def test_join_is_counted_as_a_whole_not_unit_by_unit():
    """Summing per-unit counts undercounts here, and the budget check would then raise."""

    class LeadingSpaceFreeTokenizer:
        """One token per character, except that leading spaces cost nothing."""

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": list(range(len(text.lstrip(" "))))}

    # "a" and " b" cost 1 each, but "a b" costs 3.
    result = fill_to_cutoff(["a", " b"], 2, LeadingSpaceFreeTokenizer(), sep="")

    assert result == "a "


def test_tokens_merging_across_units_leave_room_for_more():
    """Summing per-unit counts overcounts here and would cut text that fits."""

    class MergingTokenizer:
        """Each character is a token, but "ab" merges into one."""

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": list(range(len(text.replace("ab", "X"))))}

    assert fill_to_cutoff(["a", "b", "c"], 2, MergingTokenizer(), sep="") == "abc"


def test_stops_after_the_first_unit_that_does_not_fit():
    result = fill_to_cutoff(["aaaa", "bbbbbbbb", "c"], 6, CharacterTokenizer())

    assert result == "aaaa\nb"


def test_many_units_are_not_all_tokenized_after_the_budget_is_spent():
    tokenizer = CountingTokenizer()

    fill_to_cutoff(["x" * 10] * 10_000, 50, tokenizer)

    assert tokenizer.calls < 100
