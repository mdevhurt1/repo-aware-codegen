"""Tests for model loading and timed generation, using CPU stand-ins."""

import pytest
import torch
from transformers import BatchEncoding

from rac.models import load
from rac.models.load import generate, load_model


class CharTokenizer:
    """Stand-in for a HF tokenizer: one token per character."""

    pad_token_id = None
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        return BatchEncoding({"input_ids": torch.tensor([[ord(c) for c in text]])})

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids.tolist() if i != self.eos_token_id)


class ScriptedModel:
    """Emits a fixed continuation one token at a time, honoring stopping criteria."""

    device = torch.device("cpu")

    def __init__(self, continuation):
        self.continuation = continuation
        self.kwargs = None
        self.steps = 0

    def generate(self, input_ids, max_new_tokens, stopping_criteria, **kwargs):
        self.kwargs = {"max_new_tokens": max_new_tokens, **kwargs}
        ids = input_ids
        for char in self.continuation[:max_new_tokens]:
            ids = torch.cat([ids, torch.tensor([[ord(char)]])], dim=1)
            self.steps += 1
            if stopping_criteria(ids, None):
                break
        return ids


def test_returns_first_generated_line_and_seconds():
    model = ScriptedModel("x = 1\ny = 2\n")

    line, seconds = generate(model, CharTokenizer(), "def f():\n    ", max_new_tokens=64)

    assert line == "x = 1"
    assert isinstance(seconds, float)
    assert seconds >= 0


def test_greedy_and_max_new_tokens_passed():
    model = ScriptedModel("abc")

    generate(model, CharTokenizer(), "p", max_new_tokens=7)

    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 7


def test_pad_falls_back_to_eos():
    model = ScriptedModel("abc")

    generate(model, CharTokenizer(), "p")

    assert model.kwargs["pad_token_id"] == CharTokenizer.eos_token_id


def test_stops_at_first_newline():
    model = ScriptedModel("ab\ncdef\n")

    generate(model, CharTokenizer(), "p")

    assert model.steps == 3


def test_newline_in_prompt_does_not_stop():
    model = ScriptedModel("abc")

    line, _ = generate(model, CharTokenizer(), "line one\nline two\n")

    assert line == "abc"
    assert model.steps == 3


def test_blank_lines_are_skipped():
    model = ScriptedModel("\n\nabc\nd")

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "abc"
    assert model.steps == 6


def test_comment_lines_are_skipped():
    model = ScriptedModel("# todo\n    # more\nx = 1\ny = 2\n")

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1"
    assert model.steps == len("# todo\n    # more\nx = 1\n")


def test_triple_double_quote_lines_are_skipped():
    model = ScriptedModel('"""Docstring."""\n    """More."""\nx = 1\ny = 2\n')

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1"
    assert model.steps == len('"""Docstring."""\n    """More."""\nx = 1\n')


def test_triple_single_quote_lines_are_skipped():
    model = ScriptedModel("'''Docstring.'''\n    '''More.'''\nx = 1\ny = 2\n")

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1"
    assert model.steps == len("'''Docstring.'''\n    '''More.'''\nx = 1\n")


def test_multiline_double_quote_block_is_skipped():
    model = ScriptedModel('"""\n    Summary text.\n    """\nx = 1\ny = 2\n')

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1"
    assert model.steps == len('"""\n    Summary text.\n    """\nx = 1\n')


def test_multiline_single_quote_block_is_skipped():
    model = ScriptedModel("'''\n    Summary text.\n    '''\nx = 1\ny = 2\n")

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1"
    assert model.steps == len("'''\n    Summary text.\n    '''\nx = 1\n")


def test_multiline_block_with_text_on_opening_line_is_skipped():
    model = ScriptedModel('"""Summary.\n\n    More text.\n    """\nx = 1\n')

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1"


def test_multiline_block_after_comment_is_skipped():
    model = ScriptedModel('# note\n"""\nbody\n"""\nx = 1\n')

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1"


def test_unclosed_block_within_cap_gives_empty_line():
    model = ScriptedModel('"""\nbody\nmore body\n')

    line, _ = generate(model, CharTokenizer(), "p", max_new_tokens=12)

    assert line == ""


def test_indented_code_line_keeps_indent():
    model = ScriptedModel("\n    return 1\n")

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "    return 1"


def test_only_comments_within_cap_gives_empty_line():
    model = ScriptedModel("# a\n# b\n# c")

    line, _ = generate(model, CharTokenizer(), "p", max_new_tokens=8)

    assert line == ""


def test_trailing_comment_on_code_line_is_code():
    model = ScriptedModel("x = 1  # set x\ny\n")

    line, _ = generate(model, CharTokenizer(), "p")

    assert line == "x = 1  # set x"


def test_max_new_tokens_caps_unterminated_line():
    model = ScriptedModel("abcdef")

    line, _ = generate(model, CharTokenizer(), "p", max_new_tokens=4)

    assert line == "abcd"


def test_unknown_quant_rejected():
    with pytest.raises(ValueError, match="quant"):
        load_model(quant="8bit")


def test_4bit_load_uses_nf4(monkeypatch):
    calls = {}

    class FakeModel:
        def eval(self):
            calls["eval"] = True
            return self

    def fake_model_from_pretrained(name, **kwargs):
        calls["model"] = (name, kwargs)
        return FakeModel()

    monkeypatch.setattr(load.AutoTokenizer, "from_pretrained", lambda name: f"tok:{name}")
    monkeypatch.setattr(load.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    model, tokenizer = load_model("some/model")

    name, kwargs = calls["model"]
    config = kwargs["quantization_config"]
    assert name == "some/model"
    assert tokenizer == "tok:some/model"
    assert isinstance(model, FakeModel)
    assert calls["eval"]
    assert config.load_in_4bit
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.float16
