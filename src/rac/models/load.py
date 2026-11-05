"""Model loading and timed generation."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


def load_model(name: str = "deepseek-ai/deepseek-coder-1.3b-base", quant: str | None = "4bit"):
    """Return (model, tokenizer); 4-bit via bitsandbytes must fit an 8 GB GPU."""
    if quant == "4bit":
        # NF4 weights with fp16 compute: about 1 GB of weights for the 1.3B model.
        kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            "device_map": "auto",
        }
    elif quant is None:
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    else:
        raise ValueError(f"Unsupported quant {quant!r}; expected '4bit' or None.")

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


_BLOCK_QUOTES = ('"""', "'''")


def _code_lines(text: str):
    """Yield the code lines of text: not blank, a # comment, or inside a triple-quoted block."""
    open_quote = None
    for line in text.split("\n"):
        stripped = line.strip()
        if open_quote:
            # An odd number of quotes on the line closes the block.
            if stripped.count(open_quote) % 2:
                open_quote = None
            continue
        quote = next((q for q in _BLOCK_QUOTES if stripped.startswith(q)), None)
        if quote:
            # Even count ("""One line.""") opens and closes on the same line.
            if stripped.count(quote) % 2:
                open_quote = quote
            continue
        if stripped and not stripped.startswith("#"):
            yield line


def _first_code_line(text: str) -> str | None:
    """First line of text that is code, with its indentation; None if there is none."""
    return next(_code_lines(text), None)


class _StopAfterCodeLine(StoppingCriteria):
    """Stop once the generated text contains a completed line of code.

    Blank and comment-only lines do not stop generation, so the model keeps going
    until it produces a real statement. Decodes the new tokens instead of matching
    token IDs, because BPE tokens such as "\\n    " carry the newline together with
    other characters.
    """

    def __init__(self, tokenizer: Any, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new_text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        completed = new_text.rsplit("\n", 1)[0] if "\n" in new_text else ""
        return _first_code_line(completed) is not None


def generate(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 64
) -> tuple[str, float]:
    """Greedy-decode up to the next line of code.

    Return (first generated code line, wall-clock seconds). Blank and comment-only
    lines are skipped. If no code line appears within max_new_tokens, the line is
    "" unless the unterminated tail is code, in which case that partial line is returned.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Time only the decode; synchronize so queued CUDA work is counted.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            stopping_criteria=StoppingCriteriaList([_StopAfterCodeLine(tokenizer, prompt_len)]),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    return _first_code_line(text) or "", seconds
