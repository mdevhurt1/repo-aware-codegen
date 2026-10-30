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


class _StopAtNewline(StoppingCriteria):
    """Stop once the generated text contains a newline.

    Decodes the new tokens instead of matching token IDs, because BPE tokens
    such as "\\n    " carry the newline together with other characters.
    """

    def __init__(self, tokenizer: Any, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new_text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        return "\n" in new_text


def generate(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 64
) -> tuple[str, float]:
    """Greedy-decode the next line. Return (first generated line, wall-clock seconds)."""
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
            stopping_criteria=StoppingCriteriaList([_StopAtNewline(tokenizer, prompt_len)]),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    return text.split("\n", 1)[0], seconds
