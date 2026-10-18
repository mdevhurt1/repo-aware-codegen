"""Model loading and timed generation."""

from __future__ import annotations

from typing import Any


def load_model(name: str = "deepseek-ai/deepseek-coder-1.3b-base", quant: str | None = "4bit"):
    """Return (model, tokenizer); 4-bit via bitsandbytes must fit an 8 GB GPU."""
    raise NotImplementedError("load DeepSeek-Coder-1.3B 4-bit")

def generate(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 64
) -> tuple[str, float]:
    """Greedy-decode the next line. Return (first generated line, wall-clock seconds)."""
    raise NotImplementedError("timed generation")
