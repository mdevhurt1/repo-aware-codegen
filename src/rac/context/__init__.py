"""
build_context(example, method, N, tokenizer) -> str.

Every method returns a cross-file block of at most N tokens, counted with the model's
own tokenizer. The dispatcher enforces that so a builder bug can't silently inflate a
budget. Register new methods (dense, signatures) in METHODS.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rac.context import methods
from rac.types import Example

Builder = Callable[[Example, int, Any], str]

METHODS: dict[str, Builder] = {
    "none": methods.none,
    "bm25_long": methods.bm25_long,
    "bm25_short": methods.bm25_short,
    "oracle": methods.oracle,
}


class BudgetExceeded(ValueError):
    """A builder returned more than N tokens."""


def count_tokens(text: str, tokenizer: Any) -> int:
    """Tokens in `text` under `tokenizer`, without special tokens."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_context(example: Example, method: str, n: int, tokenizer: Any) -> str:
    """Return the cross-file block for `example` using `method`, at most `n` tokens."""
    if method not in METHODS:
        raise ValueError(f"unknown context method {method!r}; known: {sorted(METHODS)}")
    text = METHODS[method](example, n, tokenizer)
    used = count_tokens(text, tokenizer)
    if used > n:
        raise BudgetExceeded(
            f"{method} returned {used} tokens for budget {n} ({example.example_id})"
        )
    return text
