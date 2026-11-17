"""Shared token counting for context budgets and model prompts."""

from typing import Any


def count_tokens(text: str, tokenizer: Any, special_tokens: bool = False) -> int:
    """Count context tokens by default; include special tokens for full prompts."""
    return len(tokenizer(text, add_special_tokens=special_tokens)["input_ids"])
