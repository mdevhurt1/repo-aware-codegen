"""Prompt assembly."""

from __future__ import annotations

from typing import Any

from rac.context import count_tokens
from rac.types import Example


def build_prompt(
    example: Example, context_text: str, tokenizer: Any, max_new_tokens: int = 64
) -> str:
    """Cross-file block (from build_context) + import statements + in-file code to the cursor."""
    prompt = ""
    if context_text:
        prompt += context_text
        prompt += "\n\n"

    if example.import_statement:
        prompt += example.import_statement
        prompt += "\n\n"

    prompt += example.cropped_code

    tokens_used = count_tokens(prompt, tokenizer)
    if tokens_used + max_new_tokens > tokenizer.model_max_length:
        raise ValueError(
            f"Prompt {tokens_used} + {max_new_tokens} new tokens exceeds "
            f"{tokenizer.model_max_length} ({example.example_id})"
        )
    return prompt
