"""Prompt assembly."""

from __future__ import annotations

from typing import Any

from rac.types import Example


def build_prompt(example: Example, context_text: str, tokenizer: Any) -> str:
    """Cross-file block (from build_context) + import statements + in-file code to the cursor."""
    prompt = ""
    if context_text:
        prompt += context_text
        prompt += "\n\n"
    
    prompt += example.import_statement
    prompt += "\n\n"

    prompt += example.cropped_code

    if len(tokenizer.tokenize(prompt)) > tokenizer.model_max_length:
        raise ValueError("Prompt exceeds model max length.")

    return prompt