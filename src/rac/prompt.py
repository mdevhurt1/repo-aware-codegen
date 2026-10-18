"""Prompt assembly."""

from __future__ import annotations

from typing import Any

from rac.types import Example


def build_prompt(example: Example, context_text: str, tokenizer: Any) -> str:
    """Cross-file block (from build_context) + import statements + in-file code to the cursor."""
    raise NotImplementedError("prompt builder")
