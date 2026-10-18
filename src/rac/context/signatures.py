"""Signature map: function/class signatures parsed with tree-sitter, written as .pyi-style stubs."""

from __future__ import annotations

from typing import Any

from rac.types import Example


def signature_map(example: Example, n: int, tokenizer: Any) -> str:
    raise NotImplementedError("signature map")
