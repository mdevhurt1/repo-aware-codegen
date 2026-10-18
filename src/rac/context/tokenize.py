"""Code-aware lexical tokens for BM25."""

from __future__ import annotations


def code_tokens(text: str) -> list[str]:
    """Split identifiers on snake_case and camelCase, lowercase, drop punctuation.

    e.g. "getUserName(user_id)" -> ["get", "user", "name", "user", "id"]
    """
    raise NotImplementedError("code-aware tokenization")
