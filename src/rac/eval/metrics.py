"""Exact match and edit similarity as RepoBench defines them."""

from __future__ import annotations


def exact_match(prediction: str, reference: str) -> bool:
    """RepoBench EM on the next line (compare after stripping whitespace)."""
    raise NotImplementedError("exact match")


def edit_similarity(prediction: str, reference: str) -> float:
    """RepoBench ES: fuzz.ratio, 0-100."""
    raise NotImplementedError("edit similarity")
