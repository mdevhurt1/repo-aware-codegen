"""Exact match and edit similarity as RepoBench defines them."""

from __future__ import annotations

from rapidfuzz import fuzz


def exact_match(prediction: str, reference: str) -> bool:
    """RepoBench EM on the next line (compare after stripping whitespace).

    Like RepoBench's pred.split() == gt.split(), this also ignores runs of
    whitespace inside the line, not just at its ends.
    """
    return prediction.split() == reference.split()


def edit_similarity(prediction: str, reference: str) -> float:
    """RepoBench ES: fuzz.ratio, 0-100.

    Scores the raw strings, as RepoBench does. rapidfuzz returns an unrounded
    float where RepoBench's fuzzywuzzy rounds to an int.
    """
    return fuzz.ratio(prediction, reference)
