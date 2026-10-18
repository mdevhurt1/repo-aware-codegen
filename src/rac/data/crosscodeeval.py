"""CrossCodeEval Python loader for the final held-out test."""

from __future__ import annotations

from rac.types import Example


def load_examples(path: str) -> list[Example]:
    """Map CrossCodeEval Python records onto Example."""
    raise NotImplementedError("CrossCodeEval final test")
