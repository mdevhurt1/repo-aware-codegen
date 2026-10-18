"""
Budget fill rule.

Fill to cutoff: take units in rank order and append until the budget is reached; the
unit that crosses N is truncated at the token boundary so the block is <= N tokens.
"""

from __future__ import annotations

from typing import Any


def fill_to_cutoff(units: list[str], n: int, tokenizer: Any, sep: str = "\n") -> str:
    """Concatenate ranked `units` with `sep`, truncating the crossing unit, to <= n tokens."""
    raise NotImplementedError("fill to cutoff")
