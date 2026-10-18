"""
Pass 2 features, all computed before generation.

In-repo import count, number and total length of candidates, identifier overlap between
the last lines and candidate names, top BM25 score and its gap to the second, in-file
prefix length, RepoBench length level.
"""

from __future__ import annotations

from typing import Any

from rac.types import Example


def extract(example: Example) -> dict[str, Any]:
    raise NotImplementedError("selector features")
