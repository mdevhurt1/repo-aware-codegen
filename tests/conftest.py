import json
from pathlib import Path

import pytest

from rac.types import Example

FIXTURE = Path(__file__).parent / "fixtures" / "repobench_tiny.jsonl"


class WhitespaceTokenizer:
    """Stand-in for a HF tokenizer: one token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split())))}


@pytest.fixture
def examples() -> list[Example]:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    return [Example.from_row(r, f"fixture:{i}") for i, r in enumerate(rows)]


@pytest.fixture
def tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer()
