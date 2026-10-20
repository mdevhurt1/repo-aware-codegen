import json
from pathlib import Path

import datasets

import rac.data.repobench as rb
from rac.types import Example

FIXTURE = Path(__file__).parent / "fixtures" / "repobench_tiny.jsonl"
ROWS = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def test_load_examples(monkeypatch):
    calls = []

    def fake_load_dataset(path, split, revision):
        calls.append((path, split, revision))
        return ROWS

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    examples = rb.load_examples(split="in_file", revision="abc123")

    ids = [f"in_file:{i}" for i in range(len(ROWS))]
    assert [e.example_id for e in examples] == ids
    assert examples == [Example.from_row(r, i) for r, i in zip(ROWS, ids, strict=True)]
    assert calls == [(rb.DATASET, "in_file", "abc123")]
