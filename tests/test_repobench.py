import json
import random
from dataclasses import replace
from pathlib import Path

import datasets
import pytest

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


def many(examples, k=100):
    """k Examples with distinct IDs, so seeded draws have room to differ."""
    return [replace(examples[0], example_id=f"fixture:{i}") for i in range(k)]


def test_sample_ids_is_seeded(examples):
    pool = many(examples)
    ids = rb.sample_ids(pool, n=10, seed=7)

    assert ids == rb.sample_ids(pool, n=10, seed=7)
    assert ids != rb.sample_ids(pool, n=10, seed=8)


def test_sample_ids_draws_distinct_ids_from_examples(examples):
    pool = many(examples)
    ids = rb.sample_ids(pool, n=10)

    assert len(ids) == len(set(ids)) == 10
    assert set(ids) <= {e.example_id for e in pool}


def test_sample_ids_n_larger_than_pool_raises(examples):
    with pytest.raises(ValueError):
        rb.sample_ids(examples, n=len(examples) + 1)


def test_sample_ids_leaves_global_random_state_alone(examples):
    state = random.getstate()
    rb.sample_ids(many(examples), n=10, seed=7)
    assert random.getstate() == state


def test_save_then_load_sample_round_trips(examples, tmp_path):
    ids = rb.sample_ids(many(examples), n=10)
    path = tmp_path / "sample.json"

    rb.save_sample(ids, path)

    assert json.loads(path.read_text()) == ids
    assert rb.load_sample(path) == ids
    assert rb.load_sample(str(path)) == ids
