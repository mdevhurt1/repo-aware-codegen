"""RepoBench v1.1 Python loading and the seeded sample."""

from __future__ import annotations

import json
import random
from pathlib import Path

from rac.types import Example

DATASET = "tianyang/repobench_python_v1.1"


def load_examples(split: str = "cross_file_first", revision: str | None = None) -> list[Example]:
    """Load one RepoBench split as Examples with example_id = "<split>:<row_index>".

    Downloads the split's parquet files from the Hub (cached under ~/.cache/huggingface);
    pass a commit sha as `revision` so example IDs stay stable.
    """
    from datasets import load_dataset  # deferred: heavy import, not needed by the sample helpers

    ds = load_dataset(DATASET, split=split, revision=revision)
    return [Example.from_row(row, f"{split}:{i}") for i, row in enumerate(ds)]


def sample_ids(examples: list[Example], n: int = 500, seed: int = 0) -> list[str]:
    """Seeded random sample of example IDs."""
    return random.Random(seed).sample([ex.example_id for ex in examples], n)


def save_sample(ids: list[str], path: str | Path) -> None:
    """Write sample IDs to data/samples/<name>.json."""
    with open(path, "w") as f:
        json.dump(ids, f)


def load_sample(path: str | Path) -> list[str]:
    """Read sample IDs written by save_sample."""
    with open(path) as f:
        return json.load(f)
