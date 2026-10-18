"""RepoBench v1.1 Python loading and the seeded sample."""

from __future__ import annotations

from pathlib import Path

from rac.types import Example

DATASET = "tianyang/repobench_python_v1.1"

def load_examples(split: str = "cross_file_first", revision: str | None = None) -> list[Example]:
    """Load one RepoBench split as Examples with example_id = "<split>:<row_index>"."""
    raise NotImplementedError("load RepoBench v1.1")


def sample_ids(examples: list[Example], n: int = 500, seed: int = 0) -> list[str]:
    """Seeded random sample of example IDs."""
    raise NotImplementedError("seeded sample")


def save_sample(ids: list[str], path: str | Path) -> None:
    """Write sample IDs to data/samples/<name>.json."""
    raise NotImplementedError("save sample")


def load_sample(path: str | Path) -> list[str]:
    """Read sample IDs written by save_sample."""
    raise NotImplementedError("load sample")
