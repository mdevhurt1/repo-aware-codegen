"""RepoBench v1.1 Python loading and the seeded sample."""

from __future__ import annotations

import os
from pathlib import Path

import urllib

from rac.types import Example

DATASET = "tianyang/repobench_python_v1.1"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100

def _fetch_page(split: str = "cross_file_first", offset: int = 0, length: int = PAGE):
    """Fetch a page of rows from the RepoBench dataset."""
    # build the URL for the request
    request_url = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": length})

    # Check if the huggingface token is set in the environment variables
    token = None
    if "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]

    # Make the request to the Hugging Face datasets server
    response = None

    for attempt in range(3):  # Retry up to 3 times
        response = urllib.request.urlopen(f"{ROWS_URL}?{request_url}", headers={"Authorization": f"Bearer {token}"} if token else {})
        if response.status == 200:
            break
        
    return response

def get_dataset_sha():
    """Get the SHA of the RepoBench dataset."""
    response = urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}")
    return response["sha"]

def load_examples(split: str = "cross_file_first", revision: str | None = None) -> list[Example]:
    """Load one RepoBench split as Examples with example_id = "<split>:<row_index>"."""
    if revision and revision != get_dataset_sha():
        raise ValueError(f"Revision {revision} does not match the current dataset SHA {get_dataset_sha()}.")

    # Check for a cache in data/raw/repobench/<split>@<sha[:12]>.jsonl
    cache_path = Path(f"data/raw/repobench/{split}@{get_dataset_sha()[:12]}.jsonl")
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return [Example.parse_raw(line) for line in f]

    num_rows_total = 8083
    examples = []
    offset = 0

    while offset < num_rows_total:
        response = _fetch_page(split, offset, PAGE)
        if response is None:
            raise ValueError("Failed to fetch data from the RepoBench dataset.")

        rows = response["rows"]
        for i, row in enumerate(rows):
            example_id = f"{split}:{offset + i}"
            example = Example(id=example_id, data=row)
            examples.append(example)

        offset += PAGE

    # Save the examples to the cache
    with open(cache_path, "w") as f:
        for example in examples:
            f.write(example.json() + "\n")

    return examples


def sample_ids(examples: list[Example], n: int = 500, seed: int = 0) -> list[str]:
    """Seeded random sample of example IDs."""
    raise NotImplementedError("seeded sample")


def save_sample(ids: list[str], path: str | Path) -> None:
    """Write sample IDs to data/samples/<name>.json."""
    raise NotImplementedError("save sample")


def load_sample(path: str | Path) -> list[str]:
    """Read sample IDs written by save_sample."""
    raise NotImplementedError("load sample")
    
