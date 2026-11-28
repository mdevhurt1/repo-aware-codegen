"""Build scalar feature caches for the examples in a selector dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rac.data import repobench
from rac.selector.cache import SCALAR_COLUMNS, write_family
from rac.selector.features import extract
from rac.types import Example


def build_scalar_cache(
    dataset: dict,
    examples: list[Example],
    output: str | Path,
) -> None:
    """Extract features in dataset order after checking IDs and repositories."""
    rows = dataset["rows"]
    if not rows:
        raise ValueError("Selector dataset rows must not be empty.")

    requested = [row["example_id"] for row in rows]
    if len(set(requested)) != len(requested):
        raise ValueError("Selector dataset example IDs must be unique.")

    requested_set = set(requested)
    selected = {}
    for example in examples:
        if example.example_id not in requested_set:
            continue
        if example.example_id in selected:
            raise ValueError(f"Duplicate source example: {example.example_id}")
        selected[example.example_id] = example

    missing = [key for key in requested if key not in selected]
    if missing:
        raise ValueError(f"Missing source examples: {missing[:5]}")

    # Validate all repositories before extracting features or writing output.
    for row in rows:
        example = selected[row["example_id"]]
        if example.repo_name != row["repo_name"]:
            raise ValueError(f"Repository mismatch: {example.example_id}")

    records = {}
    for example_id in requested:
        features = extract(selected[example_id])
        records[example_id] = [features[name] for name in SCALAR_COLUMNS]

    write_family(output, list(SCALAR_COLUMNS), records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="cross_file_first")
    parser.add_argument(
        "--revision",
        required=True,
        help="Full 40-character dataset commit SHA used by the prediction runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/selector/features/scalars.json"),
    )
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-fA-F]{40}", args.revision) is None:
        parser.error("--revision must be a full 40-character dataset commit SHA.")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    examples = repobench.load_examples(
        split=args.split,
        revision=args.revision,
    )
    build_scalar_cache(dataset, examples, args.output)
    print(f"Saved {len(dataset['rows'])} examples to {args.output}")


if __name__ == "__main__":
    main()