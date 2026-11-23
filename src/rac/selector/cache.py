"""Store feature families and assemble matrices in explicit example-ID order."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCALAR_COLUMNS = (
    "import_count",
    "candidate_count",
    "candidate_total_chars",
    "identifier_overlap",
    "bm25_top_score",
    "bm25_score_gap",
    "prefix_chars",
)


def write_family(
    path: str | Path,
    columns: list[str],
    records: dict[str, list[float]],
) -> None:
    """Save one numeric feature family, keyed by example ID.

    Paths should normally be under outputs/selector/features/.
    """
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Feature columns must be nonempty and unique.")
    if not records:
        raise ValueError("Feature records must not be empty.")

    cleaned = {}
    for example_id, values in records.items():
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("Example IDs must be nonempty strings.")
        array = np.asarray(values, dtype=float)
        if array.shape != (len(columns),) or not np.isfinite(array).all():
            raise ValueError(f"{example_id}: expected finite values for every column.")
        cleaned[example_id] = array.tolist()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "columns": columns,
        "records": cleaned,
    }
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_matrix(
    paths: list[str | Path],
    example_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Join feature families in caller-specified row and family order.

    Missing IDs raise an error rather than silently removing examples.
    """
    if not paths or not example_ids:
        raise ValueError("Provide feature families and example IDs.")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("Example IDs must be unique.")

    matrices = []
    names = []
    for path in paths:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"{path}: unsupported feature-cache version.")

        columns = payload["columns"]
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError(f"{path}: invalid feature columns.")

        records = payload["records"]
        missing = [example_id for example_id in example_ids if example_id not in records]
        if missing:
            raise ValueError(f"{path}: missing example IDs: {missing[:5]}")

        matrix = np.asarray([records[key] for key in example_ids], dtype=float)
        if matrix.shape != (len(example_ids), len(columns)):
            raise ValueError(f"{path}: feature dimensions do not match columns.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path}: nonfinite feature values.")

        qualified = [f"{path.stem}:{column}" for column in columns]
        if set(qualified) & set(names):
            raise ValueError("Feature family names overlap; use distinct filenames.")
        matrices.append(matrix)
        names.extend(qualified)

    return np.concatenate(matrices, axis=1), names