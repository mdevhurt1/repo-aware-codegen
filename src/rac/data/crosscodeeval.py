"""Load local Python CrossCodeEval JSONL records for held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from rac.types import Example, Snippet


def _string(record: dict, key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def load_examples(path: str | Path) -> list[Example]:
    """Map baseline or retrieval-augmented Python records to Examples.

    Baseline rows have no candidates. Augmented rows use the structured
    crossfile_context.list chunks in their supplied order, not the formatted
    context text. Use only retrieval generated without the reference answer.

    The prompt already contains in-file imports; preserve it as supplied.
    CrossCodeEval supplies no RepoBench gold index or length bucket. We use
    -1 and "unknown", respectively. Do not use these examples with oracle.
    """
    path = Path(path)
    examples: list[Example] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                meta = row["metadata"]
                task_id = _string(meta, "task_id")
                if not task_id.strip() or task_id in seen:
                    raise ValueError(f"empty or duplicate task_id: {task_id!r}")

                retrieved = row.get("crossfile_context")
                if retrieved is None or retrieved == "" or retrieved == {}:
                    chunks = []
                elif isinstance(retrieved, dict) and isinstance(retrieved.get("list"), list):
                    chunks = retrieved["list"]
                else:
                    raise ValueError("crossfile_context must contain a structured chunk list")

                example = Example(
                    example_id=task_id,
                    repo_name=_string(meta, "repository"),
                    file_path=_string(meta, "file"),
                    context=[
                        Snippet(
                            identifier=f"{task_id}:{index}",
                            path=_string(chunk, "filename"),
                            snippet=_string(chunk, "retrieved_chunk"),
                        )
                        for index, chunk in enumerate(chunks)
                    ],
                    import_statement="",
                    cropped_code=_string(row, "prompt"),
                    next_line=_string(row, "groundtruth"),
                    gold_snippet_index=-1,
                    level="unknown",
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

            seen.add(task_id)
            examples.append(example)
    return examples
