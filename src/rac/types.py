"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Snippet:
    """One cross-file candidate as RepoBench v1.1 ships it."""

    identifier: str
    path: str
    snippet: str


@dataclass(frozen=True)
class Example:
    """
    One RepoBench v1.1 completion task.

    Fields mirror the dataset (tianyang/repobench_python_v1.1). RepoBench has no ID
    column, so `example_id` is derived as "<split>:<row_index>" at a fixed dataset
    revision.
    """

    example_id: str
    repo_name: str
    file_path: str
    context: list[Snippet]
    import_statement: str
    cropped_code: str  # in-file code up to the cursor
    next_line: str  # target
    gold_snippet_index: int  # index into `context` of the snippet the next line needs
    level: str  # RepoBench length bucket, e.g. "2k"
    token_num: int | None = None
    all_code: str | None = None
    created_at: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any], example_id: str) -> Example:
        return cls(
            example_id=example_id,
            repo_name=row["repo_name"],
            file_path=row["file_path"],
            context=[Snippet(**s) for s in row["context"]],
            import_statement=row["import_statement"],
            cropped_code=row["cropped_code"],
            next_line=row["next_line"],
            gold_snippet_index=row["gold_snippet_index"],
            level=row["level"],
            token_num=row.get("token_num"),
            all_code=row.get("all_code"),
            created_at=row.get("created_at"),
        )


@dataclass
class ResultRow:
    """One line of a run's predictions.jsonl. Pass 2 labels are built from these."""

    example_id: str
    method: str
    budget: int
    prediction: str
    em: bool
    es: float  # edit similarity, 0-100 as RepoBench reports it
    gen_seconds: float
    context_tokens: int  # tokens of cross-file context actually used (<= budget)
    extra: dict[str, Any] = field(default_factory=dict)
