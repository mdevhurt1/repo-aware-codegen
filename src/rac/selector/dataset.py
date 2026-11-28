"""Assemble selector outcomes and repository-grouped folds."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

BUDGETS = (256, 512, 1024, 2048, 4096)
ARMS = ("none",) + tuple(
    f"{method}:{budget}"
    for method in ("bm25_long", "bm25_short", "signatures")
    for budget in BUDGETS
)


def assemble(
    prediction_files: list[str | Path],
    repositories: dict[str, str],
    seed: int = 0,
) -> dict:
    """Build complete outcomes and five repository-grouped folds.

    Repeated no-context rows collapse only when their outcomes agree.
    Other duplicate arms are rejected. Oracle rows are excluded.

    Ties favor the smallest configured budget, then actual context tokens,
    then the fixed arm order. Real run files must come from the same model,
    dataset revision, and generation settings.
    """
    outcomes: dict[str, dict] = {}

    for filename in prediction_files:
        with Path(filename).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                    method = row["method"]
                    if method in {"oracle", "oracle_long", "oracle_short"}:
                        continue

                    budget = row["budget"]
                    arm = "none" if method == "none" else f"{method}:{budget}"
                    if arm not in ARMS:
                        raise ValueError(f"unsupported arm: {arm}")

                    example_id = row["example_id"]
                    if example_id not in repositories or not repositories[example_id]:
                        raise ValueError(f"missing repository for {example_id}")

                    es = row["es"]
                    tokens = row["context_tokens"]
                    if (
                        isinstance(es, bool)
                        or not isinstance(es, (float, int))
                        or not math.isfinite(es)
                        or not 0 <= es <= 100
                    ):
                        raise ValueError("ES must be finite and between 0 and 100")

                    if type(tokens) is not int or tokens < 0:
                        raise ValueError("context_tokens must be a nonnegative integer")

                    if type(budget) is not int or budget not in BUDGETS or tokens > budget:
                        raise ValueError("invalid budget or context cost")

                    if arm == "none" and tokens != 0:
                        raise ValueError("none must have zero context tokens")

                    value = {
                        "es": float(es),
                        "tokens": tokens,
                        "prediction": row["prediction"],
                    }
                    entries = outcomes.setdefault(example_id, {})
                    if arm in entries:
                        if arm != "none" or entries[arm] != value:
                            raise ValueError(
                                f"duplicate or inconsistent arm: {example_id}/{arm}"
                            )
                    entries[arm] = value

                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{filename}:{line_number}: {exc}") from exc

    if not outcomes:
        raise ValueError("no selectable predictions")

    rows = []
    for example_id, entries in sorted(outcomes.items()):
        missing = set(ARMS) - entries.keys()
        if missing:
            raise ValueError(f"{example_id}: missing arms {sorted(missing)}")

        scores = [entries[arm]["es"] for arm in ARMS]
        costs = [entries[arm]["tokens"] for arm in ARMS]
        winners = [i for i, score in enumerate(scores) if score == max(scores)]
        label = min(
            winners,
            key=lambda i: (
                0 if i == 0 else int(ARMS[i].split(":")[1]),
                costs[i],
                i,
            ),
        )
        rows.append(
            {
                "example_id": example_id,
                "repo_name": repositories[example_id],
                "es": scores,
                "context_tokens": costs,
                "label": ARMS[label],
                "context_helps": max(scores[1:]) > scores[0],
                "tied": len(winners) > 1,
            }
        )

    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["repo_name"], []).append(row["example_id"])

    if len(groups) < 5:
        raise ValueError("five grouped folds require at least five repositories")

    names = sorted(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda name: -len(groups[name]))

    bins: list[list[str]] = [[] for _ in range(5)]
    counts = [0] * 5
    for name in names:
        fold_index = min(range(5), key=lambda i: (counts[i], i))
        bins[fold_index].append(name)
        counts[fold_index] += len(groups[name])

    folds = []
    for test_repos in bins:
        test_set = set(test_repos)
        train_ids = [r["example_id"] for r in rows if r["repo_name"] not in test_set]
        test_ids = [r["example_id"] for r in rows if r["repo_name"] in test_set]
        folds.append(
            {
                "train_ids": train_ids,
                "test_ids": test_ids,
                "train_examples": len(train_ids),
                "test_examples": len(test_ids),
                "train_repositories": len(groups) - len(test_repos),
                "test_repositories": len(test_repos),
            }
        )

    means = [
        sum(row["es"][i] for row in rows) / len(rows)
        for i in range(len(ARMS))
    ]
    return {
        "arms": list(ARMS),
        "rows": rows,
        "folds": folds,
        "seed": seed,
        "mean_context_tokens": {
            arm: sum(row["context_tokens"][i] for row in rows) / len(rows)
            for i, arm in enumerate(ARMS)
        },
        "preview": {
            "none_label_share": sum(r["label"] == "none" for r in rows) / len(rows),
            "tie_share": sum(r["tied"] for r in rows) / len(rows),
            "oracle_choice_es": sum(max(r["es"]) for r in rows) / len(rows),
            "best_fixed_arm": ARMS[max(range(len(ARMS)), key=lambda i: means[i])],
            "best_fixed_es": max(means),
        },
    }


def save(dataset: dict, output: str | Path, folds_dir: str | Path) -> None:
    """Write the outcome matrix and separate fold ID files."""
    output = Path(output)
    folds_dir = Path(folds_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    for index, fold in enumerate(dataset["folds"]):
        path = folds_dir / f"selector_fold_{index}.json"
        path.write_text(json.dumps(fold, indent=2) + "\n", encoding="utf-8")