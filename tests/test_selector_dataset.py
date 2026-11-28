"""Test selector labels, complete outcomes, and repository-grouped folds."""

import json

import pytest

from rac.selector.dataset import ARMS, assemble, save


def make_runs(tmp_path):
    rows = []
    repositories = {}

    for index in range(10):
        example_id = f"sample:{index}"
        repositories[example_id] = f"repo/{index // 2}"

        for arm in ARMS:
            method, _, budget = arm.partition(":")
            rows.append(
                {
                    "example_id": example_id,
                    "method": method,
                    "budget": int(budget or 512),
                    "es": 50,
                    "context_tokens": 0 if method == "none" else 100,
                    "prediction": "return x",
                }
            )

    path = tmp_path / "predictions.jsonl"
    write_rows(path, rows)
    return path, repositories, rows


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_complete_matrix_ties_and_repository_grouping(tmp_path):
    path, repositories, _ = make_runs(tmp_path)

    dataset = assemble([path], repositories)

    assert len(dataset["arms"]) == 16
    assert len(dataset["rows"]) == 10
    assert all(
        row["label"] == "none" and not row["context_helps"]
        for row in dataset["rows"]
    )
    assert dataset["preview"]["tie_share"] == 1
    assert dataset["preview"]["none_label_share"] == 1
    assert dataset["preview"]["oracle_choice_es"] == 50
    assert dataset["preview"]["best_fixed_es"] == 50
    assert assemble([path], repositories) == dataset

    tested = []
    for fold in dataset["folds"]:
        train_repos = {repositories[i] for i in fold["train_ids"]}
        test_repos = {repositories[i] for i in fold["test_ids"]}

        assert train_repos.isdisjoint(test_repos)
        assert set(fold["train_ids"]) | set(fold["test_ids"]) == set(repositories)
        tested.extend(fold["test_ids"])

    assert sorted(tested) == sorted(repositories)

    save(dataset, tmp_path / "output/data.json", tmp_path / "folds")

    assert json.loads((tmp_path / "output/data.json").read_text()) == dataset
    assert len(list((tmp_path / "folds").glob("*.json"))) == 5


def test_missing_arm_rejected(tmp_path):
    path, repositories, rows = make_runs(tmp_path)
    write_rows(path, rows[:-1])

    with pytest.raises(ValueError, match="missing arms"):
        assemble([path], repositories)


def test_consistent_none_duplicates_and_strict_gate(tmp_path):
    path, repositories, rows = make_runs(tmp_path)
    rows.append(dict(rows[0], budget=1024))
    rows[1]["es"] = 60
    write_rows(path, rows)

    dataset = assemble([path], repositories)

    assert dataset["rows"][0]["label"] == "bm25_long:256"
    assert dataset["rows"][0]["context_helps"]
    assert dataset["preview"]["oracle_choice_es"] == 51

    rows[-1]["es"] = 51
    write_rows(path, rows)

    with pytest.raises(ValueError, match="inconsistent"):
        assemble([path], repositories)


def test_ties_prefer_smallest_budget_then_actual_cost(tmp_path):
    path, repositories, rows = make_runs(tmp_path)
    for row in rows:
        if row["example_id"] == "sample:0" and row["method"] != "none":
            row["es"] = 60
            row["context_tokens"] = 50 if row["method"] == "signatures" else 100
    write_rows(path, rows)

    dataset = assemble([path], repositories)

    assert dataset["rows"][0]["label"] == "signatures:256"


@pytest.mark.parametrize("method", ["oracle", "oracle_long", "oracle_short"])
def test_oracle_is_not_a_selectable_arm(tmp_path, method):
    path, repositories, rows = make_runs(tmp_path)
    rows.append(dict(rows[0], method=method, es=100))
    write_rows(path, rows)

    dataset = assemble([path], repositories)

    assert len(dataset["arms"]) == 16
    assert dataset["preview"]["oracle_choice_es"] == 50


def test_requires_at_least_five_repositories(tmp_path):
    path, repositories, _ = make_runs(tmp_path)
    repositories = dict.fromkeys(repositories, "same/repository")

    with pytest.raises(ValueError, match="at least five repositories"):
        assemble([path], repositories)


def test_unknown_method_is_rejected(tmp_path):
    path, repositories, rows = make_runs(tmp_path)
    rows.append(dict(rows[0], method="unknown_method"))
    write_rows(path, rows)

    with pytest.raises(ValueError, match="unsupported arm"):
        assemble([path], repositories)
