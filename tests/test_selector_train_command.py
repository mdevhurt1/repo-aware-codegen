"""Check training-command alignment, output files, and failure handling."""

import json

import numpy as np
import pytest

from rac.selector import train_command
from rac.selector.cache import write_family


def prepare_inputs(tmp_path):
    dataset = {
        "rows": [
            {"example_id": "example:b"},
            {"example_id": "example:a"},
        ]
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    feature_path = tmp_path / "scalars.json"
    write_family(
        feature_path,
        ["value"],
        {"example:a": [1.0], "example:b": [2.0]},
    )
    return dataset_path, feature_path


def test_aligns_features_and_saves_results(tmp_path, monkeypatch):
    dataset_path, feature_path = prepare_inputs(tmp_path)
    grids = {"lr": [{"C": 1.0}], "gbt": [{"n_estimators": 3}]}

    def fake_train(dataset, matrix, columns, grids=None):
        assert [row["example_id"] for row in dataset["rows"]] == [
            "example:b",
            "example:a",
        ]
        np.testing.assert_array_equal(matrix, [[2.0], [1.0]])
        assert columns == ["scalars:value"]
        assert grids == {"lr": [{"C": 1.0}], "gbt": [{"n_estimators": 3}]}
        return {
            "version": 1,
            "arms": ["none", "context"],
            "feature_columns": columns,
            "fits": [
                {
                    "model": "lr",
                    "parameters": {"C": 1.0},
                    "validation_log_loss": 0.25,
                }
            ],
            "predictions": [
                {"example_id": "example:b", "arm_log_scores": [-0.2, -1.7]},
                {"example_id": "example:a", "arm_log_scores": [-1.7, -0.2]},
            ],
        }

    monkeypatch.setattr(train_command, "train", fake_train)
    output = tmp_path / "results"

    result = train_command.run_training(
        dataset_path,
        [feature_path],
        output,
        grids=grids,
    )

    saved = json.loads((output / "training.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    parameters = json.loads(
        (output / "selected_parameters.json").read_text(encoding="utf-8")
    )

    assert saved == result
    assert predictions == result["predictions"]
    assert parameters == result["fits"]
    assert result["sources"] == {
        "dataset": str(dataset_path),
        "features": [str(feature_path)],
    }
    assert set(result["package_versions"]) == {"numpy", "scikit-learn"}
    assert all(result["package_versions"].values())


def test_missing_features_stop_before_training(tmp_path, monkeypatch):
    dataset_path, feature_path = prepare_inputs(tmp_path)
    write_family(feature_path, ["value"], {"example:a": [1.0]})

    def unexpected_train(*args, **kwargs):
        pytest.fail("Training should not run with missing feature IDs.")

    monkeypatch.setattr(train_command, "train", unexpected_train)
    output = tmp_path / "results"

    with pytest.raises(ValueError, match="missing example IDs"):
        train_command.run_training(dataset_path, [feature_path], output)

    assert not output.exists()


def test_training_failure_does_not_create_output(tmp_path, monkeypatch):
    dataset_path, feature_path = prepare_inputs(tmp_path)

    def failing_train(*args, **kwargs):
        raise ValueError("Invalid grouped folds.")

    monkeypatch.setattr(train_command, "train", failing_train)
    output = tmp_path / "results"

    with pytest.raises(ValueError, match="Invalid grouped folds"):
        train_command.run_training(dataset_path, [feature_path], output)

    assert not output.exists()


def test_nonfinite_results_are_rejected_before_writing(tmp_path, monkeypatch):
    dataset_path, feature_path = prepare_inputs(tmp_path)

    def invalid_train(*args, **kwargs):
        return {
            "fits": [],
            "predictions": [{"arm_log_scores": [float("nan")]}],
        }

    monkeypatch.setattr(train_command, "train", invalid_train)
    output = tmp_path / "results"

    with pytest.raises(ValueError):
        train_command.run_training(dataset_path, [feature_path], output)

    assert not output.exists()