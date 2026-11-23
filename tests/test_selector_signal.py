"""Tests for grouped feature-signal evaluation."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from rac.selector import signal


@pytest.fixture
def signal_data():
    """Five repositories, each containing both target classes."""
    rows = []
    for repository in range(5):
        for label in (False, True):
            rows.append(
                {
                    "example_id": f"repo{repository}:{int(label)}",
                    "repo_name": f"repo{repository}",
                    "context_helps": label,
                }
            )

    folds = []
    for repository in range(5):
        folds.append(
            {
                "train_ids": [
                    row["example_id"]
                    for row in rows
                    if row["repo_name"] != f"repo{repository}"
                ],
                "test_ids": [
                    row["example_id"]
                    for row in rows
                    if row["repo_name"] == f"repo{repository}"
                ],
            }
        )

    matrix = np.asarray(
        [[int(row["context_helps"])] for row in rows],
        dtype=float,
    )
    return {"rows": rows, "folds": folds}, matrix


def test_predictive_feature_and_joint_model(signal_data):
    dataset, matrix = signal_data

    result = signal.evaluate(dataset, matrix, ["matching_terms"])

    for name in ("matching_terms", "all_features_lr"):
        summary = result["summary"][name]
        assert summary["mean_auc"] == pytest.approx(1.0)
        assert summary["std_auc"] == pytest.approx(0.0)
        assert summary["valid_folds"] == 5


def test_raw_feature_direction_is_preserved(signal_data):
    dataset, matrix = signal_data

    result = signal.evaluate(dataset, -matrix, ["inverse_feature"])

    assert result["summary"]["inverse_feature"]["mean_auc"] == 0.0
    assert result["summary"]["all_features_lr"]["mean_auc"] == 1.0


def test_scaler_fits_only_training_examples(signal_data, monkeypatch):
    dataset, matrix = signal_data
    matrix = matrix.copy()
    matrix[:2] += 1000

    fitted_inputs = []
    original_fit = StandardScaler.fit

    def recording_fit(self, values, y=None, sample_weight=None):
        fitted_inputs.append(np.asarray(values).copy())
        return original_fit(self, values, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)

    signal.evaluate(dataset, matrix, ["value"])

    positions = {
        row["example_id"]: index
        for index, row in enumerate(dataset["rows"])
    }
    assert len(fitted_inputs) == 5
    for observed, fold in zip(fitted_inputs, dataset["folds"], strict=True):
        indices = [positions[key] for key in fold["train_ids"]]
        np.testing.assert_array_equal(observed, matrix[indices])


def test_repository_leakage_is_rejected(signal_data):
    dataset, matrix = signal_data
    dataset = deepcopy(dataset)
    dataset["rows"][2]["repo_name"] = dataset["rows"][0]["repo_name"]

    with pytest.raises(ValueError, match="repository leakage"):
        signal.evaluate(dataset, matrix, ["value"])


def test_train_test_overlap_is_rejected(signal_data):
    dataset, matrix = signal_data
    dataset = deepcopy(dataset)
    fold = dataset["folds"][0]
    fold["train_ids"].append(fold["test_ids"][0])

    with pytest.raises(ValueError, match="example overlap"):
        signal.evaluate(dataset, matrix, ["value"])


def test_duplicate_fold_ids_are_rejected(signal_data):
    dataset, matrix = signal_data
    dataset = deepcopy(dataset)
    fold = dataset["folds"][0]
    fold["train_ids"].append(fold["train_ids"][0])

    with pytest.raises(ValueError, match="duplicate example IDs"):
        signal.evaluate(dataset, matrix, ["value"])


def test_missing_fold_example_is_rejected(signal_data):
    dataset, matrix = signal_data
    dataset = deepcopy(dataset)
    dataset["folds"][0]["train_ids"].pop()

    with pytest.raises(ValueError, match="partition the dataset"):
        signal.evaluate(dataset, matrix, ["value"])


def test_repeated_test_fold_is_rejected(signal_data):
    dataset, matrix = signal_data
    dataset = deepcopy(dataset)
    dataset["folds"][1] = deepcopy(dataset["folds"][0])

    with pytest.raises(ValueError, match="exactly one test fold"):
        signal.evaluate(dataset, matrix, ["value"])


def test_single_class_test_auc_is_undefined(signal_data):
    dataset, matrix = signal_data
    dataset = deepcopy(dataset)

    for row in dataset["rows"]:
        row["context_helps"] = False

    result = signal.evaluate(dataset, matrix, ["value"])

    for fold in result["folds"]:
        assert fold["single_class_training"] is True
        assert fold["auc"]["value"] is None
        assert fold["auc"]["all_features_lr"] is None

    assert result["summary"]["all_features_lr"]["mean_auc"] is None
    assert result["summary"]["all_features_lr"]["valid_folds"] == 0


def test_single_class_training_uses_constant_probability(signal_data):
    dataset, matrix = signal_data
    dataset = deepcopy(dataset)

    for row in dataset["rows"][2:]:
        row["context_helps"] = False

    result = signal.evaluate(dataset, matrix, ["value"])

    first_fold = result["folds"][0]
    assert first_fold["single_class_training"] is True
    assert first_fold["auc"]["all_features_lr"] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_features_are_rejected(signal_data, value):
    dataset, matrix = signal_data
    matrix = matrix.copy()
    matrix[0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        signal.evaluate(dataset, matrix, ["value"])


def test_matrix_dimensions_are_checked(signal_data):
    dataset, matrix = signal_data

    with pytest.raises(ValueError, match="dimensions"):
        signal.evaluate(dataset, matrix[:-1], ["value"])


def test_evaluation_is_reproducible(signal_data):
    dataset, matrix = signal_data

    first = signal.evaluate(dataset, matrix, ["value"])
    second = signal.evaluate(dataset, matrix, ["value"])

    assert first == second