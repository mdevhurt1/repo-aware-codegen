"""Evaluate feature signal using repository-grouped held-out folds."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _validate(dataset: dict) -> tuple[list[dict], list[dict]]:
    """Reject incomplete folds and example or repository leakage."""
    rows = dataset["rows"]
    folds = dataset["folds"]

    if not rows or len(folds) != 5:
        raise ValueError("Provide nonempty dataset rows and five grouped folds.")

    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset example IDs must be unique.")

    repositories = {}
    for row in rows:
        example_id = row["example_id"]
        repository = row["repo_name"]
        if not isinstance(repository, str) or not repository:
            raise ValueError("Every example needs a repository name.")
        if type(row["context_helps"]) is not bool:
            raise ValueError("context_helps must be a boolean.")
        repositories[example_id] = repository

    all_ids = set(ids)
    held_out = Counter()

    for index, fold in enumerate(folds):
        train_ids = fold["train_ids"]
        test_ids = fold["test_ids"]
        train = set(train_ids)
        test = set(test_ids)

        if not train or not test:
            raise ValueError(f"Fold {index}: train and test must be nonempty.")
        if len(train) != len(train_ids) or len(test) != len(test_ids):
            raise ValueError(f"Fold {index}: duplicate example IDs.")
        if train & test:
            raise ValueError(f"Fold {index}: train/test example overlap.")
        if train | test != all_ids:
            raise ValueError(f"Fold {index}: IDs must partition the dataset.")

        train_repos = {repositories[key] for key in train}
        test_repos = {repositories[key] for key in test}
        if train_repos & test_repos:
            raise ValueError(f"Fold {index}: repository leakage.")

        held_out.update(test_ids)

    if set(held_out) != all_ids or any(count != 1 for count in held_out.values()):
        raise ValueError("Every example must appear in exactly one test fold.")

    return rows, folds


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUC is undefined when the held-out labels contain only one class."""
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def evaluate(
    dataset: dict,
    matrix: np.ndarray,
    columns: list[str],
) -> dict:
    """Evaluate individual features and joint logistic regression.

    Matrix rows must follow dataset["rows"] order.

    Individual features use their raw values as scores: larger values
    predict that context helps. Their direction is fixed in advance,
    without selecting a direction using held-out labels.

    Joint logistic regression uses a training-only StandardScaler and
    fixed C=1.0. A single-class training fold uses its constant class
    probability instead, because logistic regression cannot fit it.
    """
    rows, folds = _validate(dataset)
    matrix = np.asarray(matrix, dtype=float)

    if (
        not columns
        or not all(isinstance(name, str) and name for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("Feature names must be nonempty and unique.")
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError("Matrix dimensions must match dataset rows and columns.")
    if not np.isfinite(matrix).all():
        raise ValueError("Features must contain only finite values.")

    positions = {row["example_id"]: index for index, row in enumerate(rows)}
    labels = np.asarray([row["context_helps"] for row in rows], dtype=int)
    results = []

    for index, fold in enumerate(folds):
        train = np.asarray([positions[key] for key in fold["train_ids"]])
        test = np.asarray([positions[key] for key in fold["test_ids"]])
        train_labels = labels[train]
        test_labels = labels[test]

        single_class = np.unique(train_labels).size < 2
        if single_class:
            probabilities = np.full(len(test), float(train_labels[0]))
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=0,
                ),
            )
            model.fit(matrix[train], train_labels)
            probabilities = model.predict_proba(matrix[test])[:, 1]

        scores = {
            name: _auc(test_labels, matrix[test, column])
            for column, name in enumerate(columns)
        }
        scores["all_features_lr"] = _auc(test_labels, probabilities)

        results.append(
            {
                "fold": index,
                "train_examples": len(train),
                "test_examples": len(test),
                "train_positive_count": int(train_labels.sum()),
                "test_positive_count": int(test_labels.sum()),
                "single_class_training": bool(single_class),
                "auc": scores,
            }
        )

    summary = {}
    for name in [*columns, "all_features_lr"]:
        values = [
            fold["auc"][name]
            for fold in results
            if fold["auc"][name] is not None
        ]
        summary[name] = {
            "mean_auc": float(np.mean(values)) if values else None,
            "std_auc": float(np.std(values)) if values else None,
            "valid_folds": len(values),
        }

    return {
        "target": "context_helps",
        "feature_columns": columns,
        "single_feature_direction": "higher_predicts_context_helps",
        "normalization": "training_fold_only",
        "summary": summary,
        "folds": results,
    }