"""Train classical arm classifiers and context gates using grouped folds."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from rac.selector.dataset import ARMS
from rac.selector.models import (
    GBT_SEEDS,
    class_probabilities,
    fit_model,
    log_scores,
    parameter_grid,
)
from rac.selector.signal import _validate


def _classification_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Validation log loss in the complete class order."""
    scores = log_scores(probabilities)
    return float(-scores[np.arange(len(labels)), labels].mean())


def _best_context_arm(outcomes: np.ndarray, indices: np.ndarray) -> int:
    """Choose the best fixed context arm using training outcomes only."""
    return 1 + int(np.argmax(outcomes[indices, 1:].mean(axis=0)))


def _gate_probabilities(
    binary_probabilities: np.ndarray,
    context_arm: int,
) -> np.ndarray:
    """Map the gate to no context or its training-selected fixed arm."""
    probabilities = np.zeros((len(binary_probabilities), len(ARMS)))
    probabilities[:, 0] = binary_probabilities[:, 0]
    probabilities[:, context_arm] = binary_probabilities[:, 1]
    return probabilities


def train(
    dataset: dict,
    matrix: np.ndarray,
    columns: list[str],
    grids: dict[str, list[dict]] | None = None,
) -> dict:
    """Tune and evaluate LR and GBT arm classifiers and binary gates.

    For outer test fold i, fold (i + 1) % 5 supplies validation examples.
    The other three folds supply the tuning training examples. After
    parameter selection, refit on all four outer training folds.

    Parameters minimize validation log loss. Ties retain grid order.
    Each GBT seed is tuned separately. Test outcomes never select
    parameters, a fixed context arm, normalization, or token costs.

    Returned predictions contain arm-aligned finite log scores. Their
    token costs are means over the outer training examples only.
    """
    rows, folds = _validate(dataset)
    if dataset["arms"] != list(ARMS):
        raise ValueError("Dataset arms must match the fixed selector arm order.")

    matrix = np.asarray(matrix, dtype=float)
    if (
        not columns
        or not all(isinstance(name, str) and name for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("Feature names must be nonempty and unique.")
    if matrix.shape != (len(rows), len(columns)):
        raise ValueError("Matrix dimensions must match rows and feature columns.")
    if not np.isfinite(matrix).all():
        raise ValueError("Features must be finite.")

    outcomes = np.asarray([row["es"] for row in rows], dtype=float)
    costs = np.asarray([row["context_tokens"] for row in rows], dtype=float)
    expected_shape = (len(rows), len(ARMS))
    if outcomes.shape != expected_shape or costs.shape != expected_shape:
        raise ValueError("Every example must provide outcomes and costs for all arms.")
    if (
        not np.isfinite(outcomes).all()
        or np.any(outcomes < 0)
        or np.any(outcomes > 100)
    ):
        raise ValueError("ES values must be finite and between 0 and 100.")
    if (
        not np.isfinite(costs).all()
        or np.any(costs < 0)
        or np.any(costs != np.floor(costs))
        or np.any(costs[:, 0] != 0)
    ):
        raise ValueError("Token costs must be nonnegative integers; none costs zero.")

    limits = np.asarray([0] + [int(arm.split(":")[1]) for arm in ARMS[1:]])
    if np.any(costs > limits):
        raise ValueError("Token costs exceed their configured budgets.")

    if any(row["label"] not in ARMS for row in rows):
        raise ValueError("Unknown arm label.")
    arm_labels = np.asarray([ARMS.index(row["label"]) for row in rows])
    gate_labels = np.asarray([int(row["context_helps"]) for row in rows])
    expected_gate = outcomes[:, 1:].max(axis=1) > outcomes[:, 0]
    if not np.array_equal(gate_labels, expected_gate.astype(int)):
        raise ValueError("context_helps labels disagree with ES outcomes.")

    grids = (
        {name: parameter_grid(name) for name in ("lr", "gbt")}
        if grids is None
        else grids
    )
    if set(grids) != {"lr", "gbt"} or any(not grid for grid in grids.values()):
        raise ValueError("Provide nonempty grids for lr and gbt.")

    positions = {row["example_id"]: index for index, row in enumerate(rows)}
    predictions = []
    fits = []

    for fold_index, fold in enumerate(folds):
        validation_fold = (fold_index + 1) % len(folds)
        validation_ids = folds[validation_fold]["test_ids"]
        validation_set = set(validation_ids)
        fit_ids = [
            key for key in fold["train_ids"] if key not in validation_set
        ]
        if not fit_ids or not validation_set <= set(fold["train_ids"]):
            raise ValueError("Validation must be a nonempty subset of outer training.")

        fit_indices = np.asarray([positions[key] for key in fit_ids])
        validation = np.asarray([positions[key] for key in validation_ids])
        outer_train = np.asarray([positions[key] for key in fold["train_ids"]])
        test = np.asarray([positions[key] for key in fold["test_ids"]])

        fixed_arm = _best_context_arm(outcomes, outer_train)
        mean_costs = costs[outer_train].mean(axis=0).tolist()

        for model_name in ("lr", "gbt"):
            seeds = (0,) if model_name == "lr" else GBT_SEEDS
            for seed in seeds:
                for policy in ("arm", "gate"):
                    labels = arm_labels if policy == "arm" else gate_labels
                    class_count = len(ARMS) if policy == "arm" else 2

                    best_parameters = None
                    best_loss = float("inf")
                    for parameters in grids[model_name]:
                        model = fit_model(
                            model_name,
                            parameters,
                            matrix[fit_indices],
                            labels[fit_indices],
                            seed,
                        )
                        probabilities = class_probabilities(
                            model, matrix[validation], class_count
                        )
                        loss = _classification_loss(
                            labels[validation], probabilities
                        )
                        if loss < best_loss:
                            best_loss = loss
                            best_parameters = dict(parameters)

                    model = fit_model(
                        model_name,
                        best_parameters,
                        matrix[outer_train],
                        labels[outer_train],
                        seed,
                    )
                    probabilities = class_probabilities(
                        model, matrix[test], class_count
                    )
                    policy_predictions = probabilities.argmax(axis=1)
                    arm_probabilities = (
                        probabilities
                        if policy == "arm"
                        else _gate_probabilities(probabilities, fixed_arm)
                    )
                    scores = log_scores(arm_probabilities)
                    chosen_arms = arm_probabilities.argmax(axis=1)

                    fits.append(
                        {
                            "fold": fold_index,
                            "model": model_name,
                            "policy": policy,
                            "seed": seed,
                            "parameters": best_parameters,
                            "validation_log_loss": best_loss,
                            "validation_fold": validation_fold,
                            "tuning_train_ids": fit_ids,
                            "validation_ids": validation_ids,
                            "refit_train_ids": fold["train_ids"],
                            "test_ids": fold["test_ids"],
                            "fixed_context_arm": (
                                ARMS[fixed_arm] if policy == "gate" else None
                            ),
                            "secondary_accuracy": float(
                                accuracy_score(labels[test], policy_predictions)
                            ),
                            "secondary_macro_f1": float(
                                f1_score(
                                    labels[test],
                                    policy_predictions,
                                    labels=list(range(class_count)),
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                        }
                    )

                    for offset, row_index in enumerate(test):
                        predictions.append(
                            {
                                "example_id": rows[row_index]["example_id"],
                                "fold": fold_index,
                                "model": model_name,
                                "policy": policy,
                                "seed": seed,
                                "arm_log_scores": scores[offset].tolist(),
                                "arm_mean_train_tokens": mean_costs,
                                "chosen_arm": ARMS[int(chosen_arms[offset])],
                            }
                        )

    return {
        "version": 1,
        "arms": list(ARMS),
        "feature_columns": columns,
        "selection_metric": "validation_log_loss",
        "validation_rule": "next_grouped_fold",
        "fits": fits,
        "predictions": predictions,
    }