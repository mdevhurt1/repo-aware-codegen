"""Model fitting and fixed-width scores for the classical selectors."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GBT_SEEDS = (0, 1, 2)


def parameter_grid(model_name: str) -> list[dict]:
    """Small, predefined grids searched using validation examples only."""
    if model_name == "lr":
        return [
            {"C": strength, "class_weight": weighting}
            for strength, weighting in product(
                (0.1, 1.0, 10.0),
                (None, "balanced"),
            )
        ]

    if model_name == "gbt":
        return [
            {
                "n_estimators": trees,
                "max_depth": depth,
                "learning_rate": rate,
                "min_samples_leaf": leaf_size,
            }
            for trees, depth, rate, leaf_size in product(
                (50, 100),
                (1, 3),
                (0.05, 0.1),
                (5, 20),
            )
        ]

    raise ValueError(f"Unknown model: {model_name}")


def fit_model(
    model_name: str,
    parameters: dict,
    matrix: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
):
    """Fit a model; LR normalization sees only the supplied training rows."""
    if model_name not in {"lr", "gbt"}:
        raise ValueError(f"Unknown model: {model_name}")

    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)

    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("Training features must be a nonempty 2D matrix.")
    if labels.shape != (matrix.shape[0],):
        raise ValueError("Provide one label per training example.")
    if not np.isfinite(matrix).all():
        raise ValueError("Training features must be finite.")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Class labels must be nonnegative integers.")

    if np.unique(labels).size == 1:
        model = DummyClassifier(strategy="prior")
    elif model_name == "lr":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                **parameters,
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
            ),
        )
    else:
        model = GradientBoostingClassifier(
            **parameters,
            random_state=seed,
        )

    model.fit(matrix, labels)
    return model


def class_probabilities(
    model,
    matrix: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """Align probabilities with the complete, fixed class order.

    Classes absent from training receive zero probability. This also
    handles single-class training folds without changing output width.
    """
    if type(class_count) is not int or class_count < 2:
        raise ValueError("class_count must be an integer of at least two.")

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0]:
        raise ValueError("Prediction features must be a nonempty 2D matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Prediction features must be finite.")

    probabilities = model.predict_proba(matrix)
    classes = np.asarray(model.classes_, dtype=int)

    if np.any(classes < 0) or np.any(classes >= class_count):
        raise ValueError("Model classes fall outside the requested class order.")

    aligned = np.zeros((len(matrix), class_count), dtype=float)
    aligned[:, classes] = probabilities
    return aligned


def log_scores(probabilities: np.ndarray) -> np.ndarray:
    """Return finite log scores suitable for JSON and token-penalty sweeps.

    Zero probabilities use the smallest positive normal float before
    taking logs. The original probability matrix is not modified.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.ndim != 2
        or not probabilities.size
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Provide finite probability rows summing to one.")

    return np.log(np.clip(probabilities, np.finfo(float).tiny, 1.0))