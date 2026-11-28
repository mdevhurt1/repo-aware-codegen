"""Check grouped selector training with small synthetic datasets."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="requires the optional selector extra")

from rac.selector.dataset import ARMS  # noqa: E402
from rac.selector.models import class_probabilities, fit_model, log_scores  # noqa: E402
from rac.selector.train import train  # noqa: E402

SMALL_GRIDS = {
    "lr": [{"C": 1.0, "class_weight": None}],
    "gbt": [
        {
            "n_estimators": 3,
            "max_depth": 1,
            "learning_rate": 0.1,
            "min_samples_leaf": 1,
        }
    ],
}


@pytest.fixture(scope="module")
def training_data():
    rows = []
    for repository in range(5):
        for index in range(4):
            helps = bool(index % 2)
            scores = [20.0] * len(ARMS)
            scores[0] = 40.0 if helps else 80.0
            scores[1] = 80.0 if helps else 20.0
            rows.append(
                {
                    "example_id": f"repo{repository}:{index}",
                    "repo_name": f"repo{repository}",
                    "es": scores,
                    "context_tokens": [0] + [10] * (len(ARMS) - 1),
                    "label": ARMS[1] if helps else "none",
                    "context_helps": helps,
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
        [[float(row["context_helps"])] for row in rows]
    )
    dataset = {"arms": list(ARMS), "rows": rows, "folds": folds}
    return dataset, matrix


@pytest.fixture(scope="module")
def trained(training_data):
    dataset, matrix = training_data
    return train(dataset, matrix, ["signal"], grids=SMALL_GRIDS)


def test_all_models_policies_and_seeds_are_present(trained):
    combinations = {
        (fit["model"], fit["policy"], fit["seed"])
        for fit in trained["fits"]
    }
    expected = {
        (model, policy, seed)
        for model, seeds in (("lr", (0,)), ("gbt", (0, 1, 2)))
        for policy in ("arm", "gate")
        for seed in seeds
    }

    assert combinations == expected
    assert len(trained["fits"]) == 5 * len(expected)


def test_each_example_has_one_prediction_per_policy(training_data, trained):
    dataset, _ = training_data
    keys = [
        (
            row["example_id"],
            row["model"],
            row["policy"],
            row["seed"],
        )
        for row in trained["predictions"]
    ]

    assert len(keys) == len(dataset["rows"]) * 8
    assert len(set(keys)) == len(keys)


def test_scores_use_complete_arm_order(trained):
    assert trained["arms"] == list(ARMS)

    for prediction in trained["predictions"]:
        scores = np.asarray(prediction["arm_log_scores"])
        costs = prediction["arm_mean_train_tokens"]

        assert scores.shape == (len(ARMS),)
        assert np.isfinite(scores).all()
        assert np.exp(scores).sum() == pytest.approx(1.0)
        assert len(costs) == len(ARMS)
        assert costs[0] == 0
        assert prediction["chosen_arm"] == ARMS[int(scores.argmax())]


def test_tuning_validation_and_test_repositories_are_separate(training_data, trained):
    dataset, _ = training_data
    repositories = {
        row["example_id"]: row["repo_name"] for row in dataset["rows"]
    }

    for fit in trained["fits"]:
        tuning = set(fit["tuning_train_ids"])
        validation = set(fit["validation_ids"])
        test = set(fit["test_ids"])
        refit = set(fit["refit_train_ids"])

        assert tuning
        assert validation
        assert tuning | validation == refit
        assert not tuning & validation
        assert not refit & test

        tuning_repos = {repositories[key] for key in tuning}
        validation_repos = {repositories[key] for key in validation}
        test_repos = {repositories[key] for key in test}

        assert not tuning_repos & validation_repos
        assert not tuning_repos & test_repos
        assert not validation_repos & test_repos


def test_gate_only_selects_none_or_its_fixed_arm(trained):
    fixed_arms = {
        (fit["fold"], fit["model"], fit["seed"]): fit["fixed_context_arm"]
        for fit in trained["fits"]
        if fit["policy"] == "gate"
    }

    for prediction in trained["predictions"]:
        if prediction["policy"] != "gate":
            continue
        key = (
            prediction["fold"],
            prediction["model"],
            prediction["seed"],
        )
        assert prediction["chosen_arm"] in {"none", fixed_arms[key]}


def test_test_outcomes_do_not_select_parameters_or_costs(training_data, trained):
    dataset, matrix = training_data
    changed = deepcopy(dataset)

    # These examples are held out by outer fold zero.
    for row in changed["rows"][:4]:
        row["es"] = [10.0] * len(ARMS)
        row["es"][2] = 99.0
        row["label"] = ARMS[2]
        row["context_helps"] = True
        row["context_tokens"] = [0] + [200] * (len(ARMS) - 1)

    updated = train(changed, matrix, ["signal"], grids=SMALL_GRIDS)

    before = [row for row in trained["predictions"] if row["fold"] == 0]
    after = [row for row in updated["predictions"] if row["fold"] == 0]
    assert before == after

    before_fits = [fit for fit in trained["fits"] if fit["fold"] == 0]
    after_fits = [fit for fit in updated["fits"] if fit["fold"] == 0]
    for original, modified in zip(before_fits, after_fits, strict=True):
        assert original["parameters"] == modified["parameters"]
        assert original["validation_log_loss"] == modified["validation_log_loss"]
        assert original["fixed_context_arm"] == modified["fixed_context_arm"]


def test_training_is_reproducible(training_data, trained):
    dataset, matrix = training_data

    repeated = train(dataset, matrix, ["signal"], grids=SMALL_GRIDS)

    assert repeated == trained


@pytest.mark.parametrize("model_name", ["lr", "gbt"])
def test_single_class_training_keeps_complete_output_width(model_name):
    matrix = np.asarray([[0.0], [1.0], [2.0]])
    labels = np.asarray([3, 3, 3])
    parameters = SMALL_GRIDS[model_name][0]

    model = fit_model(model_name, parameters, matrix, labels)
    probabilities = class_probabilities(model, matrix, len(ARMS))

    assert probabilities.shape == (3, len(ARMS))
    np.testing.assert_array_equal(probabilities[:, 3], np.ones(3))
    assert np.isfinite(log_scores(probabilities)).all()


def test_absent_training_classes_are_aligned():
    matrix = np.asarray([[0.0], [0.1], [0.9], [1.0]])
    labels = np.asarray([2, 2, 7, 7])
    model = fit_model("lr", SMALL_GRIDS["lr"][0], matrix, labels)

    probabilities = class_probabilities(model, matrix, len(ARMS))

    unused = [index for index in range(len(ARMS)) if index not in (2, 7)]
    assert np.all(probabilities[:, unused] == 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_inconsistent_gate_labels_are_rejected(training_data):
    dataset, matrix = training_data
    dataset = deepcopy(dataset)
    dataset["rows"][0]["context_helps"] = True

    with pytest.raises(ValueError, match="disagree with ES"):
        train(dataset, matrix, ["signal"], grids=SMALL_GRIDS)


def test_invalid_feature_dimensions_are_rejected(training_data):
    dataset, matrix = training_data

    with pytest.raises(ValueError, match="dimensions"):
        train(dataset, matrix[:-1], ["signal"], grids=SMALL_GRIDS)
