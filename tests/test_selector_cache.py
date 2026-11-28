"""Check feature-cache alignment and input validation."""

import json

import numpy as np
import pytest

from rac.selector.cache import load_matrix, write_family


def test_families_follow_requested_example_order(tmp_path):
    scalars = tmp_path / "scalars.json"
    confidence = tmp_path / "confidence.json"

    write_family(
        scalars,
        ["candidate_count", "prefix_chars"],
        {"example:b": [2, 80], "example:a": [1, 40]},
    )
    write_family(
        confidence,
        ["mean_log_probability"],
        {"example:a": [-0.2], "example:b": [-0.8]},
    )

    matrix, names = load_matrix(
        [scalars, confidence],
        ["example:a", "example:b"],
    )

    np.testing.assert_allclose(matrix, [[1, 40, -0.2], [2, 80, -0.8]])
    assert names == [
        "scalars:candidate_count",
        "scalars:prefix_chars",
        "confidence:mean_log_probability",
    ]


def test_missing_ids_raise_instead_of_dropping_examples(tmp_path):
    path = tmp_path / "scalars.json"
    write_family(path, ["value"], {"example:a": [1]})

    with pytest.raises(ValueError, match="missing example IDs"):
        load_matrix([path], ["example:a", "example:b"])


@pytest.mark.parametrize(
    ("columns", "records"),
    [
        ([], {"a": []}),
        (["x", "x"], {"a": [1, 2]}),
        (["x"], {}),
        (["x"], {"a": [1, 2]}),
        (["x"], {"a": [float("nan")]}),
        (["x"], {"a": [float("inf")]}),
    ],
)
def test_invalid_families_are_not_written(tmp_path, columns, records):
    path = tmp_path / "invalid.json"

    with pytest.raises(ValueError):
        write_family(path, columns, records)

    assert not path.exists()


def test_duplicate_requested_ids_are_rejected(tmp_path):
    path = tmp_path / "scalars.json"
    write_family(path, ["value"], {"a": [1]})

    with pytest.raises(ValueError, match="must be unique"):
        load_matrix([path], ["a", "a"])


def test_duplicate_family_names_are_rejected(tmp_path):
    path = tmp_path / "scalars.json"
    write_family(path, ["value"], {"a": [1]})

    with pytest.raises(ValueError, match="family names overlap"):
        load_matrix([path, path], ["a"])


def test_modified_cache_with_nonfinite_values_is_rejected(tmp_path):
    path = tmp_path / "scalars.json"
    write_family(path, ["value"], {"a": [1]})

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"]["a"] = [float("inf")]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="nonfinite"):
        load_matrix([path], ["a"])