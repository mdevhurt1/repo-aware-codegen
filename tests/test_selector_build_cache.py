"""Check scalar-cache construction using local examples."""

from dataclasses import replace

import numpy as np
import pytest

from rac.selector.build_cache import build_scalar_cache
from rac.selector.cache import SCALAR_COLUMNS, load_matrix
from rac.selector.features import extract


def test_builds_features_in_requested_order(tmp_path, examples):
    chosen = list(reversed(examples))
    dataset = {
        "rows": [
            {"example_id": ex.example_id, "repo_name": ex.repo_name}
            for ex in chosen
        ]
    }
    output = tmp_path / "scalars.json"

    build_scalar_cache(dataset, examples, output)

    matrix, columns = load_matrix(
        [output],
        [ex.example_id for ex in chosen],
    )
    expected = [
        [extract(ex)[name] for name in SCALAR_COLUMNS]
        for ex in chosen
    ]
    np.testing.assert_allclose(matrix, expected)
    assert columns == [f"scalars:{name}" for name in SCALAR_COLUMNS]


def test_ignores_unrequested_examples(tmp_path, examples):
    chosen = examples[0]
    dataset = {
        "rows": [
            {"example_id": chosen.example_id, "repo_name": chosen.repo_name}
        ]
    }
    output = tmp_path / "scalars.json"

    build_scalar_cache(dataset, examples, output)

    matrix, _ = load_matrix([output], [chosen.example_id])
    assert matrix.shape == (1, len(SCALAR_COLUMNS))


def test_missing_source_example_is_rejected(tmp_path, examples):
    dataset = {
        "rows": [
            {"example_id": "missing:0", "repo_name": "missing/repo"}
        ]
    }
    output = tmp_path / "scalars.json"

    with pytest.raises(ValueError, match="Missing source examples"):
        build_scalar_cache(dataset, examples, output)

    assert not output.exists()


def test_repository_mismatch_is_rejected(tmp_path, examples):
    chosen = examples[0]
    dataset = {
        "rows": [
            {"example_id": chosen.example_id, "repo_name": "wrong/repository"}
        ]
    }
    output = tmp_path / "scalars.json"

    with pytest.raises(ValueError, match="Repository mismatch"):
        build_scalar_cache(dataset, examples, output)

    assert not output.exists()


def test_duplicate_requested_id_is_rejected(tmp_path, examples):
    chosen = examples[0]
    row = {"example_id": chosen.example_id, "repo_name": chosen.repo_name}
    output = tmp_path / "scalars.json"

    with pytest.raises(ValueError, match="must be unique"):
        build_scalar_cache({"rows": [row, row]}, examples, output)

    assert not output.exists()


def test_duplicate_source_id_is_rejected(tmp_path, examples):
    chosen = examples[0]
    dataset = {
        "rows": [
            {"example_id": chosen.example_id, "repo_name": chosen.repo_name}
        ]
    }
    output = tmp_path / "scalars.json"

    with pytest.raises(ValueError, match="Duplicate source example"):
        build_scalar_cache(dataset, [chosen, chosen], output)

    assert not output.exists()


def test_target_fields_do_not_change_cached_features(tmp_path, examples):
    original = examples[0]
    modified = replace(
        original,
        next_line="THIS_TARGET_MUST_NOT_BECOME_A_FEATURE",
        gold_snippet_index=-1,
    )
    dataset = {
        "rows": [
            {"example_id": original.example_id, "repo_name": original.repo_name}
        ]
    }
    first = tmp_path / "original.json"
    second = tmp_path / "modified.json"

    build_scalar_cache(dataset, [original], first)
    build_scalar_cache(dataset, [modified], second)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_empty_dataset_is_rejected(tmp_path):
    output = tmp_path / "scalars.json"

    with pytest.raises(ValueError, match="must not be empty"):
        build_scalar_cache({"rows": []}, [], output)

    assert not output.exists()