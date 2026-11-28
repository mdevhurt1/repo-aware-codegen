"""Check signal report generation using synthetic outcomes and features."""

import json

import pytest

pytest.importorskip("sklearn", reason="requires the optional selector extra")

from rac.selector.cache import write_family  # noqa: E402
from rac.selector.signal_report import generate_report, latex_table  # noqa: E402


def make_dataset():
    rows = [
        {
            "example_id": f"repo{repo}:{label}",
            "repo_name": f"repo{repo}",
            "context_helps": bool(label),
        }
        for repo in range(5)
        for label in (0, 1)
    ]

    folds = [
        {
            "train_ids": [
                row["example_id"]
                for row in rows
                if row["repo_name"] != f"repo{repo}"
            ],
            "test_ids": [
                row["example_id"]
                for row in rows
                if row["repo_name"] == f"repo{repo}"
            ],
        }
        for repo in range(5)
    ]
    return {"rows": rows, "folds": folds}


def test_report_generation_aligns_ids_and_writes_outputs(tmp_path):
    dataset = make_dataset()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    feature_path = tmp_path / "scalars.json"
    records = {
        row["example_id"]: [float(row["context_helps"])]
        for row in reversed(dataset["rows"])
    }
    write_family(feature_path, ["matching_terms"], records)

    output_dir = tmp_path / "reports"
    report = generate_report(dataset_path, [feature_path], output_dir)

    saved = json.loads(
        (output_dir / "feature_signal.json").read_text(encoding="utf-8")
    )
    assert saved == report
    assert saved["summary"]["scalars:matching_terms"]["mean_auc"] == 1.0
    assert saved["summary"]["all_features_lr"]["valid_folds"] == 5
    assert saved["sources"]["features"] == [str(feature_path)]

    table = (output_dir / "feature_signal.tex").read_text(encoding="utf-8")
    assert table == latex_table(report)
    assert r"scalars:matching\_terms" in table
    assert r"all\_features\_lr" in table
    assert "1.000" in table


def test_undefined_auc_is_saved_as_json_null(tmp_path):
    dataset = make_dataset()
    for row in dataset["rows"]:
        row["context_helps"] = False

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    feature_path = tmp_path / "scalars.json"
    write_family(
        feature_path,
        ["value"],
        {row["example_id"]: [1.0] for row in dataset["rows"]},
    )

    output_dir = tmp_path / "reports"
    generate_report(dataset_path, [feature_path], output_dir)

    text = (output_dir / "feature_signal.json").read_text(encoding="utf-8")
    saved = json.loads(text)
    assert saved["summary"]["all_features_lr"]["mean_auc"] is None
    assert '"mean_auc": null' in text
    assert "NaN" not in text
    assert "-- & -- & 0" in (
        output_dir / "feature_signal.tex"
    ).read_text(encoding="utf-8")


def test_missing_features_do_not_create_report_outputs(tmp_path):
    dataset = make_dataset()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    feature_path = tmp_path / "scalars.json"
    write_family(feature_path, ["value"], {"repo0:0": [1.0]})

    output_dir = tmp_path / "reports"
    with pytest.raises(ValueError, match="missing example IDs"):
        generate_report(dataset_path, [feature_path], output_dir)

    assert not output_dir.exists()


def test_table_escapes_feature_names():
    report = {
        "summary": {
            "a_b&c%": {
                "mean_auc": 0.75,
                "std_auc": 0.1,
                "valid_folds": 5,
            }
        }
    }

    table = latex_table(report)

    assert r"a\_b\&c\%" in table
    assert "0.750 & 0.100 & 5" in table
