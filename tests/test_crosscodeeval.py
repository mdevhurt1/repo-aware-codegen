"""Check Python CrossCodeEval baseline and structured retrieval records."""

import json

import pytest

from rac.data.crosscodeeval import load_examples


@pytest.fixture
def record():
    return {
        "metadata": {"task_id": "python/0", "repository": "owner/repo", "file": "main.py"},
        "prompt": "from util import café\n\ndef run():\n    ",
        "groundtruth": "return café()\n",
        "right_context": "# excluded from the prompt",
    }


def test_baseline_preserves_prompt_and_reference(tmp_path, record):
    path = tmp_path / "baseline.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8-sig")

    (example,) = load_examples(path)

    assert (example.example_id, example.repo_name, example.file_path) == (
        "python/0", "owner/repo", "main.py"
    )
    assert example.cropped_code == record["prompt"]
    assert example.next_line == record["groundtruth"]
    assert example.import_statement == ""
    assert example.context == []
    assert example.gold_snippet_index == -1


def test_augmented_records_use_structured_chunks(tmp_path, record):
    path = tmp_path / "retrieval.jsonl"
    record["crossfile_context"] = {
        "text": "# formatted context",
        "list": [
            {"filename": "util.py", "retrieved_chunk": "def café(): ..."},
            {"filename": "util.py", "retrieved_chunk": "def other(): ..."},
        ],
    }
    path.write_text(json.dumps(record), encoding="utf-8")

    (example,) = load_examples(path)

    assert [snippet.path for snippet in example.context] == ["util.py", "util.py"]
    assert [snippet.snippet for snippet in example.context] == [
        "def café(): ...", "def other(): ..."
    ]
    assert len({snippet.identifier for snippet in example.context}) == 2
    assert example.gold_snippet_index == -1


def test_invalid_json_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\nnot JSON\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2:"):
        load_examples(path)


def test_rejects_duplicate_task_ids(tmp_path, record):
    path = tmp_path / "duplicate.jsonl"
    path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate\.jsonl:2:.*duplicate task_id"):
        load_examples(path)
