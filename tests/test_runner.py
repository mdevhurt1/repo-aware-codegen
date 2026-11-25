"""
Tests for the experiment runner: one config in, one run directory out.

Only the parts that need a GPU or the network are stubbed (load_model, generate,
load_examples). build_context, build_prompt, the metrics, the sample loader and the file
output are all real, so these tests also cover how the runner wires them together.

write_config is the one place that knows the config layout, so changing the keys the
runner reads means changing that helper, not every test.
"""

import inspect
import json
import shutil
from pathlib import Path

import pytest
import yaml

from rac import cli, context
from rac.context import BudgetExceeded
from rac.data import repobench
from rac.eval import runner
from rac.models import load
from rac.prompt import build_prompt
from rac.types import ResultRow

REPO_ROOT = Path(__file__).parent.parent
SHIPPED_CONFIG = REPO_ROOT / "configs" / "m1_n512.yaml"
UNSET = object()


class WordTokenizer:
    """One token per whitespace-separated word, plus BOS when special tokens are on."""

    def __init__(self, model_max_length=10_000):
        self.model_max_length = model_max_length

    def __call__(self, text, add_special_tokens=True):
        ids = list(range(len(text.split())))
        return {"input_ids": [-1, *ids] if add_special_tokens else ids}


class Harness:
    """Stands in for the model and dataset, and records how the runner called them."""

    def __init__(self, examples):
        self.examples = examples
        self.model = object()
        self.tokenizer = WordTokenizer()
        self.reply = lambda prompt: ("pass", 0.25)  # prompt -> (prediction, seconds)
        self.load_model_calls = []
        self.load_examples_calls = []
        self.generate_calls = []

    def load_model(self, name="default-model", quant=UNSET):
        self.load_model_calls.append((name, quant))
        return self.model, self.tokenizer

    def load_examples(self, split="default-split", revision=UNSET):
        self.load_examples_calls.append((split, revision))
        return self.examples

    def generate(self, model, tokenizer, prompt, max_new_tokens=UNSET):
        self.generate_calls.append((model, tokenizer, prompt, max_new_tokens))
        return self.reply(prompt)

    @property
    def prompts(self):
        return [call[2] for call in self.generate_calls]


@pytest.fixture
def harness(monkeypatch, tmp_path, examples):
    """Stub the heavy calls and run from tmp_path, because the runner writes under the cwd."""
    monkeypatch.chdir(tmp_path)
    h = Harness(examples)
    monkeypatch.setattr(runner, "load_model", h.load_model)
    monkeypatch.setattr(runner, "load_examples", h.load_examples)
    monkeypatch.setattr(runner, "generate", h.generate)
    return h


@pytest.fixture
def stub_methods(monkeypatch):
    """Context methods with a known size: `three` is 3 tokens, `full` uses the whole budget."""
    monkeypatch.setitem(context.METHODS, "three", lambda ex, n, tok: "alpha beta gamma")
    monkeypatch.setitem(context.METHODS, "full", lambda ex, n, tok: " ".join(["word"] * n))


def write_config(
    sample=("fixture:0", "fixture:1", "fixture:2"),
    name="run",
    run_name=None,
    output_dir="outputs",
    methods=("none",),
    budgets=(8,),
    max_new_tokens=16,
):
    """Write <name>.yaml and its sample file to the cwd; return the config path."""
    Path(f"{name}.sample.json").write_text(json.dumps(list(sample)))
    config = {
        "run_name": run_name or name,
        "model": {"name": "stub-model", "quant": "4bit"},
        "data": {
            "split": "stub-split",
            "revision": "stub-revision",
            "sample": f"{name}.sample.json",
        },
        "methods": list(methods),
        "budgets": list(budgets),
        "generation": {"max_new_tokens": max_new_tokens},
        "output_dir": output_dir,
    }
    path = Path(f"{name}.yaml")
    path.write_text(yaml.safe_dump(config))
    return path


def read_rows(run_dir):
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def read_summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text())


# Run directory and files


@pytest.mark.parametrize("as_type", [str, Path])
def test_returns_run_directory_and_creates_it(harness, as_type):
    run_dir = runner.run(as_type(write_config()))

    assert run_dir == Path("outputs") / "run"
    assert run_dir.is_dir()
    assert (run_dir / "predictions.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()


def test_config_is_copied_byte_for_byte(harness):
    path = write_config()
    original = b"# why this run exists\n" + path.read_bytes()
    path.write_bytes(original)

    run_dir = runner.run(path)

    assert (run_dir / "config.yaml").read_bytes() == original


def test_missing_config_raises_before_loading_the_model(harness):
    with pytest.raises(FileNotFoundError):
        runner.run("no_such_config.yaml")

    assert harness.load_model_calls == []


def test_rerun_replaces_predictions_instead_of_appending(harness):
    path = write_config()

    runner.run(path)
    run_dir = runner.run(path)

    assert len(read_rows(run_dir)) == 3


def test_run_directory_uses_run_name_and_output_dir(harness):
    path = write_config(name="file_stem", run_name="experiment", output_dir="results")

    run_dir = runner.run(path)

    assert run_dir == Path("results") / "experiment"


# The loop: every sampled example x method x budget


def test_one_row_per_example_method_budget_in_order(harness, stub_methods):
    path = write_config(methods=["none", "three"], budgets=[4, 8])

    rows = read_rows(runner.run(path))

    assert [(r["example_id"], r["method"], r["budget"]) for r in rows] == [
        (f"fixture:{i}", method, budget)
        for i in range(3)
        for method in ["none", "three"]
        for budget in [4, 8]
    ]


def test_only_sampled_examples_run(harness):
    rows = read_rows(runner.run(write_config(sample=["fixture:2"])))

    assert [r["example_id"] for r in rows] == ["fixture:2"]
    assert len(harness.generate_calls) == 1


def test_rows_follow_dataset_order_not_sample_order(harness):
    rows = read_rows(runner.run(write_config(sample=["fixture:2", "fixture:0"])))

    assert [r["example_id"] for r in rows] == ["fixture:0", "fixture:2"]


def test_duplicate_sample_id_runs_once(harness):
    rows = read_rows(runner.run(write_config(sample=["fixture:1", "fixture:1"])))

    assert [r["example_id"] for r in rows] == ["fixture:1"]


# Wiring between the steps


def test_model_and_dataset_load_once_from_config(harness, stub_methods):
    runner.run(write_config(methods=["none", "three"], budgets=[4, 8]))

    assert [call[0] for call in harness.load_model_calls] == ["stub-model"]
    assert harness.load_examples_calls == [("stub-split", "stub-revision")]


def test_each_row_is_on_disk_before_the_next_generate(harness):
    """A killed runtime never gets to close the file, so rows must be flushed as they finish."""
    predictions = Path("outputs") / "run" / "predictions.jsonl"
    lines_seen = []

    def reply(prompt):
        lines_seen.append(len(predictions.read_text().splitlines()))
        return "pass", 0.25

    harness.reply = reply

    runner.run(write_config())

    assert lines_seen == [0, 1, 2]


def test_stale_summary_is_gone_after_a_failed_rerun(harness):
    run_dir = runner.run(write_config())

    def crash(prompt):
        raise RuntimeError("CUDA out of memory")

    harness.reply = crash

    with pytest.raises(RuntimeError):
        runner.run(write_config())

    assert not (run_dir / "summary.json").exists()


def test_generate_gets_the_loaded_model_and_max_new_tokens(harness):
    runner.run(write_config(sample=["fixture:0"], max_new_tokens=33))

    [(model, tokenizer, _, max_new_tokens)] = harness.generate_calls
    assert model is harness.model
    assert tokenizer is harness.tokenizer
    assert max_new_tokens == 33


def test_context_builder_gets_method_budget_and_tokenizer(harness, monkeypatch):
    seen = []

    def spy(example, n, tokenizer):
        seen.append((example.example_id, n, tokenizer))
        return ""

    monkeypatch.setitem(context.METHODS, "spy", spy)

    runner.run(write_config(sample=["fixture:1"], methods=["spy"], budgets=[4, 8]))

    assert seen == [
        ("fixture:1", 4, harness.tokenizer),
        ("fixture:1", 8, harness.tokenizer),
    ]


def test_prompt_is_context_then_imports_then_code(harness, examples, stub_methods):
    runner.run(write_config(sample=["fixture:0"], methods=["three"]))

    [prompt] = harness.prompts
    assert prompt == (
        f"alpha beta gamma\n\n{examples[0].import_statement}\n\n{examples[0].cropped_code}"
    )


def test_none_method_has_no_context_in_prompt(harness, examples):
    runner.run(write_config(sample=["fixture:0"]))

    [prompt] = harness.prompts
    assert prompt == f"{examples[0].import_statement}\n\n{examples[0].cropped_code}"


def test_prompt_length_check_uses_configured_max_new_tokens(harness, examples):
    prompt = build_prompt(examples[0], "", harness.tokenizer, max_new_tokens=0)
    prompt_tokens = len(harness.tokenizer(prompt, add_special_tokens=True)["input_ids"])
    harness.tokenizer.model_max_length = prompt_tokens + 16

    runner.run(write_config(sample=["fixture:0"], max_new_tokens=16))  # exactly fits

    with pytest.raises(ValueError, match="fixture:0"):
        runner.run(write_config(sample=["fixture:0"], max_new_tokens=17))


def test_unknown_method_raises(harness):
    with pytest.raises(ValueError, match="no_such_method"):
        runner.run(write_config(methods=["no_such_method"]))


def test_method_over_budget_raises(harness, monkeypatch):
    monkeypatch.setitem(context.METHODS, "too_long", lambda ex, n, tok: "word " * (n + 1))

    with pytest.raises(BudgetExceeded):
        runner.run(write_config(methods=["too_long"]))


# Scoring and row contents


def test_row_loads_back_as_a_result_row(harness):
    [row] = read_rows(runner.run(write_config(sample=["fixture:0"])))

    assert ResultRow(**row) == ResultRow(
        example_id="fixture:0",
        method="none",
        budget=8,
        prediction="pass",
        em=False,
        es=row["es"],
        gen_seconds=0.25,
        context_tokens=0,
    )


def test_correct_prediction_scores_full_marks(harness, examples):
    harness.reply = lambda prompt: (examples[0].next_line, 0.5)

    [row] = read_rows(runner.run(write_config(sample=["fixture:0"])))

    assert row["prediction"] == examples[0].next_line
    assert row["em"] is True
    assert row["es"] == 100.0


def test_em_ignores_whitespace_but_es_does_not(harness, examples):
    harness.reply = lambda prompt: (examples[0].next_line.strip(), 0.5)

    [row] = read_rows(runner.run(write_config(sample=["fixture:0"])))

    assert row["em"] is True
    assert 0.0 < row["es"] < 100.0


def test_each_example_is_scored_against_its_own_next_line(harness, examples):
    harness.reply = lambda prompt: (examples[0].next_line, 0.5)

    rows = read_rows(runner.run(write_config()))

    assert [r["em"] for r in rows] == [True, False, False]


def test_empty_prediction_scores_zero(harness):
    harness.reply = lambda prompt: ("", 0.5)

    [row] = read_rows(runner.run(write_config(sample=["fixture:0"])))

    assert row["prediction"] == ""
    assert row["em"] is False
    assert row["es"] == 0.0


def test_gen_seconds_comes_from_generate(harness):
    harness.reply = lambda prompt: ("pass", 1.75)

    [row] = read_rows(runner.run(write_config(sample=["fixture:0"])))

    assert row["gen_seconds"] == 1.75


def test_context_tokens_counts_context_without_special_tokens(harness, stub_methods):
    """WordTokenizer adds BOS when asked, so counting special tokens would give 1, 4 and 9."""
    path = write_config(sample=["fixture:0"], methods=["none", "three", "full"], budgets=[8])

    rows = read_rows(runner.run(path))

    assert {r["method"]: r["context_tokens"] for r in rows} == {"none": 0, "three": 3, "full": 8}


def test_non_ascii_prediction_round_trips(harness):
    harness.reply = lambda prompt: ("名前 = 'é'  # λ", 0.5)

    [row] = read_rows(runner.run(write_config(sample=["fixture:0"])))

    assert row["prediction"] == "名前 = 'é'  # λ"


def test_newline_in_prediction_stays_on_one_jsonl_line(harness):
    harness.reply = lambda prompt: ("first\nsecond", 0.5)

    rows = read_rows(runner.run(write_config(sample=["fixture:0"])))

    assert [r["prediction"] for r in rows] == ["first\nsecond"]


# summary.json


def test_summary_counts_the_run(harness, stub_methods):
    path = write_config(
        sample=["fixture:0", "fixture:2"], methods=["none", "three"], budgets=[4, 8, 16]
    )

    run_dir = runner.run(path)

    summary = read_summary(run_dir)
    assert summary["num_examples"] == 2
    assert summary["num_methods"] == 2
    assert summary["num_budgets"] == 3
    assert summary["total_predictions"] == 12 == len(read_rows(run_dir))


# CLI


def test_cli_runs_the_config_and_prints_the_run_directory(harness, capsys):
    cli.main(["run", str(write_config())])

    assert capsys.readouterr().out.strip() == str(Path("outputs") / "run")


# The stubs must not drift from the functions they replace


@pytest.mark.parametrize(
    "real",
    [load.load_model, repobench.load_examples, load.generate],
    ids=lambda real: real.__name__,
)
def test_stub_signature_matches_the_real_function(real):
    real_params = list(inspect.signature(real).parameters)
    stub = getattr(Harness, real.__name__)
    stub_params = list(inspect.signature(stub).parameters)[1:]  # drop self

    assert stub_params == real_params


# The shipped config, crash safety, and sample validation


def test_runs_the_shipped_config(harness, tmp_path):
    shipped = yaml.safe_load(SHIPPED_CONFIG.read_text())
    config_path = tmp_path / SHIPPED_CONFIG.name
    shutil.copyfile(SHIPPED_CONFIG, config_path)
    sample_path = tmp_path / shipped["data"]["sample"]
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text(json.dumps(["fixture:0", "fixture:1", "fixture:2"]))

    run_dir = runner.run(config_path)

    assert harness.load_model_calls == [(shipped["model"]["name"], shipped["model"]["quant"])]
    assert harness.load_examples_calls == [(shipped["data"]["split"], shipped["data"]["revision"])]
    assert {call[3] for call in harness.generate_calls} == {
        shipped["generation"]["max_new_tokens"]
    }
    assert len(read_rows(run_dir)) == 3 * len(shipped["methods"]) * len(shipped["budgets"])


def test_rows_finished_before_a_crash_are_on_disk(harness):
    calls = []

    def reply(prompt):
        calls.append(prompt)
        if len(calls) == 3:
            raise RuntimeError("CUDA out of memory")
        return "pass", 0.25

    harness.reply = reply
    path = write_config()

    with pytest.raises(RuntimeError):
        runner.run(path)

    predictions = Path("outputs") / "run" / "predictions.jsonl"
    assert predictions.exists()
    assert len(predictions.read_text().splitlines()) == 2


def test_failed_rerun_does_not_leave_old_predictions_beside_new_config(harness, stub_methods):
    runner.run(write_config(methods=["none"]))

    def crash(prompt):
        raise RuntimeError("CUDA out of memory")

    harness.reply = crash

    with pytest.raises(RuntimeError):
        runner.run(write_config(methods=["three"]))

    run_dir = Path("outputs") / "run"
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["methods"] == ["three"]
    predictions = run_dir / "predictions.jsonl"
    rows = read_rows(run_dir) if predictions.exists() else []
    assert all(row["method"] == "three" for row in rows)


@pytest.mark.parametrize(
    "sample",
    [
        pytest.param(["fixture:0", "fixture:999"], id="some-missing"),
        pytest.param(["other_split:0", "other_split:1"], id="all-missing"),
    ],
)
def test_sample_id_missing_from_dataset_raises(harness, sample):
    with pytest.raises(ValueError):
        runner.run(write_config(sample=sample))


def test_missing_sample_id_is_named_in_the_error(harness):
    with pytest.raises(ValueError, match="fixture:999"):
        runner.run(write_config(sample=["fixture:0", "fixture:999"]))


def test_bad_sample_is_rejected_before_loading_the_model(harness):
    with pytest.raises(ValueError):
        runner.run(write_config(sample=["fixture:999"]))

    assert harness.load_model_calls == []


def test_rejected_rerun_leaves_the_previous_run_intact(harness):
    """Nothing in the run directory changes until the run can actually start."""
    run_dir = runner.run(write_config(methods=["none"]))
    config_before = (run_dir / "config.yaml").read_bytes()
    predictions_before = (run_dir / "predictions.jsonl").read_bytes()

    with pytest.raises(ValueError):
        runner.run(write_config(sample=["fixture:999"], budgets=[16]))

    assert (run_dir / "config.yaml").read_bytes() == config_before
    assert (run_dir / "predictions.jsonl").read_bytes() == predictions_before
    assert (run_dir / "summary.json").exists()


def test_rejected_first_run_creates_no_run_directory(harness):
    with pytest.raises(ValueError):
        runner.run(write_config(sample=["fixture:999"]))

    assert not (Path("outputs") / "run").exists()
