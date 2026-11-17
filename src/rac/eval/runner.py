"""Config-driven runs. One config = one run directory."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import yaml

from rac.context import build_context, count_tokens
from rac.data.repobench import load_examples, load_sample
from rac.eval.metrics import edit_similarity, exact_match
from rac.models.load import generate, load_model
from rac.prompt import build_prompt
from rac.types import ResultRow


def run(config_path: str | Path) -> Path:
    """
    For each sampled example x method x budget: context -> prompt -> generate -> score.

    Writes <output_dir>/<run_name>/predictions.jsonl (one ResultRow per line, keyed by
    example_id, appended as each row finishes so a crash keeps the finished rows),
    summary.json (written last, so its presence means the run finished), and a copy of the
    config. A run that cannot start leaves an earlier run in that directory untouched; one
    that starts replaces it. Returns the run directory.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_dir = Path(config["output_dir"]) / config["run_name"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    # Load the examples and sample IDs once per run, and check them before the slow model load.
    data = config["data"]
    examples = load_examples(split=data["split"], revision=data["revision"])
    sample_ids_set = set(load_sample(data["sample"]))
    missing = sample_ids_set - {example.example_id for example in examples}
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(sample_ids_set)} sample IDs are not in split "
            f"{data['split']!r} at revision {data['revision']!r}, e.g. {sorted(missing)[:5]}"
        )

    # Load the model and tokenizer once per run.
    model, tokenizer = load_model(config["model"]["name"], config["model"]["quant"])

    # The run can start: clear any earlier results so they never sit beside this config.
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.unlink(missing_ok=True)

    # Save a copy of the config for reproducibility.
    shutil.copyfile(config_path, run_dir / "config.yaml")

    total_predictions = 0
    with open(run_dir / "predictions.jsonl", "w") as predictions:
        for example in examples:
            if example.example_id not in sample_ids_set:
                continue

            for method in config["methods"]:
                for budget in config["budgets"]:
                    context_text = build_context(example, method, budget, tokenizer)
                    prompt = build_prompt(
                        example, context_text, tokenizer, max_new_tokens=max_new_tokens
                    )
                    prediction, gen_seconds = generate(
                        model, tokenizer, prompt, max_new_tokens=max_new_tokens
                    )
                    result_row = ResultRow(
                        example_id=example.example_id,
                        method=method,
                        budget=budget,
                        prediction=prediction,
                        em=exact_match(prediction, example.next_line),
                        es=edit_similarity(prediction, example.next_line),
                        gen_seconds=gen_seconds,
                        context_tokens=count_tokens(context_text, tokenizer),
                    )
                    predictions.write(json.dumps(asdict(result_row)) + "\n")
                    predictions.flush()
                    total_predictions += 1

    summary_data = {
        "num_examples": len(sample_ids_set),
        "num_methods": len(config["methods"]),
        "num_budgets": len(config["budgets"]),
        "total_predictions": total_predictions,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f)

    return run_dir
