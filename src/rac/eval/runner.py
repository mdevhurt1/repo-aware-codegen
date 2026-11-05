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

    Writes outputs/<run_name>/predictions.jsonl (one ResultRow per line, keyed by
    example_id), summary.json, and a copy of the config. Returns the run directory.
    """
    config_path = Path(config_path)
    run_dir = Path("outputs") / config_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Save a copy of the config for reproducibility.
    shutil.copyfile(config_path, run_dir / "config.yaml")

    # Load the model and tokenizer once per run.
    model, tokenizer = load_model(config["model_name"])

    # Load the examples and sample IDs once per run.
    examples = load_examples(split=config["split"], revision=config.get("revision"))
    sample_ids_set = set(load_sample(config["sample_ids"]))

    results: list[ResultRow] = []
    for example in examples:
        if example.example_id not in sample_ids_set:
            continue

        for method in config["methods"]:
            for budget in config["budgets"]:
                context_text = build_context(example, method, budget, tokenizer)
                prompt = build_prompt(example, context_text, tokenizer, max_new_tokens=config["max_new_tokens"])
                prediction, gen_seconds = generate(model, tokenizer, prompt, max_new_tokens=config["max_new_tokens"])
                em = exact_match(prediction, example.next_line)
                es = edit_similarity(prediction, example.next_line)
                context_tokens = count_tokens(context_text, tokenizer)

                result_row = ResultRow(
                    example_id=example.example_id,
                    method=method,
                    budget=budget,
                    prediction=prediction,
                    em=em,
                    es=es,
                    gen_seconds=gen_seconds,
                    context_tokens=context_tokens,
                )
                results.append(result_row)

    # Write predictions.jsonl.
    predictions_path = run_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for row in results:
            f.write(json.dumps(asdict(row)) + "\n")

    # Write summary.json.
    summary_path = run_dir / "summary.json"
    summary_data = {
        "num_examples": len(sample_ids_set),
        "num_methods": len(config["methods"]),
        "num_budgets": len(config["budgets"]),
        "total_predictions": len(results),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f)

    return run_dir
