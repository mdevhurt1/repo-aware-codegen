"""Train selectors from cached features and save held-out predictions."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from rac.selector.cache import load_matrix
from rac.selector.train import train


def run_training(
    dataset_path: str | Path,
    feature_paths: list[str | Path],
    output_dir: str | Path,
    grids: dict[str, list[dict]] | None = None,
) -> dict:
    """Align features, train selectors, and write reproducible results.

    Training predictions are out-of-fold: each example is scored by
    models fitted without its repository. Saved scores support the
    subsequent token-cost policy evaluation.
    """
    dataset_path = Path(dataset_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    example_ids = [row["example_id"] for row in dataset["rows"]]

    matrix, columns = load_matrix(feature_paths, example_ids)
    result = train(dataset, matrix, columns, grids=grids)
    result["sources"] = {
        "dataset": str(dataset_path),
        "features": [str(path) for path in feature_paths],
    }
    result["package_versions"] = {
        name: version(name)
        for name in ("numpy", "scikit-learn")
    }

    # Serialize everything before creating output files. Reject NaN/Infinity.
    training_text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    predictions_text = "".join(
        json.dumps(row, allow_nan=False) + "\n"
        for row in result["predictions"]
    )
    fits_text = json.dumps(result["fits"], indent=2, allow_nan=False) + "\n"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training.json").write_text(training_text, encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text(
        predictions_text,
        encoding="utf-8",
    )
    (output_dir / "selected_parameters.json").write_text(
        fits_text,
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/selector/training"),
    )
    args = parser.parse_args()

    result = run_training(args.dataset, args.features, args.output_dir)
    print(f"Completed {len(result['fits'])} model/policy/fold fits.")
    print(f"Saved {len(result['predictions'])} held-out prediction rows.")
    print(f"Results directory: {args.output_dir}")


if __name__ == "__main__":
    main()