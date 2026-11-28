"""Generate a feature-signal report from cached features and grouped folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rac.selector.cache import load_matrix
from rac.selector.signal import evaluate


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_table(report: dict) -> str:
    """Render macro-average held-out AUC and population standard deviation."""
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Feature / model & Mean AUC & Std. & Valid folds \\",
        r"\hline",
    ]

    for name, values in report["summary"].items():
        mean = values["mean_auc"]
        std = values["std_auc"]
        mean_text = "--" if mean is None else f"{mean:.3f}"
        std_text = "--" if std is None else f"{std:.3f}"
        label = _latex_escape(name)
        count = values["valid_folds"]
        lines.append(f"{label} & {mean_text} & {std_text} & {count} " + r"\\")

    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def generate_report(
    dataset_path: str | Path,
    feature_paths: list[str | Path],
    output_dir: str | Path,
) -> dict:
    """Align cached features with outcome rows, evaluate, and save results."""
    dataset_path = Path(dataset_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    example_ids = [row["example_id"] for row in dataset["rows"]]

    matrix, columns = load_matrix(feature_paths, example_ids)
    report = evaluate(dataset, matrix, columns)
    report["sources"] = {
        "dataset": str(dataset_path),
        "features": [str(path) for path in feature_paths],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_signal.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "feature_signal.tex").write_text(
        latex_table(report),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/selector/signal"),
    )
    args = parser.parse_args()

    report = generate_report(args.dataset, args.features, args.output_dir)
    print(f"Evaluated {len(report['feature_columns'])} features across five folds.")
    print(f"Saved feature_signal.json and feature_signal.tex to {args.output_dir}")


if __name__ == "__main__":
    main()