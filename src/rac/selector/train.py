"""
Pass 2 selector: logistic regression and gradient-boosted trees.

Label: per example, the method and smallest budget with the highest ES in pass 1; ties
go to the cheaper option. Split by repository. Compare against the best fixed method,
random choice, and the per-example oracle. Needs `pip install -e '.[selector]'`.
"""

from __future__ import annotations

from pathlib import Path


def train(predictions_dir: str | Path) -> None:
    raise NotImplementedError("selector training")
