"""Config-driven runs. One config = one run directory."""

from __future__ import annotations

from pathlib import Path


def run(config_path: str | Path) -> Path:
    """
    For each sampled example x method x budget: context -> prompt -> generate -> score.

    Writes outputs/<run_name>/predictions.jsonl (one ResultRow per line, keyed by
    example_id), summary.json, and a copy of the config. Returns the run directory.
    """
    raise NotImplementedError("eval runner")
