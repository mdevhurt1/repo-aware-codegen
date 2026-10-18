"""Paired significance tests against the no-context run."""

from __future__ import annotations


def paired_bootstrap_es(
    es_method: list[float], es_baseline: list[float], n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean ES difference (method - baseline) on the same examples, with a 95% bootstrap CI."""
    raise NotImplementedError("paired bootstrap")

def mcnemar_em(em_method: list[bool], em_baseline: list[bool]) -> float:
    """Exact McNemar p-value for paired exact-match outcomes."""
    raise NotImplementedError("McNemar")
