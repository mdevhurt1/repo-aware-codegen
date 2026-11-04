"""Paired significance tests against the no-context run."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def _check_paired(a: list, b: list) -> None:
    if len(a) != len(b):
        raise ValueError(f"paired inputs differ in length: {len(a)} != {len(b)}")
    if not a:
        raise ValueError("paired inputs are empty")


def paired_bootstrap_es(
    es_method: list[float], es_baseline: list[float], n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean ES difference (method - baseline) on the same examples, with a 95% bootstrap CI."""
    _check_paired(es_method, es_baseline)
    rng = np.random.default_rng(seed)

    # Per-example edit similarity differences
    diffs = np.asarray(es_method, dtype=float) - np.asarray(es_baseline, dtype=float)

    # Resample examples with replacement, all bootstrap draws at once
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)

    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(ci_lower), float(ci_upper)


def mcnemar_em(em_method: list[bool], em_baseline: list[bool]) -> float:
    """Exact McNemar p-value for paired exact-match outcomes."""
    _check_paired(em_method, em_baseline)

    # Discordant pairs: only method correct (b), only baseline correct (c)
    b = sum(m and not k for m, k in zip(em_method, em_baseline, strict=True))
    c = sum(k and not m for m, k in zip(em_method, em_baseline, strict=True))

    if b + c == 0:
        return 1.0

    # Exact test: two-sided binomial on the discordant pairs under p = 0.5
    return float(binomtest(b, b + c, 0.5).pvalue)
