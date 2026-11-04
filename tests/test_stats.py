"""Tests for paired significance tests against the no-context run."""

import pytest

from rac.eval.stats import mcnemar_em, paired_bootstrap_es


def _em_pairs(b, c, both=0, neither=0):
    """Build paired EM lists with b method-only wins and c baseline-only wins."""
    method = [True] * b + [False] * c + [True] * both + [False] * neither
    baseline = [False] * b + [True] * c + [True] * both + [False] * neither
    return method, baseline


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (5, 0, 0.0625),
        (0, 5, 0.0625),
        (12, 3, 0.03515625),
        (3, 3, 1.0),
        (1, 0, 1.0),
    ],
)
def test_mcnemar_exact_p_value(b, c, expected):
    method, baseline = _em_pairs(b, c, both=4, neither=7)
    assert mcnemar_em(method, baseline) == pytest.approx(expected)


def test_mcnemar_no_discordant_pairs():
    method, baseline = _em_pairs(0, 0, both=3, neither=2)
    assert mcnemar_em(method, baseline) == 1.0


def test_mcnemar_returns_probability():
    method, baseline = _em_pairs(40, 2)
    p = mcnemar_em(method, baseline)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_bootstrap_identical_runs():
    es = [0.2, 0.5, 0.9, 1.0]
    assert paired_bootstrap_es(es, es, n_boot=200) == (0.0, 0.0, 0.0)


def test_bootstrap_constant_shift():
    baseline = [0.1, 0.4, 0.6]
    method = [x + 0.25 for x in baseline]
    mean, lo, hi = paired_bootstrap_es(method, baseline, n_boot=200)
    assert mean == pytest.approx(0.25)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(0.25)


def test_bootstrap_ci_brackets_mean_and_returns_floats():
    method = [0.9, 0.2, 0.7, 0.5, 0.8, 0.3, 0.6, 0.4]
    baseline = [0.5, 0.3, 0.4, 0.5, 0.6, 0.1, 0.6, 0.2]
    result = paired_bootstrap_es(method, baseline, n_boot=2_000)
    assert all(type(x) is float for x in result)
    mean, lo, hi = result
    assert mean == pytest.approx(0.15)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_seed_is_deterministic():
    method = [0.9, 0.2, 0.7, 0.5, 0.8]
    baseline = [0.5, 0.3, 0.4, 0.5, 0.6]
    first = paired_bootstrap_es(method, baseline, n_boot=500, seed=3)
    assert paired_bootstrap_es(method, baseline, n_boot=500, seed=3) == first
    assert paired_bootstrap_es(method, baseline, n_boot=500, seed=4) != first


@pytest.mark.parametrize("fn", [paired_bootstrap_es, mcnemar_em])
@pytest.mark.parametrize(("a", "b"), [([1], [1, 0]), ([1, 0], [1]), ([], [])])
def test_rejects_unpaired_or_empty(fn, a, b):
    with pytest.raises(ValueError):
        fn(a, b)
