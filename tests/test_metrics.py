"""Tests for RepoBench exact match and edit similarity."""

import pytest

from rac.eval.metrics import edit_similarity, exact_match


def test_exact_match_identical():
    assert exact_match("return x + 1", "return x + 1")


def test_exact_match_ignores_surrounding_whitespace():
    assert exact_match("    return x + 1\n", "return x + 1")


def test_exact_match_ignores_whitespace_runs_inside_line():
    # RepoBench compares pred.split() == gt.split()
    assert exact_match("return  x +\t1", "return x + 1")


def test_exact_match_respects_token_boundaries():
    assert not exact_match("return x+1", "return x + 1")


def test_exact_match_different_lines():
    assert not exact_match("return x", "return y")


def test_exact_match_empty_prediction():
    assert not exact_match("", "return x")
    assert exact_match("   ", "")


def test_edit_similarity_identical_is_100():
    assert edit_similarity("return x + 1", "return x + 1") == 100.0


def test_edit_similarity_disjoint_is_0():
    assert edit_similarity("abc", "xyz") == 0.0


def test_edit_similarity_known_value():
    # Indel similarity: 1 - 2 / (4 + 4) = 0.75
    assert edit_similarity("abcd", "abce") == pytest.approx(75.0)


def test_edit_similarity_is_symmetric():
    a, b = "self.fc = nn.Linear(4, 2)", "self.fc = nn.Linear(8, 2)"
    assert edit_similarity(a, b) == edit_similarity(b, a)


def test_edit_similarity_does_not_strip():
    # RepoBench scores the raw strings, so whitespace counts against ES
    assert edit_similarity("  return x", "return x") < 100.0


def test_edit_similarity_returns_float_in_range():
    score = edit_similarity("for i in range(n):", "for j in range(m):")
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0
