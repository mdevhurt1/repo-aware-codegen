"""Verify selector features against real BM25 scores and edge cases."""

from dataclasses import replace

from rac.selector.features import extract
from rac.types import Example, Snippet


def _example() -> Example:
    return Example(
        example_id="fixture:0",
        repo_name="demo/repo",
        file_path="main.py",
        context=[
            Snippet("fetch_widget", "widget.py", "widget"),
            Snippet("weather", "weather.py", "rain"),
            Snippet("distance", "distance.py", "miles"),
        ],
        import_statement="from widget import fetch_widget\n\nfrom weather import weather\n",
        cropped_code="weather\n\n\n\n\nfetchWidget()",
        next_line="not used",
        gold_snippet_index=0,
        level="2k",
    )


def test_features_measure_candidates_and_recent_query():
    features = extract(_example())

    assert features["import_count"] == 2
    assert features["candidate_count"] == 3
    assert features["candidate_total_chars"] == 15
    assert features["identifier_overlap"] == 2
    assert features["bm25_top_score"] > 0
    assert features["bm25_score_gap"] == features["bm25_top_score"]
    assert features["prefix_chars"] == len(_example().cropped_code)
    assert features["level"] == "2k"


def test_no_candidates_or_query_terms_have_zero_retrieval_features():
    example = _example()
    for row in (replace(example, context=[]), replace(example, cropped_code="!!!")):
        features = extract(row)
        assert features["identifier_overlap"] == 0
        assert features["bm25_top_score"] == 0
        assert features["bm25_score_gap"] == 0


def test_single_candidate_has_no_gap_and_labels_do_not_affect_features():
    example = _example()
    example = replace(example, context=example.context[:1])
    features = extract(example)
    assert features["candidate_count"] == 1
    assert features["bm25_score_gap"] == 0
    assert extract(replace(example, next_line="different", gold_snippet_index=-1)) == features
