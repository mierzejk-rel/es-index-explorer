"""Unit tests for auditable Stage A analysis calculations."""

import pandas as pd
import pytest

from es_index_explorer.mlflow_analysis.stage_a import (
    _comparison_type,
    _orient_comparison,
    _pairwise_comparisons,
    _pareto_sets,
    _to_boolean,
    _use_case_from_rubric,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        (None, None),
    ],
)
def test_to_boolean_supports_exported_representations(
    value: object, expected: bool | None
) -> None:
    assert _to_boolean(value) is expected


def test_use_case_is_derived_from_versioned_rubric_key() -> None:
    assert (
        _use_case_from_rubric("invoke_entity_and_relationship_analysis_21_v2")
        == "entity_and_relationship_analysis"
    )


def test_call_comparison_is_oriented_from_fewer_to_more_calls() -> None:
    c3 = {
        "retrieval_family": "bm25",
        "calls": 3,
        "context": 20,
    }
    c1 = {
        "retrieval_family": "bm25",
        "calls": 1,
        "context": 20,
    }
    comparison_type = _comparison_type(c3, c1)
    left, right = _orient_comparison(c3, c1, comparison_type)
    assert comparison_type == "call_count_at_matched_mode_context"
    assert left["calls"] == 1
    assert right["calls"] == 3


def test_retrieval_comparison_is_oriented_from_bm25_to_dense() -> None:
    dense = {
        "retrieval_family": "dense",
        "calls": 2,
        "context": 15,
    }
    bm25 = {
        "retrieval_family": "bm25",
        "calls": 2,
        "context": 15,
    }
    comparison_type = _comparison_type(dense, bm25)
    left, right = _orient_comparison(dense, bm25, comparison_type)
    assert comparison_type == "bm25_vs_dense_at_matched_calls_context"
    assert left["retrieval_family"] == "bm25"
    assert right["retrieval_family"] == "dense"


def test_pareto_set_marks_only_nondominated_rows() -> None:
    rows = pd.DataFrame(
        [
            _run_row("fast-low", quality=0.4, latency=5.0),
            _run_row("balanced", quality=0.7, latency=7.0),
            _run_row("slow-low", quality=0.5, latency=9.0),
        ]
    )
    result = _pareto_sets(pd, rows).set_index("arm_id")
    assert bool(result.loc["fast-low", "pareto_pass_p50"])
    assert bool(result.loc["balanced", "pareto_pass_p50"])
    assert not bool(result.loc["slow-low", "pareto_pass_p50"])


def test_pairwise_comparisons_emit_each_unordered_pair_once() -> None:
    """Orienting a pair does not mutate the outer comparison-loop baseline."""
    runs = pd.DataFrame(
        [
            _dimension_row("c2", calls=2),
            _dimension_row("c1", calls=1),
            _dimension_row("c3", calls=3),
        ]
    )
    traces = pd.DataFrame(
        [
            _trace_row("c1", grade=1),
            _trace_row("c2", grade=2),
            _trace_row("c3", grade=3),
        ]
    )
    result, details = _pairwise_comparisons(pd, traces, runs)
    assert len(result) == 3
    assert len(details) == 3
    assert not result.duplicated(
        ["dataset_segment", "comparison_type", "left_arm", "right_arm"]
    ).any()


def _run_row(arm_id: str, *, quality: float, latency: float) -> dict[str, object]:
    """Return the minimum run-level row required by the Pareto calculation."""
    return {
        "dataset_segment": "dataset",
        "arm_id": arm_id,
        "good_acceptable_rate": quality,
        "good_rate": quality,
        "rubric_v2_mean": quality,
        "latency_p50_s": latency,
        "latency_p95_s": latency,
        "critical_grade_rate": 0.0,
        "critical_error_mode_rate": 0.0,
        "source_fidelity_rate": 1.0,
        "citation_completeness_rate": 1.0,
        "avg_total_tokens": 1_000.0,
    }


def _dimension_row(arm_id: str, *, calls: int) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "dataset_segment": "dataset",
        "retrieval_family": "bm25",
        "calls": calls,
        "fetch": 10,
        "context": 10,
    }


def _trace_row(arm_id: str, *, grade: int) -> dict[str, object]:
    return {
        "dataset_segment": "dataset",
        "arm_id": arm_id,
        "rubric_key": "invoke_test_1",
        "grade_score": grade,
        "is_pass": grade >= 3,
        "RubricV2": grade / 4,
        "end_to_end_latency_s": float(grade),
    }
