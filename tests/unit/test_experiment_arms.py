"""Unit tests for the stage-neutral arm-analysis core."""

import pandas as pd
import pytest

from es_index_explorer.mlflow_analysis import experiment_arms as core

pytestmark = pytest.mark.unit


def test_split_arm_and_dataset_strips_known_suffix() -> None:
    arm_id, dataset_segment = core.split_arm_and_dataset(
        "S-A-bm25-c1-rr-f10-g10-rnone-mallinckrodt",
        ("emc2_set1", "emc2_set2", "mallinckrodt"),
    )
    assert arm_id == "S-A-bm25-c1-rr-f10-g10-rnone"
    assert dataset_segment == "mallinckrodt"


def test_split_arm_and_dataset_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError, match="Cannot parse dataset segment"):
        core.split_arm_and_dataset("S-A-bm25-c1-unknown", ("mallinckrodt",))


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
    assert core.to_boolean(value) is expected


def test_use_case_is_derived_from_versioned_rubric_key() -> None:
    assert (
        core.use_case_from_rubric("invoke_entity_and_relationship_analysis_21_v2")
        == "entity_and_relationship_analysis"
    )


def test_pareto_set_marks_only_nondominated_rows() -> None:
    rows = pd.DataFrame(
        [
            _run_row("fast-low", quality=0.4, latency=5.0),
            _run_row("balanced", quality=0.7, latency=7.0),
            _run_row("slow-low", quality=0.5, latency=9.0),
        ]
    )
    result = core.pareto_sets(pd, rows).set_index("arm_id")
    assert bool(result.loc["fast-low", "pareto_pass_p50"])
    assert bool(result.loc["balanced", "pareto_pass_p50"])
    assert not bool(result.loc["slow-low", "pareto_pass_p50"])


def test_pairwise_comparisons_engine_is_agnostic_to_stage_specific_rules() -> None:
    """A trivial always-comparable classifier still yields one row per pair."""
    runs = pd.DataFrame(
        [
            {"arm_id": "x", "dataset_segment": "dataset", "dim": 1},
            {"arm_id": "y", "dataset_segment": "dataset", "dim": 2},
        ]
    )
    traces = pd.DataFrame(
        [
            _trace_row("x", grade=1),
            _trace_row("y", grade=3),
        ]
    )

    def comparison_type_fn(
        left: dict[str, object], right: dict[str, object]
    ) -> str | None:
        return "any_pair" if left["dim"] != right["dim"] else None

    def orient_fn(
        left: dict[str, object], right: dict[str, object], comparison_type: str
    ) -> tuple[dict[str, object], dict[str, object]]:
        return (left, right) if left["dim"] < right["dim"] else (right, left)

    summary, details = core.pairwise_comparisons(
        pd,
        traces,
        runs,
        dimension_columns=["dim"],
        comparison_type_fn=comparison_type_fn,
        orient_fn=orient_fn,
    )
    assert len(summary) == 1
    assert len(details) == 1
    assert summary.iloc[0]["left_arm"] == "x"
    assert summary.iloc[0]["right_arm"] == "y"
    assert summary.iloc[0]["right_grade_wins"] == 1


def test_empty_pairwise_comparisons_return_typed_empty_outputs() -> None:
    """A valid stage without controlled pairs still produces audit files."""
    summary, details = core.pairwise_comparisons(
        pd,
        pd.DataFrame(
            columns=[
                "dataset_segment",
                "arm_id",
                "rubric_key",
                "grade_score",
                "is_pass",
                "RubricV2",
                "end_to_end_latency_s",
            ]
        ),
        pd.DataFrame(columns=["arm_id", "dataset_segment", "dimension"]),
        dimension_columns=["dimension"],
        comparison_type_fn=lambda left, right: None,
        orient_fn=lambda left, right, comparison_type: (left, right),
    )
    assert list(summary.columns) == [
        "dataset_segment",
        "comparison_type",
        "left_arm",
        "right_arm",
        "matched_trace_count",
        "right_grade_wins",
        "ties",
        "right_grade_losses",
        "right_minus_left_pass_rate",
        "right_minus_left_rubric_v2_mean",
        "right_minus_left_latency_median_s",
    ]
    assert list(details.columns)[:5] == [
        "dataset_segment",
        "comparison_type",
        "left_arm",
        "right_arm",
        "rubric_key",
    ]


def test_empty_overlap_and_operation_summaries_return_typed_outputs() -> None:
    """Missing matched chunks or Simple spans must not abort a stage report."""
    timings = pd.DataFrame(columns=["span_type", "span_name", "run_id", "trace_id"])
    retrieval = pd.DataFrame(
        columns=[
            "simple_operation",
            "retrieval_mode",
            "run_id",
            "trace_id",
            "ranked_chunk_ids",
        ]
    )
    runs = pd.DataFrame(
        columns=["run_id", "arm_id", "dataset_segment", "retrieval_family", "calls", "context"]
    )
    between = core.between_arm_family_overlap(
        pd,
        retrieval,
        timings,
        runs,
        family_column="retrieval_family",
        left_family="bm25",
        right_family="dense",
        match_columns=["calls", "context"],
    )
    within = core.within_trace_family_overlap(
        pd,
        retrieval,
        timings,
        runs[["run_id", "arm_id", "dataset_segment"]],
        left_family="bm25",
        right_family="dense",
    )
    operations = core.operation_timing_summary(
        pd,
        pd.DataFrame(
            columns=[
                "simple_operation",
                "run_id",
                "es_success_attempt_count",
                "llm_successful_attempt_count",
            ]
        ),
        runs[["run_id", "arm_id", "dataset_segment"]],
    )
    assert list(between.columns)[:3] == ["dataset_segment", "calls", "context"]
    assert list(within.columns)[:4] == [
        "dataset_segment",
        "arm_id",
        "trace_id",
        "rubric_key",
    ]
    assert list(operations.columns)[:3] == [
        "dataset_segment",
        "arm_id",
        "simple_operation",
    ]


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
