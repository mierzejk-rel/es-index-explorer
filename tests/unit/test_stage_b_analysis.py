"""Unit tests for auditable Stage B analysis calculations."""

import json

import pandas as pd
import pytest

from es_index_explorer.mlflow_analysis.stage_b import (
    _EXPECTED_STAGE_B_ARMS,
    _MODE_TOOL_NAMES,
    _augment_runs,
    _comparison_type,
    _orient_comparison,
    _pairwise_comparisons,
    _pareto_sets,
    _retrieval_overlap,
    _stage_b_contract_validation,
)

pytestmark = pytest.mark.unit


def test_augment_runs_parses_tool_multiset_and_composition() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "experiment_name": (
                    "/Shared/DSAS-2836/SimpleMode/"
                    "S-B-bm25-dense-c2-union-f15-rnone-mallinckrodt"
                ),
            },
            {
                "run_id": "r2",
                "experiment_name": (
                    "/Shared/DSAS-2836/SimpleMode/"
                    "S-B-bm25-bm25-c2-union-f20-rnone-mallinckrodt"
                ),
            },
        ]
    )
    result = _augment_runs(runs).set_index("run_id")
    assert result.loc["r1", "arm_id"] == "S-B-bm25-dense-c2-union-f15-rnone"
    assert result.loc["r1", "tool_multiset"] == "bm25×1 + dense×1"
    assert bool(result.loc["r1", "homogeneous"]) is False
    assert result.loc["r2", "tool_multiset"] == "bm25×2"
    assert bool(result.loc["r2", "homogeneous"]) is True


def test_augment_runs_rejects_unparseable_experiment_names() -> None:
    runs = pd.DataFrame(
        [{"run_id": "r1", "experiment_name": "S-B-unknown-c2-union-f15-rnone-mallinckrodt"}]
    )
    with pytest.raises(ValueError, match="Cannot parse Stage B experiment identity"):
        _augment_runs(runs)


def test_fetch_comparison_is_oriented_from_lower_to_higher_fetch() -> None:
    high_fetch = {
        "tool_multiset": "bm25×1 + dense×1",
        "fetch": 20,
        "calls": 2,
        "homogeneous": False,
    }
    low_fetch = {
        "tool_multiset": "bm25×1 + dense×1",
        "fetch": 15,
        "calls": 2,
        "homogeneous": False,
    }
    comparison_type = _comparison_type(high_fetch, low_fetch)
    left, right = _orient_comparison(high_fetch, low_fetch, comparison_type)
    assert comparison_type == "fetch_at_matched_tool_multiset"
    assert left["fetch"] == 15
    assert right["fetch"] == 20


def test_composition_comparison_is_oriented_from_homogeneous_to_heterogeneous() -> None:
    heterogeneous = {
        "tool_multiset": "bm25×1 + dense×1",
        "fetch": 20,
        "calls": 2,
        "homogeneous": False,
    }
    homogeneous = {
        "tool_multiset": "bm25×2",
        "fetch": 20,
        "calls": 2,
        "homogeneous": True,
    }
    comparison_type = _comparison_type(heterogeneous, homogeneous)
    left, right = _orient_comparison(heterogeneous, homogeneous, comparison_type)
    assert comparison_type == "tool_composition_at_matched_calls_fetch"
    assert bool(left["homogeneous"]) is True
    assert bool(right["homogeneous"]) is False


def test_call_count_only_differences_are_not_controlled_comparisons() -> None:
    """B3/B4 (c3) versus B5 (c4) mixes call count with composition; excluded."""
    c3 = {
        "tool_multiset": "bm25×2 + dense×1",
        "fetch": 20,
        "calls": 3,
        "homogeneous": False,
    }
    c4 = {
        "tool_multiset": "bm25×2 + dense×2",
        "fetch": 15,
        "calls": 4,
        "homogeneous": False,
    }
    assert _comparison_type(c3, c4) is None


def test_composition_comparison_excludes_two_heterogeneous_multisets() -> None:
    """Only homogeneous-versus-heterogeneous comparisons have that label."""
    one_dense = {
        "tool_multiset": "bm25×2 + dense×1",
        "fetch": 20,
        "calls": 3,
        "homogeneous": False,
    }
    two_dense = {
        "tool_multiset": "bm25×1 + dense×2",
        "fetch": 20,
        "calls": 3,
        "homogeneous": False,
    }
    assert _comparison_type(one_dense, two_dense) is None


def test_stage_b_contract_validation_requires_exact_matrix_and_runtime_config() -> None:
    runs = _stage_b_runs()
    retrieval = pd.DataFrame(columns=["run_id", "trace_id", "retrieval_mode"])

    validation = _stage_b_contract_validation(_augment_runs(runs), retrieval)

    assert validation["stage_b_arm_dataset_matrix_matches"]
    assert validation["runtime_configuration_matches"]
    assert validation["retrieval_mode_contract_matches"]

    homogeneous_run = runs[
        runs["experiment_name"].str.contains("S-B-bm25-bm25-c2-union-f20-rnone")
    ].iloc[0]["run_id"]
    unexpected_mode = pd.DataFrame(
        [
            {
                "run_id": homogeneous_run,
                "trace_id": "trace-1",
                "retrieval_mode": "dense",
            }
        ]
    )
    validation = _stage_b_contract_validation(_augment_runs(runs), unexpected_mode)
    assert validation["retrieval_mode_contract_violation_count"] == 1
    assert not validation["retrieval_mode_contract_matches"]

    invalid_runs = _augment_runs(runs.copy())
    invalid_runs.loc[0, "simple_per_call_fetch_count"] = 999
    invalid_runs.loc[1, "arm_id"] = "S-B-dense-dense-c2-union-f20-rnone"
    validation = _stage_b_contract_validation(invalid_runs, retrieval)

    assert not validation["stage_b_arm_dataset_matrix_matches"]
    assert validation["runtime_configuration_mismatch_count"] == 2


def test_retrieval_overlap_excludes_homogeneous_arms() -> None:
    """A malformed homogeneous trace cannot appear in mixed-arm overlap."""
    runs = _augment_runs(_stage_b_runs())
    mixed_run = runs[
        runs["arm_id"].eq("S-B-bm25-dense-c2-union-f15-rnone")
    ].iloc[0]["run_id"]
    homogeneous_run = runs[
        runs["arm_id"].eq("S-B-bm25-bm25-c2-union-f20-rnone")
    ].iloc[0]["run_id"]
    retrieval = pd.DataFrame(
        [
            *_retrieval_rows(mixed_run, "mixed-trace"),
            *_retrieval_rows(homogeneous_run, "homogeneous-trace"),
        ]
    )
    timings = pd.DataFrame(
        [
            {
                "run_id": mixed_run,
                "trace_id": "mixed-trace",
                "span_type": "UNKNOWN",
                "span_name": "invoke_test_1",
            },
            {
                "run_id": homogeneous_run,
                "trace_id": "homogeneous-trace",
                "span_type": "UNKNOWN",
                "span_name": "invoke_test_1",
            },
        ]
    )

    result = _retrieval_overlap(pd, retrieval, timings, runs)

    assert set(result["arm_id"]) == {"S-B-bm25-dense-c2-union-f15-rnone"}


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
    """Only single-dimension-changed pairs are controlled comparisons.

    ``bm25-dense-f15`` vs. ``bm25-bm25-f20`` differs in both fetch and tool
    composition simultaneously, so it must not be emitted as a comparison.
    """
    runs = pd.DataFrame(
        [
            _dimension_row("bm25-dense-f15", modes=("bm25", "dense"), fetch=15, calls=2),
            _dimension_row("bm25-dense-f20", modes=("bm25", "dense"), fetch=20, calls=2),
            _dimension_row("bm25-bm25-f20", modes=("bm25", "bm25"), fetch=20, calls=2),
        ]
    )
    traces = pd.DataFrame(
        [
            _trace_row("bm25-dense-f15", grade=1),
            _trace_row("bm25-dense-f20", grade=2),
            _trace_row("bm25-bm25-f20", grade=3),
        ]
    )
    result, details = _pairwise_comparisons(pd, traces, runs)
    assert len(result) == 2
    assert len(details) == 2
    assert not result.duplicated(
        ["dataset_segment", "comparison_type", "left_arm", "right_arm"]
    ).any()
    assert set(result["comparison_type"]) == {
        "fetch_at_matched_tool_multiset",
        "tool_composition_at_matched_calls_fetch",
    }


def _stage_b_runs() -> pd.DataFrame:
    """Return the full valid Stage B arm-by-dataset matrix."""
    rows: list[dict[str, object]] = []
    for arm_id, modes in _EXPECTED_STAGE_B_ARMS.items():
        fetch = int(arm_id.split("-f", maxsplit=1)[1].split("-", maxsplit=1)[0])
        tools = [_MODE_TOOL_NAMES[mode] for mode in modes]
        for dataset in ("emc2_set1", "emc2_set2", "mallinckrodt"):
            rows.append(
                {
                    "run_id": f"{arm_id}-{dataset}",
                    "experiment_name": f"/DSAS-2836/SimpleMode/{arm_id}-{dataset}",
                    "simple_required_tools": json.dumps(tools),
                    "configured_retrieval_call_count": len(tools),
                    "simple_merge_policy": "current_union",
                    "simple_per_call_fetch_count": fetch,
                    "simple_global_context_chunk_count": None,
                    "reasoning_effort": "none",
                    "invocation_concurrency": 1,
                }
            )
    return pd.DataFrame(rows)


def _retrieval_rows(run_id: str, trace_id: str) -> list[dict[str, object]]:
    """Return one BM25 and one dense ranked-chunk row for overlap testing."""
    return [
        {
            "run_id": run_id,
            "trace_id": trace_id,
            "retrieval_mode": "bm25",
            "ranked_chunk_ids": '[{"document_id": 1, "chunk_id": "1"}]',
        },
        {
            "run_id": run_id,
            "trace_id": trace_id,
            "retrieval_mode": "dense",
            "ranked_chunk_ids": '[{"document_id": 2, "chunk_id": "2"}]',
        },
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


def _dimension_row(
    arm_id: str, *, modes: tuple[str, ...], fetch: int, calls: int
) -> dict[str, object]:
    mode_counts = {mode: modes.count(mode) for mode in ("bm25", "dense")}
    return {
        "arm_id": arm_id,
        "dataset_segment": "dataset",
        "tool_multiset": " + ".join(
            f"{mode}×{count}" for mode, count in mode_counts.items() if count
        ),
        "calls": calls,
        "fetch": fetch,
        "homogeneous": len(set(modes)) == 1,
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
