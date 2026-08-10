"""Generic, stage-neutral arm-level experiment analysis core.

This module contains only algorithms that do not depend on any specific
experiment stage's naming scheme or parameter set: trace/run-level
aggregation, pairwise comparisons, Pareto sets, retrieval overlap, operation
timing summaries, grade distributions, and shared I/O helpers.

Per-stage modules (for example ``stage_a.py`` and ``stage_b.py``) supply the
stage-specific pieces this core needs as plain callables/values:

- an arm-identity parser (regex over the experiment short name);
- the list of extra per-arm dimension columns to carry through analysis;
- a pairwise ``comparison_type``/``orient_comparison`` pair;
- an overlap strategy (:func:`between_arm_family_overlap` or
  :func:`within_trace_family_overlap`), when retrieval-signal overlap is
  meaningful for that stage;
- expected population counts for the validation gate;
- a stage-specific report renderer.

See ``README-mlflow-rubric-analysis.md`` for the adapter pattern and a guide
to writing a new stage adapter.
"""

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

GRADE_ORDER = {
    "Critical Error": 0,
    "Poor": 1,
    "Partial": 2,
    "Acceptable": 3,
    "Good": 4,
}
PASSING_GRADES = {"Good", "Acceptable"}
ERROR_ASSESSMENTS = [
    "errors_attribution_v2",
    "errors_citation_support_v2",
    "errors_context_stripping_v2",
    "errors_entity_resolution_v2",
    "errors_gap_acknowledgment_v2",
    "errors_opinion_v2",
    "errors_overstating_certainty_v2",
    "errors_paraphrase_drift_v2",
    "errors_timeline_v2",
    "errors_unsupported_assertion_v2",
]
POSITIVE_BOOLEAN_ASSESSMENTS = [
    "answer_completeness",
    "citation_attribution",
    "citation_completeness",
    "citation_format",
    "citation_format_valid",
    "entity_completeness",
    "markdown_validation",
    "source_fidelity",
]
CONTINUOUS_ASSESSMENTS = [
    "RubricV2",
    "citation_in_snippet_matching_summary",
    "citation_validation_summary",
    "reference_in_response_validation_summary",
]
PERCENTILE_METHOD = "linear"

_RUN_BASE_COLUMNS = [
    "run_id",
    "experiment_id",
    "experiment_name",
    "model_version",
    "arm_id",
    "dataset_segment",
    "start_time_utc",
    "start_hour_utc",
]

ComparisonTypeFn = Callable[[dict[str, object], dict[str, object]], str | None]
OrientFn = Callable[
    [dict[str, object], dict[str, object], str],
    tuple[dict[str, object], dict[str, object]],
]

_PAIRWISE_SUMMARY_COLUMNS = [
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
_PAIRWISE_DETAIL_COLUMNS = [
    "dataset_segment",
    "comparison_type",
    "left_arm",
    "right_arm",
    "rubric_key",
    "left_grade_score",
    "right_grade_score",
    "grade_score_delta",
    "left_is_pass",
    "right_is_pass",
    "left_rubric_v2",
    "right_rubric_v2",
    "rubric_v2_delta",
    "left_latency_s",
    "right_latency_s",
    "latency_delta_s",
]
_OPERATION_TIMING_COLUMNS = [
    "dataset_segment",
    "arm_id",
    "simple_operation",
    "span_count",
    "observed_duration_p50_ms",
    "observed_duration_p90_ms",
    "observed_duration_p95_ms",
    "llm_successful_attempt_latency_p50_ms",
    "es_success_duration_p50_ms",
    "llm_attempt_observation_count",
    "llm_retry_rate",
    "es_attempt_observation_count",
    "es_retry_rate",
]


def split_arm_and_dataset(
    short_name: str, dataset_suffixes: tuple[str, ...]
) -> tuple[str, str]:
    """Split an experiment short name into its arm ID and dataset segment.

    Parameters
    ----------
    short_name
        The last path segment of the MLflow experiment name.
    dataset_suffixes
        Recognized dataset-segment suffixes, checked longest-match-first
        order is not required since suffixes are expected to be disjoint.

    Returns
    -------
    tuple[str, str]
        ``(arm_id, dataset_segment)``.

    Raises
    ------
    ValueError
        If no known dataset suffix matches the experiment short name.
    """
    dataset_segment = next(
        (suffix for suffix in dataset_suffixes if short_name.endswith(f"-{suffix}")),
        None,
    )
    if dataset_segment is None:
        raise ValueError(f"Cannot parse dataset segment from {short_name!r}.")
    return short_name.removesuffix(f"-{dataset_segment}"), dataset_segment


def build_trace_level(
    *,
    pandas: Any,
    runs: Any,
    quality: Any,
    retrieval: Any,
    timings: Any,
    dimension_columns: list[str],
) -> tuple[Any, dict[str, object]]:
    """Build one auditable row per run/trace and return validation diagnostics.

    Parameters
    ----------
    dimension_columns
        Extra per-arm columns (beyond ``arm_id``/``dataset_segment``) already
        present on ``runs``, carried through to every trace row.
    """
    quality = quality.copy()
    quality["assessment_value"] = quality.apply(decode_assessment_row, axis=1)
    duplicate_mask = quality.duplicated(
        ["run_id", "trace_id", "assessment_name", "assessment_value"],
        keep="first",
    )
    identical_duplicate_count = int(duplicate_mask.sum())
    quality = quality[~duplicate_mask].copy()

    conflicts = (
        quality.groupby(["run_id", "trace_id", "assessment_name"])["assessment_value"]
        .nunique(dropna=False)
        .reset_index(name="value_count")
    )
    conflicting_count = int((conflicts["value_count"] > 1).sum())
    if conflicting_count:
        raise ValueError(
            f"Found {conflicting_count} conflicting duplicate assessment keys."
        )

    quality_pivot = quality.pivot(
        index=["run_id", "trace_id"],
        columns="assessment_name",
        values="assessment_value",
    ).reset_index()

    rubric_spans = timings[
        (timings["span_type"] == "UNKNOWN")
        & timings["span_name"].str.startswith("invoke_", na=False)
    ][["run_id", "trace_id", "span_name", "observed_duration_ms"]].copy()
    rubric_spans = rubric_spans.rename(
        columns={
            "span_name": "rubric_key",
            "observed_duration_ms": "end_to_end_latency_ms",
        }
    )
    rubric_spans["use_case"] = rubric_spans["rubric_key"].map(use_case_from_rubric)

    retrieval_summary = retrieval[retrieval["actual_retrieval_calls"].notna()][
        [
            "run_id",
            "trace_id",
            "actual_retrieval_calls",
            "generic_retrieval_calls",
            "metadata_filter_retrieval_calls",
            "selected_chunk_count",
            "context_size_chars",
        ]
    ].drop_duplicates(["run_id", "trace_id"])

    run_dimensions = runs[[*_RUN_BASE_COLUMNS, *dimension_columns]]
    trace_level = (
        rubric_spans.merge(
            run_dimensions, on="run_id", how="left", validate="many_to_one"
        )
        .merge(
            quality_pivot, on=["run_id", "trace_id"], how="left", validate="one_to_one"
        )
        .merge(
            retrieval_summary,
            on=["run_id", "trace_id"],
            how="left",
            validate="one_to_one",
        )
    )
    trace_level["end_to_end_latency_s"] = trace_level["end_to_end_latency_ms"] / 1_000
    trace_level["grade_score"] = trace_level["ordinal_grade"].map(GRADE_ORDER)
    trace_level["is_pass"] = trace_level["ordinal_grade"].isin(PASSING_GRADES)
    trace_level["is_good"] = trace_level["ordinal_grade"].eq("Good")
    trace_level["is_critical"] = trace_level["ordinal_grade"].eq("Critical Error")
    for assessment in ERROR_ASSESSMENTS:
        trace_level[assessment] = trace_level[assessment].map(to_boolean)
    for assessment in POSITIVE_BOOLEAN_ASSESSMENTS:
        trace_level[assessment] = trace_level[assessment].map(to_boolean)
    trace_level["has_any_critical_error_mode"] = (
        trace_level[ERROR_ASSESSMENTS].fillna(False).any(axis=1)
    )

    validation = {
        "raw_quality_row_count": int(len(quality) + identical_duplicate_count),
        "identical_duplicate_assessment_row_count": identical_duplicate_count,
        "conflicting_duplicate_assessment_count": conflicting_count,
        "unique_trace_count": int(
            trace_level[["run_id", "trace_id"]].drop_duplicates().shape[0]
        ),
        "trace_without_rubric_key_count": int(trace_level["rubric_key"].isna().sum()),
        "trace_without_grade_count": int(trace_level["ordinal_grade"].isna().sum()),
        "trace_without_latency_count": int(
            trace_level["end_to_end_latency_ms"].isna().sum()
        ),
        "trace_without_exported_row_id_count": int(
            quality["row_id"]
            .isna()
            .groupby([quality["run_id"], quality["trace_id"]])
            .all()
            .sum()
        ),
        "pairing_key": "dataset_segment + rubric_key",
        "duplicate_policy": (
            "Drop only exact duplicate run_id/trace_id/assessment_name/value rows; "
            "fail validation if duplicate keys disagree."
        ),
    }
    return trace_level.sort_values(
        ["dataset_segment", "arm_id", "rubric_key"]
    ), validation


def build_run_level(
    *,
    pandas: Any,
    runs: Any,
    metrics: Any,
    trace_level: Any,
    dimension_columns: list[str],
) -> Any:
    """Compute reproducible per-run quality, latency, usage, and context metrics."""
    metric_pivot = metrics.pivot(
        index="run_id", columns="metric_key", values="metric_value"
    ).reset_index()
    rows: list[dict[str, object]] = []
    for run_id, traces in trace_level.groupby("run_id", sort=False):
        run = runs[runs["run_id"] == run_id].iloc[0]
        metric_row = metric_pivot[metric_pivot["run_id"] == run_id].iloc[0]
        grades = traces["ordinal_grade"].value_counts()
        row: dict[str, object] = {
            "run_id": run_id,
            "experiment_id": run["experiment_id"],
            "experiment_name": run["experiment_name"],
            "model_version": run["model_version"],
            "arm_id": run["arm_id"],
            "dataset_segment": run["dataset_segment"],
            **{column: run[column] for column in dimension_columns},
            "start_time_utc": run["start_time_utc"],
            "start_hour_utc": int(run["start_hour_utc"]),
            "trace_count": len(traces),
            "grade_good_count": int(grades.get("Good", 0)),
            "grade_acceptable_count": int(grades.get("Acceptable", 0)),
            "grade_partial_count": int(grades.get("Partial", 0)),
            "grade_poor_count": int(grades.get("Poor", 0)),
            "grade_critical_count": int(grades.get("Critical Error", 0)),
            "good_acceptable_rate": float(traces["is_pass"].mean()),
            "good_rate": float(traces["is_good"].mean()),
            "critical_grade_rate": float(traces["is_critical"].mean()),
            "critical_error_mode_rate": float(
                traces["has_any_critical_error_mode"].mean()
            ),
            "rubric_v2_mean": float(traces["RubricV2"].mean()),
            "latency_p50_s": float(
                traces["end_to_end_latency_s"].quantile(
                    0.50, interpolation=PERCENTILE_METHOD
                )
            ),
            "latency_p90_s": float(
                traces["end_to_end_latency_s"].quantile(
                    0.90, interpolation=PERCENTILE_METHOD
                )
            ),
            "latency_p95_s": float(
                traces["end_to_end_latency_s"].quantile(
                    0.95, interpolation=PERCENTILE_METHOD
                )
            ),
            "latency_mean_s": float(traces["end_to_end_latency_s"].mean()),
            "selected_chunk_mean": float(traces["selected_chunk_count"].mean()),
            "context_chars_mean": float(traces["context_size_chars"].mean()),
            "actual_retrieval_calls_mean": float(
                traces["actual_retrieval_calls"].mean()
            ),
            "reported_total_pass_rate": metric_value(metric_row, "total_pass_rate"),
            "reported_execution_time_p50_s": metric_value(
                metric_row, "execution_time_p50_s"
            ),
            "avg_total_tokens": metric_value(metric_row, "avg_total_tokens"),
            "avg_prompt_tokens": metric_value(metric_row, "avg_prompt_tokens"),
            "avg_completion_tokens": metric_value(metric_row, "avg_completion_tokens"),
            "avg_reasoning_tokens": metric_value(metric_row, "avg_reasoning_tokens"),
            "avg_llm_calls_per_trace": metric_value(
                metric_row, "avg_llm_calls_per_trace"
            ),
        }
        for assessment in POSITIVE_BOOLEAN_ASSESSMENTS:
            row[f"{assessment}_rate"] = float(traces[assessment].mean())
        for assessment in ERROR_ASSESSMENTS:
            row[f"{assessment}_rate"] = float(traces[assessment].mean())
        for assessment in CONTINUOUS_ASSESSMENTS[1:]:
            row[f"{assessment}_mean"] = float(traces[assessment].mean())
        rows.append(row)
    return pandas.DataFrame(rows).sort_values(["dataset_segment", "arm_id"])


def quality_summary(pandas: Any, run_level: Any) -> Any:
    """Return the auditable quality columns used for recommendations."""
    columns = [
        "dataset_segment",
        "arm_id",
        "trace_count",
        "grade_good_count",
        "grade_acceptable_count",
        "grade_partial_count",
        "grade_poor_count",
        "grade_critical_count",
        "good_acceptable_rate",
        "good_rate",
        "critical_grade_rate",
        "critical_error_mode_rate",
        "rubric_v2_mean",
        "citation_attribution_rate",
        "citation_completeness_rate",
        "source_fidelity_rate",
        "answer_completeness_rate",
        "citation_validation_summary_mean",
        "reference_in_response_validation_summary_mean",
    ]
    return pandas.DataFrame(run_level[columns])


def latency_summary(pandas: Any, run_level: Any) -> Any:
    """Return latency, usage, retrieval, and run-time confounder columns."""
    columns = [
        "dataset_segment",
        "arm_id",
        "trace_count",
        "start_time_utc",
        "start_hour_utc",
        "latency_p50_s",
        "latency_p90_s",
        "latency_p95_s",
        "latency_mean_s",
        "avg_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_llm_calls_per_trace",
        "actual_retrieval_calls_mean",
        "selected_chunk_mean",
        "context_chars_mean",
    ]
    return pandas.DataFrame(run_level[columns])


def pairwise_comparisons(
    pandas: Any,
    trace_level: Any,
    runs: Any,
    *,
    dimension_columns: list[str],
    comparison_type_fn: ComparisonTypeFn,
    orient_fn: OrientFn,
) -> tuple[Any, Any]:
    """Compute paired, directly inspectable experiment effects.

    Parameters
    ----------
    dimension_columns
        Extra per-arm columns available to ``comparison_type_fn``/``orient_fn``.
    comparison_type_fn
        Classify an unordered arm pair into a named single-dimension
        comparison, or return ``None`` when the pair is not controlled
        (differs on more than one dimension, or is otherwise incomparable).
    orient_fn
        Reorder a pair from baseline (``left``) to alternative (``right``)
        for a given comparison type.
    """
    dimensions = runs[["arm_id", "dataset_segment", *dimension_columns]].drop_duplicates()
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for dataset, dataset_dimensions in dimensions.groupby("dataset_segment"):
        records = dataset_dimensions.to_dict(orient="records")
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                comparison_type = comparison_type_fn(left, right)
                if comparison_type is None:
                    continue
                oriented_left, oriented_right = orient_fn(left, right, comparison_type)
                left_traces = trace_level[
                    (trace_level["dataset_segment"] == dataset)
                    & (trace_level["arm_id"] == oriented_left["arm_id"])
                ]
                right_traces = trace_level[
                    (trace_level["dataset_segment"] == dataset)
                    & (trace_level["arm_id"] == oriented_right["arm_id"])
                ]
                paired = left_traces[
                    [
                        "rubric_key",
                        "grade_score",
                        "is_pass",
                        "RubricV2",
                        "end_to_end_latency_s",
                    ]
                ].merge(
                    right_traces[
                        [
                            "rubric_key",
                            "grade_score",
                            "is_pass",
                            "RubricV2",
                            "end_to_end_latency_s",
                        ]
                    ],
                    on="rubric_key",
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                grade_delta = paired["grade_score_right"] - paired["grade_score_left"]
                for _, trace_pair in paired.iterrows():
                    detail_rows.append(
                        {
                            "dataset_segment": dataset,
                            "comparison_type": comparison_type,
                            "left_arm": oriented_left["arm_id"],
                            "right_arm": oriented_right["arm_id"],
                            "rubric_key": trace_pair["rubric_key"],
                            "left_grade_score": trace_pair["grade_score_left"],
                            "right_grade_score": trace_pair["grade_score_right"],
                            "grade_score_delta": (
                                trace_pair["grade_score_right"]
                                - trace_pair["grade_score_left"]
                            ),
                            "left_is_pass": trace_pair["is_pass_left"],
                            "right_is_pass": trace_pair["is_pass_right"],
                            "left_rubric_v2": trace_pair["RubricV2_left"],
                            "right_rubric_v2": trace_pair["RubricV2_right"],
                            "rubric_v2_delta": (
                                trace_pair["RubricV2_right"]
                                - trace_pair["RubricV2_left"]
                            ),
                            "left_latency_s": trace_pair["end_to_end_latency_s_left"],
                            "right_latency_s": trace_pair["end_to_end_latency_s_right"],
                            "latency_delta_s": (
                                trace_pair["end_to_end_latency_s_right"]
                                - trace_pair["end_to_end_latency_s_left"]
                            ),
                        }
                    )
                rows.append(
                    {
                        "dataset_segment": dataset,
                        "comparison_type": comparison_type,
                        "left_arm": oriented_left["arm_id"],
                        "right_arm": oriented_right["arm_id"],
                        "matched_trace_count": len(paired),
                        "right_grade_wins": int((grade_delta > 0).sum()),
                        "ties": int((grade_delta == 0).sum()),
                        "right_grade_losses": int((grade_delta < 0).sum()),
                        "right_minus_left_pass_rate": float(
                            paired["is_pass_right"].mean()
                            - paired["is_pass_left"].mean()
                        ),
                        "right_minus_left_rubric_v2_mean": float(
                            paired["RubricV2_right"].mean()
                            - paired["RubricV2_left"].mean()
                        ),
                        "right_minus_left_latency_median_s": float(
                            (
                                paired["end_to_end_latency_s_right"]
                                - paired["end_to_end_latency_s_left"]
                            ).median()
                        ),
                    }
                )
    summary = pandas.DataFrame(rows, columns=_PAIRWISE_SUMMARY_COLUMNS).sort_values(
        ["dataset_segment", "comparison_type", "left_arm", "right_arm"]
    )
    details = pandas.DataFrame(
        detail_rows, columns=_PAIRWISE_DETAIL_COLUMNS
    ).sort_values(
        [
            "dataset_segment",
            "comparison_type",
            "left_arm",
            "right_arm",
            "rubric_key",
        ]
    )
    return summary, details


def pareto_sets(pandas: Any, run_level: Any) -> Any:
    """Compute per-dataset Pareto membership under three quality definitions."""
    rows = run_level.copy()
    for output_column, quality_column in (
        ("pareto_pass_p50", "good_acceptable_rate"),
        ("pareto_good_p50", "good_rate"),
        ("pareto_rubric_p50", "rubric_v2_mean"),
    ):
        rows[output_column] = False
        for _, group in rows.groupby("dataset_segment"):
            for index, candidate in group.iterrows():
                dominated = (
                    (group[quality_column] >= candidate[quality_column])
                    & (group["latency_p50_s"] <= candidate["latency_p50_s"])
                    & (
                        (group[quality_column] > candidate[quality_column])
                        | (group["latency_p50_s"] < candidate["latency_p50_s"])
                    )
                ).any()
                rows.loc[index, output_column] = not dominated
    columns = [
        "dataset_segment",
        "arm_id",
        "good_acceptable_rate",
        "good_rate",
        "rubric_v2_mean",
        "latency_p50_s",
        "latency_p95_s",
        "critical_grade_rate",
        "critical_error_mode_rate",
        "source_fidelity_rate",
        "citation_completeness_rate",
        "avg_total_tokens",
        "pareto_pass_p50",
        "pareto_good_p50",
        "pareto_rubric_p50",
    ]
    return pandas.DataFrame(rows[columns]).sort_values(
        ["dataset_segment", "latency_p50_s"]
    )


def use_case_summary(pandas: Any, trace_level: Any) -> Any:
    """Aggregate raw quality outcomes by dataset, arm, and derived use case."""
    return (
        trace_level.groupby(["dataset_segment", "arm_id", "use_case"], dropna=False)
        .agg(
            trace_count=("trace_id", "count"),
            good_acceptable_rate=("is_pass", "mean"),
            good_rate=("is_good", "mean"),
            rubric_v2_mean=("RubricV2", "mean"),
            critical_grade_rate=("is_critical", "mean"),
            latency_p50_s=("end_to_end_latency_s", "median"),
        )
        .reset_index()
    )


def operation_timing_summary(pandas: Any, timings: Any, runs: Any) -> Any:
    """Aggregate Simple operation timing and retry diagnostics per run."""
    operations = timings[timings["simple_operation"].notna()].merge(
        runs[["run_id", "arm_id", "dataset_segment"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (dataset, arm, operation), spans in operations.groupby(
        ["dataset_segment", "arm_id", "simple_operation"]
    ):
        es_attempts = spans["es_success_attempt_count"].dropna()
        llm_attempts = spans["llm_successful_attempt_count"].dropna()
        rows.append(
            {
                "dataset_segment": dataset,
                "arm_id": arm,
                "simple_operation": operation,
                "span_count": len(spans),
                "observed_duration_p50_ms": float(
                    spans["observed_duration_ms"].quantile(
                        0.50, interpolation=PERCENTILE_METHOD
                    )
                ),
                "observed_duration_p90_ms": float(
                    spans["observed_duration_ms"].quantile(
                        0.90, interpolation=PERCENTILE_METHOD
                    )
                ),
                "observed_duration_p95_ms": float(
                    spans["observed_duration_ms"].quantile(
                        0.95, interpolation=PERCENTILE_METHOD
                    )
                ),
                "llm_successful_attempt_latency_p50_ms": float(
                    spans["llm_successful_attempt_latency_ms"].quantile(
                        0.50, interpolation=PERCENTILE_METHOD
                    )
                ),
                "es_success_duration_p50_ms": float(
                    spans["es_success_duration_ms"].quantile(
                        0.50, interpolation=PERCENTILE_METHOD
                    )
                ),
                "llm_attempt_observation_count": len(llm_attempts),
                "llm_retry_rate": float((llm_attempts > 1).mean())
                if len(llm_attempts)
                else float("nan"),
                "es_attempt_observation_count": len(es_attempts),
                "es_retry_rate": float((es_attempts > 1).mean())
                if len(es_attempts)
                else float("nan"),
            }
        )
    return pandas.DataFrame(rows, columns=_OPERATION_TIMING_COLUMNS).sort_values(
        ["dataset_segment", "arm_id", "simple_operation"]
    )


def grade_distribution(pandas: Any, trace_level: Any) -> Any:
    """Publish raw grade counts and rates by arm and dataset."""
    counts = (
        trace_level.groupby(["dataset_segment", "arm_id", "ordinal_grade"])
        .size()
        .reset_index(name="grade_count")
    )
    totals = counts.groupby(["dataset_segment", "arm_id"])["grade_count"].transform(
        "sum"
    )
    counts["trace_count"] = totals
    counts["grade_rate"] = counts["grade_count"] / totals
    return pandas.DataFrame(counts).sort_values(
        ["dataset_segment", "arm_id", "ordinal_grade"]
    )


def _rubric_key_map(timings: Any) -> Any:
    """Return one root-invocation ``rubric_key`` row per trace."""
    return timings[
        (timings["span_type"] == "UNKNOWN")
        & timings["span_name"].str.startswith("invoke_", na=False)
    ][["run_id", "trace_id", "span_name"]].rename(columns={"span_name": "rubric_key"})


def _chunk_identities(payloads: Any) -> set[str]:
    """Union ranked chunk identities from JSON-encoded rank-list payloads."""
    chunk_ids: set[str] = set()
    for payload in payloads.dropna():
        for item in json.loads(payload):
            chunk_ids.add(f"{item.get('document_id')}-{item.get('chunk_id')}")
    return chunk_ids


def between_arm_family_overlap(
    pandas: Any,
    retrieval: Any,
    timings: Any,
    runs: Any,
    *,
    family_column: str,
    left_family: str,
    right_family: str,
    match_columns: list[str],
) -> Any:
    """Measure ranked-chunk overlap between two arms that isolate one signal.

    Use this when each arm uses exactly one retrieval family (for example
    Stage A's ``bm25``-only versus ``dense``-only arms) and two arms are
    matched on ``match_columns`` (for example equal call count and context).
    """
    rubric_map = _rubric_key_map(timings)
    tool_rows = retrieval[retrieval["simple_operation"].notna()].merge(
        rubric_map, on=["run_id", "trace_id"], how="left", validate="many_to_one"
    )
    tool_rows = tool_rows.merge(
        runs[["run_id", "arm_id", "dataset_segment", family_column, *match_columns]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    inventories: list[dict[str, object]] = []
    for (dataset, arm, rubric_key), rows in tool_rows.groupby(
        ["dataset_segment", "arm_id", "rubric_key"]
    ):
        dimension = rows.iloc[0]
        inventories.append(
            {
                "dataset_segment": dataset,
                "arm_id": arm,
                "rubric_key": rubric_key,
                family_column: dimension[family_column],
                **{column: dimension[column] for column in match_columns},
                "chunk_ids": _chunk_identities(rows["ranked_chunk_ids"]),
            }
        )
    overlap_columns = [
        "dataset_segment",
        *match_columns,
        "rubric_key",
        f"{left_family}_arm",
        f"{right_family}_arm",
        f"{left_family}_chunk_count",
        f"{right_family}_chunk_count",
        "intersection_count",
        "union_count",
        "jaccard",
        f"{left_family}_unique_count",
        f"{right_family}_unique_count",
    ]
    if not inventories:
        return pandas.DataFrame(columns=overlap_columns)
    inventory = pandas.DataFrame(inventories)
    left_inventory = inventory[inventory[family_column] == left_family]
    right_inventory = inventory[inventory[family_column] == right_family]
    paired = left_inventory.merge(
        right_inventory,
        on=["dataset_segment", *match_columns, "rubric_key"],
        suffixes=(f"_{left_family}", f"_{right_family}"),
        validate="one_to_one",
    )
    rows_out: list[dict[str, object]] = []
    for _, row in paired.iterrows():
        left = row[f"chunk_ids_{left_family}"]
        right = row[f"chunk_ids_{right_family}"]
        union = left | right
        rows_out.append(
            {
                "dataset_segment": row["dataset_segment"],
                **{column: row[column] for column in match_columns},
                "rubric_key": row["rubric_key"],
                f"{left_family}_arm": row[f"arm_id_{left_family}"],
                f"{right_family}_arm": row[f"arm_id_{right_family}"],
                f"{left_family}_chunk_count": len(left),
                f"{right_family}_chunk_count": len(right),
                "intersection_count": len(left & right),
                "union_count": len(union),
                "jaccard": len(left & right) / len(union) if union else 1.0,
                f"{left_family}_unique_count": len(left - right),
                f"{right_family}_unique_count": len(right - left),
            }
        )
    return pandas.DataFrame(rows_out, columns=overlap_columns).sort_values(
        ["dataset_segment", *match_columns, "rubric_key"]
    )


def within_trace_family_overlap(
    pandas: Any,
    retrieval: Any,
    timings: Any,
    runs: Any,
    *,
    left_family: str,
    right_family: str,
    eligible_run_ids: set[str] | None = None,
) -> Any:
    """Measure ranked-chunk overlap between two signals within one trace.

    Use this when a single arm issues concurrent calls of more than one
    retrieval mode within the same tool round (for example a Stage B
    heterogeneous ``current_union`` arm mixing ``bm25`` and ``dense`` calls).
    Only traces where both families contributed at least one chunk are
    included. ``eligible_run_ids`` can restrict the calculation to arms whose
    configured tool multiset contains both families.
    """
    rubric_map = _rubric_key_map(timings)
    tool_rows = retrieval[retrieval["retrieval_mode"].notna()].merge(
        rubric_map, on=["run_id", "trace_id"], how="left", validate="many_to_one"
    )
    if eligible_run_ids is not None:
        tool_rows = tool_rows[tool_rows["run_id"].isin(eligible_run_ids)]
    tool_rows = tool_rows.merge(
        runs[["run_id", "arm_id", "dataset_segment"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (dataset, arm, trace_id, rubric_key), group in tool_rows.groupby(
        ["dataset_segment", "arm_id", "trace_id", "rubric_key"]
    ):
        left = _chunk_identities(
            group[group["retrieval_mode"] == left_family]["ranked_chunk_ids"]
        )
        right = _chunk_identities(
            group[group["retrieval_mode"] == right_family]["ranked_chunk_ids"]
        )
        if not left or not right:
            continue
        union = left | right
        rows.append(
            {
                "dataset_segment": dataset,
                "arm_id": arm,
                "trace_id": trace_id,
                "rubric_key": rubric_key,
                f"{left_family}_chunk_count": len(left),
                f"{right_family}_chunk_count": len(right),
                "intersection_count": len(left & right),
                "union_count": len(union),
                "jaccard": len(left & right) / len(union) if union else 1.0,
                f"{left_family}_unique_count": len(left - right),
                f"{right_family}_unique_count": len(right - left),
            }
        )
    overlap_columns = [
        "dataset_segment",
        "arm_id",
        "trace_id",
        "rubric_key",
        f"{left_family}_chunk_count",
        f"{right_family}_chunk_count",
        "intersection_count",
        "union_count",
        "jaccard",
        f"{left_family}_unique_count",
        f"{right_family}_unique_count",
    ]
    return pandas.DataFrame(rows, columns=overlap_columns).sort_values(
        ["dataset_segment", "arm_id", "rubric_key"]
    )


def population_validation(
    *,
    snapshot_dir: Path,
    manifest: dict[str, Any],
    experiments: Any,
    runs: Any,
    expected_experiment_count: int,
    expected_run_count: int,
    expected_arm_count: int,
    expected_dataset_count: int,
) -> dict[str, object]:
    """Return expected-versus-observed population and checksum diagnostics."""
    return {
        "manifest_experiment_count": manifest.get("experiment_count"),
        "manifest_run_count": manifest.get("run_count"),
        "expected_experiment_count": expected_experiment_count,
        "expected_run_count": expected_run_count,
        "expected_arm_count": expected_arm_count,
        "expected_dataset_count": expected_dataset_count,
        "observed_experiment_count": len(experiments),
        "observed_run_count": len(runs),
        "observed_arm_count": int(runs["arm_id"].nunique()),
        "observed_dataset_count": int(runs["dataset_segment"].nunique()),
        "snapshot_file_checksums_match": manifest_checksums_match(
            snapshot_dir, manifest
        ),
    }


def markdown_table(dataframe: Any) -> str:
    """Render a compact Markdown table without optional dependencies."""
    headers = [str(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in dataframe.to_dict(orient="records"):
        cells: list[str] = []
        for column in dataframe.columns:
            value = record[column]
            if is_missing(value):
                text = "—"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def decode_assessment_row(row: Any) -> object:
    """Decode the snapshot's type-tagged assessment scalar."""
    value_type = row["assessment_value_type"]
    if value_type is None or is_missing(value_type):
        return None
    columns = {
        "string": "assessment_value_string",
        "bool": "assessment_value_bool",
        "int": "assessment_value_int",
        "float": "assessment_value_float",
    }
    column = columns.get(value_type)
    if column is None:
        raise ValueError(f"Unknown assessment value type {value_type!r}.")
    return row[column]


def to_boolean(value: object) -> bool | None:
    """Normalize supported boolean assessment representations."""
    if value is None or is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Unsupported boolean assessment value {value!r}.")


def use_case_from_rubric(rubric_key: str) -> str:
    """Derive a use-case name from a stable invoke span name."""
    import re

    match = re.fullmatch(r"^invoke_(?P<use_case>.+?)_\d+(?:_v\d+)?$", rubric_key)
    return match.group("use_case") if match else rubric_key.removeprefix("invoke_")


def metric_value(metric_row: Any, key: str) -> float:
    """Return a scalar run metric."""
    value = metric_row.get(key)
    return float(value) if value is not None and not is_missing(value) else float("nan")


def is_missing(value: object) -> bool:
    """Return whether a scalar is pandas/IEEE missing."""
    return isinstance(value, float) and math.isnan(value)


def manifest_checksums_match(snapshot_dir: Path, manifest: dict[str, Any]) -> bool:
    """Verify every snapshot file checksum recorded in the manifest."""
    return all(
        sha256(snapshot_dir / str(item["name"])) == item["sha256"]
        for item in manifest.get("files", [])
    )


def file_hashes(paths: list[Path]) -> list[dict[str, object]]:
    """Return reproducibility metadata for files."""
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write stable human-readable JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_pandas() -> Any:
    """Load the optional MLflow analysis dependency."""
    try:
        import pandas
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional dependency group before analysis: "
            "uv sync --group mlflow"
        ) from exc
    return pandas
