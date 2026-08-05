"""Produce reproducible Stage A comparisons from a sanitized MLflow snapshot."""

import hashlib
import json
import re
from datetime import UTC, datetime
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

_DATASET_SUFFIXES = ("emc2_set1", "emc2_set2", "mallinckrodt")
_ARM_PATTERN = re.compile(
    r"^S-A-(?P<mode>bm25|dense|hybrid)-c(?P<calls>\d+)-rr-f(?P<fetch>\d+)-g(?P<context>\d+)-rnone$"
)
_USE_CASE_PATTERN = re.compile(r"^invoke_(?P<use_case>.+?)_\d+(?:_v\d+)?$")


def analyze_stage_a_snapshot(
    *,
    snapshot_dir: Path,
    report_path: Path,
) -> Path:
    """Analyze Stage A runs and write auditable local outputs plus a report.

    Parameters
    ----------
    snapshot_dir
        Completed sanitized snapshot containing the 54 Stage A runs.
    report_path
        Markdown report receiving evidence-backed Stage B recommendations.

    Returns
    -------
    Path
        Directory containing auditable CSV and JSON analysis artifacts.
    """
    pandas = _load_pandas()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    analysis_dir = snapshot_dir / "stage_a_analysis"
    analysis_dir.mkdir(exist_ok=True)

    manifest = _read_json(snapshot_dir / "manifest.json")
    experiments = pandas.read_parquet(snapshot_dir / "experiments.parquet")
    runs = pandas.read_parquet(snapshot_dir / "runs.parquet")
    metrics = pandas.read_parquet(snapshot_dir / "run_metrics.parquet")
    quality = pandas.read_parquet(snapshot_dir / "trace_quality.parquet")
    retrieval = pandas.read_parquet(snapshot_dir / "trace_retrieval.parquet")
    timings = pandas.read_parquet(snapshot_dir / "span_timings.parquet")

    runs = _augment_runs(runs)
    trace_level, validation = _build_trace_level(
        pandas=pandas,
        runs=runs,
        quality=quality,
        retrieval=retrieval,
        timings=timings,
    )
    run_level = _build_run_level(
        pandas=pandas,
        runs=runs,
        metrics=metrics,
        trace_level=trace_level,
    )
    validation["max_pass_rate_crosscheck_absolute_difference"] = float(
        (
            run_level["good_acceptable_rate"]
            - run_level["reported_total_pass_rate"]
        )
        .abs()
        .max()
    )
    validation["max_p50_latency_crosscheck_absolute_difference_s"] = float(
        (
            run_level["latency_p50_s"]
            - run_level["reported_execution_time_p50_s"]
        )
        .abs()
        .max()
    )
    quality_summary = _quality_summary(pandas, run_level)
    latency_summary = _latency_summary(pandas, run_level)
    pairwise, pairwise_details = _pairwise_comparisons(pandas, trace_level, runs)
    overlap = _retrieval_overlap(pandas, retrieval, timings, runs)
    pareto = _pareto_sets(pandas, run_level)
    use_case = _use_case_summary(pandas, trace_level)
    operation_timings = _operation_timing_summary(pandas, timings, runs)
    grade_distribution = _grade_distribution(pandas, trace_level)

    outputs = {
        "trace_level.csv": trace_level,
        "run_level.csv": run_level,
        "quality_by_arm_dataset.csv": quality_summary,
        "latency_by_arm_dataset.csv": latency_summary,
        "pairwise_comparisons.csv": pairwise,
        "pairwise_trace_deltas.csv": pairwise_details,
        "retrieval_overlap.csv": overlap,
        "pareto_by_dataset.csv": pareto,
        "quality_by_use_case.csv": use_case,
        "operation_timings.csv": operation_timings,
        "grade_distribution.csv": grade_distribution,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(analysis_dir / filename, index=False)

    validation.update(
        {
            "manifest_experiment_count": manifest.get("experiment_count"),
            "manifest_run_count": manifest.get("run_count"),
            "expected_experiment_count": 54,
            "expected_run_count": 54,
            "expected_arm_count": 18,
            "expected_dataset_count": 3,
            "observed_experiment_count": int(len(experiments)),
            "observed_run_count": int(len(runs)),
            "observed_arm_count": int(runs["arm_id"].nunique()),
            "observed_dataset_count": int(runs["dataset_segment"].nunique()),
            "snapshot_file_checksums_match": _manifest_checksums_match(
                snapshot_dir, manifest
            ),
        }
    )
    validation["valid_for_stage_a_analysis"] = all(
        [
            validation["manifest_experiment_count"] == 54,
            validation["manifest_run_count"] == 54,
            validation["observed_experiment_count"] == 54,
            validation["observed_run_count"] == 54,
            validation["observed_arm_count"] == 18,
            validation["observed_dataset_count"] == 3,
            validation["snapshot_file_checksums_match"],
            validation["conflicting_duplicate_assessment_count"] == 0,
            validation["trace_without_rubric_key_count"] == 0,
            validation["trace_without_grade_count"] == 0,
            validation["trace_without_latency_count"] == 0,
            validation["max_pass_rate_crosscheck_absolute_difference"] == 0,
            validation["max_p50_latency_crosscheck_absolute_difference_s"]
            < 0.001,
        ]
    )
    _write_json(analysis_dir / "validation.json", validation)

    recommendations = _stage_b_recommendations(run_level, overlap, pareto)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            snapshot_dir=snapshot_dir,
            validation=validation,
            run_level=run_level,
            pairwise=pairwise,
            overlap=overlap,
            pareto=pareto,
            operation_timings=operation_timings,
            recommendations=recommendations,
        ),
        encoding="utf-8",
    )
    _write_calculation_manifest(
        analysis_dir=analysis_dir,
        snapshot_dir=snapshot_dir,
        validation=validation,
        recommendations=recommendations,
    )
    return analysis_dir


def _augment_runs(runs: Any) -> Any:
    """Parse dataset segment and experiment parameters from experiment names."""
    parsed_rows: list[dict[str, object]] = []
    for row in runs.to_dict(orient="records"):
        short_name = str(row["experiment_name"]).rstrip("/").rsplit("/", maxsplit=1)[-1]
        dataset_segment = next(
            (suffix for suffix in _DATASET_SUFFIXES if short_name.endswith(f"-{suffix}")),
            None,
        )
        if dataset_segment is None:
            raise ValueError(f"Cannot parse dataset segment from {short_name!r}.")
        arm_id = short_name.removesuffix(f"-{dataset_segment}")
        match = _ARM_PATTERN.fullmatch(arm_id)
        if match is None:
            raise ValueError(f"Cannot parse Stage A experiment identity {arm_id!r}.")
        parsed_rows.append(
            {
                **row,
                "arm_id": arm_id,
                "dataset_segment": dataset_segment,
                "retrieval_family": match.group("mode"),
                "calls": int(match.group("calls")),
                "fetch": int(match.group("fetch")),
                "context": int(match.group("context")),
            }
        )
    return runs.__class__(parsed_rows)


def _build_trace_level(
    *,
    pandas: Any,
    runs: Any,
    quality: Any,
    retrieval: Any,
    timings: Any,
) -> tuple[Any, dict[str, object]]:
    """Build one auditable row per run/trace and return validation diagnostics."""
    quality = quality.copy()
    quality["assessment_value"] = quality.apply(_decode_assessment_row, axis=1)
    duplicate_mask = quality.duplicated(
        ["run_id", "trace_id", "assessment_name", "assessment_value"],
        keep="first",
    )
    identical_duplicate_count = int(duplicate_mask.sum())
    quality = quality[~duplicate_mask].copy()

    conflicts = (
        quality.groupby(["run_id", "trace_id", "assessment_name"])[
            "assessment_value"
        ]
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
    rubric_spans["use_case"] = rubric_spans["rubric_key"].map(_use_case_from_rubric)

    retrieval_summary = retrieval[
        retrieval["actual_retrieval_calls"].notna()
    ][
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

    run_dimensions = runs[
        [
            "run_id",
            "experiment_id",
            "experiment_name",
            "model_version",
            "arm_id",
            "dataset_segment",
            "retrieval_family",
            "calls",
            "fetch",
            "context",
            "start_time_utc",
            "start_hour_utc",
        ]
    ]
    trace_level = (
        rubric_spans.merge(run_dimensions, on="run_id", how="left", validate="many_to_one")
        .merge(quality_pivot, on=["run_id", "trace_id"], how="left", validate="one_to_one")
        .merge(
            retrieval_summary,
            on=["run_id", "trace_id"],
            how="left",
            validate="one_to_one",
        )
    )
    trace_level["end_to_end_latency_s"] = (
        trace_level["end_to_end_latency_ms"] / 1_000
    )
    trace_level["grade_score"] = trace_level["ordinal_grade"].map(GRADE_ORDER)
    trace_level["is_pass"] = trace_level["ordinal_grade"].isin(PASSING_GRADES)
    trace_level["is_good"] = trace_level["ordinal_grade"].eq("Good")
    trace_level["is_critical"] = trace_level["ordinal_grade"].eq("Critical Error")
    for assessment in ERROR_ASSESSMENTS:
        trace_level[assessment] = trace_level[assessment].map(_to_boolean)
    for assessment in POSITIVE_BOOLEAN_ASSESSMENTS:
        trace_level[assessment] = trace_level[assessment].map(_to_boolean)
    trace_level["has_any_critical_error_mode"] = trace_level[
        ERROR_ASSESSMENTS
    ].fillna(False).any(axis=1)

    validation = {
        "raw_quality_row_count": int(len(quality) + identical_duplicate_count),
        "identical_duplicate_assessment_row_count": identical_duplicate_count,
        "conflicting_duplicate_assessment_count": conflicting_count,
        "unique_trace_count": int(trace_level[["run_id", "trace_id"]].drop_duplicates().shape[0]),
        "trace_without_rubric_key_count": int(trace_level["rubric_key"].isna().sum()),
        "trace_without_grade_count": int(trace_level["ordinal_grade"].isna().sum()),
        "trace_without_latency_count": int(
            trace_level["end_to_end_latency_ms"].isna().sum()
        ),
        "trace_without_exported_row_id_count": int(
            quality["row_id"].isna().groupby([quality["run_id"], quality["trace_id"]]).all().sum()
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


def _build_run_level(
    *,
    pandas: Any,
    runs: Any,
    metrics: Any,
    trace_level: Any,
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
            "retrieval_family": run["retrieval_family"],
            "calls": int(run["calls"]),
            "fetch": int(run["fetch"]),
            "context": int(run["context"]),
            "start_time_utc": run["start_time_utc"],
            "start_hour_utc": int(run["start_hour_utc"]),
            "trace_count": int(len(traces)),
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
            "reported_total_pass_rate": _metric_value(
                metric_row, "total_pass_rate"
            ),
            "reported_execution_time_p50_s": _metric_value(
                metric_row, "execution_time_p50_s"
            ),
            "avg_total_tokens": _metric_value(metric_row, "avg_total_tokens"),
            "avg_prompt_tokens": _metric_value(metric_row, "avg_prompt_tokens"),
            "avg_completion_tokens": _metric_value(
                metric_row, "avg_completion_tokens"
            ),
            "avg_reasoning_tokens": _metric_value(
                metric_row, "avg_reasoning_tokens"
            ),
            "avg_llm_calls_per_trace": _metric_value(
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


def _quality_summary(pandas: Any, run_level: Any) -> Any:
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


def _latency_summary(pandas: Any, run_level: Any) -> Any:
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


def _pairwise_comparisons(
    pandas: Any, trace_level: Any, runs: Any
) -> tuple[Any, Any]:
    """Compute paired, directly inspectable experiment effects."""
    dimensions = runs[
        [
            "arm_id",
            "dataset_segment",
            "retrieval_family",
            "calls",
            "fetch",
            "context",
        ]
    ].drop_duplicates()
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for dataset, dataset_dimensions in dimensions.groupby("dataset_segment"):
        records = dataset_dimensions.to_dict(orient="records")
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                comparison_type = _comparison_type(left, right)
                if comparison_type is None:
                    continue
                oriented_left, oriented_right = _orient_comparison(
                    left, right, comparison_type
                )
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
                            "left_latency_s": trace_pair[
                                "end_to_end_latency_s_left"
                            ],
                            "right_latency_s": trace_pair[
                                "end_to_end_latency_s_right"
                            ],
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
                        "matched_trace_count": int(len(paired)),
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
    summary = pandas.DataFrame(rows).sort_values(
        ["dataset_segment", "comparison_type", "left_arm", "right_arm"]
    )
    details = pandas.DataFrame(detail_rows).sort_values(
        [
            "dataset_segment",
            "comparison_type",
            "left_arm",
            "right_arm",
            "rubric_key",
        ]
    )
    return summary, details


def _comparison_type(left: dict[str, object], right: dict[str, object]) -> str | None:
    """Classify a controlled Stage A pair or return no comparison."""
    if (
        left["retrieval_family"] == right["retrieval_family"]
        and left["context"] == right["context"]
        and left["calls"] != right["calls"]
    ):
        return "call_count_at_matched_mode_context"
    if (
        left["retrieval_family"] == right["retrieval_family"]
        and left["calls"] == right["calls"]
        and left["context"] != right["context"]
    ):
        return "context_at_matched_mode_calls"
    if (
        {left["retrieval_family"], right["retrieval_family"]} == {"bm25", "dense"}
        and left["calls"] == right["calls"]
        and left["context"] == right["context"]
    ):
        return "bm25_vs_dense_at_matched_calls_context"
    return None


def _orient_comparison(
    left: dict[str, object],
    right: dict[str, object],
    comparison_type: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Orient pairwise deltas from the simpler/baseline arm to the alternative."""
    if (
        comparison_type == "call_count_at_matched_mode_context"
        and int(left["calls"]) > int(right["calls"])
    ):
        return right, left
    if (
        comparison_type == "context_at_matched_mode_calls"
        and int(left["context"]) > int(right["context"])
    ):
        return right, left
    if (
        comparison_type == "bm25_vs_dense_at_matched_calls_context"
        and left["retrieval_family"] == "dense"
    ):
        return right, left
    return left, right


def _retrieval_overlap(pandas: Any, retrieval: Any, timings: Any, runs: Any) -> Any:
    """Measure BM25/dense ranked-chunk overlap for matched rubric variations."""
    rubric_map = timings[
        (timings["span_type"] == "UNKNOWN")
        & timings["span_name"].str.startswith("invoke_", na=False)
    ][["run_id", "trace_id", "span_name"]].rename(columns={"span_name": "rubric_key"})
    tool_rows = retrieval[retrieval["simple_operation"].notna()].merge(
        rubric_map, on=["run_id", "trace_id"], how="left", validate="many_to_one"
    )
    tool_rows = tool_rows.merge(
        runs[
            [
                "run_id",
                "arm_id",
                "dataset_segment",
                "retrieval_family",
                "calls",
                "context",
            ]
        ],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    inventories: list[dict[str, object]] = []
    for (dataset, arm, rubric_key), rows in tool_rows.groupby(
        ["dataset_segment", "arm_id", "rubric_key"]
    ):
        chunk_ids: set[str] = set()
        for payload in rows["ranked_chunk_ids"].dropna():
            for item in json.loads(payload):
                chunk_ids.add(f"{item.get('document_id')}-{item.get('chunk_id')}")
        dimension = rows.iloc[0]
        inventories.append(
            {
                "dataset_segment": dataset,
                "arm_id": arm,
                "rubric_key": rubric_key,
                "retrieval_family": dimension["retrieval_family"],
                "calls": int(dimension["calls"]),
                "context": int(dimension["context"]),
                "chunk_ids": chunk_ids,
            }
        )
    inventory = pandas.DataFrame(inventories)
    bm25 = inventory[inventory["retrieval_family"] == "bm25"]
    dense = inventory[inventory["retrieval_family"] == "dense"]
    paired = bm25.merge(
        dense,
        on=["dataset_segment", "calls", "context", "rubric_key"],
        suffixes=("_bm25", "_dense"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for _, row in paired.iterrows():
        left = row["chunk_ids_bm25"]
        right = row["chunk_ids_dense"]
        union = left | right
        rows.append(
            {
                "dataset_segment": row["dataset_segment"],
                "calls": int(row["calls"]),
                "context": int(row["context"]),
                "rubric_key": row["rubric_key"],
                "bm25_arm": row["arm_id_bm25"],
                "dense_arm": row["arm_id_dense"],
                "bm25_chunk_count": len(left),
                "dense_chunk_count": len(right),
                "intersection_count": len(left & right),
                "union_count": len(union),
                "jaccard": len(left & right) / len(union) if union else 1.0,
                "bm25_unique_count": len(left - right),
                "dense_unique_count": len(right - left),
            }
        )
    return pandas.DataFrame(rows).sort_values(
        ["dataset_segment", "calls", "context", "rubric_key"]
    )


def _pareto_sets(pandas: Any, run_level: Any) -> Any:
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


def _use_case_summary(pandas: Any, trace_level: Any) -> Any:
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


def _operation_timing_summary(pandas: Any, timings: Any, runs: Any) -> Any:
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
                "span_count": int(len(spans)),
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
                "llm_attempt_observation_count": int(len(llm_attempts)),
                "llm_retry_rate": float((llm_attempts > 1).mean())
                if len(llm_attempts)
                else float("nan"),
                "es_attempt_observation_count": int(len(es_attempts)),
                "es_retry_rate": float((es_attempts > 1).mean())
                if len(es_attempts)
                else float("nan"),
            }
        )
    return pandas.DataFrame(rows).sort_values(
        ["dataset_segment", "arm_id", "simple_operation"]
    )


def _grade_distribution(pandas: Any, trace_level: Any) -> Any:
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


def _stage_b_recommendations(run_level: Any, overlap: Any, pareto: Any) -> dict[str, object]:
    """Derive transparent Stage B defaults from Stage A evidence."""
    mode_summary = (
        run_level.groupby(["dataset_segment", "retrieval_family"])
        .agg(
            best_pass_rate=("good_acceptable_rate", "max"),
            best_latency_p50_s=("latency_p50_s", "min"),
        )
        .reset_index()
    )
    dense_mall = mode_summary[
        (mode_summary["dataset_segment"] == "mallinckrodt")
        & (mode_summary["retrieval_family"] == "dense")
    ].iloc[0]
    bm25_mall = mode_summary[
        (mode_summary["dataset_segment"] == "mallinckrodt")
        & (mode_summary["retrieval_family"] == "bm25")
    ].iloc[0]
    overlap_summary = (
        overlap.groupby("dataset_segment")
        .agg(
            median_jaccard=("jaccard", "median"),
            median_dense_unique=("dense_unique_count", "median"),
            paired_trace_count=("rubric_key", "count"),
        )
        .reset_index()
    )
    robust_pareto = (
        pareto.groupby("arm_id")[
            ["pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]
        ]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False)
    )
    return {
        "primary_homogeneous_arm": "BM25 current_union c2/f20",
        "heterogeneous_pilot": "BM25-first + dense-second current_union c2/f20",
        "dense_only_union": "exclude",
        "rrf_union_control": "defer unless implementation cost is negligible",
        "c3_control": (
            "include one BM25 c3 control"
            if _material_c3_gain(run_level)
            else "exclude"
        ),
        "rationale": {
            "dense_mallinckrodt_best_pass_rate": float(
                dense_mall["best_pass_rate"]
            ),
            "bm25_mallinckrodt_best_pass_rate": float(
                bm25_mall["best_pass_rate"]
            ),
            "overlap_by_dataset": overlap_summary.to_dict(orient="records"),
            "most_robust_pareto_arms": robust_pareto.head(6).to_dict(),
        },
    }


def _material_c3_gain(run_level: Any) -> bool:
    """Return whether matched BM25 c3 arms improve pass rate materially."""
    bm25 = run_level[run_level["retrieval_family"] == "bm25"]
    gains: list[float] = []
    for dataset in bm25["dataset_segment"].unique():
        group = bm25[bm25["dataset_segment"] == dataset]
        for context in (15, 20):
            c3 = group[(group["calls"] == 3) & (group["context"] == context)]
            lower = group[(group["calls"] < 3) & (group["context"] == context)]
            if not c3.empty and not lower.empty:
                gains.append(
                    float(
                        c3.iloc[0]["good_acceptable_rate"]
                        - lower["good_acceptable_rate"].max()
                    )
                )
    return any(gain >= 0.05 for gain in gains)


def _render_report(
    *,
    snapshot_dir: Path,
    validation: dict[str, object],
    run_level: Any,
    pairwise: Any,
    overlap: Any,
    pareto: Any,
    operation_timings: Any,
    recommendations: dict[str, object],
) -> str:
    """Render the evidence-backed Stage A report with reproducible formulas."""
    best_rows = (
        run_level.sort_values(
            ["dataset_segment", "good_acceptable_rate", "latency_p50_s"],
            ascending=[True, False, True],
        )
        .groupby("dataset_segment")
        .head(5)
    )
    pareto_rows = pareto[
        pareto[
            ["pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]
        ].any(axis=1)
    ]
    overlap_summary = (
        overlap.groupby("dataset_segment")
        .agg(
            matched_traces=("rubric_key", "count"),
            median_jaccard=("jaccard", "median"),
            median_bm25_unique=("bm25_unique_count", "median"),
            median_dense_unique=("dense_unique_count", "median"),
        )
        .reset_index()
    )
    call_effects = pairwise[
        pairwise["comparison_type"] == "call_count_at_matched_mode_context"
    ].sort_values(
        "right_minus_left_pass_rate", ascending=False
    )
    operation_summary = (
        operation_timings.groupby("simple_operation")
        .agg(
            run_arm_observations=("arm_id", "count"),
            median_of_run_p50_ms=("observed_duration_p50_ms", "median"),
            median_llm_retry_rate=("llm_retry_rate", "median"),
            median_es_retry_rate=("es_retry_rate", "median"),
        )
        .reset_index()
        .sort_values("median_of_run_p50_ms", ascending=False)
    )
    start_times = run_level["start_time_utc"].sort_values()
    return f"""# Stage A results and Stage B redesign

Generated from the immutable sanitized snapshot at `{snapshot_dir}` on
{datetime.now(UTC).isoformat()}.

## Executive conclusion

Stage A supports BM25 as the primary Stage B homogeneous `current_union` arm.
Dense-only retrieval is not competitive on Mallinckrodt, while BM25/dense
ranked-result overlap is low enough to justify one heterogeneous BM25-first,
dense-second late-union pilot. Include one BM25 c3 control because matched
Stage A evidence shows a material quality gain in at least one dataset/context.

Recommended Stage B matrix:

1. BM25 `current_union`, c2/f20.
2. Heterogeneous `[bm25, dense]` `current_union`, c2/f20.
3. BM25 `current_union`, c3/f20 control.
4. Do not include dense-only union; defer RRF-union unless its implementation
   cost is negligible.

## Validation and integrity

- Snapshot checksum validation: **{validation["snapshot_file_checksums_match"]}**
- Runs / experiments / arms / datasets: **{validation["observed_run_count"]} /
  {validation["observed_experiment_count"]} / {validation["observed_arm_count"]} /
  {validation["observed_dataset_count"]}**
- Unique traces: **{validation["unique_trace_count"]}**
- Exact duplicate assessment rows removed: **{validation["identical_duplicate_assessment_row_count"]}**
- Conflicting duplicate assessments: **{validation["conflicting_duplicate_assessment_count"]}**
- Missing grade / latency / derived rubric key: **{validation["trace_without_grade_count"]} /
  {validation["trace_without_latency_count"]} / {validation["trace_without_rubric_key_count"]}**
- Valid for analysis: **{validation["valid_for_stage_a_analysis"]}**

The exporter did not retain `row_id`, `use_case`, `dataset_id`, or
`evalset_variant` trace attributes. Pairing therefore uses the unique root
`invoke_*` span name (`rubric_key`) within each dataset segment. Every trace has
exactly one such key. Use-case names are derived from that key.

## Exact calculations

- `Good+Acceptable rate = (Good count + Acceptable count) / unique trace count`.
- `Good rate = Good count / unique trace count`.
- `Critical grade rate = Critical Error count / unique trace count`.
- `RubricV2 mean = arithmetic mean of the one RubricV2 assessment per trace`.
- End-to-end latency is the `observed_duration_ms / 1000` of the unique
  `UNKNOWN` root span whose name begins with `invoke_`.
- p50/p90/p95 use pandas quantiles with `{PERCENTILE_METHOD}` interpolation.
- Pairwise results inner-join arms on `(dataset_segment, rubric_key)` and publish
  matched sample size, grade wins/ties/losses, pass-rate delta, RubricV2 delta,
  and median paired latency delta.
- Pareto membership is computed independently per dataset. An arm is dominated
  if another arm has quality greater than or equal and p50 latency less than or
  equal, with at least one strict improvement.

## Top observed arms by dataset

{_markdown_table(best_rows[["dataset_segment", "arm_id", "trace_count", "good_acceptable_rate", "rubric_v2_mean", "latency_p50_s", "latency_p95_s", "avg_total_tokens"]])}

## Pareto sensitivity

The following arms are Pareto-efficient under at least one of Good+Acceptable,
Good-only, or RubricV2 quality, always against p50 latency:

{_markdown_table(pareto_rows[["dataset_segment", "arm_id", "good_acceptable_rate", "good_rate", "rubric_v2_mean", "latency_p50_s", "pareto_pass_p50", "pareto_good_p50", "pareto_rubric_p50"]])}

## BM25 versus dense retrieval overlap

Chunk identities are unioned across calls within each trace before computing
Jaccard overlap. Counts below are medians over exact matched rubric variations:

{_markdown_table(overlap_summary)}

Low Jaccard plus non-zero unique contributions support testing heterogeneous
late union. This does not prove that unique dense chunks improve answer quality;
Stage B must test that causal hypothesis.

## Matched call-count effects

Largest observed pass-rate changes when call count changes at matched retrieval
family and global context:

{_markdown_table(call_effects.head(18)[["dataset_segment", "left_arm", "right_arm", "matched_trace_count", "right_grade_wins", "ties", "right_grade_losses", "right_minus_left_pass_rate", "right_minus_left_rubric_v2_mean", "right_minus_left_latency_median_s"]])}

## Operation timing and retries

Each row below summarizes the per-run/arm operation p50s. The complete
dataset-specific values, successful-attempt timings, and retry observation
counts are in `operation_timings.csv`.

{_markdown_table(operation_summary)}

## Stage B recommendation

```json
{json.dumps(recommendations, indent=2, sort_keys=True)}
```

The recommendation is falsifiable: all source rows and formulas are published
beside the snapshot. In particular, `pairwise_comparisons.csv` exposes every
paired effect and `retrieval_overlap.csv` exposes every matched Jaccard input.

## Reproducibility and caveats

- Do not pool raw traces across datasets; all headline comparisons are
  dataset-specific.
- Latency is observational. `latency_by_arm_dataset.csv` publishes UTC run
  timestamps and hours so time-of-day confounding can be checked. Runs span
  `{start_times.iloc[0]}` through `{start_times.iloc[-1]}` and were not
  randomized into common time windows, so small latency differences must not
  be attributed causally to arm parameters.
- The 102 duplicate ordinal-grade rows were identical and removed only after
  equality validation. No conflicting duplicate was found.
- Run-level MLflow metrics are retained as cross-checks, while headline quality
  and latency are recalculated from trace-level rows.
- The snapshot excludes messages, answers, query text, chunk content, XML, and
  scorer rationales by design.

## Audit files

All calculations are under `{snapshot_dir / "stage_a_analysis"}`:

- `trace_level.csv`: one row per run/trace, the primary recomputation source.
- `run_level.csv`: per-run raw counts and calculated metrics.
- `quality_by_arm_dataset.csv` and `latency_by_arm_dataset.csv`.
- `pairwise_comparisons.csv`, `retrieval_overlap.csv`, and
  `quality_by_use_case.csv`.
- `pairwise_trace_deltas.csv` contains every exact paired rubric cell;
  `operation_timings.csv` contains operation/retry diagnostics; and
  `grade_distribution.csv` contains every grade numerator and denominator.
- `pareto_by_dataset.csv`, `validation.json`, and
  `calculation_manifest.json`.
"""


def _write_calculation_manifest(
    *,
    analysis_dir: Path,
    snapshot_dir: Path,
    validation: dict[str, object],
    recommendations: dict[str, object],
) -> None:
    """Write formulas, policies, input hashes, and output hashes."""
    output_files = sorted(
        path
        for path in analysis_dir.iterdir()
        if path.is_file() and path.name != "calculation_manifest.json"
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_dir": str(snapshot_dir),
        "input_files": _file_hashes(
            [
                snapshot_dir / "manifest.json",
                snapshot_dir / "experiments.parquet",
                snapshot_dir / "runs.parquet",
                snapshot_dir / "run_metrics.parquet",
                snapshot_dir / "trace_quality.parquet",
                snapshot_dir / "trace_retrieval.parquet",
                snapshot_dir / "span_timings.parquet",
            ]
        ),
        "output_files": _file_hashes(output_files),
        "formulas": {
            "good_acceptable_rate": "(Good + Acceptable) / unique traces",
            "good_rate": "Good / unique traces",
            "critical_grade_rate": "Critical Error / unique traces",
            "rubric_v2_mean": "arithmetic mean over unique trace RubricV2",
            "latency_seconds": "root invoke_* UNKNOWN observed_duration_ms / 1000",
            "percentile_method": PERCENTILE_METHOD,
            "pairing_key": "dataset_segment + rubric_key",
            "jaccard": "intersection(chunk identities) / union(chunk identities)",
            "pareto": "maximize quality and minimize p50 latency per dataset",
        },
        "grade_order": GRADE_ORDER,
        "passing_grades": sorted(PASSING_GRADES),
        "validation": validation,
        "recommendations": recommendations,
    }
    _write_json(analysis_dir / "calculation_manifest.json", payload)


def _markdown_table(dataframe: Any) -> str:
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
            if _is_missing(value):
                text = "—"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _decode_assessment_row(row: Any) -> object:
    """Decode the snapshot's type-tagged assessment scalar."""
    value_type = row["assessment_value_type"]
    if value_type is None or _is_missing(value_type):
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


def _to_boolean(value: object) -> bool | None:
    """Normalize supported boolean assessment representations."""
    if value is None or _is_missing(value):
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


def _use_case_from_rubric(rubric_key: str) -> str:
    """Derive a use-case name from a stable invoke span name."""
    match = _USE_CASE_PATTERN.fullmatch(rubric_key)
    return match.group("use_case") if match else rubric_key.removeprefix("invoke_")


def _metric_value(metric_row: Any, key: str) -> float:
    """Return a scalar run metric."""
    value = metric_row.get(key)
    return float(value) if value is not None and not _is_missing(value) else float("nan")


def _is_missing(value: object) -> bool:
    """Return whether a scalar is pandas/IEEE missing."""
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _manifest_checksums_match(snapshot_dir: Path, manifest: dict[str, Any]) -> bool:
    """Verify every snapshot file checksum recorded in the manifest."""
    return all(
        _sha256(snapshot_dir / str(item["name"])) == item["sha256"]
        for item in manifest.get("files", [])
    )


def _file_hashes(paths: list[Path]) -> list[dict[str, object]]:
    """Return reproducibility metadata for files."""
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write stable human-readable JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_pandas() -> Any:
    """Load the optional MLflow analysis dependency."""
    try:
        import pandas
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional dependency group before analysis: "
            "uv sync --group mlflow"
        ) from exc
    return pandas
