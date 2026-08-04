"""Export privacy-reduced MLflow snapshots and analyze them offline."""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_EXPERIMENT_FOLDER = "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/"
SCHEMA_VERSION = 1
DEFAULT_TRACE_FETCH_CONCURRENCY = 10
logger = logging.getLogger(__name__)

_EXPERIMENT_COLUMNS = ["experiment_id", "experiment_name", "lifecycle_stage"]
_RUN_COLUMNS = [
    "experiment_id",
    "experiment_name",
    "run_id",
    "status",
    "start_time_ms",
    "start_time_utc",
    "start_hour_utc",
    "start_weekday_utc",
    "dataset",
    "model_version",
    "simple_retrieval_mode",
    "simple_merge_policy",
    "requested_retrieval_calls",
    "simple_per_call_fetch_count",
    "simple_global_context_chunk_count",
    "reasoning_effort",
    "invocation_concurrency",
]
_METRIC_COLUMNS = ["experiment_id", "run_id", "metric_key", "metric_value"]
_TRACE_QUALITY_COLUMNS = [
    "experiment_id",
    "run_id",
    "trace_id",
    "trace_status",
    "execution_time_ms",
    "use_case",
    "dataset_id",
    "evalset_variant",
    "row_id",
    "assessment_name",
    "assessment_value",
    "ordinal_grade",
    "detected_error_modes",
]
_RETRIEVAL_COLUMNS = [
    "experiment_id",
    "run_id",
    "trace_id",
    "span_name",
    "simple_operation",
    "tool_ordinal",
    "retrieval_mode",
    "per_call_fetch_count",
    "returned_chunk_count",
    "requested_retrieval_calls",
    "actual_retrieval_calls",
    "generic_retrieval_calls",
    "metadata_filter_retrieval_calls",
    "selected_chunk_count",
    "context_size_chars",
    "merge_policy",
    "global_context_chunk_count",
    "ranked_chunk_ids",
]
_SPAN_TIMING_COLUMNS = [
    "experiment_id",
    "run_id",
    "trace_id",
    "span_name",
    "span_type",
    "simple_operation",
    "observed_duration_ms",
    "llm_successful_attempt_latency_ms",
    "llm_successful_attempt_count",
    "es_success_duration_ms",
    "es_success_attempt_count",
    "query_embedding_duration_ms",
]


@dataclass(frozen=True, slots=True)
class SnapshotSelector:
    """Select MLflow experiments by prefix or folder path.

    Parameters
    ----------
    experiment_prefix
        Literal MLflow experiment-name prefix. Mutually exclusive with
        ``experiment_folder``.
    experiment_folder
        MLflow folder path. Selection is recursive unless ``direct_children``
        is true.
    direct_children
        Limit folder matching to immediate child experiments.
    """

    experiment_prefix: str | None = None
    experiment_folder: str | None = None
    direct_children: bool = False

    def __post_init__(self) -> None:
        """Validate mutually exclusive selection arguments."""
        if self.experiment_prefix and self.experiment_folder:
            raise ValueError("experiment_prefix and experiment_folder are mutually exclusive.")
        if self.direct_children and not self.experiment_folder:
            raise ValueError("direct_children requires experiment_folder.")

    @property
    def resolved_folder(self) -> str | None:
        """Return the explicitly selected or default recursive folder."""
        if self.experiment_prefix:
            return None
        return self.experiment_folder or DEFAULT_EXPERIMENT_FOLDER

    def matches(self, experiment_name: str) -> bool:
        """Return whether an experiment name satisfies this selector."""
        if self.experiment_prefix:
            return experiment_name.startswith(self.experiment_prefix)

        folder = self.resolved_folder
        assert folder is not None
        normalized_folder = f"{folder.rstrip('/')}/"
        if not experiment_name.startswith(normalized_folder):
            return False
        if not self.direct_children:
            return True
        remainder = experiment_name.removeprefix(normalized_folder)
        return bool(remainder) and "/" not in remainder


def export_snapshot(
    *,
    profile: str,
    selector: SnapshotSelector,
    output_dir: Path,
    all_runs: bool,
    trace_fetch_concurrency: int = DEFAULT_TRACE_FETCH_CONCURRENCY,
) -> Path:
    """Export a sanitized MLflow snapshot using an explicit Databricks profile.

    The export uses MLflow read APIs only. It does not persist raw user prompts,
    search queries, model answers, document/chunk content, XML, or scorer
    rationales.

    Parameters
    ----------
    profile
        Explicit Databricks CLI profile name.
    selector
        Experiment selector.
    output_dir
        Directory receiving the immutable local snapshot.
    all_runs
        If true, include all finished runs; otherwise include only the latest
        finished run from each selected experiment.
    trace_fetch_concurrency
        Maximum simultaneous full-trace downloads. Must be between 1 and 10
        so the exporter does not exceed MLflow's default connection-pool size.

    Returns
    -------
    Path
        Resolved snapshot directory.
    """
    if not 1 <= trace_fetch_concurrency <= DEFAULT_TRACE_FETCH_CONCURRENCY:
        raise ValueError(
            "trace_fetch_concurrency must be between 1 and "
            f"{DEFAULT_TRACE_FETCH_CONCURRENCY}."
        )

    client, pandas = _load_mlflow_dependencies(profile)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering MLflow experiments.")
    # MLflow entity types differ across client versions, so third-party payloads
    # are intentionally handled as Any at this I/O boundary.
    experiments: list[Any] = [
        experiment
        for experiment in client.search_experiments(max_results=10_000)
        if selector.matches(experiment.name)
    ]
    logger.info("Selected %d MLflow experiments.", len(experiments))
    experiment_rows = [
        {
            "experiment_id": str(experiment.experiment_id),
            "experiment_name": experiment.name,
            "lifecycle_stage": getattr(experiment, "lifecycle_stage", None),
        }
        for experiment in experiments
    ]

    run_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    trace_quality_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    span_timing_rows: list[dict[str, object]] = []
    selected_run_ids: list[str] = []

    for experiment_ordinal, experiment in enumerate(experiments, start=1):
        logger.info(
            "Exporting experiment %d/%d: %s",
            experiment_ordinal,
            len(experiments),
            experiment.name,
        )
        runs = [
            run
            for run in client.search_runs([experiment.experiment_id], max_results=10_000)
            if str(run.info.status) == "FINISHED"
        ]
        if not all_runs and runs:
            runs = [max(runs, key=lambda run: run.info.start_time or 0)]

        logger.info("Selected %d finished run(s).", len(runs))
        for run_ordinal, run in enumerate(runs, start=1):
            logger.info(
                "Exporting run %d/%d with trace downloads limited to %d concurrent requests.",
                run_ordinal,
                len(runs),
                trace_fetch_concurrency,
            )
            selected_run_ids.append(run.info.run_id)
            run_rows.append(_sanitize_run(experiment, run))
            metric_rows.extend(_sanitize_metrics(experiment, run))
            traces = _fetch_traces_bounded(
                client=client,
                experiment_id=str(experiment.experiment_id),
                run_id=run.info.run_id,
                trace_fetch_concurrency=trace_fetch_concurrency,
            )
            for trace in traces:
                quality, retrieval, timings = _sanitize_trace(
                    experiment_id=str(experiment.experiment_id),
                    run_id=run.info.run_id,
                    trace=trace,
                )
                trace_quality_rows.extend(quality)
                retrieval_rows.extend(retrieval)
                span_timing_rows.extend(timings)

    logger.info("Writing sanitized snapshot tables.")
    _write_parquet(pandas, output_dir / "experiments.parquet", experiment_rows, _EXPERIMENT_COLUMNS)
    _write_parquet(pandas, output_dir / "runs.parquet", run_rows, _RUN_COLUMNS)
    _write_parquet(pandas, output_dir / "run_metrics.parquet", metric_rows, _METRIC_COLUMNS)
    _write_parquet(
        pandas,
        output_dir / "trace_quality.parquet",
        trace_quality_rows,
        _TRACE_QUALITY_COLUMNS,
    )
    _write_parquet(
        pandas,
        output_dir / "trace_retrieval.parquet",
        retrieval_rows,
        _RETRIEVAL_COLUMNS,
    )
    _write_parquet(
        pandas,
        output_dir / "span_timings.parquet",
        span_timing_rows,
        _SPAN_TIMING_COLUMNS,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "tracking_uri": f"databricks://{profile}",
        "profile": profile,
        "selector": asdict(selector),
        "all_runs": all_runs,
        "trace_fetch_concurrency": trace_fetch_concurrency,
        "experiment_count": len(experiment_rows),
        "run_count": len(selected_run_ids),
        "run_ids": selected_run_ids,
        "privacy_exclusions": [
            "raw_messages",
            "generated_answers",
            "search_query_text",
            "document_chunk_content",
            "retrieval_xml",
            "scorer_rationales",
        ],
        "files": _file_manifest(output_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "Snapshot complete: %d experiments and %d runs.",
        len(experiment_rows),
        len(selected_run_ids),
    )
    return output_dir


def analyze_snapshot(*, snapshot_dir: Path, selector: SnapshotSelector) -> Path:
    """Analyze a local sanitized snapshot without contacting MLflow.

    Parameters
    ----------
    snapshot_dir
        Existing directory produced by :func:`export_snapshot`.
    selector
        Local experiment selector applied to the saved experiment names.

    Returns
    -------
    Path
        Directory containing compact CSV, JSON, and Markdown analysis outputs.
    """
    pandas = _load_pandas()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    _validate_snapshot(snapshot_dir)
    analysis_dir = snapshot_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    experiments = pandas.read_parquet(snapshot_dir / "experiments.parquet")
    selected_experiments = experiments[
        experiments["experiment_name"].map(selector.matches)
    ]
    selected_ids = set(selected_experiments["experiment_id"].astype(str))
    runs = pandas.read_parquet(snapshot_dir / "runs.parquet")
    metrics = pandas.read_parquet(snapshot_dir / "run_metrics.parquet")
    quality = pandas.read_parquet(snapshot_dir / "trace_quality.parquet")
    retrieval = pandas.read_parquet(snapshot_dir / "trace_retrieval.parquet")

    runs = runs[runs["experiment_id"].astype(str).isin(selected_ids)].copy()
    metrics = metrics[metrics["experiment_id"].astype(str).isin(selected_ids)].copy()
    quality = quality[quality["experiment_id"].astype(str).isin(selected_ids)].copy()
    retrieval = retrieval[retrieval["experiment_id"].astype(str).isin(selected_ids)].copy()

    run_summary = _build_run_summary(pandas, runs, metrics, quality, retrieval)
    pareto = _build_pareto_candidates(pandas, run_summary)
    grade_distribution = _build_grade_distribution(pandas, quality)

    run_summary.to_csv(analysis_dir / "run_summary.csv", index=False)
    pareto.to_csv(analysis_dir / "pareto_candidates.csv", index=False)
    grade_distribution.to_csv(analysis_dir / "grade_distribution.csv", index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_dir": str(snapshot_dir),
        "selected_experiment_count": len(selected_experiments),
        "selected_run_count": len(runs),
        "pareto_candidate_count": len(pareto),
        "time_of_day_caveat": (
            "Latency is retained with UTC start times. Do not attribute small "
            "latency differences to arm parameters without considering time of day."
        ),
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (analysis_dir / "summary.md").write_text(
        _render_summary_markdown(summary, pareto),
        encoding="utf-8",
    )
    return analysis_dir


def _load_mlflow_dependencies(profile: str) -> tuple[Any, Any]:
    """Load optional export dependencies and configure profile-aware MLflow."""
    # `MlflowClient(tracking_uri=...)` selects the profile for REST calls, but
    # MLflow's Databricks SDK helpers also consult this environment variable.
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    try:
        import mlflow
        import pandas
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional dependency group before exporting: "
            "uv sync --group mlflow"
        ) from exc

    tracking_uri = f"databricks://{profile}"
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri), pandas


def _fetch_traces_bounded(
    *,
    client: Any,
    experiment_id: str,
    run_id: str,
    trace_fetch_concurrency: int,
) -> list[Any]:
    """Fetch all traces in pages without exceeding the default pool size.

    The metadata page is requested without spans, then complete traces are
    fetched with a bounded worker pool. This avoids MLflow's unbounded
    artifact fan-out and uses ``locations`` rather than deprecated
    ``experiment_ids``.
    """
    traces: list[Any] = []
    page_token: str | None = None
    page_number = 0
    while True:
        page_number += 1
        page = client.search_traces(
            locations=[experiment_id],
            run_id=run_id,
            include_spans=False,
            max_results=trace_fetch_concurrency,
            page_token=page_token,
        )
        trace_ids = [trace.info.trace_id for trace in page]
        logger.info(
            "Fetching trace page %d: %d traces.",
            page_number,
            len(trace_ids),
        )
        executor = ThreadPoolExecutor(max_workers=trace_fetch_concurrency)
        try:
            traces.extend(
                executor.map(
                    lambda trace_id: client.get_trace(trace_id, display=False),
                    trace_ids,
                )
            )
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Trace download interrupted; cancelling queued requests.")
            raise
        else:
            executor.shutdown(wait=True)
        page_token = page.token
        if not page_token:
            return traces


def _load_pandas() -> Any:
    """Load the optional local analysis dependency."""
    try:
        import pandas
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional dependency group before analyzing: "
            "uv sync --group mlflow"
        ) from exc
    return pandas


def _sanitize_run(experiment: Any, run: Any) -> dict[str, object]:
    """Return approved non-sensitive run metadata and parameters."""
    params = run.data.params
    start_time_ms = run.info.start_time
    start_time = (
        datetime.fromtimestamp(start_time_ms / 1000, UTC) if start_time_ms else None
    )
    return {
        "experiment_id": str(experiment.experiment_id),
        "experiment_name": experiment.name,
        "run_id": run.info.run_id,
        "status": str(run.info.status),
        "start_time_ms": start_time_ms,
        "start_time_utc": start_time.isoformat() if start_time else None,
        "start_hour_utc": start_time.hour if start_time else None,
        "start_weekday_utc": start_time.weekday() if start_time else None,
        "dataset": params.get("dataset"),
        "model_version": params.get("model_version"),
        "simple_retrieval_mode": params.get("simple_retrieval_mode"),
        "simple_merge_policy": params.get("simple_merge_policy"),
        "requested_retrieval_calls": _to_int(params.get("requested_retrieval_calls")),
        "simple_per_call_fetch_count": _to_int(
            params.get("simple_per_call_fetch_count")
        ),
        "simple_global_context_chunk_count": _to_int(
            params.get("simple_global_context_chunk_count")
        ),
        "reasoning_effort": params.get("reasoning_effort"),
        "invocation_concurrency": _to_int(params.get("invocation_concurrency")),
    }


def _sanitize_metrics(experiment: Any, run: Any) -> list[dict[str, object]]:
    """Return metric key/value rows without model content."""
    return [
        {
            "experiment_id": str(experiment.experiment_id),
            "run_id": run.info.run_id,
            "metric_key": key,
            "metric_value": value,
        }
        for key, value in run.data.metrics.items()
    ]


def _sanitize_trace(
    *,
    experiment_id: str,
    run_id: str,
    trace: Any,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return privacy-reduced assessment, retrieval, and timing rows for a trace."""
    info = trace.info
    trace_id = str(info.trace_id)
    trace_attributes = _as_mapping(getattr(info, "attributes", {}))
    quality_rows: list[dict[str, object]] = []
    ordinal_grade: str | None = None
    detected_error_modes = "[]"

    for assessment in getattr(info, "assessments", []) or []:
        name = str(getattr(assessment, "name", ""))
        value = getattr(assessment, "value", None)
        metadata = _as_mapping(getattr(assessment, "metadata", {}))
        if name == "ordinal_grade":
            ordinal_grade = str(value) if value is not None else None
        if name == "RubricV2":
            modes = metadata.get("detected_error_modes", [])
            detected_error_modes = json.dumps(
                [
                    {
                        "name": item.get("name"),
                        "detected": item.get("detected"),
                    }
                    for item in modes
                    if isinstance(item, dict)
                ],
                sort_keys=True,
            )
        quality_rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "trace_id": trace_id,
                "trace_status": str(getattr(info, "status", "")),
                "execution_time_ms": getattr(info, "execution_duration", None),
                "use_case": trace_attributes.get("use_case"),
                "dataset_id": trace_attributes.get("dataset_id"),
                "evalset_variant": trace_attributes.get("evalset_variant"),
                "row_id": trace_attributes.get("row_id"),
                "assessment_name": name,
                "assessment_value": _assessment_value(value),
                "ordinal_grade": None,
                "detected_error_modes": "[]",
            }
        )

    for row in quality_rows:
        row["ordinal_grade"] = ordinal_grade
        row["detected_error_modes"] = detected_error_modes

    retrieval_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for span in getattr(trace.data, "spans", []) or []:
        attributes = _as_mapping(getattr(span, "attributes", {}))
        operation = attributes.get("simple.operation")
        timing_rows.append(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "trace_id": trace_id,
                "span_name": getattr(span, "name", None),
                "span_type": getattr(span, "span_type", None),
                "simple_operation": operation,
                "observed_duration_ms": _span_duration_ms(span),
                "llm_successful_attempt_latency_ms": attributes.get(
                    "llm.successful_attempt_latency_ms"
                ),
                "llm_successful_attempt_count": attributes.get(
                    "llm.successful_attempt_count"
                ),
                "es_success_duration_ms": attributes.get("simple.es_success_duration_ms"),
                "es_success_attempt_count": attributes.get(
                    "simple.es_success_attempt_count"
                ),
                "query_embedding_duration_ms": attributes.get(
                    "simple.query_embedding_duration_ms"
                ),
            }
        )
        if operation in {
            "simple.retrieval_generic",
            "simple.retrieval_metadata_filter",
        } or "simple.actual_retrieval_calls" in attributes:
            retrieval_rows.append(
                {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "span_name": getattr(span, "name", None),
                    "simple_operation": operation,
                    "tool_ordinal": attributes.get("simple.tool_ordinal"),
                    "retrieval_mode": attributes.get("simple.retrieval_mode"),
                    "per_call_fetch_count": attributes.get(
                        "simple.per_call_fetch_count"
                    ),
                    "returned_chunk_count": attributes.get(
                        "simple.returned_chunk_count"
                    ),
                    "requested_retrieval_calls": attributes.get(
                        "simple.requested_retrieval_calls"
                    ),
                    "actual_retrieval_calls": attributes.get(
                        "simple.actual_retrieval_calls"
                    ),
                    "generic_retrieval_calls": attributes.get(
                        "simple.generic_retrieval_calls"
                    ),
                    "metadata_filter_retrieval_calls": attributes.get(
                        "simple.metadata_filter_retrieval_calls"
                    ),
                    "selected_chunk_count": attributes.get(
                        "simple.selected_chunk_count"
                    ),
                    "context_size_chars": attributes.get("simple.context_size_chars"),
                    "merge_policy": attributes.get("simple.merge_policy"),
                    "global_context_chunk_count": attributes.get(
                        "simple.global_context_chunk_count"
                    ),
                    "ranked_chunk_ids": _ranked_chunk_ids(span),
                }
            )
    return quality_rows, retrieval_rows, timing_rows


def _write_parquet(
    pandas: Any,
    path: Path,
    rows: list[dict[str, object]],
    columns: list[str],
) -> None:
    """Write a normalized table with stable empty-table columns."""
    dataframe = pandas.DataFrame(rows, columns=columns)
    dataframe.to_parquet(path, index=False)


def _file_manifest(snapshot_dir: Path) -> list[dict[str, object]]:
    """Return names, sizes, and checksums for current snapshot files."""
    files: list[dict[str, object]] = []
    for path in sorted(snapshot_dir.glob("*.parquet")):
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return files


def _validate_snapshot(snapshot_dir: Path) -> None:
    """Raise when a directory is not a completed sanitized snapshot."""
    required_files = {
        "manifest.json",
        "experiments.parquet",
        "runs.parquet",
        "run_metrics.parquet",
        "trace_quality.parquet",
        "trace_retrieval.parquet",
        "span_timings.parquet",
    }
    missing = sorted(
        filename for filename in required_files if not (snapshot_dir / filename).exists()
    )
    if missing:
        raise ValueError(f"Snapshot is missing required files: {', '.join(missing)}")


def _build_run_summary(
    pandas: Any,
    runs: Any,
    metrics: Any,
    quality: Any,
    retrieval: Any,
) -> Any:
    """Build per-run quality, latency, and context summary rows."""
    metric_pivot = metrics.pivot_table(
        index="run_id",
        columns="metric_key",
        values="metric_value",
        aggfunc="last",
    ).reset_index()
    grades = quality[quality["assessment_name"] == "ordinal_grade"].copy()
    grade_counts = (
        grades.groupby(["run_id", "ordinal_grade"]).size().unstack(fill_value=0).reset_index()
    )
    retrieval_summary = (
        retrieval.groupby("run_id")
        .agg(
            actual_retrieval_calls=("actual_retrieval_calls", "max"),
            selected_chunk_count=("selected_chunk_count", "max"),
            context_size_chars=("context_size_chars", "max"),
        )
        .reset_index()
    )
    return (
        runs.merge(metric_pivot, on="run_id", how="left")
        .merge(grade_counts, on="run_id", how="left")
        .merge(retrieval_summary, on="run_id", how="left")
    )


def _build_pareto_candidates(pandas: Any, run_summary: Any) -> Any:
    """Return non-dominated runs by dataset pass rate and p50 latency."""
    quality_column = (
        "total_pass_rate"
        if "total_pass_rate" in run_summary.columns
        else "RubricV2_avg"
    )
    latency_column = "execution_time_p50_s"
    if quality_column not in run_summary or latency_column not in run_summary:
        return pandas.DataFrame(columns=list(run_summary.columns) + ["pareto_rank"])

    candidates: list[Any] = []
    for _, group in run_summary.groupby("dataset", dropna=False):
        eligible = group.dropna(subset=[quality_column, latency_column]).copy()
        for index, row in eligible.iterrows():
            dominated = (
                (eligible[quality_column] >= row[quality_column])
                & (eligible[latency_column] <= row[latency_column])
                & (
                    (eligible[quality_column] > row[quality_column])
                    | (eligible[latency_column] < row[latency_column])
                )
            ).any()
            if not dominated:
                candidate = row.copy()
                candidate["pareto_rank"] = 1
                candidates.append(candidate)
    if not candidates:
        return pandas.DataFrame(columns=list(run_summary.columns) + ["pareto_rank"])
    return pandas.DataFrame(candidates)


def _build_grade_distribution(pandas: Any, quality: Any) -> Any:
    """Aggregate ordinal-grade counts by dataset and use case."""
    grades = quality[quality["assessment_name"] == "ordinal_grade"].copy()
    return (
        grades.groupby(["dataset_id", "use_case", "ordinal_grade"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )


def _render_summary_markdown(summary: dict[str, object], pareto: Any) -> str:
    """Render a compact local analysis summary without sensitive content."""
    lines = [
        "# Local MLflow Snapshot Analysis",
        "",
        f"- Selected experiments: {summary['selected_experiment_count']}",
        f"- Selected runs: {summary['selected_run_count']}",
        f"- Pareto candidates: {summary['pareto_candidate_count']}",
        f"- Latency caveat: {summary['time_of_day_caveat']}",
        "",
        "## Pareto candidates",
        "",
    ]
    if pareto.empty:
        lines.append("No comparable runs with both quality and p50 latency metrics were found.")
    else:
        lines.append(pareto.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def _as_mapping(value: object) -> dict[str, Any]:
    """Return a third-party metadata value as a mapping when possible."""
    return value if isinstance(value, dict) else {}


def _assessment_value(value: object) -> str | float | int | bool | None:
    """Retain primitive assessment values while excluding rich rationale data."""
    if isinstance(value, str | float | int | bool) or value is None:
        return value
    return str(value)


def _span_duration_ms(span: Any) -> float | None:
    """Return native MLflow span wall duration in milliseconds."""
    start = getattr(span, "start_time_ns", None)
    end = getattr(span, "end_time_ns", None)
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return (end - start) / 1_000_000


def _ranked_chunk_ids(span: Any) -> str:
    """Extract rank identifiers only from a retrieval span output."""
    outputs = _as_mapping(getattr(span, "outputs", {}))
    ranked_chunks = outputs.get("ranked_chunks", [])
    if not isinstance(ranked_chunks, list):
        return "[]"
    reduced = [
        {
            "document_id": item.get("document_id"),
            "chunk_id": item.get("chunk_id"),
            "rank": item.get("rank"),
        }
        for item in ranked_chunks
        if isinstance(item, dict)
    ]
    return json.dumps(reduced, sort_keys=True)


def _to_int(value: object) -> int | None:
    """Coerce a numeric run parameter to an integer when possible."""
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None
