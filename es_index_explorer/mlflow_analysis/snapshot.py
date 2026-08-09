"""Export privacy-reduced MLflow snapshots and analyze them offline.

Exports are resumable. Each export session stores atomic per-run shards under
``checkpoint-{unix_epoch}/``. Each shard is a trusted-local Python-native payload
so interruption/resume never changes the type or value of a sanitized row. Final
root artifacts are published atomically after every planned run commits.
"""

import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

DEFAULT_EXPERIMENT_FOLDER = "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/"
SCHEMA_VERSION = 2
CHECKPOINT_KIND = "mlflow_snapshot_export"
DEFAULT_TRACE_FETCH_CONCURRENCY = 10
logger = logging.getLogger(__name__)

_PICKLE_PROTOCOL = 5
_RUN_PAYLOAD_FILE = "run_payload.pickle"
_MAX_TRACE_REQUEST_ATTEMPTS = 5
_MAX_AUTH_RECOVERY_CYCLES = 2
_RETRY_BASE_DELAY_SECONDS = 2.0
_RETRY_MAX_DELAY_SECONDS = 30.0


class _AuthenticationRecoveryRequired(RuntimeError):
    """Signal that retries require interactive Databricks authentication."""


class _TraceRequestRetryExhausted(RuntimeError):
    """Signal that bounded retries could not complete an MLflow request."""

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
    "simple_required_tools",
    "configured_retrieval_call_count",
    "simple_merge_policy",
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
    "assessment_value_type",
    "assessment_value_string",
    "assessment_value_bool",
    "assessment_value_int",
    "assessment_value_float",
    "ordinal_grade",
    "detected_error_modes",
]
_TRACE_FAILURE_COLUMNS = [
    "experiment_id",
    "run_id",
    "trace_id",
    "trace_status",
    "failure_type",
    "failure_reason",
    "required_tool_multiset",
    "actual_tool_multiset",
]
_RETRIEVAL_COLUMNS = [
    "experiment_id",
    "run_id",
    "trace_id",
    "span_name",
    "simple_operation",
    "tool_name",
    "retrieval_mode",
    "per_call_fetch_count",
    "returned_chunk_count",
    "required_tool_multiset",
    "actual_tool_multiset",
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
_PRIVACY_EXCLUSIONS = [
    "raw_messages",
    "generated_answers",
    "search_query_text",
    "document_chunk_content",
    "retrieval_xml",
    "scorer_rationales",
]
_SNAPSHOT_PARQUET_FILES = (
    "experiments.parquet",
    "runs.parquet",
    "run_metrics.parquet",
    "trace_quality.parquet",
    "trace_failures.parquet",
    "trace_retrieval.parquet",
    "span_timings.parquet",
)


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
    resume: bool = True,
    fresh: bool = False,
    checkpoint_epoch: int | None = None,
) -> Path:
    """Export a sanitized MLflow snapshot using an explicit Databricks profile.

    The export uses MLflow read APIs only. It does not persist raw user prompts,
    search queries, model answers, document/chunk content, XML, or scorer
    rationales.

    Progress is checkpointed per experiment run under
    ``output_dir/checkpoint-{unix_epoch}/``. Each checkpoint is a trusted-local
    Python-native payload. Resume requires the same ``output_dir`` and matching
    CLI identity (profile, selector, all_runs, concurrency). Changed remote
    runs invalidate only that run's shard.

    Parameters
    ----------
    profile
        Explicit Databricks CLI profile name.
    selector
        Experiment selector.
    output_dir
        Directory receiving the immutable local snapshot and checkpoints.
    all_runs
        If true, include all finished runs; otherwise include only the latest
        finished run from each selected experiment.
    trace_fetch_concurrency
        Maximum simultaneous full-trace downloads. Must be between 1 and 10
        so the exporter does not exceed MLflow's default connection-pool size.
    resume
        When true (default), resume the newest matching incomplete checkpoint.
    fresh
        When true, ignore matching checkpoints and start a new session.
    checkpoint_epoch
        Resume a specific ``checkpoint-{epoch}`` directory when present.

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
    if fresh and checkpoint_epoch is not None:
        raise ValueError("fresh and checkpoint_epoch are mutually exclusive.")

    client, pandas = _load_mlflow_dependencies(profile)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_identity = _cli_identity(
        profile=profile,
        selector=selector,
        all_runs=all_runs,
        trace_fetch_concurrency=trace_fetch_concurrency,
    )
    checkpoint_dir = _resolve_checkpoint_session(
        output_dir=output_dir,
        cli_identity=cli_identity,
        resume=resume,
        fresh=fresh,
        checkpoint_epoch=checkpoint_epoch,
    )
    checkpoint = _read_checkpoint(checkpoint_dir)
    _cleanup_incomplete_tmp_dirs(checkpoint_dir)

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
    _write_parquet(
        pandas,
        checkpoint_dir / "experiments.parquet",
        experiment_rows,
        _EXPERIMENT_COLUMNS,
    )

    planned_runs = _discover_planned_runs(
        client=client,
        experiments=experiments,
        all_runs=all_runs,
    )
    planned_run_ids = [run_id for _, _, run_id in planned_runs]
    completed_run_ids = set(checkpoint.get("discovery", {}).get("completed_run_ids", []))
    failed_run_ids = list(checkpoint.get("discovery", {}).get("failed_run_ids", []))

    # Drop completed shards for runs no longer in the remote selection.
    remote_run_ids = set(planned_run_ids)
    for stale_run_id in sorted(completed_run_ids - remote_run_ids):
        logger.info("Omitting remote-deleted run shard %s.", stale_run_id)
        _delete_run_shard(checkpoint_dir, stale_run_id)
        completed_run_ids.discard(stale_run_id)

    checkpoint["discovery"] = {
        "experiment_ids": [str(experiment.experiment_id) for experiment in experiments],
        "planned_run_ids": planned_run_ids,
        "completed_run_ids": sorted(completed_run_ids),
        "failed_run_ids": failed_run_ids,
        "invalidated_run_ids": list(
            checkpoint.get("discovery", {}).get("invalidated_run_ids", [])
        ),
    }
    checkpoint["status"] = "in_progress"
    _write_checkpoint(checkpoint_dir, checkpoint)

    invalidated_run_ids: list[str] = list(
        checkpoint["discovery"].get("invalidated_run_ids", [])
    )
    logger.info(
        "Checkpoint %s: %d planned runs, %d already committed.",
        checkpoint_dir.name,
        len(planned_run_ids),
        len(completed_run_ids),
    )

    try:
        for run_ordinal, (experiment, run, run_id) in enumerate(planned_runs, start=1):
            logger.info(
                "Processing run %d/%d (%s).",
                run_ordinal,
                len(planned_runs),
                run_id,
            )
            remote_fingerprint = _compute_remote_fingerprint(
                client=client,
                experiment=experiment,
                run=run,
                trace_fetch_concurrency=trace_fetch_concurrency,
            )
            if run_id in completed_run_ids and _run_shard_is_committed(checkpoint_dir, run_id):
                local_fingerprint = _read_run_fingerprint(checkpoint_dir, run_id)
                if local_fingerprint is not None and _fingerprints_match(
                    local_fingerprint, remote_fingerprint
                ):
                    logger.info("Reusing committed run shard %s.", run_id)
                    continue
                logger.info(
                    "Invalidating changed run shard %s (metadata fingerprint mismatch).",
                    run_id,
                )
                _delete_run_shard(checkpoint_dir, run_id)
                completed_run_ids.discard(run_id)
                if run_id not in invalidated_run_ids:
                    invalidated_run_ids.append(run_id)
                checkpoint["discovery"]["completed_run_ids"] = sorted(completed_run_ids)
                checkpoint["discovery"]["invalidated_run_ids"] = invalidated_run_ids
                _write_checkpoint(checkpoint_dir, checkpoint)

            logger.info(
                "Downloading run %s with trace downloads limited to %d concurrent requests.",
                run_id,
                trace_fetch_concurrency,
            )
            client = _export_single_run(
                client=client,
                profile=profile,
                checkpoint_dir=checkpoint_dir,
                experiment=experiment,
                run=run,
                remote_fingerprint=remote_fingerprint,
                trace_fetch_concurrency=trace_fetch_concurrency,
            )
            completed_run_ids.add(run_id)
            if run_id in failed_run_ids:
                failed_run_ids = [item for item in failed_run_ids if item != run_id]
            checkpoint["discovery"]["completed_run_ids"] = sorted(completed_run_ids)
            checkpoint["discovery"]["failed_run_ids"] = failed_run_ids
            checkpoint["discovery"]["invalidated_run_ids"] = invalidated_run_ids
            _write_checkpoint(checkpoint_dir, checkpoint)
            logger.info(
                "Committed run shard %s (%d/%d complete).",
                run_id,
                len(completed_run_ids),
                len(planned_run_ids),
            )
    except KeyboardInterrupt:
        checkpoint["status"] = "interrupted"
        checkpoint["discovery"]["completed_run_ids"] = sorted(completed_run_ids)
        checkpoint["discovery"]["failed_run_ids"] = failed_run_ids
        checkpoint["discovery"]["invalidated_run_ids"] = invalidated_run_ids
        _write_checkpoint(checkpoint_dir, checkpoint)
        logger.info(
            "Interrupted; checkpoint retained at %s with %d/%d runs committed.",
            checkpoint_dir,
            len(completed_run_ids),
            len(planned_run_ids),
        )
        raise

    if completed_run_ids != set(planned_run_ids):
        missing = sorted(set(planned_run_ids) - completed_run_ids)
        raise RuntimeError(
            "Export incomplete; missing committed run shards: " + ", ".join(missing)
        )

    logger.info(
        "Aggregating %d committed run shards into final snapshot.",
        len(planned_run_ids),
    )
    checkpoint["status"] = "aggregating"
    _write_checkpoint(checkpoint_dir, checkpoint)
    _aggregate_and_publish(
        pandas=pandas,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        cli_identity=cli_identity,
        planned_run_ids=planned_run_ids,
        experiment_rows=experiment_rows,
    )
    logger.info(
        "Snapshot complete: %d experiments and %d runs. Checkpoint cleaned up.",
        len(experiment_rows),
        len(planned_run_ids),
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
    failures = pandas.read_parquet(snapshot_dir / "trace_failures.parquet")
    retrieval = pandas.read_parquet(snapshot_dir / "trace_retrieval.parquet")

    runs = runs[runs["experiment_id"].astype(str).isin(selected_ids)].copy()
    metrics = metrics[metrics["experiment_id"].astype(str).isin(selected_ids)].copy()
    quality = quality[quality["experiment_id"].astype(str).isin(selected_ids)].copy()
    failures = failures[failures["experiment_id"].astype(str).isin(selected_ids)].copy()
    retrieval = retrieval[retrieval["experiment_id"].astype(str).isin(selected_ids)].copy()

    run_summary = _build_run_summary(pandas, runs, metrics, quality, retrieval)
    pareto = _build_pareto_candidates(pandas, run_summary)
    grade_distribution = _build_grade_distribution(pandas, quality)

    run_summary.to_csv(analysis_dir / "run_summary.csv", index=False)
    pareto.to_csv(analysis_dir / "pareto_candidates.csv", index=False)
    grade_distribution.to_csv(analysis_dir / "grade_distribution.csv", index=False)
    failures.to_csv(analysis_dir / "plan_validation_failures.csv", index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_dir": str(snapshot_dir),
        "selected_experiment_count": len(selected_experiments),
        "selected_run_count": len(runs),
        "plan_validation_failure_count": len(failures),
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


def _cli_identity(
    *,
    profile: str,
    selector: SnapshotSelector,
    all_runs: bool,
    trace_fetch_concurrency: int,
) -> dict[str, object]:
    """Return the reproducible export identity used for checkpoint matching."""
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "profile": profile,
        "tracking_uri": f"databricks://{profile}",
        "selector": asdict(selector),
        "all_runs": all_runs,
        "trace_fetch_concurrency": trace_fetch_concurrency,
    }


def _resolve_checkpoint_session(
    *,
    output_dir: Path,
    cli_identity: dict[str, object],
    resume: bool,
    fresh: bool,
    checkpoint_epoch: int | None,
) -> Path:
    """Create or resume a matching incomplete checkpoint session."""
    if fresh or not resume:
        checkpoint_dir = _create_checkpoint_session(output_dir, cli_identity)
        logger.info("Created fresh checkpoint session %s.", checkpoint_dir.name)
        return checkpoint_dir

    if checkpoint_epoch is not None:
        checkpoint_dir = output_dir / f"checkpoint-{checkpoint_epoch}"
        if not checkpoint_dir.is_dir():
            raise ValueError(f"Checkpoint epoch {checkpoint_epoch} not found under {output_dir}.")
        checkpoint = _read_checkpoint(checkpoint_dir)
        if not _cli_identity_matches(checkpoint.get("cli_identity", {}), cli_identity):
            raise ValueError(
                f"Checkpoint {checkpoint_dir.name} does not match the current export identity."
            )
        if checkpoint.get("status") == "completed":
            raise ValueError(
                f"Checkpoint {checkpoint_dir.name} is already completed; use --fresh."
            )
        logger.info("Resuming requested checkpoint session %s.", checkpoint_dir.name)
        return checkpoint_dir

    matching = _find_newest_matching_checkpoint(output_dir, cli_identity)
    if matching is None:
        checkpoint_dir = _create_checkpoint_session(output_dir, cli_identity)
        logger.info("Created checkpoint session %s.", checkpoint_dir.name)
        return checkpoint_dir

    logger.info("Resuming matching checkpoint session %s.", matching.name)
    return matching


def _create_checkpoint_session(output_dir: Path, cli_identity: dict[str, object]) -> Path:
    """Create a new checkpoint directory and process metadata file."""
    epoch = int(time.time())
    while True:
        checkpoint_dir = output_dir / f"checkpoint-{epoch}"
        if not checkpoint_dir.exists():
            break
        epoch += 1
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (checkpoint_dir / "runs").mkdir()
    now = datetime.now(UTC).isoformat()
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "created_at_utc": now,
        "updated_at_utc": now,
        "status": "in_progress",
        "cli_identity": cli_identity,
        "discovery": {
            "experiment_ids": [],
            "planned_run_ids": [],
            "completed_run_ids": [],
            "failed_run_ids": [],
            "invalidated_run_ids": [],
        },
        "integrity_boundary": (
            "Metadata fingerprints detect run config/metric changes, trace "
            "membership, assessment/status/timing changes, and exported trace "
            "metadata changes. In-place span payload changes without corresponding "
            "trace metadata changes are not guaranteed detectable without "
            "re-fetching full spans."
        ),
    }
    _write_checkpoint(checkpoint_dir, checkpoint)
    return checkpoint_dir


def _find_newest_matching_checkpoint(
    output_dir: Path,
    cli_identity: dict[str, object],
) -> Path | None:
    """Return the newest incomplete checkpoint matching the CLI identity."""
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        checkpoint_path = path / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        try:
            checkpoint = _read_checkpoint(path)
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("Skipping unreadable checkpoint %s.", path.name)
            continue
        if checkpoint.get("status") == "completed":
            continue
        if not _cli_identity_matches(checkpoint.get("cli_identity", {}), cli_identity):
            continue
        candidates.append((int(suffix), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _cli_identity_matches(stored: object, expected: dict[str, object]) -> bool:
    """Compare stored and expected CLI identities by canonical JSON."""
    if not isinstance(stored, dict):
        return False
    return _canonical_json(stored) == _canonical_json(expected)


def _read_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    """Load checkpoint process metadata."""
    path = checkpoint_dir / "checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint metadata at {path}.")
    return payload


def _write_checkpoint(checkpoint_dir: Path, checkpoint: dict[str, Any]) -> None:
    """Atomically rewrite checkpoint process metadata."""
    checkpoint = dict(checkpoint)
    checkpoint["updated_at_utc"] = datetime.now(UTC).isoformat()
    _atomic_write_text(
        checkpoint_dir / "checkpoint.json",
        json.dumps(checkpoint, indent=2, sort_keys=True),
    )


def _cleanup_incomplete_tmp_dirs(checkpoint_dir: Path) -> None:
    """Remove incomplete temporary run directories left by prior interruptions."""
    for path in checkpoint_dir.glob(".tmp-*"):
        if path.is_dir():
            logger.info("Removing incomplete temporary directory %s.", path.name)
            shutil.rmtree(path, ignore_errors=True)


def _discover_planned_runs(
    *,
    client: Any,
    experiments: list[Any],
    all_runs: bool,
) -> list[tuple[Any, Any, str]]:
    """Discover selected finished runs in stable experiment/run order."""
    planned: list[tuple[Any, Any, str]] = []
    for experiment in experiments:
        runs = [
            run
            for run in client.search_runs([experiment.experiment_id], max_results=10_000)
            if str(run.info.status) == "FINISHED"
        ]
        if not all_runs and runs:
            runs = [max(runs, key=lambda run: run.info.start_time or 0)]
        logger.info(
            "Selected %d finished run(s) for experiment %s.",
            len(runs),
            experiment.name,
        )
        for run in runs:
            planned.append((experiment, run, str(run.info.run_id)))
    return planned


def _export_single_run(
    *,
    client: Any,
    profile: str,
    checkpoint_dir: Path,
    experiment: Any,
    run: Any,
    remote_fingerprint: dict[str, Any],
    trace_fetch_concurrency: int,
) -> Any:
    """Download, sanitize, and atomically commit one experiment-run shard.

    The shard stores a Python-native payload so the in-memory data is not
    coerced by any intermediate Parquet/JSON conversion during interruption.
    """
    run_id = str(run.info.run_id)
    experiment_id = str(experiment.experiment_id)
    run_row = _sanitize_run(experiment, run)
    metric_rows = _sanitize_metrics(experiment, run)
    traces, active_client = _fetch_traces_bounded(
        client=client,
        profile=profile,
        experiment_id=experiment_id,
        run_id=run_id,
        trace_fetch_concurrency=trace_fetch_concurrency,
    )
    quality_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for trace in traces:
        quality, failures, retrieval, timings = _sanitize_trace(
            experiment_id=experiment_id,
            run_id=run_id,
            trace=trace,
        )
        quality_rows.extend(quality)
        failure_rows.extend(failures)
        retrieval_rows.extend(retrieval)
        timing_rows.extend(timings)

    tmp_dir = checkpoint_dir / f".tmp-{run_id}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "run_row": run_row,
            "metric_rows": metric_rows,
            "quality_rows": quality_rows,
            "failure_rows": failure_rows,
            "retrieval_rows": retrieval_rows,
            "timing_rows": timing_rows,
        }
        _atomic_write_pickle(tmp_dir / _RUN_PAYLOAD_FILE, payload)
        fingerprint_payload = {
            **remote_fingerprint,
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "trace_count": len(traces),
        }
        _atomic_write_text(
            tmp_dir / "fingerprint.json",
            json.dumps(fingerprint_payload, indent=2, sort_keys=True),
        )
        (tmp_dir / ".committed").write_text("1\n", encoding="utf-8")
        _fsync_directory(tmp_dir)
        final_dir = checkpoint_dir / "runs" / run_id
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
        _fsync_directory(checkpoint_dir / "runs")
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return active_client


def _compute_remote_fingerprint(
    *,
    client: Any,
    experiment: Any,
    run: Any,
    trace_fetch_concurrency: int,
) -> dict[str, Any]:
    """Build a read-only remote fingerprint for selective shard invalidation."""
    run_payload = {
        "run_id": str(run.info.run_id),
        "experiment_id": str(experiment.experiment_id),
        "experiment_name": experiment.name,
        "status": str(run.info.status),
        "start_time": getattr(run.info, "start_time", None),
        "end_time": getattr(run.info, "end_time", None),
        "lifecycle_stage": getattr(run.info, "lifecycle_stage", None),
        "params": dict(sorted(run.data.params.items())),
        "metrics": {
            key: value
            for key, value in sorted(run.data.metrics.items(), key=lambda item: item[0])
        },
    }
    run_digest = _stable_object_hash(run_payload)
    trace_records = _collect_trace_inventory(
        client=client,
        experiment_id=str(experiment.experiment_id),
        run_id=str(run.info.run_id),
        page_size=trace_fetch_concurrency,
    )
    per_trace = [
        {
            "trace_id": record["trace_id"],
            "digest": _stable_object_hash(record),
            "record": record,
        }
        for record in trace_records
    ]
    trace_set_digest = _stable_object_hash(
        [{"trace_id": item["trace_id"], "digest": item["digest"]} for item in per_trace]
    )
    return {
        "run_digest": run_digest,
        "trace_set_digest": trace_set_digest,
        "trace_count": len(per_trace),
        "run_payload": run_payload,
        "per_trace": per_trace,
    }


def _collect_trace_inventory(
    *,
    client: Any,
    experiment_id: str,
    run_id: str,
    page_size: int,
) -> list[dict[str, Any]]:
    """Paginate lightweight trace metadata for fingerprinting."""
    records: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        page = client.search_traces(
            locations=[experiment_id],
            run_id=run_id,
            include_spans=False,
            max_results=page_size,
            page_token=page_token,
        )
        for trace in page:
            info = trace.info
            assessments = []
            for assessment in getattr(info, "assessments", []) or []:
                assessments.append(
                    {
                        "name": str(getattr(assessment, "name", "")),
                        "value": _fingerprint_assessment_value(
                            getattr(assessment, "value", None)
                        ),
                        "metadata": _sanitized_assessment_metadata(
                            _as_mapping(getattr(assessment, "metadata", {}))
                        ),
                    }
                )
            tags = _as_mapping(getattr(info, "tags", {}))
            attributes = _as_mapping(getattr(info, "attributes", {}))
            records.append(
                {
                    "trace_id": str(info.trace_id),
                    "status": str(getattr(info, "status", "")),
                    "request_time": getattr(info, "request_time", None)
                    or getattr(info, "timestamp_ms", None),
                    "execution_duration": getattr(info, "execution_duration", None),
                    "tags": {
                        key: tags[key]
                        for key in sorted(tags)
                        if key
                        in {
                            "mlflow.traceName",
                            "mlflow.trace.status",
                        }
                        or key.startswith("eval.")
                    },
                    "attributes": {
                        key: attributes.get(key)
                        for key in (
                            "use_case",
                            "dataset_id",
                            "evalset_variant",
                            "row_id",
                        )
                    },
                    "assessments": sorted(assessments, key=lambda item: item["name"]),
                }
            )
        page_token = page.token
        if not page_token:
            break
    records.sort(key=lambda item: str(item["trace_id"]))
    return records


def _fingerprint_assessment_value(value: object) -> object:
    """Normalize an assessment value for fingerprinting without JSON conversion."""
    if value is None:
        return None
    if isinstance(value, str | bool | int | float):
        return value
    return str(value)


def _sanitized_assessment_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep assessment identity metadata while dropping rationale text."""
    modes = metadata.get("detected_error_modes", [])
    sanitized_modes: list[dict[str, Any]] = []
    if isinstance(modes, list):
        for item in modes:
            if isinstance(item, dict):
                sanitized_modes.append(
                    {
                        "name": item.get("name"),
                        "detected": item.get("detected"),
                    }
                )
    return {
        "detected_error_modes": sanitized_modes,
        "updated_at": metadata.get("updated_at"),
        "source": metadata.get("source"),
    }


def _fingerprints_match(local: dict[str, Any], remote: dict[str, Any]) -> bool:
    """Return whether local and remote metadata fingerprints agree."""
    return (
        local.get("run_digest") == remote.get("run_digest")
        and local.get("trace_set_digest") == remote.get("trace_set_digest")
    )


def _read_run_fingerprint(checkpoint_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Load a committed run fingerprint when present."""
    path = checkpoint_dir / "runs" / run_id / "fingerprint.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _run_shard_is_committed(checkpoint_dir: Path, run_id: str) -> bool:
    """Return whether a run shard directory contains the commit sentinel."""
    return (checkpoint_dir / "runs" / run_id / ".committed").exists()


def _delete_run_shard(checkpoint_dir: Path, run_id: str) -> None:
    """Delete only one experiment-run shard, leaving the parent checkpoint intact."""
    shard = checkpoint_dir / "runs" / run_id
    if shard.exists():
        shutil.rmtree(shard)


def _read_run_payload(checkpoint_dir: Path, run_id: str) -> dict[str, Any]:
    """Deserialize a committed run payload without pandas coercion.

    Raises
    ------
    RuntimeError
        If the payload is missing or unreadable.
    """
    payload_path = checkpoint_dir / "runs" / run_id / _RUN_PAYLOAD_FILE
    if not payload_path.exists():
        raise RuntimeError(
            f"Committed run shard {run_id} is missing payload file {payload_path.name}."
        )
    try:
        with payload_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read committed run payload for {run_id}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid payload structure for run {run_id}.")
    return payload


def _aggregate_and_publish(
    *,
    pandas: Any,
    output_dir: Path,
    checkpoint_dir: Path,
    cli_identity: dict[str, object],
    planned_run_ids: list[str],
    experiment_rows: list[dict[str, object]],
) -> None:
    """Aggregate committed shards into validated root snapshot artifacts."""
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".final-staging-", dir=str(output_dir))
    )
    try:
        run_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        quality_rows: list[dict[str, object]] = []
        failure_rows: list[dict[str, object]] = []
        retrieval_rows: list[dict[str, object]] = []
        timing_rows: list[dict[str, object]] = []

        for run_id in planned_run_ids:
            shard = checkpoint_dir / "runs" / run_id
            if not (shard / ".committed").exists():
                raise RuntimeError(f"Missing committed shard for run {run_id}.")
            payload = _read_run_payload(checkpoint_dir, run_id)
            run_rows.append(payload["run_row"])
            metric_rows.extend(payload["metric_rows"])
            quality_rows.extend(payload["quality_rows"])
            failure_rows.extend(payload["failure_rows"])
            retrieval_rows.extend(payload["retrieval_rows"])
            timing_rows.extend(payload["timing_rows"])

        quality_rows = [_encode_quality_row(row) for row in quality_rows]

        _write_parquet(
            pandas,
            staging_dir / "experiments.parquet",
            experiment_rows,
            _EXPERIMENT_COLUMNS,
        )
        _write_parquet(pandas, staging_dir / "runs.parquet", run_rows, _RUN_COLUMNS)
        _write_parquet(
            pandas, staging_dir / "run_metrics.parquet", metric_rows, _METRIC_COLUMNS
        )
        _write_parquet(
            pandas,
            staging_dir / "trace_quality.parquet",
            quality_rows,
            _TRACE_QUALITY_COLUMNS,
        )
        _write_parquet(
            pandas,
            staging_dir / "trace_failures.parquet",
            failure_rows,
            _TRACE_FAILURE_COLUMNS,
        )
        _write_parquet(
            pandas,
            staging_dir / "trace_retrieval.parquet",
            retrieval_rows,
            _RETRIEVAL_COLUMNS,
        )
        _write_parquet(
            pandas,
            staging_dir / "span_timings.parquet",
            timing_rows,
            _SPAN_TIMING_COLUMNS,
        )

        for filename in _SNAPSHOT_PARQUET_FILES:
            if not (staging_dir / filename).exists():
                raise RuntimeError(f"Aggregation missing required file {filename}.")

        for filename in _SNAPSHOT_PARQUET_FILES:
            target = output_dir / filename
            if target.exists():
                target.unlink()
            os.replace(staging_dir / filename, target)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "tracking_uri": cli_identity["tracking_uri"],
            "profile": cli_identity["profile"],
            "selector": cli_identity["selector"],
            "all_runs": cli_identity["all_runs"],
            "trace_fetch_concurrency": cli_identity["trace_fetch_concurrency"],
            "experiment_count": len(experiment_rows),
            "run_count": len(planned_run_ids),
            "run_ids": planned_run_ids,
            "privacy_exclusions": _PRIVACY_EXCLUSIONS,
            "checkpoint_epoch": int(checkpoint_dir.name.removeprefix("checkpoint-")),
            "files": _file_manifest(output_dir),
        }
        _atomic_write_text(
            output_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        _validate_snapshot(output_dir)
        checkpoint = _read_checkpoint(checkpoint_dir)
        checkpoint["status"] = "completed"
        _write_checkpoint(checkpoint_dir, checkpoint)
        shutil.rmtree(checkpoint_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _encode_quality_row(row: dict[str, object]) -> dict[str, object]:
    """Encode a quality row's value into type-tagged final columns."""
    encoded: dict[str, object] = {
        "experiment_id": row["experiment_id"],
        "run_id": row["run_id"],
        "trace_id": row["trace_id"],
        "trace_status": row["trace_status"],
        "execution_time_ms": row["execution_time_ms"],
        "use_case": row["use_case"],
        "dataset_id": row["dataset_id"],
        "evalset_variant": row["evalset_variant"],
        "row_id": row["row_id"],
        "assessment_name": row["assessment_name"],
        "assessment_value_type": None,
        "assessment_value_string": None,
        "assessment_value_bool": None,
        "assessment_value_int": None,
        "assessment_value_float": None,
        "ordinal_grade": row["ordinal_grade"],
        "detected_error_modes": row["detected_error_modes"],
    }
    value = row["assessment_value"]
    if value is None:
        return encoded
    if isinstance(value, bool):
        encoded["assessment_value_type"] = "bool"
        encoded["assessment_value_bool"] = value
    elif isinstance(value, int):
        encoded["assessment_value_type"] = "int"
        encoded["assessment_value_int"] = value
    elif isinstance(value, float):
        encoded["assessment_value_type"] = "float"
        encoded["assessment_value_float"] = value
    elif isinstance(value, str):
        encoded["assessment_value_type"] = "string"
        encoded["assessment_value_string"] = value
    else:
        raise TypeError(
            f"Unsupported assessment value type {type(value).__name__}: "
            f"{value!r} for assessment {row.get('assessment_name')!r}"
        )
    return encoded


def decode_quality_value(row: dict[str, object]) -> object:
    """Reconstruct the original assessment value from a type-tagged final row.

    Parameters
    ----------
    row
        A dictionary or pandas row with one of the typed quality columns set.

    Returns
    -------
    object
        The original bool, int, float, str, or None value.
    """
    value_type = row.get("assessment_value_type")
    if value_type is None:
        return None
    if value_type == "bool":
        return row["assessment_value_bool"]
    if value_type == "int":
        return row["assessment_value_int"]
    if value_type == "float":
        return row["assessment_value_float"]
    if value_type == "string":
        return row["assessment_value_string"]
    raise ValueError(f"Unknown assessment_value_type: {value_type!r}")


def _load_mlflow_dependencies(profile: str) -> tuple[Any, Any]:
    """Load optional export dependencies and configure profile-aware MLflow."""
    try:
        import pandas
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional dependency group before exporting: "
            "uv sync --group mlflow"
        ) from exc
    return _create_mlflow_client(profile), pandas


def _create_mlflow_client(profile: str) -> Any:
    """Create a fresh MLflow client using the current Databricks profile credentials."""
    # `MlflowClient(tracking_uri=...)` selects the profile for REST calls, but
    # MLflow's Databricks SDK helpers also consult this environment variable.
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional dependency group before exporting: "
            "uv sync --group mlflow"
        ) from exc

    tracking_uri = f"databricks://{profile}"
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _fetch_traces_bounded(
    *,
    client: Any,
    profile: str,
    experiment_id: str,
    run_id: str,
    trace_fetch_concurrency: int,
    client_factory: Callable[[], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    input_reader: Callable[[str], str] = input,
    is_interactive: Callable[[], bool] | None = None,
) -> tuple[list[Any], Any]:
    """Fetch all traces with bounded retries and interactive auth recovery.

    The metadata page is requested without spans, then complete traces are
    fetched with a bounded worker pool. This avoids MLflow's unbounded
    artifact fan-out and uses ``locations`` rather than deprecated
    ``experiment_ids``. A full run remains atomic: traces from a failed run
    stay in memory and are discarded unless every page completes.
    """
    traces: list[Any] = []
    page_token: str | None = None
    page_number = 0
    auth_recovery_cycles = 0
    active_client = client
    client_factory = client_factory or (lambda: _create_mlflow_client(profile))
    is_interactive = is_interactive or sys.stdin.isatty

    while True:
        try:
            page = _retry_trace_request(
                operation=f"trace metadata page {page_number + 1} for run {run_id}",
                request=lambda: active_client.search_traces(
                    locations=[experiment_id],
                    run_id=run_id,
                    include_spans=False,
                    max_results=trace_fetch_concurrency,
                    page_token=page_token,
                ),
                sleep=sleep,
            )
        except _AuthenticationRecoveryRequired as exc:
            auth_recovery_cycles += 1
            active_client = _recover_interactive_authentication(
                profile=profile,
                recovery_cycle=auth_recovery_cycles,
                source_error=exc,
                client_factory=client_factory,
                input_reader=input_reader,
                is_interactive=is_interactive,
            )
            continue

        page_number += 1
        trace_ids = [trace.info.trace_id for trace in page]
        logger.info(
            "Fetching trace page %d: %d traces.",
            page_number,
            len(trace_ids),
        )
        while True:
            page_client = active_client
            executor = ThreadPoolExecutor(max_workers=trace_fetch_concurrency)
            try:
                page_traces = list(
                    executor.map(
                        lambda trace_id: _retry_trace_request(
                            operation=f"trace {trace_id} for run {run_id}",
                            request=lambda: page_client.get_trace(trace_id, display=False),
                            sleep=sleep,
                        ),
                        trace_ids,
                    )
                )
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                logger.info("Trace download interrupted; cancelling queued requests.")
                raise
            except _AuthenticationRecoveryRequired as exc:
                executor.shutdown(wait=True)
                auth_recovery_cycles += 1
                active_client = _recover_interactive_authentication(
                    profile=profile,
                    recovery_cycle=auth_recovery_cycles,
                    source_error=exc,
                    client_factory=client_factory,
                    input_reader=input_reader,
                    is_interactive=is_interactive,
                )
                continue
            else:
                executor.shutdown(wait=True)
                traces.extend(page_traces)
                break

        page_token = page.token
        if not page_token:
            return traces, active_client


def _retry_trace_request(
    *,
    operation: str,
    request: Callable[[], Any],
    sleep: Callable[[float], None],
) -> Any:
    """Execute one MLflow request with bounded transient-failure retries."""
    for attempt in range(1, _MAX_TRACE_REQUEST_ATTEMPTS + 1):
        try:
            return request()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if not _is_retryable_trace_error(exc):
                raise
            if attempt == _MAX_TRACE_REQUEST_ATTEMPTS:
                if _is_authentication_error(exc):
                    raise _AuthenticationRecoveryRequired(
                        f"Authentication retries exhausted for {operation}: "
                        f"{_safe_exception_summary(exc)}"
                    ) from exc
                raise _TraceRequestRetryExhausted(
                    f"Transient retries exhausted for {operation}: "
                    f"{_safe_exception_summary(exc)}"
                ) from exc
            delay = _retry_delay_seconds(attempt)
            logger.warning(
                "Retrying %s after attempt %d/%d failed (%s); waiting %.1fs.",
                operation,
                attempt,
                _MAX_TRACE_REQUEST_ATTEMPTS,
                _safe_exception_summary(exc),
                delay,
            )
            sleep(delay)
    raise AssertionError("retry loop should always return or raise")


def _recover_interactive_authentication(
    *,
    profile: str,
    recovery_cycle: int,
    source_error: _AuthenticationRecoveryRequired,
    client_factory: Callable[[], Any],
    input_reader: Callable[[str], str],
    is_interactive: Callable[[], bool],
) -> Any:
    """Pause for user-managed profile refresh and return a newly created client."""
    command = f"databricks auth login --profile {profile}"
    if recovery_cycle > _MAX_AUTH_RECOVERY_CYCLES:
        raise RuntimeError(
            "Databricks authentication did not recover after "
            f"{_MAX_AUTH_RECOVERY_CYCLES} manual refresh cycles. Run `{command}` "
            "and resume the same export command."
        ) from source_error
    if not is_interactive():
        raise RuntimeError(
            f"Databricks authentication needs refresh. Run `{command}` and resume "
            "the same export command; non-interactive execution cannot pause for login."
        ) from source_error

    logger.error(
        "Databricks authentication retries are exhausted. In another terminal, run `%s`. "
        "Then return here and press Enter to recreate the MLflow client (recovery %d/%d).",
        command,
        recovery_cycle,
        _MAX_AUTH_RECOVERY_CYCLES,
    )
    response = input_reader("Press Enter after successful Databricks login, or type 'q' to stop: ")
    if response.strip().lower() in {"q", "quit", "stop"}:
        raise RuntimeError(
            f"Authentication recovery cancelled. Run `{command}` and resume the same export command."
        ) from source_error
    return client_factory()


def _is_retryable_trace_error(error: Exception) -> bool:
    """Return whether a trace API error is appropriate for bounded retry."""
    message = str(error).lower()
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 429} or isinstance(status_code, int) and status_code >= 500:
        return True
    return any(
        marker in message
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "credential was not sent",
            "unsupported type for this api",
            "access token",
            "token refresh",
            "oidc",
            "rate limit",
            "too many requests",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
    )


def _is_authentication_error(error: Exception) -> bool:
    """Return whether a retryable error requires profile refresh on exhaustion."""
    message = str(error).lower()
    status_code = getattr(error, "status_code", None)
    return status_code == 401 or any(
        marker in message
        for marker in (
            "credential was not sent",
            "unsupported type for this api",
            "access token",
            "token refresh",
            "oidc",
            "authentication",
        )
    )


def _retry_delay_seconds(attempt: int) -> float:
    """Return capped exponential backoff with bounded jitter."""
    base_delay = min(_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), _RETRY_MAX_DELAY_SECONDS)
    return base_delay + random.uniform(0.0, base_delay * 0.2)


def _safe_exception_summary(error: Exception) -> str:
    """Return a content-free exception summary suitable for terminal logs."""
    status_code = getattr(error, "status_code", None)
    status = f" status={status_code}" if status_code is not None else ""
    return f"{type(error).__name__}{status}"


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
        "simple_required_tools": params.get("simple_required_tools"),
        "configured_retrieval_call_count": _simple_required_tool_count(
            params.get("simple_required_tools")
        ),
        "simple_merge_policy": params.get("simple_merge_policy"),
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
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Return privacy-reduced assessment, failure, retrieval, and timing rows."""
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
                "assessment_value": value,
                "ordinal_grade": None,
                "detected_error_modes": "[]",
            }
        )

    for row in quality_rows:
        row["ordinal_grade"] = ordinal_grade
        row["detected_error_modes"] = detected_error_modes

    retrieval_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for span in getattr(trace.data, "spans", []) or []:
        attributes = _as_mapping(getattr(span, "attributes", {}))
        operation = attributes.get("simple.operation")
        plan_failure = attributes.get("simple.retrieval_plan_failure")
        if plan_failure is not None:
            failure_rows.append(
                {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "trace_status": str(getattr(info, "status", "")),
                    "failure_type": "simple_retrieval_plan_validation",
                    "failure_reason": plan_failure,
                    "required_tool_multiset": attributes.get(
                        "simple.required_tool_multiset"
                    ),
                    "actual_tool_multiset": attributes.get(
                        "simple.actual_tool_multiset"
                    ),
                }
            )
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
                    "tool_name": attributes.get("simple.tool_name"),
                    "retrieval_mode": attributes.get("simple.retrieval_mode"),
                    "per_call_fetch_count": attributes.get(
                        "simple.per_call_fetch_count"
                    ),
                    "returned_chunk_count": attributes.get(
                        "simple.returned_chunk_count"
                    ),
                    "required_tool_multiset": attributes.get(
                        "simple.required_tool_multiset"
                    ),
                    "actual_tool_multiset": attributes.get(
                        "simple.actual_tool_multiset"
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
    return quality_rows, failure_rows, retrieval_rows, timing_rows


def _write_parquet(
    pandas: Any,
    path: Path,
    rows: list[dict[str, object]],
    columns: list[str],
) -> None:
    """Write a normalized table with stable empty-table columns."""
    dataframe = pandas.DataFrame(rows, columns=columns)
    dataframe.to_parquet(path, index=False)
    _fsync_file(path)


def _atomic_write_pickle(path: Path, payload: object) -> None:
    """Write a pickle payload via a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=_PICKLE_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_file(path)
        _fsync_directory(path.parent)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text via a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_file(path)
        _fsync_directory(path.parent)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    """Flush file contents to durable storage when the OS supports it."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Flush directory metadata when the OS supports it."""
    directory_fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _stable_object_hash(value: object) -> str:
    """Return a SHA-256 digest of a canonical JSON encoding."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    """Serialize a value with stable key ordering for fingerprinting."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


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
        *_SNAPSHOT_PARQUET_FILES,
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
        f"- Simple retrieval plan-validation failures: {summary['plan_validation_failure_count']}",
        f"- Pareto candidates: {summary['pareto_candidate_count']}",
        f"- Latency caveat: {summary['time_of_day_caveat']}",
        "",
        "## Pareto candidates",
        "",
    ]
    if pareto.empty:
        lines.append("No comparable runs with both quality and p50 latency metrics were found.")
    else:
        lines.append(_markdown_table(pareto))
    return "\n".join(lines) + "\n"


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


def _is_missing(value: object) -> bool:
    """Return whether a scalar is pandas/IEEE missing."""
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _as_mapping(value: object) -> dict[str, Any]:
    """Return a third-party metadata value as a mapping when possible."""
    return value if isinstance(value, dict) else {}


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


def _simple_required_tool_count(value: object) -> int | None:
    """Return the configured Simple Mode call count from a JSON tool multiset."""
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return len(parsed) if isinstance(parsed, list) else None
