"""Unit tests for the MLflow snapshot lossless checkpoint and assessment encoding."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from es_index_explorer.mlflow_analysis.snapshot import (
    DEFAULT_EXPERIMENT_FOLDER,
    SnapshotSelector,
    _aggregate_and_publish,
    _atomic_write_pickle,
    _AuthenticationRecoveryRequired,
    _cli_identity,
    _download_run_rubrics,
    _encode_quality_row,
    _export_context,
    _fetch_traces_bounded,
    _find_newest_matching_checkpoint,
    _markdown_table,
    _publish_needed,
    _read_published_view_signature,
    _read_run_payload,
    _recover_interactive_authentication,
    _resolve_checkpoint_session,
    _retry_trace_request,
    _sanitize_trace,
    _SNAPSHOT_PARQUET_FILES,
    _write_checkpoint,
    decode_quality_value,
)

pytestmark = pytest.mark.unit


def test_markdown_table_does_not_require_tabulate() -> None:
    """Render a Markdown table with only pandas data structures."""
    table = _markdown_table(pd.DataFrame([{"name": "A|B", "score": 1.25}]))

    assert table == "| name | score |\n| --- | --- |\n| A\\|B | 1.2500 |"


def _sample_cli_identity() -> dict[str, object]:
    """Return a reusable CLI identity for checkpoint-matching tests."""
    return _cli_identity(profile="applied-science")


def test_find_newest_matching_checkpoint_includes_completed(tmp_path: Path) -> None:
    """A retained, already-completed checkpoint remains a valid resume target."""
    cli_identity = _sample_cli_identity()
    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "runs").mkdir()
    _write_checkpoint(
        checkpoint_dir,
        {"status": "completed", "cli_identity": cli_identity, "discovery": {}},
    )

    found = _find_newest_matching_checkpoint(tmp_path, cli_identity)

    assert found == checkpoint_dir


def test_find_newest_matching_checkpoint_ignores_mismatched_identity(
    tmp_path: Path,
) -> None:
    """A completed checkpoint with a different profile is not reused."""
    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "runs").mkdir()
    _write_checkpoint(
        checkpoint_dir,
        {
            "status": "completed",
            "cli_identity": _cli_identity(profile="other-profile"),
            "discovery": {},
        },
    )

    found = _find_newest_matching_checkpoint(tmp_path, _sample_cli_identity())

    assert found is None


def test_cli_identity_is_independent_of_selector_and_all_runs() -> None:
    """The cache identity is stable across different selectors/all_runs.

    This is the key property that lets the same checkpoint be resumed and
    extended when the user changes, narrows, or widens the selected MLflow
    paths between invocations.
    """
    identity_one = _cli_identity(profile="applied-science")
    identity_two = _cli_identity(profile="applied-science")

    assert identity_one == identity_two
    assert "selector" not in identity_one
    assert "all_runs" not in identity_one
    assert "trace_fetch_concurrency" not in identity_one


def test_snapshot_selector_unions_multiple_prefixes_and_folders() -> None:
    """A selector with several prefixes and folders matches any of them."""
    selector = SnapshotSelector(
        experiment_prefixes=("S-A-", "S-B-"),
        experiment_folders=("experiments/A", "experiments/B"),
    )

    assert selector.matches("S-A-bm25-c1-rr-f10-g10-rnone")
    assert selector.matches("S-B-bm25-dense-c2-union-f15-rnone")
    assert selector.matches("experiments/A/some-run")
    assert selector.matches("experiments/B/some-run")
    assert not selector.matches("experiments/C/some-run")
    assert not selector.matches("S-C-unrelated")


def test_snapshot_selector_direct_children_applies_to_every_folder() -> None:
    """direct_children restricts matching under all given folders uniformly."""
    selector = SnapshotSelector(
        experiment_folders=("experiments/A", "experiments/B"),
        direct_children=True,
    )

    assert selector.matches("experiments/A/leaf")
    assert selector.matches("experiments/B/leaf")
    assert not selector.matches("experiments/A/nested/leaf")
    assert not selector.matches("experiments/B/nested/leaf")


def test_snapshot_selector_defaults_to_default_folder_when_unset() -> None:
    """An entirely empty selector falls back to the default recursive folder."""
    selector = SnapshotSelector()

    assert selector.matches(f"{DEFAULT_EXPERIMENT_FOLDER}some-experiment")
    assert not selector.matches("unrelated/some-experiment")


def test_snapshot_selector_direct_children_requires_a_folder() -> None:
    """direct_children without any folder is a configuration error."""
    with pytest.raises(ValueError, match="direct_children requires"):
        SnapshotSelector(direct_children=True)


def test_read_published_view_signature_returns_none_without_manifest(
    tmp_path: Path,
) -> None:
    """A fresh output directory with no manifest has no publish signature."""
    assert _read_published_view_signature(tmp_path) is None


def _touch_snapshot_parquet_files(output_dir: Path) -> None:
    """Create empty placeholder files for every required snapshot Parquet file."""
    for filename in _SNAPSHOT_PARQUET_FILES:
        (output_dir / filename).write_bytes(b"")


def test_read_published_view_signature_reads_existing_manifest(
    tmp_path: Path,
) -> None:
    """An existing manifest's signature fields are read back, sorted."""
    _touch_snapshot_parquet_files(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_ids": ["exp-2", "exp-1"],
                "run_ids": ["run-2", "run-1"],
                "selector": {"experiment_prefixes": ["S-A-"], "experiment_folders": []},
                "all_runs": True,
                "trace_fetch_concurrency": 5,
            }
        ),
        encoding="utf-8",
    )

    signature = _read_published_view_signature(tmp_path)

    assert signature == {
        "experiment_ids": ["exp-1", "exp-2"],
        "run_ids": ["run-1", "run-2"],
        "selector": {"experiment_prefixes": ["S-A-"], "experiment_folders": []},
        "all_runs": True,
    }
    assert "trace_fetch_concurrency" not in signature


def test_read_published_view_signature_tolerates_corrupt_manifest(
    tmp_path: Path,
) -> None:
    """A corrupt or unreadable manifest is treated as no prior publish."""
    (tmp_path / "manifest.json").write_text("not json", encoding="utf-8")

    assert _read_published_view_signature(tmp_path) is None


def test_read_published_view_signature_rejects_manifest_missing_experiment_ids(
    tmp_path: Path,
) -> None:
    """A pre-upgrade manifest without experiment_ids is not trusted."""
    _touch_snapshot_parquet_files(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "run_ids": ["run-1"],
                "selector": {"experiment_prefixes": ["S-A-"], "experiment_folders": []},
                "all_runs": True,
                "trace_fetch_concurrency": 5,
            }
        ),
        encoding="utf-8",
    )

    assert _read_published_view_signature(tmp_path) is None


def test_read_published_view_signature_rejects_manifest_with_missing_parquet_files(
    tmp_path: Path,
) -> None:
    """A well-formed manifest is not trusted if the snapshot Parquet files are gone.

    This guards against deciding "no changes, skip republish" when the
    published snapshot is actually incomplete or corrupted (for example, a
    Parquet file deleted or never fully written outside a normal export).
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_ids": ["exp-1"],
                "run_ids": ["run-1"],
                "selector": {"experiment_prefixes": ["S-A-"], "experiment_folders": []},
                "all_runs": True,
                "trace_fetch_concurrency": 5,
            }
        ),
        encoding="utf-8",
    )
    # Deliberately do not create the snapshot Parquet files.

    assert _read_published_view_signature(tmp_path) is None


def _sample_signature(**overrides: object) -> dict[str, object]:
    """Return a minimal publish signature, with optional field overrides."""
    signature: dict[str, object] = {
        "experiment_ids": ["exp-1"],
        "run_ids": ["run-1"],
        "selector": {"experiment_prefixes": ["S-A-"], "experiment_folders": []},
        "all_runs": True,
    }
    signature.update(overrides)
    return signature


def test_publish_needed_false_when_nothing_changed() -> None:
    """An unchanged selection with no downloads does not require a republish."""
    assert not _publish_needed(
        downloaded_this_session=False,
        current_signature=_sample_signature(),
        previous_signature=_sample_signature(),
    )


def test_publish_needed_true_when_selector_changes_but_run_ids_do_not() -> None:
    """A selector change alone must still trigger a republish (Bug 2)."""
    current = _sample_signature(
        selector={"experiment_prefixes": ["S-A-", "S-B-"], "experiment_folders": []}
    )
    previous = _sample_signature()

    assert _publish_needed(
        downloaded_this_session=False,
        current_signature=current,
        previous_signature=previous,
    )


def test_publish_needed_true_without_a_trustworthy_previous_publish() -> None:
    """A first-ever export with an empty selection still publishes (Bug 3)."""
    assert _publish_needed(
        downloaded_this_session=False,
        current_signature={
            "experiment_ids": [],
            "run_ids": [],
            "selector": {"experiment_prefixes": [], "experiment_folders": []},
            "all_runs": False,
        },
        previous_signature=None,
    )


def test_publish_needed_true_when_downloaded_this_session_regardless_of_signature() -> (
    None
):
    """A fresh download always forces a republish, even with an identical signature."""
    assert _publish_needed(
        downloaded_this_session=True,
        current_signature=_sample_signature(),
        previous_signature=_sample_signature(),
    )


def test_publish_needed_false_when_only_trace_fetch_concurrency_would_differ() -> None:
    """Concurrency is not part of the signature: it alone must not force a republish."""
    current_signature = _sample_signature()
    previous_manifest_signature = _sample_signature()
    # trace_fetch_concurrency is intentionally absent from both signatures;
    # simulate two invocations that only differed in that operational
    # parameter by confirming the (concurrency-free) signatures still match.
    assert not _publish_needed(
        downloaded_this_session=False,
        current_signature=current_signature,
        previous_signature=previous_manifest_signature,
    )


def _minimal_run_payload(run_id: str, experiment_id: str) -> dict[str, object]:
    """Return the minimum committed-shard payload accepted by _aggregate_and_publish."""
    return {
        "run_row": {
            "experiment_id": experiment_id,
            "experiment_name": "test",
            "run_id": run_id,
            "status": "FINISHED",
            "start_time_ms": 1_700_000_000_000,
            "start_time_utc": "2023-11-14T22:13:20+00:00",
            "start_hour_utc": 22,
            "start_weekday_utc": 1,
            "dataset": "ds",
            "model_version": "3.61",
            "simple_required_tools": '["get_relevant_documents_bm25"]',
            "configured_retrieval_call_count": 1,
            "simple_merge_policy": "round_robin",
            "simple_per_call_fetch_count": 10,
            "simple_global_context_chunk_count": 10,
            "reasoning_effort": "none",
            "invocation_concurrency": 1,
        },
        "metric_rows": [],
        "quality_rows": [],
        "failure_rows": [],
        "invocation_rows": [],
        "criteria_rows": [],
        "run_rubric_rows": [],
        "retrieval_rows": [],
        "timing_rows": [],
    }


def test_aggregate_and_publish_writes_manifest_from_export_context(
    tmp_path: Path,
) -> None:
    """_aggregate_and_publish must not read removed cli_identity keys (Bug 1)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoint-1"
    run_shard = checkpoint_dir / "runs" / "run-1"
    run_shard.mkdir(parents=True)
    _atomic_write_pickle(
        run_shard / "run_payload.pickle",
        _minimal_run_payload("run-1", "exp-1"),
    )
    (run_shard / ".committed").write_text("1\n", encoding="utf-8")
    _write_checkpoint(
        checkpoint_dir,
        {"status": "aggregating", "cli_identity": {}, "discovery": {}},
    )

    export_context = _export_context(
        profile="applied-science",
        selector=SnapshotSelector(experiment_prefixes=("S-A-",)),
        all_runs=True,
        trace_fetch_concurrency=5,
    )

    _aggregate_and_publish(
        pandas=pd,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        export_context=export_context,
        planned_run_ids=["run-1"],
        experiment_rows=[
            {
                "experiment_id": "exp-1",
                "experiment_name": "S-A-bm25-c1-rr-f10-g10-rnone",
                "lifecycle_stage": "active",
            }
        ],
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "applied-science"
    assert manifest["tracking_uri"] == "databricks://applied-science"
    assert manifest["selector"] == {
        "experiment_prefixes": ["S-A-"],
        "experiment_folders": [],
        "direct_children": False,
    }
    assert manifest["all_runs"] is True
    assert manifest["trace_fetch_concurrency"] == 5
    assert manifest["experiment_ids"] == ["exp-1"]
    assert manifest["run_ids"] == ["run-1"]
    for filename in _SNAPSHOT_PARQUET_FILES:
        assert (output_dir / filename).exists()


def test_aggregate_and_publish_writes_a_valid_empty_snapshot(tmp_path: Path) -> None:
    """A first-ever export with zero matching runs still publishes real files.

    Exercises the end-to-end writer side of Bug 3: an empty selection must
    still produce a complete, valid (if empty) snapshot rather than silently
    skipping publication.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoint-1"
    (checkpoint_dir / "runs").mkdir(parents=True)
    _write_checkpoint(
        checkpoint_dir,
        {"status": "aggregating", "cli_identity": {}, "discovery": {}},
    )

    export_context = _export_context(
        profile="applied-science",
        selector=SnapshotSelector(experiment_prefixes=("S-nonexistent-",)),
        all_runs=True,
        trace_fetch_concurrency=5,
    )

    _aggregate_and_publish(
        pandas=pd,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        export_context=export_context,
        planned_run_ids=[],
        experiment_rows=[],
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_ids"] == []
    assert manifest["run_ids"] == []
    assert manifest["experiment_count"] == 0
    assert manifest["run_count"] == 0
    for filename in _SNAPSHOT_PARQUET_FILES:
        parquet_path = output_dir / filename
        assert parquet_path.exists()
        assert len(pd.read_parquet(parquet_path)) == 0
    # A signature can now be read back from this valid, if empty, publish.
    assert _read_published_view_signature(output_dir) == {
        "experiment_ids": [],
        "run_ids": [],
        "selector": {
            "experiment_prefixes": ["S-nonexistent-"],
            "experiment_folders": [],
            "direct_children": False,
        },
        "all_runs": True,
    }


def test_resolve_checkpoint_session_resumes_completed_epoch(tmp_path: Path) -> None:
    """An explicit --checkpoint-epoch resumes a completed checkpoint without --fresh."""
    cli_identity = _sample_cli_identity()
    checkpoint_dir = tmp_path / "checkpoint-200"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "runs").mkdir()
    _write_checkpoint(
        checkpoint_dir,
        {"status": "completed", "cli_identity": cli_identity, "discovery": {}},
    )

    resolved = _resolve_checkpoint_session(
        output_dir=tmp_path,
        cli_identity=cli_identity,
        resume=True,
        fresh=False,
        checkpoint_epoch=200,
    )

    assert resolved == checkpoint_dir


def test_resolve_checkpoint_session_auto_resumes_completed_checkpoint(
    tmp_path: Path,
) -> None:
    """Default auto-resume picks up a retained, already-completed checkpoint."""
    cli_identity = _sample_cli_identity()
    checkpoint_dir = tmp_path / "checkpoint-300"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "runs").mkdir()
    _write_checkpoint(
        checkpoint_dir,
        {"status": "completed", "cli_identity": cli_identity, "discovery": {}},
    )

    resolved = _resolve_checkpoint_session(
        output_dir=tmp_path,
        cli_identity=cli_identity,
        resume=True,
        fresh=False,
        checkpoint_epoch=None,
    )

    assert resolved == checkpoint_dir


class FakeRestError(Exception):
    """Represent a status-bearing transient MLflow REST error."""

    def __init__(self, status_code: int, message: str = "request failed") -> None:
        """Initialize a fake MLflow REST error."""
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    "value,expected_type,expected_columns",
    [
        (None, None, {}),
        ("ok", "string", {"assessment_value_string": "ok"}),
        (True, "bool", {"assessment_value_bool": True}),
        (False, "bool", {"assessment_value_bool": False}),
        (42, "int", {"assessment_value_int": 42}),
        (-7, "int", {"assessment_value_int": -7}),
        (3.14, "float", {"assessment_value_float": 3.14}),
    ],
)
def test_encode_quality_row_preserves_scalar_value(
    value: object,
    expected_type: str | None,
    expected_columns: dict[str, object],
) -> None:
    """Type-tagged quality encoding preserves original scalar types."""
    base_row = {
        "experiment_id": "exp-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "trace_status": "OK",
        "execution_time_ms": 100.0,
        "use_case": None,
        "dataset_id": None,
        "evalset_variant": None,
        "row_id": None,
        "assessment_name": "test",
        "assessment_value": value,
        "ordinal_grade": "A",
        "detected_error_modes": "[]",
    }
    encoded = _encode_quality_row(base_row)

    assert encoded["assessment_value_type"] == expected_type
    for column, expected_value in expected_columns.items():
        assert encoded[column] == expected_value
    assert decode_quality_value(encoded) == value


def test_encode_quality_row_rejects_unsupported_types() -> None:
    """Unsupported assessment value types raise a clear error at export time."""
    base_row = {
        "experiment_id": "exp-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "trace_status": "OK",
        "execution_time_ms": 100.0,
        "use_case": None,
        "dataset_id": None,
        "evalset_variant": None,
        "row_id": None,
        "assessment_name": "test",
        "assessment_value": {"nested": "dict"},
        "ordinal_grade": "A",
        "detected_error_modes": "[]",
    }
    with pytest.raises(TypeError):
        _encode_quality_row(base_row)


def test_round_trip_quality_rows_through_parquet(tmp_path: Path) -> None:
    """Encoded quality rows survive a single Parquet round-trip."""
    from es_index_explorer.mlflow_analysis.snapshot import (
        _TRACE_QUALITY_COLUMNS,
        _write_parquet,
    )

    rows = [
        _encode_quality_row(
            {
                "experiment_id": "exp-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "trace_status": "OK",
                "execution_time_ms": 100.0,
                "use_case": None,
                "dataset_id": None,
                "evalset_variant": None,
                "row_id": None,
                "assessment_name": "RubricV2",
                "assessment_value": True,
                "ordinal_grade": "A",
                "detected_error_modes": "[]",
            }
        ),
        _encode_quality_row(
            {
                "experiment_id": "exp-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "trace_status": "OK",
                "execution_time_ms": 100.0,
                "use_case": None,
                "dataset_id": None,
                "evalset_variant": None,
                "row_id": None,
                "assessment_name": "ordinal_grade",
                "assessment_value": "A",
                "ordinal_grade": "A",
                "detected_error_modes": "[]",
            }
        ),
        _encode_quality_row(
            {
                "experiment_id": "exp-1",
                "run_id": "run-1",
                "trace_id": "trace-2",
                "trace_status": "OK",
                "execution_time_ms": 50.0,
                "use_case": None,
                "dataset_id": None,
                "evalset_variant": None,
                "row_id": None,
                "assessment_name": "score",
                "assessment_value": 7,
                "ordinal_grade": None,
                "detected_error_modes": "[]",
            }
        ),
    ]
    parquet_path = tmp_path / "quality.parquet"
    _write_parquet(pd, parquet_path, rows, _TRACE_QUALITY_COLUMNS)
    dataframe = pd.read_parquet(parquet_path)
    decoded = [decode_quality_value(row) for _, row in dataframe.iterrows()]
    assert decoded == [True, "A", 7]


def test_run_payload_round_trip_preserves_object_types(tmp_path: Path) -> None:
    """Checkpoint pickle round-trip preserves list/dict/string/bool/numeric values."""
    payload = {
        "schema_version": 3,
        "run_id": "run-1",
        "run_row": {
            "experiment_id": "exp-1",
            "experiment_name": "test",
            "run_id": "run-1",
            "status": "FINISHED",
            "start_time_ms": 1_700_000_000_000,
            "start_time_utc": "2023-11-14T22:13:20+00:00",
            "start_hour_utc": 22,
            "start_weekday_utc": 1,
            "dataset": "ds",
            "model_version": "3.61",
            "simple_required_tools": '["get_relevant_documents_bm25"]',
            "configured_retrieval_call_count": 1,
            "simple_merge_policy": "round_robin",
            "simple_per_call_fetch_count": 10,
            "simple_global_context_chunk_count": 10,
            "reasoning_effort": "none",
            "invocation_concurrency": 1,
        },
        "metric_rows": [
            {
                "experiment_id": "exp-1",
                "run_id": "run-1",
                "metric_key": "x",
                "metric_value": 1.5,
            }
        ],
        "quality_rows": [
            {
                "experiment_id": "exp-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "trace_status": "OK",
                "execution_time_ms": 100.0,
                "use_case": None,
                "dataset_id": "ds",
                "evalset_variant": "v1",
                "row_id": "row-1",
                "assessment_name": "RubricV2",
                "assessment_value": True,
                "ordinal_grade": "A",
                "detected_error_modes": "[]",
            }
        ],
        "retrieval_rows": [
            {
                "experiment_id": "exp-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "span_name": "simple.retrieval_generic",
                "simple_operation": "simple.retrieval_generic",
                "tool_name": "GetRelevantDocumentsBm25",
                "retrieval_mode": "bm25",
                "per_call_fetch_count": 10,
                "returned_chunk_count": 10,
                "required_tool_multiset": '{"get_relevant_documents_bm25": 1}',
                "actual_tool_multiset": '{"get_relevant_documents_bm25": 1}',
                "actual_retrieval_calls": 1,
                "generic_retrieval_calls": 1,
                "metadata_filter_retrieval_calls": 0,
                "selected_chunk_count": 10,
                "context_size_chars": 1234,
                "merge_policy": "round_robin",
                "global_context_chunk_count": 10,
                "ranked_chunk_ids": "[]",
            }
        ],
        "failure_rows": [],
        "invocation_rows": [],
        "criteria_rows": [],
        "run_rubric_rows": [],
        "timing_rows": [
            {
                "experiment_id": "exp-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "span_name": "simple.retrieval_generic",
                "span_type": "TOOL",
                "simple_operation": "simple.retrieval_generic",
                "observed_duration_ms": 100.0,
                "llm_successful_attempt_latency_ms": None,
                "llm_successful_attempt_count": None,
                "es_success_duration_ms": 80.0,
                "es_success_attempt_count": 1,
                "query_embedding_duration_ms": 10.0,
            }
        ],
    }
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    runs_dir = checkpoint_dir / "runs" / "run-1"
    runs_dir.mkdir(parents=True)
    _atomic_write_pickle(runs_dir / "run_payload.pickle", payload)
    (runs_dir / ".committed").write_text("1\n", encoding="utf-8")

    loaded = _read_run_payload(checkpoint_dir, "run-1")
    assert loaded == payload
    assert loaded["quality_rows"][0]["assessment_value"] is True
    assert loaded["metric_rows"][0]["metric_value"] == 1.5


def test_sanitize_trace_exports_simple_plan_validation_failure() -> None:
    """Export a safe plan-validation failure even without scorer assessments."""
    trace = SimpleNamespace(
        info=SimpleNamespace(
            trace_id="trace-1",
            status="ERROR",
            assessments=[],
            attributes={},
        ),
        data=SimpleNamespace(
            spans=[
                SimpleNamespace(
                    name="evals_complete",
                    span_type="AGENT",
                    start_time_ns=1,
                    end_time_ns=2,
                    attributes={
                        "simple.retrieval_plan_valid": False,
                        "simple.required_tool_multiset": (
                            '{"get_relevant_documents_bm25": 1}'
                        ),
                        "simple.actual_tool_multiset": (
                            '{"get_relevant_documents_dense": 1}'
                        ),
                        "simple.retrieval_plan_failure": (
                            "emitted retrieval tool multiset does not match configured plan"
                        ),
                    },
                )
            ]
        ),
    )

    quality, failures, invocations, criteria, retrieval, timings = _sanitize_trace(
        experiment_id="experiment-1",
        run_id="run-1",
        trace=trace,
    )

    assert quality == []
    assert invocations == []
    assert criteria == []
    assert retrieval == []
    assert len(timings) == 1
    assert failures == [
        {
            "experiment_id": "experiment-1",
            "run_id": "run-1",
            "trace_id": "trace-1",
            "trace_status": "ERROR",
            "failure_type": "simple_retrieval_plan_validation",
            "failure_reason": "emitted retrieval tool multiset does not match configured plan",
            "required_tool_multiset": '{"get_relevant_documents_bm25": 1}',
            "actual_tool_multiset": '{"get_relevant_documents_dense": 1}',
        }
    ]


def test_sanitize_trace_exports_invocation_identity_and_criterion_states() -> None:
    """Root span identity and scorer states are retained without rationale text."""
    trace = SimpleNamespace(
        info=SimpleNamespace(
            trace_id="trace-1",
            status="OK",
            execution_duration=100,
            assessments=[
                SimpleNamespace(
                    name="RubricV2",
                    value=0.5,
                    metadata={
                        "file_path": "/tmp/rubric_data/air_assist/test.rubric.toml",
                        "dataset_id": "emc2_set1",
                        "use_case": ["communications_analysis"],
                    },
                    rationale=(
                        "| ✓ | **expectation_one** | private rationale | ✅ PASS |\n"
                        "| ○ | **expectation_two** | private rationale | ❌ FAIL |"
                    ),
                ),
                SimpleNamespace(
                    name="ordinal_grade",
                    value="Partial",
                    metadata={},
                    rationale=None,
                ),
            ],
            attributes={},
        ),
        data=SimpleNamespace(
            spans=[
                SimpleNamespace(
                    name="invoke_communications_analysis_2_v1",
                    span_type="UNKNOWN",
                    start_time_ns=1,
                    end_time_ns=2,
                    inputs={
                        "messages": [
                            {"role": "user", "content": "What happened?"},
                        ]
                    },
                    attributes={
                        "use_case": "communications_analysis",
                        "dataset_id": "emc2_set1",
                        "evalset_variant": "v1",
                        "row_id": "row-2",
                    },
                )
            ]
        ),
    )

    quality, failures, invocations, criteria, retrieval, timings = _sanitize_trace(
        experiment_id="experiment-1",
        run_id="run-1",
        trace=trace,
    )

    assert failures == []
    assert retrieval == []
    assert len(timings) == 1
    assert quality[0]["use_case"] == "communications_analysis"
    assert invocations == [
        {
            "experiment_id": "experiment-1",
            "run_id": "run-1",
            "trace_id": "trace-1",
            "trace_status": "OK",
            "rubric_key": "invoke_communications_analysis_2_v1",
            "question": "What happened?",
            "use_case": "communications_analysis",
            "dataset_id": "emc2_set1",
            "evalset_variant": "v1",
            "row_id": "row-2",
            "rubric_index": 2,
            "variant_index": 1,
            "rubric_file_path": "/tmp/rubric_data/air_assist/test.rubric.toml",
            "criteria_parse_status": "parsed",
        }
    ]
    assert criteria == [
        {
            "experiment_id": "experiment-1",
            "run_id": "run-1",
            "trace_id": "trace-1",
            "rubric_file_path": "/tmp/rubric_data/air_assist/test.rubric.toml",
            "expectation_name": "expectation_one",
            "material": True,
            "state": "PASS",
        },
        {
            "experiment_id": "experiment-1",
            "run_id": "run-1",
            "trace_id": "trace-1",
            "rubric_file_path": "/tmp/rubric_data/air_assist/test.rubric.toml",
            "expectation_name": "expectation_two",
            "material": False,
            "state": "FAIL",
        },
    ]


def test_download_run_rubrics_sanitizes_questions_and_artifact_identity(
    tmp_path: Path,
) -> None:
    """Run rubric artifacts retain question/identity fields but not raw inputs."""

    class ArtifactClient:
        """Write a representative MLflow table artifact to the requested directory."""

        def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
            assert run_id == "run-1"
            assert path == "rubrics.json"
            artifact_path = Path(dst_path) / path
            artifact_path.write_text(
                json.dumps(
                    {
                        "columns": [
                            "name",
                            "dataset",
                            "use_case",
                            "inputs",
                            "# input variants",
                            "author",
                            "full_path",
                        ],
                        "data": [
                            [
                                "test.rubric.toml",
                                "emc2_set1",
                                ["communications_analysis"],
                                [
                                    json.dumps(
                                        {
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": "What happened?",
                                                }
                                            ]
                                        }
                                    )
                                ],
                                1,
                                "author",
                                "/tmp/rubric_data/air_assist/test.rubric.toml",
                            ]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return str(artifact_path)

    rows = _download_run_rubrics(
        client=ArtifactClient(),
        experiment_id="experiment-1",
        run_id="run-1",
    )

    assert rows[0]["question"] == "What happened?"
    assert rows[0]["rubric_file_path"].endswith("test.rubric.toml")
    assert rows[0]["artifact_status"] == "ok"
    assert rows[0]["artifact_sha256"] is not None


def test_missing_payload_raises_integrity_error(tmp_path: Path) -> None:
    """A committed shard with a missing payload raises an integrity error."""
    checkpoint_dir = tmp_path / "checkpoint-1"
    shard_dir = checkpoint_dir / "runs" / "run-1"
    shard_dir.mkdir(parents=True)
    (shard_dir / ".committed").write_text("1\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _read_run_payload(checkpoint_dir, "run-1")


def test_retry_trace_request_retries_transient_failure() -> None:
    """A transient trace request retries and returns its successful result."""
    attempts = 0
    delays: list[float] = []

    def request() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary network timeout")
        return "success"

    result = _retry_trace_request(
        operation="trace trace-1 for run run-1",
        request=request,
        sleep=delays.append,
    )

    assert result == "success"
    assert attempts == 3
    assert len(delays) == 2


def test_retry_trace_request_requires_authentication_after_exhaustion() -> None:
    """An exhausted 401 is escalated to interactive authentication recovery."""
    with pytest.raises(_AuthenticationRecoveryRequired):
        _retry_trace_request(
            operation="trace trace-1 for run run-1",
            request=lambda: (_ for _ in ()).throw(
                FakeRestError(401, "Credential was not sent")
            ),
            sleep=lambda _: None,
        )


def test_retry_trace_request_preserves_keyboard_interrupt() -> None:
    """Cancellation is never converted into a transient request retry."""
    with pytest.raises(KeyboardInterrupt):
        _retry_trace_request(
            operation="trace trace-1 for run run-1",
            request=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            sleep=lambda _: None,
        )


def test_fetch_traces_recovers_after_interactive_authentication() -> None:
    """A successful manual login recreates the client and resumes the trace page."""

    class Page(list[SimpleNamespace]):
        """Expose MLflow's page token contract."""

        token = None

    class UnauthenticatedClient:
        """Return trace metadata but reject full trace downloads."""

        def search_traces(self, **_: object) -> Page:
            """Return one page of trace metadata."""
            return Page([SimpleNamespace(info=SimpleNamespace(trace_id="trace-1"))])

        def get_trace(self, _: str, display: bool) -> object:
            """Reject full trace download while credentials are expired."""
            assert display is False
            raise FakeRestError(401, "Credential was not sent")

    class AuthenticatedClient:
        """Return one trace after authentication refresh."""

        def search_traces(self, **_: object) -> Page:
            """Return one page of trace metadata."""
            return Page([SimpleNamespace(info=SimpleNamespace(trace_id="trace-1"))])

        def get_trace(self, trace_id: str, display: bool) -> dict[str, str]:
            """Return a full trace."""
            assert display is False
            return {"trace_id": trace_id}

    initial_client = UnauthenticatedClient()
    replacement_client = AuthenticatedClient()
    prompts: list[str] = []
    factory_calls = 0

    def client_factory() -> AuthenticatedClient:
        nonlocal factory_calls
        factory_calls += 1
        return replacement_client

    traces, active_client = _fetch_traces_bounded(
        client=initial_client,
        profile="applied-science",
        experiment_id="experiment-1",
        run_id="run-1",
        trace_fetch_concurrency=1,
        client_factory=client_factory,
        sleep=lambda _: None,
        input_reader=lambda prompt: prompts.append(prompt) or "",
        is_interactive=lambda: True,
    )

    assert traces == [{"trace_id": "trace-1"}]
    assert active_client is replacement_client
    assert factory_calls == 1
    assert len(prompts) == 1


def test_authentication_recovery_fails_without_interactive_terminal() -> None:
    """A non-interactive exporter fails clearly instead of blocking for login."""
    with pytest.raises(RuntimeError, match="non-interactive"):
        _recover_interactive_authentication(
            profile="applied-science",
            recovery_cycle=1,
            source_error=_AuthenticationRecoveryRequired("authentication expired"),
            client_factory=lambda: object(),
            input_reader=lambda _: "",
            is_interactive=lambda: False,
        )
