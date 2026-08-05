"""Unit tests for the MLflow snapshot lossless checkpoint and assessment encoding."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from es_index_explorer.mlflow_analysis.snapshot import (
    _AuthenticationRecoveryRequired,
    _encode_quality_row,
    _fetch_traces_bounded,
    _recover_interactive_authentication,
    _read_run_payload,
    _retry_trace_request,
    _atomic_write_pickle,
    decode_quality_value,
)

pytestmark = pytest.mark.unit


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
    from es_index_explorer.mlflow_analysis.snapshot import _write_parquet, _TRACE_QUALITY_COLUMNS

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
        "schema_version": 1,
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
            "simple_retrieval_mode": "bm25",
            "simple_merge_policy": "round_robin",
            "requested_retrieval_calls": 1,
            "simple_per_call_fetch_count": 10,
            "simple_global_context_chunk_count": 10,
            "reasoning_effort": "none",
            "invocation_concurrency": 1,
        },
        "metric_rows": [
            {"experiment_id": "exp-1", "run_id": "run-1", "metric_key": "x", "metric_value": 1.5}
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
                "tool_ordinal": 0,
                "retrieval_mode": "bm25",
                "per_call_fetch_count": 10,
                "returned_chunk_count": 10,
                "requested_retrieval_calls": 1,
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
