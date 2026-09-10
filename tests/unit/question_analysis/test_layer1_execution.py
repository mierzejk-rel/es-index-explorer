"""Tests for deterministic Layer 1 execution and progress infrastructure."""

import io
import json
import signal
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest

from es_index_explorer.question_analysis.layer1_execution import (
    THREAD_LIMIT_ENVIRONMENT_VARIABLES,
    AutoTuneResult,
    Layer1ExecutionConfig,
    ProgressEvent,
    RateLimitedProgressWriter,
    ShutdownToken,
    SpawnProcessExecutor,
    TaskReference,
    WorkUnit,
    WorkUnitKind,
    aggregate_rss_bytes,
    auto_select_worker_count,
    discover_worker_candidates,
    execute_work_units,
    resolve_worker_count,
    signal_shutdown_context,
)

pytestmark = pytest.mark.unit


def _write_task_module(tmp_path: Path) -> TaskReference:
    module_path = tmp_path / "layer1_execution_probe.py"
    module_path.write_text(
        "\n".join(  # noqa: FLY002 - line sequence keeps generated module indentation explicit.
            (
                "import os",
                "import time",
                "OBSERVED_ENVIRONMENT = {",
                "    name: os.environ.get(name)",
                "    for name in (",
                '        "VECLIB_MAXIMUM_THREADS",',
                '        "OMP_NUM_THREADS",',
                '        "OPENBLAS_NUM_THREADS",',
                '        "MKL_NUM_THREADS",',
                "    )",
                "}",
                "",
                "def run(unit):",
                "    delay, value = unit.payload",
                "    time.sleep(delay)",
                "    return {",
                '        "value": value,',
                '        "environment": OBSERVED_ENVIRONMENT,',
                '        "worker_marker": os.environ.get("ES_INDEX_EXPLORER_LAYER1_WORKER"),',
                "    }",
                "",
                "def identity(unit):",
                "    return unit.payload",
                "",
                "def nested(unit):",
                "    from es_index_explorer.question_analysis.layer1_execution import (",
                "        Layer1ExecutionConfig,",
                "        TaskReference,",
                "        WorkUnit,",
                "        WorkUnitKind,",
                "        execute_work_units,",
                "    )",
                "    result = execute_work_units(",
                '        TaskReference(module="layer1_execution_probe", function="identity"),',
                '        (WorkUnit("inner", WorkUnitKind.START, unit.payload),),',
                '        config=Layer1ExecutionConfig(worker_mode="process", worker_count=2),',
                "    )",
                "    return result[0].value",
                "",
            )
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    return TaskReference(module="layer1_execution_probe", function="run")


@pytest.mark.parametrize("worker_count", (1, 2))
def test_spawn_executor_returns_results_in_submit_order(
    tmp_path: Path,
    worker_count: int,
) -> None:
    task = _write_task_module(tmp_path)
    units = (
        WorkUnit("first", WorkUnitKind.RUBRIC_CONTRIBUTION, (0.08, 1)),
        WorkUnit("second", WorkUnitKind.RUBRIC_CONTRIBUTION, (0.0, 2)),
        WorkUnit("third", WorkUnitKind.RUBRIC_CONTRIBUTION, (0.02, 3)),
    )
    try:
        results = execute_work_units(
            task,
            units,
            config=Layer1ExecutionConfig(
                worker_mode="process",
                worker_count=worker_count,
            ),
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert tuple(result.task_id for result in results) == ("first", "second", "third")
    values = []
    for result in results:
        assert isinstance(result.value, dict)
        values.append(result.value["value"])
    assert tuple(values) == (1, 2, 3)
    assert all(result.succeeded for result in results)


def test_worker_sets_native_thread_limits_before_lazy_task_import(
    tmp_path: Path,
) -> None:
    task = _write_task_module(tmp_path)
    try:
        result = execute_work_units(
            task,
            (WorkUnit("environment", WorkUnitKind.START, (0.0, 1)),),
            config=Layer1ExecutionConfig(worker_mode="process", worker_count=1),
        )[0]
    finally:
        sys.path.remove(str(tmp_path))

    assert isinstance(result.value, dict)
    value = result.value
    assert value["environment"] == {
        name: "1" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    }
    assert value["worker_marker"] == "1"


def test_spawn_executor_reuses_one_pool_across_batches(tmp_path: Path) -> None:
    task = _write_task_module(tmp_path)
    try:
        with SpawnProcessExecutor(1) as executor:
            first = executor.execute(
                task,
                (WorkUnit("first", WorkUnitKind.START, (0.0, 1)),),
            )
            second = executor.execute(
                task,
                (WorkUnit("second", WorkUnitKind.START, (0.0, 2)),),
            )
    finally:
        sys.path.remove(str(tmp_path))

    assert first[0].succeeded
    assert second[0].succeeded
    assert first[0].value != second[0].value


def test_process_workers_do_not_create_nested_pools(tmp_path: Path) -> None:
    _write_task_module(tmp_path)
    task = TaskReference(module="layer1_execution_probe", function="nested")
    try:
        result = execute_work_units(
            task,
            (WorkUnit("outer", WorkUnitKind.START, 17),),
            config=Layer1ExecutionConfig(worker_mode="process", worker_count=1),
        )[0]
    finally:
        sys.path.remove(str(tmp_path))

    assert result.succeeded
    assert result.value == 17


def test_sequential_executor_and_auto_single_worker_match() -> None:
    task = TaskReference(module=__name__, function="_sequential_probe")
    units = (
        WorkUnit("a", WorkUnitKind.PI_CONDITIONAL, 1),
        WorkUnit("b", WorkUnitKind.PI_CONDITIONAL, 2),
    )

    sequential = execute_work_units(
        task,
        units,
        config=Layer1ExecutionConfig(worker_mode="sequential"),
    )
    automatic = execute_work_units(
        task,
        units,
        config=Layer1ExecutionConfig(worker_mode="auto"),
        auto_tune_result=AutoTuneResult(
            selected_worker_count=1,
            candidates=(1,),
            samples=(),
        ),
    )

    assert automatic == sequential


def _sequential_probe(unit: WorkUnit) -> object:
    return unit.payload


def test_auto_selection_uses_median_memory_guard_and_smallest_two_percent_tie() -> None:
    timings = {
        1: iter((50.0, 1.01, 1.01, 1.01)),
        2: iter((50.0, 1.00, 1.00, 1.00)),
        4: iter((50.0, 0.80, 0.80, 0.80)),
    }
    memory_calls = {1: 0, 2: 0, 4: 0}

    def benchmark(worker_count: int) -> float:
        return next(timings[worker_count])

    def memory(worker_count: int) -> int:
        memory_calls[worker_count] += 1
        return 900 if worker_count == 4 else 500

    result = auto_select_worker_count(
        (4, 2, 1, 2),
        benchmark_probe=benchmark,
        memory_probe=memory,
        memory_limit_bytes=800,
    )

    assert result.candidates == (1, 2, 4)
    assert result.selected_worker_count == 1
    assert tuple(sample.memory_eligible for sample in result.samples) == (
        True,
        True,
        False,
    )
    assert memory_calls == {1: 4, 2: 4, 4: 4}


def test_auto_selection_explicit_override_skips_probes() -> None:
    def unexpected_benchmark(_worker_count: int) -> float:
        raise AssertionError("Explicit override must not benchmark")

    def unexpected_memory(_worker_count: int) -> int:
        raise AssertionError("Explicit override must not probe memory")

    result = auto_select_worker_count(
        (1, 2),
        benchmark_probe=unexpected_benchmark,
        memory_probe=unexpected_memory,
        memory_limit_bytes=1,
        explicit_worker_count=3,
    )

    assert result.selected_worker_count == 3
    assert result.explicit_override
    assert result.samples == ()


def test_candidate_discovery_respects_affinity_and_proportional_reserve() -> None:
    def unavailable_fallback() -> int:
        raise AssertionError("CPU fallback must not run when affinity is available")

    candidates = discover_worker_candidates(
        affinity_probe=lambda: {0, 2, 4, 6},
        cpu_count_probe=unavailable_fallback,
        reserve_fraction=0.25,
    )

    assert candidates == (1, 2, 3)


def test_resolve_worker_count_prefers_explicit_then_environment_override() -> None:
    auto = AutoTuneResult(selected_worker_count=2, candidates=(1, 2), samples=())

    assert (
        resolve_worker_count(
            Layer1ExecutionConfig(worker_mode="auto", worker_count=4),
            auto_tune_result=auto,
            environment={"ES_INDEX_EXPLORER_LAYER1_WORKERS": "3"},
        )
        == 4
    )
    assert (
        resolve_worker_count(
            Layer1ExecutionConfig(worker_mode="auto"),
            auto_tune_result=auto,
            environment={"ES_INDEX_EXPLORER_LAYER1_WORKERS": "3"},
        )
        == 3
    )


def test_progress_fields_and_outputs_exclude_substantive_data(tmp_path: Path) -> None:
    field_names = {field.name for field in fields(ProgressEvent)}
    substantive_fields = {
        "payload",
        "question",
        "document",
        "source_text",
        "counts",
        "secret",
        "error_message",
    }
    assert field_names.isdisjoint(substantive_fields)

    timestamps = iter((0.0, 1.0, 6.0))
    stderr = io.StringIO()
    path = tmp_path / "layer1-progress.jsonl"
    writer = RateLimitedProgressWriter(
        jsonl_path=path,
        stderr=stderr,
        interval_seconds=5.0,
        clock=lambda: next(timestamps),
    )
    assert writer.write(ProgressEvent(phase="fit", completed=1, total=3))
    assert not writer.write(ProgressEvent(phase="fit", completed=2, total=3))
    assert writer.write(ProgressEvent(phase="fit", completed=3, total=3))

    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["completed"] for record in records] == [1, 3]
    assert all(set(record).isdisjoint(substantive_fields) for record in records)
    assert "progress=1/3" in stderr.getvalue()


def test_terminal_progress_includes_optimizer_values() -> None:
    stderr = io.StringIO()
    writer = RateLimitedProgressWriter(
        stderr=stderr,
        interval_seconds=0,
    )

    writer.write(
        ProgressEvent(
            phase="primary_mml",
            work_unit_id="start:0.5",
            iteration=7,
            objective=12.5,
            gradient_maximum=0.001,
        )
    )

    rendered = stderr.getvalue()
    assert "unit=start:0.5" in rendered
    assert "iteration=7" in rendered
    assert "objective=12.5" in rendered
    assert "gradient_max=0.001" in rendered


def test_shutdown_token_records_first_signal_request() -> None:
    timestamps = iter((10.0, 20.0))
    token = ShutdownToken(clock=lambda: next(timestamps))

    with signal_shutdown_context(token):
        token.request(signal.SIGINT)
        token.request(signal.SIGTERM)

    assert token.requested
    assert token.interruption is not None
    assert token.interruption.signal_number == signal.SIGINT
    assert token.interruption.exit_code == 130
    assert token.interruption.requested_at == 10.0


def test_aggregate_rss_includes_current_process() -> None:
    assert aggregate_rss_bytes() > 0


@pytest.mark.parametrize(
    "factory",
    (
        lambda: Layer1ExecutionConfig(worker_mode="invalid"),  # ty: ignore[invalid-argument-type]
        lambda: Layer1ExecutionConfig(worker_mode="process", worker_count=0),
        lambda: Layer1ExecutionConfig(worker_mode="sequential", worker_count=2),
        lambda: Layer1ExecutionConfig(worker_mode="process", max_in_flight=0),
        lambda: Layer1ExecutionConfig(progress_interval_seconds=-1.0),
    ),
)
def test_execution_config_rejects_invalid_operational_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ProgressEvent(phase="x", iteration=-1),
        lambda: ProgressEvent(phase="x", elapsed_seconds=-1),
        lambda: ProgressEvent(phase="x", completed=2, total=1),
    ),
)
def test_progress_event_rejects_invalid_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    ("module", "function"),
    (("", "run"), ("bad-module!", "run"), ("module", "bad-name!")),
)
def test_task_reference_rejects_unsafe_names(
    module: str,
    function: str,
) -> None:
    with pytest.raises(ValueError):
        TaskReference(module=module, function=function)


def test_executor_rejects_duplicate_task_ids() -> None:
    task = TaskReference(module=__name__, function="_sequential_probe")
    units = (
        WorkUnit("duplicate", WorkUnitKind.START, 1),
        WorkUnit("duplicate", WorkUnitKind.START, 2),
    )

    with pytest.raises(ValueError, match="must be unique"):
        execute_work_units(task, units)


def test_executor_returns_typed_task_failure() -> None:
    result = execute_work_units(
        TaskReference(module=__name__, function="_failing_probe"),
        (WorkUnit("failure", WorkUnitKind.START),),
    )[0]

    assert not result.succeeded
    assert result.error_type == "RuntimeError"
    assert result.error_message == "controlled"


def _failing_probe(_unit: WorkUnit) -> object:
    raise RuntimeError("controlled")


def test_auto_selection_rejects_invalid_inputs_and_memory_exhaustion() -> None:
    with pytest.raises(ValueError, match="positive"):
        auto_select_worker_count(
            (0,),
            benchmark_probe=lambda _count: 1.0,
            memory_probe=lambda _count: 1,
            memory_limit_bytes=1,
        )
    with pytest.raises(ValueError, match="memory_limit"):
        auto_select_worker_count(
            (1,),
            benchmark_probe=lambda _count: 1.0,
            memory_probe=lambda _count: 1,
            memory_limit_bytes=0,
        )
    with pytest.raises(MemoryError):
        auto_select_worker_count(
            (1, 2),
            benchmark_probe=lambda _count: 1.0,
            memory_probe=lambda _count: 100,
            memory_limit_bytes=10,
        )


def test_resolve_worker_count_rejects_invalid_environment_and_missing_auto() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resolve_worker_count(
            Layer1ExecutionConfig(worker_mode="process"),
            environment={"ES_INDEX_EXPLORER_LAYER1_WORKERS": "bad"},
        )
    with pytest.raises(ValueError, match="positive"):
        resolve_worker_count(
            Layer1ExecutionConfig(worker_mode="process"),
            environment={"ES_INDEX_EXPLORER_LAYER1_WORKERS": "0"},
        )
    with pytest.raises(ValueError, match="auto-tune"):
        resolve_worker_count(
            Layer1ExecutionConfig(worker_mode="auto"),
            environment={},
        )
