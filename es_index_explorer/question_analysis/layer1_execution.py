"""Deterministic execution, worker selection, progress, and shutdown primitives."""

import importlib
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from types import FrameType
from typing import Literal, Self, TextIO, cast

THREAD_LIMIT_ENVIRONMENT_VARIABLES = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
WORKER_PROCESS_ENVIRONMENT_VARIABLE = "ES_INDEX_EXPLORER_LAYER1_WORKER"
WORKER_COUNT_ENVIRONMENT_VARIABLE = "ES_INDEX_EXPLORER_LAYER1_WORKERS"
AUTO_TUNE_REPETITIONS = 3
AUTO_TUNE_TIE_FRACTION = 0.02
DEFAULT_CPU_RESERVE_FRACTION = 0.10
DEFAULT_PROGRESS_INTERVAL_SECONDS = 5.0

WorkerMode = Literal["sequential", "process", "auto"]
TaskCallable = Callable[["WorkUnit"], object]
BenchmarkProbe = Callable[[int], float]
MemoryProbe = Callable[[int], int]
AffinityProbe = Callable[[], Collection[int] | None]
CpuCountProbe = Callable[[], int | None]


class WorkUnitKind(StrEnum):
    """Identify stable Layer 1 task families."""

    START = "start"
    MML_START = "start"
    OBJECTIVE_GRADIENT_PERTURBATION = "objective_gradient_perturbation"
    RUBRIC_CONTRIBUTION = "rubric_contribution"
    GLOBAL_OUTER_ATTEMPT = "global_outer_attempt"
    RUBRIC_CONDITIONAL_BLOCK = "rubric_conditional_block"
    PI_CONDITIONAL = "pi_conditional"
    IMPORTANCE_CHECK = "importance_check"
    LEAVE_OUT_REFIT = "leave_out_refit"


@dataclass(frozen=True, slots=True)
class TaskReference:
    """Reference a top-level task callable without importing its module."""

    module: str
    function: str

    def __post_init__(self) -> None:
        """Validate that the reference names an importable top-level attribute."""
        if not self.module or not all(
            part.isidentifier() for part in self.module.split(".")
        ):
            raise ValueError("Task module must be a dotted Python identifier")
        if not self.function.isidentifier():
            raise ValueError("Task function must be a top-level Python identifier")


@dataclass(frozen=True, slots=True)
class WorkUnit:
    """Store one stable, pickleable Layer 1 task request."""

    task_id: str
    kind: WorkUnitKind
    payload: object = None

    def __post_init__(self) -> None:
        """Reject empty task identities before scheduling."""
        if not self.task_id:
            raise ValueError("Work unit task_id must not be empty")


@dataclass(frozen=True, slots=True)
class WorkResult:
    """Store one stable, pickleable Layer 1 task outcome."""

    task_id: str
    kind: WorkUnitKind
    value: object = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the task callable completed successfully."""
        return self.error_type is None


@dataclass(frozen=True, slots=True)
class Layer1ExecutionConfig:
    """Configure Layer 1 scheduling without affecting scientific identities."""

    worker_mode: WorkerMode = "sequential"
    worker_count: int | None = None
    max_in_flight: int | None = None
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        """Validate operational scheduling values."""
        if self.worker_mode not in ("sequential", "process", "auto"):
            raise ValueError(f"Unsupported worker mode: {self.worker_mode}")
        if self.worker_count is not None and self.worker_count < 1:
            raise ValueError("worker_count must be positive")
        if self.worker_mode == "sequential" and self.worker_count not in (None, 1):
            raise ValueError("Sequential execution supports exactly one worker")
        if self.max_in_flight is not None and self.max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        if self.progress_interval_seconds < 0:
            raise ValueError("progress_interval_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class AutoTuneSample:
    """Store bounded timing and aggregate-memory observations for one candidate."""

    worker_count: int
    elapsed_seconds: tuple[float, float, float]
    median_elapsed_seconds: float
    peak_rss_bytes: int
    memory_eligible: bool


@dataclass(frozen=True, slots=True)
class AutoTuneResult:
    """Store deterministic worker auto-selection evidence."""

    selected_worker_count: int
    candidates: tuple[int, ...]
    samples: tuple[AutoTuneSample, ...]
    explicit_override: bool = False


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Store safe operational progress without source or count-row content."""

    phase: str
    work_unit_id: str | None = None
    start: int | None = None
    iteration: int | None = None
    objective: float | None = None
    gradient_maximum: float | None = None
    completed: int | None = None
    total: int | None = None
    retained_attempts: int | None = None
    failed_attempts: int | None = None
    elapsed_seconds: float | None = None
    throughput: float | None = None
    eta_seconds: float | None = None
    checkpoint_path: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate progress metadata before it reaches logs."""
        if not self.phase:
            raise ValueError("Progress phase must not be empty")
        for name in (
            "start",
            "iteration",
            "completed",
            "total",
            "retained_attempts",
            "failed_attempts",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"Progress field {name} must be non-negative")
        for name in ("elapsed_seconds", "throughput", "eta_seconds"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"Progress field {name} must be non-negative")
        if (
            self.completed is not None
            and self.total is not None
            and self.completed > self.total
        ):
            raise ValueError("Progress completed count cannot exceed total")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-safe progress mapping."""
        return {key: value for key, value in asdict(self).items() if value is not None}


class RateLimitedProgressWriter:
    """Write rate-limited progress to stderr and canonical JSONL."""

    def __init__(
        self,
        *,
        jsonl_path: Path | None = None,
        stderr: TextIO | None = None,
        interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize a progress writer.

        Parameters
        ----------
        jsonl_path
            Optional canonical JSONL destination.
        stderr
            Text stream used for concise terminal progress.
        interval_seconds
            Minimum interval between emitted events.
        clock
            Monotonic clock used to enforce the interval.
        """
        if interval_seconds < 0:
            raise ValueError("Progress interval must be non-negative")
        self._jsonl_path = jsonl_path
        self._stderr = stderr if stderr is not None else sys.stderr
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._last_emitted: float | None = None

    def write(self, event: ProgressEvent, *, force: bool = False) -> bool:
        """Write an event when its rate-limit interval has elapsed."""
        current = self._clock()
        if (
            not force
            and self._last_emitted is not None
            and current - self._last_emitted < self._interval_seconds
        ):
            return False
        payload = event.to_dict()
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._stderr.write(self._render_terminal(event))
        self._stderr.flush()
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(f"{encoded}\n")
        self._last_emitted = current
        return True

    @staticmethod
    def _render_terminal(event: ProgressEvent) -> str:
        parts = [f"layer1 phase={event.phase}"]
        if event.work_unit_id is not None:
            parts.append(f"unit={event.work_unit_id}")
        if event.start is not None:
            parts.append(f"start={event.start}")
        if event.iteration is not None:
            parts.append(f"iteration={event.iteration}")
        if event.objective is not None:
            parts.append(f"objective={event.objective:.10g}")
        if event.gradient_maximum is not None:
            parts.append(f"gradient_max={event.gradient_maximum:.6g}")
        if event.completed is not None and event.total is not None:
            parts.append(f"progress={event.completed}/{event.total}")
        if event.retained_attempts is not None:
            parts.append(f"retained={event.retained_attempts}")
        if event.failed_attempts is not None:
            parts.append(f"failed={event.failed_attempts}")
        if event.elapsed_seconds is not None:
            parts.append(f"elapsed={event.elapsed_seconds:.1f}s")
        if event.throughput is not None:
            parts.append(f"throughput={event.throughput:.3g}/s")
        if event.eta_seconds is not None:
            parts.append(f"eta={event.eta_seconds:.1f}s")
        return " ".join(parts) + "\n"


@dataclass(frozen=True, slots=True)
class InterruptedRequest:
    """Record the first graceful-shutdown request received by the process."""

    signal_number: int
    exit_code: int
    requested_at: float


class ShutdownToken:
    """Expose a thread-safe graceful-shutdown request to schedulers."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Initialize an unset shutdown token."""
        self._clock = clock
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._interruption: InterruptedRequest | None = None

    @property
    def requested(self) -> bool:
        """Return whether graceful shutdown has been requested."""
        return self._event.is_set()

    @property
    def interruption(self) -> InterruptedRequest | None:
        """Return the recorded first interruption request."""
        with self._lock:
            return self._interruption

    def request(self, signal_number: int) -> None:
        """Record the first signal requesting graceful shutdown."""
        with self._lock:
            if self._interruption is None:
                self._interruption = InterruptedRequest(
                    signal_number=signal_number,
                    exit_code=128 + signal_number,
                    requested_at=self._clock(),
                )
                self._event.set()


@contextmanager
def signal_shutdown_context(token: ShutdownToken) -> Iterator[ShutdownToken]:
    """Record SIGINT and SIGTERM requests while restoring prior handlers on exit."""

    def handle_signal(signal_number: int, _frame: FrameType | None) -> None:
        token.request(signal_number)

    previous_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for signal_number in previous_handlers:
            signal.signal(signal_number, handle_signal)
        yield token
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


def aggregate_rss_bytes(process_id: int | None = None) -> int:
    """Return aggregate RSS for one process and all recursive children."""
    import psutil

    try:
        process = (
            psutil.Process(process_id) if process_id is not None else psutil.Process()
        )
        processes = (process, *process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    total = 0
    for current in processes:
        try:
            total += current.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def discover_worker_candidates(
    *,
    affinity_probe: AffinityProbe | None = None,
    cpu_count_probe: CpuCountProbe = os.cpu_count,
    reserve_fraction: float = DEFAULT_CPU_RESERVE_FRACTION,
) -> tuple[int, ...]:
    """Discover a bounded, sorted worker set from process-available CPUs."""
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in [0, 1)")
    affinity = _probe_affinity(affinity_probe)
    cpu_count = len(affinity) if affinity else cpu_count_probe()
    available = max(1, cpu_count or 1)
    reserve = (
        min(available - 1, max(1, ceil(available * reserve_fraction)))
        if available > 1
        else 0
    )
    usable = max(1, available - reserve)
    candidates = {1, usable}
    power = 2
    while power < usable:
        candidates.add(power)
        power *= 2
    return tuple(sorted(candidates))


def auto_select_worker_count(
    candidates: Sequence[int],
    *,
    benchmark_probe: BenchmarkProbe,
    memory_probe: MemoryProbe,
    memory_limit_bytes: int,
    explicit_worker_count: int | None = None,
) -> AutoTuneResult:
    """Select the smallest memory-safe candidate within 2% of fastest."""
    normalized = tuple(sorted(set(candidates)))
    if not normalized or normalized[0] < 1:
        raise ValueError("Worker candidates must be positive")
    if memory_limit_bytes < 1:
        raise ValueError("memory_limit_bytes must be positive")
    if explicit_worker_count is not None:
        if explicit_worker_count < 1:
            raise ValueError("Explicit worker count must be positive")
        return AutoTuneResult(
            selected_worker_count=explicit_worker_count,
            candidates=tuple(sorted({*normalized, explicit_worker_count})),
            samples=(),
            explicit_override=True,
        )

    samples: list[AutoTuneSample] = []
    for worker_count in normalized:
        benchmark_probe(worker_count)
        memory_probe(worker_count)
        elapsed = tuple(
            benchmark_probe(worker_count) for _ in range(AUTO_TUNE_REPETITIONS)
        )
        if any(not isfinite(value) or value < 0 for value in elapsed):
            raise ValueError("Benchmark timings must be finite and non-negative")
        peak_rss = max(memory_probe(worker_count) for _ in range(AUTO_TUNE_REPETITIONS))
        if peak_rss < 0:
            raise ValueError("Memory observations must be non-negative")
        samples.append(
            AutoTuneSample(
                worker_count=worker_count,
                elapsed_seconds=cast(tuple[float, float, float], elapsed),
                median_elapsed_seconds=median(elapsed),
                peak_rss_bytes=peak_rss,
                memory_eligible=peak_rss <= memory_limit_bytes,
            )
        )
    eligible = tuple(sample for sample in samples if sample.memory_eligible)
    if not eligible:
        raise MemoryError("No worker candidate satisfies the aggregate RSS guard")
    fastest = min(sample.median_elapsed_seconds for sample in eligible)
    tied = tuple(
        sample.worker_count
        for sample in eligible
        if sample.median_elapsed_seconds <= fastest * (1.0 + AUTO_TUNE_TIE_FRACTION)
    )
    return AutoTuneResult(
        selected_worker_count=min(tied),
        candidates=normalized,
        samples=tuple(samples),
    )


def resolve_worker_count(
    config: Layer1ExecutionConfig,
    *,
    auto_tune_result: AutoTuneResult | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Resolve explicit, environment, auto-tuned, or discovered worker count."""
    if config.worker_mode == "sequential":
        return 1
    if config.worker_count is not None:
        return config.worker_count
    selected_environment = os.environ if environment is None else environment
    environment_value = selected_environment.get(WORKER_COUNT_ENVIRONMENT_VARIABLE)
    if environment_value and environment_value != "auto":
        try:
            override = int(environment_value)
        except ValueError as error:
            raise ValueError(
                f"{WORKER_COUNT_ENVIRONMENT_VARIABLE} must be 'auto' or a positive integer"
            ) from error
        if override < 1:
            raise ValueError(f"{WORKER_COUNT_ENVIRONMENT_VARIABLE} must be positive")
        return override
    if config.worker_mode == "auto":
        if auto_tune_result is None:
            raise ValueError("Auto execution requires an auto-tune result")
        return auto_tune_result.selected_worker_count
    return discover_worker_candidates()[-1]


class SequentialExecutor:
    """Execute work units serially in canonical submission order."""

    def execute(
        self,
        task: TaskReference,
        work_units: Sequence[WorkUnit],
        *,
        shutdown_token: ShutdownToken | None = None,
    ) -> tuple[WorkResult, ...]:
        """Execute work until completion or a graceful-shutdown request."""
        _validate_work_units(work_units)
        results: list[WorkResult] = []
        for unit in work_units:
            if shutdown_token is not None and shutdown_token.requested:
                break
            results.append(_execute_work_unit(task, unit))
        return tuple(results)


class SpawnProcessExecutor:
    """Execute work in spawn processes without completion-order leakage."""

    def __init__(self, worker_count: int, *, max_in_flight: int | None = None) -> None:
        """Initialize bounded process scheduling."""
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if max_in_flight is not None and max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        self._worker_count = worker_count
        self._max_in_flight = max_in_flight or worker_count * 2
        self._executor: ProcessPoolExecutor | None = None

    @property
    def worker_count(self) -> int:
        """Return the configured process count."""
        return self._worker_count

    @property
    def max_in_flight(self) -> int:
        """Return the bounded number of submitted tasks."""
        return self._max_in_flight

    def __enter__(self) -> Self:
        """Start one reusable spawn pool for repeated numerical task batches."""
        _set_worker_thread_limits()
        self._executor = ProcessPoolExecutor(
            max_workers=self._worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker_process,
        )
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        """Close the reusable process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def execute(
        self,
        task: TaskReference,
        work_units: Sequence[WorkUnit],
        *,
        shutdown_token: ShutdownToken | None = None,
    ) -> tuple[WorkResult, ...]:
        """Execute work and return outcomes in canonical submission order."""
        _validate_work_units(work_units)
        if _inside_worker_process():
            return SequentialExecutor().execute(
                task, work_units, shutdown_token=shutdown_token
            )
        results: dict[int, WorkResult] = {}
        next_index = 0
        owns_executor = self._executor is None
        if owns_executor:
            self.__enter__()
        executor = self._executor
        if executor is None:
            raise RuntimeError("Layer 1 process executor failed to initialize")
        try:
            pending: dict[Future[WorkResult], int] = {}
            while next_index < len(work_units) or pending:
                while (
                    next_index < len(work_units)
                    and len(pending) < self._max_in_flight
                    and not (shutdown_token is not None and shutdown_token.requested)
                ):
                    future = executor.submit(
                        _execute_work_unit, task, work_units[next_index]
                    )
                    pending[future] = next_index
                    next_index += 1
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = pending.pop(future)
                    results[index] = future.result()
                if shutdown_token is not None and shutdown_token.requested:
                    for future, index in tuple(pending.items()):
                        if future.cancel():
                            pending.pop(future)
                            next_index = min(next_index, index)
            return tuple(results[index] for index in sorted(results))
        finally:
            if owns_executor:
                self.__exit__(None, None, None)


def execute_work_units(
    task: TaskReference,
    work_units: Sequence[WorkUnit],
    *,
    config: Layer1ExecutionConfig | None = None,
    auto_tune_result: AutoTuneResult | None = None,
    shutdown_token: ShutdownToken | None = None,
) -> tuple[WorkResult, ...]:
    """Execute Layer 1 tasks with the configured deterministic backend."""
    selected_config = config or Layer1ExecutionConfig()
    worker_count = resolve_worker_count(
        selected_config, auto_tune_result=auto_tune_result
    )
    if selected_config.worker_mode == "sequential" or (
        selected_config.worker_mode == "auto" and worker_count == 1
    ):
        return SequentialExecutor().execute(
            task, work_units, shutdown_token=shutdown_token
        )
    return SpawnProcessExecutor(
        worker_count,
        max_in_flight=selected_config.max_in_flight,
    ).execute(task, work_units, shutdown_token=shutdown_token)


def _probe_affinity(affinity_probe: AffinityProbe | None) -> Collection[int] | None:
    if affinity_probe is not None:
        try:
            return affinity_probe()
        except (OSError, NotImplementedError):
            return None
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return None
    try:
        return cast(Collection[int], get_affinity(0))
    except (OSError, NotImplementedError):
        return None


def _set_worker_thread_limits() -> None:
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"


def _initialize_worker_process() -> None:
    _set_worker_thread_limits()
    os.environ[WORKER_PROCESS_ENVIRONMENT_VARIABLE] = "1"


def _inside_worker_process() -> bool:
    return os.environ.get(WORKER_PROCESS_ENVIRONMENT_VARIABLE) == "1"


def _load_task(reference: TaskReference) -> TaskCallable:
    module = importlib.import_module(reference.module)
    candidate = getattr(module, reference.function, None)
    if not callable(candidate):
        raise TypeError(
            f"Task reference {reference.module}.{reference.function} is not callable"
        )
    return cast(TaskCallable, candidate)


def _execute_work_unit(reference: TaskReference, unit: WorkUnit) -> WorkResult:
    _set_worker_thread_limits()
    try:
        value = _load_task(reference)(unit)
    except Exception as error:  # noqa: BLE001 - task failures are stable executor outcomes.
        return WorkResult(
            task_id=unit.task_id,
            kind=unit.kind,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    return WorkResult(task_id=unit.task_id, kind=unit.kind, value=value)


def _validate_work_units(work_units: Sequence[WorkUnit]) -> None:
    task_ids = tuple(unit.task_id for unit in work_units)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Work unit task IDs must be unique")
