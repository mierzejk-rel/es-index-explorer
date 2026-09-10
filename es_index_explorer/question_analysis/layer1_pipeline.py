"""Gated Layer 1 fitting, artifact construction, and persistence."""

import json
import platform
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from es_index_explorer.question_analysis.contracts import (
    RUBRIC_RECOMMENDATION_COLUMNS,
    VARIANT_RECOMMENDATION_COLUMNS,
    FailureKind,
    Layer1EventKind,
    SuitabilityTier,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    AnalysisError,
    GateFailureError,
    Layer1ModeError,
    Layer1ReplenishmentExhaustedError,
    MalformedInputError,
    NumericalError,
)
from es_index_explorer.question_analysis.layer1_checkpoint import (
    Layer1CheckpointCompatibilityIdentity,
    Layer1CheckpointSession,
    Layer1ExecutionProvenance,
    Layer1TaskFailure,
    Layer1TaskLedgerEntry,
    Layer1TaskStatus,
)
from es_index_explorer.question_analysis.layer1_decisions import (
    TIER_FLOORS,
    PrimaryEvent,
    build_primary_events,
    layer1_leave_out_datasets,
    leave_out_tier_changes,
    pooled_mean_tier,
    proportion_diagnostic,
    score_band_probabilities,
    shrunken_variant_means,
    tier_for_events,
    tier_from_probabilities,
    uncertain_flag,
)
from es_index_explorer.question_analysis.layer1_execution import (
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
    resolve_worker_count,
    signal_shutdown_context,
)
from es_index_explorer.question_analysis.layer1_laplace import (
    CONDITIONAL_DRAW_COUNT,
    IMPORTANCE_PARTICLES,
    INNER_DRAWS_PER_OUTER,
    ImportanceDiagnostic,
    RubricDrawBlock,
    conditional_mode,
    draw_rubric_v2,
    importance_resampling_diagnostic,
    make_rubric_draw_block,
    select_importance_rubrics,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MmlFit,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1StartFit,
    Layer1VariantData,
    MarginalEvaluationPayload,
    ModelTaskExecutor,
    evaluate_marginal_work_unit,
    fit_layer1_mml,
    method_of_moments_start,
    pack_hyperparameters,
    unpack_hyperparameters,
)
from es_index_explorer.question_analysis.layer1_packed import compress_pfu_counts
from es_index_explorer.question_analysis.layer1_propagation import (
    ADAPTIVE_DEPTHS,
    ELEVATED_FAILURE_RATE,
    MAXIMUM_REPLENISHMENT_MULTIPLIER,
    PREFIX_DIAGNOSTIC_DEPTHS,
    OuterHyperparameterDraw,
    RubricPropagation,
    simulate_layer1_dataset,
)
from es_index_explorer.question_analysis.seeds import derive_child_seed
from es_index_explorer.question_analysis.storage import versioned_frame
from es_index_explorer.question_analysis.workspace import (
    AnalysisWorkspace,
    StepExecutionResult,
)

LAYER1_FIT_ARTIFACT = "statistics/layer1_fit.json"
LAYER1_HYPERPARAMETER_DIAGNOSTICS = "statistics/layer1_hyperparameter_diagnostics.json"
LAYER1_FAILURE_ARTIFACT = "statistics/layer1_numerical_failures.json"
LAYER1_IMPORTANCE_ARTIFACT = "statistics/layer1_importance_diagnostics.parquet"
LAYER1_EVENT_ARTIFACT = "statistics/layer1_primary_events.parquet"
LAYER1_PREFIX_ARTIFACT = "statistics/layer1_prefix_diagnostics.parquet"
LAYER1_STABILITY_ARTIFACT = "statistics/layer1_stability.parquet"
RUBRIC_RECOMMENDATION_ARTIFACT = "tables/recommendation_table_rubric.parquet"
VARIANT_RECOMMENDATION_ARTIFACT = "tables/recommendation_table_variant.parquet"
LAYER1_REPORT = "partial_reports/05-suitability-estimates.md"
FLOOR_SUFFIX = {0.75: "0_75", 0.60: "0_60", 0.50: "0_50"}

Layer1Refitter = Callable[[Layer1Dataset], Layer1MmlFit]
Layer1ProgressCallback = Callable[[ProgressEvent], None]
OuterAttemptCallback = Callable[[int, Layer1Hyperparameters | None, str | None], None]
RubricBlockCallback = Callable[[str, int, RubricDrawBlock | None, str | None], None]
ConditionalCallback = Callable[[str, np.ndarray], None]
ImportanceCallback = Callable[[ImportanceDiagnostic], None]
LeaveOutCallback = Callable[[str, Mapping[str, SuitabilityTier]], None]


@dataclass(frozen=True, slots=True)
class Layer1RunConfig:
    """Configure frozen production counts with smaller explicit test fixtures."""

    mandatory_prefix_depth: int = 1_000
    prefix_diagnostic_depths: tuple[int, ...] = PREFIX_DIAGNOSTIC_DEPTHS
    adaptive_depths: tuple[int, ...] = ADAPTIVE_DEPTHS
    inner_draw_count: int = INNER_DRAWS_PER_OUTER
    conditional_draw_count: int = CONDITIONAL_DRAW_COUNT
    importance_particles: int = IMPORTANCE_PARTICLES
    maximum_depth: int = 4_000
    run_leave_out_stability: bool = True


@dataclass(frozen=True, slots=True)
class Layer1Execution:
    """Store all in-memory artifacts produced by a complete Layer 1 execution."""

    fit: Mapping[str, object]
    hyperparameter_diagnostics: Mapping[str, object]
    numerical_failures: Mapping[str, object]
    importance: pd.DataFrame
    events: pd.DataFrame
    prefixes: pd.DataFrame
    stability: pd.DataFrame
    draw_frames: Mapping[str, pd.DataFrame]
    rubric_recommendations: pd.DataFrame
    variant_recommendations: pd.DataFrame
    report: str


class _HyperparametersCheckpoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    phi: float
    mu0: list[float]
    dataset_offsets: dict[str, list[float]]
    sigma_within: list[list[float]]
    sigma_between: list[list[float]]


class _StartCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phi_multiplier: float
    vector: list[float]
    hyperparameters: _HyperparametersCheckpoint | None = None
    objective: float
    gradient_maximum: float
    iterations: int
    converged: bool
    message: str


class _FitCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    objective: float
    vector: list[float]
    hyperparameters: _HyperparametersCheckpoint
    starts: list[_StartCheckpoint]


class _BlockCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    variant_ids: list[str]
    rubric_retained_index: int


class _ImportanceCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric_id: str
    particle_count: int
    effective_sample_size: float
    effective_sample_size_ratio: float
    adequate: bool
    laplace_mean: list[float]
    importance_mean: list[float]
    laplace_covariance: list[list[float]]
    importance_covariance: list[list[float]]


class _LeaveOutCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_id: str
    tiers: dict[str, str]


@dataclass(frozen=True, slots=True)
class OuterAttemptPayload:
    """Store one spawn-safe parametric-bootstrap outer attempt."""

    data: Layer1Dataset
    fitted: Layer1Hyperparameters
    global_outer_attempt_id: int


@dataclass(frozen=True, slots=True)
class ConditionalBlockPayload:
    """Store one spawn-safe rubric conditional block request."""

    rubric: Layer1RubricData
    hyperparameters: Layer1Hyperparameters
    moments: Layer1MomentStart
    global_outer_attempt_id: int
    draw_count: int


@dataclass(frozen=True, slots=True)
class ConditionalBlockValue:
    """Return a conditional block before parent-ordered retained indexing."""

    rubric_id: str
    variant_ids: tuple[str, ...]
    global_outer_attempt_id: int
    rubric_v2_draws: np.ndarray


@dataclass(frozen=True, slots=True)
class LeaveOutRefitPayload:
    """Store one spawn-safe full leave-out Layer 1 refit."""

    comparison_id: str
    data: Layer1Dataset
    config: Layer1RunConfig


def execute_outer_attempt_work_unit(unit: WorkUnit) -> object:
    """Simulate and fully refit one independent global outer attempt."""
    if not isinstance(unit.payload, OuterAttemptPayload):
        raise MalformedInputError("Outer-attempt work unit payload is invalid")
    payload = unit.payload
    rng = np.random.default_rng(
        derive_child_seed(
            "layer1_bootstrap",
            f"outer:{payload.global_outer_attempt_id}",
        )
    )
    simulated = simulate_layer1_dataset(payload.data, payload.fitted, rng=rng)
    return fit_layer1_mml(simulated).hyperparameters


def execute_conditional_block_work_unit(unit: WorkUnit) -> object:
    """Compute one rubric conditional block in a spawn-safe worker."""
    if not isinstance(unit.payload, ConditionalBlockPayload):
        raise MalformedInputError("Conditional-block work unit payload is invalid")
    payload = unit.payload
    rng = np.random.default_rng(
        derive_child_seed(
            "laplace_draws",
            f"{payload.rubric.rubric_id}:outer:{payload.global_outer_attempt_id}",
        )
    )
    mode = conditional_mode(
        payload.rubric,
        payload.hyperparameters,
        payload.moments,
    )
    return ConditionalBlockValue(
        rubric_id=payload.rubric.rubric_id,
        variant_ids=tuple(variant.variant_id for variant in payload.rubric.variants),
        global_outer_attempt_id=payload.global_outer_attempt_id,
        rubric_v2_draws=draw_rubric_v2(
            payload.rubric,
            mode,
            rng=rng,
            draw_count=payload.draw_count,
        ),
    )


def execute_leave_out_work_unit(unit: WorkUnit) -> object:
    """Run one complete leave-out fit without nested leave-out recursion."""
    if not isinstance(unit.payload, LeaveOutRefitPayload):
        raise MalformedInputError("Leave-out work unit payload is invalid")
    payload = unit.payload
    execution = _execute_layer1(
        payload.data,
        config=replace(payload.config, run_leave_out_stability=False),
    )
    return {
        str(row["rubric_id"]): SuitabilityTier(str(row["tier"]))
        for row in execution.rubric_recommendations.to_dict(orient="records")
    }


def _checkpoint_identity(
    workspace: AnalysisWorkspace,
    run_config: Layer1RunConfig,
) -> Layer1CheckpointCompatibilityIdentity:
    manifest = workspace.load_manifest()
    state = workspace.load_state()
    required_inputs = (
        "tables/trace_pfu_table.parquet",
        "statistics/r_oracle_verification.json",
        "validation_unlock.json",
    )
    missing = [name for name in required_inputs if name not in state.artifacts]
    if missing:
        raise GateFailureError(
            f"Layer 1 checkpoint identity is missing artifacts: {missing}"
        )
    numpy_configuration = getattr(np.__config__, "CONFIG", {})
    return Layer1CheckpointCompatibilityIdentity(
        specification_sha256=manifest.specification.sha256,
        input_sha256s={name: state.artifacts[name].sha256 for name in required_inputs},
        resource_sha256s={
            name: fingerprint.sha256
            for name, fingerprint in sorted(manifest.resources.items())
        },
        numerical_contract_version="layer1-v1",
        statistical_run_config={
            "mandatory_prefix_depth": run_config.mandatory_prefix_depth,
            "prefix_diagnostic_depths": list(run_config.prefix_diagnostic_depths),
            "adaptive_depths": list(run_config.adaptive_depths),
            "inner_draw_count": run_config.inner_draw_count,
            "conditional_draw_count": run_config.conditional_draw_count,
            "importance_particles": run_config.importance_particles,
            "maximum_depth": run_config.maximum_depth,
            "run_leave_out_stability": run_config.run_leave_out_stability,
        },
        seed_contract={
            name: manifest.stream_seeds[name]
            for name in (
                "layer1_bootstrap",
                "laplace_draws",
                "diagnostic_resampling",
            )
        },
        numerical_backend="adaptive-finite-difference-v1",
        architecture=platform.machine(),
        float_abi=f"{np.dtype(np.float64).str}:{sys.byteorder}",
        python_version=platform.python_version(),
        numpy_version=version("numpy"),
        scipy_version=version("scipy"),
        blas_lapack_identity=json.dumps(
            numpy_configuration,
            sort_keys=True,
            default=str,
        ),
    )


def _execution_provenance(
    execution_config: Layer1ExecutionConfig,
    worker_count: int,
    auto_tune: AutoTuneResult | None,
) -> Layer1ExecutionProvenance:
    return Layer1ExecutionProvenance(
        cpu_model=platform.processor() or platform.machine(),
        os_details=platform.platform(),
        executor_type=execution_config.worker_mode,
        worker_count=worker_count,
        max_in_flight=execution_config.max_in_flight or max(1, 2 * worker_count),
        progress_interval_seconds=max(execution_config.progress_interval_seconds, 1e-9),
        auto_tuning_samples=(
            {
                "selected_worker_count": auto_tune.selected_worker_count,
                "candidates": list(auto_tune.candidates),
                "samples": [
                    {
                        "worker_count": sample.worker_count,
                        "elapsed_seconds": list(sample.elapsed_seconds),
                        "median_elapsed_seconds": sample.median_elapsed_seconds,
                        "peak_rss_bytes": sample.peak_rss_bytes,
                        "memory_eligible": sample.memory_eligible,
                    }
                    for sample in auto_tune.samples
                ],
            }
            if auto_tune is not None
            else {}
        ),
    )


def _start_payload(start: Layer1StartFit) -> dict[str, object]:
    return {
        "phi_multiplier": start.phi_multiplier,
        "vector": start.vector.tolist(),
        "hyperparameters": _hyperparameters_payload(start.hyperparameters),
        "objective": start.objective,
        "gradient_maximum": start.gradient_maximum,
        "iterations": start.iterations,
        "converged": start.converged,
        "message": start.message,
    }


def _hyperparameters_from_payload(
    payload: Mapping[str, object],
) -> Layer1Hyperparameters:
    validated = _HyperparametersCheckpoint.model_validate(payload)
    offsets = {
        str(name): np.asarray(value, dtype=float)
        for name, value in validated.dataset_offsets.items()
    }
    return Layer1Hyperparameters(
        phi=validated.phi,
        mu0=np.asarray(validated.mu0, dtype=float),
        dataset_offsets=offsets,
        sigma_within=np.asarray(validated.sigma_within, dtype=float),
        sigma_between=np.asarray(validated.sigma_between, dtype=float),
    )


def _start_from_payload(payload: Mapping[str, object]) -> Layer1StartFit:
    validated = _StartCheckpoint.model_validate(payload)
    if validated.hyperparameters is None:
        raise MalformedInputError("Primary start checkpoint is missing hyperparameters")
    return Layer1StartFit(
        phi_multiplier=validated.phi_multiplier,
        vector=np.asarray(validated.vector, dtype=float),
        hyperparameters=_hyperparameters_from_payload(
            validated.hyperparameters.model_dump()
        ),
        objective=validated.objective,
        gradient_maximum=validated.gradient_maximum,
        iterations=validated.iterations,
        converged=validated.converged,
        message=validated.message,
    )


def _load_propagation_checkpoint(
    checkpoint: Layer1CheckpointSession,
    data: Layer1Dataset,
) -> tuple[
    tuple[OuterHyperparameterDraw, ...],
    dict[str, tuple[RubricDrawBlock, ...]],
    dict[str, int],
    int,
    dict[str, int],
    int,
]:
    outer_draws: list[OuterHyperparameterDraw] = []
    blocks: dict[str, list[RubricDrawBlock]] = {
        rubric.rubric_id: [] for rubric in data.rubrics
    }
    processed = {rubric.rubric_id: 0 for rubric in data.rubrics}
    local_failures = {rubric.rubric_id: 0 for rubric in data.rubrics}
    global_failures = 0
    attempt_id = 0
    for entry in sorted(
        checkpoint.manifest.task_ledger,
        key=lambda current: (current.admission_group, current.admission_index),
    ):
        if entry.admission_group == "global-outer":
            attempt_id = max(attempt_id, entry.admission_index)
            if entry.status is Layer1TaskStatus.REJECTED:
                global_failures += 1
                continue
            payload = checkpoint.read_json_state(
                f"outer/attempt_{entry.admission_index:06d}.json"
            )
            if payload is None:
                raise MalformedInputError(
                    f"Missing retained outer checkpoint {entry.task_id}"
                )
            outer_draws.append(
                OuterHyperparameterDraw(
                    global_outer_attempt_id=entry.admission_index,
                    hyperparameters=_hyperparameters_from_payload(payload),
                )
            )
            continue
        if not entry.admission_group.startswith("rubric-block:"):
            continue
        rubric_id = entry.admission_group.removeprefix("rubric-block:")
        if rubric_id not in blocks:
            raise MalformedInputError(f"Checkpoint contains unknown rubric {rubric_id}")
        processed[rubric_id] += 1
        if entry.status is Layer1TaskStatus.REJECTED:
            local_failures[rubric_id] += 1
            continue
        safe_id = _safe_rubric_filename(rubric_id)
        metadata = checkpoint.read_json_state(
            f"blocks/{safe_id}/outer_{entry.admission_index:06d}.json"
        )
        values = checkpoint.read_npy_state(
            f"blocks/{safe_id}/outer_{entry.admission_index:06d}.npy"
        )
        if metadata is None or values is None:
            raise MalformedInputError(
                f"Missing retained rubric-block checkpoint {entry.task_id}"
            )
        validated_metadata = _BlockCheckpoint.model_validate(metadata)
        variant_ids = tuple(validated_metadata.variant_ids)
        blocks[rubric_id].append(
            RubricDrawBlock(
                rubric_id=rubric_id,
                variant_ids=variant_ids,
                global_outer_attempt_id=entry.admission_index,
                rubric_retained_index=validated_metadata.rubric_retained_index,
                rubric_v2_draws=values,
            )
        )
    return (
        tuple(
            sorted(
                outer_draws,
                key=lambda current: current.global_outer_attempt_id,
            )
        ),
        {
            rubric_id: tuple(
                sorted(
                    values,
                    key=lambda current: current.global_outer_attempt_id,
                )
            )
            for rubric_id, values in blocks.items()
        },
        processed,
        global_failures,
        local_failures,
        attempt_id,
    )


def _importance_payload(
    diagnostic: ImportanceDiagnostic,
) -> dict[str, object]:
    return {
        "rubric_id": diagnostic.rubric_id,
        "particle_count": diagnostic.particle_count,
        "effective_sample_size": diagnostic.effective_sample_size,
        "effective_sample_size_ratio": diagnostic.effective_sample_size_ratio,
        "adequate": diagnostic.adequate,
        "laplace_mean": diagnostic.laplace_mean.tolist(),
        "importance_mean": diagnostic.importance_mean.tolist(),
        "laplace_covariance": diagnostic.laplace_covariance.tolist(),
        "importance_covariance": diagnostic.importance_covariance.tolist(),
    }


def _load_diagnostic_checkpoint(
    checkpoint: Layer1CheckpointSession,
    data: Layer1Dataset,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, ImportanceDiagnostic],
    dict[str, Mapping[str, SuitabilityTier]],
]:
    conditional: dict[str, np.ndarray] = {}
    importance: dict[str, ImportanceDiagnostic] = {}
    leave_outs: dict[str, Mapping[str, SuitabilityTier]] = {}
    for rubric in data.rubrics:
        safe_id = _safe_rubric_filename(rubric.rubric_id)
        values = checkpoint.read_npy_state(f"diagnostics/conditional_{safe_id}.npy")
        if values is not None:
            conditional[rubric.rubric_id] = values
        payload = checkpoint.read_json_state(f"diagnostics/importance_{safe_id}.json")
        if payload is not None:
            validated = _ImportanceCheckpoint.model_validate(payload)
            importance[rubric.rubric_id] = ImportanceDiagnostic(
                rubric_id=validated.rubric_id,
                particle_count=validated.particle_count,
                effective_sample_size=validated.effective_sample_size,
                effective_sample_size_ratio=validated.effective_sample_size_ratio,
                adequate=validated.adequate,
                laplace_mean=np.asarray(validated.laplace_mean, dtype=float),
                importance_mean=np.asarray(validated.importance_mean, dtype=float),
                laplace_covariance=np.asarray(
                    validated.laplace_covariance, dtype=float
                ),
                importance_covariance=np.asarray(
                    validated.importance_covariance, dtype=float
                ),
            )
    for shard in checkpoint.manifest.scientific_state_shards:
        if not shard.path.startswith("state/leave_out/") or not shard.path.endswith(
            ".json"
        ):
            continue
        relative_path = shard.path.removeprefix("state/")
        payload = checkpoint.read_json_state(relative_path)
        if payload is None:
            continue
        tiers = _LeaveOutCheckpoint.model_validate(payload)
        leave_outs[tiers.comparison_id] = {
            rubric_id: SuitabilityTier(tier) for rubric_id, tier in tiers.tiers.items()
        }
    return conditional, importance, leave_outs


def _auto_tune_model_executor(
    data: Layer1Dataset,
    moments: Layer1MomentStart,
    execution_config: Layer1ExecutionConfig,
) -> AutoTuneResult:
    import psutil

    candidates = discover_worker_candidates()
    subset = Layer1Dataset(
        rubrics=data.rubrics[: min(24, len(data.rubrics))],
        dataset_levels=data.dataset_levels,
    )
    vector = pack_hyperparameters(moments.hyperparameters, data.dataset_levels)
    reference = TaskReference(
        module=evaluate_marginal_work_unit.__module__,
        function=evaluate_marginal_work_unit.__name__,
    )
    units = tuple(
        WorkUnit(
            task_id=f"autotune:{index}:{side}",
            kind=WorkUnitKind.OBJECTIVE_GRADIENT_PERTURBATION,
            payload=MarginalEvaluationPayload(
                data=subset,
                latent_starts=moments.rubric_latent_starts,
                vector=vector
                + np.eye(len(vector))[index]
                * sign
                * (1e-5 * max(1.0, abs(float(vector[index])))),
            ),
        )
        for index in range(len(vector))
        for side, sign in (("plus", 1.0), ("minus", -1.0))
    )

    def benchmark(worker_count: int) -> float:
        started = time.perf_counter()
        with SpawnProcessExecutor(
            worker_count,
            max_in_flight=execution_config.max_in_flight,
        ) as executor:
            results = executor.execute(reference, units)
        if not all(result.succeeded for result in results):
            return float("inf")
        return time.perf_counter() - started

    parent_rss = max(1, aggregate_rss_bytes())

    def memory(worker_count: int) -> int:
        return parent_rss * (worker_count + 1)

    return auto_select_worker_count(
        candidates,
        benchmark_probe=benchmark,
        memory_probe=memory,
        memory_limit_bytes=max(
            parent_rss,
            int(psutil.virtual_memory().available * 0.80),
        ),
        explicit_worker_count=execution_config.worker_count,
    )


def load_layer1_dataset(workspace: AnalysisWorkspace) -> Layer1Dataset:
    """Load the eligible PFU population into deterministic hierarchy objects."""
    path = workspace.store.path_for("tables/trace_pfu_table.parquet")
    if not path.is_file():
        raise MalformedInputError("Layer 1 requires trace_pfu_table.parquet")
    frame = pd.read_parquet(path)
    required = {
        "rubric_id",
        "rubric_order",
        "variant_id",
        "variant_index",
        "eval_dataset",
        "arm_id",
        "stage",
        "P",
        "F",
        "U",
        "N_r",
        "eligible",
    }
    if not required <= set(frame.columns):
        raise MalformedInputError("Layer 1 PFU table is missing required columns")
    eligible = frame.loc[frame["eligible"].astype(bool)].copy()
    rubrics: list[Layer1RubricData] = []
    grouped = eligible.sort_values(
        ["rubric_order", "rubric_id", "variant_index", "arm_id"]
    ).groupby(
        ["rubric_id", "rubric_order", "eval_dataset", "N_r"],
        sort=False,
        dropna=False,
    )
    for (rubric_id, rubric_order, dataset, expectation_count), rubric_frame in grouped:
        variants: list[Layer1VariantData] = []
        for variant_id, variant_frame in rubric_frame.groupby(
            "variant_id", sort=False, dropna=False
        ):
            counts = variant_frame[["P", "F", "U"]].to_numpy(dtype=int)
            variants.append(
                Layer1VariantData(
                    variant_id=str(variant_id),
                    counts=counts,
                    arm_ids=tuple(variant_frame["arm_id"].astype(str)),
                    stages=tuple(variant_frame["stage"].astype(str)),
                    compressed_counts=compress_pfu_counts(counts),
                )
            )
        rubrics.append(
            Layer1RubricData(
                rubric_id=str(rubric_id),
                rubric_order=int(rubric_order),
                dataset=str(dataset),
                expectation_count=int(expectation_count),
                variants=tuple(variants),
            )
        )
    return Layer1Dataset(
        rubrics=tuple(rubrics),
        dataset_levels=tuple(sorted(eligible["eval_dataset"].astype(str).unique())),
    )


def _hyperparameters_payload(
    hyperparameters: Layer1Hyperparameters,
) -> Mapping[str, object]:
    return {
        "phi": hyperparameters.phi,
        "mu0": hyperparameters.mu0.tolist(),
        "dataset_offsets": {
            dataset: value.tolist()
            for dataset, value in sorted(hyperparameters.dataset_offsets.items())
        },
        "sigma_within": hyperparameters.sigma_within.tolist(),
        "sigma_between": hyperparameters.sigma_between.tolist(),
        "trace_sigma_within": float(np.trace(hyperparameters.sigma_within)),
        "trace_sigma_between": float(np.trace(hyperparameters.sigma_between)),
        "eigenvalues_sigma_within": np.linalg.eigvalsh(
            hyperparameters.sigma_within
        ).tolist(),
        "eigenvalues_sigma_between": np.linalg.eigvalsh(
            hyperparameters.sigma_between
        ).tolist(),
    }


def _fit_payload(fit: Layer1MmlFit) -> Mapping[str, object]:
    return {
        "schema_version": 1,
        "objective": fit.objective,
        "vector": fit.vector.tolist(),
        "hyperparameters": _hyperparameters_payload(fit.hyperparameters),
        "starts": [
            {
                "phi_multiplier": start.phi_multiplier,
                "objective": start.objective,
                "gradient_maximum": start.gradient_maximum,
                "iterations": start.iterations,
                "converged": start.converged,
                "message": start.message,
                "vector": start.vector.tolist(),
            }
            for start in fit.starts
        ],
    }


def _fit_from_payload(
    payload: Mapping[str, object],
    dataset_levels: tuple[str, ...],
) -> Layer1MmlFit:
    validated = _FitCheckpoint.model_validate(payload)
    starts = tuple(
        Layer1StartFit(
            phi_multiplier=start.phi_multiplier,
            vector=np.asarray(start.vector, dtype=float),
            hyperparameters=_hyperparameters_from_payload(
                start.hyperparameters.model_dump()
                if start.hyperparameters is not None
                else _hyperparameters_payload(
                    unpack_hyperparameters(
                        np.asarray(start.vector, dtype=float),
                        dataset_levels,
                    )
                )
            ),
            objective=start.objective,
            gradient_maximum=start.gradient_maximum,
            iterations=start.iterations,
            converged=start.converged,
            message=start.message,
        )
        for start in validated.starts
    )
    return Layer1MmlFit(
        hyperparameters=_hyperparameters_from_payload(
            validated.hyperparameters.model_dump()
        ),
        vector=np.asarray(validated.vector, dtype=float),
        objective=validated.objective,
        starts=starts,
    )


def _conditional_draws(
    rubric: Layer1RubricData,
    fit: Layer1MmlFit,
    moments: Layer1MomentStart,
    draw_count: int,
) -> np.ndarray:
    mode = conditional_mode(rubric, fit.hyperparameters, moments)
    rng = np.random.default_rng(
        derive_child_seed("laplace_draws", f"{rubric.rubric_id}:conditional")
    )
    return draw_rubric_v2(rubric, mode, rng=rng, draw_count=draw_count)


def _conditional_probability_fields(
    values: np.ndarray,
    propagated_probabilities: Mapping[float, float],
    *,
    minimum: bool,
) -> Mapping[str, float]:
    fields: dict[str, float] = {}
    infix = "_min" if minimum else ""
    for floor in TIER_FLOORS:
        suffix = FLOOR_SUFFIX[floor]
        probability = float(np.mean(values >= floor))
        mcse = float(np.sqrt(probability * (1.0 - probability) / len(values)))
        fields[f"pi_cond{infix}_ge_{suffix}"] = probability
        fields[f"mcse_cond{infix}_ge_{suffix}"] = mcse
        fields[f"pi_prop_cond_gap_{suffix}"] = (
            propagated_probabilities[floor] - probability
        )
    return fields


def _execute_layer1(
    data: Layer1Dataset,
    *,
    config: Layer1RunConfig | None = None,
    refitter: Layer1Refitter = fit_layer1_mml,
    model_executor: ModelTaskExecutor | None = None,
    moments_override: Layer1MomentStart | None = None,
    fit_override: Layer1MmlFit | None = None,
    progress_callback: Layer1ProgressCallback | None = None,
    outer_attempt_callback: OuterAttemptCallback | None = None,
    rubric_block_callback: RubricBlockCallback | None = None,
    preloaded_outer_draws: tuple[OuterHyperparameterDraw, ...] = (),
    preloaded_blocks: Mapping[str, tuple[RubricDrawBlock, ...]] | None = None,
    preloaded_processed_outer: Mapping[str, int] | None = None,
    preloaded_global_failures: int = 0,
    preloaded_local_failures: Mapping[str, int] | None = None,
    preloaded_attempt_id: int = 0,
    preloaded_conditional: Mapping[str, np.ndarray] | None = None,
    preloaded_importance: Mapping[str, ImportanceDiagnostic] | None = None,
    preloaded_leave_outs: Mapping[str, Mapping[str, SuitabilityTier]] | None = None,
    conditional_callback: ConditionalCallback | None = None,
    importance_callback: ImportanceCallback | None = None,
    leave_out_callback: LeaveOutCallback | None = None,
) -> Layer1Execution:
    selected_config = config or Layer1RunConfig()
    if (
        not selected_config.prefix_diagnostic_depths
        or not selected_config.adaptive_depths
        or tuple(sorted(selected_config.prefix_diagnostic_depths))
        != selected_config.prefix_diagnostic_depths
        or tuple(sorted(selected_config.adaptive_depths))
        != selected_config.adaptive_depths
        or selected_config.mandatory_prefix_depth
        < max(selected_config.prefix_diagnostic_depths)
        or selected_config.mandatory_prefix_depth < selected_config.adaptive_depths[0]
        or selected_config.maximum_depth != selected_config.adaptive_depths[-1]
    ):
        raise MalformedInputError("Layer 1 run depths are inconsistent")
    moments = moments_override or method_of_moments_start(data)
    fit = fit_override or (
        fit_layer1_mml(data, moments, executor=model_executor)
        if refitter is fit_layer1_mml
        else refitter(data)
    )
    rubric_by_id = {rubric.rubric_id: rubric for rubric in data.rubrics}
    blocks: dict[str, list[RubricDrawBlock]] = {
        rubric.rubric_id: list((preloaded_blocks or {}).get(rubric.rubric_id, ()))
        for rubric in data.rubrics
    }
    local_failures = {
        rubric.rubric_id: (preloaded_local_failures or {}).get(rubric.rubric_id, 0)
        for rubric in data.rubrics
    }
    processed_outer_draws = {
        rubric.rubric_id: (preloaded_processed_outer or {}).get(rubric.rubric_id, 0)
        for rubric in data.rubrics
    }
    outer_draws: list[OuterHyperparameterDraw] = list(preloaded_outer_draws)
    global_failures = preloaded_global_failures
    target_depths = {
        rubric.rubric_id: selected_config.mandatory_prefix_depth
        for rubric in data.rubrics
    }
    cap = MAXIMUM_REPLENISHMENT_MULTIPLIER * selected_config.maximum_depth
    attempt_id = preloaded_attempt_id

    def process_available_outer_draws(rubric_id: str) -> None:
        rubric = rubric_by_id[rubric_id]
        target = target_depths[rubric_id]
        while len(blocks[rubric_id]) < target and processed_outer_draws[
            rubric_id
        ] < len(outer_draws):
            outer = outer_draws[processed_outer_draws[rubric_id]]
            processed_outer_draws[rubric_id] += 1
            inner_rng = np.random.default_rng(
                derive_child_seed(
                    "laplace_draws",
                    f"{rubric_id}:outer:{outer.global_outer_attempt_id}",
                )
            )
            try:
                block = make_rubric_draw_block(
                    rubric,
                    outer.hyperparameters,
                    moments,
                    rng=inner_rng,
                    global_outer_attempt_id=outer.global_outer_attempt_id,
                    rubric_retained_index=len(blocks[rubric_id]) + 1,
                    draw_count=selected_config.inner_draw_count,
                )
            except Layer1ModeError:
                local_failures[rubric_id] += 1
                if rubric_block_callback is not None:
                    rubric_block_callback(
                        rubric_id,
                        outer.global_outer_attempt_id,
                        None,
                        "Layer1ModeError",
                    )
                continue
            blocks[rubric_id].append(block)
            if rubric_block_callback is not None:
                rubric_block_callback(
                    rubric_id,
                    outer.global_outer_attempt_id,
                    block,
                    None,
                )

    def process_all_available_outer_draws() -> None:
        if not isinstance(model_executor, SpawnProcessExecutor):
            for rubric_id in target_depths:
                process_available_outer_draws(rubric_id)
            return
        tasks: list[WorkUnit] = []
        task_positions: dict[str, tuple[str, int]] = {}
        for rubric_id, target in target_depths.items():
            if len(blocks[rubric_id]) >= target:
                continue
            start = processed_outer_draws[rubric_id]
            stop = len(outer_draws)
            for position in range(start, stop):
                outer = outer_draws[position]
                task_id = f"block:{rubric_id}:outer:{outer.global_outer_attempt_id}"
                task_positions[task_id] = (rubric_id, position)
                tasks.append(
                    WorkUnit(
                        task_id=task_id,
                        kind=WorkUnitKind.RUBRIC_CONDITIONAL_BLOCK,
                        payload=ConditionalBlockPayload(
                            rubric=rubric_by_id[rubric_id],
                            hyperparameters=outer.hyperparameters,
                            moments=moments,
                            global_outer_attempt_id=outer.global_outer_attempt_id,
                            draw_count=selected_config.inner_draw_count,
                        ),
                    )
                )
            processed_outer_draws[rubric_id] = stop
        if not tasks:
            return
        outcomes = model_executor.execute(
            TaskReference(
                module=execute_conditional_block_work_unit.__module__,
                function=execute_conditional_block_work_unit.__name__,
            ),
            tuple(tasks),
        )
        grouped: dict[str, list[ConditionalBlockValue]] = {
            rubric_id: [] for rubric_id in target_depths
        }
        for outcome in outcomes:
            rubric_id, _ = task_positions[outcome.task_id]
            if not outcome.succeeded or not isinstance(
                outcome.value, ConditionalBlockValue
            ):
                local_failures[rubric_id] += 1
                if rubric_block_callback is not None:
                    _, position = task_positions[outcome.task_id]
                    rubric_block_callback(
                        rubric_id,
                        outer_draws[position].global_outer_attempt_id,
                        None,
                        outcome.error_type or "Layer1ModeError",
                    )
                continue
            grouped[rubric_id].append(outcome.value)
        for rubric_id in target_depths:
            for value in sorted(
                grouped[rubric_id],
                key=lambda current: current.global_outer_attempt_id,
            ):
                block = RubricDrawBlock(
                    rubric_id=value.rubric_id,
                    variant_ids=value.variant_ids,
                    global_outer_attempt_id=value.global_outer_attempt_id,
                    rubric_retained_index=len(blocks[rubric_id]) + 1,
                    rubric_v2_draws=value.rubric_v2_draws,
                )
                blocks[rubric_id].append(block)
                if rubric_block_callback is not None:
                    rubric_block_callback(
                        rubric_id,
                        value.global_outer_attempt_id,
                        block,
                        None,
                    )

    def extend_targets() -> None:
        nonlocal attempt_id, global_failures
        while any(
            len(blocks[rubric_id]) < target
            for rubric_id, target in target_depths.items()
        ):
            process_all_available_outer_draws()
            if all(
                len(blocks[rubric_id]) >= target
                for rubric_id, target in target_depths.items()
            ):
                return
            if attempt_id >= cap:
                unresolved = {
                    rubric_id: {
                        "target": target_depths[rubric_id],
                        "valid": len(blocks[rubric_id]),
                    }
                    for rubric_id in target_depths
                    if len(blocks[rubric_id]) < target_depths[rubric_id]
                }
                raise Layer1ReplenishmentExhaustedError(
                    "LAYER1_REPLENISHMENT_EXHAUSTED "
                    f"scope={json.dumps(unresolved, sort_keys=True)} "
                    f"attempted={cap}"
                )
            if (
                isinstance(model_executor, SpawnProcessExecutor)
                and refitter is fit_layer1_mml
            ):
                batch_size = min(
                    model_executor.max_in_flight,
                    cap - attempt_id,
                )
                attempt_ids = tuple(range(attempt_id + 1, attempt_id + batch_size + 1))
                units = tuple(
                    WorkUnit(
                        task_id=f"outer:{current_attempt}",
                        kind=WorkUnitKind.GLOBAL_OUTER_ATTEMPT,
                        payload=OuterAttemptPayload(
                            data=data,
                            fitted=fit.hyperparameters,
                            global_outer_attempt_id=current_attempt,
                        ),
                    )
                    for current_attempt in attempt_ids
                )
                outcomes = model_executor.execute(
                    TaskReference(
                        module=execute_outer_attempt_work_unit.__module__,
                        function=execute_outer_attempt_work_unit.__name__,
                    ),
                    units,
                )
                attempt_id = attempt_ids[-1]
                for current_attempt, outcome in zip(attempt_ids, outcomes, strict=True):
                    if not outcome.succeeded or not isinstance(
                        outcome.value, Layer1Hyperparameters
                    ):
                        global_failures += 1
                        if outer_attempt_callback is not None:
                            outer_attempt_callback(
                                current_attempt,
                                None,
                                outcome.error_type or "NumericalError",
                            )
                    else:
                        outer = OuterHyperparameterDraw(
                            global_outer_attempt_id=current_attempt,
                            hyperparameters=outcome.value,
                        )
                        outer_draws.append(outer)
                        if outer_attempt_callback is not None:
                            outer_attempt_callback(
                                current_attempt,
                                outer.hyperparameters,
                                None,
                            )
                    if progress_callback is not None:
                        progress_callback(
                            ProgressEvent(
                                phase="outer_attempt",
                                work_unit_id=f"outer:{current_attempt}",
                                retained_attempts=len(outer_draws),
                                failed_attempts=global_failures,
                            )
                        )
                continue
            attempt_id += 1
            outer_rng = np.random.default_rng(
                derive_child_seed("layer1_bootstrap", f"outer:{attempt_id}")
            )
            simulated = simulate_layer1_dataset(
                data, fit.hyperparameters, rng=outer_rng
            )
            try:
                outer_fit = refitter(simulated)
            except NumericalError as error:
                global_failures += 1
                if outer_attempt_callback is not None:
                    outer_attempt_callback(
                        attempt_id,
                        None,
                        type(error).__name__,
                    )
            else:
                outer = OuterHyperparameterDraw(
                    global_outer_attempt_id=attempt_id,
                    hyperparameters=outer_fit.hyperparameters,
                )
                outer_draws.append(outer)
                if outer_attempt_callback is not None:
                    outer_attempt_callback(
                        attempt_id,
                        outer.hyperparameters,
                        None,
                    )
            if progress_callback is not None:
                progress_callback(
                    ProgressEvent(
                        phase="outer_attempt",
                        work_unit_id=f"outer:{attempt_id}",
                        retained_attempts=len(outer_draws),
                        failed_attempts=global_failures,
                    )
                )

    extend_targets()
    decision_depths: dict[str, int] = {}
    prefix_rows: list[dict[str, object]] = []
    final_events: dict[str, tuple[PrimaryEvent, ...]] = {}
    for rubric in data.rubrics:
        propagation = RubricPropagation(
            rubric_id=rubric.rubric_id,
            blocks=tuple(blocks[rubric.rubric_id]),
            attempted=processed_outer_draws[rubric.rubric_id],
            failed=local_failures[rubric.rubric_id],
            elevated_failure_conditions=(
                local_failures[rubric.rubric_id]
                / max(1, processed_outer_draws[rubric.rubric_id])
                > ELEVATED_FAILURE_RATE
            ),
        )
        for depth in selected_config.prefix_diagnostic_depths:
            events = build_primary_events(propagation, depth=depth)
            decision = tier_for_events(
                events,
                kind=Layer1EventKind.WORST_VARIANT,
                unit_id=rubric.rubric_id,
            )
            prefix_rows.append(
                {
                    "rubric_id": rubric.rubric_id,
                    "outer_depth": depth,
                    "tier": decision.tier.value,
                }
            )
        depth = selected_config.adaptive_depths[0]
        while True:
            events = build_primary_events(propagation, depth=depth)
            if not any(event.summary.refinement_triggered for event in events):
                break
            next_depth = (
                selected_config.adaptive_depths[
                    selected_config.adaptive_depths.index(depth) + 1
                ]
                if depth != selected_config.adaptive_depths[-1]
                else None
            )
            if next_depth is None:
                break
            target_depths[rubric.rubric_id] = next_depth
            extend_targets()
            propagation = RubricPropagation(
                rubric_id=rubric.rubric_id,
                blocks=tuple(blocks[rubric.rubric_id]),
                attempted=processed_outer_draws[rubric.rubric_id],
                failed=local_failures[rubric.rubric_id],
                elevated_failure_conditions=(
                    local_failures[rubric.rubric_id]
                    / max(1, processed_outer_draws[rubric.rubric_id])
                    > ELEVATED_FAILURE_RATE
                ),
            )
            depth = next_depth
        decision_depths[rubric.rubric_id] = depth
        final_events[rubric.rubric_id] = events
        if progress_callback is not None:
            progress_callback(
                ProgressEvent(
                    phase="adaptive_depth",
                    work_unit_id=f"rubric:{rubric.rubric_id}",
                    completed=len(decision_depths),
                    total=len(data.rubrics),
                )
            )

    importance_ids = set(select_importance_rubrics(data))
    importance_diagnostics: dict[str, ImportanceDiagnostic] = dict(
        preloaded_importance or {}
    )
    conditional_draws: dict[str, np.ndarray] = dict(preloaded_conditional or {})
    for rubric in data.rubrics:
        if rubric.rubric_id not in conditional_draws:
            conditional_draws[rubric.rubric_id] = _conditional_draws(
                rubric, fit, moments, selected_config.conditional_draw_count
            )
            if conditional_callback is not None:
                conditional_callback(
                    rubric.rubric_id,
                    conditional_draws[rubric.rubric_id],
                )
        if (
            rubric.rubric_id in importance_ids
            and rubric.rubric_id not in importance_diagnostics
        ):
            importance_diagnostics[rubric.rubric_id] = importance_resampling_diagnostic(
                rubric,
                fit.hyperparameters,
                moments,
                rng=np.random.default_rng(
                    derive_child_seed(
                        "diagnostic_resampling",
                        f"importance:{rubric.rubric_id}",
                    )
                ),
                particle_count=selected_config.importance_particles,
            )
            if importance_callback is not None:
                importance_callback(importance_diagnostics[rubric.rubric_id])
        if progress_callback is not None:
            progress_callback(
                ProgressEvent(
                    phase="conditional_diagnostics",
                    work_unit_id=f"rubric:{rubric.rubric_id}",
                    completed=len(conditional_draws),
                    total=len(data.rubrics),
                )
            )

    event_rows: list[dict[str, object]] = []
    rubric_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    draw_frames: dict[str, pd.DataFrame] = {}
    for rubric in data.rubrics:
        depth = decision_depths[rubric.rubric_id]
        rubric_blocks = blocks[rubric.rubric_id]
        propagation = RubricPropagation(
            rubric_id=rubric.rubric_id,
            blocks=tuple(rubric_blocks),
            attempted=processed_outer_draws[rubric.rubric_id],
            failed=local_failures[rubric.rubric_id],
            elevated_failure_conditions=(
                local_failures[rubric.rubric_id]
                / max(1, processed_outer_draws[rubric.rubric_id])
                > ELEVATED_FAILURE_RATE
            ),
        )
        events = final_events[rubric.rubric_id]
        at_cap = depth == selected_config.maximum_depth
        rubric_decision = tier_for_events(
            events,
            kind=Layer1EventKind.WORST_VARIANT,
            unit_id=rubric.rubric_id,
        )
        if at_cap and any(event.summary.refinement_triggered for event in events):
            rubric_decision = tier_from_probabilities(
                rubric_decision.probabilities,
                monte_carlo_indeterminate=True,
            )
        pooled_decision = pooled_mean_tier(propagation, depth=depth)
        means = shrunken_variant_means(propagation, depth=depth)
        conditional = conditional_draws[rubric.rubric_id]
        propagated_values = np.concatenate(
            [block.rubric_v2_draws for block in rubric_blocks[:depth]], axis=0
        )
        minimum_values = np.min(propagated_values, axis=1)
        conditional_minimum = np.min(conditional, axis=1)
        bands = score_band_probabilities(minimum_values)
        proportion = proportion_diagnostic(len(rubric.variants))
        importance = importance_diagnostics.get(rubric.rubric_id)
        event_lookup = {
            (event.kind, event.unit_id, event.floor): event for event in events
        }
        rubric_propagated_probabilities = {
            floor: event_lookup[
                (Layer1EventKind.WORST_VARIANT, rubric.rubric_id, floor)
            ].summary.estimate
            for floor in TIER_FLOORS
        }
        for event in events:
            event_rows.append(
                {
                    "rubric_id": event.rubric_id,
                    "unit_id": event.unit_id,
                    "event_kind": event.kind.value,
                    "floor": event.floor,
                    "outer_depth": depth,
                    "estimate": event.summary.estimate,
                    "mcse": event.summary.mcse,
                    "interval_lower": event.summary.interval_lower,
                    "interval_upper": event.summary.interval_upper,
                    "refinement_triggered": event.summary.refinement_triggered,
                }
            )
        rubric_rows.append(
            {
                "rubric_id": rubric.rubric_id,
                "rubric_order": rubric.rubric_order,
                "eval_dataset": rubric.dataset,
                "tier": rubric_decision.tier.value,
                "uncertain": uncertain_flag(rubric_decision, pooled_decision),
                "v_r": len(rubric.variants),
                "monte_carlo_indeterminate": rubric_decision.monte_carlo_indeterminate,
                "rubric_decision_depth": depth,
                **{
                    f"pi_min_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                        (Layer1EventKind.WORST_VARIANT, rubric.rubric_id, floor)
                    ].summary.estimate
                    for floor in TIER_FLOORS
                },
                **{
                    f"mcse_min_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                        (Layer1EventKind.WORST_VARIANT, rubric.rubric_id, floor)
                    ].summary.mcse
                    for floor in TIER_FLOORS
                },
                **{
                    f"pi_proportion_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                        (
                            Layer1EventKind.PROPORTION_KAPPA_0_75,
                            rubric.rubric_id,
                            floor,
                        )
                    ].summary.estimate
                    for floor in TIER_FLOORS
                },
                **_conditional_probability_fields(
                    conditional_minimum,
                    rubric_propagated_probabilities,
                    minimum=True,
                ),
                "pooled_mean_tier": pooled_decision.tier.value,
                "shrunken_variant_minimum": min(means.values()),
                "proportion_binding_count": proportion.binding_count,
                "proportion_grid_json": json.dumps(proportion.attainable_grid),
                "proportion_uninformative": proportion.uninformative_by_construction,
                **{f"score_band_{name}": value for name, value in bands.items()},
                "elevated_numerical_failure_conditions": (
                    propagation.elevated_failure_conditions
                ),
                "laplace_adequate": importance.adequate if importance else None,
            }
        )
        for variant_index, variant in enumerate(rubric.variants):
            decision = tier_for_events(
                events,
                kind=Layer1EventKind.SINGLE_VARIANT,
                unit_id=variant.variant_id,
                at_adaptive_cap=at_cap,
            )
            values = propagated_values[:, variant_index]
            conditional_values = conditional[:, variant_index]
            variant_bands = score_band_probabilities(values)
            variant_propagated_probabilities = {
                floor: event_lookup[
                    (
                        Layer1EventKind.SINGLE_VARIANT,
                        variant.variant_id,
                        floor,
                    )
                ].summary.estimate
                for floor in TIER_FLOORS
            }
            variant_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "rubric_id": rubric.rubric_id,
                    "variant_index": variant_index,
                    "tier": decision.tier.value,
                    "uncertain": False,
                    "monte_carlo_indeterminate": (decision.monte_carlo_indeterminate),
                    "rubric_decision_depth": depth,
                    **{
                        f"pi_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                            (
                                Layer1EventKind.SINGLE_VARIANT,
                                variant.variant_id,
                                floor,
                            )
                        ].summary.estimate
                        for floor in TIER_FLOORS
                    },
                    **{
                        f"mcse_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                            (
                                Layer1EventKind.SINGLE_VARIANT,
                                variant.variant_id,
                                floor,
                            )
                        ].summary.mcse
                        for floor in TIER_FLOORS
                    },
                    **_conditional_probability_fields(
                        conditional_values,
                        variant_propagated_probabilities,
                        minimum=False,
                    ),
                    "shrunken_mean": means[variant.variant_id],
                    **{
                        f"score_band_{name}": value
                        for name, value in variant_bands.items()
                    },
                    "elevated_numerical_failure_conditions": (
                        propagation.elevated_failure_conditions
                    ),
                    "laplace_adequate": importance.adequate if importance else None,
                }
            )
        draw_rows = [
            {
                "rubric_id": rubric.rubric_id,
                "variant_id": variant.variant_id,
                "global_outer_attempt_id": block.global_outer_attempt_id,
                "rubric_retained_index": block.rubric_retained_index,
                "inner_index": inner_index,
                "rubric_decision_depth": depth,
                "rubric_v2": block.rubric_v2_draws[inner_index, variant_index],
            }
            for block in rubric_blocks
            for inner_index in range(len(block.rubric_v2_draws))
            for variant_index, variant in enumerate(rubric.variants)
        ]
        draw_frames[rubric.rubric_id] = pd.DataFrame(draw_rows)

    importance_rows = [
        {
            "rubric_id": diagnostic.rubric_id,
            "particle_count": diagnostic.particle_count,
            "effective_sample_size": diagnostic.effective_sample_size,
            "effective_sample_size_ratio": diagnostic.effective_sample_size_ratio,
            "adequate": diagnostic.adequate,
            "laplace_mean_json": json.dumps(diagnostic.laplace_mean.tolist()),
            "importance_mean_json": json.dumps(diagnostic.importance_mean.tolist()),
            "laplace_covariance_json": json.dumps(
                diagnostic.laplace_covariance.tolist()
            ),
            "importance_covariance_json": json.dumps(
                diagnostic.importance_covariance.tolist()
            ),
        }
        for diagnostic in importance_diagnostics.values()
    ]
    failures = {
        "schema_version": 1,
        "global_attempts": attempt_id,
        "global_failures": global_failures,
        "rubric_failures": local_failures,
    }
    report = _render_layer1_report(
        pd.DataFrame(rubric_rows),
        pd.DataFrame(variant_rows),
        fit.hyperparameters,
        failures,
    )
    rubric_recommendations = pd.DataFrame(
        rubric_rows, columns=RUBRIC_RECOMMENDATION_COLUMNS
    )
    stability_rows: list[dict[str, object]] = []
    if selected_config.run_leave_out_stability:
        primary_tiers = {
            str(row["rubric_id"]): SuitabilityTier(str(row["tier"]))
            for row in rubric_rows
        }

        def refit_tiers(reduced: Layer1Dataset) -> Mapping[str, SuitabilityTier]:
            reduced_execution = _execute_layer1(
                reduced,
                config=replace(
                    selected_config,
                    run_leave_out_stability=False,
                ),
                refitter=refitter,
            )
            return {
                str(row["rubric_id"]): SuitabilityTier(str(row["tier"]))
                for row in reduced_execution.rubric_recommendations.to_dict(
                    orient="records"
                )
            }

        alternatives: dict[str, Mapping[str, SuitabilityTier]] = dict(
            preloaded_leave_outs or {}
        )
        reduced_datasets = {
            comparison_id: reduced
            for comparison_id, reduced in layer1_leave_out_datasets(data).items()
            if comparison_id not in alternatives
        }
        if (
            isinstance(model_executor, SpawnProcessExecutor)
            and refitter is fit_layer1_mml
        ):
            units = tuple(
                WorkUnit(
                    task_id=f"leave-out:{comparison_id}",
                    kind=WorkUnitKind.LEAVE_OUT_REFIT,
                    payload=LeaveOutRefitPayload(
                        comparison_id=comparison_id,
                        data=reduced,
                        config=selected_config,
                    ),
                )
                for comparison_id, reduced in sorted(reduced_datasets.items())
            )
            outcomes = model_executor.execute(
                TaskReference(
                    module=execute_leave_out_work_unit.__module__,
                    function=execute_leave_out_work_unit.__name__,
                ),
                units,
            )
            failed = [outcome for outcome in outcomes if not outcome.succeeded]
            if failed:
                first = failed[0]
                raise NumericalError(
                    "Layer 1 leave-out refit failed: "
                    f"{first.task_id} {first.error_type}: {first.error_message}"
                )
            for outcome in outcomes:
                if not isinstance(outcome.value, Mapping):
                    continue
                comparison_id = outcome.task_id.removeprefix("leave-out:")
                alternatives[comparison_id] = outcome.value
                if leave_out_callback is not None:
                    leave_out_callback(comparison_id, outcome.value)
        else:
            for comparison_id, reduced in sorted(reduced_datasets.items()):
                alternatives[comparison_id] = refit_tiers(reduced)
                if leave_out_callback is not None:
                    leave_out_callback(
                        comparison_id,
                        alternatives[comparison_id],
                    )
        changed = leave_out_tier_changes(primary_tiers, alternatives)
        stability_rows = [
            {
                "analysis": "leave_out",
                "unit_id": unit_id,
                "comparison_id": comparison_id,
                "tier_changed": unit_id in changed[comparison_id],
                "shared_draw_count": None,
            }
            for comparison_id in sorted(alternatives)
            for unit_id in sorted(primary_tiers)
        ]
    return Layer1Execution(
        fit=_fit_payload(fit),
        hyperparameter_diagnostics={
            "schema_version": 1,
            "moment_smoothed_variants": moments.smoothed_variant_count,
            "retained_phi_components": moments.retained_phi_components,
            "decision_depths": decision_depths,
        },
        numerical_failures=failures,
        importance=pd.DataFrame(importance_rows),
        events=pd.DataFrame(event_rows),
        prefixes=pd.DataFrame(prefix_rows),
        stability=pd.DataFrame(
            stability_rows,
            columns=[
                "analysis",
                "unit_id",
                "comparison_id",
                "tier_changed",
                "shared_draw_count",
            ],
        ),
        draw_frames=draw_frames,
        rubric_recommendations=rubric_recommendations,
        variant_recommendations=pd.DataFrame(
            variant_rows, columns=VARIANT_RECOMMENDATION_COLUMNS
        ),
        report=report,
    )


def _render_layer1_report(
    rubrics: pd.DataFrame,
    variants: pd.DataFrame,
    hyperparameters: Layer1Hyperparameters,
    failures: Mapping[str, object],
) -> str:
    """Render the persisted Layer 1 partial report from result tables."""
    rubric_counts = rubrics["tier"].value_counts().sort_index()
    variant_counts = variants["tier"].value_counts().sort_index()
    lines = [
        "# Layer 1 suitability estimates",
        "",
        "The decision quantity is `Pi_prop`, a propagated empirical-Bayes uncertainty measure,",
        "not a Bayesian posterior probability or a frequentist coverage guarantee.",
        "",
        "## Hyperparameters",
        "",
        f"- `phi`: {hyperparameters.phi:.8g}",
        f"- `trace(Sigma_between)`: {np.trace(hyperparameters.sigma_between):.8g}",
        f"- `trace(Sigma_within)`: {np.trace(hyperparameters.sigma_within):.8g}",
        "",
        "## Rubric tiers",
        "",
        *[f"- {tier}: {count}" for tier, count in rubric_counts.items()],
        "",
        "## Variant tiers",
        "",
        *[f"- {tier}: {count}" for tier, count in variant_counts.items()],
        "",
        "## Numerical failures",
        "",
        f"- Global failures: {failures['global_failures']}",
        f"- Global attempts: {failures['global_attempts']}",
        "",
        "Importance-resampling caveats and Monte Carlo indeterminacy remain attached to",
        "the corresponding recommendation rows.",
        "",
    ]
    return "\n".join(lines)


def _verify_layer1_gate_artifacts(workspace: AnalysisWorkspace) -> None:
    oracle_path = workspace.store.path_for("statistics/r_oracle_verification.json")
    unlock_path = workspace.store.path_for("validation_unlock.json")
    if not oracle_path.is_file() or not unlock_path.is_file():
        raise GateFailureError(
            "Layer 1 requires persisted oracle and validation unlock"
        )
    try:
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MalformedInputError("Layer 1 gate artifact is invalid") from error
    if oracle.get("passed") is not True or unlock.get("authorized") is not True:
        raise GateFailureError(
            "Layer 1 oracle or human validation gate is not authorized"
        )


def _safe_rubric_filename(rubric_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in rubric_id
    )


def _persist_layer1_execution(
    workspace: AnalysisWorkspace,
    execution: Layer1Execution,
) -> StepExecutionResult:
    artifacts = [
        workspace.store.write_json(
            LAYER1_FIT_ARTIFACT,
            execution.fit,
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_json(
            LAYER1_HYPERPARAMETER_DIAGNOSTICS,
            execution.hyperparameter_diagnostics,
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_json(
            LAYER1_FAILURE_ARTIFACT,
            execution.numerical_failures,
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_IMPORTANCE_ARTIFACT,
            versioned_frame(execution.importance),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_EVENT_ARTIFACT,
            versioned_frame(execution.events),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_PREFIX_ARTIFACT,
            versioned_frame(execution.prefixes),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_STABILITY_ARTIFACT,
            versioned_frame(execution.stability),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            RUBRIC_RECOMMENDATION_ARTIFACT,
            versioned_frame(execution.rubric_recommendations),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            VARIANT_RECOMMENDATION_ARTIFACT,
            versioned_frame(execution.variant_recommendations),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_bytes(
            LAYER1_REPORT,
            execution.report.encode("utf-8"),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
    ]
    for rubric_id, frame in execution.draw_frames.items():
        artifacts.append(
            workspace.store.write_parquet(
                Path("draws") / f"pi_prop_{_safe_rubric_filename(rubric_id)}.parquet",
                versioned_frame(frame),
                created_by=WorkflowCommand.FIT_LAYER1,
            )
        )
    return StepExecutionResult(artifacts=tuple(artifacts))


def run_fit_layer1(
    workspace: AnalysisWorkspace,
    *,
    execution_config: Layer1ExecutionConfig | None = None,
    resume: bool = True,
    fresh: bool = False,
    _checkpoint_enabled: bool = True,
) -> StepExecutionResult:
    """Run the gated Layer 1 engine and atomically register complete artifacts."""
    _verify_layer1_gate_artifacts(workspace)
    data = load_layer1_dataset(workspace)
    if not _checkpoint_enabled:
        return _persist_layer1_execution(workspace, _execute_layer1(data))
    run_config = Layer1RunConfig()
    moments = method_of_moments_start(data)
    selected_execution = execution_config or Layer1ExecutionConfig()
    auto_tune = (
        _auto_tune_model_executor(data, moments, selected_execution)
        if selected_execution.worker_mode == "auto"
        else None
    )
    worker_count = resolve_worker_count(
        selected_execution,
        auto_tune_result=auto_tune,
    )
    checkpoint = Layer1CheckpointSession.open(
        workspace.root,
        _checkpoint_identity(workspace, run_config),
        _execution_provenance(selected_execution, worker_count, auto_tune),
        resume=resume,
        fresh=fresh,
    )
    checkpoint.commit_json_state(
        "primary/moments.json",
        {
            "schema_version": 1,
            "smoothed_variant_count": moments.smoothed_variant_count,
            "retained_phi_components": moments.retained_phi_components,
            "hyperparameters": _hyperparameters_payload(moments.hyperparameters),
        },
        task_id="primary:moments",
    )
    progress = RateLimitedProgressWriter(
        jsonl_path=workspace.store.path_for("logs/layer1-progress.jsonl"),
        interval_seconds=selected_execution.progress_interval_seconds,
    )
    shutdown = ShutdownToken()

    def stop_if_requested() -> None:
        interruption = shutdown.interruption
        if interruption is None:
            return
        checkpoint.mark_interrupted(interruption.exit_code)
        progress.write(
            ProgressEvent(
                phase="interrupted",
                checkpoint_path=checkpoint.root.as_posix(),
            ),
            force=True,
        )
        raise SystemExit(interruption.exit_code)

    def report_iteration(
        multiplier: float,
        iteration: int,
        objective: float,
        gradient_maximum: float,
    ) -> None:
        progress.write(
            ProgressEvent(
                phase="primary_mml",
                work_unit_id=f"start:{multiplier}",
                iteration=iteration,
                objective=objective,
                gradient_maximum=gradient_maximum,
                checkpoint_path=checkpoint.root.as_posix(),
            )
        )
        stop_if_requested()

    def commit_start(start: Layer1StartFit) -> None:
        checkpoint.commit_json_state(
            f"primary/start_{str(start.phi_multiplier).replace('.', '_')}.json",
            _start_payload(start),
            task_id=f"primary:start:{start.phi_multiplier}",
        )
        progress.write(
            ProgressEvent(
                phase="primary_start_complete",
                work_unit_id=f"start:{start.phi_multiplier}",
                checkpoint_path=checkpoint.root.as_posix(),
            ),
            force=True,
        )
        stop_if_requested()

    def report_progress(event: ProgressEvent) -> None:
        progress.write(event)
        stop_if_requested()

    def commit_outer_attempt(
        current_attempt: int,
        hyperparameters: Layer1Hyperparameters | None,
        failure_type: str | None,
    ) -> None:
        task_id = f"outer:{current_attempt}"
        if hyperparameters is None:
            checkpoint.upsert_task(
                Layer1TaskLedgerEntry(
                    task_id=task_id,
                    admission_group="global-outer",
                    admission_index=current_attempt,
                    status=Layer1TaskStatus.REJECTED,
                    failure=Layer1TaskFailure(
                        failure_kind=FailureKind.NUMERICAL,
                        failure_type=failure_type or "NumericalError",
                        message="Global outer attempt was numerically non-computable.",
                    ),
                )
            )
        else:
            checkpoint.commit_json_state(
                f"outer/attempt_{current_attempt:06d}.json",
                dict(_hyperparameters_payload(hyperparameters)),
                task_id=task_id,
            )
            checkpoint.upsert_task(
                Layer1TaskLedgerEntry(
                    task_id=task_id,
                    admission_group="global-outer",
                    admission_index=current_attempt,
                    status=Layer1TaskStatus.ADMITTED_RETAINED,
                )
            )
        stop_if_requested()

    def commit_rubric_block(
        rubric_id: str,
        current_attempt: int,
        block: RubricDrawBlock | None,
        failure_type: str | None,
    ) -> None:
        task_id = f"block:{rubric_id}:outer:{current_attempt}"
        admission_group = f"rubric-block:{rubric_id}"
        if block is None:
            checkpoint.upsert_task(
                Layer1TaskLedgerEntry(
                    task_id=task_id,
                    admission_group=admission_group,
                    admission_index=current_attempt,
                    status=Layer1TaskStatus.REJECTED,
                    failure=Layer1TaskFailure(
                        failure_kind=FailureKind.NUMERICAL,
                        failure_type=failure_type or "Layer1ModeError",
                        message="Rubric conditional block was numerically non-computable.",
                    ),
                )
            )
        else:
            safe_id = _safe_rubric_filename(rubric_id)
            checkpoint.commit_npy_state(
                f"blocks/{safe_id}/outer_{current_attempt:06d}.npy",
                block.rubric_v2_draws,
                task_id=task_id,
            )
            checkpoint.commit_json_state(
                f"blocks/{safe_id}/outer_{current_attempt:06d}.json",
                {
                    "schema_version": 1,
                    "variant_ids": list(block.variant_ids),
                    "rubric_retained_index": block.rubric_retained_index,
                },
                task_id=task_id,
            )
            checkpoint.upsert_task(
                Layer1TaskLedgerEntry(
                    task_id=task_id,
                    admission_group=admission_group,
                    admission_index=current_attempt,
                    status=Layer1TaskStatus.ADMITTED_RETAINED,
                )
            )
        stop_if_requested()

    def commit_conditional(rubric_id: str, values: np.ndarray) -> None:
        checkpoint.commit_npy_state(
            f"diagnostics/conditional_{_safe_rubric_filename(rubric_id)}.npy",
            values,
            task_id=f"conditional:{rubric_id}",
        )
        stop_if_requested()

    def commit_importance(diagnostic: ImportanceDiagnostic) -> None:
        checkpoint.commit_json_state(
            f"diagnostics/importance_{_safe_rubric_filename(diagnostic.rubric_id)}.json",
            _importance_payload(diagnostic),
            task_id=f"importance:{diagnostic.rubric_id}",
        )
        stop_if_requested()

    def commit_leave_out(
        comparison_id: str,
        tiers: Mapping[str, SuitabilityTier],
    ) -> None:
        checkpoint.commit_json_state(
            f"leave_out/{_safe_rubric_filename(comparison_id)}.json",
            {
                "comparison_id": comparison_id,
                "tiers": {
                    rubric_id: tier.value for rubric_id, tier in sorted(tiers.items())
                },
            },
            task_id=f"leave-out:{comparison_id}",
        )
        stop_if_requested()

    completed_starts: dict[float, Layer1StartFit] = {}
    for multiplier in (0.5, 1.0, 2.0):
        payload = checkpoint.read_json_state(
            f"primary/start_{str(multiplier).replace('.', '_')}.json"
        )
        if payload is not None:
            completed_starts[multiplier] = _start_from_payload(payload)
    persisted_fit = checkpoint.read_json_state("primary/fit.json")
    fit = (
        _fit_from_payload(persisted_fit, data.dataset_levels)
        if persisted_fit is not None
        else None
    )
    (
        preloaded_outer_draws,
        preloaded_blocks,
        preloaded_processed_outer,
        preloaded_global_failures,
        preloaded_local_failures,
        preloaded_attempt_id,
    ) = _load_propagation_checkpoint(checkpoint, data)
    (
        preloaded_conditional,
        preloaded_importance,
        preloaded_leave_outs,
    ) = _load_diagnostic_checkpoint(checkpoint, data)

    def execute(model_executor: ModelTaskExecutor | None) -> Layer1Execution:
        nonlocal fit
        if fit is None:
            fit = fit_layer1_mml(
                data,
                moments,
                executor=model_executor,
                completed_starts=completed_starts,
                start_callback=commit_start,
                iteration_callback=report_iteration,
            )
            checkpoint.commit_json_state(
                "primary/fit.json",
                dict(_fit_payload(fit)),
                task_id="primary:fit",
            )
        return _execute_layer1(
            data,
            config=run_config,
            model_executor=model_executor,
            moments_override=moments,
            fit_override=fit,
            progress_callback=report_progress,
            outer_attempt_callback=commit_outer_attempt,
            rubric_block_callback=commit_rubric_block,
            preloaded_outer_draws=preloaded_outer_draws,
            preloaded_blocks=preloaded_blocks,
            preloaded_processed_outer=preloaded_processed_outer,
            preloaded_global_failures=preloaded_global_failures,
            preloaded_local_failures=preloaded_local_failures,
            preloaded_attempt_id=preloaded_attempt_id,
            preloaded_conditional=preloaded_conditional,
            preloaded_importance=preloaded_importance,
            preloaded_leave_outs=preloaded_leave_outs,
            conditional_callback=commit_conditional,
            importance_callback=commit_importance,
            leave_out_callback=commit_leave_out,
        )

    try:
        with signal_shutdown_context(shutdown):
            if selected_execution.worker_mode == "sequential":
                execution = execute(None)
            else:
                with SpawnProcessExecutor(
                    worker_count,
                    max_in_flight=selected_execution.max_in_flight,
                ) as model_executor:
                    execution = execute(model_executor)
    except SystemExit:
        raise
    except AnalysisError:
        checkpoint.mark_failed()
        raise
    checkpoint.commit_json_state(
        "final/ready.json",
        {
            "schema_version": 1,
            "compatibility_sha256": checkpoint.manifest.compatibility_sha256,
            "rubric_rows": len(execution.rubric_recommendations),
            "variant_rows": len(execution.variant_recommendations),
        },
        task_id="final:ready",
    )
    result = _persist_layer1_execution(workspace, execution)
    checkpoint.mark_completed()
    return result
