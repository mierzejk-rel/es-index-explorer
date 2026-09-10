"""Tests for the gated Layer 1 command and persisted artifact contracts."""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from es_index_explorer.question_analysis import layer1_pipeline as pipeline_module
from es_index_explorer.question_analysis.contracts import (
    RUBRIC_RECOMMENDATION_COLUMNS,
    TRACE_PFU_TABLE_COLUMNS,
    VARIANT_RECOMMENDATION_COLUMNS,
    StepRecord,
    StepStatus,
    SuitabilityTier,
    WorkflowCommand,
    WorkflowState,
)
from es_index_explorer.question_analysis.errors import GateFailureError
from es_index_explorer.question_analysis.layer1_checkpoint import (
    Layer1CheckpointCompatibilityIdentity,
    Layer1CheckpointSession,
    Layer1ExecutionProvenance,
    Layer1TaskLedgerEntry,
    Layer1TaskStatus,
)
from es_index_explorer.question_analysis.layer1_execution import (
    AutoTuneResult,
    Layer1ExecutionConfig,
    ProgressEvent,
    ShutdownToken,
    SpawnProcessExecutor,
    TaskReference,
    WorkResult,
    WorkUnit,
    WorkUnitKind,
)
from es_index_explorer.question_analysis.layer1_laplace import (
    ImportanceDiagnostic,
    RubricDrawBlock,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Hyperparameters,
    Layer1MmlFit,
    Layer1MomentStart,
    Layer1StartFit,
    MmlIterationCallback,
    MmlStartCallback,
    ModelTaskExecutor,
    pack_hyperparameters,
)
from es_index_explorer.question_analysis.layer1_pipeline import (
    LAYER1_REPORT,
    Layer1Execution,
    Layer1RunConfig,
    _execute_layer1,
    _hyperparameters_payload,
    _load_propagation_checkpoint,
    load_layer1_dataset,
    run_fit_layer1,
)
from es_index_explorer.question_analysis.storage import ArtifactStore, versioned_frame
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

pytestmark = pytest.mark.unit


def _workspace(tmp_path: Path) -> AnalysisWorkspace:
    store = ArtifactStore(tmp_path)
    store.prepare()
    return AnalysisWorkspace(root=tmp_path, store=store)


def _trace_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant_index in range(2):
        for arm_index in range(28):
            row = {column: None for column in TRACE_PFU_TABLE_COLUMNS}
            row.update(
                {
                    "rubric_id": "r1",
                    "rubric_order": 1,
                    "variant_id": f"v{variant_index}",
                    "eval_dataset": "emc2_set1",
                    "variant_index": variant_index,
                    "arm_id": f"a{arm_index:02d}",
                    "stage": "A",
                    "trace_id": f"t{variant_index}-{arm_index}",
                    "P": 5,
                    "F": 2,
                    "U": 1,
                    "N_r": 8,
                    "eligible": True,
                }
            )
            rows.append(row)
    return versioned_frame(pd.DataFrame(rows, columns=TRACE_PFU_TABLE_COLUMNS))


def _execution() -> Layer1Execution:
    return Layer1Execution(
        fit={"schema_version": 1, "objective": 1.0},
        hyperparameter_diagnostics={"schema_version": 1},
        numerical_failures={"schema_version": 1},
        importance=pd.DataFrame(
            [
                {
                    "rubric_id": "r1",
                    "particle_count": 100,
                    "effective_sample_size": 80.0,
                    "effective_sample_size_ratio": 0.8,
                    "adequate": True,
                    "laplace_mean_json": "[]",
                    "importance_mean_json": "[]",
                    "laplace_covariance_json": "[]",
                    "importance_covariance_json": "[]",
                }
            ]
        ),
        events=pd.DataFrame(
            [
                {
                    "rubric_id": "r1",
                    "unit_id": "r1",
                    "event_kind": "WORST_VARIANT",
                    "floor": 0.75,
                    "outer_depth": 500,
                    "estimate": 0.9,
                    "mcse": 0.01,
                    "interval_lower": 0.88,
                    "interval_upper": 0.92,
                    "refinement_triggered": True,
                }
            ]
        ),
        prefixes=pd.DataFrame(
            [{"rubric_id": "r1", "outer_depth": 500, "tier": "SUITABLE"}]
        ),
        stability=pd.DataFrame(
            [
                {
                    "analysis": "prefix",
                    "unit_id": "r1",
                    "comparison_id": "250:500",
                    "tier_changed": False,
                    "shared_draw_count": 20_000,
                }
            ]
        ),
        draw_frames={
            "r1": pd.DataFrame(
                [
                    {
                        "rubric_id": "r1",
                        "variant_id": "v0",
                        "global_outer_attempt_id": 1,
                        "rubric_retained_index": 1,
                        "inner_index": 0,
                        "rubric_decision_depth": 500,
                        "rubric_v2": 0.8,
                    }
                ]
            )
        },
        rubric_recommendations=pd.DataFrame(
            [["r1", *([None] * (len(RUBRIC_RECOMMENDATION_COLUMNS) - 1))]],
            columns=RUBRIC_RECOMMENDATION_COLUMNS,
        ),
        variant_recommendations=pd.DataFrame(
            [["v0", *([None] * (len(VARIANT_RECOMMENDATION_COLUMNS) - 1))]],
            columns=VARIANT_RECOMMENDATION_COLUMNS,
        ),
        report="# Layer 1 suitability estimates\n",
    )


def test_load_layer1_dataset_preserves_joint_variant_design(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=pipeline_module.WorkflowCommand.JOIN,
    )

    data = load_layer1_dataset(workspace)

    assert len(data.rubrics) == 1
    assert len(data.rubrics[0].variants) == 2
    assert all(len(variant.arm_ids) == 28 for variant in data.rubrics[0].variants)
    assert all(variant.stages == ("A",) * 28 for variant in data.rubrics[0].variants)


def test_fit_layer1_requires_physical_oracle_and_unlock_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(GateFailureError, match="requires persisted oracle"):
        run_fit_layer1(_workspace(tmp_path))


def test_fit_layer1_persists_complete_registered_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.store.path_for("statistics/r_oracle_verification.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    workspace.store.path_for("validation_unlock.json").write_text(
        json.dumps({"authorized": True}), encoding="utf-8"
    )
    workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=pipeline_module.WorkflowCommand.JOIN,
    )
    monkeypatch.setattr(pipeline_module, "_execute_layer1", lambda _data: _execution())

    result = run_fit_layer1(workspace, _checkpoint_enabled=False)

    paths = {artifact.path for artifact in result.artifacts}
    assert LAYER1_REPORT in paths
    assert "tables/recommendation_table_rubric.parquet" in paths
    assert "tables/recommendation_table_variant.parquet" in paths
    assert "draws/pi_prop_r1.parquet" in paths
    assert all(path.is_file() for path in (tmp_path / value for value in paths))


def test_small_execution_uses_common_rubric_depth_and_exact_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=pipeline_module.WorkflowCommand.JOIN,
    )
    data = load_layer1_dataset(workspace)
    hyperparameters = Layer1Hyperparameters(
        phi=10.0,
        mu0=np.array([0.5, -0.5]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.eye(2) * 0.2,
        sigma_between=np.eye(2) * 0.3,
    )
    vector = np.zeros(9)
    start_fit = Layer1StartFit(
        phi_multiplier=1.0,
        vector=vector,
        hyperparameters=hyperparameters,
        objective=1.0,
        gradient_maximum=1e-6,
        iterations=2,
        converged=True,
        message="synthetic",
    )
    fit = Layer1MmlFit(
        hyperparameters=hyperparameters,
        vector=vector,
        objective=1.0,
        starts=(start_fit, start_fit, start_fit),
    )
    eta = np.array([0.5, -0.5])
    moments = Layer1MomentStart(
        hyperparameters=hyperparameters,
        rubric_latent_starts={"r1": np.concatenate((eta, eta, eta))},
        variant_alr={"v0": eta, "v1": eta},
        smoothed_variant_count=0,
        retained_phi_components=1,
    )
    monkeypatch.setattr(
        pipeline_module,
        "method_of_moments_start",
        lambda _data: moments,
    )

    def make_block(
        _rubric: pipeline_module.Layer1RubricData,
        _hyperparameters: Layer1Hyperparameters,
        _moments: Layer1MomentStart,
        *,
        rng: np.random.Generator,
        global_outer_attempt_id: int,
        rubric_retained_index: int,
        draw_count: int,
    ) -> RubricDrawBlock:
        del rng
        return RubricDrawBlock(
            rubric_id="r1",
            variant_ids=("v0", "v1"),
            global_outer_attempt_id=global_outer_attempt_id,
            rubric_retained_index=rubric_retained_index,
            rubric_v2_draws=np.full((draw_count, 2), 0.8),
        )

    monkeypatch.setattr(pipeline_module, "make_rubric_draw_block", make_block)
    monkeypatch.setattr(
        pipeline_module,
        "_conditional_draws",
        lambda _rubric, _fit, _moments, count: np.full((count, 2), 0.8),
    )
    monkeypatch.setattr(
        pipeline_module,
        "importance_resampling_diagnostic",
        lambda rubric, _hyperparameters, _moments, **_kwargs: ImportanceDiagnostic(
            rubric_id=rubric.rubric_id,
            particle_count=20,
            effective_sample_size=20.0,
            effective_sample_size_ratio=1.0,
            adequate=True,
            laplace_mean=np.zeros(6),
            importance_mean=np.zeros(6),
            laplace_covariance=np.eye(6),
            importance_covariance=np.eye(6),
        ),
    )

    execution = _execute_layer1(
        data,
        config=Layer1RunConfig(
            mandatory_prefix_depth=4,
            prefix_diagnostic_depths=(2, 4),
            adaptive_depths=(2, 4),
            inner_draw_count=2,
            conditional_draw_count=20,
            importance_particles=20,
            maximum_depth=4,
            run_leave_out_stability=False,
        ),
        refitter=lambda _data: fit,
    )

    assert execution.hyperparameter_diagnostics["decision_depths"] == {"r1": 2}
    assert tuple(execution.rubric_recommendations.columns) == (
        RUBRIC_RECOMMENDATION_COLUMNS
    )
    assert tuple(execution.variant_recommendations.columns) == (
        VARIANT_RECOMMENDATION_COLUMNS
    )
    assert set(execution.draw_frames["r1"]["rubric_decision_depth"]) == {2}
    assert len(execution.prefixes) == 2


def test_small_execution_processes_conditional_blocks_in_spawn_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=pipeline_module.WorkflowCommand.JOIN,
    )
    data = load_layer1_dataset(workspace)
    hyperparameters = Layer1Hyperparameters(
        phi=10.0,
        mu0=np.array([0.5, -0.5]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.eye(2) * 0.2,
        sigma_between=np.eye(2) * 0.3,
    )
    eta = np.array([0.5, -0.5])
    moments = Layer1MomentStart(
        hyperparameters=hyperparameters,
        rubric_latent_starts={"r1": np.concatenate((eta, eta, eta))},
        variant_alr={"v0": eta, "v1": eta},
        smoothed_variant_count=0,
        retained_phi_components=1,
    )
    vector = pack_hyperparameters(hyperparameters, ("emc2_set1",))
    start = Layer1StartFit(
        phi_multiplier=1.0,
        vector=vector,
        hyperparameters=hyperparameters,
        objective=1.0,
        gradient_maximum=1e-6,
        iterations=2,
        converged=True,
        message="synthetic",
    )
    fit = Layer1MmlFit(
        hyperparameters=hyperparameters,
        vector=vector,
        objective=1.0,
        starts=(start, start, start),
    )
    monkeypatch.setattr(
        pipeline_module,
        "method_of_moments_start",
        lambda _data: moments,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_conditional_draws",
        lambda _rubric, _fit, _moments, count: np.full((count, 2), 0.8),
    )
    monkeypatch.setattr(
        pipeline_module,
        "importance_resampling_diagnostic",
        lambda rubric, _hyperparameters, _moments, **_kwargs: ImportanceDiagnostic(
            rubric_id=rubric.rubric_id,
            particle_count=20,
            effective_sample_size=20,
            effective_sample_size_ratio=1,
            adequate=True,
            laplace_mean=np.zeros(6),
            importance_mean=np.zeros(6),
            laplace_covariance=np.eye(6),
            importance_covariance=np.eye(6),
        ),
    )

    with SpawnProcessExecutor(2) as executor:
        execution = _execute_layer1(
            data,
            config=Layer1RunConfig(
                mandatory_prefix_depth=4,
                prefix_diagnostic_depths=(2, 4),
                adaptive_depths=(2, 4),
                inner_draw_count=2,
                conditional_draw_count=20,
                importance_particles=20,
                maximum_depth=4,
                run_leave_out_stability=False,
            ),
            refitter=lambda _data: fit,
            model_executor=executor,
            moments_override=moments,
            fit_override=fit,
        )

    assert len(execution.draw_frames["r1"]) >= 8
    assert set(execution.draw_frames["r1"]["global_outer_attempt_id"]) == {
        1,
        2,
        3,
        4,
    }


def test_process_scheduler_covers_outer_blocks_and_leave_outs_without_nesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=pipeline_module.WorkflowCommand.JOIN,
    )
    data = load_layer1_dataset(workspace)
    hyperparameters = Layer1Hyperparameters(
        phi=10.0,
        mu0=np.array([0.5, -0.5]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.eye(2) * 0.2,
        sigma_between=np.eye(2) * 0.3,
    )
    eta = np.array([0.5, -0.5])
    moments = Layer1MomentStart(
        hyperparameters=hyperparameters,
        rubric_latent_starts={"r1": np.concatenate((eta, eta, eta))},
        variant_alr={"v0": eta, "v1": eta},
        smoothed_variant_count=0,
        retained_phi_components=1,
    )
    vector = pack_hyperparameters(hyperparameters, ("emc2_set1",))
    start = Layer1StartFit(
        phi_multiplier=1.0,
        vector=vector,
        hyperparameters=hyperparameters,
        objective=1.0,
        gradient_maximum=1e-6,
        iterations=2,
        converged=True,
        message="synthetic",
    )
    fit = Layer1MmlFit(
        hyperparameters=hyperparameters,
        vector=vector,
        objective=1.0,
        starts=(start, start, start),
    )

    class FakeSpawnExecutor(SpawnProcessExecutor):
        def __init__(self) -> None:
            super().__init__(2)

        def execute(
            self,
            task: TaskReference,
            work_units: Sequence[WorkUnit],
            *,
            shutdown_token: ShutdownToken | None = None,
        ) -> tuple[WorkResult, ...]:
            del task, shutdown_token
            results: list[WorkResult] = []
            for unit in work_units:
                if unit.kind is WorkUnitKind.GLOBAL_OUTER_ATTEMPT:
                    value: object = hyperparameters
                elif unit.kind is WorkUnitKind.RUBRIC_CONDITIONAL_BLOCK:
                    payload = cast(
                        pipeline_module.ConditionalBlockPayload,
                        unit.payload,
                    )
                    value = pipeline_module.ConditionalBlockValue(
                        rubric_id="r1",
                        variant_ids=("v0", "v1"),
                        global_outer_attempt_id=payload.global_outer_attempt_id,
                        rubric_v2_draws=np.full((2, 2), 0.8),
                    )
                elif unit.kind is WorkUnitKind.LEAVE_OUT_REFIT:
                    value = {"r1": SuitabilityTier.SUITABLE}
                else:
                    raise AssertionError(f"Unexpected task kind {unit.kind}")
                results.append(
                    WorkResult(
                        task_id=unit.task_id,
                        kind=unit.kind,
                        value=value,
                    )
                )
            return tuple(results)

    monkeypatch.setattr(
        pipeline_module,
        "_conditional_draws",
        lambda _rubric, _fit, _moments, count: np.full((count, 2), 0.8),
    )
    monkeypatch.setattr(
        pipeline_module,
        "importance_resampling_diagnostic",
        lambda rubric, _hyperparameters, _moments, **_kwargs: ImportanceDiagnostic(
            rubric_id=rubric.rubric_id,
            particle_count=20,
            effective_sample_size=20,
            effective_sample_size_ratio=1,
            adequate=True,
            laplace_mean=np.zeros(6),
            importance_mean=np.zeros(6),
            laplace_covariance=np.eye(6),
            importance_covariance=np.eye(6),
        ),
    )

    execution = _execute_layer1(
        data,
        config=Layer1RunConfig(
            mandatory_prefix_depth=4,
            prefix_diagnostic_depths=(2, 4),
            adaptive_depths=(2, 4),
            inner_draw_count=2,
            conditional_draw_count=20,
            importance_particles=20,
            maximum_depth=4,
            run_leave_out_stability=True,
        ),
        model_executor=FakeSpawnExecutor(),
        moments_override=moments,
        fit_override=fit,
    )

    assert len(execution.draw_frames["r1"]) >= 8
    assert not execution.stability.empty
    assert set(execution.stability["comparison_id"]) == {
        *(f"arm:a{index:02d}" for index in range(28)),
        "stage:A",
    }


def test_propagation_checkpoint_reloads_outer_and_rubric_block_identity(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=pipeline_module.WorkflowCommand.JOIN,
    )
    data = load_layer1_dataset(workspace)
    identity = Layer1CheckpointCompatibilityIdentity(
        specification_sha256="1" * 64,
        input_sha256s={"trace": "2" * 64},
        numerical_contract_version="layer1-v1",
        statistical_run_config={"depth": 2},
        seed_contract={"master_seed": 20},
        numerical_backend="adaptive-finite-difference-v1",
        architecture="arm64",
        float_abi="<f8:little",
        python_version="3.12",
        numpy_version="2.4",
        scipy_version="1.17",
        blas_lapack_identity="test",
    )
    provenance = Layer1ExecutionProvenance(
        cpu_model="test",
        os_details="test",
        executor_type="sequential",
        worker_count=1,
        max_in_flight=1,
        progress_interval_seconds=1.0,
    )
    checkpoint = Layer1CheckpointSession.open(
        tmp_path,
        identity,
        provenance,
        resume=True,
        fresh=False,
    )
    hyperparameters = Layer1Hyperparameters(
        phi=10.0,
        mu0=np.array([0.5, -0.5]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.eye(2) * 0.2,
        sigma_between=np.eye(2) * 0.3,
    )
    checkpoint.commit_json_state(
        "outer/attempt_000001.json",
        dict(_hyperparameters_payload(hyperparameters)),
        task_id="outer:1",
    )
    checkpoint.upsert_task(
        Layer1TaskLedgerEntry(
            task_id="outer:1",
            admission_group="global-outer",
            admission_index=1,
            status=Layer1TaskStatus.ADMITTED_RETAINED,
        )
    )
    checkpoint.commit_npy_state(
        "blocks/r1/outer_000001.npy",
        np.full((2, 2), 0.8),
        task_id="block:r1:outer:1",
    )
    checkpoint.commit_json_state(
        "blocks/r1/outer_000001.json",
        {
            "schema_version": 1,
            "variant_ids": ["v0", "v1"],
            "rubric_retained_index": 1,
        },
        task_id="block:r1:outer:1",
    )
    checkpoint.upsert_task(
        Layer1TaskLedgerEntry(
            task_id="block:r1:outer:1",
            admission_group="rubric-block:r1",
            admission_index=1,
            status=Layer1TaskStatus.ADMITTED_RETAINED,
        )
    )

    outer, blocks, processed, global_failures, local_failures, attempt_id = (
        _load_propagation_checkpoint(checkpoint, data)
    )

    assert [value.global_outer_attempt_id for value in outer] == [1]
    assert blocks["r1"][0].rubric_retained_index == 1
    np.testing.assert_array_equal(blocks["r1"][0].rubric_v2_draws, np.full((2, 2), 0.8))
    assert processed == {"r1": 1}
    assert global_failures == 0
    assert local_failures == {"r1": 0}
    assert attempt_id == 1


def test_checkpointed_fit_command_promotes_outputs_and_reuses_primary_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# Test specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)
    trace_metadata = workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=WorkflowCommand.JOIN,
    )
    oracle_metadata = workspace.store.write_json(
        "statistics/r_oracle_verification.json",
        {"passed": True},
        created_by=WorkflowCommand.ORACLE,
    )
    unlock_metadata = workspace.store.write_json(
        "validation_unlock.json",
        {"authorized": True},
        created_by=WorkflowCommand.VALIDATE_FEATURES,
    )
    now = datetime.now(UTC)
    state = workspace.load_state()
    steps = dict(state.steps)
    for command in (
        WorkflowCommand.ORACLE,
        WorkflowCommand.VALIDATE_FEATURES,
    ):
        steps[command] = StepRecord(
            status=StepStatus.COMPLETED,
            attempts=1,
            started_at=now,
            completed_at=now,
        )
    workspace.save_state(
        WorkflowState.model_validate(
            {
                **state.model_dump(),
                "steps": steps,
                "artifacts": {
                    trace_metadata.path: trace_metadata,
                    oracle_metadata.path: oracle_metadata,
                    unlock_metadata.path: unlock_metadata,
                },
                "outcome_modeling_unlocked": True,
                "outcome_modeling_unlocked_at": now,
                "updated_at": now,
            }
        )
    )
    hyperparameters = Layer1Hyperparameters(
        phi=10.0,
        mu0=np.array([0.5, -0.5]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.eye(2) * 0.2,
        sigma_between=np.eye(2) * 0.3,
    )
    eta = np.array([0.5, -0.5])
    moments = Layer1MomentStart(
        hyperparameters=hyperparameters,
        rubric_latent_starts={"r1": np.concatenate((eta, eta, eta))},
        variant_alr={"v0": eta, "v1": eta},
        smoothed_variant_count=0,
        retained_phi_components=1,
    )
    vector = pack_hyperparameters(hyperparameters, ("emc2_set1",))
    starts = tuple(
        Layer1StartFit(
            phi_multiplier=multiplier,
            vector=vector,
            hyperparameters=hyperparameters,
            objective=1.0,
            gradient_maximum=1e-6,
            iterations=2,
            converged=True,
            message="synthetic",
        )
        for multiplier in (0.5, 1.0, 2.0)
    )
    fit = Layer1MmlFit(
        hyperparameters=hyperparameters,
        vector=vector,
        objective=1.0,
        starts=starts,
    )
    fit_calls = 0

    def fake_fit(
        _data: object,
        _moments: object,
        *,
        executor: ModelTaskExecutor | None,
        completed_starts: dict[float, Layer1StartFit],
        start_callback: MmlStartCallback,
        iteration_callback: MmlIterationCallback,
    ) -> Layer1MmlFit:
        nonlocal fit_calls
        fit_calls += 1
        assert completed_starts == {}
        iteration_callback(0.5, 1, 1.0, 1e-6)
        for start in starts:
            start_callback(start)
        del executor
        return fit

    monkeypatch.setattr(
        pipeline_module, "method_of_moments_start", lambda _data: moments
    )
    monkeypatch.setattr(pipeline_module, "fit_layer1_mml", fake_fit)

    execution_calls = 0

    def fake_execute(_data: object, **kwargs: object) -> Layer1Execution:
        nonlocal execution_calls
        execution_calls += 1
        progress_callback = cast(
            pipeline_module.Layer1ProgressCallback,
            kwargs["progress_callback"],
        )
        progress_callback(
            ProgressEvent(
                phase="outer_attempt",
                retained_attempts=1,
                failed_attempts=0,
            )
        )
        if execution_calls == 1:
            outer_callback = cast(
                pipeline_module.OuterAttemptCallback,
                kwargs["outer_attempt_callback"],
            )
            block_callback = cast(
                pipeline_module.RubricBlockCallback,
                kwargs["rubric_block_callback"],
            )
            conditional_callback = cast(
                pipeline_module.ConditionalCallback,
                kwargs["conditional_callback"],
            )
            importance_callback = cast(
                pipeline_module.ImportanceCallback,
                kwargs["importance_callback"],
            )
            leave_out_callback = cast(
                pipeline_module.LeaveOutCallback,
                kwargs["leave_out_callback"],
            )
            outer_callback(1, hyperparameters, None)
            outer_callback(2, None, "NumericalError")
            outer_callback(3, hyperparameters, None)
            block = RubricDrawBlock(
                rubric_id="r1",
                variant_ids=("v0", "v1"),
                global_outer_attempt_id=1,
                rubric_retained_index=1,
                rubric_v2_draws=np.full((2, 2), 0.8),
            )
            block_callback("r1", 1, block, None)
            block_callback("r1", 3, None, "Layer1ModeError")
            conditional_callback("r1", np.full((20, 2), 0.8))
            importance_callback(
                ImportanceDiagnostic(
                    rubric_id="r1",
                    particle_count=20,
                    effective_sample_size=20,
                    effective_sample_size_ratio=1,
                    adequate=True,
                    laplace_mean=np.zeros(6),
                    importance_mean=np.zeros(6),
                    laplace_covariance=np.eye(6),
                    importance_covariance=np.eye(6),
                )
            )
            leave_out_callback(
                "arm:a0",
                {"r1": SuitabilityTier.SUITABLE},
            )
        return _execution()

    monkeypatch.setattr(
        pipeline_module,
        "_execute_layer1",
        fake_execute,
    )

    first = run_fit_layer1(workspace)
    second = run_fit_layer1(workspace)

    assert fit_calls == 1
    assert execution_calls == 2
    assert {artifact.path for artifact in first.artifacts} == {
        artifact.path for artifact in second.artifacts
    }
    checkpoint = workspace.root / "checkpoints" / "fit-layer1" / "manifest.json"
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "completed"


def test_auto_tuning_uses_representative_marginal_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.store.write_parquet(
        "tables/trace_pfu_table.parquet",
        _trace_frame(),
        created_by=pipeline_module.WorkflowCommand.JOIN,
    )
    data = load_layer1_dataset(workspace)
    hyperparameters = Layer1Hyperparameters(
        phi=10.0,
        mu0=np.array([0.5, -0.5]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.eye(2) * 0.2,
        sigma_between=np.eye(2) * 0.3,
    )
    eta = np.array([0.5, -0.5])
    moments = Layer1MomentStart(
        hyperparameters=hyperparameters,
        rubric_latent_starts={"r1": np.concatenate((eta, eta, eta))},
        variant_alr={"v0": eta, "v1": eta},
        smoothed_variant_count=0,
        retained_phi_components=1,
    )
    observed: dict[str, object] = {}

    def select(
        candidates: Sequence[int],
        *,
        benchmark_probe: Callable[[int], float],
        memory_probe: Callable[[int], int],
        memory_limit_bytes: int,
        explicit_worker_count: int | None,
    ) -> AutoTuneResult:
        observed["candidates"] = tuple(candidates)
        observed["elapsed"] = benchmark_probe(1)
        observed["memory"] = memory_probe(1)
        observed["limit"] = memory_limit_bytes
        observed["override"] = explicit_worker_count
        return AutoTuneResult(
            selected_worker_count=1,
            candidates=tuple(candidates),
            samples=(),
        )

    monkeypatch.setattr(
        pipeline_module,
        "discover_worker_candidates",
        lambda: (1, 2),
    )
    monkeypatch.setattr(pipeline_module, "auto_select_worker_count", select)

    result = pipeline_module._auto_tune_model_executor(
        data,
        moments,
        Layer1ExecutionConfig(worker_mode="auto"),
    )

    assert result.selected_worker_count == 1
    assert observed["candidates"] == (1, 2)
    assert cast(float, observed["elapsed"]) >= 0
    assert cast(int, observed["memory"]) > 0
    assert cast(int, observed["limit"]) >= cast(int, observed["memory"])
    assert observed["override"] is None
