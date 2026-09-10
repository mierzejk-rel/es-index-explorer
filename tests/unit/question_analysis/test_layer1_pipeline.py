"""Tests for the gated Layer 1 command and persisted artifact contracts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from es_index_explorer.question_analysis import layer1_pipeline as pipeline_module
from es_index_explorer.question_analysis.contracts import (
    RUBRIC_RECOMMENDATION_COLUMNS,
    TRACE_PFU_TABLE_COLUMNS,
    VARIANT_RECOMMENDATION_COLUMNS,
)
from es_index_explorer.question_analysis.errors import GateFailureError
from es_index_explorer.question_analysis.layer1_laplace import (
    ImportanceDiagnostic,
    RubricDrawBlock,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Hyperparameters,
    Layer1MmlFit,
    Layer1MomentStart,
    Layer1StartFit,
)
from es_index_explorer.question_analysis.layer1_pipeline import (
    LAYER1_REPORT,
    Layer1Execution,
    Layer1RunConfig,
    _execute_layer1,
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

    result = run_fit_layer1(workspace)

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
