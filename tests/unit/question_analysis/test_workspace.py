"""Tests for manifests, atomic artifacts, and resumable state."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from es_index_explorer.question_analysis import workspace as workspace_module
from es_index_explorer.question_analysis.contracts import (
    ANALYSIS_LOCK_TAG,
    MASTER_SEED,
    ArtifactMetadata,
    StepStatus,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.seeds import named_stream_seeds
from es_index_explorer.question_analysis.storage import ARTIFACT_DIRECTORIES
from es_index_explorer.question_analysis.workspace import (
    DEFAULT_ANALYSIS_ROOT,
    AnalysisWorkspace,
)

pytestmark = pytest.mark.unit


def test_default_root_is_segment5_execution_root() -> None:
    assert DEFAULT_ANALYSIS_ROOT.name == "simplemode-v1-segment5"


def _specification(tmp_path: Path) -> Path:
    path = tmp_path / "specification.md"
    path.write_text("# Locked specification\n", encoding="utf-8")
    return path


def test_manifest_round_trip_and_frozen_layout(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    specification = _specification(tmp_path)
    workspace = AnalysisWorkspace.initialize(root, specification)
    manifest = workspace.load_manifest()

    assert manifest.master_seed == MASTER_SEED
    assert manifest.analysis_lock_tag == ANALYSIS_LOCK_TAG
    assert manifest.stream_seeds == named_stream_seeds()
    assert set(manifest.resources) == {
        "ordinal_grade_oracle",
        "ordinal_grade_oracle_provenance",
        "r_oracle_dockerfile",
        "segment2_arms_source",
        "segment2_catalogue_source",
        "segment2_grade_oracle_source",
        "segment2_join_source",
        "segment3_codebook",
        "segment3_contracts_source",
        "segment3_features_source",
        "segment3_linguistics_source",
        "segment3_nlp_resources_source",
        "segment3_quotation_audit",
        "segment3_source_dossier",
        "segment3_spacy_manifest",
        "segment3_storage_source",
        "segment4_annotations_source",
        "segment4_cli_source",
        "segment4_workflow_source",
        "segment5_decision_ledger",
        "segment5_dsl_gate_source",
        "segment5_gold_source",
        "segment5_operations",
        "segment5_validation_source",
        "stanza_en_resource_manifest",
    }
    assert workspace.load_manifest() == manifest
    assert AnalysisWorkspace.initialize(root, specification)
    assert (
        workspace.load_state().steps[WorkflowCommand.JOIN].status is StepStatus.PENDING
    )
    assert all((root / directory).is_dir() for directory in ARTIFACT_DIRECTORIES)


def test_specification_drift_blocks_reinitialize_and_open(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    specification = _specification(tmp_path)
    AnalysisWorkspace.initialize(root, specification)
    specification.write_text("# Changed specification\n", encoding="utf-8")

    with pytest.raises(MalformedInputError, match="Specification fingerprint differs"):
        AnalysisWorkspace.initialize(root, specification)
    with pytest.raises(MalformedInputError, match="Locked input changed"):
        AnalysisWorkspace.open_existing(root)


def test_resource_drift_blocks_reinitialize_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "analysis"
    specification = _specification(tmp_path)
    resource = tmp_path / "resource.md"
    resource.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        workspace_module,
        "DEFAULT_RESOURCES",
        {"test_resource": resource},
    )
    AnalysisWorkspace.initialize(root, specification)
    resource.write_text("changed\n", encoding="utf-8")

    with pytest.raises(MalformedInputError, match="Resource fingerprint differs"):
        AnalysisWorkspace.initialize(root, specification)
    with pytest.raises(MalformedInputError, match="Locked input changed"):
        AnalysisWorkspace.open_existing(root)


def test_successful_step_records_output_hash_and_skips_repeat(tmp_path: Path) -> None:
    workspace = AnalysisWorkspace.initialize(
        tmp_path / "analysis", _specification(tmp_path)
    )

    def write_output(current: AnalysisWorkspace) -> Iterator[ArtifactMetadata]:
        yield current.store.write_json(
            "tables/example.json",
            {"value": 7},
            created_by=WorkflowCommand.JOIN,
        )

    assert workspace.run_step(WorkflowCommand.JOIN, write_output)
    assert not workspace.run_step(WorkflowCommand.JOIN, write_output)

    state = workspace.load_state()
    metadata = state.artifacts["tables/example.json"]
    assert state.steps[WorkflowCommand.JOIN].status is StepStatus.COMPLETED
    assert (
        metadata.size_bytes
        == (tmp_path / "analysis" / "tables" / "example.json").stat().st_size
    )


def test_tampered_output_blocks_resume(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    workspace = AnalysisWorkspace.initialize(root, _specification(tmp_path))

    def write_output(current: AnalysisWorkspace) -> Iterator[ArtifactMetadata]:
        yield current.store.write_json(
            "tables/example.json",
            {"value": 7},
            created_by=WorkflowCommand.JOIN,
        )

    workspace.run_step(WorkflowCommand.JOIN, write_output)
    (root / "tables" / "example.json").write_text('{"value": 8}\n', encoding="utf-8")

    with pytest.raises(MalformedInputError, match="Recorded artifact changed"):
        AnalysisWorkspace.open_existing(root)


def test_artifact_path_cannot_escape_analysis_root(tmp_path: Path) -> None:
    workspace = AnalysisWorkspace.initialize(
        tmp_path / "analysis", _specification(tmp_path)
    )

    with pytest.raises(MalformedInputError, match="escapes analysis root"):
        workspace.store.path_for("../outside.json")


def test_categorized_failure_is_persisted_and_resumable(tmp_path: Path) -> None:
    workspace = AnalysisWorkspace.initialize(
        tmp_path / "analysis", _specification(tmp_path)
    )

    def fail_gate(current: AnalysisWorkspace) -> None:
        del current
        raise GateFailureError("balance gate failed")

    with pytest.raises(GateFailureError):
        workspace.run_step(WorkflowCommand.JOIN, fail_gate)

    failed = workspace.load_state().steps[WorkflowCommand.JOIN]
    assert failed.status is StepStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.message == "balance gate failed"
    assert failed.failure_history == (failed.failure,)

    def recover(current: AnalysisWorkspace) -> None:
        del current

    workspace.run_step(WorkflowCommand.JOIN, recover)
    recovered = workspace.load_state().steps[WorkflowCommand.JOIN]
    assert recovered.status is StepStatus.COMPLETED
    assert recovered.failure is None
    assert recovered.failure_history == failed.failure_history


def test_join_action_can_lock_inputs_but_cannot_change_them(tmp_path: Path) -> None:
    workspace = AnalysisWorkspace.initialize(
        tmp_path / "analysis", _specification(tmp_path)
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text('{"snapshot": 1}\n', encoding="utf-8")

    def lock_input(current: AnalysisWorkspace) -> None:
        current.register_input("snapshot", snapshot)

    workspace.run_step(WorkflowCommand.JOIN, lock_input)
    assert (
        workspace.register_input("snapshot", snapshot).inputs["snapshot"].size_bytes > 0
    )

    snapshot.write_text('{"snapshot": 2}\n', encoding="utf-8")
    with pytest.raises(
        MalformedInputError, match="differs from its locked fingerprint"
    ):
        workspace.register_input("snapshot", snapshot)


def test_interrupted_running_step_resumes_on_next_attempt(tmp_path: Path) -> None:
    workspace = AnalysisWorkspace.initialize(
        tmp_path / "analysis", _specification(tmp_path)
    )

    def crash(current: AnalysisWorkspace) -> None:
        del current
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        workspace.run_step(WorkflowCommand.JOIN, crash)

    interrupted = workspace.load_state().steps[WorkflowCommand.JOIN]
    assert interrupted.status is StepStatus.RUNNING
    assert interrupted.attempts == 1

    def recover(current: AnalysisWorkspace) -> None:
        del current

    workspace.run_step(WorkflowCommand.JOIN, recover)
    completed = workspace.load_state().steps[WorkflowCommand.JOIN]
    assert completed.status is StepStatus.COMPLETED
    assert completed.attempts == 2
