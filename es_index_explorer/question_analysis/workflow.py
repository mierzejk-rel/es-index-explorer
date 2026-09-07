"""Pure workflow transitions and release-sequence gates."""

from collections.abc import Iterable
from datetime import UTC, datetime

from es_index_explorer.question_analysis.contracts import (
    ArtifactMetadata,
    FailureRecord,
    StepRecord,
    StepStatus,
    WorkflowCommand,
    WorkflowState,
)
from es_index_explorer.question_analysis.errors import (
    AnalysisError,
    MalformedInputError,
    PrerequisiteError,
)

PREREQUISITES: dict[WorkflowCommand, frozenset[WorkflowCommand]] = {
    WorkflowCommand.JOIN: frozenset(),
    WorkflowCommand.FEATURES: frozenset({WorkflowCommand.JOIN}),
    WorkflowCommand.ANNOTATE_EMIT: frozenset({WorkflowCommand.FEATURES}),
    WorkflowCommand.ANNOTATE_RUN: frozenset({WorkflowCommand.ANNOTATE_EMIT}),
    WorkflowCommand.ANNOTATE_INGEST: frozenset({WorkflowCommand.ANNOTATE_RUN}),
    WorkflowCommand.GOLD_SAMPLE: frozenset({WorkflowCommand.ANNOTATE_INGEST}),
    WorkflowCommand.GOLD_INGEST: frozenset({WorkflowCommand.GOLD_SAMPLE}),
    WorkflowCommand.VALIDATE_FEATURES: frozenset({WorkflowCommand.GOLD_INGEST}),
    WorkflowCommand.ORACLE: frozenset(),
    WorkflowCommand.FIT_LAYER1: frozenset(
        {WorkflowCommand.VALIDATE_FEATURES, WorkflowCommand.ORACLE}
    ),
    WorkflowCommand.FIT_FAMILIES: frozenset(
        {WorkflowCommand.VALIDATE_FEATURES, WorkflowCommand.ORACLE}
    ),
    WorkflowCommand.ROBUSTNESS: frozenset(
        {WorkflowCommand.FIT_LAYER1, WorkflowCommand.FIT_FAMILIES}
    ),
    WorkflowCommand.REPORT: frozenset({WorkflowCommand.ROBUSTNESS}),
}


def assert_can_start(state: WorkflowState, command: WorkflowCommand) -> None:
    """Validate release-sequence prerequisites for a command."""
    if command is WorkflowCommand.STATUS:
        return
    if command is WorkflowCommand.ORACLE:
        started_fits = [
            fit_command.value
            for fit_command in (
                WorkflowCommand.FIT_LAYER1,
                WorkflowCommand.FIT_FAMILIES,
            )
            if state.steps[fit_command].status is not StepStatus.PENDING
        ]
        if started_fits:
            raise PrerequisiteError(
                f"oracle cannot run after outcome fitting has started: {', '.join(started_fits)}"
            )
    missing = sorted(
        prerequisite.value
        for prerequisite in PREREQUISITES[command]
        if state.steps[prerequisite].status is not StepStatus.COMPLETED
    )
    if missing:
        raise PrerequisiteError(
            f"{command.value} requires completed step(s): {', '.join(missing)}"
        )
    if (
        command in (WorkflowCommand.FIT_LAYER1, WorkflowCommand.FIT_FAMILIES)
        and not state.outcome_modeling_unlocked
    ):
        raise PrerequisiteError(f"{command.value} requires the outcome-modeling unlock")


def start_step(
    state: WorkflowState,
    command: WorkflowCommand,
    *,
    now: datetime | None = None,
) -> tuple[WorkflowState, bool]:
    """Start or resume a command; completed commands become no-ops."""
    if command is WorkflowCommand.STATUS:
        raise MalformedInputError("status is read-only and cannot be started")
    record = state.steps[command]
    if record.status is StepStatus.COMPLETED:
        return state, False
    assert_can_start(state, command)
    timestamp = now or datetime.now(UTC)
    steps = dict(state.steps)
    steps[command] = StepRecord(
        status=StepStatus.RUNNING,
        attempts=record.attempts + 1,
        started_at=timestamp,
        failure_history=record.failure_history,
    )
    return state.model_copy(update={"steps": steps, "updated_at": timestamp}), True


def complete_step(
    state: WorkflowState,
    command: WorkflowCommand,
    artifacts: Iterable[ArtifactMetadata] = (),
    *,
    authorize_outcome_modeling: bool = False,
    now: datetime | None = None,
) -> WorkflowState:
    """Complete a running command and record all output hashes."""
    record = state.steps[command]
    if record.status is not StepStatus.RUNNING:
        raise MalformedInputError(f"{command.value} is not running")
    if command is WorkflowCommand.VALIDATE_FEATURES and not authorize_outcome_modeling:
        raise MalformedInputError(
            "validate-features completion requires verified unlock authorization"
        )
    if authorize_outcome_modeling and command is not WorkflowCommand.VALIDATE_FEATURES:
        raise MalformedInputError(
            "Only validate-features may authorize outcome modeling"
        )
    timestamp = now or datetime.now(UTC)
    artifact_map = dict(state.artifacts)
    for artifact in artifacts:
        if artifact.created_by is not command:
            raise MalformedInputError(
                f"Artifact {artifact.path} was created by {artifact.created_by.value}, not {command.value}"
            )
        artifact_map[artifact.path] = artifact
    steps = dict(state.steps)
    steps[command] = record.model_copy(
        update={
            "status": StepStatus.COMPLETED,
            "completed_at": timestamp,
            "failure": None,
        }
    )
    return state.model_copy(
        update={
            "steps": steps,
            "artifacts": artifact_map,
            "outcome_modeling_unlocked": (
                state.outcome_modeling_unlocked or authorize_outcome_modeling
            ),
            "outcome_modeling_unlocked_at": (
                timestamp
                if authorize_outcome_modeling
                else state.outcome_modeling_unlocked_at
            ),
            "updated_at": timestamp,
        }
    )


def fail_step(
    state: WorkflowState,
    command: WorkflowCommand,
    error: AnalysisError,
    *,
    now: datetime | None = None,
) -> WorkflowState:
    """Persist a categorized failure for a running command."""
    record = state.steps[command]
    if record.status is not StepStatus.RUNNING:
        raise MalformedInputError(f"{command.value} is not running")
    timestamp = now or datetime.now(UTC)
    failure = FailureRecord(
        kind=error.failure_kind,
        message=error.message,
        occurred_at=timestamp,
    )
    steps = dict(state.steps)
    steps[command] = record.model_copy(
        update={
            "status": StepStatus.FAILED,
            "completed_at": timestamp,
            "failure": failure,
            "failure_history": (*record.failure_history, failure),
        }
    )
    return state.model_copy(update={"steps": steps, "updated_at": timestamp})
