"""Tests for release-sequence transitions."""

from datetime import UTC, datetime, timedelta

import pytest

from es_index_explorer.question_analysis.contracts import (
    StepStatus,
    WorkflowCommand,
    WorkflowState,
)
from es_index_explorer.question_analysis.errors import (
    MalformedInputError,
    PrerequisiteError,
)
from es_index_explorer.question_analysis.workflow import (
    PREREQUISITES,
    complete_step,
    start_step,
)

pytestmark = pytest.mark.unit

START = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)

EXPECTED_PREREQUISITES = {
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


def _complete(
    state: WorkflowState, command: WorkflowCommand, offset: int
) -> WorkflowState:
    running, should_run = start_step(
        state, command, now=START + timedelta(minutes=offset)
    )
    assert should_run
    return complete_step(
        running,
        command,
        authorize_outcome_modeling=command is WorkflowCommand.VALIDATE_FEATURES,
        now=START + timedelta(minutes=offset, seconds=1),
    )


def test_prerequisite_map_matches_locked_release_sequence() -> None:
    assert PREREQUISITES == EXPECTED_PREREQUISITES


def test_features_cannot_bypass_join() -> None:
    state = WorkflowState.initial(START)

    with pytest.raises(PrerequisiteError, match="features requires completed step"):
        start_step(state, WorkflowCommand.FEATURES, now=START)


def test_validate_unlocks_models_but_oracle_remains_required() -> None:
    state = WorkflowState.initial(START)
    sequence = (
        WorkflowCommand.JOIN,
        WorkflowCommand.FEATURES,
        WorkflowCommand.ANNOTATE_EMIT,
        WorkflowCommand.ANNOTATE_RUN,
        WorkflowCommand.ANNOTATE_INGEST,
        WorkflowCommand.GOLD_SAMPLE,
        WorkflowCommand.GOLD_INGEST,
        WorkflowCommand.VALIDATE_FEATURES,
    )
    for offset, command in enumerate(sequence):
        state = _complete(state, command, offset)

    assert state.outcome_modeling_unlocked
    assert state.outcome_modeling_unlocked_at == START + timedelta(
        minutes=len(sequence) - 1, seconds=1
    )
    with pytest.raises(PrerequisiteError, match="oracle"):
        start_step(state, WorkflowCommand.FIT_LAYER1, now=START + timedelta(hours=1))

    state = _complete(state, WorkflowCommand.ORACLE, 20)
    running, should_run = start_step(
        state, WorkflowCommand.FIT_LAYER1, now=START + timedelta(minutes=21)
    )

    assert should_run
    assert running.steps[WorkflowCommand.FIT_LAYER1].status is StepStatus.RUNNING


def test_validate_completion_without_explicit_authorization_is_rejected() -> None:
    state = WorkflowState.initial(START)
    for offset, command in enumerate(
        (
            WorkflowCommand.JOIN,
            WorkflowCommand.FEATURES,
            WorkflowCommand.ANNOTATE_EMIT,
            WorkflowCommand.ANNOTATE_RUN,
            WorkflowCommand.ANNOTATE_INGEST,
            WorkflowCommand.GOLD_SAMPLE,
            WorkflowCommand.GOLD_INGEST,
        )
    ):
        state = _complete(state, command, offset)
    running, _ = start_step(
        state,
        WorkflowCommand.VALIDATE_FEATURES,
        now=START + timedelta(minutes=8),
    )

    with pytest.raises(MalformedInputError, match="unlock authorization"):
        complete_step(
            running,
            WorkflowCommand.VALIDATE_FEATURES,
            now=START + timedelta(minutes=8, seconds=1),
        )


def test_oracle_can_run_before_join_but_not_after_fitting_starts() -> None:
    state = _complete(WorkflowState.initial(START), WorkflowCommand.ORACLE, 0)
    state = _complete(state, WorkflowCommand.JOIN, 1)
    state = _complete(state, WorkflowCommand.FEATURES, 2)
    state = _complete(state, WorkflowCommand.ANNOTATE_EMIT, 3)
    state = _complete(state, WorkflowCommand.ANNOTATE_RUN, 4)
    state = _complete(state, WorkflowCommand.ANNOTATE_INGEST, 5)
    state = _complete(state, WorkflowCommand.GOLD_SAMPLE, 6)
    state = _complete(state, WorkflowCommand.GOLD_INGEST, 7)
    state = _complete(state, WorkflowCommand.VALIDATE_FEATURES, 8)
    state, _ = start_step(
        state, WorkflowCommand.FIT_FAMILIES, now=START + timedelta(minutes=9)
    )
    steps = dict(state.steps)
    steps[WorkflowCommand.ORACLE] = steps[WorkflowCommand.ORACLE].model_copy(
        update={
            "status": StepStatus.PENDING,
            "attempts": 0,
            "started_at": None,
            "completed_at": None,
        }
    )
    state = state.model_copy(update={"steps": steps})

    with pytest.raises(PrerequisiteError, match="after outcome fitting has started"):
        start_step(state, WorkflowCommand.ORACLE, now=START + timedelta(minutes=10))


def test_running_step_resumes_and_completed_step_is_idempotent() -> None:
    state, should_run = start_step(
        WorkflowState.initial(START), WorkflowCommand.JOIN, now=START
    )
    assert should_run

    resumed, should_run = start_step(
        state, WorkflowCommand.JOIN, now=START + timedelta(minutes=1)
    )
    assert should_run
    assert resumed.steps[WorkflowCommand.JOIN].attempts == 2

    completed = complete_step(
        resumed, WorkflowCommand.JOIN, now=START + timedelta(minutes=2)
    )
    unchanged, should_run = start_step(
        completed, WorkflowCommand.JOIN, now=START + timedelta(minutes=3)
    )
    assert not should_run
    assert unchanged == completed
