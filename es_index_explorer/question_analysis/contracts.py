"""Typed contracts for the question-suitability workflow."""

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
ANALYSIS_ID = "simplemode-v1"
ANALYSIS_LOCK_TAG = "analysis-lock/simple-mode-question-suitability"
MASTER_SEED = 20
ANNOTATION_BATCH_SIZE = 24
GOLD_SAMPLE_PER_STRATUM = 12
ANNOTATOR_MODELS = ("Claude Opus 5 (high thinking)", "GPT-5.6 Sol")
OUTCOME_FIELD_DENYLIST = frozenset(
    {"rubric_v2", "pass_rate", "grade", "ordinal_grade", "tier"}
)


class ExitCode(IntEnum):
    """Locked CLI exit categories."""

    MALFORMED_INPUT = 1
    GATE_FAILURE = 2
    PREREQUISITE = 3
    NUMERICAL = 4


class WorkflowCommand(StrEnum):
    """Public workflow commands in display order."""

    JOIN = "join"
    FEATURES = "features"
    ANNOTATE_EMIT = "annotate-emit"
    ANNOTATE_RUN = "annotate-run"
    ANNOTATE_INGEST = "annotate-ingest"
    GOLD_SAMPLE = "gold-sample"
    GOLD_INGEST = "gold-ingest"
    VALIDATE_FEATURES = "validate-features"
    ORACLE = "oracle"
    FIT_LAYER1 = "fit-layer1"
    FIT_FAMILIES = "fit-families"
    ROBUSTNESS = "robustness"
    REPORT = "report"
    STATUS = "status"


STATE_CHANGING_COMMANDS = tuple(
    command for command in WorkflowCommand if command is not WorkflowCommand.STATUS
)


class StepStatus(StrEnum):
    """Persistent workflow-step states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureKind(StrEnum):
    """Persistent failure categories aligned with CLI exits."""

    MALFORMED_INPUT = "malformed_input"
    GATE_FAILURE = "gate_failure"
    PREREQUISITE = "prerequisite"
    NUMERICAL = "numerical"


class FeatureValidationStatus(StrEnum):
    """Confirmatory feature-validation decisions."""

    VALIDATED = "VALIDATED"
    VALIDATED_WITH_LIMITATIONS = "VALIDATED_WITH_LIMITATIONS"
    NOT_VALIDATED = "NOT_VALIDATED"


class AnalysisFlag(StrEnum):
    """Cross-segment diagnostic flags reserved by the specification."""

    METRIC_UNDEFINED_DEGENERATE_GOLD = "METRIC_UNDEFINED_DEGENERATE_GOLD"
    MONTE_CARLO_INDETERMINATE = "MONTE_CARLO_INDETERMINATE"
    BH_INDETERMINATE = "BH_INDETERMINATE"
    DISPUTED = "DISPUTED"


class FrozenModel(BaseModel):
    """Immutable Pydantic model base."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class FileFingerprint(FrozenModel):
    """SHA-256 identity for an immutable file."""

    path: str = Field(min_length=1)
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        return normalized


class ArtifactMetadata(FileFingerprint):
    """Identity and provenance for a persisted output."""

    created_by: WorkflowCommand
    created_at: datetime


class FailureRecord(FrozenModel):
    """Failure persisted for a workflow attempt."""

    kind: FailureKind
    message: str = Field(min_length=1)
    occurred_at: datetime


class StepRecord(FrozenModel):
    """Persistent state for one workflow command."""

    status: StepStatus = StepStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure: FailureRecord | None = None
    failure_history: tuple[FailureRecord, ...] = ()

    @model_validator(mode="after")
    def validate_timestamps(self) -> "StepRecord":
        if self.status is StepStatus.PENDING and (
            self.attempts != 0
            or any((self.started_at, self.completed_at, self.failure))
        ):
            raise ValueError("pending steps cannot carry timestamps or failures")
        if self.status is StepStatus.RUNNING and (
            self.started_at is None
            or self.completed_at is not None
            or self.failure is not None
        ):
            raise ValueError("running steps require only started_at")
        if self.status is StepStatus.COMPLETED and (
            self.started_at is None
            or self.completed_at is None
            or self.failure is not None
        ):
            raise ValueError(
                "completed steps require start/completion times and no failure"
            )
        if self.status is StepStatus.FAILED and (
            self.started_at is None or self.completed_at is None or self.failure is None
        ):
            raise ValueError(
                "failed steps require start/completion times and a failure"
            )
        if self.failure is not None and (
            not self.failure_history or self.failure_history[-1] != self.failure
        ):
            raise ValueError("current failure must be the last failure-history entry")
        return self


class AnalysisLockManifest(FrozenModel):
    """Immutable analysis identity and input/tool fingerprints."""

    schema_version: Literal[1] = SCHEMA_VERSION
    analysis_id: Literal["simplemode-v1"] = ANALYSIS_ID
    analysis_lock_tag: Literal["analysis-lock/simple-mode-question-suitability"] = (
        ANALYSIS_LOCK_TAG
    )
    specification: FileFingerprint
    master_seed: Literal[20] = MASTER_SEED
    stream_seeds: dict[str, int]
    inputs: dict[str, FileFingerprint] = Field(default_factory=dict)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    resources: dict[str, FileFingerprint] = Field(default_factory=dict)

    @field_validator("stream_seeds")
    @classmethod
    def validate_stream_seeds(cls, value: dict[str, int]) -> dict[str, int]:
        from es_index_explorer.question_analysis.seeds import named_stream_seeds

        if value != named_stream_seeds():
            raise ValueError("stream_seeds must exactly match the frozen named streams")
        return value


class WorkflowState(FrozenModel):
    """Mutable-by-replacement workflow state persisted in ``state.json``."""

    schema_version: Literal[1] = SCHEMA_VERSION
    analysis_id: Literal["simplemode-v1"] = ANALYSIS_ID
    steps: dict[WorkflowCommand, StepRecord]
    artifacts: dict[str, ArtifactMetadata] = Field(default_factory=dict)
    outcome_modeling_unlocked: bool = False
    outcome_modeling_unlocked_at: datetime | None = None
    updated_at: datetime

    @classmethod
    def initial(cls, now: datetime | None = None) -> "WorkflowState":
        timestamp = now or datetime.now(UTC)
        return cls(
            steps={command: StepRecord() for command in STATE_CHANGING_COMMANDS},
            updated_at=timestamp,
        )

    @model_validator(mode="after")
    def validate_steps_and_unlock(self) -> "WorkflowState":
        if set(self.steps) != set(STATE_CHANGING_COMMANDS):
            raise ValueError(
                "state must contain every state-changing command exactly once"
            )
        validation_complete = (
            self.steps[WorkflowCommand.VALIDATE_FEATURES].status is StepStatus.COMPLETED
        )
        if self.outcome_modeling_unlocked != validation_complete:
            raise ValueError(
                "outcome_modeling_unlocked must match validate-features completion"
            )
        if validation_complete != (self.outcome_modeling_unlocked_at is not None):
            raise ValueError(
                "outcome_modeling_unlocked_at must match validate-features completion"
            )
        return self
