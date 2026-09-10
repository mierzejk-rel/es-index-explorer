"""Typed contracts for the question-suitability workflow."""

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
ANALYSIS_ID = "simplemode-v1"
ANALYSIS_LOCK_TAG = "analysis-lock/simple-mode-question-suitability"
MASTER_SEED = 20
ANNOTATION_BATCH_SIZE = 24
GOLD_SAMPLE_PER_STRATUM = 12
GOLD_RECODE_FRACTION = 0.20
GOLD_RECODE_MINIMUM_PER_STRATUM = 2
GOLD_RECODE_MINIMUM_DELAY_DAYS = 14
VALIDATION_BOOTSTRAP_REPLICATES = 2_000
ANNOTATOR_MODELS = ("Claude Opus 5 (high thinking)", "GPT-5.6 Sol")
GOLD_VALIDATION_FEATURES = (
    "qdmr_step_count",
    "hop_structure",
    "exhaustivity_requirement",
    "negative_conclusiveness",
    "referring_form_type",
    "answer_locality",
    "recall_orientation",
)
GOLD_EXPLORATORY_EVIDENCE_FEATURES = (
    "presupposition_load",
    "cognitive_process_level",
    "qdmr_operator_set",
    "qdmr_normalized_question",
    "demand_type",
    "specificity",
    "qdmr_applicability",
)
GOLD_STRATIFYING_FEATURES = (
    "hop_structure",
    "exhaustivity_requirement",
    "negative_conclusiveness",
    "referring_form_type",
    "answer_locality",
    "recall_orientation",
)
QDMR_OPERATOR_INVENTORY = (
    "SELECT",
    "FILTER",
    "PROJECT",
    "AGGREGATE",
    "GROUP",
    "SUPERLATIVE",
    "COMPARATIVE",
    "UNION",
    "INTERSECTION",
    "DISCARD",
    "SORT",
    "BOOLEAN",
    "ARITHMETIC",
)
QDMR_OPERATORS = frozenset(QDMR_OPERATOR_INVENTORY)
QDMR_APPLICABILITY_LEVELS = (
    "applicable",
    "applicable_after_normalisation",
    "not_applicable",
)
COGNITIVE_PROCESS_LEVELS = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)
ANNOTATION_CONSTRUCT_ANOMALY_COLUMNS = (
    "item_id",
    "item_type",
    "model_id",
    "batch_id",
    "source_text_sha256",
    "raw_response_sha256",
    "feature",
    "anomaly_code",
    "qdmr_step_count",
    "qdmr_operator_count",
    "requires_validation",
)
OUTCOME_FIELD_CONTRACT_VERSION = 1
TRACE_PFU_TABLE_COLUMNS = (
    "rubric_id",
    "rubric_order",
    "variant_id",
    "eval_dataset",
    "rubric_file_path",
    "variant_index",
    "arm_id",
    "stage",
    "run_id",
    "trace_id",
    "P",
    "F",
    "U",
    "N_r",
    "rubric_v2",
    "logged_rubric_v2",
    "rubric_v2_match",
    "logged_ordinal_grade",
    "recomputed_ordinal_grade",
    "ordinal_grade_match",
    "error_override",
    "error_scorer_judgement_count",
    "expectations_to_next_grade",
    "criterion_count_match",
    "eligible",
)
EXPLICIT_OUTCOME_FIELD_NAMES = frozenset(
    {
        "p",
        "f",
        "u",
        "state",
        "error_override",
        "rubric_v2",
        "pass_rate",
        "grade",
        "ordinal_grade",
        "tier",
    }
)
FROZEN_RECOMMENDATION_FIELD_NAMES = frozenset(
    {"tier", "uncertain", "v_r", "monte_carlo_indeterminate"}
)
RUBRIC_RECOMMENDATION_COLUMNS = (
    "rubric_id",
    "rubric_order",
    "eval_dataset",
    "tier",
    "uncertain",
    "v_r",
    "monte_carlo_indeterminate",
    "rubric_decision_depth",
    "pi_min_ge_0_75",
    "mcse_min_ge_0_75",
    "pi_min_ge_0_60",
    "mcse_min_ge_0_60",
    "pi_min_ge_0_50",
    "mcse_min_ge_0_50",
    "pi_proportion_ge_0_75",
    "pi_proportion_ge_0_60",
    "pi_proportion_ge_0_50",
    "pi_cond_min_ge_0_75",
    "mcse_cond_min_ge_0_75",
    "pi_prop_cond_gap_0_75",
    "pi_cond_min_ge_0_60",
    "mcse_cond_min_ge_0_60",
    "pi_prop_cond_gap_0_60",
    "pi_cond_min_ge_0_50",
    "mcse_cond_min_ge_0_50",
    "pi_prop_cond_gap_0_50",
    "pooled_mean_tier",
    "shrunken_variant_minimum",
    "proportion_binding_count",
    "proportion_grid_json",
    "proportion_uninformative",
    "score_band_below_0_50",
    "score_band_0_50_to_0_60",
    "score_band_0_60_to_0_75",
    "score_band_0_75_to_1_00",
    "elevated_numerical_failure_conditions",
    "laplace_adequate",
)
VARIANT_RECOMMENDATION_COLUMNS = (
    "variant_id",
    "rubric_id",
    "variant_index",
    "tier",
    "uncertain",
    "monte_carlo_indeterminate",
    "rubric_decision_depth",
    "pi_ge_0_75",
    "mcse_ge_0_75",
    "pi_ge_0_60",
    "mcse_ge_0_60",
    "pi_ge_0_50",
    "mcse_ge_0_50",
    "pi_cond_ge_0_75",
    "mcse_cond_ge_0_75",
    "pi_prop_cond_gap_0_75",
    "pi_cond_ge_0_60",
    "mcse_cond_ge_0_60",
    "pi_prop_cond_gap_0_60",
    "pi_cond_ge_0_50",
    "mcse_cond_ge_0_50",
    "pi_prop_cond_gap_0_50",
    "shrunken_mean",
    "score_band_below_0_50",
    "score_band_0_50_to_0_60",
    "score_band_0_60_to_0_75",
    "score_band_0_75_to_1_00",
    "elevated_numerical_failure_conditions",
    "laplace_adequate",
)
TRACE_SHARED_STRUCTURAL_FIELD_NAMES = frozenset(
    {
        "rubric_id",
        "rubric_order",
        "variant_id",
        "eval_dataset",
        "rubric_file_path",
        "variant_index",
    }
)
OUTCOME_ARTIFACT_SCHEMAS = MappingProxyType(
    {
        "trace_pfu_table.parquet": TRACE_PFU_TABLE_COLUMNS,
        "recommendation_table_rubric.parquet": RUBRIC_RECOMMENDATION_COLUMNS,
        "recommendation_table_variant.parquet": VARIANT_RECOMMENDATION_COLUMNS,
    }
)
_REGISTERED_OUTCOME_ARTIFACT_FIELD_NAMES = frozenset(
    field_name.casefold()
    for columns in OUTCOME_ARTIFACT_SCHEMAS.values()
    for field_name in columns
)
ANNOTATION_RESPONSE_FIELD_DENYLIST = frozenset(
    {
        *_REGISTERED_OUTCOME_ARTIFACT_FIELD_NAMES,
        *EXPLICIT_OUTCOME_FIELD_NAMES,
        *FROZEN_RECOMMENDATION_FIELD_NAMES,
    }
)
PRE_OUTCOME_INPUT_FIELD_DENYLIST = frozenset(
    ANNOTATION_RESPONSE_FIELD_DENYLIST - TRACE_SHARED_STRUCTURAL_FIELD_NAMES
)
# Kept as the public short name for callers that validate pre-outcome source tables.
OUTCOME_FIELD_DENYLIST = PRE_OUTCOME_INPUT_FIELD_DENYLIST
PROTECTED_RECOMMENDATION_ARTIFACT_NAMES = frozenset(
    {"recommendation_table_rubric.parquet", "recommendation_table_variant.parquet"}
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
    GOLD_INGEST_INITIAL = "gold-ingest-initial"
    GOLD_RECODE_RELEASE = "gold-recode-release"
    GOLD_INGEST_PROVISIONAL = "gold-ingest-provisional"
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


class DslGateStatus(StrEnum):
    """Design-based surrogate-label derivation outcomes."""

    ESTABLISHED = "DSL_ESTABLISHED"
    NOT_ESTABLISHED = "DSL_NOT_ESTABLISHED"


class AnnotatorKind(StrEnum):
    """Permitted gold-workflow annotator kinds."""

    HUMAN = "human"
    PROVISIONAL_LLM = "provisional_llm"


class AnnotationConstructAnomaly(StrEnum):
    """Outcome-blind construct anomalies requiring later validity review."""

    QDMR_OPERATOR_COUNT_EXCEEDS_STEP_COUNT = "QDMR_OPERATOR_COUNT_EXCEEDS_STEP_COUNT"


class AnalysisFlag(StrEnum):
    """Cross-segment diagnostic flags reserved by the specification."""

    METRIC_UNDEFINED_DEGENERATE_GOLD = "METRIC_UNDEFINED_DEGENERATE_GOLD"
    MONTE_CARLO_INDETERMINATE = "MONTE_CARLO_INDETERMINATE"
    BH_INDETERMINATE = "BH_INDETERMINATE"
    DISPUTED = "DISPUTED"


class SuitabilityTier(StrEnum):
    """Layer 1 recommendation tiers and numerical indeterminacy state."""

    SUITABLE = "SUITABLE"
    PROMISING = "PROMISING"
    BORDERLINE = "BORDERLINE"
    NOT_SUITABLE = "NOT_SUITABLE"
    MONTE_CARLO_INDETERMINATE = "MONTE_CARLO_INDETERMINATE"


class Layer1EventKind(StrEnum):
    """Primary Layer 1 event kinds used for adaptive refinement."""

    WORST_VARIANT = "WORST_VARIANT"
    SINGLE_VARIANT = "SINGLE_VARIANT"
    PROPORTION_KAPPA_0_75 = "PROPORTION_KAPPA_0_75"


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
