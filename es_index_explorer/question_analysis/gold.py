"""Outcome-blind gold sampling and human-label ingestion."""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from es_index_explorer.question_analysis.contracts import (
    ANNOTATION_RESPONSE_FIELD_DENYLIST,
    GOLD_RECODE_FRACTION,
    GOLD_RECODE_MINIMUM_DELAY_DAYS,
    GOLD_RECODE_MINIMUM_PER_STRATUM,
    GOLD_SAMPLE_PER_STRATUM,
    GOLD_STRATIFYING_FEATURES,
    GOLD_VALIDATION_FEATURES,
    AnnotatorKind,
    ArtifactMetadata,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.seeds import rng_for
from es_index_explorer.question_analysis.storage import (
    canonical_json_bytes,
    sha256_file,
    versioned_frame,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

GOLD_SAMPLE_TABLE = "tables/gold_sample.parquet"
GOLD_INITIAL_LABELS_TABLE = "tables/gold_initial_labels.parquet"
GOLD_LABELS_TABLE = "tables/gold_labels.parquet"
GOLD_PROVISIONAL_RECODE_TABLE = "tables/gold_provisional_recode.parquet"
GOLD_DIRECTORY = "gold"
ADJUDICATION_TEMPLATE = f"{GOLD_DIRECTORY}/adjudication.csv"
RECODE_TEMPLATE = f"{GOLD_DIRECTORY}/delayed_recode.csv"
RECODE_MAPPING = f"{GOLD_DIRECTORY}/delayed_recode_mapping.json"
BUNDLE_MANIFEST = f"{GOLD_DIRECTORY}/bundle_manifest.json"
RECODE_MANIFEST = f"{GOLD_DIRECTORY}/delayed_recode_manifest.json"
VALUE_GUIDE = f"{GOLD_DIRECTORY}/value_guide.json"
GOLD_INITIAL_INGEST_VERIFICATION = "gold_initial_ingest_verification.json"
GOLD_INGEST_VERIFICATION = "gold_ingest_verification.json"
PROVISIONAL_DIRECTORY = f"{GOLD_DIRECTORY}/provisional"
PROVISIONAL_COMPLETED_RECODE = f"{PROVISIONAL_DIRECTORY}/completed_recode.csv"
PROVISIONAL_RAW_RESPONSE = f"{PROVISIONAL_DIRECTORY}/raw_response.bin"
PROVISIONAL_PROVENANCE = f"{PROVISIONAL_DIRECTORY}/provenance.json"
PROVISIONAL_VERIFICATION = "gold_provisional_recode_verification.json"
PROVISIONAL_UNRESOLVED = f"{PROVISIONAL_DIRECTORY}/unresolved_validation_items.json"
DISPUTED = "DISPUTED"

GOLD_BUNDLE_FEATURES = (
    "answer_locality",
    "cognitive_process_level",
    "demand_type",
    "exhaustivity_requirement",
    "hop_structure",
    "negative_conclusiveness",
    "presupposition_load",
    "qdmr_applicability",
    "qdmr_normalized_question",
    "qdmr_operator_set",
    "qdmr_step_count",
    "recall_orientation",
    "referring_form_type",
    "specificity",
)

FEATURE_VALUES: Mapping[str, frozenset[object]] = {
    "answer_locality": frozenset(
        {
            "single_passage",
            "cross_document_aggregation",
            "not_classifiable_binary_locality",
        }
    ),
    "cognitive_process_level": frozenset(
        {"remember", "understand", "apply", "analyze", "evaluate", "create"}
    ),
    "demand_type": frozenset(
        {
            "verbatim_citation",
            "entity_identification",
            "relational_claim",
            "temporal_ordering",
            "quantification",
            "evaluative_synthesis",
        }
    ),
    "exhaustivity_requirement": frozenset(
        {"mention_some", "weakly_exhaustive", "mention_all"}
    ),
    "hop_structure": frozenset({"atomic", "bridge", "comparison", "intersection"}),
    "negative_conclusiveness": frozenset({True, False}),
    "presupposition_load": frozenset({True, False}),
    "qdmr_applicability": frozenset(
        {"applicable", "applicable_after_normalisation", "not_applicable"}
    ),
    "recall_orientation": frozenset({"precision_oriented", "recall_oriented"}),
    "referring_form_type": frozenset(
        {"full_name_form", "alias_or_handle", "email_address", "no_focal_referent"}
    ),
    "specificity": frozenset({"specific", "broad"}),
}

CSV_COLUMNS = (
    "local_item_id",
    "item_type",
    "source_text",
    "feature",
    "first_model_id",
    "first_value_json",
    "second_model_id",
    "second_value_json",
    "human_value_json",
    "comment",
)
RECODE_CSV_COLUMNS = (
    "recode_item_id",
    "item_type",
    "source_text",
    "feature",
    "recode_value_json",
    "comment",
)


class InitialHumanGoldProvenance(BaseModel):
    """Describe qualifying initial human adjudication provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    annotator_kind: Literal[AnnotatorKind.HUMAN]
    annotator_id: str = Field(min_length=1)
    annotator_role: str = Field(min_length=1)
    expertise: str = Field(min_length=1)
    qualification: str = Field(min_length=1)
    annotation_environment: str = Field(min_length=1)
    bundle_manifest_sha256: str = Field(min_length=64, max_length=64)
    completed_adjudication_sha256: str = Field(min_length=64, max_length=64)
    schema_sha256: str = Field(min_length=64, max_length=64)
    codebook_sha256: str = Field(min_length=64, max_length=64)
    decision_ledger_sha256: str = Field(min_length=64, max_length=64)
    started_at: datetime
    completed_at: datetime
    outcome_blindness_confirmed: Literal[True]
    notes: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> "InitialHumanGoldProvenance":
        """Require a nonnegative initial annotation interval."""
        if self.completed_at < self.started_at:
            raise ValueError("Initial annotation completion precedes its start")
        return self


class HumanRecodeProvenance(BaseModel):
    """Describe qualifying delayed human re-code provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    annotator_kind: Literal[AnnotatorKind.HUMAN]
    annotator_id: str = Field(min_length=1)
    annotator_role: str = Field(min_length=1)
    expertise: str = Field(min_length=1)
    qualification: str = Field(min_length=1)
    annotation_environment: str = Field(min_length=1)
    recode_manifest_sha256: str = Field(min_length=64, max_length=64)
    completed_recode_sha256: str = Field(min_length=64, max_length=64)
    initial_labels_sha256: str = Field(min_length=64, max_length=64)
    schema_sha256: str = Field(min_length=64, max_length=64)
    codebook_sha256: str = Field(min_length=64, max_length=64)
    decision_ledger_sha256: str = Field(min_length=64, max_length=64)
    started_at: datetime
    completed_at: datetime
    context_isolated: Literal[True]
    context_isolation_description: str = Field(min_length=1)
    outcome_blindness_confirmed: Literal[True]
    notes: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> "HumanRecodeProvenance":
        """Require a nonnegative re-code annotation interval."""
        if self.completed_at < self.started_at:
            raise ValueError("Re-code completion precedes its start")
        return self


class ProvisionalLlmProvenance(BaseModel):
    """Describe non-human provisional re-code evidence without authorizing unlock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    annotator_kind: Literal[AnnotatorKind.PROVISIONAL_LLM]
    provider: str = Field(min_length=1)
    interface: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    recode_manifest_sha256: str = Field(min_length=64, max_length=64)
    completed_recode_sha256: str = Field(min_length=64, max_length=64)
    initial_labels_sha256: str = Field(min_length=64, max_length=64)
    schema_sha256: str = Field(min_length=64, max_length=64)
    codebook_sha256: str = Field(min_length=64, max_length=64)
    decision_ledger_sha256: str = Field(min_length=64, max_length=64)
    parameters: dict[str, str | int | float | bool | None]
    session_id: str = Field(min_length=1)
    context_isolated: Literal[True]
    context_isolation_description: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime
    agent_id: str | None = None
    run_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    status: str = Field(min_length=1)
    raw_response_sha256: str = Field(min_length=64, max_length=64)
    outcome_blindness_confirmed: Literal[True]

    @model_validator(mode="after")
    def validate_timing(self) -> "ProvisionalLlmProvenance":
        """Require a nonnegative provisional annotation interval."""
        if self.completed_at < self.started_at:
            raise ValueError("Provisional re-code completion precedes its start")
        return self


@dataclass(frozen=True, slots=True)
class GoldSampleResult:
    """Hold deterministic gold-sample tables and exchange files."""

    sample: pd.DataFrame
    adjudication_csv: bytes
    recode_csv: bytes
    recode_mapping: dict[str, object]
    value_guide: dict[str, object]
    manifest: dict[str, object]


def build_gold_sample(
    agreement: pd.DataFrame,
    features: pd.DataFrame,
    *,
    sampling_rng: np.random.Generator | None = None,
    recode_rng: np.random.Generator | None = None,
) -> GoldSampleResult:
    """Build the strict-partition gold sample and human exchange bundles."""
    _require_columns(
        agreement,
        {
            "item_id",
            "item_type",
            "feature",
            "first_model_id",
            "second_model_id",
            "first_value",
            "second_value",
            "agreed",
        },
        "annotation agreement",
    )
    _require_columns(
        features,
        {"item_id", "item_type", "source_text", "text_sha256"},
        "deterministic features",
    )
    if features["item_id"].duplicated().any():
        raise GateFailureError("Gold source features contain duplicate item IDs")

    candidates = _candidate_strata(agreement)
    assignments = _strict_partition(candidates)
    sample = _draw_stratified_sample(
        assignments,
        sampling_rng or rng_for("gold_sampling"),
    )
    sample = sample.merge(
        features[["item_id", "item_type", "source_text", "text_sha256"]],
        on=["item_id", "item_type"],
        how="left",
        validate="one_to_one",
    )
    if sample[["source_text", "text_sha256"]].isna().any().any():
        raise GateFailureError("Gold sample cannot be joined to source text")

    bundle_rows = _bundle_rows(sample, agreement)
    recode_items = _select_recode_items(
        sample,
        recode_rng or rng_for("gold_recode_sampling"),
    )
    adjudication_rows = _adjudication_rows(bundle_rows)
    recode_rows, recode_mapping = _recode_rows(bundle_rows, recode_items)
    adjudication_csv = _csv_bytes(CSV_COLUMNS, adjudication_rows)
    recode_csv = _csv_bytes(RECODE_CSV_COLUMNS, recode_rows)
    value_guide = _value_guide()
    manifest = _bundle_manifest(
        sample,
        adjudication_csv,
        recode_csv,
        recode_mapping,
        value_guide,
    )
    return GoldSampleResult(
        sample=sample.drop(columns="source_text"),
        adjudication_csv=adjudication_csv,
        recode_csv=recode_csv,
        recode_mapping=recode_mapping,
        value_guide=value_guide,
        manifest=manifest,
    )


def run_gold_sample(workspace: AnalysisWorkspace) -> tuple[ArtifactMetadata, ...]:
    """Persist deterministic gold sampling and outcome-blind human bundles."""
    agreement = pd.read_parquet(
        workspace.store.path_for("tables/annotation_agreement.parquet")
    ).drop(columns="artifact_schema_version")
    features = pd.read_parquet(
        workspace.store.path_for("tables/features_deterministic.parquet")
    ).drop(columns="artifact_schema_version")
    result = build_gold_sample(agreement, features)
    manifest = {
        **result.manifest,
        "schema_sha256": _gold_schema_sha256(),
        "codebook_sha256": _locked_resource_sha(workspace, "segment3_codebook"),
        "decision_ledger_sha256": _locked_resource_sha(
            workspace, "segment5_decision_ledger"
        ),
    }
    return (
        workspace.store.write_parquet(
            GOLD_SAMPLE_TABLE,
            versioned_frame(result.sample),
            created_by=WorkflowCommand.GOLD_SAMPLE,
        ),
        workspace.store.write_bytes(
            ADJUDICATION_TEMPLATE,
            result.adjudication_csv,
            created_by=WorkflowCommand.GOLD_SAMPLE,
        ),
        workspace.store.write_json(
            VALUE_GUIDE,
            result.value_guide,
            created_by=WorkflowCommand.GOLD_SAMPLE,
        ),
        workspace.store.write_json(
            BUNDLE_MANIFEST,
            manifest,
            created_by=WorkflowCommand.GOLD_SAMPLE,
        ),
    )


def run_gold_ingest_initial(
    workspace: AnalysisWorkspace,
    adjudication_csv: Path,
    provenance_json: Path,
) -> tuple[ArtifactMetadata, ...]:
    """Validate and checkpoint the initial human adjudication."""
    provenance = _load_initial_human_provenance(provenance_json)
    manifest_path = workspace.store.path_for(BUNDLE_MANIFEST)
    if sha256_file(manifest_path) != provenance.bundle_manifest_sha256:
        raise MalformedInputError(
            "Initial provenance does not match the gold bundle manifest"
        )
    if sha256_file(adjudication_csv) != provenance.completed_adjudication_sha256:
        raise MalformedInputError(
            "Initial provenance does not match the completed adjudication CSV"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_locked_provenance_hashes(
        workspace,
        schema_sha256=provenance.schema_sha256,
        codebook_sha256=provenance.codebook_sha256,
        decision_ledger_sha256=provenance.decision_ledger_sha256,
        manifest=manifest,
    )
    checkpoint_time = _trusted_step_start(
        workspace, WorkflowCommand.GOLD_INGEST_INITIAL
    )
    sample_time = _step_completed_at(workspace, WorkflowCommand.GOLD_SAMPLE)
    if provenance.started_at < sample_time:
        raise GateFailureError("Initial annotation started before gold bundle emission")
    if provenance.completed_at > checkpoint_time:
        raise GateFailureError("Initial annotation completion is future-dated")
    initial_template = _read_csv(
        workspace.store.path_for(ADJUDICATION_TEMPLATE), CSV_COLUMNS
    )
    initial = _read_csv(adjudication_csv, CSV_COLUMNS)
    labels = normalize_initial_human_labels(
        initial_template,
        initial,
        provenance,
    )
    sample = pd.read_parquet(workspace.store.path_for(GOLD_SAMPLE_TABLE)).drop(
        columns="artifact_schema_version"
    )
    local_item_map = {
        f"gold-{index + 1:04d}": {
            "item_id": str(row["item_id"]),
            "stratum_id": str(row["stratum_id"]),
            "inclusion_probability": float(row["inclusion_probability"]),
        }
        for index, row in enumerate(
            sample.sort_values("sample_order", kind="stable").to_dict(orient="records")
        )
    }
    labels = labels.assign(
        item_id=labels["local_item_id"].map(
            lambda local_id: local_item_map[str(local_id)]["item_id"]
        ),
        stratum_id=labels["local_item_id"].map(
            lambda local_id: local_item_map[str(local_id)]["stratum_id"]
        ),
        inclusion_probability=labels["local_item_id"].map(
            lambda local_id: local_item_map[str(local_id)]["inclusion_probability"]
        ),
        initial_received_at=checkpoint_time,
    )
    labels = labels[
        [
            "item_id",
            "local_item_id",
            "item_type",
            "stratum_id",
            "inclusion_probability",
            "feature",
            "human_gold_value_json",
            "human_comment",
            "initial_started_at",
            "initial_completed_at",
            "initial_received_at",
            "annotator_kind",
            "annotator_id",
            "annotator_role",
            "annotator_expertise",
            "annotator_qualification",
            "annotation_environment",
        ]
    ]
    verification = {
        "schema_version": 1,
        "passed": True,
        "annotator_kind": provenance.annotator_kind,
        "initial_label_row_count": len(labels),
        "initial_received_at": checkpoint_time.isoformat(),
        "minimum_recode_delay_days": GOLD_RECODE_MINIMUM_DELAY_DAYS,
        "provisional_evidence_can_unlock": False,
        "bundle_manifest_sha256": provenance.bundle_manifest_sha256,
        "source_manifest_schema_version": manifest.get("schema_version"),
    }
    return (
        workspace.store.write_parquet(
            GOLD_INITIAL_LABELS_TABLE,
            versioned_frame(labels),
            created_by=WorkflowCommand.GOLD_INGEST_INITIAL,
        ),
        workspace.store.write_json(
            GOLD_INITIAL_INGEST_VERIFICATION,
            verification,
            created_by=WorkflowCommand.GOLD_INGEST_INITIAL,
        ),
    )


def run_gold_recode_release(
    workspace: AnalysisWorkspace,
) -> tuple[ArtifactMetadata, ...]:
    """Release the blind re-code bundle after the trusted delay."""
    release_time = _trusted_step_start(workspace, WorkflowCommand.GOLD_RECODE_RELEASE)
    initial_completed = _step_completed_at(
        workspace, WorkflowCommand.GOLD_INGEST_INITIAL
    )
    minimum_release = initial_completed + timedelta(days=GOLD_RECODE_MINIMUM_DELAY_DAYS)
    if release_time < minimum_release:
        raise GateFailureError(
            f"Blind re-code cannot be released before {minimum_release.isoformat()}"
        )
    agreement = pd.read_parquet(
        workspace.store.path_for("tables/annotation_agreement.parquet")
    ).drop(columns="artifact_schema_version")
    features = pd.read_parquet(
        workspace.store.path_for("tables/features_deterministic.parquet")
    ).drop(columns="artifact_schema_version")
    result = build_gold_sample(agreement, features)
    persisted_sample = pd.read_parquet(
        workspace.store.path_for(GOLD_SAMPLE_TABLE)
    ).drop(columns="artifact_schema_version")
    try:
        pd.testing.assert_frame_equal(result.sample, persisted_sample)
    except AssertionError as error:
        raise GateFailureError(
            "Recomputed delayed re-code selection conflicts with the gold sample"
        ) from error
    bundle_manifest = json.loads(
        workspace.store.path_for(BUNDLE_MANIFEST).read_text(encoding="utf-8")
    )
    recode_csv_sha256 = sha256(result.recode_csv).hexdigest()
    recode_mapping_sha256 = sha256(
        canonical_json_bytes(result.recode_mapping)
    ).hexdigest()
    if (
        bundle_manifest.get("recode_csv_sha256") != recode_csv_sha256
        or bundle_manifest.get("recode_mapping_sha256") != recode_mapping_sha256
    ):
        raise GateFailureError("Delayed re-code release differs from the locked bundle")
    recode_manifest = {
        "schema_version": 1,
        "recode_csv_sha256": recode_csv_sha256,
        "recode_mapping_sha256": recode_mapping_sha256,
        "initial_labels_sha256": sha256_file(
            workspace.store.path_for(GOLD_INITIAL_LABELS_TABLE)
        ),
        "schema_sha256": _gold_schema_sha256(),
        "codebook_sha256": _locked_resource_sha(workspace, "segment3_codebook"),
        "decision_ledger_sha256": _locked_resource_sha(
            workspace, "segment5_decision_ledger"
        ),
        "released_at": release_time.isoformat(),
        "minimum_recode_delay_days": GOLD_RECODE_MINIMUM_DELAY_DAYS,
        "outcome_fields_included": [],
    }
    return (
        workspace.store.write_bytes(
            RECODE_TEMPLATE,
            result.recode_csv,
            created_by=WorkflowCommand.GOLD_RECODE_RELEASE,
        ),
        workspace.store.write_json(
            RECODE_MAPPING,
            result.recode_mapping,
            created_by=WorkflowCommand.GOLD_RECODE_RELEASE,
        ),
        workspace.store.write_json(
            RECODE_MANIFEST,
            recode_manifest,
            created_by=WorkflowCommand.GOLD_RECODE_RELEASE,
        ),
    )


def run_gold_ingest(
    workspace: AnalysisWorkspace,
    recode_csv: Path,
    provenance_json: Path,
) -> tuple[ArtifactMetadata, ...]:
    """Validate delayed human labels and persist canonical gold evidence."""
    provenance = _load_recode_human_provenance(provenance_json)
    recode_manifest_path = workspace.store.path_for(RECODE_MANIFEST)
    if sha256_file(recode_manifest_path) != provenance.recode_manifest_sha256:
        raise MalformedInputError("Re-code provenance does not match its manifest")
    if sha256_file(recode_csv) != provenance.completed_recode_sha256:
        raise MalformedInputError(
            "Re-code provenance does not match the completed re-code CSV"
        )
    recode_manifest = json.loads(recode_manifest_path.read_text(encoding="utf-8"))
    _validate_locked_provenance_hashes(
        workspace,
        schema_sha256=provenance.schema_sha256,
        codebook_sha256=provenance.codebook_sha256,
        decision_ledger_sha256=provenance.decision_ledger_sha256,
        manifest=recode_manifest,
    )
    initial_path = workspace.store.path_for(GOLD_INITIAL_LABELS_TABLE)
    if sha256_file(
        initial_path
    ) != provenance.initial_labels_sha256 or provenance.initial_labels_sha256 != str(
        recode_manifest.get("initial_labels_sha256")
    ):
        raise MalformedInputError("Re-code provenance does not match initial labels")
    release_time = _step_completed_at(workspace, WorkflowCommand.GOLD_RECODE_RELEASE)
    ingest_time = _trusted_step_start(workspace, WorkflowCommand.GOLD_INGEST)
    if provenance.started_at < release_time:
        raise GateFailureError("Human re-code started before blind bundle release")
    if provenance.completed_at > ingest_time:
        raise GateFailureError("Human re-code completion is future-dated")
    recode_template = _read_csv(
        workspace.store.path_for(RECODE_TEMPLATE), RECODE_CSV_COLUMNS
    )
    completed_recode = _read_csv(recode_csv, RECODE_CSV_COLUMNS)
    mapping = json.loads(
        workspace.store.path_for(RECODE_MAPPING).read_text(encoding="utf-8")
    )
    initial_labels = pd.read_parquet(initial_path).drop(
        columns="artifact_schema_version"
    )
    labels = attach_human_recode(
        initial_labels,
        recode_template,
        completed_recode,
        mapping,
        provenance,
        received_at=ingest_time,
    )
    verification = {
        "schema_version": 1,
        "passed": True,
        "annotator_kind": provenance.annotator_kind,
        "gold_label_row_count": len(labels),
        "recode_row_count": int(labels["human_recode_value_json"].notna().sum()),
        "recode_started_at": provenance.started_at.isoformat(),
        "recode_completed_at": provenance.completed_at.isoformat(),
        "recode_received_at": ingest_time.isoformat(),
        "minimum_recode_delay_days": GOLD_RECODE_MINIMUM_DELAY_DAYS,
        "provisional_evidence_can_unlock": False,
        "recode_manifest_sha256": provenance.recode_manifest_sha256,
    }
    return (
        workspace.store.write_parquet(
            GOLD_LABELS_TABLE,
            versioned_frame(labels),
            created_by=WorkflowCommand.GOLD_INGEST,
        ),
        workspace.store.write_json(
            GOLD_INGEST_VERIFICATION,
            verification,
            created_by=WorkflowCommand.GOLD_INGEST,
        ),
    )


def run_gold_ingest_provisional(
    workspace: AnalysisWorkspace,
    recode_csv: Path,
    provenance_json: Path,
    raw_response: Path,
) -> tuple[ArtifactMetadata, ...]:
    """Persist a provisional LLM re-code without advancing the human gate."""
    provenance = _load_provisional_llm_provenance(provenance_json)
    recode_manifest_path = workspace.store.path_for(RECODE_MANIFEST)
    if sha256_file(recode_manifest_path) != provenance.recode_manifest_sha256:
        raise MalformedInputError(
            "Provisional provenance does not match the re-code manifest"
        )
    if sha256_file(recode_csv) != provenance.completed_recode_sha256:
        raise MalformedInputError(
            "Provisional provenance does not match the completed re-code CSV"
        )
    if sha256_file(raw_response) != provenance.raw_response_sha256:
        raise MalformedInputError(
            "Provisional provenance does not match the raw response"
        )
    recode_manifest = json.loads(recode_manifest_path.read_text(encoding="utf-8"))
    _validate_locked_provenance_hashes(
        workspace,
        schema_sha256=provenance.schema_sha256,
        codebook_sha256=provenance.codebook_sha256,
        decision_ledger_sha256=provenance.decision_ledger_sha256,
        manifest=recode_manifest,
    )
    initial_path = workspace.store.path_for(GOLD_INITIAL_LABELS_TABLE)
    if sha256_file(
        initial_path
    ) != provenance.initial_labels_sha256 or provenance.initial_labels_sha256 != str(
        recode_manifest.get("initial_labels_sha256")
    ):
        raise MalformedInputError(
            "Provisional provenance does not match initial labels"
        )
    release_time = _step_completed_at(workspace, WorkflowCommand.GOLD_RECODE_RELEASE)
    ingest_time = _trusted_step_start(
        workspace, WorkflowCommand.GOLD_INGEST_PROVISIONAL
    )
    if provenance.started_at < release_time:
        raise GateFailureError(
            "Provisional re-code started before blind bundle release"
        )
    if provenance.completed_at > ingest_time:
        raise GateFailureError("Provisional re-code completion is future-dated")

    recode_template = _read_csv(
        workspace.store.path_for(RECODE_TEMPLATE), RECODE_CSV_COLUMNS
    )
    completed_recode = _read_csv(recode_csv, RECODE_CSV_COLUMNS)
    mapping = json.loads(
        workspace.store.path_for(RECODE_MAPPING).read_text(encoding="utf-8")
    )
    initial_labels = pd.read_parquet(initial_path).drop(
        columns="artifact_schema_version"
    )
    labels = normalize_provisional_recode(
        initial_labels,
        recode_template,
        completed_recode,
        mapping,
        provenance,
        received_at=ingest_time,
    )
    verification = {
        **validate_provisional_llm_provenance(provenance),
        "provisional_label_row_count": len(labels),
        "recode_manifest_sha256": provenance.recode_manifest_sha256,
        "completed_recode_sha256": provenance.completed_recode_sha256,
        "raw_response_sha256": provenance.raw_response_sha256,
        "received_at": ingest_time.isoformat(),
        "outcome_fields_included": [],
    }
    unresolved = {
        "schema_version": 1,
        "items": [
            {
                "code": "INDEPENDENT_HUMAN_EXPERT_VERIFICATION_REQUIRED",
                "blocking": True,
                "reason": (
                    "Provisional LLM re-code is not human test-retest evidence."
                ),
            }
        ],
    }
    return (
        workspace.store.write_parquet(
            GOLD_PROVISIONAL_RECODE_TABLE,
            versioned_frame(labels),
            created_by=WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        ),
        workspace.store.write_bytes(
            PROVISIONAL_COMPLETED_RECODE,
            recode_csv.read_bytes(),
            created_by=WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        ),
        workspace.store.write_bytes(
            PROVISIONAL_RAW_RESPONSE,
            raw_response.read_bytes(),
            created_by=WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        ),
        workspace.store.write_bytes(
            PROVISIONAL_PROVENANCE,
            canonical_json_bytes(provenance),
            created_by=WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        ),
        workspace.store.write_json(
            PROVISIONAL_VERIFICATION,
            verification,
            created_by=WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        ),
        workspace.store.write_json(
            PROVISIONAL_UNRESOLVED,
            unresolved,
            created_by=WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        ),
    )


def normalize_initial_human_labels(
    initial_template: Sequence[Mapping[str, str]],
    initial_rows: Sequence[Mapping[str, str]],
    provenance: InitialHumanGoldProvenance,
) -> pd.DataFrame:
    """Validate edited CSV rows and normalize initial human labels."""
    initial_by_key = _unique_rows(
        initial_rows, ("local_item_id", "feature"), "initial adjudication"
    )
    initial_template_by_key = _unique_rows(
        initial_template,
        ("local_item_id", "feature"),
        "initial adjudication template",
    )
    if set(initial_by_key) != set(initial_template_by_key):
        raise MalformedInputError(
            "Initial adjudication row identity differs from template"
        )
    normalized: list[dict[str, object]] = []
    immutable_columns = tuple(
        column
        for column in CSV_COLUMNS
        if column not in {"human_value_json", "comment"}
    )
    for key in sorted(initial_template_by_key):
        template = initial_template_by_key[key]
        completed = initial_by_key[key]
        _require_immutable_csv_fields(template, completed, immutable_columns)
        value = _parse_human_value(completed["feature"], completed["human_value_json"])
        normalized.append(
            {
                "local_item_id": completed["local_item_id"],
                "item_type": completed["item_type"],
                "feature": completed["feature"],
                "human_gold_value_json": _json_value(value),
                "human_comment": completed["comment"] or None,
                "initial_started_at": provenance.started_at,
                "initial_completed_at": provenance.completed_at,
                "annotator_kind": provenance.annotator_kind,
                "annotator_id": provenance.annotator_id,
                "annotator_role": provenance.annotator_role,
                "annotator_expertise": provenance.expertise,
                "annotator_qualification": provenance.qualification,
                "annotation_environment": provenance.annotation_environment,
            }
        )

    return (
        pd.DataFrame(normalized)
        .sort_values(["local_item_id", "feature"], kind="stable")
        .reset_index(drop=True)
    )


def normalize_provisional_recode(
    initial_labels: pd.DataFrame,
    recode_template: Sequence[Mapping[str, str]],
    recode_rows: Sequence[Mapping[str, str]],
    recode_mapping: Mapping[str, object],
    provenance: ProvisionalLlmProvenance,
    *,
    received_at: datetime,
) -> pd.DataFrame:
    """Validate and normalize physically separate provisional LLM labels."""
    recode_by_key = _unique_rows(
        recode_rows, ("recode_item_id",), "provisional re-code"
    )
    template_by_key = _unique_rows(
        recode_template, ("recode_item_id",), "delayed re-code template"
    )
    if set(recode_by_key) != set(template_by_key):
        raise MalformedInputError(
            "Provisional re-code row identity differs from template"
        )
    mapping_rows = recode_mapping.get("rows")
    if not isinstance(mapping_rows, list):
        raise MalformedInputError("Delayed re-code mapping is invalid")
    mapping_by_key: dict[tuple[str, ...], Mapping[str, object]] = {
        (str(row["recode_item_id"]),): row
        for row in mapping_rows
        if isinstance(row, dict) and "recode_item_id" in row
    }
    if len(mapping_by_key) != len(mapping_rows) or set(mapping_by_key) != set(
        template_by_key
    ):
        raise MalformedInputError("Delayed re-code mapping keys are invalid")
    initial_by_key = {
        (str(row["local_item_id"]), str(row["feature"])): row
        for row in initial_labels.to_dict(orient="records")
    }
    immutable_columns = tuple(
        column
        for column in RECODE_CSV_COLUMNS
        if column not in {"recode_value_json", "comment"}
    )
    normalized: list[dict[str, object]] = []
    for key in sorted(template_by_key):
        template = template_by_key[key]
        completed = recode_by_key[key]
        _require_immutable_csv_fields(template, completed, immutable_columns)
        target = mapping_by_key[key]
        local_key = str(target["local_item_id"])
        feature = str(completed["feature"])
        initial = initial_by_key.get((local_key, feature))
        if initial is None or str(target.get("feature")) != feature:
            raise MalformedInputError(
                "Provisional re-code mapping conflicts with initial labels"
            )
        value = _parse_human_value(feature, completed["recode_value_json"])
        conditional_probability = target.get("recode_conditional_probability")
        joint_probability = target.get("recode_joint_inclusion_probability")
        if not isinstance(conditional_probability, (int, float)) or not isinstance(
            joint_probability, (int, float)
        ):
            raise MalformedInputError(
                "Provisional re-code mapping probabilities are invalid"
            )
        normalized.append(
            {
                "item_id": str(initial["item_id"]),
                "local_item_id": local_key,
                "recode_item_id": key[0],
                "item_type": str(initial["item_type"]),
                "stratum_id": str(initial["stratum_id"]),
                "inclusion_probability": float(initial["inclusion_probability"]),
                "feature": feature,
                "provisional_value_json": _json_value(value),
                "provisional_comment": completed["comment"] or None,
                "recode_conditional_probability": float(conditional_probability),
                "recode_joint_inclusion_probability": float(joint_probability),
                "annotator_kind": provenance.annotator_kind,
                "provider": provenance.provider,
                "interface": provenance.interface,
                "model": provenance.model,
                "model_version": provenance.model_version,
                "prompt_sha256": provenance.prompt_sha256,
                "session_id": provenance.session_id,
                "context_isolated": provenance.context_isolated,
                "context_isolation_description": (
                    provenance.context_isolation_description
                ),
                "started_at": provenance.started_at,
                "completed_at": provenance.completed_at,
                "received_at": received_at,
                "raw_response_sha256": provenance.raw_response_sha256,
                "qualifies_as_human_recode": False,
                "can_unlock_outcome_modeling": False,
            }
        )
    return (
        pd.DataFrame(normalized)
        .sort_values(["recode_item_id"], kind="stable")
        .reset_index(drop=True)
    )


def attach_human_recode(
    initial_labels: pd.DataFrame,
    recode_template: Sequence[Mapping[str, str]],
    recode_rows: Sequence[Mapping[str, str]],
    recode_mapping: Mapping[str, object],
    provenance: HumanRecodeProvenance,
    *,
    received_at: datetime,
) -> pd.DataFrame:
    """Attach validated delayed human re-code values to initial labels."""
    recode_by_key = _unique_rows(recode_rows, ("recode_item_id",), "delayed re-code")
    recode_template_by_key = _unique_rows(
        recode_template, ("recode_item_id",), "delayed re-code template"
    )
    if set(recode_by_key) != set(recode_template_by_key):
        raise MalformedInputError("Delayed re-code row identity differs from template")
    mapping_rows = recode_mapping.get("rows")
    if not isinstance(mapping_rows, list):
        raise MalformedInputError("Delayed re-code mapping is invalid")
    mapping_by_key = {
        (str(row["recode_item_id"]),): row
        for row in mapping_rows
        if isinstance(row, dict) and "recode_item_id" in row
    }
    if len(mapping_by_key) != len(mapping_rows) or set(mapping_by_key) != set(
        recode_template_by_key
    ):
        raise MalformedInputError("Delayed re-code mapping keys are invalid")
    normalized = initial_labels.assign(
        human_recode_value_json=None,
        human_recode_comment=None,
        recode_started_at=None,
        recode_completed_at=None,
        recode_received_at=None,
        recode_annotator_id=None,
        recode_annotator_role=None,
        recode_annotator_expertise=None,
        recode_annotator_qualification=None,
        recode_annotation_environment=None,
        recode_conditional_probability=None,
        recode_joint_inclusion_probability=None,
    ).to_dict(orient="records")
    normalized_by_key = {
        (str(row["local_item_id"]), str(row["feature"])): row for row in normalized
    }
    immutable_recode_columns = tuple(
        column
        for column in RECODE_CSV_COLUMNS
        if column not in {"recode_value_json", "comment"}
    )
    for key in sorted(recode_template_by_key):
        template = recode_template_by_key[key]
        completed = recode_by_key[key]
        _require_immutable_csv_fields(template, completed, immutable_recode_columns)
        target = mapping_by_key.get(key)
        if target is None:
            raise MalformedInputError("Delayed re-code mapping is incomplete")
        local_key = str(target["local_item_id"])
        feature = str(completed["feature"])
        row = normalized_by_key.get((local_key, feature))
        if row is None or str(target.get("feature")) != feature:
            raise MalformedInputError(
                "Delayed re-code mapping conflicts with initial labels"
            )
        value = _parse_human_value(completed["feature"], completed["recode_value_json"])
        row["human_recode_value_json"] = _json_value(value)
        row["human_recode_comment"] = completed["comment"] or None
        row["recode_started_at"] = provenance.started_at
        row["recode_completed_at"] = provenance.completed_at
        row["recode_received_at"] = received_at
        row["recode_annotator_id"] = provenance.annotator_id
        row["recode_annotator_role"] = provenance.annotator_role
        row["recode_annotator_expertise"] = provenance.expertise
        row["recode_annotator_qualification"] = provenance.qualification
        row["recode_annotation_environment"] = provenance.annotation_environment
        row["recode_conditional_probability"] = float(
            target["recode_conditional_probability"]
        )
        row["recode_joint_inclusion_probability"] = float(
            target["recode_joint_inclusion_probability"]
        )

    return (
        pd.DataFrame(normalized)
        .sort_values(["local_item_id", "feature"], kind="stable")
        .reset_index(drop=True)
    )


def validate_provisional_llm_provenance(
    provenance: ProvisionalLlmProvenance,
) -> dict[str, object]:
    """Return explicit non-unlock evidence for a provisional LLM re-code."""
    return {
        "schema_version": 1,
        "annotator_kind": provenance.annotator_kind,
        "qualifies_as_human_recode": False,
        "can_unlock_outcome_modeling": False,
        "unresolved_validation_item": "INDEPENDENT_HUMAN_EXPERT_VERIFICATION_REQUIRED",
        "model": provenance.model,
        "model_version": provenance.model_version,
        "raw_response_sha256": provenance.raw_response_sha256,
    }


def _candidate_strata(agreement: pd.DataFrame) -> pd.DataFrame:
    selected = agreement.loc[
        agreement["feature"].isin(GOLD_STRATIFYING_FEATURES)
    ].copy()
    if selected.empty:
        raise GateFailureError("No eligible categorical P4 strata are available")
    selected["stratum_level"] = [
        _json_value(json.loads(first_value)) if bool(agreed) else DISPUTED
        for first_value, agreed in zip(
            selected["first_value"], selected["agreed"], strict=True
        )
    ]
    selected["candidate_population_n"] = selected.groupby(
        ["feature", "stratum_level"], sort=False
    )["item_id"].transform("size")
    return selected


def _strict_partition(candidates: pd.DataFrame) -> pd.DataFrame:
    ordered = candidates.assign(
        _disputed=candidates["stratum_level"].eq(DISPUTED),
    ).sort_values(
        ["item_id", "candidate_population_n", "feature", "_disputed", "stratum_level"],
        kind="stable",
    )
    assignments = ordered.drop_duplicates("item_id", keep="first").drop(
        columns="_disputed"
    )
    if assignments["item_id"].nunique() != candidates["item_id"].nunique():
        raise GateFailureError("Gold stratum assignment is not a strict partition")
    assignments = assignments.rename(
        columns={
            "feature": "stratum_feature",
            "candidate_population_n": "candidate_cell_n",
        }
    )
    assignments["stratum_id"] = [
        f"stratum_{sha256(f'{feature}\\0{level}'.encode()).hexdigest()[:16]}"
        for feature, level in zip(
            assignments["stratum_feature"],
            assignments["stratum_level"],
            strict=True,
        )
    ]
    assignments["stratum_population_n"] = assignments.groupby("stratum_id", sort=False)[
        "item_id"
    ].transform("size")
    return assignments


def _draw_stratified_sample(
    assignments: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    stratum_order = (
        assignments[
            ["stratum_id", "stratum_feature", "stratum_level", "stratum_population_n"]
        ]
        .drop_duplicates()
        .assign(_disputed=lambda frame: frame["stratum_level"].eq(DISPUTED))
        .sort_values(
            [
                "stratum_population_n",
                "stratum_feature",
                "_disputed",
                "stratum_level",
            ],
            kind="stable",
        )
    )
    sample_order = 0
    for stratum in stratum_order.to_dict(orient="records"):
        members = assignments.loc[
            assignments["stratum_id"].eq(stratum["stratum_id"])
        ].sort_values("item_id", kind="stable")
        population_n = len(members)
        sample_n = min(GOLD_SAMPLE_PER_STRATUM, population_n)
        positions = sorted(
            int(position)
            for position in rng.choice(population_n, size=sample_n, replace=False)
        )
        draw = members.iloc[positions].copy()
        draw["stratum_sample_n"] = sample_n
        draw["inclusion_probability"] = sample_n / population_n
        draw["sample_order"] = range(sample_order, sample_order + sample_n)
        sample_order += sample_n
        selected.append(draw)
    sample = pd.concat(selected, ignore_index=True)
    if sample["item_id"].duplicated().any():
        raise GateFailureError("Gold sample contains duplicate items")
    return sample.sort_values("sample_order", kind="stable").reset_index(drop=True)


def _bundle_rows(sample: pd.DataFrame, agreement: pd.DataFrame) -> pd.DataFrame:
    rows = agreement.loc[
        agreement["item_id"].isin(sample["item_id"])
        & agreement["feature"].isin(GOLD_BUNDLE_FEATURES)
    ].merge(
        sample[["item_id", "item_type", "source_text", "sample_order"]],
        on=["item_id", "item_type"],
        how="inner",
        validate="many_to_one",
    )
    return rows.sort_values(["sample_order", "feature"], kind="stable")


def _adjudication_rows(bundle_rows: pd.DataFrame) -> list[dict[str, str]]:
    local_ids = {
        item_id: f"gold-{index + 1:04d}"
        for index, item_id in enumerate(bundle_rows["item_id"].drop_duplicates())
    }
    return [
        {
            "local_item_id": local_ids[str(row["item_id"])],
            "item_type": str(row["item_type"]),
            "source_text": str(row["source_text"]),
            "feature": str(row["feature"]),
            "first_model_id": str(row["first_model_id"]),
            "first_value_json": str(row["first_value"]),
            "second_model_id": str(row["second_model_id"]),
            "second_value_json": str(row["second_value"]),
            "human_value_json": "",
            "comment": "",
        }
        for row in bundle_rows.to_dict(orient="records")
    ]


def _select_recode_items(
    sample: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for _, group in sample.groupby("stratum_id", sort=True):
        sample_n = len(group)
        recode_n = min(
            sample_n,
            max(
                GOLD_RECODE_MINIMUM_PER_STRATUM,
                math.ceil(GOLD_RECODE_FRACTION * sample_n),
            ),
        )
        ordered = group.sort_values("item_id", kind="stable")
        positions = rng.choice(sample_n, size=recode_n, replace=False)
        selected.append(ordered.iloc[positions])
    combined = pd.concat(selected, ignore_index=True)
    permutation = rng.permutation(len(combined))
    return combined.iloc[permutation].reset_index(drop=True)


def _recode_rows(
    bundle_rows: pd.DataFrame,
    recode_items: pd.DataFrame,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    item_order = {
        str(item_id): index
        for index, item_id in enumerate(recode_items["item_id"], start=1)
    }
    initial_local_ids = {
        item_id: f"gold-{index + 1:04d}"
        for index, item_id in enumerate(bundle_rows["item_id"].drop_duplicates())
    }
    selected = bundle_rows.loc[bundle_rows["item_id"].isin(item_order)].copy()
    recode_counts = recode_items.groupby("stratum_id", sort=False)["item_id"].transform(
        "size"
    )
    recode_probability_by_item = {
        str(row["item_id"]): (
            float(recode_count) / float(row["stratum_sample_n"]),
            float(row["inclusion_probability"])
            * float(recode_count)
            / float(row["stratum_sample_n"]),
        )
        for row, recode_count in zip(
            recode_items.to_dict(orient="records"), recode_counts, strict=True
        )
    }
    selected["_item_order"] = selected["item_id"].map(item_order)
    selected = selected.sort_values(["_item_order", "feature"], kind="stable")
    rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str | float]] = []
    for index, row in enumerate(selected.to_dict(orient="records"), start=1):
        recode_id = f"recode-{index:04d}"
        item_id = str(row["item_id"])
        conditional_probability, joint_probability = recode_probability_by_item[item_id]
        rows.append(
            {
                "recode_item_id": recode_id,
                "item_type": str(row["item_type"]),
                "source_text": str(row["source_text"]),
                "feature": str(row["feature"]),
                "recode_value_json": "",
                "comment": "",
            }
        )
        mapping_rows.append(
            {
                "recode_item_id": recode_id,
                "local_item_id": initial_local_ids[item_id],
                "item_id": item_id,
                "feature": str(row["feature"]),
                "recode_conditional_probability": conditional_probability,
                "recode_joint_inclusion_probability": joint_probability,
            }
        )
    return rows, {"schema_version": 1, "rows": mapping_rows}


def _value_guide() -> dict[str, object]:
    return {
        "schema_version": 1,
        "format": "Enter one strict JSON value in the editable value column.",
        "validation_features": list(GOLD_VALIDATION_FEATURES),
        "stratifying_features": list(GOLD_STRATIFYING_FEATURES),
        "categorical_values": {
            feature: sorted(values, key=str)
            for feature, values in FEATURE_VALUES.items()
        },
        "qdmr_step_count": "Positive JSON integer or null when not applicable.",
        "qdmr_operator_set": "Sorted unique JSON string array from the frozen QDMR inventory.",
        "qdmr_normalized_question": "JSON string when required, otherwise null.",
    }


def _bundle_manifest(
    sample: pd.DataFrame,
    adjudication_csv: bytes,
    recode_csv: bytes,
    recode_mapping: Mapping[str, object],
    value_guide: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sample_item_count": len(sample),
        "stratum_count": sample["stratum_id"].nunique(),
        "adjudication_csv_sha256": sha256(adjudication_csv).hexdigest(),
        "recode_csv_sha256": sha256(recode_csv).hexdigest(),
        "recode_mapping_sha256": sha256(
            canonical_json_bytes(recode_mapping)
        ).hexdigest(),
        "value_guide_sha256": sha256(canonical_json_bytes(value_guide)).hexdigest(),
        "minimum_recode_delay_days": GOLD_RECODE_MINIMUM_DELAY_DAYS,
        "outcome_fields_included": [],
    }


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_csv(
    path: Path,
    expected_columns: Sequence[str],
) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_columns):
                raise MalformedInputError(f"CSV schema mismatch: {path.name}")
            rows = list(reader)
    except OSError as error:
        raise MalformedInputError(f"Cannot read gold CSV: {path}") from error
    return [{str(key): str(value) for key, value in row.items()} for row in rows]


def _load_initial_human_provenance(path: Path) -> InitialHumanGoldProvenance:
    try:
        return InitialHumanGoldProvenance.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise MalformedInputError("Invalid initial human provenance") from error


def _load_recode_human_provenance(path: Path) -> HumanRecodeProvenance:
    try:
        return HumanRecodeProvenance.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise MalformedInputError("Invalid delayed human provenance") from error


def _load_provisional_llm_provenance(path: Path) -> ProvisionalLlmProvenance:
    try:
        return ProvisionalLlmProvenance.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise MalformedInputError("Invalid provisional LLM provenance") from error


def _unique_rows(
    rows: Sequence[Mapping[str, str]],
    key_columns: tuple[str, ...],
    description: str,
) -> dict[tuple[str, ...], Mapping[str, str]]:
    result: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        value = tuple(row.get(column, "") for column in key_columns)
        if any(not part for part in value) or value in result:
            raise MalformedInputError(
                f"{description} contains missing or duplicate keys"
            )
        result[value] = row
    return result


def _gold_schema_sha256() -> str:
    payload = {
        "schema_version": 1,
        "adjudication_columns": list(CSV_COLUMNS),
        "recode_columns": list(RECODE_CSV_COLUMNS),
        "bundle_features": list(GOLD_BUNDLE_FEATURES),
        "validation_features": list(GOLD_VALIDATION_FEATURES),
        "feature_values": {
            feature: sorted(values, key=str)
            for feature, values in FEATURE_VALUES.items()
        },
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _locked_resource_sha(workspace: AnalysisWorkspace, name: str) -> str:
    fingerprint = workspace.load_manifest().resources.get(name)
    if fingerprint is None:
        raise GateFailureError(f"Analysis lock is missing required resource: {name}")
    return fingerprint.sha256


def _validate_locked_provenance_hashes(
    workspace: AnalysisWorkspace,
    *,
    schema_sha256: str,
    codebook_sha256: str,
    decision_ledger_sha256: str,
    manifest: Mapping[str, object],
) -> None:
    expected = {
        "schema_sha256": _gold_schema_sha256(),
        "codebook_sha256": _locked_resource_sha(workspace, "segment3_codebook"),
        "decision_ledger_sha256": _locked_resource_sha(
            workspace, "segment5_decision_ledger"
        ),
    }
    supplied = {
        "schema_sha256": schema_sha256,
        "codebook_sha256": codebook_sha256,
        "decision_ledger_sha256": decision_ledger_sha256,
    }
    if supplied != expected:
        raise MalformedInputError("Human provenance hashes do not match analysis lock")
    if any(str(manifest.get(name)) != value for name, value in expected.items()):
        raise GateFailureError("Gold manifest hashes do not match analysis lock")


def _trusted_step_start(
    workspace: AnalysisWorkspace, command: WorkflowCommand
) -> datetime:
    started_at = workspace.load_state().steps[command].started_at
    if started_at is None:
        raise GateFailureError(f"{command.value} has no trusted start timestamp")
    return started_at


def _step_completed_at(
    workspace: AnalysisWorkspace, command: WorkflowCommand
) -> datetime:
    completed_at = workspace.load_state().steps[command].completed_at
    if completed_at is None:
        raise GateFailureError(f"{command.value} has no trusted completion timestamp")
    return completed_at


def _require_immutable_csv_fields(
    template: Mapping[str, str],
    completed: Mapping[str, str],
    columns: Sequence[str],
) -> None:
    changed = [column for column in columns if template[column] != completed[column]]
    if changed:
        raise MalformedInputError(f"Human CSV changed immutable fields: {changed}")
    denied = [
        column
        for column in completed
        if column.casefold() in ANNOTATION_RESPONSE_FIELD_DENYLIST
    ]
    if denied:
        raise GateFailureError(f"Human CSV contains outcome fields: {sorted(denied)}")


def _parse_human_value(feature: str, raw_value: str) -> object:
    if not raw_value:
        raise MalformedInputError(f"Human value is missing for {feature}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise MalformedInputError(
            f"Human value is not strict JSON for {feature}"
        ) from error
    if feature in FEATURE_VALUES and value not in FEATURE_VALUES[feature]:
        raise MalformedInputError(f"Unknown human label for {feature}: {value!r}")
    if feature == "qdmr_step_count" and not (
        value is None
        or (isinstance(value, int) and not isinstance(value, bool) and value >= 1)
    ):
        raise MalformedInputError("qdmr_step_count must be a positive integer or null")
    if feature == "qdmr_operator_set" and not (
        value is None
        or (
            isinstance(value, list)
            and all(isinstance(operator, str) for operator in value)
            and value == sorted(set(value))
        )
    ):
        raise MalformedInputError(
            "qdmr_operator_set must be a sorted unique string list"
        )
    if feature == "qdmr_normalized_question" and not (
        value is None or (isinstance(value, str) and bool(value.strip()))
    ):
        raise MalformedInputError(
            "qdmr_normalized_question must be a non-empty string or null"
        )
    return value


def _json_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _require_columns(
    frame: pd.DataFrame,
    expected: set[str],
    description: str,
) -> None:
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise MalformedInputError(f"{description} is missing columns: {missing}")
