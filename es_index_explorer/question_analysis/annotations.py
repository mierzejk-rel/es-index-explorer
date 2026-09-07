"""Outcome-blind P4 annotation emission, execution, and ingestion."""

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import sleep
from typing import Literal, Protocol, cast

import pandas as pd
from cursor_sdk import Agent, AgentOptions, Cursor, LocalAgentOptions
from cursor_sdk.errors import CursorAgentError
from cursor_sdk.types import SDKModel
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from es_index_explorer.question_analysis.contracts import (
    ANNOTATION_BATCH_SIZE,
    ANNOTATOR_MODELS,
    ArtifactMetadata,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
    PrerequisiteError,
)
from es_index_explorer.question_analysis.seeds import rng_for
from es_index_explorer.question_analysis.storage import (
    canonical_json_bytes,
    fingerprint_file,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

QUESTION_LABELS = {
    "exhaustivity_requirement": {"mention_some", "weakly_exhaustive", "mention_all"},
    "qdmr_applicability": {"applicable", "applicable_after_normalisation", "not_applicable"},
    "hop_structure": {"atomic", "bridge", "comparison", "intersection"},
    "referring_form_type": {"full_name_form", "alias_or_handle", "email_address"},
    "recall_orientation": {"precision_oriented", "recall_oriented"},
    "cognitive_process_level": {"remember", "understand", "apply", "analyze", "evaluate", "create"},
}
EXPECTATION_LABELS = {
    "demand_type": {
        "verbatim_citation",
        "entity_identification",
        "relational_claim",
        "temporal_ordering",
        "quantification",
        "evaluative_synthesis",
    },
    "specificity": {"specific", "broad"},
    "answer_locality": {"single_passage", "cross_document_aggregation"},
}
QDMR_OPERATORS = frozenset(
    {
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
    }
)
P4_INSTRUCTIONS = """
For questions, exhaustivity_requirement is mention_some when one valid instance or fact answers
the request, weakly_exhaustive when all salient answers in scope are expected without proving no
answer was omitted, and mention_all for exhaustive enumeration or a required negative conclusion.
negative_conclusiveness is true only when an adequate negative answer must establish absence.
presupposition_load is true only when the wording takes a disputed entity, event, relationship, or
proposition for granted rather than asking whether it exists.

QDMR applicability is applicable for a direct interrogative. It is
applicable_after_normalisation for a directive only when a faithful interrogative paraphrase exists,
and the paraphrase must be recorded. Otherwise use not_applicable and set all QDMR decomposition
values to null. Applicable decompositions have a positive step count, a hop_structure of atomic,
bridge, comparison, or intersection, and a sorted unique operator subset of SELECT, FILTER, PROJECT,
AGGREGATE, GROUP, SUPERLATIVE, COMPARATIVE, UNION, INTERSECTION, DISCARD, SORT, BOOLEAN, ARITHMETIC.

referring_form_type is email_address when a raw email identifies a focal referent, otherwise
alias_or_handle for a pseudonym, nickname, screen name, or explicit alias, otherwise full_name_form
for a personal or organization name including surname-only and title-plus-surname. If several occur,
email_address takes precedence over alias_or_handle over full_name_form. Use no_focal_referent only
when no focal named referent exists. recall_orientation is recall_oriented for requests to find,
list, review, or summarize all relevant material; known-item or single-fact requests are
precision_oriented. cognitive_process_level follows remember < understand < apply < analyze <
evaluate < create.

For expectations, demand_type is exactly one of verbatim_citation, entity_identification,
relational_claim, temporal_ordering, quantification, or evaluative_synthesis. specificity is
specific when it names the required fact, entity, relation, event, or value; otherwise broad.
answer_locality is single_passage when one coherent passage can establish the expectation and
cross_document_aggregation only when evidence from multiple documents must be combined. When
multiple passages from one document are needed, set answer_locality null and use
not_classifiable_binary_locality. Document IDs are not evidence of locality.
""".strip()
OUTCOME_KEY_PATTERN = re.compile(
    r"(^|_)(p|f|u|state|error_override|rubric_?v?2|pass_?rate|grade|ordinal_?grade|tier|"
    r"criterion_?observation_?id|trace_?id|run_?id|eligible|recommendation)(_|$)",
    re.IGNORECASE,
)
RAW_RESPONSE_DIRECTORY = "annotations/raw"
BATCH_DIRECTORY = "annotations/batches"
ATTEMPT_DIRECTORY = "annotations/attempts"
ENVELOPE_DIRECTORY = "annotations/envelopes"
REJECTED_DIRECTORY = "annotations/rejected"
SUPERSEDED_DIRECTORY = "annotations/superseded/v1"
ANNOTATION_MAX_WORKERS = 4
ORDINAL_LEVELS = {
    "exhaustivity_requirement": (
        "mention_some",
        "weakly_exhaustive",
        "mention_all",
    ),
    "cognitive_process_level": (
        "remember",
        "understand",
        "apply",
        "analyze",
        "evaluate",
        "create",
    ),
}


class QuestionAnnotation(BaseModel):
    """Strict P4 annotation response for a question item."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    exhaustivity_requirement: Literal["mention_some", "weakly_exhaustive", "mention_all"]
    negative_conclusiveness: bool
    presupposition_load: bool
    qdmr_applicability: Literal["applicable", "applicable_after_normalisation", "not_applicable"]
    qdmr_step_count: int | None = Field(default=None, ge=1)
    qdmr_operator_set: list[str] | None = None
    qdmr_normalized_question: str | None = None
    hop_structure: Literal["atomic", "bridge", "comparison", "intersection"] | None = None
    referring_form_type: Literal["full_name_form", "alias_or_handle", "email_address"] | None = None
    referring_form_missingness: Literal["no_focal_referent"] | None = None
    recall_orientation: Literal["precision_oriented", "recall_oriented"]
    cognitive_process_level: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]

    @model_validator(mode="after")
    def validate_qdmr_and_reference_form(self) -> "QuestionAnnotation":
        applicable = self.qdmr_applicability != "not_applicable"
        has_decomposition = (
            self.qdmr_step_count is not None and bool(self.qdmr_operator_set)
        )
        if applicable != has_decomposition:
            raise ValueError("QDMR applicability and decomposition values disagree")
        if applicable != (self.hop_structure is not None):
            raise ValueError("QDMR applicability and hop_structure disagree")
        if self.qdmr_applicability == "applicable_after_normalisation":
            if not self.qdmr_normalized_question:
                raise ValueError("normalized QDMR requires qdmr_normalized_question")
        elif self.qdmr_normalized_question is not None:
            raise ValueError("only normalized QDMR may carry qdmr_normalized_question")
        if not applicable and self.qdmr_operator_set is not None:
            raise ValueError("inapplicable QDMR must use null rather than an empty operator set")
        if self.qdmr_operator_set is not None:
            if self.qdmr_operator_set != sorted(set(self.qdmr_operator_set)):
                raise ValueError("qdmr_operator_set must be sorted and unique")
            if not set(self.qdmr_operator_set) <= QDMR_OPERATORS:
                raise ValueError("qdmr_operator_set contains an unknown operator")
        if self.referring_form_type is None:
            if self.referring_form_missingness != "no_focal_referent":
                raise ValueError("missing referring form requires no_focal_referent")
        elif self.referring_form_missingness is not None:
            raise ValueError("classified referring form cannot carry missingness")
        return self


class ExpectationAnnotation(BaseModel):
    """Strict P4 annotation response for an expectation item."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    demand_type: Literal[
        "verbatim_citation",
        "entity_identification",
        "relational_claim",
        "temporal_ordering",
        "quantification",
        "evaluative_synthesis",
    ]
    specificity: Literal["specific", "broad"]
    answer_locality: Literal["single_passage", "cross_document_aggregation"] | None = None
    answer_locality_missingness: Literal["not_classifiable_binary_locality"] | None = None

    @model_validator(mode="after")
    def validate_locality(self) -> "ExpectationAnnotation":
        if (self.answer_locality is None) == (self.answer_locality_missingness is None):
            raise ValueError("answer locality must be classified or explicitly missing")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Exact SDK model and parameter selection frozen for annotation."""

    requested_name: str
    model_id: str
    display_name: str
    parameters: tuple[tuple[str, str], ...]


class CursorSdkLike(Protocol):
    """Minimal Cursor SDK surface used by the annotation runner."""

    def list_models(self, api_key: str) -> Sequence[SDKModel]: ...

    def prompt(self, message: str, options: AgentOptions) -> object: ...


class CursorSdkAdapter:
    """Production adapter for the local Cursor Python SDK."""

    def list_models(self, api_key: str) -> Sequence[SDKModel]:
        """List account-visible Cursor models."""
        return Cursor.models.list(api_key=api_key)

    def prompt(self, message: str, options: AgentOptions) -> object:
        """Execute one isolated local one-shot prompt."""
        return Agent.prompt(message, options)


def run_annotate_emit(workspace: AnalysisWorkspace) -> tuple[ArtifactMetadata, ...]:
    """Emit deterministic, outcome-blind annotation batches."""
    items = _load_feature_items(workspace)
    batches = _build_batches(items)
    artifacts = [
        workspace.store.write_json(
            f"{BATCH_DIRECTORY}/{batch_id}.json",
            payload,
            created_by=WorkflowCommand.ANNOTATE_EMIT,
        )
        for batch_id, payload in batches.items()
    ]
    index = {"schema_version": 1, "batch_ids": list(batches), "batch_count": len(batches)}
    artifacts.append(
        workspace.store.write_json(
            f"{BATCH_DIRECTORY}/index.json", index, created_by=WorkflowCommand.ANNOTATE_EMIT
        )
    )
    return tuple(artifacts)


def run_annotate_run(
    workspace: AnalysisWorkspace,
    sdk: CursorSdkLike | None = None,
    api_key: str | None = None,
) -> tuple[ArtifactMetadata, ...]:
    """Run all emitted batches using the frozen Cursor model selections."""
    secret = os.environ.get("CURSOR_API_KEY") if api_key is None else api_key
    if not secret:
        raise PrerequisiteError("CURSOR_API_KEY is required before Cursor annotation can run")
    runner = sdk or CursorSdkAdapter()
    batches = _load_batches(workspace)
    _archive_v1_preflight(workspace)
    preflight_path = workspace.store.path_for("annotation_preflight_manifest.json")
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        models = _models_from_manifest(preflight)
        if canonical_json_bytes(preflight) != canonical_json_bytes(
            _annotation_manifest(models, batches)
        ):
            raise MalformedInputError(
                "Annotation preflight manifest conflicts with current batches, schemas, prompt, or SDK"
            )
    else:
        models = resolve_models(runner.list_models(secret))
        preflight = _annotation_manifest(models, batches)
        _write_once(
            workspace,
            "annotation_preflight_manifest.json",
            canonical_json_bytes(preflight),
        )
    artifacts: list[ArtifactMetadata] = [
        _existing_metadata(workspace, "annotation_preflight_manifest.json")
    ]
    artifacts.extend(_existing_rejected_metadata(workspace))
    artifacts.extend(_existing_superseded_metadata(workspace))
    jobs = [
        (model, batch_id, batch)
        for model in models
        for batch_id, batch in batches.items()
    ]
    executor = ThreadPoolExecutor(
        max_workers=ANNOTATION_MAX_WORKERS,
        thread_name_prefix="cursor-annotation",
    )
    futures: dict[Future[tuple[ArtifactMetadata, ...]], tuple[str, str]] = {
        executor.submit(
            _run_or_resume_batch,
            workspace,
            runner,
            secret,
            model,
            batch_id,
            batch,
        ): (model.model_id, batch_id)
        for model, batch_id, batch in jobs
    }
    try:
        for future in as_completed(futures):
            artifacts.extend(future.result())
    except Exception:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    completed_manifest = {
        **preflight,
        "runs": _completed_run_records(workspace, models, batches),
        "rejected_runs": _rejected_run_records(workspace),
    }
    artifacts.append(
        _write_once(
            workspace,
            "annotation_manifest.json",
            canonical_json_bytes(completed_manifest),
        )
    )
    return tuple(sorted(artifacts, key=lambda artifact: artifact.path))


def run_annotate_ingest(workspace: AnalysisWorkspace) -> tuple[ArtifactMetadata, ...]:
    """Validate raw responses, normalize annotations, and report agreement."""
    batches = _load_batches(workspace)
    manifest = _load_annotation_manifest(workspace)
    rows = _normalize_all(workspace, batches, manifest)
    normalized = pd.DataFrame(rows).sort_values(["item_id", "model_id"]).reset_index(drop=True)
    if len(normalized) != 2 * 577 or normalized["item_id"].nunique() != 577:
        raise GateFailureError("Annotation normalization does not contain 577 items × 2 models")
    agreement = _agreement_rows(normalized)
    verification = _annotation_verification(normalized, agreement)
    artifacts = (
        workspace.store.write_parquet(
            "tables/annotations_normalized.parquet",
            _versioned(normalized),
            created_by=WorkflowCommand.ANNOTATE_INGEST,
        ),
        workspace.store.write_parquet(
            "tables/annotation_agreement.parquet",
            _versioned(agreement),
            created_by=WorkflowCommand.ANNOTATE_INGEST,
        ),
        workspace.store.write_json(
            "annotation_ingest_verification.json",
            verification,
            created_by=WorkflowCommand.ANNOTATE_INGEST,
        ),
        workspace.store.write_bytes(
            "partial_reports/03-annotation-reliability.md",
            _render_report(verification).encode("utf-8"),
            created_by=WorkflowCommand.ANNOTATE_INGEST,
        ),
    )
    if not verification["passed"]:
        raise GateFailureError(f"Annotation ingestion failed: {verification['blocking_failures']}")
    return artifacts


def resolve_models(models: Sequence[SDKModel]) -> tuple[ResolvedModel, ...]:
    """Resolve the two locked annotator identities without fallbacks."""
    candidates = [
        ResolvedModel(
            requested_name=name,
            model_id=model.id,
            display_name=model.display_name,
            parameters=_model_parameters(model),
        )
        for name in ANNOTATOR_MODELS
        for model in models
        if _matches_model(name, model)
    ]
    if len(candidates) != 2 or {candidate.requested_name for candidate in candidates} != set(ANNOTATOR_MODELS):
        raise GateFailureError("Required Cursor annotation model or variant is unavailable or ambiguous")
    return tuple(sorted(candidates, key=lambda model: model.requested_name))


def _matches_model(requested: str, model: SDKModel) -> bool:
    expected_display_name = (
        "Claude Opus 5" if requested.startswith("Claude") else "GPT-5.6 Sol"
    )
    return model.display_name.casefold() == expected_display_name.casefold()


def _model_parameters(model: SDKModel) -> tuple[tuple[str, str], ...]:
    default_variants = [variant for variant in model.variants if variant.is_default]
    if len(default_variants) != 1:
        raise GateFailureError(f"Cursor model has no unique default variant: {model.id}")
    public_parameter_ids = {parameter.id for parameter in model.parameters}
    parameters = tuple(
        (parameter.id, parameter.value)
        for parameter in default_variants[0].params
        if parameter.id in public_parameter_ids
    )
    if model.display_name.casefold() == "claude opus 5" and not (
        ("thinking", "true") in parameters and ("effort", "high") in parameters
    ):
        raise GateFailureError("Claude Opus 5 default is not the frozen high-thinking variant")
    return parameters


def _load_feature_items(workspace: AnalysisWorkspace) -> pd.DataFrame:
    path = workspace.store.path_for("tables/features_deterministic.parquet")
    if not path.is_file():
        raise MalformedInputError("Segment 3 features artifact is missing")
    frame = pd.read_parquet(path).drop(columns="artifact_schema_version")
    _reject_outcome_keys(frame.columns.tolist())
    columns = ["item_id", "item_type", "source_text", "text_sha256"]
    if set(frame["item_type"]) != {"question", "expectation"} or frame["item_id"].duplicated().any():
        raise GateFailureError("Segment 3 annotation population is invalid")
    return frame[columns]


def _build_batches(items: pd.DataFrame) -> dict[str, dict[str, object]]:
    if len(items) != 577:
        raise GateFailureError("Annotation population must contain exactly 577 items")
    order = rng_for("annotation_shuffle").permutation(len(items))
    shuffled = items.iloc[order].reset_index(drop=True)
    batches: dict[str, dict[str, object]] = {}
    for index, start in enumerate(range(0, len(shuffled), ANNOTATION_BATCH_SIZE), start=1):
        rows = shuffled.iloc[start : start + ANNOTATION_BATCH_SIZE].to_dict(orient="records")
        _reject_outcome_keys_from_value(rows)
        batch_id = f"batch-{index:03d}"
        batches[batch_id] = {
            "schema_version": 1,
            "batch_id": batch_id,
            "items": rows,
            "item_ids_sha256": sha256(canonical_json_bytes({"item_ids": [row["item_id"] for row in rows]})).hexdigest(),
        }
    final_items = cast(list[dict[str, object]], batches["batch-025"]["items"])
    if len(batches) != 25 or len(final_items) != 1:
        raise GateFailureError("Frozen annotation batching contract failed")
    return batches


def _prompt_batch(batch: Mapping[str, object]) -> dict[str, object]:
    items = cast(list[dict[str, object]], batch["items"])
    return {
        "schema_version": 2,
        "batch_id": batch["batch_id"],
        "items": [
            {
                "item_id": f"item-{index:03d}",
                "item_type": item["item_type"],
                "source_text": item["source_text"],
            }
            for index, item in enumerate(items, start=1)
        ],
    }


def _load_batches(workspace: AnalysisWorkspace) -> dict[str, dict[str, object]]:
    index_path = workspace.store.path_for(f"{BATCH_DIRECTORY}/index.json")
    if not index_path.is_file():
        raise MalformedInputError("Annotation batches have not been emitted")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    batches = {
        batch_id: json.loads(workspace.store.path_for(f"{BATCH_DIRECTORY}/{batch_id}.json").read_text(encoding="utf-8"))
        for batch_id in index["batch_ids"]
    }
    return batches


def _annotation_manifest(models: Sequence[ResolvedModel], batches: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "sdk_version": version("cursor-sdk"),
        "models": [asdict(model) for model in models],
        "batch_hashes": {batch_id: str(payload["item_ids_sha256"]) for batch_id, payload in batches.items()},
        "batch_count": len(batches),
        "batch_size": ANNOTATION_BATCH_SIZE,
        "prompt_sha256": sha256(_prompt_template().encode("utf-8")).hexdigest(),
        "question_schema_sha256": sha256(json.dumps(QuestionAnnotation.model_json_schema(), sort_keys=True).encode()).hexdigest(),
        "expectation_schema_sha256": sha256(json.dumps(ExpectationAnnotation.model_json_schema(), sort_keys=True).encode()).hexdigest(),
    }


def _models_from_manifest(manifest: Mapping[str, object]) -> tuple[ResolvedModel, ...]:
    return tuple(
        ResolvedModel(
            requested_name=str(model["requested_name"]),
            model_id=str(model["model_id"]),
            display_name=str(model["display_name"]),
            parameters=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in cast(list[list[object]], model["parameters"])
            ),
        )
        for model in cast(list[dict[str, object]], manifest["models"])
    )


def _completed_run_records(
    workspace: AnalysisWorkspace,
    models: Sequence[ResolvedModel],
    batches: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    records = []
    for model in models:
        for batch_id in batches:
            path = workspace.store.path_for(
                f"{ATTEMPT_DIRECTORY}/{model.model_id}/{batch_id}.json"
            )
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                raise MalformedInputError(
                    f"Invalid annotation attempt metadata: {path}"
                ) from error
    return records


def _write_rejected_response(
    workspace: AnalysisWorkspace,
    model: ResolvedModel,
    batch_id: str,
    response_bytes: bytes,
    metadata: Mapping[str, object],
) -> None:
    rejection_id = sha256(
        f"{metadata['run_id']}:{metadata['response_sha256']}".encode()
    ).hexdigest()[:24]
    base = f"{REJECTED_DIRECTORY}/{model.model_id}/{batch_id}/{rejection_id}"
    _write_once(workspace, f"{base}.jsonl", response_bytes)
    _write_once(workspace, f"{base}.json", canonical_json_bytes(metadata))


def _rejected_run_records(workspace: AnalysisWorkspace) -> list[dict[str, object]]:
    root = workspace.store.path_for(REJECTED_DIRECTORY)
    records = []
    for path in sorted(root.rglob("*.json")) if root.is_dir() else []:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise MalformedInputError(
                f"Invalid rejected annotation metadata: {path}"
            ) from error
    return records


def _existing_rejected_metadata(
    workspace: AnalysisWorkspace,
) -> list[ArtifactMetadata]:
    root = workspace.store.path_for(REJECTED_DIRECTORY)
    if not root.is_dir():
        return []
    return [
        _existing_metadata(workspace, path.relative_to(workspace.root).as_posix())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _archive_v1_preflight(workspace: AnalysisWorkspace) -> None:
    preflight_path = workspace.store.path_for("annotation_preflight_manifest.json")
    if not preflight_path.is_file():
        return
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MalformedInputError("Invalid annotation preflight manifest") from error
    if preflight.get("schema_version") != 1:
        return

    moved_paths = []
    for directory in (RAW_RESPONSE_DIRECTORY, ATTEMPT_DIRECTORY):
        root = workspace.store.path_for(directory)
        for source in sorted(root.rglob("*")) if root.is_dir() else []:
            if not source.is_file():
                continue
            relative = source.relative_to(workspace.root)
            destination = workspace.store.path_for(
                f"{SUPERSEDED_DIRECTORY}/{relative.as_posix()}"
            )
            _archive_file(source, destination)
            moved_paths.append(relative.as_posix())
    _write_once(
        workspace,
        f"{SUPERSEDED_DIRECTORY}/migration.json",
        canonical_json_bytes(
            {
                "schema_version": 1,
                "reason": "Replaced long model-facing hash IDs with deterministic batch-local keys",
                "superseded_paths": moved_paths,
            }
        ),
    )
    _archive_file(
        preflight_path,
        workspace.store.path_for(
            f"{SUPERSEDED_DIRECTORY}/annotation_preflight_manifest.json"
        ),
    )


def _archive_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise MalformedInputError(
                f"Superseded annotation artifact conflicts: {destination}"
            )
        source.unlink()
        return
    source.replace(destination)


def _existing_superseded_metadata(
    workspace: AnalysisWorkspace,
) -> list[ArtifactMetadata]:
    root = workspace.store.path_for(SUPERSEDED_DIRECTORY)
    if not root.is_dir():
        return []
    return [
        _existing_metadata(workspace, path.relative_to(workspace.root).as_posix())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _run_or_resume_batch(
    workspace: AnalysisWorkspace,
    sdk: CursorSdkLike,
    api_key: str,
    model: ResolvedModel,
    batch_id: str,
    batch: Mapping[str, object],
) -> tuple[ArtifactMetadata, ...]:
    raw_path = f"{RAW_RESPONSE_DIRECTORY}/{model.model_id}/{batch_id}.jsonl"
    attempt_path = f"{ATTEMPT_DIRECTORY}/{model.model_id}/{batch_id}.json"
    envelope_path = f"{ENVELOPE_DIRECTORY}/{model.model_id}/{batch_id}.json"
    absolute_raw = workspace.store.path_for(raw_path)
    absolute_attempt = workspace.store.path_for(attempt_path)
    absolute_envelope = workspace.store.path_for(envelope_path)

    if absolute_envelope.is_file():
        return _materialize_envelope(
            workspace,
            model,
            batch_id,
            batch,
            envelope_path,
            raw_path,
            attempt_path,
        )
    if absolute_raw.exists() or absolute_attempt.exists():
        if not absolute_raw.is_file() or not absolute_attempt.is_file():
            raise MalformedInputError(
                f"Incomplete legacy annotation response pair: {model.model_id}/{batch_id}"
            )
        _verify_raw_response(absolute_raw, absolute_attempt, model, batch_id)
        return (
            _existing_metadata(workspace, raw_path),
            _existing_metadata(workspace, attempt_path),
        )

    result, attempts = _run_batch(sdk, api_key, model, batch_id, batch)
    if str(getattr(result, "status", "")) != "finished":
        raise GateFailureError(
            f"Cursor run failed model={model.model_id} batch={batch_id}"
        )
    response_bytes = str(getattr(result, "result", "")).encode("utf-8")
    metadata = {
        "attempts": attempts,
        **_response_metadata(result, model, batch_id, batch, response_bytes),
    }
    try:
        _parse_response_text(
            response_bytes.decode("utf-8"),
            cast(list[dict[str, object]], batch["items"]),
            f"{model.model_id}/{batch_id}",
        )
    except MalformedInputError as error:
        _write_rejected_response(
            workspace,
            model,
            batch_id,
            response_bytes,
            {**metadata, "rejection_reason": error.message},
        )
        raise

    _write_once(
        workspace,
        envelope_path,
        canonical_json_bytes(
            {"metadata": metadata, "response": response_bytes.decode("utf-8")}
        ),
    )
    return _materialize_envelope(
        workspace,
        model,
        batch_id,
        batch,
        envelope_path,
        raw_path,
        attempt_path,
    )


def _materialize_envelope(
    workspace: AnalysisWorkspace,
    model: ResolvedModel,
    batch_id: str,
    batch: Mapping[str, object],
    envelope_path: str,
    raw_path: str,
    attempt_path: str,
) -> tuple[ArtifactMetadata, ...]:
    absolute_envelope = workspace.store.path_for(envelope_path)
    try:
        envelope = json.loads(absolute_envelope.read_text(encoding="utf-8"))
        metadata = cast(dict[str, object], envelope["metadata"])
        response_bytes = str(envelope["response"]).encode("utf-8")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise MalformedInputError(
            f"Invalid annotation response envelope: {envelope_path}"
        ) from error
    if (
        metadata.get("model_id") != model.model_id
        or metadata.get("batch_id") != batch_id
        or metadata.get("batch_hash") != batch["item_ids_sha256"]
        or metadata.get("status") != "finished"
        or metadata.get("response_sha256") != sha256(response_bytes).hexdigest()
    ):
        raise MalformedInputError(
            f"Annotation response envelope metadata mismatch: {model.model_id}/{batch_id}"
        )
    _parse_response_text(
        response_bytes.decode("utf-8"),
        cast(list[dict[str, object]], batch["items"]),
        f"{model.model_id}/{batch_id}",
    )
    raw_metadata = _write_once(workspace, raw_path, response_bytes)
    attempt_metadata = _write_once(
        workspace, attempt_path, canonical_json_bytes(metadata)
    )
    _verify_raw_response(
        workspace.store.path_for(raw_path),
        workspace.store.path_for(attempt_path),
        model,
        batch_id,
    )
    return (
        _existing_metadata(workspace, envelope_path),
        raw_metadata,
        attempt_metadata,
    )


def _run_batch(
    sdk: CursorSdkLike, api_key: str, model: ResolvedModel, batch_id: str, batch: Mapping[str, object]
) -> tuple[object, int]:
    for attempt in range(1, 4):
        try:
            with tempfile.TemporaryDirectory(prefix="simplemode-annotation-") as temporary:
                directory = Path(temporary)
                prompt_batch = _prompt_batch(batch)
                (directory / "batch.json").write_bytes(
                    canonical_json_bytes(prompt_batch)
                )
                (directory / "schema-question.json").write_text(
                    json.dumps(QuestionAnnotation.model_json_schema(), sort_keys=True), encoding="utf-8"
                )
                (directory / "schema-expectation.json").write_text(
                    json.dumps(ExpectationAnnotation.model_json_schema(), sort_keys=True), encoding="utf-8"
                )
                options = AgentOptions(
                    model={"id": model.model_id, "params": [{"id": key, "value": value} for key, value in model.parameters]},
                    api_key=api_key,
                    local=LocalAgentOptions(cwd=directory, setting_sources=[], sandbox_options={"enabled": True}),
                    tools=[],
                )
                return sdk.prompt(_prompt_for_batch(batch), options), attempt
        except CursorAgentError as error:
            if not error.is_retryable or attempt == 3:
                safe_message = error.message.replace(api_key, "[REDACTED]")
                raise GateFailureError(
                    "Cursor SDK startup failed "
                    f"model={model.model_id} batch={batch_id} "
                    f"code={error.code} proto_code={error.proto_error_code} "
                    f"status={error.status_code} message={safe_message}"
                ) from error
            sleep(_retry_delay(error, attempt))
    raise AssertionError("unreachable")


def _retry_delay(error: CursorAgentError, attempt: int) -> float:
    try:
        return min(60.0, max(1.0, float(error.retry_after or 2**attempt)))
    except ValueError:
        return float(2**attempt)


def _prompt_template() -> str:
    return (
        "Return JSONL only: exactly one object per supplied item, no markdown. Annotate only the supplied "
        "source text under the supplied schemas. Do not infer or mention experimental outcomes, quality, "
        "grades, evaluation results, traces, runs, or recommendations. Questions use QuestionAnnotation; "
        "expectations use ExpectationAnnotation. Each item_id is a short batch-local key; copy it exactly "
        "and do not invent, expand, or alter it. Use sorted unique uppercase QDMR operators.\n\n"
        f"CODEBOOK:\n{P4_INSTRUCTIONS}"
    )


def _prompt_for_batch(batch: Mapping[str, object]) -> str:
    return (
        f"{_prompt_template()}\n\nQUESTION_SCHEMA:\n"
        f"{json.dumps(QuestionAnnotation.model_json_schema(), ensure_ascii=False, sort_keys=True)}"
        f"\n\nEXPECTATION_SCHEMA:\n"
        f"{json.dumps(ExpectationAnnotation.model_json_schema(), ensure_ascii=False, sort_keys=True)}"
        f"\n\nBATCH:\n{json.dumps(_prompt_batch(batch), ensure_ascii=False, sort_keys=True)}"
    )


def _response_metadata(
    result: object,
    model: ResolvedModel,
    batch_id: str,
    batch: Mapping[str, object],
    response_bytes: bytes,
) -> dict[str, object]:
    usage = getattr(result, "usage", None)
    return {
        "schema_version": 1,
        "model_id": model.model_id,
        "batch_id": batch_id,
        "batch_hash": batch["item_ids_sha256"],
        "agent_id": str(getattr(result, "agent_id", "")),
        "run_id": str(getattr(result, "id", "")),
        "status": str(getattr(result, "status", "")),
        "created_at": str(getattr(result, "created_at", "")),
        "duration_ms": int(getattr(result, "duration_ms", 0)),
        "usage": asdict(usage) if usage is not None else None,
        "response_sha256": sha256(response_bytes).hexdigest(),
    }


def _write_once(workspace: AnalysisWorkspace, relative_path: str, data: bytes) -> ArtifactMetadata:
    path = workspace.store.path_for(relative_path)
    if path.exists():
        if sha256(path.read_bytes()).digest() != sha256(data).digest():
            raise MalformedInputError(f"Immutable annotation artifact conflicts: {relative_path}")
        return ArtifactMetadata(
            **fingerprint_file(path, label=relative_path).model_dump(),
            created_by=WorkflowCommand.ANNOTATE_RUN,
            created_at=datetime.now(UTC),
        )
    return workspace.store.write_bytes(relative_path, data, created_by=WorkflowCommand.ANNOTATE_RUN)


def _existing_metadata(workspace: AnalysisWorkspace, relative_path: str) -> ArtifactMetadata:
    """Return registration metadata for a verified immutable response artifact."""
    path = workspace.store.path_for(relative_path)
    return ArtifactMetadata(
        **fingerprint_file(path, label=relative_path).model_dump(),
        created_by=WorkflowCommand.ANNOTATE_RUN,
        created_at=datetime.now(UTC),
    )


def _verify_raw_response(
    path: Path,
    attempt_path: Path,
    model: ResolvedModel,
    batch_id: str,
) -> None:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise MalformedInputError(f"Invalid existing raw annotation response: {model.model_id}/{batch_id}")
    try:
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MalformedInputError(
            f"Invalid annotation attempt metadata: {attempt_path}"
        ) from error
    if (
        attempt.get("model_id") != model.model_id
        or attempt.get("batch_id") != batch_id
        or attempt.get("status") != "finished"
        or attempt.get("response_sha256") != sha256(path.read_bytes()).hexdigest()
    ):
        raise MalformedInputError(
            f"Raw annotation response metadata mismatch: {model.model_id}/{batch_id}"
        )


def _load_annotation_manifest(workspace: AnalysisWorkspace) -> dict[str, object]:
    path = workspace.store.path_for("annotation_manifest.json")
    if not path.is_file():
        raise MalformedInputError("Annotation manifest is missing")
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _normalize_all(
    workspace: AnalysisWorkspace, batches: Mapping[str, Mapping[str, object]], manifest: Mapping[str, object]
) -> list[dict[str, object]]:
    expected = {
        str(item["item_id"]): item
        for batch in batches.values()
        for item in cast(list[dict[str, object]], batch["items"])
    }
    rows: list[dict[str, object]] = []
    for model in _models_from_manifest(manifest):
        for batch_id, batch in batches.items():
            path = workspace.store.path_for(f"{RAW_RESPONSE_DIRECTORY}/{model.model_id}/{batch_id}.jsonl")
            response_rows = _parse_response(path, cast(list[dict[str, object]], batch["items"]))
            attempt_path = workspace.store.path_for(
                f"{ATTEMPT_DIRECTORY}/{model.model_id}/{batch_id}.json"
            )
            try:
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise MalformedInputError(
                    f"Invalid annotation attempt metadata: {attempt_path}"
                ) from error
            for item, annotation in response_rows:
                annotation.pop("item_id")
                row = {
                    "item_id": item["item_id"],
                    "item_type": item["item_type"],
                    "source_text_sha256": item["text_sha256"],
                    "model_id": model.model_id,
                    "model_name": model.requested_name,
                    "batch_id": batch_id,
                    "annotation_agent_id": attempt["agent_id"],
                    "annotation_run_id": attempt["run_id"],
                    "raw_response_sha256": sha256(path.read_bytes()).hexdigest(),
                    **annotation,
                }
                rows.append(row)
    if {row["item_id"] for row in rows} != set(expected) or len(rows) != 2 * len(expected):
        raise GateFailureError("Raw annotations are incomplete")
    return rows


def _parse_response(path: Path, items: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object]]]:
    if not path.is_file():
        raise MalformedInputError(f"Missing raw annotation response: {path}")
    return _parse_response_text(path.read_text(encoding="utf-8"), items, str(path))


def _parse_response_text(
    response: str,
    items: list[dict[str, object]],
    source: str,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    item_by_id = {
        f"item-{index:03d}": item for index, item in enumerate(items, start=1)
    }
    lines = response.splitlines()
    if len(lines) != len(items):
        raise MalformedInputError(
            f"Raw annotation response has incorrect line count: {source}"
        )
    rows = []
    seen_item_ids: set[str] = set()
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise MalformedInputError(
                f"Raw annotation response is not strict JSONL: {source}"
            ) from error
        _reject_outcome_keys_from_value(value)
        item_id = str(value.get("item_id", ""))
        item = item_by_id.get(item_id)
        if item is None or item_id in seen_item_ids:
            raise MalformedInputError(
                f"Raw annotation IDs do not match batch: {source}"
            )
        seen_item_ids.add(item_id)
        model = QuestionAnnotation if item["item_type"] == "question" else ExpectationAnnotation
        try:
            annotation = model.model_validate(value).model_dump(mode="json")
        except ValidationError as error:
            raise MalformedInputError(
                f"Raw annotation response violates the strict schema: {source}"
            ) from error
        rows.append((item, annotation))
    return rows


def _agreement_rows(normalized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item_id, group in normalized.groupby("item_id", sort=True):
        if len(group) != 2:
            raise GateFailureError(f"Expected exactly two model annotations for {item_id}")
        first, second = group.sort_values("model_id").to_dict(orient="records")
        for feature, (left, right) in _comparable_features(first, second).items():
            if isinstance(left, list) or isinstance(right, list):
                left_set = set(left if isinstance(left, list) else [])
                right_set = set(right if isinstance(right, list) else [])
                union = left_set | right_set
                similarity = len(left_set & right_set) / len(union) if union else 1.0
                agreed = left_set == right_set
                category = (
                    "agreement"
                    if agreed
                    else "set_partial_overlap"
                    if left_set & right_set
                    else "set_disjoint"
                )
            elif feature in ORDINAL_LEVELS:
                levels = ORDINAL_LEVELS[feature]
                left_rank, right_rank = levels.index(left), levels.index(right)
                distance = abs(left_rank - right_rank)
                similarity = 1.0 - (distance / (len(levels) - 1))
                agreed = distance == 0
                category = "agreement" if agreed else f"ordinal_distance_{distance}"
            elif feature == "qdmr_step_count":
                if not isinstance(left, (int, float)) or not isinstance(
                    right, (int, float)
                ):
                    raise GateFailureError(
                        f"Non-numeric QDMR step count for item {item_id}"
                    )
                distance = abs(float(left) - float(right))
                similarity = 1.0 / (1.0 + distance)
                agreed = distance == 0
                category = "agreement" if agreed else "numeric_distance"
            else:
                similarity = float(left == right)
                agreed = left == right
                category = "agreement" if agreed else "different_value"
            rows.append(
                {
                    "item_id": item_id,
                    "item_type": first["item_type"],
                    "feature": feature,
                    "first_model_id": first["model_id"],
                    "second_model_id": second["model_id"],
                    "first_value": json.dumps(left, sort_keys=True),
                    "second_value": json.dumps(right, sort_keys=True),
                    "agreed": agreed,
                    "similarity": similarity,
                    "disagreement_category": category,
                }
            )
    return pd.DataFrame(rows)


def _comparable_features(
    first: Mapping[str, object], second: Mapping[str, object]
) -> dict[str, tuple[object, object]]:
    item_type = str(first["item_type"])
    if item_type == "expectation":
        return {
            "demand_type": (_clean_value(first, "demand_type"), _clean_value(second, "demand_type")),
            "specificity": (_clean_value(first, "specificity"), _clean_value(second, "specificity")),
            "answer_locality": (
                _clean_value(first, "answer_locality")
                or _clean_value(first, "answer_locality_missingness"),
                _clean_value(second, "answer_locality")
                or _clean_value(second, "answer_locality_missingness"),
            ),
        }

    features = {
        name: (_clean_value(first, name), _clean_value(second, name))
        for name in (
            "exhaustivity_requirement",
            "negative_conclusiveness",
            "presupposition_load",
            "qdmr_applicability",
            "recall_orientation",
            "cognitive_process_level",
        )
    }
    features["referring_form_type"] = (
        _clean_value(first, "referring_form_type")
        or _clean_value(first, "referring_form_missingness"),
        _clean_value(second, "referring_form_type")
        or _clean_value(second, "referring_form_missingness"),
    )
    if all(
        _clean_value(row, "qdmr_applicability") != "not_applicable"
        for row in (first, second)
    ):
        for name in ("qdmr_step_count", "qdmr_operator_set", "hop_structure"):
            features[name] = (
                _clean_value(first, name),
                _clean_value(second, name),
            )
        if any(
            _clean_value(row, "qdmr_applicability")
            == "applicable_after_normalisation"
            for row in (first, second)
        ):
            features["qdmr_normalized_question"] = (
                _clean_value(first, "qdmr_normalized_question"),
                _clean_value(second, "qdmr_normalized_question"),
            )
    return features


def _clean_value(row: Mapping[str, object], key: str) -> object:
    value = row.get(key)
    if isinstance(value, list) or value is None:
        return value
    return None if bool(pd.isna(value)) else value


def _annotation_verification(normalized: pd.DataFrame, agreement: pd.DataFrame) -> dict[str, object]:
    failures = []
    if normalized["item_id"].duplicated().sum() != 577:
        failures.append("item_model_cardinality")
    if len(agreement) == 0:
        failures.append("agreement_empty")
    feature_agreement = {
        str(feature): {
            "item_count": len(group),
            "exact_agreement_rate": float(group["agreed"].mean()),
            "mean_similarity": float(group["similarity"].mean()),
        }
        for feature, group in agreement.groupby("feature", sort=True)
    }
    return {
        "schema_version": 1,
        "passed": not failures,
        "blocking_failures": failures,
        "normalized_row_count": len(normalized),
        "unique_item_count": normalized["item_id"].nunique(),
        "agreement_row_count": len(agreement),
        "agreement_rate": float(agreement["agreed"].mean()) if len(agreement) else None,
        "feature_agreement": feature_agreement,
        "question_annotation_count": int(
            normalized["item_type"].eq("question").sum()
        ),
        "expectation_annotation_count": int(
            normalized["item_type"].eq("expectation").sum()
        ),
        "outcome_columns_loaded": [],
    }


def _render_report(verification: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Partial report 03 — Outcome-blind annotation reliability",
            "",
            f"Annotation ingestion: **{'PASS' if verification['passed'] else 'FAIL'}**.",
            f"- Normalized annotation rows: {verification['normalized_row_count']}.",
            f"- Unique items: {verification['unique_item_count']}.",
            f"- Agreement rows: {verification['agreement_row_count']}.",
            f"- Exact agreement rate: {verification['agreement_rate']}.",
            "",
            "## Preliminary feature agreement",
            "",
            *[
                f"- `{feature}`: n={summary['item_count']}, exact={summary['exact_agreement_rate']:.3f}, "
                f"mean scale-aware similarity={summary['mean_similarity']:.3f}."
                for feature, summary in cast(
                    dict[str, dict[str, int | float]],
                    verification["feature_agreement"],
                ).items()
            ],
            "",
            "- Outcome columns loaded: none.",
            "- Inter-model agreement is reliability evidence, not validity evidence.",
            "",
        ]
    )


def _versioned(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "artifact_schema_version", 1)
    return frame


def _reject_outcome_keys(keys: Iterable[object]) -> None:
    found = [str(key) for key in keys if OUTCOME_KEY_PATTERN.search(str(key))]
    if found:
        raise GateFailureError(f"Outcome field denial: {sorted(found)}")


def _reject_outcome_keys_from_value(value: object) -> None:
    if isinstance(value, Mapping):
        _reject_outcome_keys(value.keys())
        for nested in value.values():
            _reject_outcome_keys_from_value(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_outcome_keys_from_value(nested)
