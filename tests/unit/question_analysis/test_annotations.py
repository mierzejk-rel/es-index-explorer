"""Tests for strict, outcome-blind P4 annotation contracts."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pandas as pd
import pytest
from cursor_sdk import AgentOptions
from pydantic import ValidationError

from es_index_explorer.question_analysis.annotations import (
    ExpectationAnnotation,
    QuestionAnnotation,
    ResolvedModel,
    _annotation_manifest,
    _build_batches,
    _prompt_for_batch,
    _reject_outcome_keys_from_value,
    resolve_models,
    run_annotate_emit,
    run_annotate_ingest,
    run_annotate_run,
)
from es_index_explorer.question_analysis.contracts import WorkflowCommand
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    PrerequisiteError,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

pytestmark = pytest.mark.unit


class FakeParameter:
    """One minimal SDK model parameter."""

    def __init__(self, identifier: str, values: tuple[str, ...]) -> None:
        self.id = identifier
        self.values = tuple(type("Value", (), {"value": value})() for value in values)


class FakeModel:
    """One minimal account-visible SDK model."""

    def __init__(
        self,
        identifier: str,
        display_name: str,
        parameters: tuple[FakeParameter, ...] = (),
        default_parameters: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.id = identifier
        self.display_name = display_name
        self.parameters = parameters
        params = tuple(
            type("ParameterValue", (), {"id": key, "value": value})()
            for key, value in default_parameters
        )
        self.variants = (type("Variant", (), {"params": params, "is_default": True})(),)


@dataclass(frozen=True)
class FakeResult:
    """Minimal successful Cursor run result."""

    agent_id: str
    id: str
    status: str
    result: str
    created_at: str = "2026-09-07T00:00:00Z"
    duration_ms: int = 1
    usage: None = None


class FakeSdk:
    """Thread-safe deterministic Cursor SDK substitute."""

    def __init__(self, models: tuple[FakeModel, ...]) -> None:
        self.models = models

    def list_models(self, api_key: str) -> tuple[FakeModel, ...]:
        _ = api_key
        return self.models

    def prompt(self, message: str, options: AgentOptions) -> FakeResult:
        batch = json.loads(message.rsplit("\n\nBATCH:\n", maxsplit=1)[1])
        selection = cast(Mapping[str, object], options.model)
        model_id = str(selection["id"])
        rows = []
        for item in batch["items"]:
            if item["item_type"] == "question":
                rows.append(
                    {
                        "item_id": item["item_id"],
                        "exhaustivity_requirement": "mention_some",
                        "negative_conclusiveness": False,
                        "presupposition_load": False,
                        "qdmr_applicability": "not_applicable",
                        "qdmr_step_count": None,
                        "qdmr_operator_set": None,
                        "qdmr_normalized_question": None,
                        "hop_structure": None,
                        "referring_form_type": None,
                        "referring_form_missingness": "no_focal_referent",
                        "recall_orientation": "precision_oriented",
                        "cognitive_process_level": "remember",
                    }
                )
            else:
                rows.append(
                    {
                        "item_id": item["item_id"],
                        "demand_type": "relational_claim",
                        "specificity": "specific",
                        "answer_locality": "single_passage",
                        "answer_locality_missingness": None,
                    }
                )
        response = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
        batch_id = str(batch["batch_id"])
        return FakeResult(
            agent_id=f"agent-{model_id}-{batch_id}",
            id=f"run-{model_id}-{batch_id}",
            status="finished",
            result=response,
        )


def test_question_annotation_requires_explicit_missingness() -> None:
    annotation = QuestionAnnotation.model_validate(
        {
            "item_id": "question-1",
            "exhaustivity_requirement": "mention_all",
            "negative_conclusiveness": False,
            "presupposition_load": False,
            "qdmr_applicability": "not_applicable",
            "qdmr_step_count": None,
            "qdmr_operator_set": None,
            "qdmr_normalized_question": None,
            "hop_structure": None,
            "referring_form_type": None,
            "referring_form_missingness": "no_focal_referent",
            "recall_orientation": "recall_oriented",
            "cognitive_process_level": "analyze",
        }
    )

    assert annotation.qdmr_step_count is None


def test_question_annotation_rejects_unsorted_or_unknown_qdmr_operator() -> None:
    payload = {
        "item_id": "question-1",
        "exhaustivity_requirement": "mention_some",
        "negative_conclusiveness": False,
        "presupposition_load": False,
        "qdmr_applicability": "applicable",
        "qdmr_step_count": 1,
        "qdmr_operator_set": ["SELECT", "FILTER"],
        "qdmr_normalized_question": None,
        "hop_structure": "atomic",
        "referring_form_type": "full_name_form",
        "referring_form_missingness": None,
        "recall_orientation": "precision_oriented",
        "cognitive_process_level": "remember",
    }

    with pytest.raises(ValidationError, match="sorted and unique"):
        QuestionAnnotation.model_validate(payload)
    payload["qdmr_operator_set"] = ["UNKNOWN"]
    with pytest.raises(ValidationError, match="unknown operator"):
        QuestionAnnotation.model_validate(payload)


def test_expectation_annotation_requires_binary_locality_or_reason() -> None:
    with pytest.raises(ValidationError, match="classified or explicitly missing"):
        ExpectationAnnotation.model_validate(
            {
                "item_id": "expectation-1",
                "demand_type": "entity_identification",
                "specificity": "specific",
            }
        )


def test_annotation_batches_are_complete_and_deterministic() -> None:
    items = pd.DataFrame(
        {
            "item_id": [f"item-{index:03d}" for index in range(577)],
            "item_type": ["question"] * 263 + ["expectation"] * 314,
            "source_text": [f"Text {index}" for index in range(577)],
            "text_sha256": [f"{index:064x}" for index in range(577)],
        }
    )

    first = _build_batches(items)
    second = _build_batches(items)

    assert first == second
    assert len(first) == 25
    assert len(first["batch-001"]["items"]) == 24
    assert len(first["batch-025"]["items"]) == 1
    assert {
        item["item_id"] for batch in first.values() for item in batch["items"]
    } == set(items["item_id"])


def test_recursive_outcome_denial_is_case_insensitive() -> None:
    with pytest.raises(GateFailureError, match="Outcome field denial"):
        _reject_outcome_keys_from_value({"labels": [{"RubricV2": 1}]})


def test_model_resolution_requires_exact_two_selections() -> None:
    models = (
        FakeModel(
            "claude-opus-5",
            "Claude Opus 5",
            (
                FakeParameter("thinking", ("false", "true")),
                FakeParameter("effort", ("low", "high")),
            ),
            (("thinking", "true"), ("effort", "high")),
        ),
        FakeModel("gpt-5.6-sol", "GPT-5.6 Sol"),
    )

    resolved = resolve_models(models)

    assert resolved == (
        ResolvedModel(
            "Claude Opus 5 (high thinking)",
            "claude-opus-5",
            "Claude Opus 5",
            (("thinking", "true"), ("effort", "high")),
        ),
        ResolvedModel("GPT-5.6 Sol", "gpt-5.6-sol", "GPT-5.6 Sol", ()),
    )
    with pytest.raises(GateFailureError, match="unavailable or ambiguous"):
        resolve_models(models[:1])


def test_annotation_manifest_contains_deterministic_batch_hashes() -> None:
    model = ResolvedModel("GPT-5.6 Sol", "gpt", "GPT-5.6 Sol", ())
    manifest = _annotation_manifest(
        (model,),
        {"batch-001": {"item_ids_sha256": "a" * 64}},
    )

    assert manifest["batch_hashes"] == {"batch-001": "a" * 64}
    assert json.loads(json.dumps(manifest))["sdk_version"]


def test_prompt_carries_strict_schemas_and_only_batch_input() -> None:
    prompt = _prompt_for_batch(
        {
            "batch_id": "batch-001",
            "items": [{"item_id": "question-1", "item_type": "question", "source_text": "Who?", "text_sha256": "a" * 64}],
        }
    )

    assert "QUESTION_SCHEMA:" in prompt
    assert "EXPECTATION_SCHEMA:" in prompt
    assert "BATCH:" in prompt
    assert "RubricV2" not in prompt
    assert '"item_id": "item-001"' in prompt
    assert "question-1" not in prompt
    assert "a" * 64 not in prompt


def test_emit_writes_frozen_batches_and_missing_key_blocks_run(tmp_path) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)
    workspace.run_step(WorkflowCommand.JOIN, lambda current: ())
    rows = pd.DataFrame(
        {
            "artifact_schema_version": [1] * 577,
            "item_id": [f"item-{index:03d}" for index in range(577)],
            "item_type": ["question"] * 263 + ["expectation"] * 314,
            "source_text": [f"Text {index}" for index in range(577)],
            "text_sha256": [f"{index:064x}" for index in range(577)],
        }
    )
    path = workspace.store.path_for("tables/features_deterministic.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    workspace.run_step(WorkflowCommand.FEATURES, lambda current: ())

    assert workspace.run_step(WorkflowCommand.ANNOTATE_EMIT, run_annotate_emit)
    index = json.loads(
        workspace.store.path_for("annotations/batches/index.json").read_text(encoding="utf-8")
    )
    assert index == {"batch_count": 25, "batch_ids": [f"batch-{value:03d}" for value in range(1, 26)], "schema_version": 1}
    with pytest.raises(PrerequisiteError, match="CURSOR_API_KEY"):
        run_annotate_run(workspace, api_key="")


def test_concurrent_run_is_resumable_and_ingests_all_annotations(tmp_path) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)
    workspace.run_step(WorkflowCommand.JOIN, lambda current: ())
    rows = pd.DataFrame(
        {
            "artifact_schema_version": [1] * 577,
            "item_id": [f"source-{index:03d}" for index in range(577)],
            "item_type": ["question"] * 263 + ["expectation"] * 314,
            "source_text": [f"Text {index}" for index in range(577)],
            "text_sha256": [f"{index:064x}" for index in range(577)],
        }
    )
    feature_path = workspace.store.path_for("tables/features_deterministic.parquet")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(feature_path, index=False)
    workspace.run_step(WorkflowCommand.FEATURES, lambda current: ())
    workspace.run_step(WorkflowCommand.ANNOTATE_EMIT, run_annotate_emit)
    models = (
        FakeModel(
            "claude-opus-5",
            "Claude Opus 5",
            (
                FakeParameter("thinking", ("false", "true")),
                FakeParameter("effort", ("low", "high")),
            ),
            (("thinking", "true"), ("effort", "high")),
        ),
        FakeModel("gpt-5.6-sol", "GPT-5.6 Sol"),
    )
    sdk = FakeSdk(models)

    assert workspace.run_step(
        WorkflowCommand.ANNOTATE_RUN,
        lambda current: run_annotate_run(current, sdk=sdk, api_key="test-key"),
    )
    interrupted_raw = workspace.store.path_for(
        "annotations/raw/gpt-5.6-sol/batch-025.jsonl"
    )
    interrupted_attempt = workspace.store.path_for(
        "annotations/attempts/gpt-5.6-sol/batch-025.json"
    )
    interrupted_raw.unlink()
    interrupted_attempt.unlink()
    run_annotate_run(workspace, sdk=sdk, api_key="test-key")
    assert interrupted_raw.is_file()
    assert interrupted_attempt.is_file()
    assert not workspace.run_step(
        WorkflowCommand.ANNOTATE_RUN,
        lambda current: run_annotate_run(current, sdk=sdk, api_key="test-key"),
    )
    assert workspace.run_step(WorkflowCommand.ANNOTATE_INGEST, run_annotate_ingest)

    normalized = pd.read_parquet(
        workspace.store.path_for("tables/annotations_normalized.parquet")
    )
    envelopes = list(
        workspace.store.path_for("annotations/envelopes").rglob("*.json")
    )
    assert len(normalized) == 1_154
    assert normalized["item_id"].nunique() == 577
    assert len(envelopes) == 50
