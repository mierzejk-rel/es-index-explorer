"""Tests for the versioned Stage 4 outcome-field refusal contract."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from es_index_explorer.question_analysis.annotations import (
    _annotation_verification,
    _matching_outcome_keys,
    _parse_response,
    _parse_response_text,
    _reject_outcome_keys_from_value,
)
from es_index_explorer.question_analysis.contracts import (
    ANNOTATION_RESPONSE_FIELD_DENYLIST,
    OUTCOME_ARTIFACT_SCHEMAS,
    PRE_OUTCOME_INPUT_FIELD_DENYLIST,
    PROTECTED_RECOMMENDATION_ARTIFACT_NAMES,
    TRACE_PFU_TABLE_COLUMNS,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.storage import ArtifactStore

pytestmark = pytest.mark.unit

QuestionPayloadFactory = Callable[..., dict[str, object]]
PROJECT_ROOT = Path(__file__).parents[3]
OUTCOME_FIXTURE = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "outcome-field-refusal.jsonl"
)


@pytest.mark.parametrize("field_name", sorted(ANNOTATION_RESPONSE_FIELD_DENYLIST))
def test_parser_rejects_every_declared_outcome_field(
    field_name: str,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory()
    payload[field_name] = "forbidden"

    with pytest.raises(GateFailureError, match="Outcome field denial"):
        _parse_response_text(
            json.dumps(payload),
            question_batch_items,
            f"field-{field_name}",
        )


@pytest.mark.parametrize("field_name", ["GrAdE", "ARM_ID", "UnCeRtAiN"])
def test_outcome_field_refusal_is_case_insensitive(
    field_name: str,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory()
    payload[field_name] = "forbidden"

    with pytest.raises(GateFailureError, match="Outcome field denial"):
        _parse_response_text(json.dumps(payload), question_batch_items, "case")


def test_outcome_field_refusal_recurses_through_dicts_and_lists() -> None:
    with pytest.raises(
        GateFailureError, match=r"Outcome field denial.*logged_ordinal_grade"
    ):
        _reject_outcome_keys_from_value(
            {"metadata": [{"details": {"logged_ordinal_grade": "Good"}}]}
        )


def test_outcome_field_refusal_uses_exact_names() -> None:
    value = {
        "profile": "safe",
        "upgrade_reason": "safe",
        "run_identifier": "safe",
        "recommendation_context": "safe",
    }

    _reject_outcome_keys_from_value(value)

    assert _matching_outcome_keys(value) == []


def test_pre_outcome_inputs_allow_structural_ids_but_reject_outcome_fields() -> None:
    keys = ["rubric_id", "variant_id", "eval_dataset", "arm_id", "rubric_v2"]

    assert _matching_outcome_keys(
        keys, denied_fields=PRE_OUTCOME_INPUT_FIELD_DENYLIST
    ) == ["arm_id", "rubric_v2"]


def test_annotation_verification_reports_detected_outcome_columns() -> None:
    normalized = pd.DataFrame(
        {
            "item_id": ["item-001"],
            "item_type": ["question"],
            "model_id": ["model-1"],
            "qdmr_applicability": ["not_applicable"],
            "qdmr_step_count": [None],
            "qdmr_operator_set": [None],
            "cognitive_process_level": ["remember"],
            "grade": ["A"],
        }
    )
    agreement = pd.DataFrame(
        {"feature": ["qdmr_step_count"], "agreed": [True], "similarity": [1.0]}
    )

    verification = _annotation_verification(normalized, agreement)

    assert verification["outcome_columns_loaded"] == ["grade"]
    assert "outcome_columns_loaded" in cast(
        list[str], verification["blocking_failures"]
    )
    assert verification["passed"] is False


def test_frozen_outcome_response_fixture_is_rejected(
    question_batch_items: list[dict[str, object]],
) -> None:
    with pytest.raises(GateFailureError, match=r"Outcome field denial.*grade"):
        _parse_response(OUTCOME_FIXTURE, question_batch_items)


def test_trace_pfu_parquet_must_match_registered_schema(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    valid = pd.DataFrame(columns=["artifact_schema_version", *TRACE_PFU_TABLE_COLUMNS])

    metadata = store.write_parquet(
        "tables/trace_pfu_table.parquet",
        valid,
        created_by=WorkflowCommand.JOIN,
    )

    assert metadata.path == "tables/trace_pfu_table.parquet"
    with pytest.raises(MalformedInputError, match="schema mismatch"):
        store.write_parquet(
            "tables/trace_pfu_table.parquet",
            valid.drop(columns=TRACE_PFU_TABLE_COLUMNS[-1]),
            created_by=WorkflowCommand.JOIN,
        )


@pytest.mark.parametrize(
    "artifact_name", sorted(PROTECTED_RECOMMENDATION_ARTIFACT_NAMES)
)
def test_recommendation_parquet_requires_its_registered_schema(
    artifact_name: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path)
    valid = pd.DataFrame(
        columns=[
            "artifact_schema_version",
            *OUTCOME_ARTIFACT_SCHEMAS[artifact_name],
        ]
    )

    metadata = store.write_parquet(
        f"tables/{artifact_name}",
        valid,
        created_by=WorkflowCommand.FIT_LAYER1,
    )

    assert metadata.path == f"tables/{artifact_name}"
    with pytest.raises(MalformedInputError, match="schema mismatch"):
        store.write_parquet(
            f"tables/{artifact_name}",
            valid.drop(columns=OUTCOME_ARTIFACT_SCHEMAS[artifact_name][-1]),
            created_by=WorkflowCommand.FIT_LAYER1,
        )
