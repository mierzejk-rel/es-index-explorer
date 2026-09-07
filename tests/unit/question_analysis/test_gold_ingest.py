"""Tests for human gold ingestion and provisional LLM provenance."""

import csv
import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from es_index_explorer.question_analysis.contracts import AnnotatorKind
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.gold import (
    GoldSampleResult,
    HumanGoldProvenance,
    ProvisionalLlmProvenance,
    build_gold_sample,
    normalize_human_labels,
    run_gold_ingest,
    run_gold_sample,
    validate_provisional_llm_provenance,
)
from es_index_explorer.question_analysis.storage import (
    canonical_json_bytes,
    sha256_file,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

pytestmark = pytest.mark.unit


def _result() -> GoldSampleResult:
    agreement = pd.DataFrame(
        [
            {
                "item_id": f"variant-{index:03d}",
                "item_type": "question",
                "feature": "recall_orientation",
                "first_model_id": "claude-opus-5",
                "second_model_id": "gpt-5.6-sol",
                "first_value": '"precision_oriented"',
                "second_value": '"precision_oriented"',
                "agreed": True,
            }
            for index in range(3)
        ]
    )
    features = pd.DataFrame(
        [
            {
                "item_id": f"variant-{index:03d}",
                "item_type": "question",
                "source_text": f"Question {index}?",
                "text_sha256": f"{index:064x}",
            }
            for index in range(3)
        ]
    )
    return build_gold_sample(agreement, features)


def _rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(payload.decode("utf-8"))))


def _provenance(delay_days: int = 14) -> HumanGoldProvenance:
    initial = datetime(2026, 9, 8, 12, tzinfo=UTC)
    return HumanGoldProvenance(
        annotator_kind=AnnotatorKind.HUMAN,
        annotator_role="initial gold coder",
        bundle_manifest_sha256="a" * 64,
        initial_completed_at=initial,
        recode_completed_at=initial + timedelta(days=delay_days),
        outcome_blindness_confirmed=True,
    )


def test_human_gold_normalization_requires_unchanged_templates_and_valid_values() -> (
    None
):
    result = _result()
    initial_template = _rows(result.adjudication_csv)
    recode_template = _rows(result.recode_csv)
    initial = [
        dict(row, human_value_json='"precision_oriented"') for row in initial_template
    ]
    initial_values = {row["local_item_id"]: row["human_value_json"] for row in initial}
    mapping_rows = result.recode_mapping["rows"]
    assert isinstance(mapping_rows, list)
    mapping_by_recode = {
        str(row["recode_item_id"]): str(row["local_item_id"])
        for row in mapping_rows
        if isinstance(row, dict)
    }
    recode = [
        dict(
            row,
            recode_value_json=initial_values[mapping_by_recode[row["recode_item_id"]]],
        )
        for row in recode_template
    ]

    normalized = normalize_human_labels(
        initial_template,
        initial,
        recode_template,
        recode,
        result.recode_mapping,
        _provenance(),
    )

    assert len(normalized) == 3
    assert normalized["human_gold_value_json"].eq('"precision_oriented"').all()
    assert normalized["human_recode_value_json"].notna().sum() == 2
    assert set(normalized["annotator_kind"]) == {AnnotatorKind.HUMAN}


def test_human_gold_rejects_short_delay_and_changed_source_text() -> None:
    with pytest.raises(ValidationError, match="at least 14 days"):
        _provenance(delay_days=13)

    result = _result()
    initial_template = _rows(result.adjudication_csv)
    initial = [
        dict(row, human_value_json='"precision_oriented"') for row in initial_template
    ]
    initial[0]["source_text"] = "Changed"

    with pytest.raises(MalformedInputError, match="immutable fields"):
        normalize_human_labels(
            initial_template,
            initial,
            _rows(result.recode_csv),
            _rows(result.recode_csv),
            result.recode_mapping,
            _provenance(),
        )


def test_provisional_llm_provenance_can_never_unlock() -> None:
    provenance = ProvisionalLlmProvenance(
        annotator_kind=AnnotatorKind.PROVISIONAL_LLM,
        provider="Cursor",
        interface="Python SDK",
        model="frontier-model",
        model_version="2026-09",
        prompt_sha256="a" * 64,
        codebook_sha256="b" * 64,
        bundle_sha256="c" * 64,
        decision_ledger_sha256="d" * 64,
        parameters={"reasoning": "high"},
        session_id="fresh-session",
        context_isolated=True,
        context_isolation_description="Fresh empty project with bundle only.",
        started_at=datetime(2026, 9, 22, 12, tzinfo=UTC),
        completed_at=datetime(2026, 9, 22, 12, 30, tzinfo=UTC),
        status="finished",
        raw_response_sha256="e" * 64,
    )

    evidence = validate_provisional_llm_provenance(provenance)

    assert evidence["qualifies_as_human_recode"] is False
    assert evidence["can_unlock_outcome_modeling"] is False
    assert (
        evidence["unresolved_validation_item"]
        == "INDEPENDENT_HUMAN_EXPERT_VERIFICATION_REQUIRED"
    )


def test_gold_handlers_persist_human_exchange_and_normalized_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analysis"
    specification = tmp_path / "specification.md"
    specification.write_text("# Specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(root, specification)
    synthetic = _result()
    agreement = pd.DataFrame(
        [
            {
                "item_id": f"variant-{index:03d}",
                "item_type": "question",
                "feature": "recall_orientation",
                "first_model_id": "claude-opus-5",
                "second_model_id": "gpt-5.6-sol",
                "first_value": '"precision_oriented"',
                "second_value": '"precision_oriented"',
                "agreed": True,
            }
            for index in range(3)
        ]
    )
    features = pd.DataFrame(
        [
            {
                "item_id": f"variant-{index:03d}",
                "item_type": "question",
                "source_text": f"Question {index}?",
                "text_sha256": f"{index:064x}",
            }
            for index in range(3)
        ]
    )
    agreement.insert(0, "artifact_schema_version", 1)
    features.insert(0, "artifact_schema_version", 1)
    agreement.to_parquet(root / "tables" / "annotation_agreement.parquet", index=False)
    features.to_parquet(root / "tables" / "features_deterministic.parquet", index=False)

    sample_artifacts = run_gold_sample(workspace)

    assert len(sample_artifacts) == 6
    initial_template = _rows((root / "gold" / "adjudication.csv").read_bytes())
    recode_template = _rows((root / "gold" / "delayed_recode.csv").read_bytes())
    initial = [
        dict(row, human_value_json='"precision_oriented"') for row in initial_template
    ]
    mapping = json.loads(
        (root / "gold" / "delayed_recode_mapping.json").read_text(encoding="utf-8")
    )
    initial_by_local = {
        row["local_item_id"]: row["human_value_json"] for row in initial
    }
    mapping_by_recode = {
        str(row["recode_item_id"]): str(row["local_item_id"]) for row in mapping["rows"]
    }
    recode = [
        dict(
            row,
            recode_value_json=initial_by_local[
                mapping_by_recode[row["recode_item_id"]]
            ],
        )
        for row in recode_template
    ]
    initial_path = tmp_path / "completed-initial.csv"
    recode_path = tmp_path / "completed-recode.csv"
    initial_path.write_text(
        _csv_from_rows(initial, tuple(initial[0])).decode("utf-8"),
        encoding="utf-8",
    )
    recode_path.write_text(
        _csv_from_rows(recode, tuple(recode[0])).decode("utf-8"),
        encoding="utf-8",
    )
    initial_time = datetime(2026, 9, 8, 12, tzinfo=UTC)
    provenance = HumanGoldProvenance(
        annotator_kind=AnnotatorKind.HUMAN,
        annotator_role="initial gold coder",
        bundle_manifest_sha256=sha256_file(root / "gold" / "bundle_manifest.json"),
        initial_completed_at=initial_time,
        recode_completed_at=initial_time + timedelta(days=14),
        outcome_blindness_confirmed=True,
    )
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    ingest_artifacts = run_gold_ingest(
        workspace,
        initial_path,
        recode_path,
        provenance_path,
    )

    assert synthetic.manifest["sample_item_count"] == 3
    assert len(ingest_artifacts) == 2
    labels = pd.read_parquet(root / "tables" / "gold_labels.parquet")
    assert len(labels) == 3
    assert labels["item_id"].nunique() == 3
    assert labels["human_recode_value_json"].notna().sum() == 2


def _csv_from_rows(
    rows: list[dict[str, str]],
    fieldnames: tuple[str, ...],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")
