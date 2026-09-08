"""Tests for human gold ingestion and provisional LLM provenance."""

import csv
import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from es_index_explorer.question_analysis.contracts import (
    AnnotatorKind,
    StepRecord,
    StepStatus,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
    PrerequisiteError,
)
from es_index_explorer.question_analysis.gold import (
    GoldSampleResult,
    HumanRecodeProvenance,
    InitialHumanGoldProvenance,
    ProvisionalLlmProvenance,
    attach_human_recode,
    build_gold_sample,
    normalize_initial_human_labels,
    normalize_provisional_recode,
    run_gold_ingest,
    run_gold_ingest_initial,
    run_gold_ingest_provisional,
    run_gold_recode_release,
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
                "feature": feature,
                "first_model_id": "claude-opus-5",
                "second_model_id": "gpt-5.6-sol",
                "first_value": (
                    '"precision_oriented"'
                    if feature == "recall_orientation"
                    else "false"
                ),
                "second_value": (
                    '"precision_oriented"'
                    if feature == "recall_orientation"
                    else "false"
                ),
                "agreed": True,
            }
            for index in range(3)
            for feature in ("recall_orientation", "negative_conclusiveness")
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


def _initial_provenance() -> InitialHumanGoldProvenance:
    initial = datetime(2026, 9, 8, 12, tzinfo=UTC)
    return InitialHumanGoldProvenance(
        annotator_kind=AnnotatorKind.HUMAN,
        annotator_id="coder-1",
        annotator_role="initial gold coder",
        expertise="Project-domain knowledge.",
        qualification="Project author; not an independent expert.",
        annotation_environment="Isolated local CSV editor.",
        bundle_manifest_sha256="a" * 64,
        completed_adjudication_sha256="e" * 64,
        schema_sha256="b" * 64,
        codebook_sha256="c" * 64,
        decision_ledger_sha256="d" * 64,
        started_at=initial,
        completed_at=initial + timedelta(hours=1),
        outcome_blindness_confirmed=True,
    )


def _human_value(feature: str) -> str:
    return '"precision_oriented"' if feature == "recall_orientation" else "false"


def _provisional_provenance() -> ProvisionalLlmProvenance:
    return ProvisionalLlmProvenance(
        annotator_kind=AnnotatorKind.PROVISIONAL_LLM,
        provider="Cursor",
        interface="Python SDK",
        model="frontier-model",
        model_version="2026-09",
        prompt_sha256="a" * 64,
        recode_manifest_sha256="f" * 64,
        completed_recode_sha256="1" * 64,
        initial_labels_sha256="2" * 64,
        schema_sha256="3" * 64,
        codebook_sha256="b" * 64,
        decision_ledger_sha256="d" * 64,
        parameters={"reasoning": "high"},
        session_id="fresh-session",
        context_isolated=True,
        context_isolation_description="Fresh empty project with bundle only.",
        started_at=datetime(2026, 9, 22, 12, tzinfo=UTC),
        completed_at=datetime(2026, 9, 22, 12, 30, tzinfo=UTC),
        status="finished",
        raw_response_sha256="e" * 64,
        outcome_blindness_confirmed=True,
    )


def test_feature_long_initial_and_recode_round_trip_uses_composite_keys() -> None:
    result = _result()
    initial_template = _rows(result.adjudication_csv)
    recode_template = _rows(result.recode_csv)
    initial = [
        dict(row, human_value_json=_human_value(row["feature"]))
        for row in initial_template
    ]
    initial_values = {
        (row["local_item_id"], row["feature"]): row["human_value_json"]
        for row in initial
    }
    mapping_rows = result.recode_mapping["rows"]
    assert isinstance(mapping_rows, list)
    mapping_by_recode = {
        str(row["recode_item_id"]): (
            str(row["local_item_id"]),
            str(row["feature"]),
        )
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

    normalized = normalize_initial_human_labels(
        initial_template,
        initial,
        _initial_provenance(),
    )
    normalized = attach_human_recode(
        normalized,
        recode_template,
        recode,
        result.recode_mapping,
        HumanRecodeProvenance(
            annotator_kind=AnnotatorKind.HUMAN,
            annotator_id="coder-1",
            annotator_role="delayed gold coder",
            expertise="Project-domain knowledge.",
            qualification="Project author; not an independent expert.",
            annotation_environment="Fresh isolated CSV editor.",
            recode_manifest_sha256="e" * 64,
            completed_recode_sha256="a" * 64,
            initial_labels_sha256="f" * 64,
            schema_sha256="b" * 64,
            codebook_sha256="c" * 64,
            decision_ledger_sha256="d" * 64,
            started_at=datetime(2026, 9, 23, 12, tzinfo=UTC),
            completed_at=datetime(2026, 9, 23, 13, tzinfo=UTC),
            context_isolated=True,
            context_isolation_description="No initial labels or comments available.",
            outcome_blindness_confirmed=True,
        ),
        received_at=datetime(2026, 9, 23, 14, tzinfo=UTC),
    )

    assert len(normalized) == 6
    assert not normalized.duplicated(["local_item_id", "feature"]).any()
    assert normalized["human_recode_value_json"].notna().sum() == 4
    assert set(normalized["annotator_kind"]) == {AnnotatorKind.HUMAN}
    with pytest.raises(MalformedInputError, match="duplicate keys"):
        normalize_initial_human_labels(
            initial_template,
            [*initial, initial[0]],
            _initial_provenance(),
        )
    with pytest.raises(MalformedInputError, match="row identity"):
        normalize_initial_human_labels(
            initial_template,
            initial[:-1],
            _initial_provenance(),
        )
    with pytest.raises(GateFailureError, match="outcome fields"):
        normalize_initial_human_labels(
            initial_template,
            [dict(row, grade="A") for row in initial],
            _initial_provenance(),
        )


def test_provisional_normalization_rejects_mapping_and_identity_corruption() -> None:
    result = _result()
    initial_template = _rows(result.adjudication_csv)
    initial = [
        dict(row, human_value_json=_human_value(row["feature"]))
        for row in initial_template
    ]
    initial_labels = normalize_initial_human_labels(
        initial_template,
        initial,
        _initial_provenance(),
    )
    item_ids = {
        local_id: f"item-{index}"
        for index, local_id in enumerate(initial_labels["local_item_id"].unique())
    }
    initial_labels = initial_labels.assign(
        item_id=initial_labels["local_item_id"].map(item_ids),
        stratum_id="stratum",
        inclusion_probability=1.0,
    )
    recode_template = _rows(result.recode_csv)
    recode = [
        dict(row, recode_value_json=_human_value(row["feature"]))
        for row in recode_template
    ]
    received_at = datetime(2026, 9, 22, 13, tzinfo=UTC)
    normalized = normalize_provisional_recode(
        initial_labels,
        recode_template,
        recode,
        result.recode_mapping,
        _provisional_provenance(),
        received_at=received_at,
    )
    assert len(normalized) == len(recode)

    with pytest.raises(MalformedInputError, match="row identity"):
        normalize_provisional_recode(
            initial_labels,
            recode_template,
            recode[:-1],
            result.recode_mapping,
            _provisional_provenance(),
            received_at=received_at,
        )
    with pytest.raises(MalformedInputError, match="mapping is invalid"):
        normalize_provisional_recode(
            initial_labels,
            recode_template,
            recode,
            {"rows": "invalid"},
            _provisional_provenance(),
            received_at=received_at,
        )
    mapping_value = result.recode_mapping["rows"]
    assert isinstance(mapping_value, list)
    mapping_rows = list(mapping_value)
    with pytest.raises(MalformedInputError, match="mapping keys"):
        normalize_provisional_recode(
            initial_labels,
            recode_template,
            recode,
            {"rows": [*mapping_rows, mapping_rows[0]]},
            _provisional_provenance(),
            received_at=received_at,
        )
    bad_probability_rows = [dict(row) for row in mapping_rows]
    bad_probability_rows[0]["recode_joint_inclusion_probability"] = "invalid"
    with pytest.raises(MalformedInputError, match="probabilities"):
        normalize_provisional_recode(
            initial_labels,
            recode_template,
            recode,
            {"rows": bad_probability_rows},
            _provisional_provenance(),
            received_at=received_at,
        )


def test_initial_human_gold_rejects_invalid_timing_and_changed_source_text() -> None:
    invalid = _initial_provenance().model_dump()
    invalid["started_at"] = datetime(2026, 9, 8, 14, tzinfo=UTC)
    invalid["completed_at"] = datetime(2026, 9, 8, 13, tzinfo=UTC)
    with pytest.raises(ValidationError, match="precedes"):
        InitialHumanGoldProvenance.model_validate(invalid)
    result = _result()
    initial_template = _rows(result.adjudication_csv)
    initial = [
        dict(row, human_value_json=_human_value(row["feature"]))
        for row in initial_template
    ]
    initial[0]["source_text"] = "Changed"

    with pytest.raises(MalformedInputError, match="immutable fields"):
        normalize_initial_human_labels(
            initial_template,
            initial,
            _initial_provenance(),
        )


def test_provisional_llm_provenance_can_never_unlock() -> None:
    provenance = _provisional_provenance()

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
    initial_time = datetime(2026, 9, 8, 12, tzinfo=UTC)
    state = workspace.load_state()
    steps = dict(state.steps)
    for command in (
        WorkflowCommand.JOIN,
        WorkflowCommand.FEATURES,
        WorkflowCommand.ANNOTATE_EMIT,
        WorkflowCommand.ANNOTATE_RUN,
        WorkflowCommand.ANNOTATE_INGEST,
    ):
        steps[command] = StepRecord(
            status=StepStatus.COMPLETED,
            attempts=1,
            started_at=initial_time - timedelta(hours=2),
            completed_at=initial_time - timedelta(hours=1),
        )
    workspace.save_state(state.model_copy(update={"steps": steps}))
    agreement = pd.DataFrame(
        [
            {
                "item_id": f"variant-{index:03d}",
                "item_type": "question",
                "feature": feature,
                "first_model_id": "claude-opus-5",
                "second_model_id": "gpt-5.6-sol",
                "first_value": (
                    '"precision_oriented"'
                    if feature == "recall_orientation"
                    else "false"
                ),
                "second_value": (
                    '"precision_oriented"'
                    if feature == "recall_orientation"
                    else "false"
                ),
                "agreed": True,
            }
            for index in range(3)
            for feature in ("recall_orientation", "negative_conclusiveness")
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

    workspace.run_step(
        WorkflowCommand.GOLD_SAMPLE,
        run_gold_sample,
        now=initial_time,
    )

    assert not (root / "gold" / "delayed_recode.csv").exists()
    initial_template = _rows((root / "gold" / "adjudication.csv").read_bytes())
    initial = [
        dict(row, human_value_json=_human_value(row["feature"]))
        for row in initial_template
    ]
    initial_path = tmp_path / "completed-initial.csv"
    initial_path.write_text(
        _csv_from_rows(initial, tuple(initial[0])).decode("utf-8"),
        encoding="utf-8",
    )
    bundle_manifest = json.loads(
        (root / "gold" / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    initial_provenance = InitialHumanGoldProvenance(
        annotator_kind=AnnotatorKind.HUMAN,
        annotator_id="coder-1",
        annotator_role="initial gold coder",
        expertise="Project-domain knowledge.",
        qualification="Project author; not an independent expert.",
        annotation_environment="Isolated local CSV editor.",
        bundle_manifest_sha256=sha256_file(root / "gold" / "bundle_manifest.json"),
        completed_adjudication_sha256=sha256_file(initial_path),
        schema_sha256=bundle_manifest["schema_sha256"],
        codebook_sha256=bundle_manifest["codebook_sha256"],
        decision_ledger_sha256=bundle_manifest["decision_ledger_sha256"],
        started_at=initial_time,
        completed_at=initial_time + timedelta(hours=1),
        outcome_blindness_confirmed=True,
    )
    initial_provenance_path = tmp_path / "initial-provenance.json"
    initial_provenance_path.write_bytes(canonical_json_bytes(initial_provenance))
    wrong_hash_path = tmp_path / "wrong-hash-initial-provenance.json"
    wrong_hash_path.write_bytes(
        canonical_json_bytes(
            initial_provenance.model_copy(update={"codebook_sha256": "0" * 64})
        )
    )
    with pytest.raises(MalformedInputError, match="analysis lock"):
        workspace.run_step(
            WorkflowCommand.GOLD_INGEST_INITIAL,
            lambda current: run_gold_ingest_initial(
                current, initial_path, wrong_hash_path
            ),
            now=initial_time + timedelta(hours=2),
        )
    backdated_path = tmp_path / "backdated-initial-provenance.json"
    backdated_path.write_bytes(
        canonical_json_bytes(
            initial_provenance.model_copy(
                update={"started_at": initial_time - timedelta(seconds=1)}
            )
        )
    )
    with pytest.raises(GateFailureError, match="before gold bundle emission"):
        workspace.run_step(
            WorkflowCommand.GOLD_INGEST_INITIAL,
            lambda current: run_gold_ingest_initial(
                current, initial_path, backdated_path
            ),
            now=initial_time + timedelta(hours=2),
        )
    workspace.run_step(
        WorkflowCommand.GOLD_INGEST_INITIAL,
        lambda current: run_gold_ingest_initial(
            current, initial_path, initial_provenance_path
        ),
        now=initial_time + timedelta(hours=2),
    )

    with pytest.raises(GateFailureError, match="cannot be released"):
        workspace.run_step(
            WorkflowCommand.GOLD_RECODE_RELEASE,
            run_gold_recode_release,
            now=initial_time + timedelta(days=14, hours=2) - timedelta(seconds=1),
        )
    release_time = initial_time + timedelta(days=14, hours=2)
    workspace.run_step(
        WorkflowCommand.GOLD_RECODE_RELEASE,
        run_gold_recode_release,
        now=release_time,
    )
    recode_template = _rows((root / "gold" / "delayed_recode.csv").read_bytes())
    mapping = json.loads(
        (root / "gold" / "delayed_recode_mapping.json").read_text(encoding="utf-8")
    )
    initial_by_key = {
        (row["local_item_id"], row["feature"]): row["human_value_json"]
        for row in initial
    }
    mapping_by_recode = {
        str(row["recode_item_id"]): (
            str(row["local_item_id"]),
            str(row["feature"]),
        )
        for row in mapping["rows"]
    }
    recode = [
        dict(
            row,
            recode_value_json=initial_by_key[mapping_by_recode[row["recode_item_id"]]],
        )
        for row in recode_template
    ]
    recode_path = tmp_path / "completed-recode.csv"
    recode_path.write_text(
        _csv_from_rows(recode, tuple(recode[0])).decode("utf-8"),
        encoding="utf-8",
    )
    recode_manifest = json.loads(
        (root / "gold" / "delayed_recode_manifest.json").read_text(encoding="utf-8")
    )
    raw_response_path = tmp_path / "raw-provisional-response.bin"
    raw_response_path.write_bytes(b"provisional raw response")
    provisional_provenance = ProvisionalLlmProvenance(
        annotator_kind=AnnotatorKind.PROVISIONAL_LLM,
        provider="Cursor",
        interface="external isolated session",
        model="frontier-model",
        model_version="2026-09",
        prompt_sha256="a" * 64,
        recode_manifest_sha256=sha256_file(
            root / "gold" / "delayed_recode_manifest.json"
        ),
        completed_recode_sha256=sha256_file(recode_path),
        initial_labels_sha256=recode_manifest["initial_labels_sha256"],
        schema_sha256=recode_manifest["schema_sha256"],
        codebook_sha256=recode_manifest["codebook_sha256"],
        decision_ledger_sha256=recode_manifest["decision_ledger_sha256"],
        parameters={"reasoning": "high"},
        session_id="isolated-session",
        context_isolated=True,
        context_isolation_description="No outcomes or initial labels available.",
        started_at=release_time,
        completed_at=release_time + timedelta(hours=1),
        status="finished",
        raw_response_sha256=sha256_file(raw_response_path),
        outcome_blindness_confirmed=True,
    )
    provisional_provenance_path = tmp_path / "provisional-provenance.json"
    provisional_provenance_path.write_bytes(
        canonical_json_bytes(provisional_provenance)
    )
    early_provisional_path = tmp_path / "early-provisional-provenance.json"
    early_provisional_path.write_bytes(
        canonical_json_bytes(
            provisional_provenance.model_copy(
                update={"started_at": release_time - timedelta(seconds=1)}
            )
        )
    )
    with pytest.raises(GateFailureError, match="before blind bundle release"):
        workspace.run_step(
            WorkflowCommand.GOLD_INGEST_PROVISIONAL,
            lambda current: run_gold_ingest_provisional(
                current,
                recode_path,
                early_provisional_path,
                raw_response_path,
            ),
            now=release_time + timedelta(hours=2),
        )
    future_provisional_path = tmp_path / "future-provisional-provenance.json"
    future_provisional_path.write_bytes(
        canonical_json_bytes(
            provisional_provenance.model_copy(
                update={"completed_at": release_time + timedelta(hours=3)}
            )
        )
    )
    with pytest.raises(GateFailureError, match="future-dated"):
        workspace.run_step(
            WorkflowCommand.GOLD_INGEST_PROVISIONAL,
            lambda current: run_gold_ingest_provisional(
                current,
                recode_path,
                future_provisional_path,
                raw_response_path,
            ),
            now=release_time + timedelta(hours=2),
        )
    wrong_raw_response = tmp_path / "wrong-raw-response.bin"
    wrong_raw_response.write_bytes(b"tampered")
    with pytest.raises(MalformedInputError, match="raw response"):
        workspace.run_step(
            WorkflowCommand.GOLD_INGEST_PROVISIONAL,
            lambda current: run_gold_ingest_provisional(
                current,
                recode_path,
                provisional_provenance_path,
                wrong_raw_response,
            ),
            now=release_time + timedelta(hours=2),
        )
    workspace.run_step(
        WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        lambda current: run_gold_ingest_provisional(
            current,
            recode_path,
            provisional_provenance_path,
            raw_response_path,
        ),
        now=release_time + timedelta(hours=2),
    )
    provisional_labels = pd.read_parquet(
        root / "tables" / "gold_provisional_recode.parquet"
    )
    assert len(provisional_labels) == 4
    assert "human_recode_value_json" not in provisional_labels
    assert not provisional_labels["qualifies_as_human_recode"].any()
    state_after_provisional = workspace.load_state()
    assert (
        state_after_provisional.steps[WorkflowCommand.GOLD_INGEST].status
        is StepStatus.PENDING
    )
    assert not state_after_provisional.outcome_modeling_unlocked
    with pytest.raises(PrerequisiteError, match="gold-ingest"):
        workspace.run_step(
            WorkflowCommand.VALIDATE_FEATURES,
            lambda _: pytest.fail("Provisional evidence advanced validation"),
        )
    with pytest.raises(PrerequisiteError, match="validate-features"):
        workspace.run_step(
            WorkflowCommand.FIT_LAYER1,
            lambda _: pytest.fail("Provisional evidence unlocked fitting"),
        )

    recode_provenance = HumanRecodeProvenance(
        annotator_kind=AnnotatorKind.HUMAN,
        annotator_id="coder-1",
        annotator_role="delayed gold coder",
        expertise="Project-domain knowledge.",
        qualification="Project author; not an independent expert.",
        annotation_environment="Fresh isolated CSV editor.",
        recode_manifest_sha256=sha256_file(
            root / "gold" / "delayed_recode_manifest.json"
        ),
        completed_recode_sha256=sha256_file(recode_path),
        initial_labels_sha256=recode_manifest["initial_labels_sha256"],
        schema_sha256=recode_manifest["schema_sha256"],
        codebook_sha256=recode_manifest["codebook_sha256"],
        decision_ledger_sha256=recode_manifest["decision_ledger_sha256"],
        started_at=release_time,
        completed_at=release_time + timedelta(hours=1),
        context_isolated=True,
        context_isolation_description="No initial labels or comments available.",
        outcome_blindness_confirmed=True,
    )
    recode_provenance_path = tmp_path / "recode-provenance.json"
    recode_provenance_path.write_bytes(canonical_json_bytes(recode_provenance))
    future_path = tmp_path / "future-recode-provenance.json"
    future_path.write_bytes(
        canonical_json_bytes(
            recode_provenance.model_copy(
                update={"completed_at": release_time + timedelta(hours=3)}
            )
        )
    )
    with pytest.raises(GateFailureError, match="future-dated"):
        workspace.run_step(
            WorkflowCommand.GOLD_INGEST,
            lambda current: run_gold_ingest(current, recode_path, future_path),
            now=release_time + timedelta(hours=2),
        )
    workspace.run_step(
        WorkflowCommand.GOLD_INGEST,
        lambda current: run_gold_ingest(current, recode_path, recode_provenance_path),
        now=release_time + timedelta(hours=2),
    )

    labels = pd.read_parquet(root / "tables" / "gold_labels.parquet")
    assert len(labels) == 6
    assert labels["item_id"].nunique() == 3
    assert labels["human_recode_value_json"].notna().sum() == 4
    assert labels["recode_joint_inclusion_probability"].notna().sum() == 4
    assert not labels.duplicated(["item_id", "feature"]).any()
    recoded = labels.loc[labels["human_recode_value_json"].notna()]
    assert (
        recoded["recode_joint_inclusion_probability"]
        == recoded["inclusion_probability"] * recoded["recode_conditional_probability"]
    ).all()
    assert {
        "annotator_id",
        "annotator_expertise",
        "annotator_qualification",
        "annotation_environment",
        "recode_annotator_id",
        "recode_annotation_environment",
    }.issubset(labels.columns)
    state_before_replay = workspace.load_state()
    for command in (
        WorkflowCommand.GOLD_SAMPLE,
        WorkflowCommand.GOLD_INGEST_INITIAL,
        WorkflowCommand.GOLD_RECODE_RELEASE,
        WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        WorkflowCommand.GOLD_INGEST,
    ):
        assert not workspace.run_step(
            command,
            lambda _: pytest.fail("Completed gold step was executed again"),
        )
    assert workspace.load_state() == state_before_replay


def _csv_from_rows(
    rows: list[dict[str, str]],
    fieldnames: tuple[str, ...],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")
