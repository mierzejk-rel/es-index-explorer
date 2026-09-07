"""Comprehensive tests for the Stage 4 QDMR response contract."""

import json
from collections.abc import Callable

import pandas as pd
import pytest

from es_index_explorer.question_analysis.annotations import (
    _agreement_rows,
    _annotation_cardinality,
    _annotation_verification,
    _closed_inventory_counts,
    _comparable_features,
    _normalize_annotation_dtypes,
    _parse_response_text,
    _qdmr_construct_anomalies,
    _render_report,
)
from es_index_explorer.question_analysis.contracts import (
    ANNOTATION_CONSTRUCT_ANOMALY_COLUMNS,
    COGNITIVE_PROCESS_LEVELS,
    QDMR_OPERATOR_INVENTORY,
    QDMR_OPERATORS,
    AnnotationConstructAnomaly,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)

pytestmark = pytest.mark.unit

QuestionPayloadFactory = Callable[..., dict[str, object]]
EXPECTED_MODEL_IDS = frozenset({"claude-opus-5", "gpt-5.6-sol"})


def _parse_payload(
    payload: dict[str, object], items: list[dict[str, object]]
) -> list[tuple[dict[str, object], dict[str, object]]]:
    return _parse_response_text(json.dumps(payload), items, "test-response")


def _normalized_question_row(model_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "item_id": "source-question-001",
        "item_type": "question",
        "model_id": model_id,
        "batch_id": "batch-001",
        "source_text_sha256": "a" * 64,
        "raw_response_sha256": "b" * 64,
        "exhaustivity_requirement": "mention_some",
        "negative_conclusiveness": False,
        "presupposition_load": False,
        "qdmr_applicability": "applicable",
        "qdmr_step_count": 2,
        "qdmr_operator_set": ["FILTER", "SELECT"],
        "qdmr_normalized_question": None,
        "hop_structure": "atomic",
        "referring_form_type": "full_name_form",
        "referring_form_missingness": None,
        "recall_orientation": "precision_oriented",
        "cognitive_process_level": "remember",
    }
    row.update(overrides)
    return row


def _balanced_cardinality_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": f"item-{item_index:03d}",
                "model_id": model_id,
            }
            for item_index in range(577)
            for model_id in sorted(EXPECTED_MODEL_IDS)
        ]
    )


def test_annotation_cardinality_requires_exactly_two_expected_models_per_item() -> None:
    cardinality = _annotation_cardinality(
        _balanced_cardinality_frame(), EXPECTED_MODEL_IDS
    )

    assert cardinality == {
        "passed": True,
        "expected_item_count": 577,
        "expected_model_count": 2,
        "expected_model_ids": ["claude-opus-5", "gpt-5.6-sol"],
        "observed_model_ids": ["claude-opus-5", "gpt-5.6-sol"],
        "invalid_item_count": 0,
        "duplicate_item_model_row_count": 0,
    }


def test_annotation_cardinality_rejects_three_plus_one_row_redistribution() -> None:
    normalized = _balanced_cardinality_frame()
    normalized.loc[
        normalized["item_id"].eq("item-001") & normalized["model_id"].eq("gpt-5.6-sol"),
        "item_id",
    ] = "item-000"

    cardinality = _annotation_cardinality(normalized, EXPECTED_MODEL_IDS)

    assert cardinality["passed"] is False
    assert cardinality["invalid_item_count"] == 2
    assert cardinality["duplicate_item_model_row_count"] == 2


def test_annotation_cardinality_rejects_duplicate_or_unexpected_model_ids() -> None:
    duplicate_model = _balanced_cardinality_frame()
    duplicate_model.loc[
        duplicate_model["item_id"].eq("item-000")
        & duplicate_model["model_id"].eq("gpt-5.6-sol"),
        "model_id",
    ] = "claude-opus-5"
    unexpected_model = _balanced_cardinality_frame()
    unexpected_model.loc[
        unexpected_model["item_id"].eq("item-000")
        & unexpected_model["model_id"].eq("gpt-5.6-sol"),
        "model_id",
    ] = "unregistered-model"

    duplicate_cardinality = _annotation_cardinality(duplicate_model, EXPECTED_MODEL_IDS)
    unexpected_cardinality = _annotation_cardinality(
        unexpected_model, EXPECTED_MODEL_IDS
    )

    assert duplicate_cardinality["passed"] is False
    assert duplicate_cardinality["invalid_item_count"] == 1
    assert duplicate_cardinality["duplicate_item_model_row_count"] == 2
    assert unexpected_cardinality["passed"] is False
    assert unexpected_cardinality["invalid_item_count"] == 1
    assert unexpected_cardinality["observed_model_ids"] == [
        "claude-opus-5",
        "gpt-5.6-sol",
        "unregistered-model",
    ]


@pytest.mark.parametrize("operator", sorted(QDMR_OPERATORS))
def test_parser_accepts_each_frozen_qdmr_operator(
    operator: str,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(
        qdmr_step_count=1,
        qdmr_operator_set=[operator],
    )

    parsed = _parse_payload(payload, question_batch_items)

    assert parsed[0][1]["qdmr_operator_set"] == [operator]
    assert parsed[0][1]["qdmr_step_count"] == 1


@pytest.mark.parametrize("level", COGNITIVE_PROCESS_LEVELS)
def test_parser_accepts_each_cognitive_process_level(
    level: str,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(cognitive_process_level=level)

    parsed = _parse_payload(payload, question_batch_items)

    assert parsed[0][1]["cognitive_process_level"] == level


@pytest.mark.parametrize(
    "hop_structure", ["atomic", "bridge", "comparison", "intersection"]
)
def test_parser_accepts_each_frozen_hop_structure(
    hop_structure: str,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(hop_structure=hop_structure)

    parsed = _parse_payload(payload, question_batch_items)

    assert parsed[0][1]["hop_structure"] == hop_structure


def test_parser_accepts_applicable_after_normalisation_with_rewrite(
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(
        qdmr_applicability="applicable_after_normalisation",
        qdmr_normalized_question="Which records match?",
    )

    parsed = _parse_payload(payload, question_batch_items)

    assert parsed[0][1]["qdmr_applicability"] == ("applicable_after_normalisation")
    assert parsed[0][1]["qdmr_normalized_question"] == "Which records match?"


def test_parser_accepts_explicitly_inapplicable_qdmr(
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(
        qdmr_applicability="not_applicable",
        qdmr_step_count=None,
        qdmr_operator_set=None,
        qdmr_normalized_question=None,
        hop_structure=None,
    )

    parsed = _parse_payload(payload, question_batch_items)

    assert parsed[0][1]["qdmr_step_count"] is None
    assert parsed[0][1]["qdmr_operator_set"] is None
    assert parsed[0][1]["hop_structure"] is None


@pytest.mark.parametrize(
    "operator_set",
    [
        pytest.param(["SELECT", "SELECT"], id="duplicate"),
        pytest.param(["SELECT", "FILTER"], id="unsorted"),
        pytest.param(["UNKNOWN"], id="unknown"),
        pytest.param(["select"], id="lowercase"),
        pytest.param([" SELECT"], id="leading-whitespace"),
        pytest.param(["SELECT "], id="trailing-whitespace"),
        pytest.param(["SELECT."], id="punctuation"),
        pytest.param([1], id="non-string"),
        pytest.param("SELECT", id="not-a-list"),
    ],
)
def test_parser_rejects_noncanonical_qdmr_operator_sets(
    operator_set: object,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(qdmr_operator_set=operator_set)

    with pytest.raises(MalformedInputError, match="violates the strict schema"):
        _parse_payload(payload, question_batch_items)


@pytest.mark.parametrize("step_count", [0, -1])
def test_parser_rejects_nonpositive_qdmr_step_count(
    step_count: int,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(qdmr_step_count=step_count)

    with pytest.raises(MalformedInputError, match="violates the strict schema"):
        _parse_payload(payload, question_batch_items)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"qdmr_step_count": None}, id="applicable-without-count"),
        pytest.param({"qdmr_operator_set": None}, id="applicable-without-operators"),
        pytest.param({"qdmr_operator_set": []}, id="applicable-with-empty-operators"),
        pytest.param({"hop_structure": None}, id="applicable-without-hop"),
        pytest.param(
            {"qdmr_normalized_question": "Already interrogative?"},
            id="direct-applicable-with-rewrite",
        ),
        pytest.param(
            {
                "qdmr_applicability": "applicable_after_normalisation",
                "qdmr_normalized_question": None,
            },
            id="normalised-without-rewrite",
        ),
        pytest.param(
            {
                "qdmr_applicability": "not_applicable",
                "qdmr_step_count": None,
                "qdmr_operator_set": [],
                "hop_structure": None,
            },
            id="inapplicable-with-empty-operators",
        ),
        pytest.param(
            {
                "qdmr_applicability": "not_applicable",
                "qdmr_step_count": None,
                "qdmr_operator_set": None,
                "hop_structure": "atomic",
            },
            id="inapplicable-with-hop",
        ),
    ],
)
def test_parser_rejects_inconsistent_qdmr_cross_fields(
    overrides: dict[str, object],
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory(**overrides)

    with pytest.raises(MalformedInputError, match="violates the strict schema"):
        _parse_payload(payload, question_batch_items)


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("", id="empty"),
        pytest.param("not-json", id="malformed-json"),
        pytest.param("```json", id="markdown-fence"),
        pytest.param("Result: {}", id="prose"),
    ],
)
def test_parser_refuses_non_jsonl_responses(
    response: str, question_batch_items: list[dict[str, object]]
) -> None:
    with pytest.raises(MalformedInputError):
        _parse_response_text(response, question_batch_items, "non-jsonl")


def test_parser_rejects_wrong_line_count(
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    line = json.dumps(question_payload_factory())

    with pytest.raises(MalformedInputError, match="incorrect line count"):
        _parse_response_text(f"{line}\n{line}", question_batch_items, "line-count")


@pytest.mark.parametrize(
    "item_id",
    [pytest.param(None, id="missing"), pytest.param("item-999", id="unknown")],
)
def test_parser_rejects_missing_or_unknown_local_item_id(
    item_id: str | None,
    question_payload_factory: QuestionPayloadFactory,
    question_batch_items: list[dict[str, object]],
) -> None:
    payload = question_payload_factory()
    if item_id is None:
        payload.pop("item_id")
    else:
        payload["item_id"] = item_id

    with pytest.raises(MalformedInputError, match="IDs do not match batch"):
        _parse_payload(payload, question_batch_items)


def test_parser_rejects_duplicate_local_item_ids(
    question_payload_factory: QuestionPayloadFactory,
) -> None:
    items: list[dict[str, object]] = [
        {
            "item_id": f"source-question-{index:03d}",
            "item_type": "question",
            "source_text": f"Question {index}?",
            "text_sha256": f"{index:064x}",
        }
        for index in (1, 2)
    ]
    line = json.dumps(question_payload_factory())

    with pytest.raises(MalformedInputError, match="IDs do not match batch"):
        _parse_response_text(f"{line}\n{line}", items, "duplicate")


def test_parser_rejects_partially_valid_batch_without_returning_partial_rows(
    question_payload_factory: QuestionPayloadFactory,
) -> None:
    items: list[dict[str, object]] = [
        {
            "item_id": f"source-question-{index:03d}",
            "item_type": "question",
            "source_text": f"Question {index}?",
            "text_sha256": f"{index:064x}",
        }
        for index in (1, 2)
    ]
    valid = question_payload_factory()
    invalid = question_payload_factory(
        item_id="item-002", qdmr_operator_set=["UNKNOWN"]
    )

    with pytest.raises(MalformedInputError, match="violates the strict schema"):
        _parse_response_text(
            f"{json.dumps(valid)}\n{json.dumps(invalid)}",
            items,
            "partial",
        )


def test_parser_maps_local_ids_by_identity_and_is_deterministic(
    question_payload_factory: QuestionPayloadFactory,
) -> None:
    items: list[dict[str, object]] = [
        {
            "item_id": f"source-question-{index:03d}",
            "item_type": "question",
            "source_text": f"Question {index}?",
            "text_sha256": f"{index:064x}",
        }
        for index in (1, 2)
    ]
    first_payload = question_payload_factory(item_id="item-001")
    second_payload = question_payload_factory(item_id="item-002")
    response = f"{json.dumps(second_payload)}\n{json.dumps(first_payload)}"

    first = _parse_response_text(response, items, "identity")
    second = _parse_response_text(response, items, "identity")

    assert first == second
    assert [row[0]["item_id"] for row in first] == [
        "source-question-002",
        "source-question-001",
    ]
    assert [row[1]["item_id"] for row in first] == ["item-002", "item-001"]


@pytest.mark.parametrize(
    ("left", "right", "agreed", "similarity", "category"),
    [
        (
            ["FILTER", "SELECT"],
            ["FILTER", "SELECT"],
            True,
            1.0,
            "agreement",
        ),
        (
            ["FILTER", "SELECT"],
            ["PROJECT", "SELECT"],
            False,
            1 / 3,
            "set_partial_overlap",
        ),
        (
            ["FILTER"],
            ["SELECT"],
            False,
            0.0,
            "set_disjoint",
        ),
    ],
)
def test_qdmr_operator_set_agreement_uses_set_semantics(
    left: list[str],
    right: list[str],
    agreed: bool,
    similarity: float,
    category: str,
) -> None:
    normalized = pd.DataFrame(
        [
            _normalized_question_row("claude-opus-5", qdmr_operator_set=left),
            _normalized_question_row("gpt-5.6-sol", qdmr_operator_set=right),
        ]
    )

    agreement = _agreement_rows(normalized)
    operator = agreement.loc[agreement["feature"].eq("qdmr_operator_set")].iloc[0]

    assert bool(operator["agreed"]) is agreed
    assert float(operator["similarity"]) == pytest.approx(similarity)
    assert operator["disagreement_category"] == category


def test_qdmr_step_count_agreement_uses_numeric_distance() -> None:
    normalized = pd.DataFrame(
        [
            _normalized_question_row("claude-opus-5", qdmr_step_count=2),
            _normalized_question_row("gpt-5.6-sol", qdmr_step_count=4),
        ]
    )

    agreement = _agreement_rows(normalized)
    step_count = agreement.loc[agreement["feature"].eq("qdmr_step_count")].iloc[0]

    assert bool(step_count["agreed"]) is False
    assert float(step_count["similarity"]) == pytest.approx(1 / 3)
    assert step_count["disagreement_category"] == "numeric_distance"
    assert step_count["first_value"] == "2"
    assert step_count["second_value"] == "4"


def test_annotation_dtype_normalization_uses_nullable_integers() -> None:
    normalized = pd.DataFrame({"qdmr_step_count": [1.0, None, 3.0]})

    result = _normalize_annotation_dtypes(normalized)

    assert result["qdmr_step_count"].dtype == pd.Int64Dtype()
    assert result["qdmr_step_count"].iloc[0] == 1
    assert pd.isna(result["qdmr_step_count"].iloc[1])
    assert result["qdmr_step_count"].iloc[2] == 3
    assert normalized["qdmr_step_count"].dtype == float


def test_annotation_dtype_normalization_rejects_fractional_steps() -> None:
    normalized = pd.DataFrame({"qdmr_step_count": [1.5]})

    with pytest.raises(
        GateFailureError,
        match="cannot be represented as a nullable integer",
    ):
        _normalize_annotation_dtypes(normalized)


def test_qdmr_comparison_fields_are_gated_by_applicability() -> None:
    applicable = _normalized_question_row("claude-opus-5")
    inapplicable = _normalized_question_row(
        "gpt-5.6-sol",
        qdmr_applicability="not_applicable",
        qdmr_step_count=None,
        qdmr_operator_set=None,
        hop_structure=None,
    )

    features = _comparable_features(applicable, inapplicable)

    assert "qdmr_applicability" in features
    assert "qdmr_step_count" not in features
    assert "qdmr_operator_set" not in features
    assert "hop_structure" not in features


def test_normalized_question_is_compared_when_either_model_rewrites() -> None:
    direct = _normalized_question_row("claude-opus-5", qdmr_normalized_question=None)
    normalized = _normalized_question_row(
        "gpt-5.6-sol",
        qdmr_applicability="applicable_after_normalisation",
        qdmr_normalized_question="Which records match?",
    )

    features = _comparable_features(direct, normalized)

    assert features["qdmr_normalized_question"] == (
        None,
        "Which records match?",
    )


def test_exhaustivity_is_nominal_and_cognitive_process_is_ordinal() -> None:
    normalized = pd.DataFrame(
        [
            _normalized_question_row(
                "claude-opus-5",
                exhaustivity_requirement="mention_some",
                cognitive_process_level="remember",
            ),
            _normalized_question_row(
                "gpt-5.6-sol",
                exhaustivity_requirement="mention_all",
                cognitive_process_level="analyze",
            ),
        ]
    )

    agreement = _agreement_rows(normalized).set_index("feature")

    assert agreement.loc["exhaustivity_requirement", "similarity"] == 0.0
    assert (
        agreement.loc["exhaustivity_requirement", "disagreement_category"]
        == "different_value"
    )
    assert agreement.loc["cognitive_process_level", "similarity"] == pytest.approx(0.4)
    assert (
        agreement.loc["cognitive_process_level", "disagreement_category"]
        == "ordinal_distance_3"
    )


def test_qdmr_construct_anomaly_is_deterministic_and_attributable() -> None:
    normalized = pd.DataFrame(
        [
            _normalized_question_row(
                "gpt-5.6-sol",
                item_id="question-b",
                qdmr_step_count=1,
                qdmr_operator_set=["FILTER", "SELECT"],
            ),
            _normalized_question_row(
                "claude-opus-5",
                item_id="question-a",
                qdmr_step_count=1,
                qdmr_operator_set=["FILTER", "PROJECT", "SELECT"],
            ),
            _normalized_question_row(
                "gpt-5.6-sol",
                item_id="question-c",
                qdmr_step_count=2,
                qdmr_operator_set=["FILTER", "SELECT"],
            ),
            _normalized_question_row(
                "claude-opus-5",
                item_id="question-d",
                qdmr_step_count=3,
                qdmr_operator_set=["FILTER", "SELECT"],
            ),
            _normalized_question_row(
                "gpt-5.6-sol",
                item_id="question-e",
                qdmr_applicability="not_applicable",
                qdmr_step_count=None,
                qdmr_operator_set=None,
                hop_structure=None,
            ),
        ]
    )

    first = _qdmr_construct_anomalies(normalized)
    second = _qdmr_construct_anomalies(normalized)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(ANNOTATION_CONSTRUCT_ANOMALY_COLUMNS)
    assert first["item_id"].tolist() == ["question-a", "question-b"]
    assert first["qdmr_step_count"].tolist() == [1, 1]
    assert first["qdmr_operator_count"].tolist() == [3, 2]
    assert (
        first["anomaly_code"]
        .eq(AnnotationConstructAnomaly.QDMR_OPERATOR_COUNT_EXCEEDS_STEP_COUNT)
        .all()
    )
    assert first["requires_validation"].all()
    assert {"source_text", "grade", "rubric_v2"}.isdisjoint(first.columns)
    assert first["source_text_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert first["raw_response_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_qdmr_construct_anomaly_empty_table_has_stable_schema() -> None:
    normalized = pd.DataFrame(
        [
            _normalized_question_row("claude-opus-5"),
            _normalized_question_row("gpt-5.6-sol"),
        ]
    )

    anomalies = _qdmr_construct_anomalies(normalized)

    assert anomalies.empty
    assert list(anomalies.columns) == list(ANNOTATION_CONSTRUCT_ANOMALY_COLUMNS)


def test_qdmr_construct_anomaly_rejects_invalid_applicable_values() -> None:
    normalized = pd.DataFrame(
        [_normalized_question_row("claude-opus-5", qdmr_operator_set=None)]
    )

    with pytest.raises(
        GateFailureError, match="Applicable QDMR construct values are invalid"
    ):
        _qdmr_construct_anomalies(normalized)


def test_annotation_verification_persists_zero_applicability_levels() -> None:
    normalized = pd.DataFrame(
        [
            _normalized_question_row("claude-opus-5"),
            _normalized_question_row("gpt-5.6-sol"),
        ]
    )
    agreement = _agreement_rows(normalized)
    anomalies = _qdmr_construct_anomalies(normalized)

    verification = _annotation_verification(normalized, agreement, anomalies)
    report = _render_report(verification)

    assert verification["schema_version"] == 3
    assert verification["qdmr_applicability_counts"] == {
        "applicable": 2,
        "applicable_after_normalisation": 0,
        "not_applicable": 0,
    }
    assert verification["qdmr_applicability_counts_by_model"] == {
        "claude-opus-5": {
            "applicable": 1,
            "applicable_after_normalisation": 0,
            "not_applicable": 0,
        },
        "gpt-5.6-sol": {
            "applicable": 1,
            "applicable_after_normalisation": 0,
            "not_applicable": 0,
        },
    }
    assert verification["construct_anomaly_count"] == 0
    assert verification["construct_anomaly_counts"] == {}
    assert verification["construct_anomalies_block_ingest"] is False
    assert verification["qdmr_operator_counts"] == {
        operator: 2 if operator in {"FILTER", "SELECT"} else 0
        for operator in QDMR_OPERATOR_INVENTORY
    }
    assert verification["cognitive_process_level_counts"] == {
        level: 2 if level == "remember" else 0 for level in COGNITIVE_PROCESS_LEVELS
    }
    assert "not_applicable=0" in report
    assert '["PROJECT", "AGGREGATE"' in report
    assert '["understand", "apply", "analyze", "evaluate", "create"]' in report
    assert "Krippendorff" not in report
    assert "VALIDATED" not in report


def test_closed_inventory_count_rejects_non_list_operator_values() -> None:
    normalized = pd.DataFrame(
        [_normalized_question_row("claude-opus-5", qdmr_operator_set="FILTER")]
    )

    with pytest.raises(
        GateFailureError, match="cannot be counted as a closed inventory"
    ):
        _closed_inventory_counts(normalized)
