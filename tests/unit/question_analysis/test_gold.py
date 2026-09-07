"""Tests for deterministic strict-partition gold sampling."""

import csv
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.gold import (
    DISPUTED,
    _parse_human_value,
    build_gold_sample,
)

pytestmark = pytest.mark.unit


def _sampling_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    agreement_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for index in range(16):
        item_id = f"variant-{index:03d}"
        first = '"precision_oriented"'
        second = first if index < 13 else '"recall_oriented"'
        agreement_rows.append(
            {
                "item_id": item_id,
                "item_type": "question",
                "feature": "recall_orientation",
                "first_model_id": "claude-opus-5",
                "second_model_id": "gpt-5.6-sol",
                "first_value": first,
                "second_value": second,
                "agreed": first == second,
            }
        )
        feature_rows.append(
            {
                "item_id": item_id,
                "item_type": "question",
                "source_text": f"Question {index}?",
                "text_sha256": f"{index:064x}",
            }
        )
    return pd.DataFrame(agreement_rows), pd.DataFrame(feature_rows)


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(payload.decode("utf-8"))))


def test_gold_sample_is_strict_deterministic_and_has_exact_probabilities() -> None:
    agreement, features = _sampling_frames()

    first = build_gold_sample(
        agreement,
        features,
        sampling_rng=np.random.default_rng(123),
        recode_rng=np.random.default_rng(456),
    )
    second = build_gold_sample(
        agreement,
        features,
        sampling_rng=np.random.default_rng(123),
        recode_rng=np.random.default_rng(456),
    )

    assert len(first.sample) == 15
    assert first.sample["item_id"].is_unique
    assert first.sample["stratum_id"].nunique() == 2
    assert set(first.sample["stratum_level"]) == {'"precision_oriented"', DISPUTED}
    assert sorted(first.sample["inclusion_probability"].unique()) == [
        pytest.approx(12 / 13),
        1.0,
    ]
    pd.testing.assert_frame_equal(first.sample, second.sample)
    assert first.adjudication_csv == second.adjudication_csv
    assert first.recode_csv == second.recode_csv
    assert first.manifest == second.manifest


def test_recode_bundle_is_blind_and_uses_twenty_percent_minimum_two() -> None:
    agreement, features = _sampling_frames()

    result = build_gold_sample(
        agreement,
        features,
        sampling_rng=np.random.default_rng(123),
        recode_rng=np.random.default_rng(456),
    )
    recode_rows = _csv_rows(result.recode_csv)
    adjudication_rows = _csv_rows(result.adjudication_csv)

    assert len(recode_rows) == 5
    assert len(adjudication_rows) == 15
    assert set(recode_rows[0]) == {
        "recode_item_id",
        "item_type",
        "source_text",
        "feature",
        "recode_value_json",
        "comment",
    }
    assert not {
        "first_model_id",
        "first_value_json",
        "second_model_id",
        "second_value_json",
        "human_value_json",
        "stratum_id",
    }.intersection(recode_rows[0])
    assert result.manifest["outcome_fields_included"] == []


def test_gold_sampling_rejects_missing_schema_duplicate_sources_and_no_strata() -> None:
    agreement, features = _sampling_frames()

    with pytest.raises(MalformedInputError, match="missing columns"):
        build_gold_sample(agreement.drop(columns="agreed"), features)

    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(GateFailureError, match="duplicate item IDs"):
        build_gold_sample(agreement, duplicated)

    with pytest.raises(GateFailureError, match="No eligible"):
        build_gold_sample(
            agreement.assign(feature="presupposition_load"),
            features,
        )


@pytest.mark.parametrize(
    ("feature", "value", "message"),
    (
        ("recall_orientation", "", "missing"),
        ("recall_orientation", "not-json", "strict JSON"),
        ("recall_orientation", '"unknown"', "Unknown human label"),
        ("qdmr_step_count", "0", "positive integer"),
        ("qdmr_operator_set", '["SELECT", "SELECT"]', "sorted unique"),
        ("qdmr_normalized_question", '""', "non-empty string"),
    ),
)
def test_human_value_contract_rejects_invalid_values(
    feature: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(MalformedInputError, match=message):
        _parse_human_value(feature, value)
