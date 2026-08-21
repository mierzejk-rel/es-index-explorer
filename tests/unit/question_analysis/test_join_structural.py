"""Unit tests for join-time structural decisions."""

import pandas as pd
import pytest

from es_index_explorer.question_analysis.catalogue import Catalogue
from es_index_explorer.question_analysis.errors import GateFailureError
from es_index_explorer.question_analysis.join import (
    _arm_balance_failure_count,
    _attach_catalogue_source_paths,
    _f3_design_rank,
    _materiality_verification,
)

pytestmark = pytest.mark.unit


def test_snapshot_filename_maps_to_canonical_source_path() -> None:
    catalogue = _catalogue_with_rubrics(
        [
            {
                "eval_dataset": "emc2_set1",
                "rubric_file_name": "001.rubric.toml",
                "source_path": "air_assist/EMC2/uat/set_1/001.rubric.toml",
            }
        ]
    )
    trace = pd.DataFrame(
        [{"eval_dataset": "emc2_set1", "rubric_file_path": "001.rubric.toml"}]
    )

    result = _attach_catalogue_source_paths(trace, catalogue)

    assert result["source_path"].tolist() == [
        "air_assist/EMC2/uat/set_1/001.rubric.toml"
    ]


def test_duplicate_catalogue_filename_mapping_fails_closed() -> None:
    catalogue = _catalogue_with_rubrics(
        [
            {
                "eval_dataset": "emc2_set1",
                "rubric_file_name": "001.rubric.toml",
                "source_path": "first/001.rubric.toml",
            },
            {
                "eval_dataset": "emc2_set1",
                "rubric_file_name": "001.rubric.toml",
                "source_path": "second/001.rubric.toml",
            },
        ]
    )
    trace = pd.DataFrame(
        [{"eval_dataset": "emc2_set1", "rubric_file_path": "001.rubric.toml"}]
    )

    with pytest.raises(GateFailureError, match="duplicate keys"):
        _attach_catalogue_source_paths(trace, catalogue)


def test_missing_catalogue_filename_mapping_fails_closed() -> None:
    catalogue = _catalogue_with_rubrics(
        [
            {
                "eval_dataset": "emc2_set1",
                "rubric_file_name": "001.rubric.toml",
                "source_path": "air_assist/EMC2/uat/set_1/001.rubric.toml",
            }
        ]
    )
    trace = pd.DataFrame(
        [{"eval_dataset": "emc2_set1", "rubric_file_path": "missing.rubric.toml"}]
    )

    with pytest.raises(GateFailureError, match="without a unique catalogue source path"):
        _attach_catalogue_source_paths(trace, catalogue)


def test_arm_balance_requires_28_rows_and_28_distinct_arms() -> None:
    balanced = pd.DataFrame(
        {
            "rubric_id": ["rubric"] * 28,
            "variant_index": [0] * 28,
            "arm_id": [f"arm-{index}" for index in range(28)],
            "eligible": [True] * 28,
        }
    )
    duplicate_arm = pd.concat([balanced, balanced.iloc[[0]]], ignore_index=True)

    assert _arm_balance_failure_count(balanced) == 0
    assert _arm_balance_failure_count(duplicate_arm) == 1


def test_materiality_verification_requires_every_expectation_to_be_material() -> None:
    all_material = _material_catalogue([True, True])
    mixed_materiality = _material_catalogue([True, False])

    assert _materiality_verification(all_material) == {
        "expectation_count": 2,
        "material_expectation_count": 2,
        "non_material_expectation_count": 0,
        "rubric_mismatch_count": 0,
        "passed": True,
    }
    assert _materiality_verification(mixed_materiality) == {
        "expectation_count": 2,
        "material_expectation_count": 1,
        "non_material_expectation_count": 1,
        "rubric_mismatch_count": 1,
        "passed": False,
    }


def test_f3_interaction_remains_confirmatory_when_only_c_is_constant() -> None:
    result = _f3_design_rank(_f3_fixture(contexts=(10, 20, 30)))

    assert result["rank"] == 4
    assert result["column_count"] == 5
    assert not result["full_rank"]
    assert result["rank_without_interaction"] == 3
    assert result["column_count_without_interaction"] == 4
    assert result["interaction_identified"]
    assert result["interaction_status"] == "confirmatory"


def test_f3_interaction_is_exploratory_when_it_adds_no_rank() -> None:
    result = _f3_design_rank(_f3_fixture(contexts=(10, 10, 10)))

    assert not result["interaction_identified"]
    assert result["interaction_status"] == "exploratory"


def _f3_fixture(*, contexts: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rubric_index, expectation_count in enumerate((1, 2, 3)):
        for context_index, global_context in enumerate(contexts):
            for expectation_index in range(expectation_count):
                rows.append(
                    {
                        "eligible": True,
                        "stage": "A",
                        "retrieval_family": "hybrid",
                        "rubric_id": f"rubric-{rubric_index}",
                        "expectation_id": (
                            f"rubric-{rubric_index}-expectation-{expectation_index}"
                        ),
                        "global_context": global_context,
                        "calls": 1,
                        "arm_id": f"arm-{context_index}",
                        "criterion_observation_id": (
                            f"rubric-{rubric_index}-context-{context_index}-"
                            f"expectation-{expectation_index}"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _catalogue_with_rubrics(rows: list[dict[str, str]]) -> Catalogue:
    return Catalogue(
        rubrics=pd.DataFrame(rows),
        variants=pd.DataFrame(),
        expectations=pd.DataFrame(),
        input_paths=(),
    )


def _material_catalogue(material: list[bool]) -> Catalogue:
    material_count = sum(material)
    return Catalogue(
        rubrics=pd.DataFrame(
            [
                {
                    "expectation_count": len(material),
                    "material_expectation_count": material_count,
                }
            ]
        ),
        variants=pd.DataFrame(),
        expectations=pd.DataFrame({"material": material}),
        input_paths=(),
    )
