"""Tests for the all-input catalogue builder."""

from pathlib import Path

import pytest

from es_index_explorer.question_analysis.catalogue import build_catalogue

pytestmark = pytest.mark.unit

TASK = """
[[specification.use_cases]]
name = "communications_analysis"
description = "Analyze communications."

[[specification.use_cases]]
name = "incident_event_analysis"
description = "Analyze incidents and events."
"""

RUBRIC = """
[[input]]
[[input.messages]]
role = "user"
content = "Question zero?"

[[input]]
[[input.messages]]
role = "user"
content = "Question one?"

[[input]]
[[input.messages]]
role = "user"
content = "Question two?"

[[expectations]]
name = "duplicate"
description = "First meaning."
material = true
document_ids = ["A", "A", "B"]

[[expectations]]
name = "duplicate"
description = "Second meaning."
material = true

[meta]
author = "author@example.com"
date = "2026-01-01"
dataset_id = "EMC-2"
task_name = "air_assist"
version = "1"
schema_version = "7"
use_case = [
    "communications_analysis",
    "incident_event_analysis",
    "communications_analysis",
]
"""


def test_catalogue_enumerates_every_variant_and_expectation(tmp_path: Path) -> None:
    rubric_root = tmp_path / "rubric_data"
    rubric_dir = rubric_root / "air_assist" / "EMC2" / "uat" / "set_1"
    rubric_dir.mkdir(parents=True)
    (rubric_dir / "001.rubric.toml").write_text(RUBRIC, encoding="utf-8")
    _populate_other_dataset_roots(rubric_root)
    task = tmp_path / "task.toml"
    task.write_text(TASK, encoding="utf-8")

    catalogue = build_catalogue(rubric_root, task)
    target = catalogue.variants[catalogue.variants["eval_dataset"].eq("emc2_set1")]
    expectations = catalogue.expectations[
        catalogue.expectations["eval_dataset"].eq("emc2_set1")
    ]

    assert target["variant_index"].tolist() == [0, 1, 2]
    assert target["question"].tolist() == [
        "Question zero?",
        "Question one?",
        "Question two?",
    ]
    assert target["variant_id"].nunique() == 3
    assert target["use_cases"].tolist() == [
        ["communications_analysis", "incident_event_analysis"],
        ["communications_analysis", "incident_event_analysis"],
        ["communications_analysis", "incident_event_analysis"],
    ]
    assert expectations["expectation_id"].nunique() == 2
    assert expectations["expectation_name_occurrence_index"].tolist() == [0, 1]
    assert expectations["expectation_document_count"].tolist() == [2, 0]
    assert expectations.iloc[0]["document_ids"] == ["A", "B"]


def test_catalogue_preserves_variants_with_identical_questions(tmp_path: Path) -> None:
    rubric_root = tmp_path / "rubric_data"
    rubric_dir = rubric_root / "air_assist" / "EMC2" / "uat" / "set_1"
    rubric_dir.mkdir(parents=True)
    rubric = RUBRIC.replace("Question one?", "Question zero?")
    (rubric_dir / "001.rubric.toml").write_text(rubric, encoding="utf-8")
    _populate_other_dataset_roots(rubric_root)
    task = tmp_path / "task.toml"
    task.write_text(TASK, encoding="utf-8")

    catalogue = build_catalogue(rubric_root, task)
    variants = catalogue.variants[catalogue.variants["eval_dataset"].eq("emc2_set1")]

    assert variants["variant_index"].tolist() == [0, 1, 2]
    assert variants["question"].tolist() == [
        "Question zero?",
        "Question zero?",
        "Question two?",
    ]
    assert variants["variant_id"].nunique() == 3


def test_catalogue_order_is_independent_of_file_creation_order(tmp_path: Path) -> None:
    rubric_root = tmp_path / "rubric_data"
    rubric_dir = rubric_root / "air_assist" / "EMC2" / "uat" / "set_1"
    rubric_dir.mkdir(parents=True)
    (rubric_dir / "002.rubric.toml").write_text(RUBRIC, encoding="utf-8")
    (rubric_dir / "001.rubric.toml").write_text(RUBRIC, encoding="utf-8")
    _populate_other_dataset_roots(rubric_root)
    task = tmp_path / "task.toml"
    task.write_text(TASK, encoding="utf-8")

    catalogue = build_catalogue(rubric_root, task)
    set_1 = catalogue.rubrics[catalogue.rubrics["eval_dataset"].eq("emc2_set1")]

    assert set_1["rubric_file_name"].tolist() == [
        "001.rubric.toml",
        "002.rubric.toml",
    ]


def _populate_other_dataset_roots(rubric_root: Path) -> None:
    for relative in (
        Path("air_assist/EMC2/uat/set_2"),
        Path("air_assist/mallinckrodt/rubrics_for_ga"),
    ):
        directory = rubric_root / relative
        directory.mkdir(parents=True)
        (directory / "001.rubric.toml").write_text(RUBRIC, encoding="utf-8")
