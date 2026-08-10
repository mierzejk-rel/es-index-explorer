"""Unit tests for provenance-checked rubric analysis."""

import json
from pathlib import Path

import pandas as pd
import pytest

from es_index_explorer.mlflow_analysis.rubric_analysis import (
    _join_invocations,
    analyze_rubrics,
)

pytestmark = pytest.mark.unit


def test_analyze_rubrics_joins_multilabel_catalogue_without_legacy_weight(
    tmp_path: Path,
) -> None:
    """Join trace results to TOML questions, expectations, and multi-label use cases."""
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_count": 1,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    rubric_root = tmp_path / "rubric_data"
    rubric_file = rubric_root / "air_assist" / "test.rubric.toml"
    rubric_file.parent.mkdir(parents=True)
    rubric_file.write_text(
        """
[[expectations]]
name = "expects_fact"
weight = 9
category = "deprecated"
description = "States the relevant fact."

[meta]
dataset_id = "emc2_set1"
task_name = "air_assist"
use_case = ["communications_analysis", "entity_and_relationship_analysis"]

[input]
[[input.messages]]
role = "user"
content = "What happened?"
""".strip(),
        encoding="utf-8",
    )
    task_path = tmp_path / "air_assist.toml"
    task_path.write_text(
        """
[specification]
[[specification.use_cases]]
name = "communications_analysis"
description = "Communication analysis."

[[specification.use_cases]]
name = "entity_and_relationship_analysis"
description = "Entity analysis."
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "experiment_id": "experiment-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "trace_status": "OK",
                "rubric_key": "invoke_communications_analysis_0",
                "question": "What happened?",
                "use_case": "communications_analysis",
                "dataset_id": "emc2_set1",
                "evalset_variant": None,
                "row_id": None,
                "rubric_index": 0,
                "variant_index": 0,
                "rubric_file_path": "/old/rubric_data/air_assist/test.rubric.toml",
                "criteria_parse_status": "parsed",
            }
        ]
    ).to_parquet(snapshot_dir / "trace_invocations.parquet", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "experiment-1",
                "run_id": "run-1",
                "rubric_index": 0,
                "variant_index": 0,
                "rubric_name": "test.rubric.toml",
                "rubric_file_path": "/old/rubric_data/air_assist/test.rubric.toml",
                "dataset_id": "emc2_set1",
                "use_case": json.dumps(["communications_analysis"]),
                "question": "What happened?",
                "input_variant_count": 1,
                "author": "author",
                "artifact_sha256": "hash",
                "artifact_status": "ok",
            }
        ]
    ).to_parquet(snapshot_dir / "run_rubrics.parquet", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "experiment-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "rubric_file_path": "/old/rubric_data/air_assist/test.rubric.toml",
                "expectation_name": "expects_fact",
                "material": True,
                "state": "PASS",
            }
        ]
    ).to_parquet(snapshot_dir / "trace_criteria.parquet", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "experiment-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "trace_status": "OK",
                "execution_time_ms": 10,
                "use_case": "communications_analysis",
                "dataset_id": "emc2_set1",
                "evalset_variant": None,
                "row_id": None,
                "assessment_name": "RubricV2",
                "assessment_value_type": "float",
                "assessment_value_string": None,
                "assessment_value_bool": None,
                "assessment_value_int": None,
                "assessment_value_float": 1.0,
                "ordinal_grade": "Good",
                "detected_error_modes": "[]",
            }
        ]
    ).to_parquet(snapshot_dir / "trace_quality.parquet", index=False)

    analysis_dir = analyze_rubrics(
        snapshot_dir=snapshot_dir,
        rubric_root=rubric_root,
        task_path=task_path,
    )

    expectations = pd.read_csv(analysis_dir / "rubric_expectations.csv")
    use_cases = pd.read_csv(analysis_dir / "use_case_summary.csv")
    criteria = pd.read_csv(analysis_dir / "expectation_summary.csv")
    assert "weight" not in expectations.columns
    assert "category" not in expectations.columns
    assert expectations.loc[0, "material"]
    assert set(use_cases["use_case_label"]) == {
        "Communications Analysis",
        "Entity And Relationship Analysis",
    }
    assert criteria.loc[0, "state"] == "PASS"


def test_direct_rubric_identity_remains_eligible_without_run_artifact() -> None:
    """Direct RubricV2 file identity is primary; artifact absence is auditable."""
    invocations = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "trace_id": "trace-1",
                "rubric_key": "unexpected_root_name",
                "question": "What happened?",
                "dataset_id": "emc2_set1",
                "rubric_file_path": "/old/rubric_data/air_assist/test.rubric.toml",
                "rubric_index": None,
                "variant_index": None,
            }
        ]
    )
    catalogue = pd.DataFrame(
        [
            {
                "catalogue_id": "catalogue-1",
                "source_path": "air_assist/test.rubric.toml",
                "question": "What happened?",
                "dataset_id": "emc2_set1",
                "use_cases": '["communications_analysis"]',
                "use_case_descriptions": "{}",
            }
        ]
    )
    run_rubrics = pd.DataFrame(
        columns=[
            "run_id",
            "rubric_index",
            "variant_index",
            "artifact_status",
        ]
    )

    joins, discrepancies = _join_invocations(
        pandas=pd,
        invocations=invocations,
        run_rubrics=run_rubrics,
        catalogue=catalogue,
    )

    assert joins.loc[0, "join_status"] == "eligible"
    assert joins.loc[0, "join_method"] == "direct_rubric_v2_file_path"
    assert set(discrepancies["discrepancy_type"]) == {
        "missing_run_rubric_artifact",
        "unparsable_span_identity",
    }
