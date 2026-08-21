"""Integration tests for the frozen Simple Mode join."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from es_index_explorer.question_analysis.contracts import WorkflowCommand
from es_index_explorer.question_analysis.errors import GateFailureError
from es_index_explorer.question_analysis.join import (
    DEFAULT_RUBRIC_ROOT,
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_TASK_PATH,
    JoinConfig,
    JoinResult,
    _require_unique,
    _versioned,
    build_join,
)
from es_index_explorer.question_analysis.storage import ArtifactStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DEFAULT_SNAPSHOT_DIR.is_dir()
        or not DEFAULT_RUBRIC_ROOT.is_dir()
        or not DEFAULT_TASK_PATH.is_file(),
        reason="Frozen snapshot and sibling r1-evals checkout are required",
    ),
]


@pytest.fixture(scope="module")
def result() -> JoinResult:
    return build_join()


def test_full_join_passes_every_structural_gate(result: JoinResult) -> None:
    verification = result.structural_verification

    assert verification["passed"]
    assert verification["blocking_failures"] == []
    assert verification["derived_totals"] == {
        "rubrics": 63,
        "variants": 263,
        "expectations": 314,
        "traces": 7364,
        "expected_criterion_observations": 36624,
        "actual_eligible_criterion_observations": 36624,
    }
    coverage = cast(dict[str, int], verification["coverage"])
    integrity = cast(dict[str, int | float], verification["integrity"])
    materiality = cast(dict[str, int | bool], verification["materiality"])
    variant_range = cast(dict[str, int | bool], verification["variant_range"])
    assert coverage["arm_balance_failure_count"] == 0
    assert coverage["warning_discrepancy_count"] == 7
    assert integrity["rubric_v2_mismatch_count"] == 0
    assert integrity["grade_mismatch_count"] == 0
    assert variant_range == {
        "min_V_r": 2,
        "max_V_r": 7,
        "min_V_r_at_least_2": True,
        "V_r_equals_1_rubric_count": 0,
        "blocking": False,
    }
    assert materiality == {
        "expectation_count": 314,
        "material_expectation_count": 314,
        "non_material_expectation_count": 0,
        "rubric_mismatch_count": 0,
        "passed": True,
    }
    assert "Designed variants per rubric: 2–7." in result.report
    assert "Non-blocking join warnings: 7" in result.report


def test_expected_join_warnings_remain_non_blocking(result: JoinResult) -> None:
    assert len(result.discrepancies) == 7
    assert result.discrepancies["severity"].eq("warning").all()
    assert result.discrepancies["discrepancy_type"].value_counts().to_dict() == {
        "expectation_name_outer_whitespace_normalized": 3,
        "assessment_expectation_name_outer_whitespace_normalized": 2,
        "duplicate_expectation_name_resolved_by_occurrence": 2,
    }


def test_nonfirst_variants_and_duplicate_expectations_survive(
    result: JoinResult,
) -> None:
    assert int((result.catalogue.variants["variant_index"] > 0).sum()) == 200

    rubric = result.catalogue.rubrics[
        result.catalogue.rubrics["rubric_file_name"].eq(
            "21_qna_anomaly_detection_unusual_email_patterns.rubric.toml"
        )
    ].iloc[0]
    expectations = result.catalogue.expectations[
        result.catalogue.expectations["rubric_id"].eq(rubric["rubric_id"])
        & result.catalogue.expectations["expectation_name"].eq(
            "cite_routine_maintenance_email"
        )
    ]
    criteria = result.criterion_table[
        result.criterion_table["rubric_id"].eq(rubric["rubric_id"])
        & result.criterion_table["expectation_name"].eq(
            "cite_routine_maintenance_email"
        )
    ]

    assert expectations["expectation_id"].nunique() == 2
    assert expectations["expectation_name_occurrence_index"].tolist() == [0, 1]
    assert len(criteria) == 84 * 2
    assert criteria.groupby("trace_id")["expectation_id"].nunique().eq(2).all()


def test_join_keys_and_order_are_deterministic(result: JoinResult) -> None:
    repeated = build_join()

    pd.testing.assert_frame_equal(result.catalogue.rubrics, repeated.catalogue.rubrics)
    pd.testing.assert_frame_equal(result.catalogue.variants, repeated.catalogue.variants)
    pd.testing.assert_frame_equal(
        result.catalogue.expectations, repeated.catalogue.expectations
    )
    pd.testing.assert_frame_equal(result.criterion_table, repeated.criterion_table)
    pd.testing.assert_frame_equal(result.trace_pfu_table, repeated.trace_pfu_table)
    pd.testing.assert_frame_equal(result.discrepancies, repeated.discrepancies)
    assert result.structural_verification == repeated.structural_verification
    assert result.f3_design_rank == repeated.f3_design_rank
    assert result.report == repeated.report
    assert result.criterion_table["criterion_observation_id"].is_unique
    assert result.trace_pfu_table[["run_id", "trace_id"]].duplicated().sum() == 0


def test_segment2_parquet_hashes_are_stable_in_same_environment(
    result: JoinResult, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path)
    frames = {
        "rubric_catalogue.parquet": result.catalogue.rubrics,
        "variant_catalogue.parquet": result.catalogue.variants,
        "expectation_catalogue.parquet": result.catalogue.expectations,
        "criterion_table.parquet": result.criterion_table,
        "trace_pfu_table.parquet": result.trace_pfu_table,
        "join_discrepancies.parquet": result.discrepancies,
    }

    for name, frame in frames.items():
        first = store.write_parquet(
            f"first/{name}",
            _versioned(frame),
            created_by=WorkflowCommand.JOIN,
        )
        second = store.write_parquet(
            f"second/{name}",
            _versioned(frame),
            created_by=WorkflowCommand.JOIN,
        )

        assert first.sha256 == second.sha256
        assert first.size_bytes == second.size_bytes


def test_duplicate_join_keys_fail_closed() -> None:
    frame = pd.DataFrame([{"run_id": "run"}, {"run_id": "run"}])

    with pytest.raises(GateFailureError, match="duplicate keys"):
        _require_unique(frame, ["run_id"], "synthetic")


def test_missing_criterion_is_a_blocking_gate(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for path in DEFAULT_SNAPSHOT_DIR.glob("*.parquet"):
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(DEFAULT_SNAPSHOT_DIR / "manifest.json", snapshot / "manifest.json")

    criteria_path = snapshot / "trace_criteria.parquet"
    criteria = pd.read_parquet(criteria_path)
    criteria.iloc[1:].to_parquet(criteria_path, index=False)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["files"] if item["name"] == criteria_path.name)
    row["size_bytes"] = criteria_path.stat().st_size
    row["sha256"] = hashlib.sha256(criteria_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    corrupted = build_join(JoinConfig(snapshot_dir=snapshot))

    assert not corrupted.structural_verification["passed"]
    blocking_failures = cast(
        list[str], corrupted.structural_verification["blocking_failures"]
    )
    assert "criterion_observation_count" in blocking_failures
    assert "criterion_count_mismatch" in set(
        corrupted.discrepancies["discrepancy_type"]
    )
