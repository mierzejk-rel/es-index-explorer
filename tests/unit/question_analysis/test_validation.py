"""Tests for Horvitz-Thompson feature-validation metrics."""

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from es_index_explorer.question_analysis import validation as validation_module
from es_index_explorer.question_analysis.contracts import FeatureValidationStatus
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.storage import canonical_json_bytes
from es_index_explorer.question_analysis.validation import (
    FeatureDecision,
    balanced_accuracy,
    build_feature_validation,
    emit_next_feature_dossier,
    ht_confusion_totals,
    krippendorff_alpha,
    macro_f1,
    mae_skill,
    quadratically_weighted_kappa,
    run_validate_features,
    stratified_bootstrap_interval,
    weighted_median,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

pytestmark = pytest.mark.unit


def test_ht_confusion_and_binary_metrics_use_inverse_probabilities() -> None:
    matrix = ht_confusion_totals(
        gold=[False, False, True, True],
        predicted=[False, True, True, True],
        inclusion_probabilities=[0.5, 0.5, 0.25, 0.25],
        levels=[False, True],
    )

    np.testing.assert_array_equal(matrix, np.array([[2.0, 2.0], [0.0, 8.0]]))
    assert balanced_accuracy(matrix) == pytest.approx(0.75)
    assert macro_f1(matrix) == pytest.approx((2 / 3 + 8 / 9) / 2)


def test_quadratic_kappa_and_krippendorff_alpha_respect_distance() -> None:
    matrix = np.diag([3.0, 4.0, 5.0])

    assert quadratically_weighted_kappa(matrix) == pytest.approx(1.0)
    assert krippendorff_alpha(
        [["low", "low"], ["middle", "middle"], ["high", "high"]],
        level="ordinal",
        ordinal_levels=["low", "middle", "high"],
    ) == pytest.approx(1.0)
    assert krippendorff_alpha(
        [[1, 1], [2, 2], [3, 3]],
        level="interval",
    ) == pytest.approx(1.0)


def test_weighted_median_and_mae_skill_use_matched_naive_predictor() -> None:
    assert weighted_median([1, 2, 5], [1, 4, 1]) == 2

    perfect, naive_mae = mae_skill(
        [1, 2, 5],
        [1, 2, 5],
        [1.0, 0.25, 1.0],
    )
    naive, _ = mae_skill(
        [1, 2, 5],
        [2, 2, 2],
        [1.0, 0.25, 1.0],
    )

    assert perfect == pytest.approx(1.0)
    assert naive == pytest.approx(0.0)
    assert naive_mae == pytest.approx(4 / 6)


def test_stratified_bootstrap_is_deterministic_and_preserves_stratum_sizes() -> None:
    frame = pd.DataFrame(
        {
            "stratum_id": ["a", "a", "b", "b", "b"],
            "value": [1.0, 2.0, 4.0, 5.0, 6.0],
        }
    )

    first = stratified_bootstrap_interval(
        frame,
        lambda sample: float(sample["value"].mean()),
        rng=np.random.default_rng(123),
        replicates=100,
    )
    second = stratified_bootstrap_interval(
        frame,
        lambda sample: float(sample["value"].mean()),
        rng=np.random.default_rng(123),
        replicates=100,
    )

    assert first == second
    assert first[0] <= first[1]


def test_metric_failure_and_degenerate_paths_are_explicit() -> None:
    with pytest.raises(MalformedInputError, match="non-empty and aligned"):
        ht_confusion_totals([], [], [], [False, True])
    with pytest.raises(MalformedInputError, match="Inclusion probabilities"):
        ht_confusion_totals([False], [False], [0.0], [False, True])
    with pytest.raises(MalformedInputError, match="2-by-2"):
        balanced_accuracy(np.zeros((3, 3)))
    assert np.isnan(balanced_accuracy(np.array([[0.0, 0.0], [0.0, 1.0]])))
    with pytest.raises(MalformedInputError, match="square"):
        macro_f1(np.zeros((2, 3)))
    assert macro_f1(np.zeros((2, 2))) == 0.0
    with pytest.raises(MalformedInputError, match="square"):
        quadratically_weighted_kappa(np.zeros((2, 3)))
    assert np.isnan(quadratically_weighted_kappa(np.zeros((1, 1))))
    assert np.isnan(quadratically_weighted_kappa(np.array([[1.0, 0.0], [0.0, 0.0]])))
    with pytest.raises(MalformedInputError, match="non-empty and aligned"):
        weighted_median([], [])
    with pytest.raises(MalformedInputError, match="positive total weight"):
        weighted_median([1], [0])
    with pytest.raises(MalformedInputError, match="non-empty and aligned"):
        mae_skill([], [], [])
    skill, naive_mae = mae_skill([2, 2], [2, 2], [1.0, 1.0])
    assert np.isnan(skill)
    assert naive_mae == 0.0
    assert np.isnan(krippendorff_alpha([[1], [2]], level="interval"))
    assert np.isnan(
        krippendorff_alpha([["same", "same"], ["same", "same"]], level="nominal")
    )
    with pytest.raises(MalformedInputError, match="Unsupported alpha"):
        krippendorff_alpha([[1, 2], [2, 1]], level="ratio")
    with pytest.raises(MalformedInputError, match="positive"):
        stratified_bootstrap_interval(
            pd.DataFrame({"stratum_id": ["a"], "value": [1]}),
            lambda frame: float(frame["value"].mean()),
            rng=np.random.default_rng(1),
            replicates=0,
        )
    with pytest.raises(MalformedInputError, match="non-empty stratum_id"):
        stratified_bootstrap_interval(
            pd.DataFrame(),
            lambda frame: 1.0,
            rng=np.random.default_rng(1),
        )
    assert all(
        np.isnan(value)
        for value in stratified_bootstrap_interval(
            pd.DataFrame({"stratum_id": ["a"], "value": [1]}),
            lambda frame: float("nan"),
            rng=np.random.default_rng(1),
            replicates=2,
        )
    )


def _complete_validation_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_values: dict[str, list[object]] = {
        "qdmr_step_count": [1, 2, 3],
        "hop_structure": ["atomic", "bridge", "comparison", "intersection"],
        "exhaustivity_requirement": [
            "mention_some",
            "weakly_exhaustive",
            "mention_all",
        ],
        "negative_conclusiveness": [False, True],
        "referring_form_type": [
            "full_name_form",
            "alias_or_handle",
            "email_address",
        ],
        "answer_locality": ["single_passage", "cross_document_aggregation"],
        "recall_orientation": ["precision_oriented", "recall_oriented"],
    }
    gold_rows: list[dict[str, object]] = []
    normalized_rows: list[dict[str, object]] = []
    for feature, values in feature_values.items():
        for level_index, value in enumerate(values):
            for repeat in range(12):
                item_id = f"{feature}-{level_index}-{repeat}"
                gold_rows.append(
                    {
                        "item_id": item_id,
                        "feature": feature,
                        "human_gold_value_json": (
                            str(value).lower()
                            if isinstance(value, bool)
                            else f'"{value}"'
                            if isinstance(value, str)
                            else str(value)
                        ),
                        "stratum_id": f"{feature}-{level_index}",
                        "inclusion_probability": 1.0,
                    }
                )
                for model_id in ("claude-opus-5", "gpt-5.6-sol"):
                    normalized_rows.append(
                        {
                            "item_id": item_id,
                            "model_id": model_id,
                            feature: value,
                        }
                    )
    return pd.DataFrame(gold_rows), pd.DataFrame(normalized_rows)


def test_feature_validation_requires_both_models_and_preserves_dossiers() -> None:
    gold, normalized = _complete_validation_frames()

    validation, dossiers = build_feature_validation(
        gold,
        normalized,
        bootstrap_replicates=30,
    )

    assert len(validation) == 14
    assert set(dossiers) == {
        "qdmr_step_count",
        "hop_structure",
        "exhaustivity_requirement",
        "negative_conclusiveness",
        "referring_form_type",
        "answer_locality",
        "recall_orientation",
    }
    assert all(dossier["structural_rules_passed"] for dossier in dossiers.values())
    assert all(
        dossier["both_models_metric_rule_passed"] for dossier in dossiers.values()
    )
    assert all(
        dossier["minimum_class_coverage_passed"] for dossier in dossiers.values()
    )

    bad_gpt = normalized.copy()
    bad_gpt.loc[
        bad_gpt["model_id"].eq("gpt-5.6-sol")
        & bad_gpt["item_id"].str.startswith("negative_conclusiveness"),
        "negative_conclusiveness",
    ] = False
    _, bad_dossiers = build_feature_validation(
        gold,
        bad_gpt,
        bootstrap_replicates=30,
    )

    assert (
        bad_dossiers["negative_conclusiveness"]["both_models_metric_rule_passed"]
        is False
    )
    assert bad_dossiers["negative_conclusiveness"]["structural_rules_passed"] is False


def test_feature_dossiers_are_exposed_one_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "analysis"
    specification = tmp_path / "specification.md"
    specification.write_text("# Specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(root, specification)
    pd.DataFrame({"artifact_schema_version": [1], "placeholder": [1]}).to_parquet(
        root / "tables" / "gold_labels.parquet", index=False
    )
    pd.DataFrame({"artifact_schema_version": [1], "placeholder": [1]}).to_parquet(
        root / "tables" / "annotations_normalized.parquet", index=False
    )
    order = validation_module._feature_decision_order()
    dossiers = {feature: {"schema_version": 1, "feature": feature} for feature in order}
    monkeypatch.setattr(
        validation_module,
        "build_feature_validation",
        lambda *args, **kwargs: (pd.DataFrame(), dossiers),
    )
    decisions_dir = tmp_path / "decisions"

    first_path = emit_next_feature_dossier(
        workspace,
        decisions_dir,
        bootstrap_replicates=1,
    )
    first_feature = order[0]
    first_dossier_bytes = canonical_json_bytes(dossiers[first_feature])
    decision = FeatureDecision(
        order=1,
        feature=first_feature,
        dossier_sha256=sha256(first_dossier_bytes).hexdigest(),
        status=FeatureValidationStatus.NOT_VALIDATED,
        material_failure_mode=False,
        justification="Structural fixture decision.",
        git_commit="abcdef0",
    )
    (decisions_dir / f"01-{first_feature}.json").write_text(
        json.dumps(decision.model_dump(mode="json")),
        encoding="utf-8",
    )

    second_path = emit_next_feature_dossier(
        workspace,
        decisions_dir,
        bootstrap_replicates=1,
    )

    assert first_path.name == f"01-{first_feature}-dossier.json"
    assert second_path.name == f"02-{order[1]}-dossier.json"
    assert len(list(decisions_dir.glob("*-dossier.json"))) == 2


def test_validate_features_persists_outputs_and_explicit_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "analysis"
    specification = tmp_path / "specification.md"
    specification.write_text("# Specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(root, specification)
    initial = datetime(2026, 9, 8, tzinfo=UTC)
    pd.DataFrame(
        {
            "artifact_schema_version": [1],
            "annotator_kind": ["human"],
            "initial_completed_at": [initial],
            "recode_completed_at": [initial + timedelta(days=14)],
            "human_recode_value_json": ['"value"'],
        }
    ).to_parquet(root / "tables" / "gold_labels.parquet", index=False)
    pd.DataFrame({"artifact_schema_version": [1], "model_id": ["model"]}).to_parquet(
        root / "tables" / "annotations_normalized.parquet", index=False
    )
    order = validation_module._feature_decision_order()
    dossiers = {
        feature: {
            "schema_version": 1,
            "feature": feature,
            "structural_rules_passed": True,
            "minimum_class_coverage_passed": True,
            "metric_undefined_degenerate_gold": False,
        }
        for feature in order
    }
    validation_frame = pd.DataFrame(
        {
            "feature": list(order),
            "model_id": ["claude-opus-5"] * len(order),
            "metric_value": [1.0] * len(order),
        }
    )
    monkeypatch.setattr(
        validation_module,
        "build_feature_validation",
        lambda *args, **kwargs: (validation_frame, dossiers),
    )
    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir()
    for index, feature in enumerate(order, start=1):
        dossier_sha = sha256(canonical_json_bytes(dossiers[feature])).hexdigest()
        decision = FeatureDecision(
            order=index,
            feature=feature,
            dossier_sha256=dossier_sha,
            status=FeatureValidationStatus.VALIDATED,
            material_failure_mode=False,
            justification="All frozen structural rules pass.",
            git_commit=f"abcde{index:02d}",
        )
        (decisions_dir / f"{index:02d}-{feature}.json").write_text(
            json.dumps(decision.model_dump(mode="json")),
            encoding="utf-8",
        )

    result = run_validate_features(workspace, decisions_dir)

    assert result.authorize_outcome_modeling is True
    assert len(result.artifacts) == 13
    assert (root / "validation_unlock.json").is_file()
    unlock = json.loads((root / "validation_unlock.json").read_text(encoding="utf-8"))
    assert unlock["authorized"] is True
    assert unlock["provisional_llm_can_unlock"] is False
    dsl_gate = json.loads((root / "dsl_gate.json").read_text(encoding="utf-8"))
    assert dsl_gate["status"] == "DSL_NOT_ESTABLISHED"
