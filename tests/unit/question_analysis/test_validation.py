"""Tests for Horvitz-Thompson feature-validation metrics."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from es_index_explorer.question_analysis import validation as validation_module
from es_index_explorer.question_analysis.contracts import FeatureValidationStatus
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.storage import (
    canonical_json_bytes,
    sha256_file,
    versioned_frame,
)
from es_index_explorer.question_analysis.validation import (
    ExploratoryFeatureDossier,
    FeatureDecision,
    FeatureValidationDossier,
    balanced_accuracy,
    build_exploratory_p4_evidence,
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


def test_feature_decision_requires_evidence_for_qualitative_limitations() -> None:
    base: dict[str, object] = {
        "order": 1,
        "feature": "answer_locality",
        "dossier_sha256": "a" * 64,
        "status": FeatureValidationStatus.VALIDATED_WITH_LIMITATIONS,
        "justification": "Pre-outcome materiality assessment.",
        "git_commit": "abcdef0",
    }
    with pytest.raises(ValidationError, match="failure-pattern IDs"):
        FeatureDecision.model_validate({**base, "material_failure_mode": True})
    with pytest.raises(ValidationError, match="interval IDs and justification"):
        FeatureDecision.model_validate(
            {
                **base,
                "material_failure_mode": False,
                "class_interval_material_limitation": True,
            }
        )


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


def test_explicit_model_missingness_is_retained_as_substantive_error() -> None:
    matrix = ht_confusion_totals(
        gold=[False, True],
        predicted=[False, "MISSING"],
        inclusion_probabilities=[1.0, 1.0],
        levels=[False, True],
        forecast_levels=[False, True, "MISSING"],
    )

    np.testing.assert_array_equal(
        matrix,
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert balanced_accuracy(matrix) == pytest.approx(0.5)
    assert macro_f1(matrix) == pytest.approx(0.5)
    with pytest.raises(MalformedInputError, match="explicit label inventories"):
        ht_confusion_totals(
            gold=[False],
            predicted=["UNKNOWN"],
            inclusion_probabilities=[1.0],
            levels=[False, True],
            forecast_levels=[False, True, "MISSING"],
        )


def test_model_dossier_reports_missingness_error_and_failure_example() -> None:
    rows = pd.DataFrame(
        [
            {
                "item_id": f"item-{index:02d}",
                "stratum_id": "single" if index < 12 else "cross",
                "human_value": (
                    "single_passage" if index < 12 else "cross_document_aggregation"
                ),
                "model_value": (
                    "not_classifiable_binary_locality"
                    if index == 12
                    else "single_passage"
                    if index < 12
                    else "cross_document_aggregation"
                ),
                "inclusion_probability": 0.5,
            }
            for index in range(24)
        ]
    )

    evidence = validation_module._model_validation_evidence(
        "answer_locality",
        "model-a",
        rows,
        bootstrap_replicates=10,
    )

    assert evidence.forecast_missingness_error_count == 1
    assert evidence.confusion is not None
    assert evidence.confusion.ht_cells[1][2] == pytest.approx(2.0)
    assert evidence.class_metrics[1].recall < 1.0
    assert evidence.failure_patterns[0].example_item_ids == ("item-12",)
    with pytest.raises(GateFailureError, match="null qdmr_step_count"):
        validation_module._model_validation_evidence(
            "qdmr_step_count",
            "model-a",
            pd.DataFrame(
                {
                    "item_id": ["q"],
                    "stratum_id": ["s"],
                    "human_value": [2],
                    "model_value": [None],
                    "inclusion_probability": [1.0],
                }
            ),
            bootstrap_replicates=1,
        )


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


def test_alpha_matches_coincidence_references_and_ht_expansion() -> None:
    assert krippendorff_alpha(
        [["a", "a"], ["a", "b"], ["b", "b"]],
        level="nominal",
    ) == pytest.approx(4 / 9)
    assert krippendorff_alpha(
        [["a", "b"], ["a", None], ["b", "b"]],
        level="nominal",
    ) == pytest.approx(0.0)
    assert krippendorff_alpha(
        [["a", "b"], ["a", "a"], ["b", "b"]],
        level="nominal",
        unit_weights=[2.0, 1.0, 1.0],
    ) == pytest.approx(0.125)

    near = krippendorff_alpha(
        [["low", "middle"], ["high", "high"], ["low", "low"]],
        level="ordinal",
        ordinal_levels=["low", "middle", "high"],
    )
    far = krippendorff_alpha(
        [["low", "high"], ["high", "high"], ["low", "low"]],
        level="ordinal",
        ordinal_levels=["low", "middle", "high"],
    )
    assert near > far
    assert validation_module._alpha_level("exhaustivity_requirement") == "nominal"
    assert validation_module._alpha_level("cognitive_process_level") == "ordinal"
    assert validation_module._alpha_level("qdmr_step_count") == "interval"


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


def test_human_test_retest_uses_joint_gold_recode_probabilities() -> None:
    gold = pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "feature": ["qdmr_step_count"] * 3,
            "human_gold_value_json": ["1", "10", "10"],
            "human_recode_value_json": ["1", "10", "1"],
            "recode_joint_inclusion_probability": [0.1, 1.0, 1.0],
            "stratum_id": ["s1", "s2", "s2"],
        }
    )

    evidence = validation_module._human_test_retest(
        "qdmr_step_count",
        gold,
        bootstrap_replicates=10,
    )

    assert evidence.metric_value == pytest.approx(0.5)
    assert evidence.weight == "inverse_joint_gold_recode_inclusion_probability"
    assert evidence.measurement_level == "interval"


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
    with pytest.raises(MalformedInputError, match="two gold rows"):
        balanced_accuracy(np.zeros((3, 3)))
    assert np.isnan(balanced_accuracy(np.array([[0.0, 0.0], [0.0, 1.0]])))
    with pytest.raises(MalformedInputError, match="every gold class"):
        macro_f1(np.zeros((3, 2)))
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
                        "human_recode_value_json": (
                            str(value).lower()
                            if isinstance(value, bool)
                            else f'"{value}"'
                            if isinstance(value, str)
                            else str(value)
                        ),
                        "recode_joint_inclusion_probability": 1.0,
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
    for feature, dossier in dossiers.items():
        validated_dossier = FeatureValidationDossier.model_validate(dossier)
        assert len(validated_dossier.model_summaries) == 2
        assert validated_dossier.human_test_retest.item_count >= 12
        assert validated_dossier.outcome_columns_loaded == ()
        if feature == "qdmr_step_count":
            assert all(
                summary.confusion is None and summary.numeric_error_summary is not None
                for summary in validated_dossier.model_summaries
            )
        else:
            assert all(
                summary.confusion is not None and summary.class_metrics
                for summary in validated_dossier.model_summaries
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


def test_exploratory_evidence_is_comprehensive_and_non_gating() -> None:
    feature_values: dict[str, list[object]] = {
        "presupposition_load": [False, True],
        "cognitive_process_level": ["remember", "analyze"],
        "qdmr_operator_set": [["SELECT"], ["FILTER", "SELECT"]],
        "qdmr_normalized_question": [
            "Who approved the plan?",
            "Which records mention the event?",
        ],
        "demand_type": ["entity_identification", "relational_claim"],
        "specificity": ["specific", "broad"],
        "qdmr_applicability": ["applicable", "not_applicable"],
    }
    gold_rows: list[dict[str, object]] = []
    normalized_rows: list[dict[str, object]] = []
    for feature, values in feature_values.items():
        for level_index, value in enumerate(values):
            for repeat in range(3):
                item_id = f"{feature}-{level_index}-{repeat}"
                encoded = json.dumps(value)
                gold_rows.append(
                    {
                        "item_id": item_id,
                        "feature": feature,
                        "human_gold_value_json": encoded,
                        "human_recode_value_json": encoded,
                        "stratum_id": f"{feature}-{level_index}",
                        "inclusion_probability": 1.0,
                        "recode_joint_inclusion_probability": 1.0,
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

    table, dossiers = build_exploratory_p4_evidence(
        pd.DataFrame(gold_rows),
        pd.DataFrame(normalized_rows),
        bootstrap_replicates=5,
    )

    assert set(dossiers) == set(feature_values)
    assert table["non_gating"].all()
    assert "validation_status" not in table
    for feature, dossier in dossiers.items():
        parsed = ExploratoryFeatureDossier.model_validate(dossier)
        assert parsed.availability == "AVAILABLE"
        assert parsed.non_gating
        assert len(parsed.model_summaries) == 2
        assert parsed.human_test_retest_metrics
        assert "validation_status" not in dossier
        if feature == "qdmr_operator_set":
            assert parsed.operator_evidence_by_model is not None
            assert {metric.metric for metric in parsed.model_summaries[0].metrics} == {
                "ht_weighted_mean_jaccard",
                "ht_weighted_exact_set_match",
            }
        if feature == "qdmr_normalized_question":
            assert parsed.evidence_kind == "faithfulness_text"


def test_validation_evidence_replay_is_deterministic() -> None:
    gold, normalized = _complete_validation_frames()

    first_table, first_dossiers = build_feature_validation(
        gold,
        normalized,
        bootstrap_replicates=5,
    )
    second_table, second_dossiers = build_feature_validation(
        gold,
        normalized,
        bootstrap_replicates=5,
    )

    pd.testing.assert_frame_equal(first_table, second_table)
    assert canonical_json_bytes(first_dossiers["qdmr_step_count"]) == (
        canonical_json_bytes(second_dossiers["qdmr_step_count"])
    )


def test_feature_dossiers_are_exposed_one_at_a_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analysis"
    specification = tmp_path / "specification.md"
    specification.write_text("# Specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(root, specification)
    gold, normalized = _complete_validation_frames()
    versioned_frame(gold).to_parquet(
        root / "tables" / "gold_labels.parquet", index=False
    )
    versioned_frame(
        gold[["item_id", "stratum_id", "inclusion_probability"]].drop_duplicates()
    ).to_parquet(root / "tables" / "gold_sample.parquet", index=False)
    versioned_frame(normalized).to_parquet(
        root / "tables" / "annotations_normalized.parquet", index=False
    )
    (root / "annotation_manifest.json").write_text("{}\n", encoding="utf-8")
    order = validation_module._feature_decision_order()
    decisions_dir = tmp_path / "decisions"

    first_path = emit_next_feature_dossier(
        workspace,
        decisions_dir,
        bootstrap_replicates=5,
    )
    first_feature = order[0]
    first_dossier_bytes = first_path.read_bytes()
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
        bootstrap_replicates=5,
    )

    assert first_path.name == f"01-{first_feature}-dossier.json"
    assert second_path.name == f"02-{order[1]}-dossier.json"
    assert len(list(decisions_dir.glob("*-dossier.json"))) == 2


def test_validate_features_persists_outputs_and_explicit_unlock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analysis"
    specification = tmp_path / "specification.md"
    specification.write_text("# Specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(root, specification)
    gold, normalized = _complete_validation_frames()
    gold["annotator_kind"] = "human"
    versioned_frame(gold).to_parquet(
        root / "tables" / "gold_labels.parquet", index=False
    )
    versioned_frame(
        gold[["item_id", "stratum_id", "inclusion_probability"]].drop_duplicates()
    ).to_parquet(root / "tables" / "gold_sample.parquet", index=False)
    versioned_frame(normalized).to_parquet(
        root / "tables" / "annotations_normalized.parquet", index=False
    )
    (root / "annotation_manifest.json").write_text("{}\n", encoding="utf-8")
    order = validation_module._feature_decision_order()
    codebook = workspace.load_manifest().resources["segment3_codebook"].sha256
    hashes = {
        "gold_sample_sha256": sha256_file(root / "tables" / "gold_sample.parquet"),
        "gold_labels_sha256": sha256_file(root / "tables" / "gold_labels.parquet"),
        "annotation_manifest_sha256": sha256_file(root / "annotation_manifest.json"),
        "codebook_sha256": codebook,
    }
    _, dossiers = build_feature_validation(
        gold,
        normalized,
        bootstrap_replicates=10,
        evidence_hashes=hashes,
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

    result = run_validate_features(
        workspace,
        decisions_dir,
        bootstrap_replicates=10,
    )

    assert result.authorize_outcome_modeling is True
    assert len(result.artifacts) == 22
    assert (root / "validation_unlock.json").is_file()
    unlock = json.loads((root / "validation_unlock.json").read_text(encoding="utf-8"))
    assert unlock["authorized"] is True
    assert unlock["provisional_llm_can_unlock"] is False
    dsl_gate = json.loads((root / "dsl_gate.json").read_text(encoding="utf-8"))
    assert dsl_gate["status"] == "DSL_NOT_ESTABLISHED"
    validation_rows = pd.read_parquet(root / "tables" / "feature_validation.parquet")
    validated_rows = pd.read_parquet(root / "tables" / "validated_features.parquet")
    assert len(validation_rows) == 14
    assert len(validated_rows) == 7
    assert {
        "scored_item_count",
        "gold_missingness_count",
        "forecast_missingness_error_count",
    }.issubset(validation_rows.columns)
    qdmr_dossier = json.loads(
        (root / "statistics" / "feature_validation" / "qdmr_step_count.json").read_text(
            encoding="utf-8"
        )
    )
    assert FeatureValidationDossier.model_validate(qdmr_dossier).schema_version == 2
    report = (root / "partial_reports" / "04-feature-validation.md").read_text(
        encoding="utf-8"
    )
    assert "weakest" in report
    assert "widest model interval" in report
    exploratory = json.loads(
        (
            root / "statistics" / "exploratory_p4_evidence" / "qdmr_operator_set.json"
        ).read_text(encoding="utf-8")
    )
    parsed_exploratory = ExploratoryFeatureDossier.model_validate(exploratory)
    assert parsed_exploratory.non_gating
    assert parsed_exploratory.availability == "UNAVAILABLE"
