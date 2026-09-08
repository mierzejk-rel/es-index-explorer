"""Tests for three-term CGM covariance and stacked score construction."""

from collections.abc import Sequence

import numpy as np
import pytest

from es_index_explorer.question_analysis.cluster_covariance import (
    joint_wald_statistic,
    project_psd,
    three_term_cluster_meat,
)
from es_index_explorer.question_analysis.errors import (
    MalformedInputError,
    NumericalError,
)
from es_index_explorer.question_analysis.stacked_scores import (
    apply_arm_weights,
    generate_offdiagonal_oracle_scores,
    stack_score_rows,
    stacked_bread,
    stacked_restriction,
    validate_common_design_columns,
)

pytestmark = pytest.mark.unit


def _independent_outer_sum(scores: np.ndarray, labels: Sequence[object]) -> np.ndarray:
    unique = list(dict.fromkeys(labels))
    cluster_sums = [
        scores[np.asarray([label == current for label in labels])].sum(axis=0)
        for current in unique
    ]
    factor = len(unique) / (len(unique) - 1)
    return factor * sum(
        (np.outer(value, value) for value in cluster_sums),
        start=np.zeros((scores.shape[1], scores.shape[1])),
    )


def test_three_term_meat_uses_realised_per_term_corrections() -> None:
    scores = np.array([[1.0], [2.0], [3.0], [5.0], [7.0]])
    rubric = ["r1", "r1", "r2", "r2", "r2"]
    arm = ["a1", "a2", "a1", "a2", "a3"]

    result = three_term_cluster_meat(scores, rubric, arm)

    intersections = list(zip(rubric, arm, strict=True))
    expected_rubric = _independent_outer_sum(scores, rubric)
    expected_arm = _independent_outer_sum(scores, arm)
    expected_intersection = _independent_outer_sum(scores, intersections)
    np.testing.assert_allclose(result.rubric, expected_rubric)
    np.testing.assert_allclose(result.arm, expected_arm)
    np.testing.assert_allclose(result.intersection, expected_intersection)
    np.testing.assert_allclose(
        result.combined, expected_rubric + expected_arm - expected_intersection
    )
    assert result.rubric_cluster_count == 2
    assert result.arm_cluster_count == 3
    assert result.intersection_cluster_count == 5


def test_psd_projection_symmetrizes_and_clips_negative_eigenvalue() -> None:
    result = project_psd(np.array([[1.0, 2.0 + 1e-12], [2.0, 1.0]]))

    assert result.projection_applied
    np.testing.assert_allclose(result.matrix, result.matrix.T)
    assert np.linalg.eigvalsh(result.matrix).min() >= -1e-12
    assert result.eigenvalues_after.min() == 0.0


def test_singular_restricted_covariance_is_non_computable() -> None:
    with pytest.raises(NumericalError, match="singular after PSD"):
        joint_wald_statistic(
            np.array([1.0, 2.0]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 1.0]]),
        )


def test_stacked_scores_preserve_resolved_conservative_cross_meat() -> None:
    system = generate_offdiagonal_oracle_scores()
    scores = system.score_rows
    rubric = list(system.rubric_clusters)
    arm = list(system.arm_clusters)

    result = three_term_cluster_meat(
        system.score_rows, system.rubric_clusters, system.arm_clusters
    )
    independent = (
        _independent_outer_sum(scores, rubric)
        + _independent_outer_sum(scores, arm)
        - _independent_outer_sum(scores, list(zip(rubric, arm, strict=True)))
    )

    np.testing.assert_allclose(result.combined[0, 1], independent[0, 1], rtol=1e-6)
    assert result.combined[0, 1] != 0.0
    np.testing.assert_array_equal(
        stacked_bread(np.array([[2.0]]), np.array([[3.0]])),
        np.diag([2.0, 3.0]),
    )


def test_unresolved_rows_are_zero_padded_in_resolved_block() -> None:
    system = stack_score_rows(
        ["resolved"],
        np.array([[2.0]]),
        ["resolved", "unresolved"],
        np.array([[3.0], [5.0]]),
        {"resolved": "r1", "unresolved": "r2"},
        {"resolved": "a1", "unresolved": "a2"},
    )

    assert system.row_ids == ("resolved", "unresolved")
    np.testing.assert_array_equal(system.score_rows, np.array([[2.0, 3.0], [0.0, 5.0]]))


def test_resolved_unresolved_rows_contribute_to_cross_outcome_cluster_meat() -> None:
    conservative_ids = ["r1-a1", "r1-a2", "r2-a1", "r2-a2"]
    system = stack_score_rows(
        ["r1-a1", "r2-a2"],
        np.array([[1.0], [2.0]]),
        conservative_ids,
        np.array([[3.0], [5.0], [7.0], [11.0]]),
        {
            "r1-a1": "r1",
            "r1-a2": "r1",
            "r2-a1": "r2",
            "r2-a2": "r2",
        },
        {
            "r1-a1": "a1",
            "r1-a2": "a2",
            "r2-a1": "a1",
            "r2-a2": "a2",
        },
    )

    meat = three_term_cluster_meat(
        system.score_rows, system.rubric_clusters, system.arm_clusters
    )
    assert meat.rubric_cluster_count == 2
    assert meat.arm_cluster_count == 2
    assert meat.intersection_cluster_count == 4
    assert meat.combined[0, 1] != 0.0


def test_both_outcomes_must_retain_declared_design_columns() -> None:
    assert validate_common_design_columns(
        ("intercept", "feature"),
        ("intercept", "feature"),
        ("intercept", "feature"),
    ) == ("intercept", "feature")

    with pytest.raises(MalformedInputError, match="retain the declared"):
        validate_common_design_columns(
            ("intercept", "feature"),
            ("intercept", "feature"),
            ("intercept",),
        )
    with pytest.raises(MalformedInputError, match="non-empty and unique"):
        validate_common_design_columns(("feature", "feature"), (), ())


def test_stacked_score_input_contracts_reject_ambiguous_rows() -> None:
    with pytest.raises(MalformedInputError, match="row IDs must be unique"):
        stack_score_rows(
            ["row", "row"],
            np.ones((2, 1)),
            ["row"],
            np.ones((1, 1)),
            {"row": "r1"},
            {"row": "a1"},
        )
    with pytest.raises(MalformedInputError, match="rubric and arm"):
        stack_score_rows(
            ["row"],
            np.ones((1, 1)),
            ["row"],
            np.ones((1, 1)),
            {},
            {},
        )


def test_stacked_matrix_and_weight_contracts_fail_closed() -> None:
    with pytest.raises(MalformedInputError, match="bread must be finite and square"):
        stacked_bread(np.ones((2, 1)), np.eye(1))
    with pytest.raises(MalformedInputError, match="restriction must be a finite"):
        stacked_restriction(np.array([1.0]), np.ones((1, 1)))
    with pytest.raises(MalformedInputError, match="Missing bootstrap weight"):
        apply_arm_weights(np.ones((2, 1)), ("a1", "a2"), {"a1": 1.0})
    with pytest.raises(MalformedInputError, match="must be finite"):
        apply_arm_weights(np.ones((2, 1)), ("a1", "a2"), {"a1": 1.0, "a2": np.nan})


def test_cluster_covariance_input_contracts_fail_closed() -> None:
    with pytest.raises(MalformedInputError, match="non-empty finite matrix"):
        three_term_cluster_meat(np.array([[np.nan]]), ("r1",), ("a1",))
    with pytest.raises(MalformedInputError, match="align with score rows"):
        three_term_cluster_meat(np.ones((2, 1)), ("r1",), ("a1", "a2"))
    with pytest.raises(NumericalError, match="at least two realised"):
        three_term_cluster_meat(np.ones((2, 1)), ("r1", "r1"), ("a1", "a2"))
