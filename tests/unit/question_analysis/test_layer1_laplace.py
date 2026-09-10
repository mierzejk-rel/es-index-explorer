"""Tests for Layer 1 conditional Laplace and importance diagnostics."""

import numpy as np
import pytest

from es_index_explorer.question_analysis.errors import Layer1ModeError
from es_index_explorer.question_analysis.layer1_laplace import (
    draw_rubric_v2,
    importance_resampling_diagnostic,
    make_rubric_draw_block,
    observed_rubric_performance,
    select_importance_rubrics,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1VariantData,
    _accepted_curvature,
    find_rubric_mode,
)

pytestmark = pytest.mark.unit


def _rubric(
    rubric_id: str = "r1",
    *,
    rubric_order: int = 1,
    dataset: str = "emc2_set1",
    expectation_count: int = 8,
    probabilities: tuple[float, float, float] = (0.55, 0.30, 0.15),
    variant_count: int = 2,
) -> Layer1RubricData:
    variants = tuple(
        Layer1VariantData(
            variant_id=f"{rubric_id}-v{variant_index}",
            counts=np.random.default_rng(
                100 + rubric_order * 10 + variant_index
            ).multinomial(
                expectation_count,
                probabilities,
                size=28,
            ),
            arm_ids=tuple(f"a{index:02d}" for index in range(28)),
        )
        for variant_index in range(variant_count)
    )
    return Layer1RubricData(
        rubric_id=rubric_id,
        rubric_order=rubric_order,
        dataset=dataset,
        expectation_count=expectation_count,
        variants=variants,
    )


def _hyperparameters() -> Layer1Hyperparameters:
    return Layer1Hyperparameters(
        phi=15.0,
        mu0=np.array([0.5, -0.5]),
        dataset_offsets={
            "emc2_set1": np.zeros(2),
            "emc2_set2": np.array([0.1, 0.0]),
            "mallinckrodt": np.array([-0.1, 0.1]),
        },
        sigma_within=np.array([[0.25, 0.03], [0.03, 0.20]]),
        sigma_between=np.array([[0.30, 0.02], [0.02, 0.28]]),
    )


def _moments(rubric: Layer1RubricData) -> Layer1MomentStart:
    eta = np.array([0.5, -0.5])
    return Layer1MomentStart(
        hyperparameters=_hyperparameters(),
        rubric_latent_starts={
            rubric.rubric_id: np.concatenate((eta, np.tile(eta, len(rubric.variants))))
        },
        variant_alr={variant.variant_id: eta for variant in rubric.variants},
        smoothed_variant_count=0,
        retained_phi_components=3,
    )


def test_conditional_draws_are_joint_deterministic_and_bounded() -> None:
    rubric = _rubric()
    moments = _moments(rubric)
    mode = find_rubric_mode(
        rubric,
        _hyperparameters(),
        moments.rubric_latent_starts[rubric.rubric_id],
    )

    first = draw_rubric_v2(rubric, mode, rng=np.random.default_rng(41), draw_count=40)
    second = draw_rubric_v2(rubric, mode, rng=np.random.default_rng(41), draw_count=40)

    assert first.shape == (40, 2)
    assert np.all((0 <= first) & (first <= 1))
    np.testing.assert_array_equal(first, second)


def test_draw_block_preserves_global_and_local_indices() -> None:
    rubric = _rubric()
    block = make_rubric_draw_block(
        rubric,
        _hyperparameters(),
        _moments(rubric),
        rng=np.random.default_rng(5),
        global_outer_attempt_id=17,
        rubric_retained_index=11,
    )

    assert block.rubric_id == rubric.rubric_id
    assert block.variant_ids == ("r1-v0", "r1-v1")
    assert block.global_outer_attempt_id == 17
    assert block.rubric_retained_index == 11
    assert block.rubric_v2_draws.shape == (40, 2)


def test_importance_resampling_reports_ess_and_weighted_moments() -> None:
    rubric = _rubric()
    diagnostic = importance_resampling_diagnostic(
        rubric,
        _hyperparameters(),
        _moments(rubric),
        rng=np.random.default_rng(9),
        particle_count=400,
    )

    assert 0 < diagnostic.effective_sample_size <= 400
    assert diagnostic.effective_sample_size_ratio == pytest.approx(
        diagnostic.effective_sample_size / 400
    )
    assert diagnostic.laplace_mean.shape == diagnostic.importance_mean.shape == (6,)
    assert (
        diagnostic.laplace_covariance.shape
        == diagnostic.importance_covariance.shape
        == (6, 6)
    )


def test_curvature_requires_strict_positive_definiteness_at_tolerance() -> None:
    accepted, _ = _accepted_curvature(-np.diag([1.0, 1.1e-10]))

    np.testing.assert_allclose(np.diag(accepted), [1.0, 1.1e-10])
    with pytest.raises(Layer1ModeError, match="not positive definite"):
        _accepted_curvature(-np.diag([1.0, 1e-10]))
    with pytest.raises(Layer1ModeError, match="not positive definite"):
        _accepted_curvature(-np.diag([1.0, 0.0]))


def test_purposive_selection_uses_stable_union_of_four_selectors() -> None:
    rubrics = (
        _rubric("small-v", rubric_order=4, variant_count=1),
        _rubric("small-n", rubric_order=3, expectation_count=2),
        _rubric(
            "low",
            rubric_order=2,
            probabilities=(0.1, 0.8, 0.1),
        ),
        _rubric(
            "high",
            rubric_order=1,
            probabilities=(0.85, 0.1, 0.05),
        ),
        _rubric("middle", rubric_order=0),
    )
    data = Layer1Dataset(
        rubrics=rubrics,
        dataset_levels=("emc2_set1",),
    )

    assert select_importance_rubrics(data) == (
        "small-v",
        "small-n",
        "low",
        "high",
    )
    assert observed_rubric_performance(rubrics[2]) < observed_rubric_performance(
        rubrics[3]
    )
