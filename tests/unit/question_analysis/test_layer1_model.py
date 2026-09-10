"""Tests for Layer 1 transforms, moments, density, and marginal-fit contracts."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import dirichlet_multinomial

from es_index_explorer.question_analysis import layer1_model as layer1_module
from es_index_explorer.question_analysis.errors import (
    Layer1InitialiserSensitiveError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1StartFit,
    Layer1VariantData,
    alr_from_counts,
    covariance_to_log_cholesky,
    dirichlet_multinomial_logpmf,
    find_rubric_mode,
    fit_mml_start,
    inverse_alr,
    log_cholesky_to_covariance,
    method_of_moments_start,
    pack_hyperparameters,
    rubric_log_density_derivatives,
    unpack_hyperparameters,
)

pytestmark = pytest.mark.unit


def _variant(
    variant_id: str,
    probabilities: tuple[float, float, float],
    *,
    seed: int,
    traces: int = 28,
    expectation_count: int = 12,
) -> Layer1VariantData:
    rng = np.random.default_rng(seed)
    return Layer1VariantData(
        variant_id=variant_id,
        counts=rng.multinomial(expectation_count, probabilities, size=traces),
        arm_ids=tuple(f"arm-{index:02d}" for index in range(traces)),
    )


def _dataset() -> Layer1Dataset:
    rubrics: list[Layer1RubricData] = []
    datasets = ("emc2_set1", "emc2_set2", "mallinckrodt")
    for dataset_index, dataset in enumerate(datasets):
        for rubric_index in range(3):
            base = 0.42 + 0.03 * dataset_index + 0.01 * rubric_index
            variants = (
                _variant(
                    f"{dataset}-r{rubric_index}-v0",
                    (base, 0.38, 1.0 - base - 0.38),
                    seed=100 * dataset_index + 10 * rubric_index,
                ),
                _variant(
                    f"{dataset}-r{rubric_index}-v1",
                    (base - 0.04, 0.40, 1.0 - base + 0.04 - 0.40),
                    seed=100 * dataset_index + 10 * rubric_index + 1,
                ),
            )
            rubrics.append(
                Layer1RubricData(
                    rubric_id=f"{dataset}-r{rubric_index}",
                    rubric_order=len(rubrics),
                    dataset=dataset,
                    expectation_count=12,
                    variants=variants,
                )
            )
    return Layer1Dataset(rubrics=tuple(rubrics), dataset_levels=datasets)


def _hyperparameters() -> Layer1Hyperparameters:
    return Layer1Hyperparameters(
        phi=20.0,
        mu0=np.array([0.2, -0.4]),
        dataset_offsets={
            "emc2_set1": np.zeros(2),
            "emc2_set2": np.array([0.1, -0.05]),
            "mallinckrodt": np.array([-0.1, 0.08]),
        },
        sigma_within=np.array([[0.30, 0.05], [0.05, 0.20]]),
        sigma_between=np.array([[0.40, -0.03], [-0.03, 0.25]]),
    )


def _optimizer_dataset() -> Layer1Dataset:
    rng = np.random.default_rng(321)
    rubrics: list[Layer1RubricData] = []
    for dataset_index, dataset in enumerate(("emc2_set1", "emc2_set2", "mallinckrodt")):
        for rubric_index in range(2):
            variants: list[Layer1VariantData] = []
            for variant_index in range(2):
                probabilities = np.array(
                    [
                        0.50 + 0.04 * dataset_index - 0.03 * variant_index,
                        0.32,
                        0.18 - 0.04 * dataset_index + 0.03 * variant_index,
                    ]
                )
                variants.append(
                    Layer1VariantData(
                        variant_id=f"{dataset}-r{rubric_index}-v{variant_index}",
                        counts=rng.multinomial(8, probabilities, size=8),
                        arm_ids=tuple(f"a{index}" for index in range(8)),
                    )
                )
            rubrics.append(
                Layer1RubricData(
                    rubric_id=f"{dataset}-r{rubric_index}",
                    rubric_order=len(rubrics),
                    dataset=dataset,
                    expectation_count=8,
                    variants=tuple(variants),
                )
            )
    return Layer1Dataset(
        rubrics=tuple(rubrics),
        dataset_levels=("emc2_set1", "emc2_set2", "mallinckrodt"),
    )


def test_alr_boundary_adds_epsilon_to_all_components() -> None:
    coordinates, smoothed = alr_from_counts([2, 0, 3])
    expected = np.array([np.log(2.5 / 0.5), np.log(3.5 / 0.5)])

    assert smoothed
    np.testing.assert_allclose(coordinates, expected)
    probabilities = inverse_alr(coordinates)
    np.testing.assert_allclose(probabilities, np.array([2.5, 0.5, 3.5]) / 6.5)


def test_alr_positive_counts_are_unsmoothed_and_round_trip() -> None:
    coordinates, smoothed = alr_from_counts([2, 3, 5])

    assert not smoothed
    np.testing.assert_allclose(inverse_alr(coordinates), np.array([0.2, 0.3, 0.5]))


def test_alr_rejects_malformed_counts() -> None:
    for counts in ([0, 0, 0], [1, -1, 2], [1, 2], [1, np.nan, 2]):
        with pytest.raises(MalformedInputError):
            alr_from_counts(counts)


def test_log_cholesky_round_trip_is_positive_definite() -> None:
    covariance = np.array([[0.7, 0.2], [0.2, 0.4]])
    parameters = covariance_to_log_cholesky(covariance)
    rebuilt = log_cholesky_to_covariance(parameters)

    np.testing.assert_allclose(rebuilt, covariance)
    assert np.linalg.eigvalsh(rebuilt).min() > 0


def test_hyperparameter_pack_round_trip_fixes_reference_offset() -> None:
    data = _dataset()
    expected = _hyperparameters()

    vector = pack_hyperparameters(expected, data.dataset_levels)
    actual = unpack_hyperparameters(vector, data.dataset_levels)

    assert actual.phi == pytest.approx(expected.phi)
    np.testing.assert_allclose(actual.mu0, expected.mu0)
    np.testing.assert_array_equal(actual.dataset_offsets["emc2_set1"], np.zeros(2))
    for dataset in ("emc2_set2", "mallinckrodt"):
        np.testing.assert_allclose(
            actual.dataset_offsets[dataset], expected.dataset_offsets[dataset]
        )
    np.testing.assert_allclose(actual.sigma_within, expected.sigma_within)
    np.testing.assert_allclose(actual.sigma_between, expected.sigma_between)


def test_dirichlet_multinomial_logpmf_matches_scipy_on_raw_counts() -> None:
    counts = np.array([[3, 2, 1], [0, 4, 2]])
    eta = np.array([0.3, -0.2])
    phi = 8.0
    theta = inverse_alr(eta)
    expected = sum(
        dirichlet_multinomial.logpmf(row, alpha=phi * theta, n=int(row.sum()))
        for row in counts
    )

    assert dirichlet_multinomial_logpmf(counts, eta, phi) == pytest.approx(expected)


def test_method_of_moments_uses_reference_offset_and_pooled_covariance() -> None:
    data = _dataset()
    result = method_of_moments_start(data)

    assert result.hyperparameters.phi >= 0.1
    assert result.hyperparameters.phi <= 1_000
    np.testing.assert_array_equal(
        result.hyperparameters.dataset_offsets["emc2_set1"], np.zeros(2)
    )
    assert np.linalg.eigvalsh(result.hyperparameters.sigma_within).min() > 0
    assert np.linalg.eigvalsh(result.hyperparameters.sigma_between).min() > 0
    assert result.retained_phi_components > 0
    assert len(result.variant_alr) == 18


def test_latent_gradient_and_hessian_match_finite_differences() -> None:
    rubric = _dataset().rubrics[0]
    hyperparameters = _hyperparameters()
    initial = method_of_moments_start(_dataset()).rubric_latent_starts[rubric.rubric_id]
    _, gradient, hessian = rubric_log_density_derivatives(
        rubric, hyperparameters, initial
    )
    step = 1e-5
    numerical_gradient = np.zeros_like(gradient)
    numerical_hessian = np.zeros_like(hessian)

    def score(value: np.ndarray) -> np.ndarray:
        return rubric_log_density_derivatives(rubric, hyperparameters, value)[1]

    for index in range(len(initial)):
        direction = np.zeros_like(initial)
        direction[index] = step
        plus_density = rubric_log_density_derivatives(
            rubric, hyperparameters, initial + direction
        )[0]
        minus_density = rubric_log_density_derivatives(
            rubric, hyperparameters, initial - direction
        )[0]
        numerical_gradient[index] = (plus_density - minus_density) / (2 * step)
        numerical_hessian[:, index] = (
            score(initial + direction) - score(initial - direction)
        ) / (2 * step)

    np.testing.assert_allclose(gradient, numerical_gradient, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)


def test_newton_mode_satisfies_gradient_and_positive_curvature() -> None:
    data = _dataset()
    moments = method_of_moments_start(data)
    rubric = data.rubrics[0]
    mode = find_rubric_mode(
        rubric,
        _hyperparameters(),
        moments.rubric_latent_starts[rubric.rubric_id],
    )

    assert mode.gradient_maximum < 1e-8
    assert mode.iterations <= 100
    assert np.linalg.eigvalsh(mode.curvature).min() > 0
    np.testing.assert_allclose(mode.covariance @ mode.curvature, np.eye(6), atol=1e-8)


def test_three_start_failure_is_typed_and_does_not_select_a_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _dataset()
    moments = method_of_moments_start(data)
    vector = pack_hyperparameters(moments.hyperparameters, data.dataset_levels)
    call_count = 0

    def fake_fit(
        _data: Layer1Dataset,
        _moments: Layer1MomentStart,
        *,
        phi_multiplier: float,
    ) -> Layer1StartFit:
        nonlocal call_count
        call_count += 1
        return Layer1StartFit(
            phi_multiplier=phi_multiplier,
            vector=vector + (0.1 if call_count == 3 else 0.0),
            hyperparameters=moments.hyperparameters,
            objective=1.0,
            gradient_maximum=1e-6,
            iterations=2,
            converged=True,
            message="synthetic",
        )

    monkeypatch.setattr(layer1_module, "fit_mml_start", fake_fit)

    with pytest.raises(Layer1InitialiserSensitiveError):
        layer1_module.fit_layer1_mml(data, moments)


def test_one_failed_start_is_non_equivalent() -> None:
    data = _dataset()
    moments = method_of_moments_start(data)
    vector = pack_hyperparameters(moments.hyperparameters, data.dataset_levels)
    passing = Layer1StartFit(
        phi_multiplier=1.0,
        vector=vector,
        hyperparameters=moments.hyperparameters,
        objective=1.0,
        gradient_maximum=1e-6,
        iterations=2,
        converged=True,
        message="passing",
    )

    assert not layer1_module._starts_equivalent(
        (
            passing,
            replace(passing, phi_multiplier=0.5),
            replace(passing, converged=False),
        )
    )


def test_real_mml_start_returns_complete_strict_convergence_diagnostics() -> None:
    data = _optimizer_dataset()
    moments = method_of_moments_start(data)

    result = fit_mml_start(data, moments, phi_multiplier=1.0)

    assert result.vector.shape == (13,)
    assert np.isfinite(result.objective)
    assert np.isfinite(result.hyperparameters.phi)
    assert result.iterations <= 200
    assert result.message
