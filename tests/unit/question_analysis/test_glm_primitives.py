"""Tests for score, bread, and frozen Newton solver primitives."""

from collections.abc import Callable

import numpy as np
import pytest
import statsmodels.api as sm

from es_index_explorer.question_analysis.errors import (
    MalformedInputError,
    NumericalError,
)
from es_index_explorer.question_analysis.glm_primitives import (
    bernoulli_bread,
    bernoulli_means,
    bernoulli_score_rows,
    linear_bread,
    linear_score_rows,
    solve_restricted_linear,
    solve_restricted_logit,
    solve_unrestricted_linear,
    solve_unrestricted_logit,
)

pytestmark = pytest.mark.unit


def _logit_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    predictor = rng.normal(size=400)
    design = np.column_stack((np.ones(len(predictor)), predictor))
    probabilities = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * predictor)))
    response = rng.binomial(1, probabilities).astype(float)
    return design, response


def test_logit_scores_bread_and_newton_match_statsmodels() -> None:
    design, response = _logit_fixture()

    result = solve_unrestricted_logit(design, response)
    reference = sm.GLM(response, design, family=sm.families.Binomial()).fit()

    np.testing.assert_allclose(result.coefficients, reference.params, rtol=1e-8)
    np.testing.assert_allclose(
        bernoulli_score_rows(design, response, result.coefficients).sum(axis=0),
        np.zeros(design.shape[1]),
        atol=1e-8,
    )
    means = 1.0 / (1.0 + np.exp(-(design @ result.coefficients)))
    np.testing.assert_allclose(
        bernoulli_bread(design, result.coefficients),
        design.T @ (design * (means * (1.0 - means))[:, None]),
    )


def test_restricted_logit_enforces_exact_constraint() -> None:
    design, response = _logit_fixture()
    unrestricted = solve_unrestricted_logit(design, response)
    restriction = np.array([[0.0, 1.0]])

    result = solve_restricted_logit(
        design,
        response,
        restriction,
        initial=unrestricted.coefficients,
    )

    np.testing.assert_allclose(restriction @ result.coefficients, 0.0, atol=1e-10)
    assert result.max_residual < 1e-8


def test_linear_solvers_satisfy_score_and_restriction() -> None:
    design = np.array([[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]])
    response = np.array([0.0, 1.0, 3.0, 4.0])
    restriction = np.array([[0.0, 1.0]])

    unrestricted = solve_unrestricted_linear(design, response)
    restricted = solve_restricted_linear(design, response, restriction)

    np.testing.assert_allclose(
        linear_score_rows(design, response, unrestricted.coefficients).sum(axis=0),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(restriction @ restricted.coefficients, 0.0)


def test_singular_design_fails_numerically() -> None:
    design = np.ones((20, 2))
    response = np.tile(np.array([0.0, 1.0]), 10)

    with pytest.raises(NumericalError, match="bread is singular"):
        solve_unrestricted_logit(design, response)

    with pytest.raises(NumericalError, match="Linear bread is singular"):
        solve_unrestricted_linear(design, response)


@pytest.mark.parametrize(
    ("operation", "message"),
    (
        (lambda: bernoulli_means(np.ones(3), np.ones(1)), "not aligned"),
        (
            lambda: bernoulli_means(np.array([[np.nan]]), np.ones(1)),
            "must be finite",
        ),
        (
            lambda: bernoulli_score_rows(
                np.ones((2, 1)), np.array([0.0, 2.0]), np.zeros(1)
            ),
            "responses must lie",
        ),
        (lambda: linear_bread(np.ones(3)), "finite matrix"),
        (
            lambda: linear_bread(np.ones((2, 1)), estimating_weights=np.ones(3)),
            "weights are invalid",
        ),
    ),
)
def test_primitive_input_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(MalformedInputError, match=message):
        operation()


def test_newton_iteration_caps_are_consequential() -> None:
    design, response = _logit_fixture()
    restriction = np.array([[0.0, 1.0]])

    with pytest.raises(NumericalError, match="did not converge in 0"):
        solve_unrestricted_logit(design, response, max_iterations=0)
    with pytest.raises(NumericalError, match="did not converge in 0"):
        solve_restricted_logit(
            design,
            response,
            restriction,
            initial=np.array([0.0, 1.0]),
            max_iterations=0,
        )


@pytest.mark.parametrize(
    "restriction",
    (
        np.array([1.0, 0.0]),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, np.nan]]),
    ),
)
def test_invalid_restrictions_are_rejected(restriction: np.ndarray) -> None:
    design, response = _logit_fixture()
    unrestricted = solve_unrestricted_logit(design, response)

    with pytest.raises(MalformedInputError, match="full row rank"):
        solve_restricted_logit(
            design,
            response,
            restriction,
            initial=unrestricted.coefficients,
        )
    with pytest.raises(MalformedInputError, match="full row rank"):
        solve_restricted_linear(design, response, restriction)
