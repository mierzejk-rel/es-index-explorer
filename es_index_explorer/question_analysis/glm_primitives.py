"""Numerical score, bread, and Newton-solver primitives."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from es_index_explorer.question_analysis.errors import (
    MalformedInputError,
    NumericalError,
)

DEFAULT_SOLVER_TOLERANCE = 1e-8
DEFAULT_MAX_ITERATIONS = 50


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Store a converged estimating-equation solution."""

    coefficients: np.ndarray
    iterations: int
    max_residual: float
    converged: bool = True


def _validated_inputs(
    design: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    estimating_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(design, dtype=float)
    outcome = np.asarray(response, dtype=float)
    beta = np.asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or outcome.ndim != 1 or beta.ndim != 1:
        raise MalformedInputError(
            "Design, response, and coefficients have invalid dimensions"
        )
    if len(outcome) != matrix.shape[0] or len(beta) != matrix.shape[1]:
        raise MalformedInputError("Design, response, and coefficients are not aligned")
    weights = (
        np.ones(matrix.shape[0], dtype=float)
        if estimating_weights is None
        else np.asarray(estimating_weights, dtype=float)
    )
    if weights.ndim != 1 or len(weights) != matrix.shape[0]:
        raise MalformedInputError("Estimating weights are not aligned with design rows")
    if not all(np.isfinite(value).all() for value in (matrix, outcome, beta, weights)):
        raise MalformedInputError("Numerical inputs must be finite")
    return matrix, outcome, beta, weights


def bernoulli_means(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Compute stable Bernoulli-logit fitted means."""
    matrix = np.asarray(design, dtype=float)
    beta = np.asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or beta.ndim != 1 or matrix.shape[1] != len(beta):
        raise MalformedInputError("Design and coefficients are not aligned")
    if not np.isfinite(matrix).all() or not np.isfinite(beta).all():
        raise MalformedInputError("Design and coefficients must be finite")
    return expit(matrix @ beta)


def bernoulli_score_rows(
    design: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    *,
    estimating_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one Bernoulli-logit score contribution per row."""
    matrix, outcome, beta, weights = _validated_inputs(
        design, response, coefficients, estimating_weights
    )
    if np.any((outcome < 0) | (outcome > 1)):
        raise MalformedInputError("Bernoulli responses must lie in [0, 1]")
    means = expit(matrix @ beta)
    return matrix * (weights * (outcome - means))[:, None]


def bernoulli_bread(
    design: np.ndarray,
    coefficients: np.ndarray,
    *,
    estimating_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Bernoulli-logit information ``X'W(beta)X``."""
    matrix = np.asarray(design, dtype=float)
    beta = np.asarray(coefficients, dtype=float)
    placeholder = np.zeros(matrix.shape[0] if matrix.ndim == 2 else 0, dtype=float)
    matrix, _, beta, weights = _validated_inputs(
        matrix, placeholder, beta, estimating_weights
    )
    means = expit(matrix @ beta)
    diagonal = weights * means * (1.0 - means)
    return matrix.T @ (matrix * diagonal[:, None])


def linear_score_rows(
    design: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    *,
    estimating_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one linear-model score contribution per row."""
    matrix, outcome, beta, weights = _validated_inputs(
        design, response, coefficients, estimating_weights
    )
    return matrix * (weights * (outcome - matrix @ beta))[:, None]


def linear_bread(
    design: np.ndarray, *, estimating_weights: np.ndarray | None = None
) -> np.ndarray:
    """Compute the linear-model score derivative ``X'WX``."""
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise MalformedInputError("Linear design must be a finite matrix")
    weights = (
        np.ones(matrix.shape[0], dtype=float)
        if estimating_weights is None
        else np.asarray(estimating_weights, dtype=float)
    )
    if (
        weights.ndim != 1
        or len(weights) != matrix.shape[0]
        or not np.isfinite(weights).all()
    ):
        raise MalformedInputError("Linear estimating weights are invalid")
    return matrix.T @ (matrix * weights[:, None])


def solve_unrestricted_logit(
    design: np.ndarray,
    response: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    estimating_weights: np.ndarray | None = None,
    tolerance: float = DEFAULT_SOLVER_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> SolverResult:
    """Solve an unrestricted Bernoulli-logit estimating equation by Newton iteration."""
    matrix = np.asarray(design, dtype=float)
    beta = (
        np.zeros(matrix.shape[1], dtype=float)
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    _validated_inputs(matrix, response, beta, estimating_weights)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise NumericalError("Unrestricted Newton bread is singular")
    for iteration in range(max_iterations + 1):
        scores = bernoulli_score_rows(
            matrix, response, beta, estimating_weights=estimating_weights
        ).sum(axis=0)
        residual = float(np.max(np.abs(scores)))
        if residual < tolerance:
            return SolverResult(beta.copy(), iteration, residual)
        if iteration == max_iterations:
            break
        bread = bernoulli_bread(matrix, beta, estimating_weights=estimating_weights)
        try:
            update = np.linalg.solve(bread, scores)
        except np.linalg.LinAlgError as error:
            raise NumericalError("Unrestricted Newton bread is singular") from error
        beta += update
        if not np.isfinite(beta).all():
            raise NumericalError("Unrestricted Newton iteration became non-finite")
    raise NumericalError(
        f"Unrestricted Newton solver did not converge in {max_iterations} iterations"
    )


def solve_restricted_logit(
    design: np.ndarray,
    response: np.ndarray,
    restriction: np.ndarray,
    *,
    initial: np.ndarray,
    estimating_weights: np.ndarray | None = None,
    tolerance: float = DEFAULT_SOLVER_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> SolverResult:
    """Solve ``score(beta)=0`` under ``R beta=0`` by a KKT Newton iteration."""
    matrix = np.asarray(design, dtype=float)
    beta = np.asarray(initial, dtype=float).copy()
    constraints = np.asarray(restriction, dtype=float)
    _validated_inputs(matrix, response, beta, estimating_weights)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise NumericalError("Restricted Newton bread is singular")
    if (
        constraints.ndim != 2
        or constraints.shape[1] != matrix.shape[1]
        or constraints.shape[0] == 0
        or not np.isfinite(constraints).all()
        or np.linalg.matrix_rank(constraints) != constraints.shape[0]
    ):
        raise MalformedInputError("Restriction matrix must have full row rank")

    for iteration in range(max_iterations + 1):
        score = bernoulli_score_rows(
            matrix, response, beta, estimating_weights=estimating_weights
        ).sum(axis=0)
        try:
            multiplier = np.linalg.solve(
                constraints @ constraints.T, constraints @ score
            )
        except np.linalg.LinAlgError as error:
            raise NumericalError("Restricted multiplier system is singular") from error
        lagrangian_residual = score - constraints.T @ multiplier
        constraint_residual = constraints @ beta
        residual = float(
            max(
                np.max(np.abs(lagrangian_residual)),
                np.max(np.abs(constraint_residual)),
            )
        )
        if residual < tolerance:
            return SolverResult(beta.copy(), iteration, residual)
        if iteration == max_iterations:
            break

        bread = bernoulli_bread(matrix, beta, estimating_weights=estimating_weights)
        zeros = np.zeros((constraints.shape[0], constraints.shape[0]), dtype=float)
        kkt = np.block([[bread, constraints.T], [constraints, zeros]])
        right_hand_side = np.concatenate((score, -constraint_residual))
        try:
            update = np.linalg.solve(kkt, right_hand_side)[: matrix.shape[1]]
        except np.linalg.LinAlgError as error:
            raise NumericalError("Restricted KKT system is singular") from error
        beta += update
        if not np.isfinite(beta).all():
            raise NumericalError("Restricted Newton iteration became non-finite")
    raise NumericalError(
        f"Restricted Newton solver did not converge in {max_iterations} iterations"
    )


def solve_unrestricted_linear(design: np.ndarray, response: np.ndarray) -> SolverResult:
    """Solve the unrestricted linear score equation exactly."""
    matrix = np.asarray(design, dtype=float)
    outcome = np.asarray(response, dtype=float)
    initial = np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0)
    _validated_inputs(matrix, outcome, initial)
    try:
        coefficients = np.linalg.solve(linear_bread(matrix), matrix.T @ outcome)
    except np.linalg.LinAlgError as error:
        raise NumericalError("Linear bread is singular") from error
    residual = float(
        np.max(np.abs(linear_score_rows(matrix, outcome, coefficients).sum(axis=0)))
    )
    return SolverResult(coefficients, 1, residual)


def solve_restricted_linear(
    design: np.ndarray,
    response: np.ndarray,
    restriction: np.ndarray,
) -> SolverResult:
    """Solve a linear score equation under ``R beta=0`` with a KKT system."""
    unrestricted = solve_unrestricted_linear(design, response)
    matrix = np.asarray(design, dtype=float)
    outcome = np.asarray(response, dtype=float)
    constraints = np.asarray(restriction, dtype=float)
    if (
        constraints.ndim != 2
        or constraints.shape[1] != matrix.shape[1]
        or not np.isfinite(constraints).all()
        or np.linalg.matrix_rank(constraints) != constraints.shape[0]
    ):
        raise MalformedInputError("Linear restriction matrix must have full row rank")
    bread = linear_bread(matrix)
    zeros = np.zeros((constraints.shape[0], constraints.shape[0]), dtype=float)
    kkt = np.block([[bread, constraints.T], [constraints, zeros]])
    right_hand_side = np.concatenate(
        (matrix.T @ outcome, np.zeros(constraints.shape[0]))
    )
    try:
        coefficients = np.linalg.solve(kkt, right_hand_side)[: matrix.shape[1]]
    except np.linalg.LinAlgError as error:
        raise NumericalError("Restricted linear KKT system is singular") from error
    score = linear_score_rows(matrix, outcome, coefficients).sum(axis=0)
    multiplier = np.linalg.solve(constraints @ constraints.T, constraints @ score)
    residual = float(
        max(
            np.max(np.abs(score - constraints.T @ multiplier)),
            np.max(np.abs(constraints @ coefficients)),
        )
    )
    return SolverResult(coefficients, unrestricted.iterations, residual)
