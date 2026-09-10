"""Intentionally simple serial references for Layer 1 numerical kernels."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import log

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, polygamma

REFERENCE_DATASET = "emc2_set1"
LOG_2PI = log(2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class ReferenceRubric:
    """Store plain arrays required by the independent dense reference."""

    rubric_id: str
    dataset: str
    variant_counts: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class ReferenceHyperparameters:
    """Store natural-scale hyperparameters for the independent reference."""

    phi: float
    mu0: np.ndarray
    offsets: dict[str, np.ndarray]
    sigma_within: np.ndarray
    sigma_between: np.ndarray


def reference_inverse_alr(eta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Back-transform FAIL-reference ALR coordinates without production helpers."""
    values = np.asarray(eta, dtype=float)
    logits = np.array([values[0], 0.0, values[1]], dtype=float)
    return np.exp(logits - logsumexp(logits))


def reference_alr_from_counts(
    counts: Sequence[float] | np.ndarray,
    *,
    epsilon: float = 0.5,
) -> np.ndarray:
    """Transform counts with the frozen all-component boundary correction."""
    values = np.asarray(counts, dtype=float)
    adjusted = values + epsilon if np.any(values == 0) else values
    probabilities = adjusted / adjusted.sum()
    return np.array(
        [
            np.log(probabilities[0] / probabilities[1]),
            np.log(probabilities[2] / probabilities[1]),
        ]
    )


def reference_dirichlet_multinomial_logpmf(
    counts: np.ndarray,
    eta: Sequence[float] | np.ndarray,
    phi: float,
) -> float:
    """Evaluate the raw-count Dirichlet-multinomial log probability serially."""
    theta = reference_inverse_alr(eta)
    alpha = phi * theta
    total = 0.0
    for observation in np.asarray(counts, dtype=float):
        sample_size = float(observation.sum())
        total += float(
            gammaln(sample_size + 1.0)
            - gammaln(observation + 1.0).sum()
            + gammaln(phi)
            - gammaln(sample_size + phi)
            + (gammaln(observation + alpha) - gammaln(alpha)).sum()
        )
    return total


def _reference_softmax_derivatives(
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = reference_inverse_alr(eta)
    positions = (0, 2)
    jacobian = np.empty((3, 2), dtype=float)
    hessians = np.empty((3, 2, 2), dtype=float)
    for component in range(3):
        for first, first_position in enumerate(positions):
            first_delta = float(component == first_position)
            jacobian[component, first] = theta[component] * (
                first_delta - theta[first_position]
            )
            for second, second_position in enumerate(positions):
                second_delta = float(component == second_position)
                cross_delta = float(first_position == second_position)
                hessians[component, first, second] = theta[component] * (
                    (first_delta - theta[first_position])
                    * (second_delta - theta[second_position])
                    - theta[second_position] * (cross_delta - theta[first_position])
                )
    return theta, jacobian, hessians


def reference_variant_derivatives(
    counts: np.ndarray,
    eta: Sequence[float] | np.ndarray,
    phi: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate one variant likelihood and ALR derivatives serially."""
    coordinates = np.asarray(eta, dtype=float)
    observations = np.asarray(counts, dtype=float)
    theta, jacobian, theta_hessians = _reference_softmax_derivatives(coordinates)
    alpha = phi * theta
    first_theta = np.zeros(3)
    second_theta = np.zeros(3)
    for observation in observations:
        first_theta += phi * (digamma(observation + alpha) - digamma(alpha))
        second_theta += phi**2 * (
            polygamma(1, observation + alpha) - polygamma(1, alpha)
        )
    gradient = jacobian.T @ first_theta
    hessian = jacobian.T @ np.diag(second_theta) @ jacobian
    for component in range(3):
        hessian += first_theta[component] * theta_hessians[component]
    return (
        reference_dirichlet_multinomial_logpmf(observations, coordinates, phi),
        gradient,
        hessian,
    )


def reference_rubric_v2_draws(eta_draws: np.ndarray) -> np.ndarray:
    """Back-transform a draw-by-variant ALR tensor using explicit serial loops."""
    values = np.asarray(eta_draws, dtype=float)
    output = np.empty(values.shape[:2], dtype=float)
    for draw_index in range(values.shape[0]):
        for variant_index in range(values.shape[1]):
            probability = reference_inverse_alr(values[draw_index, variant_index])
            output[draw_index, variant_index] = probability[0] + 0.5 * probability[2]
    return output


def reference_ordered_sum(values: Sequence[float]) -> float:
    """Accumulate floats in their supplied order without tree reduction."""
    total = 0.0
    for value in values:
        total += float(value)
    return total


def reference_unpack_hyperparameters(
    vector: np.ndarray,
    dataset_levels: Sequence[str],
) -> ReferenceHyperparameters:
    """Unpack the frozen vector without production parameterization helpers."""
    values = np.asarray(vector, dtype=float)
    datasets = tuple(sorted(set(dataset_levels) - {REFERENCE_DATASET}))
    cursor = 3
    offsets = {REFERENCE_DATASET: np.zeros(2)}
    for dataset in datasets:
        offsets[dataset] = values[cursor : cursor + 2].copy()
        cursor += 2

    def covariance(parameters: np.ndarray) -> np.ndarray:
        lower = np.array(
            [
                [np.exp(parameters[0]), 0.0],
                [parameters[1], np.exp(parameters[2])],
            ]
        )
        return lower @ lower.T

    return ReferenceHyperparameters(
        phi=float(np.exp(values[0])),
        mu0=values[1:3].copy(),
        offsets=offsets,
        sigma_within=covariance(values[cursor : cursor + 3]),
        sigma_between=covariance(values[cursor + 3 : cursor + 6]),
    )


def _reference_normal(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, np.ndarray]:
    precision = np.linalg.solve(covariance, np.eye(2))
    difference = value - mean
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("Reference covariance is not positive definite")
    log_density = -0.5 * (
        2 * LOG_2PI + log_determinant + difference @ precision @ difference
    )
    return float(log_density), precision


def reference_rubric_density_derivatives(
    rubric: ReferenceRubric,
    hyperparameters: ReferenceHyperparameters,
    latent: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate a rubric density and dense derivatives independently."""
    values = np.asarray(latent, dtype=float)
    variant_count = len(rubric.variant_counts)
    mu = values[:2]
    etas = values[2:].reshape(variant_count, 2)
    dataset_mean = hyperparameters.mu0 + hyperparameters.offsets[rubric.dataset]
    total, between_precision = _reference_normal(
        mu, dataset_mean, hyperparameters.sigma_between
    )
    _, within_precision = _reference_normal(
        np.zeros(2), np.zeros(2), hyperparameters.sigma_within
    )
    gradient = np.zeros_like(values)
    hessian = np.zeros((len(values), len(values)))
    gradient[:2] -= between_precision @ (mu - dataset_mean)
    hessian[:2, :2] -= between_precision
    for index, (counts, eta) in enumerate(
        zip(rubric.variant_counts, etas, strict=True)
    ):
        likelihood, likelihood_gradient, likelihood_hessian = (
            reference_variant_derivatives(counts, eta, hyperparameters.phi)
        )
        normal, _ = _reference_normal(eta, mu, hyperparameters.sigma_within)
        total += likelihood + normal
        difference = eta - mu
        variant_slice = slice(2 + 2 * index, 4 + 2 * index)
        gradient[variant_slice] += likelihood_gradient - within_precision @ difference
        gradient[:2] += within_precision @ difference
        hessian[variant_slice, variant_slice] += likelihood_hessian - within_precision
        hessian[:2, :2] -= within_precision
        hessian[:2, variant_slice] += within_precision
        hessian[variant_slice, :2] += within_precision
    return float(total), gradient, hessian


def reference_find_mode(
    rubric: ReferenceRubric,
    hyperparameters: ReferenceHyperparameters,
    initial: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Find the frozen conditional mode using independent dense Newton algebra."""
    value = np.asarray(initial, dtype=float).copy()
    previous: float | None = None
    for _ in range(100):
        density, gradient, hessian = reference_rubric_density_derivatives(
            rubric, hyperparameters, value
        )
        if (
            np.max(np.abs(gradient)) < 1e-8
            and previous is not None
            and abs(density - previous) < 1e-10
        ):
            curvature = -(hessian + hessian.T) / 2.0
            eigenvalues = np.linalg.eigvalsh(curvature)
            threshold = 1e-10 * max(1.0, float(eigenvalues[-1]))
            if eigenvalues[0] <= threshold:
                raise np.linalg.LinAlgError(
                    "Reference curvature is not positive definite"
                )
            return value, density, curvature
        direction = np.linalg.solve(-hessian, gradient)
        directional_derivative = float(gradient @ direction)
        step = 1.0
        accepted: tuple[np.ndarray, float] | None = None
        for _ in range(21):
            candidate = value + step * direction
            candidate_density = reference_rubric_density_derivatives(
                rubric, hyperparameters, candidate
            )[0]
            slack = 1e-12 * max(1.0, abs(density))
            if candidate_density + slack >= (
                density + 1e-4 * step * directional_derivative
            ):
                accepted = candidate, candidate_density
                break
            step /= 2.0
        if accepted is None or step < 2.0**-20:
            raise np.linalg.LinAlgError("Reference line search failed")
        previous = density
        value = accepted[0]
    raise np.linalg.LinAlgError("Reference mode did not converge")


def reference_laplace_log_marginal(
    vector: np.ndarray,
    dataset_levels: Sequence[str],
    rubrics: Sequence[ReferenceRubric],
    latent_starts: dict[str, np.ndarray],
) -> float:
    """Evaluate the full ordered nested-Laplace objective independently."""
    hyperparameters = reference_unpack_hyperparameters(vector, dataset_levels)
    contributions: list[float] = []
    for rubric in rubrics:
        mode, density, curvature = reference_find_mode(
            rubric,
            hyperparameters,
            latent_starts[rubric.rubric_id],
        )
        sign, log_determinant = np.linalg.slogdet(curvature)
        if sign <= 0:
            raise np.linalg.LinAlgError("Reference curvature determinant is invalid")
        contributions.append(
            density + 0.5 * len(mode) * LOG_2PI - 0.5 * log_determinant
        )
    return reference_ordered_sum(contributions)
