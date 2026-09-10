"""Layer 1 transforms, hierarchy density, starts, and marginal fitting."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import log

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import digamma, gammaln, logsumexp, polygamma

from es_index_explorer.question_analysis.errors import (
    Layer1InitialiserSensitiveError,
    Layer1ModeError,
    MalformedInputError,
    NumericalError,
)

ALR_EPSILON = 0.5
PHI_MINIMUM = 0.1
PHI_MAXIMUM = 1_000.0
REFERENCE_DATASET = "emc2_set1"
OUTER_OBJECTIVE_TOLERANCE = 1e-6
OUTER_GRADIENT_TOLERANCE = 1e-4
OUTER_MAX_ITERATIONS = 200
INNER_GRADIENT_TOLERANCE = 1e-8
INNER_OBJECTIVE_TOLERANCE = 1e-10
INNER_MAX_ITERATIONS = 100
INNER_ARMIJO_COEFFICIENT = 1e-4
INNER_ARMIJO_ROUNDOFF_TOLERANCE = 1e-12
INNER_MAX_HALVINGS = 20
CURVATURE_RELATIVE_TOLERANCE = 1e-10
START_OBJECTIVE_TOLERANCE = 1e-6
START_PARAMETER_TOLERANCE = 1e-5
OUTER_DIFFERENCE_STEP = 1e-5
OUTER_DIFFERENCE_MAX_HALVINGS = 12
LOG_2PI = log(2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class Layer1VariantData:
    """Store one variant's raw trace counts and arm identities."""

    variant_id: str
    counts: np.ndarray
    arm_ids: tuple[str, ...]
    stages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Layer1RubricData:
    """Store one rubric's variants and fixed hierarchy design values."""

    rubric_id: str
    rubric_order: int
    dataset: str
    expectation_count: int
    variants: tuple[Layer1VariantData, ...]


@dataclass(frozen=True, slots=True)
class Layer1Dataset:
    """Store the complete ordered Layer 1 fitting population."""

    rubrics: tuple[Layer1RubricData, ...]
    dataset_levels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Layer1Hyperparameters:
    """Store natural-scale Layer 1 hyperparameters."""

    phi: float
    mu0: np.ndarray
    dataset_offsets: Mapping[str, np.ndarray]
    sigma_within: np.ndarray
    sigma_between: np.ndarray


@dataclass(frozen=True, slots=True)
class Layer1MomentStart:
    """Store method-of-moments hyperparameters and latent starts."""

    hyperparameters: Layer1Hyperparameters
    rubric_latent_starts: Mapping[str, np.ndarray]
    variant_alr: Mapping[str, np.ndarray]
    smoothed_variant_count: int
    retained_phi_components: int


@dataclass(frozen=True, slots=True)
class Layer1Mode:
    """Store a converged conditional mode and observed curvature."""

    value: np.ndarray
    log_density: float
    gradient_maximum: float
    curvature: np.ndarray
    covariance: np.ndarray
    iterations: int


@dataclass(frozen=True, slots=True)
class Layer1StartFit:
    """Store one marginal-likelihood optimizer result."""

    phi_multiplier: float
    vector: np.ndarray
    hyperparameters: Layer1Hyperparameters
    objective: float
    gradient_maximum: float
    iterations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class Layer1MmlFit:
    """Store an accepted three-start marginal maximum-likelihood fit."""

    hyperparameters: Layer1Hyperparameters
    vector: np.ndarray
    objective: float
    starts: tuple[Layer1StartFit, ...]


def alr_from_counts(
    counts: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Transform a three-component count vector to FAIL-reference ALR coordinates."""
    values = np.asarray(counts, dtype=float)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or float(values.sum()) <= 0
    ):
        raise MalformedInputError(
            "ALR counts must be a finite non-negative three-vector"
        )
    smoothed = bool(np.any(values == 0))
    transformed = values + ALR_EPSILON if smoothed else values
    probabilities = transformed / transformed.sum()
    return (
        np.array(
            [
                np.log(probabilities[0] / probabilities[1]),
                np.log(probabilities[2] / probabilities[1]),
            ],
            dtype=float,
        ),
        smoothed,
    )


def inverse_alr(eta: Sequence[float] | np.ndarray) -> np.ndarray:
    """Back-transform FAIL-reference ALR coordinates to simplex probabilities."""
    values = np.asarray(eta, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise MalformedInputError("ALR coordinates must be a finite two-vector")
    logits = np.array([values[0], 0.0, values[1]], dtype=float)
    return np.exp(logits - logsumexp(logits))


def log_cholesky_to_covariance(
    parameters: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Map three unconstrained log-Cholesky parameters to a 2x2 covariance."""
    values = np.asarray(parameters, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise MalformedInputError(
            "Log-Cholesky parameters must be a finite three-vector"
        )
    if abs(values[0]) > 350 or abs(values[2]) > 350:
        raise NumericalError("Log-Cholesky diagonal is outside finite numerical range")
    lower = np.array(
        [[np.exp(values[0]), 0.0], [values[1], np.exp(values[2])]],
        dtype=float,
    )
    covariance = lower @ lower.T
    if not np.isfinite(covariance).all():
        raise NumericalError("Log-Cholesky covariance is non-finite")
    return covariance


def covariance_to_log_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Map a finite positive-definite 2x2 covariance to log-Cholesky parameters."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise MalformedInputError("Covariance must be a finite 2x2 matrix")
    try:
        lower = np.linalg.cholesky((matrix + matrix.T) / 2.0)
    except np.linalg.LinAlgError as error:
        raise NumericalError("Covariance is not positive definite") from error
    return np.array([np.log(lower[0, 0]), lower[1, 0], np.log(lower[1, 1])])


def validate_layer1_dataset(
    data: Layer1Dataset, *, require_balance: bool = True
) -> None:
    """Validate Layer 1 identities, dimensions, raw counts, and arm balance."""
    if not data.rubrics or REFERENCE_DATASET not in data.dataset_levels:
        raise MalformedInputError("Layer 1 data must include the reference dataset")
    rubric_ids: set[str] = set()
    variant_ids: set[str] = set()
    for rubric in data.rubrics:
        if (
            not rubric.rubric_id
            or rubric.rubric_id in rubric_ids
            or rubric.dataset not in data.dataset_levels
            or rubric.expectation_count <= 0
            or not rubric.variants
        ):
            raise MalformedInputError("Layer 1 rubric contract is invalid")
        rubric_ids.add(rubric.rubric_id)
        for variant in rubric.variants:
            counts = np.asarray(variant.counts, dtype=float)
            if (
                not variant.variant_id
                or variant.variant_id in variant_ids
                or counts.ndim != 2
                or counts.shape[1] != 3
                or len(variant.arm_ids) != len(counts)
                or (variant.stages and len(variant.stages) != len(counts))
                or not np.isfinite(counts).all()
                or np.any(counts < 0)
                or not np.all(counts.sum(axis=1) == rubric.expectation_count)
                or len(set(variant.arm_ids)) != len(variant.arm_ids)
            ):
                raise MalformedInputError("Layer 1 variant contract is invalid")
            if require_balance and len(variant.arm_ids) != 28:
                raise MalformedInputError(
                    "Layer 1 requires exactly 28 arms per variant"
                )
            variant_ids.add(variant.variant_id)


def _sample_covariance(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    return centered.T @ centered / (len(values) - 1)


def method_of_moments_start(data: Layer1Dataset) -> Layer1MomentStart:
    """Compute the frozen Layer 1 method-of-moments initialization."""
    validate_layer1_dataset(data, require_balance=False)
    variant_alr: dict[str, np.ndarray] = {}
    rubric_means: dict[str, np.ndarray] = {}
    smoothed = 0
    phi_estimates: list[float] = []
    for rubric in data.rubrics:
        coordinates: list[np.ndarray] = []
        for variant in rubric.variants:
            aggregate = np.asarray(variant.counts, dtype=float).sum(axis=0)
            eta, used_smoothing = alr_from_counts(aggregate)
            smoothed += int(used_smoothing)
            variant_alr[variant.variant_id] = eta
            coordinates.append(eta)
            trace_count = len(variant.counts)
            means = aggregate / (trace_count * rubric.expectation_count)
            if trace_count > 1:
                variances = np.var(variant.counts, axis=0, ddof=1)
                for component in range(3):
                    theta = float(means[component])
                    denominator = rubric.expectation_count * theta * (1.0 - theta)
                    if not 0.0 < theta < 1.0 or denominator <= 0:
                        continue
                    overdispersion = float(variances[component] / denominator)
                    if 1.0 < overdispersion < rubric.expectation_count:
                        phi_estimates.append(
                            (rubric.expectation_count - overdispersion)
                            / (overdispersion - 1.0)
                        )
        rubric_means[rubric.rubric_id] = np.mean(coordinates, axis=0)
    reference_values = np.stack(
        [
            rubric_means[rubric.rubric_id]
            for rubric in data.rubrics
            if rubric.dataset == REFERENCE_DATASET
        ]
    )
    mu0 = reference_values.mean(axis=0)
    offsets: dict[str, np.ndarray] = {REFERENCE_DATASET: np.zeros(2)}
    for dataset in data.dataset_levels:
        if dataset == REFERENCE_DATASET:
            continue
        dataset_values = np.stack(
            [
                rubric_means[rubric.rubric_id]
                for rubric in data.rubrics
                if rubric.dataset == dataset
            ]
        )
        offsets[dataset] = dataset_values.mean(axis=0) - mu0
    within_sum = np.zeros((2, 2), dtype=float)
    within_degrees = 0
    between_values: list[np.ndarray] = []
    latent_starts: dict[str, np.ndarray] = {}
    for rubric in data.rubrics:
        rubric_mean = rubric_means[rubric.rubric_id]
        rubric_coordinates = np.stack(
            [variant_alr[variant.variant_id] for variant in rubric.variants]
        )
        deviations = rubric_coordinates - rubric_mean
        within_sum += deviations.T @ deviations
        within_degrees += len(rubric.variants) - 1
        between_values.append(rubric_mean - mu0 - offsets[rubric.dataset])
        latent_starts[rubric.rubric_id] = np.concatenate(
            (rubric_mean, rubric_coordinates.ravel())
        )
    if within_degrees <= 0 or len(between_values) <= 1:
        raise NumericalError("Layer 1 moments require estimable covariance degrees")
    sigma_within = within_sum / within_degrees
    sigma_between = _sample_covariance(np.stack(between_values))
    covariance_to_log_cholesky(sigma_within)
    covariance_to_log_cholesky(sigma_between)
    phi = (
        float(np.clip(np.median(phi_estimates), PHI_MINIMUM, PHI_MAXIMUM))
        if phi_estimates
        else PHI_MINIMUM
    )
    return Layer1MomentStart(
        hyperparameters=Layer1Hyperparameters(
            phi=phi,
            mu0=mu0,
            dataset_offsets=offsets,
            sigma_within=sigma_within,
            sigma_between=sigma_between,
        ),
        rubric_latent_starts=latent_starts,
        variant_alr=variant_alr,
        smoothed_variant_count=smoothed,
        retained_phi_components=len(phi_estimates),
    )


def pack_hyperparameters(
    hyperparameters: Layer1Hyperparameters,
    dataset_levels: Sequence[str],
) -> np.ndarray:
    """Pack natural-scale hyperparameters into the frozen unconstrained vector."""
    if hyperparameters.phi <= 0 or not np.isfinite(hyperparameters.phi):
        raise MalformedInputError("Layer 1 phi must be finite and positive")
    values: list[float] = [np.log(hyperparameters.phi), *hyperparameters.mu0]
    for dataset in sorted(set(dataset_levels) - {REFERENCE_DATASET}):
        offset = np.asarray(hyperparameters.dataset_offsets[dataset], dtype=float)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise MalformedInputError("Layer 1 dataset offset is invalid")
        values.extend(offset)
    values.extend(covariance_to_log_cholesky(hyperparameters.sigma_within))
    values.extend(covariance_to_log_cholesky(hyperparameters.sigma_between))
    return np.asarray(values, dtype=float)


def unpack_hyperparameters(
    vector: Sequence[float] | np.ndarray,
    dataset_levels: Sequence[str],
) -> Layer1Hyperparameters:
    """Unpack the frozen unconstrained hyperparameter vector."""
    datasets = tuple(sorted(set(dataset_levels) - {REFERENCE_DATASET}))
    values = np.asarray(vector, dtype=float)
    expected = 1 + 2 + 2 * len(datasets) + 3 + 3
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise MalformedInputError("Layer 1 hyperparameter vector has wrong shape")
    if abs(values[0]) > 700:
        raise NumericalError("Layer 1 log-phi is outside finite numerical range")
    cursor = 3
    offsets: dict[str, np.ndarray] = {REFERENCE_DATASET: np.zeros(2)}
    for dataset in datasets:
        offsets[dataset] = values[cursor : cursor + 2].copy()
        cursor += 2
    return Layer1Hyperparameters(
        phi=float(np.exp(values[0])),
        mu0=values[1:3].copy(),
        dataset_offsets=offsets,
        sigma_within=log_cholesky_to_covariance(values[cursor : cursor + 3]),
        sigma_between=log_cholesky_to_covariance(values[cursor + 3 : cursor + 6]),
    )


def dirichlet_multinomial_logpmf(
    counts: np.ndarray,
    eta: Sequence[float] | np.ndarray,
    phi: float,
) -> float:
    """Evaluate the raw-count Dirichlet-multinomial log probability."""
    observations = np.asarray(counts, dtype=float)
    if (
        observations.ndim != 2
        or observations.shape[1] != 3
        or not np.isfinite(observations).all()
        or np.any(observations < 0)
        or phi <= 0
        or not np.isfinite(phi)
    ):
        raise MalformedInputError("Dirichlet-multinomial inputs are invalid")
    theta = inverse_alr(eta)
    alpha = phi * theta
    totals = observations.sum(axis=1)
    values = (
        gammaln(totals + 1.0)
        - gammaln(observations + 1.0).sum(axis=1)
        + gammaln(phi)
        - gammaln(totals + phi)
        + (gammaln(observations + alpha) - gammaln(alpha)).sum(axis=1)
    )
    return float(values.sum())


def _softmax_derivatives(
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = inverse_alr(eta)
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


def _variant_likelihood_derivatives(
    counts: np.ndarray,
    eta: np.ndarray,
    phi: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    observations = np.asarray(counts, dtype=float)
    theta, jacobian, theta_hessians = _softmax_derivatives(eta)
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
    hessian += np.tensordot(first_theta, theta_hessians, axes=(0, 0))
    return (
        dirichlet_multinomial_logpmf(observations, eta, phi),
        gradient,
        hessian,
    )


def _normal_log_density(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise NumericalError("Layer 1 Gaussian covariance is not positive definite")
    difference = value - mean
    return float(
        -0.5
        * (
            len(value) * LOG_2PI
            + log_determinant
            + difference @ np.linalg.solve(covariance, difference)
        )
    )


def rubric_log_density_derivatives(
    rubric: Layer1RubricData,
    hyperparameters: Layer1Hyperparameters,
    latent: Sequence[float] | np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate one rubric's joint conditional log density, gradient, and Hessian."""
    values = np.asarray(latent, dtype=float)
    dimension = 2 * (len(rubric.variants) + 1)
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise MalformedInputError("Layer 1 latent vector has wrong shape")
    mu = values[:2]
    etas = values[2:].reshape(len(rubric.variants), 2)
    try:
        within_precision = np.linalg.inv(hyperparameters.sigma_within)
        between_precision = np.linalg.inv(hyperparameters.sigma_between)
    except np.linalg.LinAlgError as error:
        raise NumericalError("Layer 1 Gaussian covariance is singular") from error
    dataset_mean = hyperparameters.mu0 + hyperparameters.dataset_offsets[rubric.dataset]
    log_density = _normal_log_density(mu, dataset_mean, hyperparameters.sigma_between)
    gradient = np.zeros(dimension)
    hessian = np.zeros((dimension, dimension))
    difference_between = mu - dataset_mean
    gradient[:2] -= between_precision @ difference_between
    hessian[:2, :2] -= between_precision
    for index, (variant, eta) in enumerate(zip(rubric.variants, etas, strict=True)):
        likelihood, likelihood_gradient, likelihood_hessian = (
            _variant_likelihood_derivatives(variant.counts, eta, hyperparameters.phi)
        )
        eta_slice = slice(2 + 2 * index, 4 + 2 * index)
        difference = eta - mu
        log_density += likelihood + _normal_log_density(
            eta, mu, hyperparameters.sigma_within
        )
        gradient[eta_slice] += likelihood_gradient - within_precision @ difference
        gradient[:2] += within_precision @ difference
        hessian[eta_slice, eta_slice] += likelihood_hessian - within_precision
        hessian[:2, :2] -= within_precision
        hessian[eta_slice, :2] += within_precision
        hessian[:2, eta_slice] += within_precision
    return float(log_density), gradient, hessian


def _accepted_curvature(hessian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curvature = -(hessian + hessian.T) / 2.0
    if not np.isfinite(curvature).all():
        raise Layer1ModeError("Layer 1 observed curvature is non-finite")
    eigenvalues = np.linalg.eigvalsh(curvature)
    threshold = CURVATURE_RELATIVE_TOLERANCE * max(1.0, float(eigenvalues[-1]))
    if float(eigenvalues[0]) <= threshold:
        raise Layer1ModeError("Layer 1 observed curvature is not positive definite")
    try:
        np.linalg.cholesky(curvature)
        covariance = np.linalg.inv(curvature)
    except np.linalg.LinAlgError as error:
        raise Layer1ModeError("Layer 1 observed curvature is non-computable") from error
    return curvature, covariance


def find_rubric_mode(
    rubric: Layer1RubricData,
    hyperparameters: Layer1Hyperparameters,
    initial: Sequence[float] | np.ndarray,
) -> Layer1Mode:
    """Find one conditional mode using the frozen Newton-Armijo contract."""
    value = np.asarray(initial, dtype=float).copy()
    previous_log_density: float | None = None
    for iteration in range(1, INNER_MAX_ITERATIONS + 1):
        log_density, gradient, hessian = rubric_log_density_derivatives(
            rubric, hyperparameters, value
        )
        gradient_maximum = float(np.max(np.abs(gradient)))
        if (
            gradient_maximum < INNER_GRADIENT_TOLERANCE
            and previous_log_density is not None
            and abs(log_density - previous_log_density) < INNER_OBJECTIVE_TOLERANCE
        ):
            curvature, covariance = _accepted_curvature(hessian)
            return Layer1Mode(
                value=value,
                log_density=log_density,
                gradient_maximum=gradient_maximum,
                curvature=curvature,
                covariance=covariance,
                iterations=iteration,
            )
        curvature = -hessian
        try:
            direction = np.linalg.solve(curvature, gradient)
        except np.linalg.LinAlgError as error:
            raise Layer1ModeError("Layer 1 Newton curvature is singular") from error
        directional_derivative = float(gradient @ direction)
        step = 1.0
        accepted: tuple[np.ndarray, float] | None = None
        for _ in range(INNER_MAX_HALVINGS + 1):
            candidate = value + step * direction
            if np.isfinite(candidate).all():
                candidate_density = rubric_log_density_derivatives(
                    rubric, hyperparameters, candidate
                )[0]
                roundoff_slack = INNER_ARMIJO_ROUNDOFF_TOLERANCE * max(
                    1.0, abs(log_density)
                )
                if candidate_density + roundoff_slack >= (
                    log_density
                    + INNER_ARMIJO_COEFFICIENT * step * directional_derivative
                ):
                    accepted = candidate, candidate_density
                    break
            step /= 2.0
        if accepted is None or step < 2.0**-INNER_MAX_HALVINGS:
            raise Layer1ModeError("Layer 1 Newton line search failed")
        previous_log_density = log_density
        value = accepted[0]
    raise Layer1ModeError("Layer 1 Newton mode did not converge")


def laplace_log_marginal(
    data: Layer1Dataset,
    hyperparameters: Layer1Hyperparameters,
    latent_starts: Mapping[str, np.ndarray],
) -> tuple[float, Mapping[str, Layer1Mode]]:
    """Approximate the Layer 1 marginal log likelihood by nested Laplace."""
    modes: dict[str, Layer1Mode] = {}
    total = 0.0
    for rubric in data.rubrics:
        try:
            initial = latent_starts[rubric.rubric_id]
        except KeyError as error:
            raise MalformedInputError("Missing rubric latent start") from error
        mode = find_rubric_mode(rubric, hyperparameters, initial)
        sign, log_determinant = np.linalg.slogdet(mode.curvature)
        if sign <= 0:
            raise Layer1ModeError("Layer 1 Laplace curvature determinant is invalid")
        dimension = len(mode.value)
        total += mode.log_density + 0.5 * dimension * LOG_2PI - 0.5 * log_determinant
        modes[rubric.rubric_id] = mode
    if not np.isfinite(total):
        raise NumericalError("Layer 1 marginal log likelihood is non-finite")
    return total, modes


def fit_mml_start(
    data: Layer1Dataset,
    moments: Layer1MomentStart,
    *,
    phi_multiplier: float,
) -> Layer1StartFit:
    """Fit one frozen Layer 1 marginal-likelihood start."""
    initial_hyperparameters = replace(
        moments.hyperparameters,
        phi=float(
            np.clip(
                moments.hyperparameters.phi * phi_multiplier,
                PHI_MINIMUM,
                PHI_MAXIMUM,
            )
        ),
    )
    initial = pack_hyperparameters(initial_hyperparameters, data.dataset_levels)
    history: list[float] = []
    gradient_failure = False

    def evaluate(vector: np.ndarray) -> float:
        hyperparameters = unpack_hyperparameters(vector, data.dataset_levels)
        return float(
            -laplace_log_marginal(data, hyperparameters, moments.rubric_latent_starts)[
                0
            ]
        )

    numerical_exceptions = (
        NumericalError,
        MalformedInputError,
        FloatingPointError,
        np.linalg.LinAlgError,
    )

    def objective(vector: np.ndarray) -> float:
        try:
            return evaluate(vector)
        except numerical_exceptions:
            displacement = np.clip(vector - initial, -1e20, 1e20)
            return float(1e50 + displacement @ displacement)

    def numerical_gradient(vector: np.ndarray) -> np.ndarray:
        nonlocal gradient_failure
        try:
            baseline = evaluate(vector)
        except numerical_exceptions:
            displacement = np.clip(vector - initial, -1e20, 1e20)
            return 2.0 * displacement
        result = np.empty_like(vector)
        for index in range(len(vector)):
            step = OUTER_DIFFERENCE_STEP * max(1.0, abs(float(vector[index])))
            derivative: float | None = None
            for _ in range(OUTER_DIFFERENCE_MAX_HALVINGS + 1):
                direction = np.zeros_like(vector)
                direction[index] = step
                try:
                    plus = evaluate(vector + direction)
                except numerical_exceptions:
                    plus = None
                try:
                    minus = evaluate(vector - direction)
                except numerical_exceptions:
                    minus = None
                if plus is not None and minus is not None:
                    derivative = (plus - minus) / (2.0 * step)
                    break
                if plus is not None:
                    derivative = (plus - baseline) / step
                    break
                if minus is not None:
                    derivative = (baseline - minus) / step
                    break
                step /= 2.0
            if derivative is None or not np.isfinite(derivative):
                gradient_failure = True
                result[index] = 0.0
            else:
                result[index] = derivative
        return result

    def callback(intermediate: np.ndarray) -> None:
        history.append(objective(intermediate))

    history.append(objective(initial))
    result: OptimizeResult = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=numerical_gradient,
        callback=callback,
        options={
            "maxiter": OUTER_MAX_ITERATIONS,
            "ftol": 1e-12,
            "gtol": OUTER_GRADIENT_TOLERANCE,
            "maxls": 20,
        },
    )
    result_vector = np.asarray(result.x, dtype=float)
    result_gradient = np.asarray(result.jac, dtype=float)
    finite_result = bool(
        result_vector.shape == initial.shape
        and np.isfinite(result_vector).all()
        and result_gradient.shape == initial.shape
        and np.isfinite(result_gradient).all()
        and np.isfinite(result.fun)
    )
    vector = result_vector if finite_result else initial
    final_gradient = result_gradient if finite_result else np.full_like(initial, np.inf)
    gradient_maximum = float(np.max(np.abs(final_gradient)))
    objective_change = abs(history[-1] - history[-2]) if len(history) >= 2 else 0.0
    converged = bool(
        finite_result
        and not gradient_failure
        and result.success
        and int(result.nit) <= OUTER_MAX_ITERATIONS
        and gradient_maximum < OUTER_GRADIENT_TOLERANCE
        and objective_change < OUTER_OBJECTIVE_TOLERANCE
    )
    return Layer1StartFit(
        phi_multiplier=phi_multiplier,
        vector=vector,
        hyperparameters=unpack_hyperparameters(vector, data.dataset_levels),
        objective=float(result.fun),
        gradient_maximum=gradient_maximum,
        iterations=int(result.nit),
        converged=converged,
        message=str(result.message),
    )


def _starts_equivalent(starts: Sequence[Layer1StartFit]) -> bool:
    if len(starts) != 3 or not all(start.converged for start in starts):
        return False
    for first_index, first in enumerate(starts):
        for second in starts[first_index + 1 :]:
            if abs(first.objective - second.objective) > START_OBJECTIVE_TOLERANCE:
                return False
            scale = np.maximum(
                1.0, np.maximum(np.abs(first.vector), np.abs(second.vector))
            )
            if float(np.max(np.abs(first.vector - second.vector) / scale)) > (
                START_PARAMETER_TOLERANCE
            ):
                return False
    return True


def fit_layer1_mml(
    data: Layer1Dataset,
    moments: Layer1MomentStart | None = None,
) -> Layer1MmlFit:
    """Fit and verify the frozen three-start Layer 1 marginal likelihood."""
    start = moments or method_of_moments_start(data)
    fits = tuple(
        fit_mml_start(data, start, phi_multiplier=multiplier)
        for multiplier in (0.5, 1.0, 2.0)
    )
    if not _starts_equivalent(fits):
        raise Layer1InitialiserSensitiveError(
            "Layer 1 starts did not converge to an equivalent optimum"
        )
    selected = fits[1]
    return Layer1MmlFit(
        hyperparameters=selected.hyperparameters,
        vector=selected.vector,
        objective=selected.objective,
        starts=fits,
    )
