"""Layer 1 transforms, hierarchy density, starts, and marginal fitting."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import log

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import digamma, gammaln, polygamma

from es_index_explorer.question_analysis.errors import (
    Layer1InitialiserSensitiveError,
    Layer1ModeError,
    MalformedInputError,
    NumericalError,
)
from es_index_explorer.question_analysis.layer1_execution import (
    SequentialExecutor,
    SpawnProcessExecutor,
    TaskReference,
    WorkUnit,
    WorkUnitKind,
)
from es_index_explorer.question_analysis.layer1_packed import (
    CompressedPfuCounts,
    compress_pfu_counts,
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
OBJECTIVE_CACHE_MAX_ENTRIES = 256
LOG_2PI = log(2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class Layer1VariantData:
    """Store one variant's raw trace counts and arm identities."""

    variant_id: str
    counts: np.ndarray
    arm_ids: tuple[str, ...]
    stages: tuple[str, ...] = ()
    compressed_counts: CompressedPfuCounts | None = None


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
class HyperparameterEvaluationContext:
    """Store reusable covariance algebra for one exact hyperparameter vector."""

    hyperparameters: Layer1Hyperparameters
    within_precision: np.ndarray
    between_precision: np.ndarray
    within_log_determinant: float
    between_log_determinant: float


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


@dataclass(frozen=True, slots=True)
class MarginalEvaluationPayload:
    """Store one pickleable exact marginal-objective evaluation request."""

    data: Layer1Dataset
    latent_starts: Mapping[str, np.ndarray]
    vector: np.ndarray


ModelTaskExecutor = SequentialExecutor | SpawnProcessExecutor
MmlIterationCallback = Callable[[float, int, float, float], None]
MmlStartCallback = Callable[[Layer1StartFit], None]


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
    return inverse_alr_batch(values)


def inverse_alr_batch(eta: np.ndarray) -> np.ndarray:
    """Back-transform an arbitrary batch of finite FAIL-reference ALR coordinates."""
    values = np.asarray(eta, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 2 or not np.isfinite(values).all():
        raise MalformedInputError("Batched ALR coordinates must end in dimension two")
    maximum = np.maximum(np.maximum(values[..., 0], 0.0), values[..., 1])
    pass_weight = np.exp(values[..., 0] - maximum)
    fail_weight = np.exp(-maximum)
    undetermined_weight = np.exp(values[..., 1] - maximum)
    denominator = pass_weight + fail_weight + undetermined_weight
    return np.stack(
        (
            pass_weight / denominator,
            fail_weight / denominator,
            undetermined_weight / denominator,
        ),
        axis=-1,
    )


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
            if (
                variant.compressed_counts is not None
                and variant.compressed_counts.trace_count != len(counts)
            ):
                raise MalformedInputError(
                    "Compressed Layer 1 counts do not match raw trace count"
                )
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
    counts: np.ndarray | CompressedPfuCounts,
    eta: Sequence[float] | np.ndarray,
    phi: float,
) -> float:
    """Evaluate the raw-count Dirichlet-multinomial log probability."""
    compressed = (
        counts
        if isinstance(counts, CompressedPfuCounts)
        else compress_pfu_counts(counts)
    )
    if phi <= 0 or not np.isfinite(phi):
        raise MalformedInputError("Dirichlet-multinomial inputs are invalid")
    observations = compressed.patterns.astype(float, copy=False)
    multiplicities = compressed.multiplicities.astype(float, copy=False)
    theta = inverse_alr(eta)
    alpha = phi * theta
    totals = observations.sum(axis=1)
    values = (
        compressed.combinatorial_log_terms
        + gammaln(phi)
        - gammaln(totals + phi)
        + (gammaln(observations + alpha) - gammaln(alpha)).sum(axis=1)
    )
    return float(values @ multiplicities)


def _softmax_derivatives(
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = inverse_alr(eta)
    indicator = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    centered = indicator - theta[[0, 2]]
    jacobian = theta[:, None] * centered
    selected_jacobian = jacobian[[0, 2]]
    hessians = theta[:, None, None] * (
        centered[:, :, None] * centered[:, None, :] - selected_jacobian[None, :, :]
    )
    return theta, jacobian, hessians


def _variant_likelihood_derivatives(
    counts: np.ndarray | CompressedPfuCounts,
    eta: np.ndarray,
    phi: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    compressed = (
        counts
        if isinstance(counts, CompressedPfuCounts)
        else compress_pfu_counts(counts)
    )
    observations = compressed.patterns.astype(float, copy=False)
    multiplicities = compressed.multiplicities.astype(float, copy=False)
    theta, jacobian, theta_hessians = _softmax_derivatives(eta)
    alpha = phi * theta
    first_terms = phi * (digamma(observations + alpha) - digamma(alpha))
    second_terms = phi**2 * (polygamma(1, observations + alpha) - polygamma(1, alpha))
    first_theta = multiplicities @ first_terms
    second_theta = multiplicities @ second_terms
    gradient = jacobian.T @ first_theta
    hessian = jacobian.T @ np.diag(second_theta) @ jacobian
    hessian += np.tensordot(first_theta, theta_hessians, axes=(0, 0))
    return (
        dirichlet_multinomial_logpmf(compressed, eta, phi),
        gradient,
        hessian,
    )


def _normal_log_density(
    value: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
    log_determinant: float,
) -> float:
    difference = value - mean
    return float(
        -0.5
        * (len(value) * LOG_2PI + log_determinant + difference @ precision @ difference)
    )


def build_hyperparameter_context(
    hyperparameters: Layer1Hyperparameters,
) -> HyperparameterEvaluationContext:
    """Factor reusable Gaussian covariance terms once for an exact psi."""
    within = np.asarray(hyperparameters.sigma_within, dtype=float)
    between = np.asarray(hyperparameters.sigma_between, dtype=float)
    try:
        within_cholesky = np.linalg.cholesky(within)
        between_cholesky = np.linalg.cholesky(between)
        within_precision = np.linalg.solve(within, np.eye(2))
        between_precision = np.linalg.solve(between, np.eye(2))
    except np.linalg.LinAlgError as error:
        raise NumericalError("Layer 1 Gaussian covariance is singular") from error
    return HyperparameterEvaluationContext(
        hyperparameters=hyperparameters,
        within_precision=within_precision,
        between_precision=between_precision,
        within_log_determinant=float(2.0 * np.log(np.diag(within_cholesky)).sum()),
        between_log_determinant=float(2.0 * np.log(np.diag(between_cholesky)).sum()),
    )


def rubric_log_density(
    rubric: Layer1RubricData,
    context: HyperparameterEvaluationContext,
    latent: Sequence[float] | np.ndarray,
) -> float:
    """Evaluate one rubric conditional log density without unused derivatives."""
    values = np.asarray(latent, dtype=float)
    dimension = 2 * (len(rubric.variants) + 1)
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise MalformedInputError("Layer 1 latent vector has wrong shape")
    hyperparameters = context.hyperparameters
    mu = values[:2]
    etas = values[2:].reshape(len(rubric.variants), 2)
    dataset_mean = hyperparameters.mu0 + hyperparameters.dataset_offsets[rubric.dataset]
    total = _normal_log_density(
        mu,
        dataset_mean,
        context.between_precision,
        context.between_log_determinant,
    )
    for variant, eta in zip(rubric.variants, etas, strict=True):
        total += dirichlet_multinomial_logpmf(
            variant.compressed_counts or variant.counts,
            eta,
            hyperparameters.phi,
        )
        total += _normal_log_density(
            eta,
            mu,
            context.within_precision,
            context.within_log_determinant,
        )
    return float(total)


def rubric_log_density_batch(
    rubric: Layer1RubricData,
    context: HyperparameterEvaluationContext,
    latent: np.ndarray,
) -> np.ndarray:
    """Evaluate a batch of rubric conditional log densities without derivatives."""
    values = np.asarray(latent, dtype=float)
    dimension = 2 * (len(rubric.variants) + 1)
    if (
        values.ndim != 2
        or values.shape[1] != dimension
        or not np.isfinite(values).all()
    ):
        raise MalformedInputError("Layer 1 latent batch has wrong shape")
    hyperparameters = context.hyperparameters
    mu = values[:, :2]
    etas = values[:, 2:].reshape(len(values), len(rubric.variants), 2)
    dataset_mean = hyperparameters.mu0 + hyperparameters.dataset_offsets[rubric.dataset]
    difference_between = mu - dataset_mean
    total = -0.5 * (
        2 * LOG_2PI
        + context.between_log_determinant
        + np.einsum(
            "ni,ij,nj->n",
            difference_between,
            context.between_precision,
            difference_between,
        )
    )
    for variant_index, variant in enumerate(rubric.variants):
        eta = etas[:, variant_index]
        probabilities = inverse_alr_batch(eta)
        alpha = hyperparameters.phi * probabilities
        compressed = variant.compressed_counts or compress_pfu_counts(variant.counts)
        patterns = compressed.patterns.astype(float, copy=False)
        multiplicities = compressed.multiplicities.astype(float, copy=False)
        totals = patterns.sum(axis=1)
        likelihood_terms = (
            compressed.combinatorial_log_terms[None, :]
            + gammaln(hyperparameters.phi)
            - gammaln(totals[None, :] + hyperparameters.phi)
            + (
                gammaln(patterns[None, :, :] + alpha[:, None, :])
                - gammaln(alpha[:, None, :])
            ).sum(axis=2)
        )
        total += likelihood_terms @ multiplicities
        difference = eta - mu
        total += -0.5 * (
            2 * LOG_2PI
            + context.within_log_determinant
            + np.einsum(
                "ni,ij,nj->n",
                difference,
                context.within_precision,
                difference,
            )
        )
    return np.asarray(total, dtype=float)


def rubric_log_density_derivatives(
    rubric: Layer1RubricData,
    hyperparameters: Layer1Hyperparameters,
    latent: Sequence[float] | np.ndarray,
    *,
    context: HyperparameterEvaluationContext | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate one rubric's joint conditional log density, gradient, and Hessian."""
    values = np.asarray(latent, dtype=float)
    dimension = 2 * (len(rubric.variants) + 1)
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise MalformedInputError("Layer 1 latent vector has wrong shape")
    mu = values[:2]
    etas = values[2:].reshape(len(rubric.variants), 2)
    selected_context = context or build_hyperparameter_context(hyperparameters)
    within_precision = selected_context.within_precision
    between_precision = selected_context.between_precision
    dataset_mean = hyperparameters.mu0 + hyperparameters.dataset_offsets[rubric.dataset]
    log_density = _normal_log_density(
        mu,
        dataset_mean,
        between_precision,
        selected_context.between_log_determinant,
    )
    gradient = np.zeros(dimension)
    hessian = np.zeros((dimension, dimension))
    difference_between = mu - dataset_mean
    gradient[:2] -= between_precision @ difference_between
    hessian[:2, :2] -= between_precision
    for index, (variant, eta) in enumerate(zip(rubric.variants, etas, strict=True)):
        likelihood, likelihood_gradient, likelihood_hessian = (
            _variant_likelihood_derivatives(
                variant.compressed_counts or variant.counts,
                eta,
                hyperparameters.phi,
            )
        )
        eta_slice = slice(2 + 2 * index, 4 + 2 * index)
        difference = eta - mu
        log_density += likelihood + _normal_log_density(
            eta,
            mu,
            within_precision,
            selected_context.within_log_determinant,
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


def solve_arrowhead_curvature(
    curvature: np.ndarray,
    right_hand_side: np.ndarray,
    variant_count: int,
) -> np.ndarray:
    """Solve the exact rubric-mean/variant arrowhead system by block elimination."""
    matrix = np.asarray(curvature, dtype=float)
    right = np.asarray(right_hand_side, dtype=float)
    dimension = 2 * (variant_count + 1)
    if (
        variant_count <= 0
        or matrix.shape != (dimension, dimension)
        or right.shape != (dimension,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(right).all()
    ):
        raise MalformedInputError("Layer 1 arrowhead system has invalid dimensions")
    schur = matrix[:2, :2].copy()
    reduced_right = right[:2].copy()
    solved_variant_right: list[np.ndarray] = []
    solved_variant_cross: list[np.ndarray] = []
    for variant_index in range(variant_count):
        variant_slice = slice(2 + 2 * variant_index, 4 + 2 * variant_index)
        block = matrix[variant_slice, variant_slice]
        cross = matrix[:2, variant_slice]
        solved_right = np.linalg.solve(block, right[variant_slice])
        solved_cross = np.linalg.solve(block, cross.T)
        solved_variant_right.append(solved_right)
        solved_variant_cross.append(solved_cross)
        schur -= cross @ solved_cross
        reduced_right -= cross @ solved_right
    rubric_solution = np.linalg.solve(schur, reduced_right)
    solution = np.empty(dimension, dtype=float)
    solution[:2] = rubric_solution
    for variant_index, (solved_right, solved_cross) in enumerate(
        zip(solved_variant_right, solved_variant_cross, strict=True)
    ):
        variant_slice = slice(2 + 2 * variant_index, 4 + 2 * variant_index)
        solution[variant_slice] = solved_right - solved_cross @ rubric_solution
    return solution


def find_rubric_mode(
    rubric: Layer1RubricData,
    hyperparameters: Layer1Hyperparameters,
    initial: Sequence[float] | np.ndarray,
    *,
    context: HyperparameterEvaluationContext | None = None,
) -> Layer1Mode:
    """Find one conditional mode using the frozen Newton-Armijo contract."""
    value = np.asarray(initial, dtype=float).copy()
    selected_context = context or build_hyperparameter_context(hyperparameters)
    previous_log_density: float | None = None
    for iteration in range(1, INNER_MAX_ITERATIONS + 1):
        log_density, gradient, hessian = rubric_log_density_derivatives(
            rubric,
            hyperparameters,
            value,
            context=selected_context,
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
            direction = solve_arrowhead_curvature(
                curvature,
                gradient,
                len(rubric.variants),
            )
        except np.linalg.LinAlgError as error:
            raise Layer1ModeError("Layer 1 Newton curvature is singular") from error
        directional_derivative = float(gradient @ direction)
        step = 1.0
        accepted: tuple[np.ndarray, float] | None = None
        for _ in range(INNER_MAX_HALVINGS + 1):
            candidate = value + step * direction
            if np.isfinite(candidate).all():
                candidate_density = rubric_log_density(
                    rubric, selected_context, candidate
                )
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
    context = build_hyperparameter_context(hyperparameters)
    for rubric in data.rubrics:
        try:
            initial = latent_starts[rubric.rubric_id]
        except KeyError as error:
            raise MalformedInputError("Missing rubric latent start") from error
        mode = find_rubric_mode(
            rubric,
            hyperparameters,
            initial,
            context=context,
        )
        sign, log_determinant = np.linalg.slogdet(mode.curvature)
        if sign <= 0:
            raise Layer1ModeError("Layer 1 Laplace curvature determinant is invalid")
        dimension = len(mode.value)
        total += mode.log_density + 0.5 * dimension * LOG_2PI - 0.5 * log_determinant
        modes[rubric.rubric_id] = mode
    if not np.isfinite(total):
        raise NumericalError("Layer 1 marginal log likelihood is non-finite")
    return total, modes


def evaluate_marginal_work_unit(unit: WorkUnit) -> object:
    """Evaluate one exact marginal objective in a spawn-safe worker."""
    if not isinstance(unit.payload, MarginalEvaluationPayload):
        raise MalformedInputError("Marginal work unit payload is invalid")
    payload = unit.payload
    hyperparameters = unpack_hyperparameters(
        payload.vector, payload.data.dataset_levels
    )
    return -laplace_log_marginal(
        payload.data,
        hyperparameters,
        payload.latent_starts,
    )[0]


def fit_mml_start(
    data: Layer1Dataset,
    moments: Layer1MomentStart,
    *,
    phi_multiplier: float,
    executor: ModelTaskExecutor | None = None,
    iteration_callback: MmlIterationCallback | None = None,
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
    latest_gradient_maximum = float("inf")
    objective_cache: dict[bytes, float] = {}
    cache_order: list[bytes] = []

    def cache_value(vector: np.ndarray, value: float) -> None:
        key = np.ascontiguousarray(vector, dtype=np.float64).tobytes()
        if key in objective_cache:
            return
        objective_cache[key] = value
        cache_order.append(key)
        if len(cache_order) > OBJECTIVE_CACHE_MAX_ENTRIES:
            expired = cache_order.pop(0)
            objective_cache.pop(expired, None)

    def evaluate(vector: np.ndarray) -> float:
        key = np.ascontiguousarray(vector, dtype=np.float64).tobytes()
        cached = objective_cache.get(key)
        if cached is not None:
            return cached
        hyperparameters = unpack_hyperparameters(vector, data.dataset_levels)
        value = float(
            -laplace_log_marginal(data, hyperparameters, moments.rubric_latent_starts)[
                0
            ]
        )
        cache_value(vector, value)
        return value

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
        nonlocal gradient_failure, latest_gradient_maximum
        try:
            baseline = evaluate(vector)
        except numerical_exceptions:
            displacement = np.clip(vector - initial, -1e20, 1e20)
            result = 2.0 * displacement
            latest_gradient_maximum = float(np.max(np.abs(result)))
            return result
        if executor is not None and isinstance(executor, SpawnProcessExecutor):
            result = parallel_numerical_gradient(vector, baseline)
            latest_gradient_maximum = float(np.max(np.abs(result)))
            return result
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
        latest_gradient_maximum = float(np.max(np.abs(result)))
        return result

    def parallel_numerical_gradient(
        vector: np.ndarray,
        baseline: float,
    ) -> np.ndarray:
        nonlocal gradient_failure
        if not isinstance(executor, SpawnProcessExecutor):
            raise TypeError("Parallel gradient requires a process executor")
        result = np.zeros_like(vector)
        unresolved = {
            index: OUTER_DIFFERENCE_STEP * max(1.0, abs(float(vector[index])))
            for index in range(len(vector))
        }
        reference = TaskReference(
            module=__name__,
            function="evaluate_marginal_work_unit",
        )
        for halving in range(OUTER_DIFFERENCE_MAX_HALVINGS + 1):
            units: list[WorkUnit] = []
            candidates: dict[str, np.ndarray] = {}
            for index, step in unresolved.items():
                for side, sign in (("plus", 1.0), ("minus", -1.0)):
                    candidate = vector.copy()
                    candidate[index] += sign * step
                    task_id = f"fd:{index}:{side}:{halving}"
                    candidates[task_id] = candidate
                    units.append(
                        WorkUnit(
                            task_id=task_id,
                            kind=WorkUnitKind.OBJECTIVE_GRADIENT_PERTURBATION,
                            payload=MarginalEvaluationPayload(
                                data=data,
                                latent_starts=moments.rubric_latent_starts,
                                vector=candidate,
                            ),
                        )
                    )
            outcomes = {
                outcome.task_id: outcome
                for outcome in executor.execute(reference, tuple(units))
            }
            next_unresolved: dict[int, float] = {}
            for index, step in unresolved.items():
                plus_id = f"fd:{index}:plus:{halving}"
                minus_id = f"fd:{index}:minus:{halving}"
                plus_outcome = outcomes.get(plus_id)
                minus_outcome = outcomes.get(minus_id)
                plus_value = plus_outcome.value if plus_outcome is not None else None
                minus_value = minus_outcome.value if minus_outcome is not None else None
                plus = (
                    float(plus_value)
                    if plus_outcome is not None
                    and plus_outcome.succeeded
                    and isinstance(plus_value, (int, float, np.floating))
                    else None
                )
                minus = (
                    float(minus_value)
                    if minus_outcome is not None
                    and minus_outcome.succeeded
                    and isinstance(minus_value, (int, float, np.floating))
                    else None
                )
                if plus is not None:
                    cache_value(candidates[plus_id], plus)
                if minus is not None:
                    cache_value(candidates[minus_id], minus)
                if plus is not None and minus is not None:
                    result[index] = (plus - minus) / (2.0 * step)
                elif plus is not None:
                    result[index] = (plus - baseline) / step
                elif minus is not None:
                    result[index] = (baseline - minus) / step
                else:
                    next_unresolved[index] = step / 2.0
            unresolved = next_unresolved
            if not unresolved:
                break
        if unresolved:
            gradient_failure = True
            for index in unresolved:
                result[index] = 0.0
        return result

    def callback(intermediate: np.ndarray) -> None:
        history.append(objective(intermediate))
        if iteration_callback is not None:
            iteration_callback(
                phi_multiplier,
                len(history) - 1,
                history[-1],
                latest_gradient_maximum,
            )

    history.append(objective(initial))
    result: OptimizeResult = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=numerical_gradient,
        callback=callback,
        options={
            "maxiter": OUTER_MAX_ITERATIONS,
            "ftol": 0.0,
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
    *,
    executor: ModelTaskExecutor | None = None,
    completed_starts: Mapping[float, Layer1StartFit] | None = None,
    start_callback: MmlStartCallback | None = None,
    iteration_callback: MmlIterationCallback | None = None,
) -> Layer1MmlFit:
    """Fit and verify the frozen three-start Layer 1 marginal likelihood."""
    start = moments or method_of_moments_start(data)
    existing = completed_starts or {}
    fits_list: list[Layer1StartFit] = []
    for multiplier in (0.5, 1.0, 2.0):
        current = existing.get(multiplier)
        if current is None:
            current = fit_mml_start(
                data,
                start,
                phi_multiplier=multiplier,
                executor=executor,
                iteration_callback=iteration_callback,
            )
            if start_callback is not None:
                start_callback(current)
        fits_list.append(current)
    fits = tuple(fits_list)
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
