"""Conditional Laplace draws and importance-resampling diagnostics."""

from dataclasses import dataclass

import numpy as np

from es_index_explorer.question_analysis.errors import (
    Layer1ModeError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1Mode,
    Layer1MomentStart,
    Layer1RubricData,
    find_rubric_mode,
    inverse_alr,
    rubric_log_density_derivatives,
)

INNER_DRAWS_PER_OUTER = 40
CONDITIONAL_DRAW_COUNT = 20_000
IMPORTANCE_PARTICLES = 5_000
IMPORTANCE_ESS_RATIO_MINIMUM = 0.10


@dataclass(frozen=True, slots=True)
class RubricDrawBlock:
    """Store one synchronized conditional draw block for a rubric."""

    rubric_id: str
    variant_ids: tuple[str, ...]
    global_outer_attempt_id: int
    rubric_retained_index: int
    rubric_v2_draws: np.ndarray


@dataclass(frozen=True, slots=True)
class ImportanceDiagnostic:
    """Store a fixed importance-resampling adequacy comparison."""

    rubric_id: str
    particle_count: int
    effective_sample_size: float
    effective_sample_size_ratio: float
    adequate: bool
    laplace_mean: np.ndarray
    importance_mean: np.ndarray
    laplace_covariance: np.ndarray
    importance_covariance: np.ndarray


def conditional_mode(
    rubric: Layer1RubricData,
    hyperparameters: Layer1Hyperparameters,
    moments: Layer1MomentStart,
) -> Layer1Mode:
    """Find one rubric's conditional mode against the observed counts."""
    try:
        initial = moments.rubric_latent_starts[rubric.rubric_id]
    except KeyError as error:
        raise MalformedInputError("Missing Layer 1 rubric moment start") from error
    return find_rubric_mode(rubric, hyperparameters, initial)


def draw_rubric_v2(
    rubric: Layer1RubricData,
    mode: Layer1Mode,
    *,
    rng: np.random.Generator,
    draw_count: int,
) -> np.ndarray:
    """Draw joint variant RubricV2 values from one Laplace approximation."""
    if draw_count <= 0:
        raise MalformedInputError("Conditional draw count must be positive")
    latent_draws = rng.multivariate_normal(
        mode.value,
        mode.covariance,
        size=draw_count,
        check_valid="raise",
    )
    eta_draws = latent_draws[:, 2:].reshape(draw_count, len(rubric.variants), 2)
    rubric_v2 = np.empty((draw_count, len(rubric.variants)), dtype=float)
    for draw_index in range(draw_count):
        for variant_index in range(len(rubric.variants)):
            probabilities = inverse_alr(eta_draws[draw_index, variant_index])
            rubric_v2[draw_index, variant_index] = (
                probabilities[0] + 0.5 * probabilities[2]
            )
    if not np.isfinite(rubric_v2).all():
        raise Layer1ModeError("Conditional RubricV2 draws are non-finite")
    return rubric_v2


def make_rubric_draw_block(
    rubric: Layer1RubricData,
    hyperparameters: Layer1Hyperparameters,
    moments: Layer1MomentStart,
    *,
    rng: np.random.Generator,
    global_outer_attempt_id: int,
    rubric_retained_index: int,
    draw_count: int = INNER_DRAWS_PER_OUTER,
) -> RubricDrawBlock:
    """Create one identified joint conditional draw block."""
    if global_outer_attempt_id <= 0 or rubric_retained_index <= 0:
        raise MalformedInputError("Layer 1 draw indices must be positive")
    mode = conditional_mode(rubric, hyperparameters, moments)
    return RubricDrawBlock(
        rubric_id=rubric.rubric_id,
        variant_ids=tuple(variant.variant_id for variant in rubric.variants),
        global_outer_attempt_id=global_outer_attempt_id,
        rubric_retained_index=rubric_retained_index,
        rubric_v2_draws=draw_rubric_v2(rubric, mode, rng=rng, draw_count=draw_count),
    )


def _multivariate_normal_logpdf(
    values: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    difference = values - mean
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise Layer1ModeError("Laplace proposal covariance is not positive definite")
    solved = np.linalg.solve(covariance, difference.T).T
    return -0.5 * (
        values.shape[1] * np.log(2.0 * np.pi)
        + log_determinant
        + np.einsum("ij,ij->i", difference, solved)
    )


def importance_resampling_diagnostic(
    rubric: Layer1RubricData,
    hyperparameters: Layer1Hyperparameters,
    moments: Layer1MomentStart,
    *,
    rng: np.random.Generator,
    particle_count: int = IMPORTANCE_PARTICLES,
) -> ImportanceDiagnostic:
    """Compare the conditional Laplace approximation with self-importance sampling."""
    if particle_count <= 1:
        raise MalformedInputError("Importance particle count must exceed one")
    mode = conditional_mode(rubric, hyperparameters, moments)
    particles = rng.multivariate_normal(
        mode.value,
        mode.covariance,
        size=particle_count,
        check_valid="raise",
    )
    target_log = np.asarray(
        [
            rubric_log_density_derivatives(rubric, hyperparameters, particle)[0]
            for particle in particles
        ]
    )
    proposal_log = _multivariate_normal_logpdf(particles, mode.value, mode.covariance)
    log_weights = target_log - proposal_log
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise Layer1ModeError("Importance weights are non-computable")
    normalized = weights / total
    effective_sample_size = float(total**2 / np.square(weights).sum())
    importance_mean = normalized @ particles
    centered = particles - importance_mean
    importance_covariance = (centered * normalized[:, None]).T @ centered
    ratio = effective_sample_size / particle_count
    return ImportanceDiagnostic(
        rubric_id=rubric.rubric_id,
        particle_count=particle_count,
        effective_sample_size=effective_sample_size,
        effective_sample_size_ratio=ratio,
        adequate=ratio >= IMPORTANCE_ESS_RATIO_MINIMUM,
        laplace_mean=mode.value,
        importance_mean=importance_mean,
        laplace_covariance=mode.covariance,
        importance_covariance=importance_covariance,
    )


def observed_rubric_performance(rubric: Layer1RubricData) -> float:
    """Compute equal-variant, equal-trace observed RubricV2 performance."""
    variant_values = [
        np.mean(
            (variant.counts[:, 0] + 0.5 * variant.counts[:, 2])
            / rubric.expectation_count
        )
        for variant in rubric.variants
    ]
    return float(np.mean(variant_values))


def select_importance_rubrics(data: Layer1Dataset) -> tuple[str, ...]:
    """Select the frozen deterministic purposive importance-resampling subset."""
    if not data.rubrics:
        raise MalformedInputError("Cannot select importance rubrics from empty data")
    ordered = sorted(
        data.rubrics, key=lambda rubric: (rubric.rubric_order, rubric.rubric_id)
    )
    selectors = (
        min(
            ordered,
            key=lambda rubric: (
                len(rubric.variants),
                rubric.rubric_order,
                rubric.rubric_id,
            ),
        ),
        min(
            ordered,
            key=lambda rubric: (
                rubric.expectation_count,
                rubric.rubric_order,
                rubric.rubric_id,
            ),
        ),
        min(
            ordered,
            key=lambda rubric: (
                observed_rubric_performance(rubric),
                rubric.rubric_order,
                rubric.rubric_id,
            ),
        ),
        min(
            ordered,
            key=lambda rubric: (
                -observed_rubric_performance(rubric),
                rubric.rubric_order,
                rubric.rubric_id,
            ),
        ),
    )
    return tuple(dict.fromkeys(rubric.rubric_id for rubric in selectors))
