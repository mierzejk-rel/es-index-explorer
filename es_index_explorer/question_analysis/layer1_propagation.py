"""Parametric hyperparameter propagation and adaptive Monte Carlo controls."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from es_index_explorer.question_analysis.errors import (
    Layer1ModeError,
    Layer1ReplenishmentExhaustedError,
    NumericalError,
)
from es_index_explorer.question_analysis.layer1_laplace import (
    INNER_DRAWS_PER_OUTER,
    RubricDrawBlock,
    make_rubric_draw_block,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MmlFit,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1VariantData,
    fit_layer1_mml,
    inverse_alr,
)
from es_index_explorer.question_analysis.layer1_packed import compress_pfu_counts
from es_index_explorer.question_analysis.seeds import derive_child_seed

BASE_OUTER_DRAWS = 500
PREFIX_DIAGNOSTIC_DEPTHS = (250, 500, 1_000)
ADAPTIVE_DEPTHS = (500, 1_000, 2_000, 4_000)
MAXIMUM_REPLENISHMENT_MULTIPLIER = 10
GAMMA = 0.90
ELEVATED_FAILURE_RATE = 0.05

Layer1Refitter = Callable[[Layer1Dataset], Layer1MmlFit]


@dataclass(frozen=True, slots=True)
class OuterHyperparameterDraw:
    """Store one globally retained parametric-bootstrap hyperparameter draw."""

    global_outer_attempt_id: int
    hyperparameters: Layer1Hyperparameters


@dataclass(frozen=True, slots=True)
class OuterSequenceResult:
    """Store retained outer draws and globally rejected attempt counts."""

    draws: tuple[OuterHyperparameterDraw, ...]
    attempted: int
    failed: int


@dataclass(frozen=True, slots=True)
class RubricPropagation:
    """Store one rubric's valid conditional blocks and failure disclosure."""

    rubric_id: str
    blocks: tuple[RubricDrawBlock, ...]
    attempted: int
    failed: int
    elevated_failure_conditions: bool


@dataclass(frozen=True, slots=True)
class EventMonteCarloSummary:
    """Store an event probability and outer-batch Monte Carlo uncertainty."""

    estimate: float
    mcse: float
    interval_lower: float
    interval_upper: float
    outer_draw_count: int
    refinement_triggered: bool


@dataclass(frozen=True, slots=True)
class Layer1Simulation:
    """Store one simulated dataset and its known latent generating values."""

    data: Layer1Dataset
    rubric_latents: Mapping[str, np.ndarray]
    variant_latents: Mapping[str, np.ndarray]


def simulate_layer1_dataset(
    design: Layer1Dataset,
    hyperparameters: Layer1Hyperparameters,
    *,
    rng: np.random.Generator,
) -> Layer1Dataset:
    """Simulate raw PFU counts while holding the observed hierarchy design fixed."""
    return simulate_layer1_dataset_with_truth(
        design,
        hyperparameters,
        rng=rng,
    ).data


def simulate_layer1_dataset_with_truth(
    design: Layer1Dataset,
    hyperparameters: Layer1Hyperparameters,
    *,
    rng: np.random.Generator,
) -> Layer1Simulation:
    """Simulate PFU counts and retain latent truth for recovery calibration."""
    rubrics: list[Layer1RubricData] = []
    rubric_latents: dict[str, np.ndarray] = {}
    variant_latents: dict[str, np.ndarray] = {}
    for rubric in design.rubrics:
        rubric_mean = (
            hyperparameters.mu0 + hyperparameters.dataset_offsets[rubric.dataset]
        )
        rubric_latent = rng.multivariate_normal(
            rubric_mean,
            hyperparameters.sigma_between,
            check_valid="raise",
        )
        rubric_latents[rubric.rubric_id] = rubric_latent
        variants: list[Layer1VariantData] = []
        for variant in rubric.variants:
            variant_latent = rng.multivariate_normal(
                rubric_latent,
                hyperparameters.sigma_within,
                check_valid="raise",
            )
            variant_latents[variant.variant_id] = variant_latent
            variant_probability = inverse_alr(variant_latent)
            counts = np.empty_like(variant.counts, dtype=int)
            for trace_index in range(len(variant.arm_ids)):
                trace_probability = rng.dirichlet(
                    hyperparameters.phi * variant_probability
                )
                counts[trace_index] = rng.multinomial(
                    rubric.expectation_count, trace_probability
                )
            variants.append(
                Layer1VariantData(
                    variant_id=variant.variant_id,
                    counts=counts,
                    arm_ids=variant.arm_ids,
                    stages=variant.stages,
                    compressed_counts=compress_pfu_counts(counts),
                )
            )
        rubrics.append(
            Layer1RubricData(
                rubric_id=rubric.rubric_id,
                rubric_order=rubric.rubric_order,
                dataset=rubric.dataset,
                expectation_count=rubric.expectation_count,
                variants=tuple(variants),
            )
        )
    return Layer1Simulation(
        data=Layer1Dataset(
            rubrics=tuple(rubrics),
            dataset_levels=design.dataset_levels,
        ),
        rubric_latents=rubric_latents,
        variant_latents=variant_latents,
    )


def generate_outer_sequence(
    design: Layer1Dataset,
    fitted: Layer1Hyperparameters,
    *,
    target_depth: int,
    refitter: Layer1Refitter = fit_layer1_mml,
    maximum_attempts: int | None = None,
) -> OuterSequenceResult:
    """Generate and replenish the deterministic global outer hyperparameter sequence."""
    if target_depth <= 0:
        raise Layer1ReplenishmentExhaustedError(
            "Layer 1 outer target depth must be positive"
        )
    cap = maximum_attempts or MAXIMUM_REPLENISHMENT_MULTIPLIER * target_depth
    retained: list[OuterHyperparameterDraw] = []
    failed = 0
    for attempt_id in range(1, cap + 1):
        rng = np.random.default_rng(
            derive_child_seed("layer1_bootstrap", f"outer:{attempt_id}")
        )
        simulated = simulate_layer1_dataset(design, fitted, rng=rng)
        try:
            refit = refitter(simulated)
        except NumericalError:
            failed += 1
            continue
        retained.append(
            OuterHyperparameterDraw(
                global_outer_attempt_id=attempt_id,
                hyperparameters=refit.hyperparameters,
            )
        )
        if len(retained) == target_depth:
            return OuterSequenceResult(
                draws=tuple(retained),
                attempted=attempt_id,
                failed=failed,
            )
    raise Layer1ReplenishmentExhaustedError(
        "LAYER1_REPLENISHMENT_EXHAUSTED "
        f"scope=global target={target_depth} attempted={cap} valid={len(retained)}"
    )


def propagate_rubric(
    rubric: Layer1RubricData,
    moments: Layer1MomentStart,
    outer_draws: Sequence[OuterHyperparameterDraw],
    *,
    target_depth: int,
    inner_draw_count: int = INNER_DRAWS_PER_OUTER,
) -> RubricPropagation:
    """Create one rubric's retained conditional blocks from the global sequence."""
    blocks: list[RubricDrawBlock] = []
    failures = 0
    attempted = 0
    for outer in outer_draws:
        attempted += 1
        rng = np.random.default_rng(
            derive_child_seed(
                "laplace_draws",
                f"{rubric.rubric_id}:outer:{outer.global_outer_attempt_id}",
            )
        )
        try:
            block = make_rubric_draw_block(
                rubric,
                outer.hyperparameters,
                moments,
                rng=rng,
                global_outer_attempt_id=outer.global_outer_attempt_id,
                rubric_retained_index=len(blocks) + 1,
                draw_count=inner_draw_count,
            )
        except Layer1ModeError:
            failures += 1
            continue
        blocks.append(block)
        if len(blocks) == target_depth:
            rate = failures / attempted
            return RubricPropagation(
                rubric_id=rubric.rubric_id,
                blocks=tuple(blocks),
                attempted=attempted,
                failed=failures,
                elevated_failure_conditions=rate > ELEVATED_FAILURE_RATE,
            )
    raise Layer1ReplenishmentExhaustedError(
        "LAYER1_REPLENISHMENT_EXHAUSTED "
        f"scope={rubric.rubric_id} target={target_depth} "
        f"attempted={attempted} valid={len(blocks)}"
    )


def event_monte_carlo_summary(
    indicator_blocks: Sequence[np.ndarray],
    *,
    gamma: float = GAMMA,
) -> EventMonteCarloSummary:
    """Compute Pi_prop and MCSE from one mean per retained outer draw."""
    if len(indicator_blocks) <= 1:
        raise Layer1ReplenishmentExhaustedError(
            "Layer 1 MCSE requires at least two outer draws"
        )
    batch_means = np.asarray(
        [np.asarray(block, dtype=float).mean() for block in indicator_blocks]
    )
    if not np.isfinite(batch_means).all():
        raise Layer1ModeError("Layer 1 event indicators are non-finite")
    estimate = float(batch_means.mean())
    mcse = float(np.std(batch_means, ddof=1) / np.sqrt(len(batch_means)))
    lower = estimate - 2.0 * mcse
    upper = estimate + 2.0 * mcse
    return EventMonteCarloSummary(
        estimate=estimate,
        mcse=mcse,
        interval_lower=lower,
        interval_upper=upper,
        outer_draw_count=len(batch_means),
        refinement_triggered=lower <= gamma <= upper,
    )


def next_refinement_depth(current_depth: int) -> int | None:
    """Return the next frozen rubric-wide adaptive depth."""
    try:
        position = ADAPTIVE_DEPTHS.index(current_depth)
    except ValueError as error:
        raise Layer1ReplenishmentExhaustedError(
            "Current Layer 1 depth is not on the adaptive grid"
        ) from error
    return (
        ADAPTIVE_DEPTHS[position + 1] if position + 1 < len(ADAPTIVE_DEPTHS) else None
    )


def shared_draw_keys(
    first: RubricPropagation,
    second: RubricPropagation,
) -> frozenset[tuple[int, int]]:
    """Return exact global outer/inner keys shared by two rubrics."""
    first_keys = {
        (block.global_outer_attempt_id, inner_index)
        for block in first.blocks
        for inner_index in range(len(block.rubric_v2_draws))
    }
    second_keys = {
        (block.global_outer_attempt_id, inner_index)
        for block in second.blocks
        for inner_index in range(len(block.rubric_v2_draws))
    }
    return frozenset(first_keys & second_keys)


def block_mapping(
    propagation: RubricPropagation,
) -> Mapping[tuple[int, int], np.ndarray]:
    """Map synchronized draw keys to one rubric's variant-value vector."""
    return {
        (block.global_outer_attempt_id, inner_index): values
        for block in propagation.blocks
        for inner_index, values in enumerate(block.rubric_v2_draws)
    }
