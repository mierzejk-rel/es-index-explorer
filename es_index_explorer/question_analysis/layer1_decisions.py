"""Layer 1 event probabilities, tiers, diagnostics, and stability."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np

from es_index_explorer.question_analysis.contracts import (
    Layer1EventKind,
    SuitabilityTier,
)
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1VariantData,
)
from es_index_explorer.question_analysis.layer1_propagation import (
    GAMMA,
    EventMonteCarloSummary,
    RubricPropagation,
    block_mapping,
    event_monte_carlo_summary,
    shared_draw_keys,
)

KAPPA = 0.75
TIER_FLOORS = (0.75, 0.60, 0.50)
TIER_BY_FLOOR = {
    0.75: SuitabilityTier.SUITABLE,
    0.60: SuitabilityTier.PROMISING,
    0.50: SuitabilityTier.BORDERLINE,
}
TIER_ORDER = {
    SuitabilityTier.NOT_SUITABLE: 0,
    SuitabilityTier.BORDERLINE: 1,
    SuitabilityTier.PROMISING: 2,
    SuitabilityTier.SUITABLE: 3,
}
SCORE_BANDS = (
    ("below_0_50", 0.0, 0.50, False),
    ("0_50_to_0_60", 0.50, 0.60, False),
    ("0_60_to_0_75", 0.60, 0.75, False),
    ("0_75_to_1_00", 0.75, 1.00, True),
)
MINIMUM_SHARED_OUTER_ATTEMPTS = 250
Layer1TierEstimator = Callable[[Layer1Dataset], Mapping[str, SuitabilityTier]]


@dataclass(frozen=True, slots=True)
class PrimaryEvent:
    """Store one explicit adaptive-refinement event and its MC summary."""

    unit_id: str
    rubric_id: str
    kind: Layer1EventKind
    floor: float
    summary: EventMonteCarloSummary


@dataclass(frozen=True, slots=True)
class TierDecision:
    """Store one deterministic tier and its supporting floor probabilities."""

    tier: SuitabilityTier
    probabilities: Mapping[float, float]
    monte_carlo_indeterminate: bool


@dataclass(frozen=True, slots=True)
class ProportionDiagnostic:
    """Store the finite-grid variant-proportion diagnostic."""

    variant_count: int
    binding_count: int
    attainable_grid: tuple[float, ...]
    uninformative_by_construction: bool


@dataclass(frozen=True, slots=True)
class RankReversal:
    """Store synchronized rank-reversal evidence for one unit pair."""

    first_rubric_id: str
    second_rubric_id: str
    shared_outer_attempts: int
    shared_draws: int
    comparable: bool
    reversal_frequency: float | None


def _blocks_at_depth(
    propagation: RubricPropagation,
    depth: int,
) -> tuple[np.ndarray, ...]:
    if depth <= 1 or len(propagation.blocks) < depth:
        raise MalformedInputError(
            "Layer 1 propagation does not contain requested depth"
        )
    return tuple(block.rubric_v2_draws for block in propagation.blocks[:depth])


def build_primary_events(
    propagation: RubricPropagation,
    *,
    depth: int,
    gamma: float = GAMMA,
) -> tuple[PrimaryEvent, ...]:
    """Materialize every rubric, variant, and proportion event at one common depth."""
    blocks = _blocks_at_depth(propagation, depth)
    variant_ids = propagation.blocks[0].variant_ids
    if not variant_ids or any(
        block.variant_ids != variant_ids for block in propagation.blocks[:depth]
    ):
        raise MalformedInputError("Rubric draw blocks have inconsistent variants")
    events: list[PrimaryEvent] = []
    binding_count = ceil(KAPPA * len(variant_ids))
    for floor in TIER_FLOORS:
        events.append(
            PrimaryEvent(
                unit_id=propagation.rubric_id,
                rubric_id=propagation.rubric_id,
                kind=Layer1EventKind.WORST_VARIANT,
                floor=floor,
                summary=event_monte_carlo_summary(
                    tuple(np.min(block, axis=1) >= floor for block in blocks),
                    gamma=gamma,
                ),
            )
        )
        events.append(
            PrimaryEvent(
                unit_id=propagation.rubric_id,
                rubric_id=propagation.rubric_id,
                kind=Layer1EventKind.PROPORTION_KAPPA_0_75,
                floor=floor,
                summary=event_monte_carlo_summary(
                    tuple(
                        np.count_nonzero(block >= floor, axis=1) >= binding_count
                        for block in blocks
                    ),
                    gamma=gamma,
                ),
            )
        )
        for variant_index, variant_id in enumerate(variant_ids):
            events.append(
                PrimaryEvent(
                    unit_id=variant_id,
                    rubric_id=propagation.rubric_id,
                    kind=Layer1EventKind.SINGLE_VARIANT,
                    floor=floor,
                    summary=event_monte_carlo_summary(
                        tuple(block[:, variant_index] >= floor for block in blocks),
                        gamma=gamma,
                    ),
                )
            )
    return tuple(events)


def tier_from_probabilities(
    probabilities: Mapping[float, float],
    *,
    gamma: float = GAMMA,
    monte_carlo_indeterminate: bool = False,
) -> TierDecision:
    """Assign the highest frozen floor clearing gamma."""
    if set(probabilities) != set(TIER_FLOORS) or any(
        not np.isfinite(value) or not 0 <= value <= 1
        for value in probabilities.values()
    ):
        raise MalformedInputError("Tier probabilities must cover every frozen floor")
    if monte_carlo_indeterminate:
        tier = SuitabilityTier.MONTE_CARLO_INDETERMINATE
    else:
        tier = SuitabilityTier.NOT_SUITABLE
        for floor in TIER_FLOORS:
            if probabilities[floor] >= gamma:
                tier = TIER_BY_FLOOR[floor]
                break
    return TierDecision(
        tier=tier,
        probabilities=dict(probabilities),
        monte_carlo_indeterminate=monte_carlo_indeterminate,
    )


def tier_for_events(
    events: Sequence[PrimaryEvent],
    *,
    kind: Layer1EventKind,
    unit_id: str,
    at_adaptive_cap: bool = False,
) -> TierDecision:
    """Assign a tier from one unit's explicit floor events."""
    selected = [
        event for event in events if event.kind is kind and event.unit_id == unit_id
    ]
    probabilities = {event.floor: event.summary.estimate for event in selected}
    indeterminate = at_adaptive_cap and any(
        event.summary.refinement_triggered for event in selected
    )
    return tier_from_probabilities(
        probabilities, monte_carlo_indeterminate=indeterminate
    )


def proportion_diagnostic(variant_count: int) -> ProportionDiagnostic:
    """Construct the exact finite-grid kappa diagnostic metadata."""
    if variant_count <= 0:
        raise MalformedInputError("Variant count must be positive")
    return ProportionDiagnostic(
        variant_count=variant_count,
        binding_count=ceil(KAPPA * variant_count),
        attainable_grid=tuple(
            count / variant_count for count in range(variant_count + 1)
        ),
        uninformative_by_construction=variant_count <= 3,
    )


def pooled_mean_tier(
    propagation: RubricPropagation,
    *,
    depth: int,
) -> TierDecision:
    """Apply the tier rule to the per-draw arithmetic variant mean."""
    blocks = _blocks_at_depth(propagation, depth)
    probabilities = {
        floor: float(
            np.mean(
                np.concatenate([np.mean(block, axis=1) >= floor for block in blocks])
            )
        )
        for floor in TIER_FLOORS
    }
    return tier_from_probabilities(probabilities)


def uncertain_flag(primary: TierDecision, pooled_mean: TierDecision) -> bool:
    """Return whether the pooled-mean tier is strictly higher than the primary tier."""
    if (
        primary.tier is SuitabilityTier.MONTE_CARLO_INDETERMINATE
        or pooled_mean.tier is SuitabilityTier.MONTE_CARLO_INDETERMINATE
    ):
        return False
    return TIER_ORDER[pooled_mean.tier] > TIER_ORDER[primary.tier]


def score_band_probabilities(
    values: np.ndarray,
) -> Mapping[str, float]:
    """Compute probabilities for the four frozen RubricV2 score bands."""
    scores = np.asarray(values, dtype=float)
    if (
        scores.ndim != 1
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise MalformedInputError(
            "Score-band values must be a finite unit-interval vector"
        )
    probabilities: dict[str, float] = {}
    for name, lower, upper, include_upper in SCORE_BANDS:
        included = (scores >= lower) & (
            scores <= upper if include_upper else scores < upper
        )
        probabilities[name] = float(np.mean(included))
    return probabilities


def shrunken_variant_means(
    propagation: RubricPropagation,
    *,
    depth: int,
) -> Mapping[str, float]:
    """Compute propagated shrunken means for every variant at common depth."""
    blocks = _blocks_at_depth(propagation, depth)
    values = np.concatenate(blocks, axis=0)
    return {
        variant_id: float(values[:, index].mean())
        for index, variant_id in enumerate(propagation.blocks[0].variant_ids)
    }


def heterogeneity_dispersion(
    rubric_id: str,
    variant_ids: Sequence[str],
    moments: Layer1MomentStart,
) -> float | None:
    """Compute D_r from method-of-moments variant ALR residuals."""
    if len(variant_ids) < 3:
        return None
    values = np.stack([moments.variant_alr[variant_id] for variant_id in variant_ids])
    covariance = np.cov(values - values.mean(axis=0), rowvar=False, ddof=1)
    del rubric_id
    return float(np.trace(covariance) / 2.0)


def rank_reversal(
    first: RubricPropagation,
    second: RubricPropagation,
    *,
    first_point: float,
    second_point: float,
) -> RankReversal:
    """Compute rank reversal on exact draw keys shared by both rubrics."""
    keys = shared_draw_keys(first, second)
    shared_outer = len({key[0] for key in keys})
    if shared_outer < MINIMUM_SHARED_OUTER_ATTEMPTS:
        return RankReversal(
            first_rubric_id=first.rubric_id,
            second_rubric_id=second.rubric_id,
            shared_outer_attempts=shared_outer,
            shared_draws=len(keys),
            comparable=False,
            reversal_frequency=None,
        )
    first_values = block_mapping(first)
    second_values = block_mapping(second)
    point_order = first_point > second_point
    reversals = [
        (float(np.min(first_values[key])) > float(np.min(second_values[key])))
        != point_order
        for key in sorted(keys)
    ]
    return RankReversal(
        first_rubric_id=first.rubric_id,
        second_rubric_id=second.rubric_id,
        shared_outer_attempts=shared_outer,
        shared_draws=len(keys),
        comparable=True,
        reversal_frequency=float(np.mean(reversals)),
    )


def leave_out_tier_changes(
    primary: Mapping[str, SuitabilityTier],
    alternatives: Mapping[str, Mapping[str, SuitabilityTier]],
) -> Mapping[str, tuple[str, ...]]:
    """Identify units whose point tier changes under each fully refitted leave-out."""
    return {
        leave_out_id: tuple(
            sorted(
                unit_id
                for unit_id, original in primary.items()
                if refit.get(unit_id) != original
            )
        )
        for leave_out_id, refit in alternatives.items()
    }


def _filter_layer1_design(
    data: Layer1Dataset,
    *,
    excluded_arm: str | None = None,
    excluded_stage: str | None = None,
) -> Layer1Dataset:
    rubrics: list[Layer1RubricData] = []
    for rubric in data.rubrics:
        variants: list[Layer1VariantData] = []
        for variant in rubric.variants:
            mask = np.asarray(
                [
                    arm != excluded_arm
                    and (not variant.stages or stage != excluded_stage)
                    for arm, stage in zip(
                        variant.arm_ids,
                        variant.stages or ("",) * len(variant.arm_ids),
                        strict=True,
                    )
                ]
            )
            variants.append(
                Layer1VariantData(
                    variant_id=variant.variant_id,
                    counts=variant.counts[mask],
                    arm_ids=tuple(
                        arm
                        for arm, retained in zip(variant.arm_ids, mask, strict=True)
                        if retained
                    ),
                    stages=tuple(
                        stage
                        for stage, retained in zip(variant.stages, mask, strict=True)
                        if retained
                    ),
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
    return Layer1Dataset(rubrics=tuple(rubrics), dataset_levels=data.dataset_levels)


def run_full_leave_out_refits(
    data: Layer1Dataset,
    estimator: Layer1TierEstimator,
) -> Mapping[str, Mapping[str, SuitabilityTier]]:
    """Fully invoke the Layer 1 estimator for each observed arm and stage removal."""
    return {
        comparison_id: estimator(reduced)
        for comparison_id, reduced in layer1_leave_out_datasets(data).items()
    }


def layer1_leave_out_datasets(
    data: Layer1Dataset,
) -> Mapping[str, Layer1Dataset]:
    """Construct each observed arm/stage removal for an independent full refit."""
    arms = sorted(
        {
            arm
            for rubric in data.rubrics
            for variant in rubric.variants
            for arm in variant.arm_ids
        }
    )
    stages = sorted(
        {
            stage
            for rubric in data.rubrics
            for variant in rubric.variants
            for stage in variant.stages
        }
    )
    results: dict[str, Layer1Dataset] = {}
    for arm in arms:
        results[f"arm:{arm}"] = _filter_layer1_design(data, excluded_arm=arm)
    for stage in stages:
        results[f"stage:{stage}"] = _filter_layer1_design(data, excluded_stage=stage)
    return results
