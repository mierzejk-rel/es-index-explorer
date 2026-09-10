"""Tests for Layer 1 event, tier, diagnostic, and stability rules."""

import numpy as np
import pytest

from es_index_explorer.question_analysis.contracts import (
    Layer1EventKind,
    SuitabilityTier,
)
from es_index_explorer.question_analysis.layer1_decisions import (
    build_primary_events,
    heterogeneity_dispersion,
    leave_out_tier_changes,
    pooled_mean_tier,
    proportion_diagnostic,
    rank_reversal,
    run_full_leave_out_refits,
    score_band_probabilities,
    shrunken_variant_means,
    tier_for_events,
    tier_from_probabilities,
    uncertain_flag,
)
from es_index_explorer.question_analysis.layer1_laplace import RubricDrawBlock
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1VariantData,
)
from es_index_explorer.question_analysis.layer1_propagation import RubricPropagation

pytestmark = pytest.mark.unit


def _propagation(
    *,
    rubric_id: str = "r1",
    depth: int = 4,
    attempt_offset: int = 0,
) -> RubricPropagation:
    blocks = tuple(
        RubricDrawBlock(
            rubric_id=rubric_id,
            variant_ids=(f"{rubric_id}-v1", f"{rubric_id}-v2"),
            global_outer_attempt_id=attempt_offset + index,
            rubric_retained_index=index,
            rubric_v2_draws=np.array(
                [
                    [0.80, 0.70],
                    [0.76, 0.61],
                    [0.65, 0.55],
                    [0.45, 0.52],
                ]
            ),
        )
        for index in range(1, depth + 1)
    )
    return RubricPropagation(
        rubric_id=rubric_id,
        blocks=blocks,
        attempted=depth,
        failed=0,
        elevated_failure_conditions=False,
    )


def test_primary_event_registry_contains_every_unit_kind_and_floor() -> None:
    events = build_primary_events(_propagation(), depth=4)

    assert len(events) == 12
    assert {(event.unit_id, event.kind, event.floor) for event in events} == {
        *{("r1", Layer1EventKind.WORST_VARIANT, floor) for floor in (0.75, 0.60, 0.50)},
        *{
            ("r1", Layer1EventKind.PROPORTION_KAPPA_0_75, floor)
            for floor in (0.75, 0.60, 0.50)
        },
        *{
            (variant, Layer1EventKind.SINGLE_VARIANT, floor)
            for variant in ("r1-v1", "r1-v2")
            for floor in (0.75, 0.60, 0.50)
        },
    }
    assert all(event.summary.outer_draw_count == 4 for event in events)


def test_tier_rule_uses_inclusive_gamma_and_empty_set_convention() -> None:
    suitable = tier_from_probabilities({0.75: 0.90, 0.60: 1.0, 0.50: 1.0})
    none = tier_from_probabilities({0.75: 0.10, 0.60: 0.50, 0.50: 0.89})

    assert suitable.tier is SuitabilityTier.SUITABLE
    assert none.tier is SuitabilityTier.NOT_SUITABLE


def test_adaptive_cap_preserves_indeterminate_as_distinct_tier() -> None:
    events = build_primary_events(_propagation(), depth=4, gamma=0.50)
    decision = tier_for_events(
        events,
        kind=Layer1EventKind.WORST_VARIANT,
        unit_id="r1",
        at_adaptive_cap=True,
    )

    assert decision.tier is SuitabilityTier.MONTE_CARLO_INDETERMINATE
    assert decision.monte_carlo_indeterminate


@pytest.mark.parametrize(
    ("variant_count", "binding_count", "uninformative"),
    ((2, 2, True), (3, 3, True), (4, 3, False), (8, 6, False)),
)
def test_proportion_diagnostic_has_exact_grid_and_binding_count(
    variant_count: int,
    binding_count: int,
    uninformative: bool,
) -> None:
    diagnostic = proportion_diagnostic(variant_count)

    assert diagnostic.binding_count == binding_count
    assert diagnostic.uninformative_by_construction is uninformative
    assert diagnostic.attainable_grid == tuple(
        count / variant_count for count in range(variant_count + 1)
    )


def test_pooled_mean_tier_and_uncertain_flag_use_same_draws() -> None:
    propagation = _propagation()
    events = build_primary_events(propagation, depth=4)
    primary = tier_for_events(
        events,
        kind=Layer1EventKind.WORST_VARIANT,
        unit_id="r1",
    )
    pooled = pooled_mean_tier(propagation, depth=4)

    assert uncertain_flag(primary, pooled) is (
        pooled.tier.value != primary.tier.value
        and pooled.tier is not SuitabilityTier.NOT_SUITABLE
    )


def test_score_bands_partition_unit_interval_with_inclusive_top() -> None:
    probabilities = score_band_probabilities(
        np.array([0.0, 0.499, 0.50, 0.599, 0.60, 0.749, 0.75, 1.0])
    )

    assert probabilities == {
        "below_0_50": 0.25,
        "0_50_to_0_60": 0.25,
        "0_60_to_0_75": 0.25,
        "0_75_to_1_00": 0.25,
    }


def test_shrunken_variant_means_use_common_depth() -> None:
    means = shrunken_variant_means(_propagation(), depth=4)

    assert means == pytest.approx({"r1-v1": 0.665, "r1-v2": 0.595})


def test_heterogeneity_is_missing_below_three_variants() -> None:
    moments = Layer1MomentStart(
        hyperparameters=Layer1Hyperparameters(
            phi=1.0,
            mu0=np.zeros(2),
            dataset_offsets={"emc2_set1": np.zeros(2)},
            sigma_within=np.eye(2),
            sigma_between=np.eye(2),
        ),
        rubric_latent_starts={},
        variant_alr={
            "v1": np.array([0.0, 0.0]),
            "v2": np.array([1.0, 0.0]),
            "v3": np.array([0.0, 1.0]),
        },
        smoothed_variant_count=0,
        retained_phi_components=0,
    )

    assert heterogeneity_dispersion("r1", ("v1", "v2"), moments) is None
    assert heterogeneity_dispersion("r1", ("v1", "v2", "v3"), moments) == pytest.approx(
        1 / 3
    )


def test_rank_reversal_requires_250_shared_outer_attempts() -> None:
    first = _propagation(rubric_id="r1", depth=250)
    second = _propagation(rubric_id="r2", depth=250)
    comparable = rank_reversal(first, second, first_point=0.7, second_point=0.6)
    not_comparable = rank_reversal(
        first,
        _propagation(rubric_id="r3", depth=249, attempt_offset=1_000),
        first_point=0.7,
        second_point=0.6,
    )

    assert comparable.comparable
    assert comparable.shared_outer_attempts == 250
    assert comparable.reversal_frequency == 1.0
    assert not not_comparable.comparable
    assert not_comparable.reversal_frequency is None


def test_leave_out_changes_compare_tier_identity() -> None:
    primary = {
        "r1": SuitabilityTier.SUITABLE,
        "r2": SuitabilityTier.PROMISING,
    }
    alternatives = {
        "arm-a": {
            "r1": SuitabilityTier.PROMISING,
            "r2": SuitabilityTier.PROMISING,
        },
        "stage-c": primary,
    }

    assert leave_out_tier_changes(primary, alternatives) == {
        "arm-a": ("r1",),
        "stage-c": (),
    }


def test_leave_out_mechanics_invoke_estimator_on_physically_reduced_data() -> None:
    data = Layer1Dataset(
        rubrics=(
            Layer1RubricData(
                rubric_id="r1",
                rubric_order=1,
                dataset="emc2_set1",
                expectation_count=2,
                variants=(
                    Layer1VariantData(
                        variant_id="v1",
                        counts=np.array([[2, 0, 0], [1, 1, 0], [0, 2, 0]]),
                        arm_ids=("a1", "a2", "a3"),
                        stages=("A", "B", "B"),
                    ),
                ),
            ),
        ),
        dataset_levels=("emc2_set1",),
    )
    observed_sizes: dict[str, int] = {}

    def estimator(reduced: Layer1Dataset) -> dict[str, SuitabilityTier]:
        retained_arms = reduced.rubrics[0].variants[0].arm_ids
        observed_sizes[":".join(retained_arms)] = len(retained_arms)
        return {"r1": SuitabilityTier.SUITABLE}

    results = run_full_leave_out_refits(data, estimator)

    assert set(results) == {"arm:a1", "arm:a2", "arm:a3", "stage:A", "stage:B"}
    assert sorted(observed_sizes.values()) == [1, 2, 2, 2]
