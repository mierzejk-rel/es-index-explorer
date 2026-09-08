"""Tests for sampled/enumerated WCR and BH bracket adjudication."""

import numpy as np
import pytest

from es_index_explorer.question_analysis.cluster_covariance import (
    joint_wald_statistic,
    three_term_cluster_covariance,
)
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.glm_primitives import (
    bernoulli_bread,
    bernoulli_score_rows,
    solve_restricted_logit,
    solve_unrestricted_logit,
)
from es_index_explorer.question_analysis.multiplicity import (
    PValueBracket,
    adjudicate_bh_brackets,
    benjamini_hochberg,
)
from es_index_explorer.question_analysis.wild_bootstrap import (
    WildBootstrapProblem,
    enumerated_rademacher_weights,
    family_bootstrap_rngs,
    logit_full_refit_statistic,
    one_step_replicate,
    run_wild_cluster_bootstrap,
)

pytestmark = pytest.mark.unit


def _problem() -> WildBootstrapProblem:
    rng = np.random.default_rng(42)
    rubric = tuple(
        f"rubric-{rubric_index}"
        for rubric_index in range(4)
        for _arm_index in range(4)
        for _repeat in range(3)
    )
    arm = tuple(
        f"arm-{arm_index}"
        for _rubric_index in range(4)
        for arm_index in range(4)
        for _repeat in range(3)
    )
    return WildBootstrapProblem(
        restricted_coefficients=np.array([0.0]),
        restricted_bread=np.array([[50.0]]),
        restricted_score_rows=rng.normal(size=(len(arm), 1)),
        restriction=np.array([[1.0]]),
        rubric_clusters=rubric,
        arm_clusters=arm,
        observed_wald=0.01,
    )


def test_enumeration_uses_one_global_sign_representative() -> None:
    schedules = enumerated_rademacher_weights(("a", "b", "c"))

    assert len(schedules) == 4
    assert all(schedule["a"] == 1.0 for schedule in schedules)
    assert {tuple(schedule.values()) for schedule in schedules} == {
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    }


def test_family_bootstrap_rngs_are_stable_and_family_specific() -> None:
    first_draws, first_validation = family_bootstrap_rngs("F1")
    repeated_draws, repeated_validation = family_bootstrap_rngs("F1")
    other_draws, _ = family_bootstrap_rngs("F2")

    np.testing.assert_array_equal(
        first_draws.integers(0, 2**31, size=8),
        repeated_draws.integers(0, 2**31, size=8),
    )
    np.testing.assert_array_equal(
        first_validation.integers(0, 2**31, size=8),
        repeated_validation.integers(0, 2**31, size=8),
    )
    assert not np.array_equal(
        family_bootstrap_rngs("F1")[0].integers(0, 2**31, size=8),
        other_draws.integers(0, 2**31, size=8),
    )


def test_sampled_regime_replenishes_to_requested_valid_count() -> None:
    result = run_wild_cluster_bootstrap(
        _problem(),
        rng=np.random.default_rng(5),
        bootstrap_replicates=5,
    )

    assert result.regime == "sampled"
    assert result.valid_replicates == 5
    assert result.p_value.lower == result.p_value.upper
    assert len(result.attainable_grid) == 6
    assert result.invariant_blocks_passed


def test_enumerated_regime_uses_true_support_and_no_plus_one() -> None:
    result = run_wild_cluster_bootstrap(
        _problem(),
        rng=np.random.default_rng(5),
        bootstrap_replicates=20,
    )

    assert result.regime == "enumerated"
    assert result.attainable_support == 8
    assert result.valid_replicates == 8
    assert result.p_value == PValueBracket(1.0, 1.0)
    assert result.attainable_grid[0] == 1 / 8
    assert result.attainable_grid[-1] == 1.0


def test_enumerated_discards_widen_bracket_on_true_support() -> None:
    rng = np.random.default_rng(4)
    rubric = tuple(
        f"r{rubric_index}"
        for rubric_index in range(3)
        for _arm_index in range(3)
        for _repeat in range(2)
    )
    arm = tuple(
        f"a{arm_index}"
        for _rubric_index in range(3)
        for arm_index in range(3)
        for _repeat in range(2)
    )
    problem = WildBootstrapProblem(
        restricted_coefficients=np.array([0.0]),
        restricted_bread=np.array([[10.0]]),
        restricted_score_rows=rng.normal(size=(18, 1)),
        restriction=np.array([[1.0]]),
        rubric_clusters=rubric,
        arm_clusters=arm,
        observed_wald=0.1,
    )

    result = run_wild_cluster_bootstrap(
        problem,
        rng=np.random.default_rng(1),
        bootstrap_replicates=10,
    )

    assert result.attainable_support == 4
    assert result.valid_replicates == 2
    assert result.singular_replicates == 2
    assert result.p_value == PValueBracket(0.5, 1.0)


def test_full_refit_validation_passes_for_identical_statistic() -> None:
    problem = _problem()
    result = run_wild_cluster_bootstrap(
        problem,
        rng=np.random.default_rng(5),
        bootstrap_replicates=5,
        validation_rng=np.random.default_rng(7),
        full_refit=lambda weights: one_step_replicate(problem, weights).wald,
    )

    assert result.full_refit_validation is not None
    assert result.full_refit_validation.passed
    assert not result.used_full_refit


def test_failed_validation_switches_family_to_full_refit() -> None:
    result = run_wild_cluster_bootstrap(
        _problem(),
        rng=np.random.default_rng(5),
        bootstrap_replicates=5,
        validation_rng=np.random.default_rng(7),
        full_refit=lambda _weights: 0.0,
    )

    assert result.full_refit_validation is not None
    assert not result.full_refit_validation.passed
    assert result.used_full_refit
    assert result.p_value == PValueBracket(1 / 6, 1 / 6)


def test_full_refit_all_one_signs_reproduces_unperturbed_logit_statistic() -> None:
    rng = np.random.default_rng(3)
    rubric = [
        f"r{rubric_index}"
        for rubric_index in range(6)
        for _arm_index in range(5)
        for _repeat in range(8)
    ]
    arm = [
        f"a{arm_index}"
        for _rubric_index in range(6)
        for arm_index in range(5)
        for _repeat in range(8)
    ]
    predictor = rng.normal(size=len(arm))
    design = np.column_stack((np.ones(len(arm)), predictor))
    probabilities = 1.0 / (1.0 + np.exp(-(design @ np.array([-0.2, 0.5]))))
    response = rng.binomial(1, probabilities).astype(float)
    restriction = np.array([[0.0, 1.0]])
    unrestricted = solve_unrestricted_logit(design, response)
    restricted = solve_restricted_logit(
        design, response, restriction, initial=unrestricted.coefficients
    )
    covariance = three_term_cluster_covariance(
        bernoulli_bread(design, unrestricted.coefficients),
        bernoulli_score_rows(design, response, unrestricted.coefficients),
        rubric,
        arm,
    )
    expected = joint_wald_statistic(
        unrestricted.coefficients, covariance.covariance, restriction
    ).value
    full_refit = logit_full_refit_statistic(
        design,
        response,
        restriction,
        rubric,
        arm,
        unrestricted_initial=unrestricted.coefficients,
        restricted_initial=restricted.coefficients,
    )

    assert full_refit({arm_id: 1.0 for arm_id in set(arm)}) == pytest.approx(expected)


def test_bh_uses_inclusive_step_up_threshold() -> None:
    rejected = benjamini_hochberg(
        {"F1": 0.01, "F2": 0.025, "F3": 0.20},
        false_discovery_rate=0.05,
    )

    assert rejected == {"F1", "F2"}


def test_two_corner_bh_marks_spillover_family_indeterminate() -> None:
    result = adjudicate_bh_brackets(
        {
            "bracketed": PValueBracket(0.01, 0.20),
            "spillover": PValueBracket(0.03, 0.03),
            "stable": PValueBracket(0.20, 0.20),
        }
    )

    assert result.lower_rejections == {"bracketed", "spillover"}
    assert result.upper_rejections == frozenset()
    assert result.indeterminate == {"bracketed", "spillover"}
    assert result.definite_non_rejections == {"stable"}


def test_multiplicity_contracts_reject_invalid_values() -> None:
    with pytest.raises(MalformedInputError, match="must lie"):
        benjamini_hochberg({"F1": 0.5}, false_discovery_rate=1.0)
    with pytest.raises(MalformedInputError, match="Invalid p-value"):
        benjamini_hochberg({"F1": np.nan})
    with pytest.raises(MalformedInputError, match="Invalid p-value bracket"):
        adjudicate_bh_brackets({"F1": PValueBracket(0.7, 0.2)})
