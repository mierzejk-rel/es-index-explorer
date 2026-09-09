"""Tests for sampled/enumerated WCR and BH bracket adjudication."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pytest
from scipy.optimize import root

from es_index_explorer.question_analysis import wild_bootstrap as bootstrap_module
from es_index_explorer.question_analysis.cluster_covariance import (
    joint_wald_statistic,
    three_term_cluster_covariance,
)
from es_index_explorer.question_analysis.contracts import AnalysisFlag
from es_index_explorer.question_analysis.errors import (
    MalformedInputError,
    NonFiniteBootstrapReplicateError,
    NonFiniteWaldStatisticError,
    NumericalError,
    SingularRestrictionCovarianceError,
)
from es_index_explorer.question_analysis.glm_primitives import (
    SolverResult,
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
    BootstrapReplicate,
    FullRefitEvaluator,
    FullRefitReplicate,
    WildBootstrapProblem,
    _rerun_with_full_refit,
    _validate_full_refit,
    bootstrap_failure_disclosure,
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


def _typed_full_refit(
    problem: WildBootstrapProblem,
    weights: Mapping[object, float],
    *,
    wald: float,
) -> FullRefitReplicate:
    row_weights = np.asarray([weights[arm] for arm in problem.arm_clusters])
    covariance = three_term_cluster_covariance(
        problem.restricted_bread,
        problem.restricted_score_rows * row_weights[:, None],
        problem.rubric_clusters,
        problem.arm_clusters,
    )
    solution = SolverResult(
        coefficients=problem.restricted_coefficients.copy(),
        iterations=1,
        max_residual=0.0,
    )
    return FullRefitReplicate(
        wald=wald,
        weights_by_arm=dict(weights),
        restricted_solution=solution,
        unrestricted_solution=solution,
        bread=problem.restricted_bread.copy(),
        covariance=covariance,
    )


@dataclass(frozen=True, slots=True)
class _LogitRefitFixture:
    design: np.ndarray
    response: np.ndarray
    restriction: np.ndarray
    rubric: tuple[str, ...]
    arm: tuple[str, ...]
    unrestricted: SolverResult
    restricted: SolverResult
    evaluator: FullRefitEvaluator
    problem: WildBootstrapProblem
    observed_wald: float


def _logit_refit_fixture() -> _LogitRefitFixture:
    rng = np.random.default_rng(3)
    rubric = tuple(
        f"r{rubric_index}"
        for rubric_index in range(6)
        for _arm_index in range(5)
        for _repeat in range(8)
    )
    arm = tuple(
        f"a{arm_index}"
        for _rubric_index in range(6)
        for arm_index in range(5)
        for _repeat in range(8)
    )
    predictor = rng.normal(size=len(arm))
    design = np.column_stack((np.ones(len(arm)), predictor))
    probabilities = 1.0 / (1.0 + np.exp(-(design @ np.array([-0.2, 0.5]))))
    response = rng.binomial(1, probabilities).astype(float)
    restriction = np.array([[0.0, 1.0]])
    unrestricted = solve_unrestricted_logit(design, response)
    restricted = solve_restricted_logit(
        design,
        response,
        restriction,
        initial=unrestricted.coefficients,
    )
    covariance = three_term_cluster_covariance(
        bernoulli_bread(design, unrestricted.coefficients),
        bernoulli_score_rows(design, response, unrestricted.coefficients),
        rubric,
        arm,
    )
    observed_wald = joint_wald_statistic(
        unrestricted.coefficients,
        covariance.covariance,
        restriction,
    ).value
    evaluator = logit_full_refit_statistic(
        design,
        response,
        restriction,
        rubric,
        arm,
        unrestricted_initial=unrestricted.coefficients,
        restricted_initial=restricted.coefficients,
    )
    problem = WildBootstrapProblem(
        restricted_coefficients=restricted.coefficients,
        restricted_bread=bernoulli_bread(design, restricted.coefficients),
        restricted_score_rows=bernoulli_score_rows(
            design,
            response,
            restricted.coefficients,
        ),
        restriction=restriction,
        rubric_clusters=rubric,
        arm_clusters=arm,
        observed_wald=observed_wald,
    )
    return _LogitRefitFixture(
        design=design,
        response=response,
        restriction=restriction,
        rubric=rubric,
        arm=arm,
        unrestricted=unrestricted,
        restricted=restricted,
        evaluator=evaluator,
        problem=problem,
        observed_wald=observed_wald,
    )


def _reference_expit(values: np.ndarray) -> np.ndarray:
    """Compute logistic means without using the production score helper."""
    output = np.empty_like(values, dtype=float)
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponentials = np.exp(values[~nonnegative])
    output[~nonnegative] = exponentials / (1.0 + exponentials)
    return output


def _reference_cluster_meat(
    score_rows: np.ndarray,
    labels: tuple[object, ...],
) -> np.ndarray:
    """Build one finite-sample-corrected cluster meat independently."""
    cluster_sums: dict[object, np.ndarray] = {}
    for label, score in zip(labels, score_rows, strict=True):
        cluster_sums[label] = (
            cluster_sums.get(
                label,
                np.zeros(score_rows.shape[1], dtype=float),
            )
            + score
        )
    cluster_count = len(cluster_sums)
    outer_sum = np.stack(tuple(cluster_sums.values())).T
    return cluster_count / (cluster_count - 1) * (outer_sum @ outer_sum.T)


def _reference_cgm_psd_wald(
    coefficients: np.ndarray,
    bread: np.ndarray,
    score_rows: np.ndarray,
    restriction: np.ndarray,
    rubric: tuple[object, ...],
    arm: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute CGM meat, PSD covariance, and Wald without production helpers."""
    rubric_meat = _reference_cluster_meat(score_rows, rubric)
    arm_meat = _reference_cluster_meat(score_rows, arm)
    intersections = tuple(zip(rubric, arm, strict=True))
    intersection_meat = _reference_cluster_meat(score_rows, intersections)
    combined_meat = rubric_meat + arm_meat - intersection_meat
    inverse_bread = np.linalg.inv(bread)
    raw_covariance = inverse_bread @ combined_meat @ inverse_bread.T
    raw_covariance = (raw_covariance + raw_covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(raw_covariance)
    projected_covariance = (
        eigenvectors * np.where(eigenvalues <= 1e-10, 0.0, eigenvalues)
    ) @ eigenvectors.T
    projected_covariance = (projected_covariance + projected_covariance.T) / 2.0
    contrast = restriction @ coefficients
    restriction_covariance = restriction @ projected_covariance @ restriction.T
    wald = float(contrast.T @ np.linalg.solve(restriction_covariance, contrast))
    return (
        rubric_meat,
        arm_meat,
        intersection_meat,
        raw_covariance,
        projected_covariance,
        wald,
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
    assert result.one_step_invariant_blocks_passed
    assert result.failure_disclosure.combined_count == 0
    assert result.failure_disclosure.denominator == 5
    assert result.failure_disclosure.rate == 0.0
    assert not result.failure_disclosure.triggered


def test_sampled_plus_one_formula_handles_zero_and_all_exceedances() -> None:
    problem = _problem()
    no_exceedances = run_wild_cluster_bootstrap(
        replace(problem, observed_wald=1e12),
        rng=np.random.default_rng(5),
        bootstrap_replicates=5,
    )
    all_exceedances = run_wild_cluster_bootstrap(
        replace(problem, observed_wald=0.0),
        rng=np.random.default_rng(5),
        bootstrap_replicates=5,
    )

    assert no_exceedances.p_value == PValueBracket(1 / 6, 1 / 6)
    assert all_exceedances.p_value == PValueBracket(1.0, 1.0)


def test_sampled_exceedance_rule_is_inclusive_on_exact_tie() -> None:
    problem = _problem()
    arms = tuple(dict.fromkeys(problem.arm_clusters))
    probe_rng = np.random.default_rng(77)
    signs = probe_rng.choice(np.array([-1.0, 1.0]), size=len(arms), replace=True)
    weights = {arm: float(sign) for arm, sign in zip(arms, signs, strict=True)}
    tied_wald = one_step_replicate(problem, weights).wald

    result = run_wild_cluster_bootstrap(
        replace(problem, observed_wald=tied_wald),
        rng=np.random.default_rng(77),
        bootstrap_replicates=1,
    )

    assert result.p_value == PValueBracket(1.0, 1.0)


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
    assert result.failure_disclosure.singular_count == 2
    assert result.failure_disclosure.non_finite_count == 0
    assert result.failure_disclosure.combined_count == 2
    assert result.failure_disclosure.denominator == result.attainable_support == 4
    assert result.failure_disclosure.rate == 0.5
    assert result.failure_disclosure.triggered


def test_failure_disclosure_uses_exact_strict_one_percent_boundary() -> None:
    above = bootstrap_failure_disclosure(
        singular_count=1,
        non_finite_count=0,
        denominator=99,
    )
    at_boundary = bootstrap_failure_disclosure(
        singular_count=0,
        non_finite_count=1,
        denominator=100,
    )
    none = bootstrap_failure_disclosure(
        singular_count=0,
        non_finite_count=0,
        denominator=99,
    )

    assert above.rate == 1 / 99
    assert above.triggered
    assert at_boundary.rate == 0.01
    assert not at_boundary.triggered
    assert none.rate == 0.0
    assert not none.triggered


def test_sampled_replenishment_keeps_attempts_out_of_disclosure_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(42)
    rubric = tuple(
        f"r{rubric_index}"
        for rubric_index in range(4)
        for _arm_index in range(8)
        for _repeat in range(2)
    )
    arm = tuple(
        f"a{arm_index}"
        for _rubric_index in range(4)
        for arm_index in range(8)
        for _repeat in range(2)
    )
    problem = WildBootstrapProblem(
        restricted_coefficients=np.array([0.0]),
        restricted_bread=np.array([[50.0]]),
        restricted_score_rows=rng.normal(size=(len(arm), 1)),
        restriction=np.array([[1.0]]),
        rubric_clusters=rubric,
        arm_clusters=arm,
        observed_wald=0.01,
    )
    original = bootstrap_module.one_step_replicate
    call_count = 0

    def fail_once(
        current_problem: WildBootstrapProblem,
        weights: Mapping[object, float],
    ) -> BootstrapReplicate:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise SingularRestrictionCovarianceError(
                "Restricted covariance is singular after PSD projection"
            )
        return original(current_problem, weights)

    monkeypatch.setattr(bootstrap_module, "one_step_replicate", fail_once)
    result = run_wild_cluster_bootstrap(
        problem,
        rng=np.random.default_rng(9),
        bootstrap_replicates=99,
    )

    assert result.regime == "sampled"
    assert result.valid_replicates == 99
    assert result.attempted_replicates == 100
    assert result.failure_disclosure.denominator == 99
    assert result.failure_disclosure.rate == 1 / 99
    assert result.failure_disclosure.triggered


@pytest.mark.parametrize(
    ("singular_count", "non_finite_count", "denominator", "message"),
    (
        (-1, 0, 100, "counts must be non-negative"),
        (0, -1, 100, "counts must be non-negative"),
        (0, 0, 0, "denominator must be positive"),
        (0, 0, -1, "denominator must be positive"),
    ),
)
def test_failure_disclosure_rejects_invalid_contract(
    singular_count: int,
    non_finite_count: int,
    denominator: int,
    message: str,
) -> None:
    with pytest.raises(MalformedInputError, match=message):
        bootstrap_failure_disclosure(
            singular_count=singular_count,
            non_finite_count=non_finite_count,
            denominator=denominator,
        )


def test_every_enumerated_discard_is_non_computable() -> None:
    base_problem = _problem()
    problem = replace(
        base_problem,
        restricted_score_rows=np.zeros_like(base_problem.restricted_score_rows),
        observed_wald=0.0,
    )

    with pytest.raises(NumericalError, match="Every enumerated bootstrap vector"):
        run_wild_cluster_bootstrap(
            problem,
            rng=np.random.default_rng(1),
            bootstrap_replicates=20,
        )


def test_failed_validation_switches_family_to_full_refit() -> None:
    problem = _problem()
    result = run_wild_cluster_bootstrap(
        problem,
        rng=np.random.default_rng(5),
        bootstrap_replicates=5,
        validation_rng=np.random.default_rng(7),
        full_refit=lambda weights: _typed_full_refit(problem, weights, wald=0.0),
    )

    assert result.full_refit_validation is not None
    assert not result.full_refit_validation.passed
    assert result.used_full_refit
    assert result.p_value == PValueBracket(1 / 6, 1 / 6)
    assert result.one_step_invariant_blocks_passed


def test_full_refit_validation_uses_exact_indicator_and_discrepancy_thresholds() -> (
    None
):
    problem = _problem()
    arm_weights = {arm: 1.0 for arm in set(problem.arm_clusters)}
    template = _typed_full_refit(problem, arm_weights, wald=1.001)
    one_step = one_step_replicate(problem, arm_weights)
    replicates = tuple(
        BootstrapReplicate(
            wald=1.001,
            weights_by_arm={"draw_index": float(index)},
            meat=one_step.meat,
        )
        for index in range(5_000)
    )
    seed = 41
    selected = {
        int(index)
        for index in np.random.default_rng(seed).choice(
            len(replicates),
            size=100,
            replace=False,
        )
    }
    mismatches = set(sorted(selected)[:1])

    def evaluator(weights: Mapping[object, float]) -> FullRefitReplicate:
        draw_index = int(weights["draw_index"])
        return replace(
            template,
            wald=0.999 if draw_index in mismatches else 1.001,
            weights_by_arm=dict(weights),
        )

    passing = _validate_full_refit(
        replicates,
        evaluator,
        np.random.default_rng(seed),
        observed_wald=1.0,
    )
    assert passing.indicator_agreement == 0.99
    assert passing.maximum_relative_discrepancy < 0.01
    assert passing.passed

    mismatches.add(sorted(selected)[1])
    failing = _validate_full_refit(
        replicates,
        evaluator,
        np.random.default_rng(seed),
        observed_wald=1.0,
    )
    assert failing.indicator_agreement == 0.98
    assert not failing.passed


def test_full_refit_validation_handles_zero_wald_without_division() -> None:
    problem = _problem()
    weights = {arm: 1.0 for arm in set(problem.arm_clusters)}
    template = _typed_full_refit(problem, weights, wald=0.0)
    one_step = one_step_replicate(problem, weights)
    zero_replicate = replace(one_step, wald=0.0)

    exact = _validate_full_refit(
        (zero_replicate,),
        lambda _weights: template,
        np.random.default_rng(1),
        observed_wald=0.0,
    )
    assert exact.maximum_relative_discrepancy == 0.0
    assert exact.passed

    nonzero_replicate = replace(one_step, wald=1.0)
    divergent = _validate_full_refit(
        (nonzero_replicate,),
        lambda _weights: template,
        np.random.default_rng(1),
        observed_wald=0.0,
    )
    assert divergent.maximum_relative_discrepancy == float("inf")
    assert not divergent.passed


def test_full_refit_rerun_classifies_failures_without_placeholder_meat() -> None:
    problem = _problem()
    weights = {arm: 1.0 for arm in set(problem.arm_clusters)}
    template = _typed_full_refit(problem, weights, wald=2.0)

    def evaluator(schedule: Mapping[object, float]) -> FullRefitReplicate:
        kind = int(schedule["kind"])
        if kind == 1:
            raise SingularRestrictionCovarianceError(
                "Restricted covariance is singular after PSD projection"
            )
        if kind == 2:
            raise NonFiniteBootstrapReplicateError(
                "Full-refit Wald statistic is non-finite"
            )
        return replace(template, weights_by_arm=dict(schedule))

    schedules: tuple[Mapping[object, float], ...] = (
        {"kind": 1.0},
        {"kind": 2.0},
        {"kind": 3.0},
        {"kind": 4.0},
    )
    valid, singular, non_finite, attempted = _rerun_with_full_refit(
        schedules,
        evaluator,
        target_valid=1,
    )

    assert len(valid) == 1
    assert valid[0].wald == 2.0
    assert singular == 1
    assert non_finite == 1
    assert attempted == 3


def test_unexpected_numerical_error_is_not_silently_reclassified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_unexpectedly(
        _problem: WildBootstrapProblem,
        _weights: Mapping[object, float],
    ) -> BootstrapReplicate:
        raise NumericalError("Unexpected bootstrap implementation failure")

    monkeypatch.setattr(
        bootstrap_module,
        "one_step_replicate",
        fail_unexpectedly,
    )

    with pytest.raises(
        NumericalError,
        match="Unexpected bootstrap implementation failure",
    ):
        run_wild_cluster_bootstrap(
            _problem(),
            rng=np.random.default_rng(5),
            bootstrap_replicates=5,
        )

    def full_refit_failure(
        _weights: Mapping[object, float],
    ) -> FullRefitReplicate:
        raise NumericalError("Unexpected full-refit implementation failure")

    with pytest.raises(
        NumericalError,
        match="Unexpected full-refit implementation failure",
    ):
        unexpected_schedules: tuple[Mapping[object, float], ...] = ({"arm": 1.0},)
        _rerun_with_full_refit(
            unexpected_schedules,
            full_refit_failure,
        )


def test_sampled_full_refit_fallback_replenishes_typed_failures() -> None:
    problem = _problem()
    weights = {arm: 1.0 for arm in set(problem.arm_clusters)}
    template = _typed_full_refit(problem, weights, wald=0.0)
    call_count = 0

    def evaluator(schedule: Mapping[object, float]) -> FullRefitReplicate:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise NonFiniteBootstrapReplicateError(
                "Full-refit Wald statistic is non-finite"
            )
        return replace(template, weights_by_arm=dict(schedule))

    result = run_wild_cluster_bootstrap(
        problem,
        rng=np.random.default_rng(5),
        bootstrap_replicates=5,
        validation_rng=np.random.default_rng(7),
        full_refit=evaluator,
    )

    assert result.regime == "sampled"
    assert result.used_full_refit
    assert result.valid_replicates == 5
    assert result.attempted_replicates == 6
    assert result.singular_replicates == 0
    assert result.non_finite_replicates == 1
    assert result.failure_disclosure.denominator == 5
    assert result.failure_disclosure.rate == 0.2
    assert result.p_value == PValueBracket(1 / 6, 1 / 6)


def test_sampled_full_refit_fallback_fails_after_replenishment_cap() -> None:
    problem = _problem()
    weights = {arm: 1.0 for arm in set(problem.arm_clusters)}
    template = _typed_full_refit(problem, weights, wald=0.0)
    call_count = 0

    def evaluator(schedule: Mapping[object, float]) -> FullRefitReplicate:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return replace(template, weights_by_arm=dict(schedule))
        raise NonFiniteBootstrapReplicateError(
            "Full-refit Wald statistic is non-finite"
        )

    with pytest.raises(
        NumericalError,
        match="could not replenish the requested valid replicates",
    ):
        run_wild_cluster_bootstrap(
            problem,
            rng=np.random.default_rng(5),
            bootstrap_replicates=5,
            validation_rng=np.random.default_rng(7),
            full_refit=evaluator,
        )


def test_full_refit_all_one_signs_reproduces_unperturbed_logit_statistic() -> None:
    fixture = _logit_refit_fixture()
    weights: dict[object, float] = {arm_id: 1.0 for arm_id in set(fixture.arm)}
    replicate = fixture.evaluator(weights)

    assert replicate.wald == pytest.approx(fixture.observed_wald)
    np.testing.assert_allclose(
        fixture.restriction @ replicate.restricted_solution.coefficients,
        0.0,
        atol=1e-10,
    )
    assert replicate.restricted_solution.max_residual < 1e-8
    assert replicate.unrestricted_solution.max_residual < 1e-8
    np.testing.assert_allclose(
        replicate.bread,
        bernoulli_bread(
            fixture.design,
            replicate.unrestricted_solution.coefficients,
        ),
    )


def test_one_step_translates_non_finite_wald_to_discardable_replicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    weights = {arm: 1.0 for arm in set(problem.arm_clusters)}

    def raise_non_finite_wald(
        _coefficients: np.ndarray,
        _covariance: np.ndarray,
        _restriction: np.ndarray,
    ) -> None:
        raise NonFiniteWaldStatisticError("Unexpected primitive wording")

    monkeypatch.setattr(
        bootstrap_module,
        "joint_wald_statistic",
        raise_non_finite_wald,
    )

    with pytest.raises(
        NonFiniteBootstrapReplicateError,
        match="One-step Wald statistic is non-finite",
    ) as raised:
        one_step_replicate(problem, weights)

    assert isinstance(raised.value.__cause__, NonFiniteWaldStatisticError)


def test_full_refit_translates_non_finite_wald_to_discardable_replicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _logit_refit_fixture()
    weights: dict[object, float] = {arm_id: 1.0 for arm_id in set(fixture.arm)}

    def raise_non_finite_wald(
        _coefficients: np.ndarray,
        _covariance: np.ndarray,
        _restriction: np.ndarray,
    ) -> None:
        raise NonFiniteWaldStatisticError("Unexpected primitive wording")

    monkeypatch.setattr(
        bootstrap_module,
        "joint_wald_statistic",
        raise_non_finite_wald,
    )

    with pytest.raises(
        NonFiniteBootstrapReplicateError,
        match="Full-refit Wald statistic is non-finite",
    ) as raised:
        fixture.evaluator(weights)

    assert isinstance(raised.value.__cause__, NonFiniteWaldStatisticError)


def test_nontrivial_full_refit_matches_independent_scipy_root() -> None:
    fixture = _logit_refit_fixture()
    weights: dict[object, float] = {
        "a0": 1.0,
        "a1": -1.0,
        "a2": 1.0,
        "a3": -1.0,
        "a4": 1.0,
    }
    row_weights = np.asarray([weights[arm] for arm in fixture.arm])

    def weighted_score(coefficients: np.ndarray) -> np.ndarray:
        means = _reference_expit(fixture.design @ coefficients)
        return fixture.design.T @ (row_weights * (fixture.response - means))

    def restricted_equations(parameters: np.ndarray) -> np.ndarray:
        coefficients = parameters[: fixture.design.shape[1]]
        multiplier = parameters[fixture.design.shape[1] :]
        return np.concatenate(
            (
                weighted_score(coefficients) - fixture.restriction.T @ multiplier,
                fixture.restriction @ coefficients,
            )
        )

    restricted_reference = root(
        restricted_equations,
        np.concatenate((fixture.restricted.coefficients, np.zeros(1))),
        method="hybr",
        tol=1e-11,
    )
    unrestricted_reference = root(
        weighted_score,
        fixture.unrestricted.coefficients,
        method="hybr",
        tol=1e-11,
    )
    assert restricted_reference.success
    assert unrestricted_reference.success

    replicate = fixture.evaluator(weights)

    np.testing.assert_allclose(
        replicate.restricted_solution.coefficients,
        restricted_reference.x[: fixture.design.shape[1]],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        replicate.unrestricted_solution.coefficients,
        unrestricted_reference.x,
        rtol=1e-8,
        atol=1e-10,
    )
    reference_means = _reference_expit(fixture.design @ unrestricted_reference.x)
    expected_scores = (
        fixture.design * (row_weights * (fixture.response - reference_means))[:, None]
    )
    expected_bread = fixture.design.T @ (
        fixture.design * (reference_means * (1.0 - reference_means))[:, None]
    )
    (
        rubric_meat,
        arm_meat,
        intersection_meat,
        raw_covariance,
        projected_covariance,
        expected_wald,
    ) = _reference_cgm_psd_wald(
        unrestricted_reference.x,
        expected_bread,
        expected_scores,
        fixture.restriction,
        fixture.rubric,
        fixture.arm,
    )
    np.testing.assert_allclose(replicate.bread, expected_bread, rtol=1e-8)
    np.testing.assert_allclose(
        replicate.covariance.meat.rubric,
        rubric_meat,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        replicate.covariance.meat.arm,
        arm_meat,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        replicate.covariance.meat.intersection,
        intersection_meat,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        replicate.covariance.meat.combined,
        rubric_meat + arm_meat - intersection_meat,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        replicate.covariance.unprojected_covariance,
        raw_covariance,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        replicate.covariance.covariance,
        projected_covariance,
        rtol=1e-8,
        atol=1e-10,
    )
    assert replicate.weights_by_arm == weights
    assert replicate.wald == pytest.approx(expected_wald, rel=1e-8)


def test_actual_full_refit_fallback_keeps_one_step_invariant_separate() -> None:
    fixture = _logit_refit_fixture()
    all_positive: dict[object, float] = {f"a{index}": 1.0 for index in range(5)}
    alternating: dict[object, float] = {
        "a0": 1.0,
        "a1": -1.0,
        "a2": 1.0,
        "a3": -1.0,
        "a4": 1.0,
    }
    positive_refit = fixture.evaluator(all_positive)
    alternating_refit = fixture.evaluator(alternating)
    assert not np.allclose(
        positive_refit.covariance.meat.arm,
        alternating_refit.covariance.meat.arm,
        rtol=1e-9,
        atol=0.0,
    )

    result = run_wild_cluster_bootstrap(
        fixture.problem,
        rng=np.random.default_rng(1),
        bootstrap_replicates=5,
        validation_rng=np.random.default_rng(7),
        full_refit=fixture.evaluator,
    )

    assert result.used_full_refit
    assert result.full_refit_validation is not None
    assert not result.full_refit_validation.passed
    assert result.one_step_invariant_blocks_passed
    assert result.p_value == PValueBracket(1 / 3, 1 / 3)


def test_bh_uses_inclusive_step_up_threshold() -> None:
    rejected = benjamini_hochberg(
        {"F1": 0.01, "F2": 0.025, "F3": 0.20},
        false_discovery_rate=0.05,
    )

    assert rejected == {"F1", "F2"}


def test_bh_equal_p_values_are_independent_of_input_order() -> None:
    forward = {
        "F1": 0.01,
        "F2": 0.025,
        "F3": 0.025,
        "F4": 0.90,
    }
    reverse = dict(reversed(tuple(forward.items())))

    assert benjamini_hochberg(forward) == {"F1", "F2", "F3"}
    assert benjamini_hochberg(reverse) == {"F1", "F2", "F3"}
    forward_decisions = adjudicate_bh_brackets(
        {family: PValueBracket(p_value, p_value) for family, p_value in forward.items()}
    )
    reverse_decisions = adjudicate_bh_brackets(
        {family: PValueBracket(p_value, p_value) for family, p_value in reverse.items()}
    )
    assert forward_decisions == reverse_decisions


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
    decisions = {decision.family_id: decision for decision in result.family_decisions}
    assert tuple(decision.family_id for decision in result.family_decisions) == (
        "bracketed",
        "spillover",
        "stable",
    )
    assert decisions["bracketed"].rejected is None
    assert decisions["bracketed"].analysis_flag is AnalysisFlag.BH_INDETERMINATE
    assert decisions["spillover"].rejected is None
    assert decisions["spillover"].analysis_flag is AnalysisFlag.BH_INDETERMINATE
    assert decisions["stable"].rejected is False
    assert decisions["stable"].analysis_flag is None


def test_identical_bh_corners_produce_definite_typed_decisions() -> None:
    result = adjudicate_bh_brackets(
        {
            "F2": PValueBracket(0.20, 0.20),
            "F1": PValueBracket(0.01, 0.01),
        }
    )

    assert result.indeterminate == frozenset()
    assert result.definite_rejections == {"F1"}
    assert result.definite_non_rejections == {"F2"}
    assert result.family_decisions[0].family_id == "F1"
    assert result.family_decisions[0].rejected is True
    assert result.family_decisions[0].analysis_flag is None
    assert result.family_decisions[1].family_id == "F2"
    assert result.family_decisions[1].rejected is False
    assert result.family_decisions[1].analysis_flag is None


def test_empty_bh_family_has_no_decisions() -> None:
    result = adjudicate_bh_brackets({})

    assert result.lower_rejections == frozenset()
    assert result.upper_rejections == frozenset()
    assert result.family_decisions == ()


def test_multiplicity_contracts_reject_invalid_values() -> None:
    with pytest.raises(MalformedInputError, match="must lie"):
        benjamini_hochberg({"F1": 0.5}, false_discovery_rate=1.0)
    with pytest.raises(MalformedInputError, match="Invalid p-value"):
        benjamini_hochberg({"F1": np.nan})
    with pytest.raises(MalformedInputError, match="Invalid p-value bracket"):
        adjudicate_bh_brackets({"F1": PValueBracket(0.7, 0.2)})
