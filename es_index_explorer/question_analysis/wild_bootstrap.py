"""Restricted arm-clustered wild score bootstrap primitives."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from es_index_explorer.question_analysis.cluster_covariance import (
    ClusterCovariance,
    ClusterMeat,
    joint_wald_statistic,
    three_term_cluster_covariance,
)
from es_index_explorer.question_analysis.errors import (
    MalformedInputError,
    NonFiniteBootstrapReplicateError,
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
from es_index_explorer.question_analysis.multiplicity import PValueBracket
from es_index_explorer.question_analysis.seeds import derive_child_seed

DEFAULT_BOOTSTRAP_REPLICATES = 9_999
MAX_REPLENISHMENT_MULTIPLIER = 10
FULL_REFIT_FRACTION = 0.02
FULL_REFIT_INDICATOR_AGREEMENT = 0.99
FULL_REFIT_MAX_RELATIVE_DISCREPANCY = 0.01
INVARIANT_BLOCK_RTOL = 1e-9

FullRefitEvaluator = Callable[[Mapping[object, float]], "FullRefitReplicate"]


def family_bootstrap_rngs(
    family_id: str,
) -> tuple[np.random.Generator, np.random.Generator]:
    """Create deterministic draw and validation RNGs for one family."""
    if not family_id:
        raise MalformedInputError("Family ID must be non-empty")
    return (
        np.random.default_rng(
            derive_child_seed("glm_bootstrap", f"{family_id}:replicates")
        ),
        np.random.default_rng(
            derive_child_seed("glm_bootstrap", f"{family_id}:full-refit-validation")
        ),
    )


@dataclass(frozen=True, slots=True)
class WildBootstrapProblem:
    """Define the restricted-score objects required by the WCR loop."""

    restricted_coefficients: np.ndarray
    restricted_bread: np.ndarray
    restricted_score_rows: np.ndarray
    restriction: np.ndarray
    rubric_clusters: tuple[object, ...]
    arm_clusters: tuple[object, ...]
    observed_wald: float


@dataclass(frozen=True, slots=True)
class BootstrapReplicate:
    """Store one valid bootstrap statistic and its meat blocks."""

    wald: float
    weights_by_arm: Mapping[object, float]
    meat: ClusterMeat


@dataclass(frozen=True, slots=True)
class FullRefitReplicate:
    """Store one actual restricted-then-unrestricted full-refit result."""

    wald: float
    weights_by_arm: Mapping[object, float]
    restricted_solution: SolverResult
    unrestricted_solution: SolverResult
    bread: np.ndarray
    covariance: ClusterCovariance


@dataclass(frozen=True, slots=True)
class FullRefitValidation:
    """Store the one-step/full-refit validation outcome."""

    replicate_count: int
    indicator_agreement: float
    maximum_relative_discrepancy: float
    passed: bool


@dataclass(frozen=True, slots=True)
class BootstrapFailureDisclosure:
    """Store the frozen combined bootstrap-failure disclosure calculation."""

    singular_count: int
    non_finite_count: int
    combined_count: int
    denominator: int
    rate: float
    triggered: bool


@dataclass(frozen=True, slots=True)
class WildBootstrapResult:
    """Store sampled or enumerated WCR output and diagnostics."""

    regime: str
    p_value: PValueBracket
    valid_replicates: int
    attempted_replicates: int
    singular_replicates: int
    non_finite_replicates: int
    arm_cluster_count: int
    attainable_support: int
    attainable_grid: tuple[float, ...]
    failure_disclosure: BootstrapFailureDisclosure
    one_step_invariant_blocks_passed: bool
    full_refit_validation: FullRefitValidation | None
    used_full_refit: bool


def _validate_problem(problem: WildBootstrapProblem) -> tuple[object, ...]:
    beta = np.asarray(problem.restricted_coefficients, dtype=float)
    bread = np.asarray(problem.restricted_bread, dtype=float)
    scores = np.asarray(problem.restricted_score_rows, dtype=float)
    restriction = np.asarray(problem.restriction, dtype=float)
    if (
        beta.ndim != 1
        or bread.shape != (len(beta), len(beta))
        or scores.ndim != 2
        or scores.shape[1] != len(beta)
        or restriction.ndim != 2
        or restriction.shape[1] != len(beta)
        or len(problem.rubric_clusters) != len(scores)
        or len(problem.arm_clusters) != len(scores)
        or not all(
            np.isfinite(value).all() for value in (beta, bread, scores, restriction)
        )
        or not np.isfinite(problem.observed_wald)
    ):
        raise MalformedInputError(
            "Wild-bootstrap problem has invalid dimensions or values"
        )
    arms = tuple(dict.fromkeys(problem.arm_clusters))
    if len(arms) <= 1:
        raise NumericalError(
            "Wild bootstrap requires at least two realised arm clusters"
        )
    return arms


def _row_weights(
    arm_clusters: Sequence[object], weights_by_arm: Mapping[object, float]
) -> np.ndarray:
    try:
        values = np.asarray(
            [weights_by_arm[cluster] for cluster in arm_clusters], dtype=float
        )
    except KeyError as error:
        raise MalformedInputError("Missing a realised arm bootstrap weight") from error
    if not np.isfinite(values).all():
        raise MalformedInputError("Bootstrap weights must be finite")
    return values


def one_step_replicate(
    problem: WildBootstrapProblem,
    weights_by_arm: Mapping[object, float],
) -> BootstrapReplicate:
    """Compute one fixed-bread, studentized restricted-score replicate."""
    _validate_problem(problem)
    weights = _row_weights(problem.arm_clusters, weights_by_arm)
    perturbed_scores = problem.restricted_score_rows * weights[:, None]
    score_sum = perturbed_scores.sum(axis=0)
    try:
        update = np.linalg.solve(problem.restricted_bread, score_sum)
    except np.linalg.LinAlgError as error:
        raise NumericalError("One-step restricted bread is singular") from error
    coefficients = problem.restricted_coefficients + update
    if not np.isfinite(coefficients).all():
        raise NonFiniteBootstrapReplicateError("One-step coefficients are non-finite")
    covariance = three_term_cluster_covariance(
        problem.restricted_bread,
        perturbed_scores,
        problem.rubric_clusters,
        problem.arm_clusters,
    )
    wald = joint_wald_statistic(
        coefficients, covariance.covariance, problem.restriction
    ).value
    if not np.isfinite(wald):
        raise NonFiniteBootstrapReplicateError("One-step Wald statistic is non-finite")
    return BootstrapReplicate(wald, dict(weights_by_arm), covariance.meat)


def enumerated_rademacher_weights(
    arms: Sequence[object],
) -> tuple[dict[object, float], ...]:
    """Enumerate one representative from each global sign-flip pair."""
    unique_arms = tuple(dict.fromkeys(arms))
    if not unique_arms:
        raise MalformedInputError("Cannot enumerate weights without arms")
    return tuple(
        {
            unique_arms[0]: 1.0,
            **{
                arm: float(sign)
                for arm, sign in zip(unique_arms[1:], signs, strict=True)
            },
        }
        for signs in product((-1, 1), repeat=len(unique_arms) - 1)
    )


def _sampled_rademacher_weights(
    arms: Sequence[object], rng: np.random.Generator
) -> dict[object, float]:
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(arms), replace=True)
    return {arm: float(sign) for arm, sign in zip(arms, signs, strict=True)}


def bootstrap_failure_disclosure(
    *,
    singular_count: int,
    non_finite_count: int,
    denominator: int,
) -> BootstrapFailureDisclosure:
    """Compute the strict one-percent combined bootstrap-failure disclosure."""
    if singular_count < 0 or non_finite_count < 0:
        raise MalformedInputError("Bootstrap failure counts must be non-negative")
    if denominator <= 0:
        raise MalformedInputError("Bootstrap failure denominator must be positive")
    combined_count = singular_count + non_finite_count
    rate = combined_count / denominator
    return BootstrapFailureDisclosure(
        singular_count=singular_count,
        non_finite_count=non_finite_count,
        combined_count=combined_count,
        denominator=denominator,
        rate=rate,
        triggered=rate > 0.01,
    )


def _invariant_blocks(replicates: Sequence[BootstrapReplicate]) -> bool:
    if not replicates:
        return False
    first = replicates[0].meat
    return all(
        np.allclose(
            replicate.meat.arm,
            first.arm,
            rtol=INVARIANT_BLOCK_RTOL,
            atol=0.0,
        )
        and np.allclose(
            replicate.meat.intersection,
            first.intersection,
            rtol=INVARIANT_BLOCK_RTOL,
            atol=0.0,
        )
        for replicate in replicates[1:]
    )


def _validate_full_refit(
    replicates: Sequence[BootstrapReplicate],
    full_refit: FullRefitEvaluator,
    validation_rng: np.random.Generator,
    observed_wald: float,
) -> FullRefitValidation:
    count = max(1, int(np.ceil(FULL_REFIT_FRACTION * len(replicates))))
    indices = np.sort(validation_rng.choice(len(replicates), size=count, replace=False))
    agreements: list[bool] = []
    discrepancies: list[float] = []
    for index in indices:
        replicate = replicates[int(index)]
        full_wald = full_refit(replicate.weights_by_arm).wald
        if not np.isfinite(full_wald):
            raise NumericalError("Full-refit Wald statistic is non-finite")
        agreements.append(
            (replicate.wald >= observed_wald) == (full_wald >= observed_wald)
        )
        denominator = abs(full_wald)
        discrepancy = (
            abs(replicate.wald - full_wald) / denominator
            if denominator > 0
            else (0.0 if replicate.wald == full_wald else float("inf"))
        )
        discrepancies.append(discrepancy)
    agreement = float(np.mean(agreements))
    maximum = float(max(discrepancies))
    return FullRefitValidation(
        replicate_count=count,
        indicator_agreement=agreement,
        maximum_relative_discrepancy=maximum,
        passed=(
            agreement >= FULL_REFIT_INDICATOR_AGREEMENT
            and maximum < FULL_REFIT_MAX_RELATIVE_DISCREPANCY
        ),
    )


def _rerun_with_full_refit(
    schedules: Sequence[Mapping[object, float]],
    full_refit: FullRefitEvaluator,
    *,
    target_valid: int | None = None,
) -> tuple[list[FullRefitReplicate], int, int, int]:
    valid: list[FullRefitReplicate] = []
    singular = 0
    non_finite = 0
    attempted = 0
    for weights in schedules:
        if target_valid is not None and len(valid) == target_valid:
            break
        attempted += 1
        try:
            replicate = full_refit(weights)
            if not np.isfinite(replicate.wald):
                raise NonFiniteBootstrapReplicateError(
                    "Full-refit Wald statistic is non-finite"
                )
            valid.append(replicate)
        except SingularRestrictionCovarianceError:
            singular += 1
        except NonFiniteBootstrapReplicateError:
            non_finite += 1
    return valid, singular, non_finite, attempted


def logit_full_refit_statistic(
    design: np.ndarray,
    response: np.ndarray,
    restriction: np.ndarray,
    rubric_clusters: Sequence[object],
    arm_clusters: Sequence[object],
    *,
    unrestricted_initial: np.ndarray,
    restricted_initial: np.ndarray,
) -> FullRefitEvaluator:
    """Build the frozen restricted-then-unrestricted perturbed-score evaluator."""
    matrix = np.asarray(design, dtype=float)
    outcome = np.asarray(response, dtype=float)
    constraints = np.asarray(restriction, dtype=float)
    unrestricted_start = np.asarray(unrestricted_initial, dtype=float)
    restricted_start = np.asarray(restricted_initial, dtype=float)
    arms = tuple(arm_clusters)
    rubrics = tuple(rubric_clusters)

    def statistic(weights_by_arm: Mapping[object, float]) -> FullRefitReplicate:
        row_weights = _row_weights(arms, weights_by_arm)
        restricted = solve_restricted_logit(
            matrix,
            outcome,
            constraints,
            initial=restricted_start,
            estimating_weights=row_weights,
        )
        unrestricted = solve_unrestricted_logit(
            matrix,
            outcome,
            initial=unrestricted_start,
            estimating_weights=row_weights,
        )
        score_rows = bernoulli_score_rows(
            matrix,
            outcome,
            unrestricted.coefficients,
            estimating_weights=row_weights,
        )
        # The full-refit bread is recomputed at beta_hat*; signs perturb scores,
        # not the Bernoulli information definition.
        bread = bernoulli_bread(matrix, unrestricted.coefficients)
        covariance = three_term_cluster_covariance(bread, score_rows, rubrics, arms)
        wald = joint_wald_statistic(
            unrestricted.coefficients, covariance.covariance, constraints
        ).value
        return FullRefitReplicate(
            wald=wald,
            weights_by_arm=dict(weights_by_arm),
            restricted_solution=restricted,
            unrestricted_solution=unrestricted,
            bread=bread,
            covariance=covariance,
        )

    return statistic


def run_wild_cluster_bootstrap(
    problem: WildBootstrapProblem,
    *,
    rng: np.random.Generator,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    validation_rng: np.random.Generator | None = None,
    full_refit: FullRefitEvaluator | None = None,
) -> WildBootstrapResult:
    """Run the frozen sampled-or-enumerated arm-clustered restricted WCR."""
    arms = _validate_problem(problem)
    if bootstrap_replicates <= 0:
        raise MalformedInputError("Bootstrap replicate count must be positive")
    support = 2 ** (len(arms) - 1)
    regime = "enumerated" if support <= bootstrap_replicates else "sampled"
    schedules: Sequence[dict[object, float]]
    if regime == "enumerated":
        schedules = enumerated_rademacher_weights(arms)
        maximum_attempts = len(schedules)
        target_valid = len(schedules)
    else:
        schedules = ()
        maximum_attempts = MAX_REPLENISHMENT_MULTIPLIER * bootstrap_replicates
        target_valid = bootstrap_replicates

    valid: list[BootstrapReplicate] = []
    attempted_schedules: list[dict[object, float]] = []
    singular = 0
    non_finite = 0
    attempts = 0
    while attempts < maximum_attempts and (
        attempts < len(schedules)
        if regime == "enumerated"
        else len(valid) < target_valid
    ):
        weights = (
            schedules[attempts]
            if regime == "enumerated"
            else _sampled_rademacher_weights(arms, rng)
        )
        attempts += 1
        attempted_schedules.append(weights)
        try:
            valid.append(one_step_replicate(problem, weights))
        except SingularRestrictionCovarianceError:
            singular += 1
        except NonFiniteBootstrapReplicateError:
            non_finite += 1

    if regime == "sampled" and len(valid) != bootstrap_replicates:
        raise NumericalError(
            f"Wild bootstrap could not replenish {bootstrap_replicates} valid replicates"
        )

    one_step_invariant = _invariant_blocks(valid)
    if valid and not one_step_invariant:
        raise NumericalError(
            "Arm or intersection covariance block varied across replicates"
        )

    validation: FullRefitValidation | None = None
    used_full_refit = False
    final_walds = [replicate.wald for replicate in valid]
    if full_refit is not None:
        if validation_rng is None:
            raise MalformedInputError("Full-refit validation requires a dedicated RNG")
        validation = _validate_full_refit(
            valid, full_refit, validation_rng, problem.observed_wald
        )
        if not validation.passed:
            schedules_for_full = (
                list(schedules) if regime == "enumerated" else list(attempted_schedules)
            )
            full_refit_valid, singular, non_finite, full_attempts = (
                _rerun_with_full_refit(
                    schedules_for_full,
                    full_refit,
                    target_valid=bootstrap_replicates if regime == "sampled" else None,
                )
            )
            while (
                regime == "sampled"
                and len(full_refit_valid) < bootstrap_replicates
                and full_attempts < MAX_REPLENISHMENT_MULTIPLIER * bootstrap_replicates
            ):
                weights = _sampled_rademacher_weights(arms, rng)
                additional, failed_singular, failed_non_finite, attempted = (
                    _rerun_with_full_refit(
                        (weights,),
                        full_refit,
                        target_valid=1,
                    )
                )
                full_attempts += attempted
                full_refit_valid.extend(additional)
                singular += failed_singular
                non_finite += failed_non_finite
            attempts = full_attempts
            if regime == "sampled" and len(full_refit_valid) != bootstrap_replicates:
                raise NumericalError(
                    "Full-refit bootstrap could not replenish the requested valid replicates"
                )
            final_walds = [replicate.wald for replicate in full_refit_valid]
            used_full_refit = True

    exceedances = sum(wald >= problem.observed_wald for wald in final_walds)
    if regime == "enumerated":
        discarded = support - len(final_walds)
        lower = exceedances / support
        upper = (exceedances + discarded) / support
        grid = tuple(index / support for index in range(1, support + 1))
        failure_denominator = support
        if discarded == support:
            raise NumericalError("Every enumerated bootstrap vector was non-computable")
    else:
        lower = upper = (1 + exceedances) / (bootstrap_replicates + 1)
        grid = tuple(
            index / (bootstrap_replicates + 1)
            for index in range(1, bootstrap_replicates + 2)
        )
        failure_denominator = bootstrap_replicates

    failure_disclosure = bootstrap_failure_disclosure(
        singular_count=singular,
        non_finite_count=non_finite,
        denominator=failure_denominator,
    )
    return WildBootstrapResult(
        regime=regime,
        p_value=PValueBracket(lower, upper),
        valid_replicates=len(final_walds),
        attempted_replicates=attempts,
        singular_replicates=singular,
        non_finite_replicates=non_finite,
        arm_cluster_count=len(arms),
        attainable_support=support,
        attainable_grid=grid,
        failure_disclosure=failure_disclosure,
        one_step_invariant_blocks_passed=one_step_invariant,
        full_refit_validation=validation,
        used_full_refit=used_full_refit,
    )
