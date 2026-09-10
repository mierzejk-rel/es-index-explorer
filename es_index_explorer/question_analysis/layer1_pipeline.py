"""Gated Layer 1 fitting, artifact construction, and persistence."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from es_index_explorer.question_analysis.contracts import (
    RUBRIC_RECOMMENDATION_COLUMNS,
    VARIANT_RECOMMENDATION_COLUMNS,
    Layer1EventKind,
    SuitabilityTier,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    Layer1ModeError,
    Layer1ReplenishmentExhaustedError,
    MalformedInputError,
    NumericalError,
)
from es_index_explorer.question_analysis.layer1_decisions import (
    TIER_FLOORS,
    PrimaryEvent,
    build_primary_events,
    leave_out_tier_changes,
    pooled_mean_tier,
    proportion_diagnostic,
    run_full_leave_out_refits,
    score_band_probabilities,
    shrunken_variant_means,
    tier_for_events,
    tier_from_probabilities,
    uncertain_flag,
)
from es_index_explorer.question_analysis.layer1_laplace import (
    CONDITIONAL_DRAW_COUNT,
    IMPORTANCE_PARTICLES,
    INNER_DRAWS_PER_OUTER,
    ImportanceDiagnostic,
    RubricDrawBlock,
    conditional_mode,
    draw_rubric_v2,
    importance_resampling_diagnostic,
    make_rubric_draw_block,
    select_importance_rubrics,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MmlFit,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1VariantData,
    fit_layer1_mml,
    method_of_moments_start,
)
from es_index_explorer.question_analysis.layer1_propagation import (
    ADAPTIVE_DEPTHS,
    ELEVATED_FAILURE_RATE,
    MAXIMUM_REPLENISHMENT_MULTIPLIER,
    PREFIX_DIAGNOSTIC_DEPTHS,
    OuterHyperparameterDraw,
    RubricPropagation,
    simulate_layer1_dataset,
)
from es_index_explorer.question_analysis.seeds import derive_child_seed
from es_index_explorer.question_analysis.storage import versioned_frame
from es_index_explorer.question_analysis.workspace import (
    AnalysisWorkspace,
    StepExecutionResult,
)

LAYER1_FIT_ARTIFACT = "statistics/layer1_fit.json"
LAYER1_HYPERPARAMETER_DIAGNOSTICS = "statistics/layer1_hyperparameter_diagnostics.json"
LAYER1_FAILURE_ARTIFACT = "statistics/layer1_numerical_failures.json"
LAYER1_IMPORTANCE_ARTIFACT = "statistics/layer1_importance_diagnostics.parquet"
LAYER1_EVENT_ARTIFACT = "statistics/layer1_primary_events.parquet"
LAYER1_PREFIX_ARTIFACT = "statistics/layer1_prefix_diagnostics.parquet"
LAYER1_STABILITY_ARTIFACT = "statistics/layer1_stability.parquet"
RUBRIC_RECOMMENDATION_ARTIFACT = "tables/recommendation_table_rubric.parquet"
VARIANT_RECOMMENDATION_ARTIFACT = "tables/recommendation_table_variant.parquet"
LAYER1_REPORT = "partial_reports/05-suitability-estimates.md"
FLOOR_SUFFIX = {0.75: "0_75", 0.60: "0_60", 0.50: "0_50"}

Layer1Refitter = Callable[[Layer1Dataset], Layer1MmlFit]


@dataclass(frozen=True, slots=True)
class Layer1RunConfig:
    """Configure frozen production counts with smaller explicit test fixtures."""

    mandatory_prefix_depth: int = 1_000
    prefix_diagnostic_depths: tuple[int, ...] = PREFIX_DIAGNOSTIC_DEPTHS
    adaptive_depths: tuple[int, ...] = ADAPTIVE_DEPTHS
    inner_draw_count: int = INNER_DRAWS_PER_OUTER
    conditional_draw_count: int = CONDITIONAL_DRAW_COUNT
    importance_particles: int = IMPORTANCE_PARTICLES
    maximum_depth: int = 4_000
    run_leave_out_stability: bool = True


@dataclass(frozen=True, slots=True)
class Layer1Execution:
    """Store all in-memory artifacts produced by a complete Layer 1 execution."""

    fit: Mapping[str, object]
    hyperparameter_diagnostics: Mapping[str, object]
    numerical_failures: Mapping[str, object]
    importance: pd.DataFrame
    events: pd.DataFrame
    prefixes: pd.DataFrame
    stability: pd.DataFrame
    draw_frames: Mapping[str, pd.DataFrame]
    rubric_recommendations: pd.DataFrame
    variant_recommendations: pd.DataFrame
    report: str


def load_layer1_dataset(workspace: AnalysisWorkspace) -> Layer1Dataset:
    """Load the eligible PFU population into deterministic hierarchy objects."""
    path = workspace.store.path_for("tables/trace_pfu_table.parquet")
    if not path.is_file():
        raise MalformedInputError("Layer 1 requires trace_pfu_table.parquet")
    frame = pd.read_parquet(path)
    required = {
        "rubric_id",
        "rubric_order",
        "variant_id",
        "variant_index",
        "eval_dataset",
        "arm_id",
        "stage",
        "P",
        "F",
        "U",
        "N_r",
        "eligible",
    }
    if not required <= set(frame.columns):
        raise MalformedInputError("Layer 1 PFU table is missing required columns")
    eligible = frame.loc[frame["eligible"].astype(bool)].copy()
    rubrics: list[Layer1RubricData] = []
    grouped = eligible.sort_values(
        ["rubric_order", "rubric_id", "variant_index", "arm_id"]
    ).groupby(
        ["rubric_id", "rubric_order", "eval_dataset", "N_r"],
        sort=False,
        dropna=False,
    )
    for (rubric_id, rubric_order, dataset, expectation_count), rubric_frame in grouped:
        variants: list[Layer1VariantData] = []
        for variant_id, variant_frame in rubric_frame.groupby(
            "variant_id", sort=False, dropna=False
        ):
            variants.append(
                Layer1VariantData(
                    variant_id=str(variant_id),
                    counts=variant_frame[["P", "F", "U"]].to_numpy(dtype=int),
                    arm_ids=tuple(variant_frame["arm_id"].astype(str)),
                    stages=tuple(variant_frame["stage"].astype(str)),
                )
            )
        rubrics.append(
            Layer1RubricData(
                rubric_id=str(rubric_id),
                rubric_order=int(rubric_order),
                dataset=str(dataset),
                expectation_count=int(expectation_count),
                variants=tuple(variants),
            )
        )
    return Layer1Dataset(
        rubrics=tuple(rubrics),
        dataset_levels=tuple(sorted(eligible["eval_dataset"].astype(str).unique())),
    )


def _hyperparameters_payload(
    hyperparameters: Layer1Hyperparameters,
) -> Mapping[str, object]:
    return {
        "phi": hyperparameters.phi,
        "mu0": hyperparameters.mu0.tolist(),
        "dataset_offsets": {
            dataset: value.tolist()
            for dataset, value in sorted(hyperparameters.dataset_offsets.items())
        },
        "sigma_within": hyperparameters.sigma_within.tolist(),
        "sigma_between": hyperparameters.sigma_between.tolist(),
        "trace_sigma_within": float(np.trace(hyperparameters.sigma_within)),
        "trace_sigma_between": float(np.trace(hyperparameters.sigma_between)),
        "eigenvalues_sigma_within": np.linalg.eigvalsh(
            hyperparameters.sigma_within
        ).tolist(),
        "eigenvalues_sigma_between": np.linalg.eigvalsh(
            hyperparameters.sigma_between
        ).tolist(),
    }


def _fit_payload(fit: Layer1MmlFit) -> Mapping[str, object]:
    return {
        "schema_version": 1,
        "objective": fit.objective,
        "hyperparameters": _hyperparameters_payload(fit.hyperparameters),
        "starts": [
            {
                "phi_multiplier": start.phi_multiplier,
                "objective": start.objective,
                "gradient_maximum": start.gradient_maximum,
                "iterations": start.iterations,
                "converged": start.converged,
                "message": start.message,
                "vector": start.vector.tolist(),
            }
            for start in fit.starts
        ],
    }


def _conditional_draws(
    rubric: Layer1RubricData,
    fit: Layer1MmlFit,
    moments: Layer1MomentStart,
    draw_count: int,
) -> np.ndarray:
    mode = conditional_mode(rubric, fit.hyperparameters, moments)
    rng = np.random.default_rng(
        derive_child_seed("laplace_draws", f"{rubric.rubric_id}:conditional")
    )
    return draw_rubric_v2(rubric, mode, rng=rng, draw_count=draw_count)


def _conditional_probability_fields(
    values: np.ndarray,
    propagated_probabilities: Mapping[float, float],
    *,
    minimum: bool,
) -> Mapping[str, float]:
    fields: dict[str, float] = {}
    infix = "_min" if minimum else ""
    for floor in TIER_FLOORS:
        suffix = FLOOR_SUFFIX[floor]
        probability = float(np.mean(values >= floor))
        mcse = float(np.sqrt(probability * (1.0 - probability) / len(values)))
        fields[f"pi_cond{infix}_ge_{suffix}"] = probability
        fields[f"mcse_cond{infix}_ge_{suffix}"] = mcse
        fields[f"pi_prop_cond_gap_{suffix}"] = (
            propagated_probabilities[floor] - probability
        )
    return fields


def _execute_layer1(
    data: Layer1Dataset,
    *,
    config: Layer1RunConfig | None = None,
    refitter: Layer1Refitter = fit_layer1_mml,
) -> Layer1Execution:
    selected_config = config or Layer1RunConfig()
    if (
        not selected_config.prefix_diagnostic_depths
        or not selected_config.adaptive_depths
        or tuple(sorted(selected_config.prefix_diagnostic_depths))
        != selected_config.prefix_diagnostic_depths
        or tuple(sorted(selected_config.adaptive_depths))
        != selected_config.adaptive_depths
        or selected_config.mandatory_prefix_depth
        < max(selected_config.prefix_diagnostic_depths)
        or selected_config.mandatory_prefix_depth < selected_config.adaptive_depths[0]
        or selected_config.maximum_depth != selected_config.adaptive_depths[-1]
    ):
        raise MalformedInputError("Layer 1 run depths are inconsistent")
    moments = method_of_moments_start(data)
    fit = refitter(data)
    rubric_by_id = {rubric.rubric_id: rubric for rubric in data.rubrics}
    blocks: dict[str, list[RubricDrawBlock]] = {
        rubric.rubric_id: [] for rubric in data.rubrics
    }
    local_failures = {rubric.rubric_id: 0 for rubric in data.rubrics}
    processed_outer_draws = {rubric.rubric_id: 0 for rubric in data.rubrics}
    outer_draws: list[OuterHyperparameterDraw] = []
    global_failures = 0
    target_depths = {
        rubric.rubric_id: selected_config.mandatory_prefix_depth
        for rubric in data.rubrics
    }
    cap = MAXIMUM_REPLENISHMENT_MULTIPLIER * selected_config.maximum_depth
    attempt_id = 0

    def process_available_outer_draws(rubric_id: str) -> None:
        rubric = rubric_by_id[rubric_id]
        target = target_depths[rubric_id]
        while len(blocks[rubric_id]) < target and processed_outer_draws[
            rubric_id
        ] < len(outer_draws):
            outer = outer_draws[processed_outer_draws[rubric_id]]
            processed_outer_draws[rubric_id] += 1
            inner_rng = np.random.default_rng(
                derive_child_seed(
                    "laplace_draws",
                    f"{rubric_id}:outer:{outer.global_outer_attempt_id}",
                )
            )
            try:
                block = make_rubric_draw_block(
                    rubric,
                    outer.hyperparameters,
                    moments,
                    rng=inner_rng,
                    global_outer_attempt_id=outer.global_outer_attempt_id,
                    rubric_retained_index=len(blocks[rubric_id]) + 1,
                    draw_count=selected_config.inner_draw_count,
                )
            except Layer1ModeError:
                local_failures[rubric_id] += 1
                continue
            blocks[rubric_id].append(block)

    def extend_targets() -> None:
        nonlocal attempt_id, global_failures
        while any(
            len(blocks[rubric_id]) < target
            for rubric_id, target in target_depths.items()
        ):
            for rubric_id in target_depths:
                process_available_outer_draws(rubric_id)
            if all(
                len(blocks[rubric_id]) >= target
                for rubric_id, target in target_depths.items()
            ):
                return
            attempt_id += 1
            if attempt_id > cap:
                unresolved = {
                    rubric_id: {
                        "target": target_depths[rubric_id],
                        "valid": len(blocks[rubric_id]),
                    }
                    for rubric_id in target_depths
                    if len(blocks[rubric_id]) < target_depths[rubric_id]
                }
                raise Layer1ReplenishmentExhaustedError(
                    "LAYER1_REPLENISHMENT_EXHAUSTED "
                    f"scope={json.dumps(unresolved, sort_keys=True)} "
                    f"attempted={cap}"
                )
            outer_rng = np.random.default_rng(
                derive_child_seed("layer1_bootstrap", f"outer:{attempt_id}")
            )
            simulated = simulate_layer1_dataset(
                data, fit.hyperparameters, rng=outer_rng
            )
            try:
                outer_fit = refitter(simulated)
            except NumericalError:
                global_failures += 1
                continue
            outer = OuterHyperparameterDraw(
                global_outer_attempt_id=attempt_id,
                hyperparameters=outer_fit.hyperparameters,
            )
            outer_draws.append(outer)

    extend_targets()
    decision_depths: dict[str, int] = {}
    prefix_rows: list[dict[str, object]] = []
    final_events: dict[str, tuple[PrimaryEvent, ...]] = {}
    for rubric in data.rubrics:
        propagation = RubricPropagation(
            rubric_id=rubric.rubric_id,
            blocks=tuple(blocks[rubric.rubric_id]),
            attempted=processed_outer_draws[rubric.rubric_id],
            failed=local_failures[rubric.rubric_id],
            elevated_failure_conditions=(
                local_failures[rubric.rubric_id]
                / max(1, processed_outer_draws[rubric.rubric_id])
                > ELEVATED_FAILURE_RATE
            ),
        )
        for depth in selected_config.prefix_diagnostic_depths:
            events = build_primary_events(propagation, depth=depth)
            decision = tier_for_events(
                events,
                kind=Layer1EventKind.WORST_VARIANT,
                unit_id=rubric.rubric_id,
            )
            prefix_rows.append(
                {
                    "rubric_id": rubric.rubric_id,
                    "outer_depth": depth,
                    "tier": decision.tier.value,
                }
            )
        depth = selected_config.adaptive_depths[0]
        while True:
            events = build_primary_events(propagation, depth=depth)
            if not any(event.summary.refinement_triggered for event in events):
                break
            next_depth = (
                selected_config.adaptive_depths[
                    selected_config.adaptive_depths.index(depth) + 1
                ]
                if depth != selected_config.adaptive_depths[-1]
                else None
            )
            if next_depth is None:
                break
            target_depths[rubric.rubric_id] = next_depth
            extend_targets()
            propagation = RubricPropagation(
                rubric_id=rubric.rubric_id,
                blocks=tuple(blocks[rubric.rubric_id]),
                attempted=processed_outer_draws[rubric.rubric_id],
                failed=local_failures[rubric.rubric_id],
                elevated_failure_conditions=(
                    local_failures[rubric.rubric_id]
                    / max(1, processed_outer_draws[rubric.rubric_id])
                    > ELEVATED_FAILURE_RATE
                ),
            )
            depth = next_depth
        decision_depths[rubric.rubric_id] = depth
        final_events[rubric.rubric_id] = events

    importance_ids = set(select_importance_rubrics(data))
    importance_diagnostics: dict[str, ImportanceDiagnostic] = {}
    conditional_draws: dict[str, np.ndarray] = {}
    for rubric in data.rubrics:
        conditional_draws[rubric.rubric_id] = _conditional_draws(
            rubric, fit, moments, selected_config.conditional_draw_count
        )
        if rubric.rubric_id in importance_ids:
            importance_diagnostics[rubric.rubric_id] = importance_resampling_diagnostic(
                rubric,
                fit.hyperparameters,
                moments,
                rng=np.random.default_rng(
                    derive_child_seed(
                        "diagnostic_resampling",
                        f"importance:{rubric.rubric_id}",
                    )
                ),
                particle_count=selected_config.importance_particles,
            )

    event_rows: list[dict[str, object]] = []
    rubric_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    draw_frames: dict[str, pd.DataFrame] = {}
    for rubric in data.rubrics:
        depth = decision_depths[rubric.rubric_id]
        rubric_blocks = blocks[rubric.rubric_id]
        propagation = RubricPropagation(
            rubric_id=rubric.rubric_id,
            blocks=tuple(rubric_blocks),
            attempted=processed_outer_draws[rubric.rubric_id],
            failed=local_failures[rubric.rubric_id],
            elevated_failure_conditions=(
                local_failures[rubric.rubric_id]
                / max(1, processed_outer_draws[rubric.rubric_id])
                > ELEVATED_FAILURE_RATE
            ),
        )
        events = final_events[rubric.rubric_id]
        at_cap = depth == selected_config.maximum_depth
        rubric_decision = tier_for_events(
            events,
            kind=Layer1EventKind.WORST_VARIANT,
            unit_id=rubric.rubric_id,
        )
        if at_cap and any(event.summary.refinement_triggered for event in events):
            rubric_decision = tier_from_probabilities(
                rubric_decision.probabilities,
                monte_carlo_indeterminate=True,
            )
        pooled_decision = pooled_mean_tier(propagation, depth=depth)
        means = shrunken_variant_means(propagation, depth=depth)
        conditional = conditional_draws[rubric.rubric_id]
        propagated_values = np.concatenate(
            [block.rubric_v2_draws for block in rubric_blocks[:depth]], axis=0
        )
        minimum_values = np.min(propagated_values, axis=1)
        conditional_minimum = np.min(conditional, axis=1)
        bands = score_band_probabilities(minimum_values)
        proportion = proportion_diagnostic(len(rubric.variants))
        importance = importance_diagnostics.get(rubric.rubric_id)
        event_lookup = {
            (event.kind, event.unit_id, event.floor): event for event in events
        }
        rubric_propagated_probabilities = {
            floor: event_lookup[
                (Layer1EventKind.WORST_VARIANT, rubric.rubric_id, floor)
            ].summary.estimate
            for floor in TIER_FLOORS
        }
        for event in events:
            event_rows.append(
                {
                    "rubric_id": event.rubric_id,
                    "unit_id": event.unit_id,
                    "event_kind": event.kind.value,
                    "floor": event.floor,
                    "outer_depth": depth,
                    "estimate": event.summary.estimate,
                    "mcse": event.summary.mcse,
                    "interval_lower": event.summary.interval_lower,
                    "interval_upper": event.summary.interval_upper,
                    "refinement_triggered": event.summary.refinement_triggered,
                }
            )
        rubric_rows.append(
            {
                "rubric_id": rubric.rubric_id,
                "rubric_order": rubric.rubric_order,
                "eval_dataset": rubric.dataset,
                "tier": rubric_decision.tier.value,
                "uncertain": uncertain_flag(rubric_decision, pooled_decision),
                "v_r": len(rubric.variants),
                "monte_carlo_indeterminate": rubric_decision.monte_carlo_indeterminate,
                "rubric_decision_depth": depth,
                **{
                    f"pi_min_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                        (Layer1EventKind.WORST_VARIANT, rubric.rubric_id, floor)
                    ].summary.estimate
                    for floor in TIER_FLOORS
                },
                **{
                    f"mcse_min_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                        (Layer1EventKind.WORST_VARIANT, rubric.rubric_id, floor)
                    ].summary.mcse
                    for floor in TIER_FLOORS
                },
                **{
                    f"pi_proportion_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                        (
                            Layer1EventKind.PROPORTION_KAPPA_0_75,
                            rubric.rubric_id,
                            floor,
                        )
                    ].summary.estimate
                    for floor in TIER_FLOORS
                },
                **_conditional_probability_fields(
                    conditional_minimum,
                    rubric_propagated_probabilities,
                    minimum=True,
                ),
                "pooled_mean_tier": pooled_decision.tier.value,
                "shrunken_variant_minimum": min(means.values()),
                "proportion_binding_count": proportion.binding_count,
                "proportion_grid_json": json.dumps(proportion.attainable_grid),
                "proportion_uninformative": proportion.uninformative_by_construction,
                **{f"score_band_{name}": value for name, value in bands.items()},
                "elevated_numerical_failure_conditions": (
                    propagation.elevated_failure_conditions
                ),
                "laplace_adequate": importance.adequate if importance else None,
            }
        )
        for variant_index, variant in enumerate(rubric.variants):
            decision = tier_for_events(
                events,
                kind=Layer1EventKind.SINGLE_VARIANT,
                unit_id=variant.variant_id,
                at_adaptive_cap=at_cap,
            )
            values = propagated_values[:, variant_index]
            conditional_values = conditional[:, variant_index]
            variant_bands = score_band_probabilities(values)
            variant_propagated_probabilities = {
                floor: event_lookup[
                    (
                        Layer1EventKind.SINGLE_VARIANT,
                        variant.variant_id,
                        floor,
                    )
                ].summary.estimate
                for floor in TIER_FLOORS
            }
            variant_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "rubric_id": rubric.rubric_id,
                    "variant_index": variant_index,
                    "tier": decision.tier.value,
                    "uncertain": False,
                    "monte_carlo_indeterminate": (decision.monte_carlo_indeterminate),
                    "rubric_decision_depth": depth,
                    **{
                        f"pi_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                            (
                                Layer1EventKind.SINGLE_VARIANT,
                                variant.variant_id,
                                floor,
                            )
                        ].summary.estimate
                        for floor in TIER_FLOORS
                    },
                    **{
                        f"mcse_ge_{FLOOR_SUFFIX[floor]}": event_lookup[
                            (
                                Layer1EventKind.SINGLE_VARIANT,
                                variant.variant_id,
                                floor,
                            )
                        ].summary.mcse
                        for floor in TIER_FLOORS
                    },
                    **_conditional_probability_fields(
                        conditional_values,
                        variant_propagated_probabilities,
                        minimum=False,
                    ),
                    "shrunken_mean": means[variant.variant_id],
                    **{
                        f"score_band_{name}": value
                        for name, value in variant_bands.items()
                    },
                    "elevated_numerical_failure_conditions": (
                        propagation.elevated_failure_conditions
                    ),
                    "laplace_adequate": importance.adequate if importance else None,
                }
            )
        draw_rows = [
            {
                "rubric_id": rubric.rubric_id,
                "variant_id": variant.variant_id,
                "global_outer_attempt_id": block.global_outer_attempt_id,
                "rubric_retained_index": block.rubric_retained_index,
                "inner_index": inner_index,
                "rubric_decision_depth": depth,
                "rubric_v2": block.rubric_v2_draws[inner_index, variant_index],
            }
            for block in rubric_blocks
            for inner_index in range(len(block.rubric_v2_draws))
            for variant_index, variant in enumerate(rubric.variants)
        ]
        draw_frames[rubric.rubric_id] = pd.DataFrame(draw_rows)

    importance_rows = [
        {
            "rubric_id": diagnostic.rubric_id,
            "particle_count": diagnostic.particle_count,
            "effective_sample_size": diagnostic.effective_sample_size,
            "effective_sample_size_ratio": diagnostic.effective_sample_size_ratio,
            "adequate": diagnostic.adequate,
            "laplace_mean_json": json.dumps(diagnostic.laplace_mean.tolist()),
            "importance_mean_json": json.dumps(diagnostic.importance_mean.tolist()),
            "laplace_covariance_json": json.dumps(
                diagnostic.laplace_covariance.tolist()
            ),
            "importance_covariance_json": json.dumps(
                diagnostic.importance_covariance.tolist()
            ),
        }
        for diagnostic in importance_diagnostics.values()
    ]
    failures = {
        "schema_version": 1,
        "global_attempts": attempt_id,
        "global_failures": global_failures,
        "rubric_failures": local_failures,
    }
    report = _render_layer1_report(
        pd.DataFrame(rubric_rows),
        pd.DataFrame(variant_rows),
        fit.hyperparameters,
        failures,
    )
    rubric_recommendations = pd.DataFrame(
        rubric_rows, columns=RUBRIC_RECOMMENDATION_COLUMNS
    )
    stability_rows: list[dict[str, object]] = []
    if selected_config.run_leave_out_stability:
        primary_tiers = {
            str(row["rubric_id"]): SuitabilityTier(str(row["tier"]))
            for row in rubric_rows
        }

        def refit_tiers(reduced: Layer1Dataset) -> Mapping[str, SuitabilityTier]:
            reduced_execution = _execute_layer1(
                reduced,
                config=replace(
                    selected_config,
                    run_leave_out_stability=False,
                ),
                refitter=refitter,
            )
            return {
                str(row["rubric_id"]): SuitabilityTier(str(row["tier"]))
                for row in reduced_execution.rubric_recommendations.to_dict(
                    orient="records"
                )
            }

        alternatives = run_full_leave_out_refits(data, refit_tiers)
        changed = leave_out_tier_changes(primary_tiers, alternatives)
        stability_rows = [
            {
                "analysis": "leave_out",
                "unit_id": unit_id,
                "comparison_id": comparison_id,
                "tier_changed": unit_id in changed[comparison_id],
                "shared_draw_count": None,
            }
            for comparison_id in sorted(alternatives)
            for unit_id in sorted(primary_tiers)
        ]
    return Layer1Execution(
        fit=_fit_payload(fit),
        hyperparameter_diagnostics={
            "schema_version": 1,
            "moment_smoothed_variants": moments.smoothed_variant_count,
            "retained_phi_components": moments.retained_phi_components,
            "decision_depths": decision_depths,
        },
        numerical_failures=failures,
        importance=pd.DataFrame(importance_rows),
        events=pd.DataFrame(event_rows),
        prefixes=pd.DataFrame(prefix_rows),
        stability=pd.DataFrame(
            stability_rows,
            columns=[
                "analysis",
                "unit_id",
                "comparison_id",
                "tier_changed",
                "shared_draw_count",
            ],
        ),
        draw_frames=draw_frames,
        rubric_recommendations=rubric_recommendations,
        variant_recommendations=pd.DataFrame(
            variant_rows, columns=VARIANT_RECOMMENDATION_COLUMNS
        ),
        report=report,
    )


def _render_layer1_report(
    rubrics: pd.DataFrame,
    variants: pd.DataFrame,
    hyperparameters: Layer1Hyperparameters,
    failures: Mapping[str, object],
) -> str:
    """Render the persisted Layer 1 partial report from result tables."""
    rubric_counts = rubrics["tier"].value_counts().sort_index()
    variant_counts = variants["tier"].value_counts().sort_index()
    lines = [
        "# Layer 1 suitability estimates",
        "",
        "The decision quantity is `Pi_prop`, a propagated empirical-Bayes uncertainty measure,",
        "not a Bayesian posterior probability or a frequentist coverage guarantee.",
        "",
        "## Hyperparameters",
        "",
        f"- `phi`: {hyperparameters.phi:.8g}",
        f"- `trace(Sigma_between)`: {np.trace(hyperparameters.sigma_between):.8g}",
        f"- `trace(Sigma_within)`: {np.trace(hyperparameters.sigma_within):.8g}",
        "",
        "## Rubric tiers",
        "",
        *[f"- {tier}: {count}" for tier, count in rubric_counts.items()],
        "",
        "## Variant tiers",
        "",
        *[f"- {tier}: {count}" for tier, count in variant_counts.items()],
        "",
        "## Numerical failures",
        "",
        f"- Global failures: {failures['global_failures']}",
        f"- Global attempts: {failures['global_attempts']}",
        "",
        "Importance-resampling caveats and Monte Carlo indeterminacy remain attached to",
        "the corresponding recommendation rows.",
        "",
    ]
    return "\n".join(lines)


def _verify_layer1_gate_artifacts(workspace: AnalysisWorkspace) -> None:
    oracle_path = workspace.store.path_for("statistics/r_oracle_verification.json")
    unlock_path = workspace.store.path_for("validation_unlock.json")
    if not oracle_path.is_file() or not unlock_path.is_file():
        raise GateFailureError(
            "Layer 1 requires persisted oracle and validation unlock"
        )
    try:
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MalformedInputError("Layer 1 gate artifact is invalid") from error
    if oracle.get("passed") is not True or unlock.get("authorized") is not True:
        raise GateFailureError(
            "Layer 1 oracle or human validation gate is not authorized"
        )


def _safe_rubric_filename(rubric_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in rubric_id
    )


def run_fit_layer1(workspace: AnalysisWorkspace) -> StepExecutionResult:
    """Run the gated Layer 1 engine and atomically register complete artifacts."""
    _verify_layer1_gate_artifacts(workspace)
    execution = _execute_layer1(load_layer1_dataset(workspace))
    artifacts = [
        workspace.store.write_json(
            LAYER1_FIT_ARTIFACT,
            execution.fit,
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_json(
            LAYER1_HYPERPARAMETER_DIAGNOSTICS,
            execution.hyperparameter_diagnostics,
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_json(
            LAYER1_FAILURE_ARTIFACT,
            execution.numerical_failures,
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_IMPORTANCE_ARTIFACT,
            versioned_frame(execution.importance),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_EVENT_ARTIFACT,
            versioned_frame(execution.events),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_PREFIX_ARTIFACT,
            versioned_frame(execution.prefixes),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            LAYER1_STABILITY_ARTIFACT,
            versioned_frame(execution.stability),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            RUBRIC_RECOMMENDATION_ARTIFACT,
            versioned_frame(execution.rubric_recommendations),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_parquet(
            VARIANT_RECOMMENDATION_ARTIFACT,
            versioned_frame(execution.variant_recommendations),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
        workspace.store.write_bytes(
            LAYER1_REPORT,
            execution.report.encode("utf-8"),
            created_by=WorkflowCommand.FIT_LAYER1,
        ),
    ]
    for rubric_id, frame in execution.draw_frames.items():
        artifacts.append(
            workspace.store.write_parquet(
                Path("draws") / f"pi_prop_{_safe_rubric_filename(rubric_id)}.parquet",
                versioned_frame(frame),
                created_by=WorkflowCommand.FIT_LAYER1,
            )
        )
    return StepExecutionResult(artifacts=tuple(artifacts))
