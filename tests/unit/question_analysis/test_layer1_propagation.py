"""Tests for Layer 1 parametric propagation and adaptive Monte Carlo."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from es_index_explorer.question_analysis import (
    layer1_propagation as propagation_module,
)
from es_index_explorer.question_analysis.errors import (
    Layer1ModeError,
    Layer1ReplenishmentExhaustedError,
    NumericalError,
)
from es_index_explorer.question_analysis.layer1_laplace import RubricDrawBlock
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1MmlFit,
    Layer1MomentStart,
    Layer1RubricData,
    Layer1StartFit,
    Layer1VariantData,
    inverse_alr,
    method_of_moments_start,
)
from es_index_explorer.question_analysis.layer1_propagation import (
    OuterHyperparameterDraw,
    RubricPropagation,
    event_monte_carlo_summary,
    generate_outer_sequence,
    next_refinement_depth,
    propagate_rubric,
    shared_draw_keys,
    simulate_layer1_dataset,
)
from es_index_explorer.question_analysis.seeds import derive_child_seed

pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_CONTRACT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "question_analysis"
    / "layer1-synthetic-contract.json"
)


def _hyperparameters() -> Layer1Hyperparameters:
    return Layer1Hyperparameters(
        phi=12.0,
        mu0=np.array([0.3, -0.4]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.array([[0.20, 0.03], [0.03, 0.18]]),
        sigma_between=np.array([[0.25, -0.02], [-0.02, 0.22]]),
    )


def _rubric() -> Layer1RubricData:
    counts = np.tile(np.array([[5, 2, 1]]), (28, 1))
    return Layer1RubricData(
        rubric_id="r1",
        rubric_order=1,
        dataset="emc2_set1",
        expectation_count=8,
        variants=(
            Layer1VariantData(
                variant_id="v1",
                counts=counts,
                arm_ids=tuple(f"a{index:02d}" for index in range(28)),
            ),
            Layer1VariantData(
                variant_id="v2",
                counts=counts.copy(),
                arm_ids=tuple(f"a{index:02d}" for index in range(28)),
            ),
        ),
    )


def _dataset() -> Layer1Dataset:
    return Layer1Dataset(rubrics=(_rubric(),), dataset_levels=("emc2_set1",))


def _mml_fit(hyperparameters: Layer1Hyperparameters) -> Layer1MmlFit:
    vector = np.zeros(9)
    start = Layer1StartFit(
        phi_multiplier=1.0,
        vector=vector,
        hyperparameters=hyperparameters,
        objective=1.0,
        gradient_maximum=1e-6,
        iterations=2,
        converged=True,
        message="synthetic",
    )
    return Layer1MmlFit(
        hyperparameters=hyperparameters,
        vector=vector,
        objective=1.0,
        starts=(
            start,
            replace(start, phi_multiplier=0.5),
            replace(start, phi_multiplier=2.0),
        ),
    )


def _moments() -> Layer1MomentStart:
    eta = np.array([0.5, -0.4])
    return Layer1MomentStart(
        hyperparameters=_hyperparameters(),
        rubric_latent_starts={"r1": np.concatenate((eta, eta, eta))},
        variant_alr={"v1": eta, "v2": eta},
        smoothed_variant_count=0,
        retained_phi_components=1,
    )


def _block(attempt: int, retained: int, value: float = 0.7) -> RubricDrawBlock:
    return RubricDrawBlock(
        rubric_id="r1",
        variant_ids=("v1", "v2"),
        global_outer_attempt_id=attempt,
        rubric_retained_index=retained,
        rubric_v2_draws=np.full((4, 2), value),
    )


def test_parametric_dgp_preserves_design_and_is_deterministic() -> None:
    first = simulate_layer1_dataset(
        _dataset(), _hyperparameters(), rng=np.random.default_rng(15)
    )
    second = simulate_layer1_dataset(
        _dataset(), _hyperparameters(), rng=np.random.default_rng(15)
    )

    assert first.dataset_levels == second.dataset_levels == ("emc2_set1",)
    for first_variant, second_variant in zip(
        first.rubrics[0].variants, second.rubrics[0].variants, strict=True
    ):
        assert first_variant.arm_ids == second_variant.arm_ids
        np.testing.assert_array_equal(first_variant.counts, second_variant.counts)
        np.testing.assert_array_equal(first_variant.counts.sum(axis=1), 8)


def test_outer_sequence_replenishes_global_fit_failure_deterministically() -> None:
    call_count = 0

    def refitter(_data: Layer1Dataset) -> Layer1MmlFit:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise NumericalError("synthetic global fit failure")
        return _mml_fit(_hyperparameters())

    result = generate_outer_sequence(
        _dataset(),
        _hyperparameters(),
        target_depth=2,
        refitter=refitter,
        maximum_attempts=4,
    )

    assert [draw.global_outer_attempt_id for draw in result.draws] == [2, 3]
    assert result.attempted == 3
    assert result.failed == 1


def test_outer_sequence_exhaustion_is_typed_with_counts() -> None:
    def fail(_data: Layer1Dataset) -> Layer1MmlFit:
        raise NumericalError("always fails")

    with pytest.raises(
        Layer1ReplenishmentExhaustedError,
        match=r"scope=global target=2 attempted=3 valid=0",
    ):
        generate_outer_sequence(
            _dataset(),
            _hyperparameters(),
            target_depth=2,
            refitter=fail,
            maximum_attempts=3,
        )


def test_rubric_local_failure_preserves_global_attempt_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def builder(
        _rubric: Layer1RubricData,
        _hyperparameters: Layer1Hyperparameters,
        _moments: Layer1MomentStart,
        *,
        rng: np.random.Generator,
        global_outer_attempt_id: int,
        rubric_retained_index: int,
        draw_count: int,
    ) -> RubricDrawBlock:
        del rng, draw_count
        if global_outer_attempt_id == 2:
            raise Layer1ModeError("rubric-local mode failure")
        return _block(global_outer_attempt_id, rubric_retained_index)

    monkeypatch.setattr(propagation_module, "make_rubric_draw_block", builder)
    outer = tuple(
        OuterHyperparameterDraw(index, _hyperparameters()) for index in (1, 2, 3)
    )
    result = propagate_rubric(
        _rubric(),
        _moments(),
        outer,
        target_depth=2,
        inner_draw_count=4,
    )

    assert [block.global_outer_attempt_id for block in result.blocks] == [1, 3]
    assert [block.rubric_retained_index for block in result.blocks] == [1, 2]
    assert result.failed == 1
    assert result.attempted == 3


def test_outer_batch_mcse_and_closed_trigger_are_exact() -> None:
    blocks = (
        np.array([True, True, True, True]),
        np.array([True, True, True, False]),
        np.array([True, True, True, True]),
        np.array([True, True, True, False]),
    )
    summary = event_monte_carlo_summary(blocks, gamma=0.90)
    batch_means = np.array([1.0, 0.75, 1.0, 0.75])

    assert summary.estimate == pytest.approx(batch_means.mean())
    assert summary.mcse == pytest.approx(
        np.std(batch_means, ddof=1) / np.sqrt(len(batch_means))
    )
    assert summary.refinement_triggered
    boundary = event_monte_carlo_summary(
        (np.array([True]), np.array([False])),
        gamma=0.5 + 2 * (np.std([1.0, 0.0], ddof=1) / np.sqrt(2)),
    )
    assert boundary.refinement_triggered


def test_adaptive_depth_grid_stops_at_four_thousand() -> None:
    assert next_refinement_depth(500) == 1_000
    assert next_refinement_depth(1_000) == 2_000
    assert next_refinement_depth(2_000) == 4_000
    assert next_refinement_depth(4_000) is None


def test_shared_keys_use_global_attempt_not_local_retained_index() -> None:
    first = RubricPropagation(
        rubric_id="r1",
        blocks=(_block(1, 1), _block(3, 2)),
        attempted=3,
        failed=1,
        elevated_failure_conditions=True,
    )
    second = RubricPropagation(
        rubric_id="r2",
        blocks=(
            replace(_block(2, 1), rubric_id="r2"),
            replace(_block(3, 2), rubric_id="r2"),
        ),
        attempted=3,
        failed=1,
        elevated_failure_conditions=True,
    )

    assert shared_draw_keys(first, second) == {
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    }


def test_frozen_single_synthetic_fixture_is_a_finite_non_gating_smoke() -> None:
    contract = json.loads(SYNTHETIC_CONTRACT.read_text(encoding="utf-8"))
    datasets = tuple(contract["dataset_offsets"])
    hyperparameters = Layer1Hyperparameters(
        phi=contract["phi"],
        mu0=np.asarray(contract["mu0"]),
        dataset_offsets={
            dataset: np.asarray(offset)
            for dataset, offset in contract["dataset_offsets"].items()
        },
        sigma_within=np.asarray(contract["sigma_within"]),
        sigma_between=np.asarray(contract["sigma_between"]),
    )
    rubrics: list[Layer1RubricData] = []
    for dataset in datasets:
        for rubric_index in range(contract["rubrics_per_dataset"]):
            variants = tuple(
                Layer1VariantData(
                    variant_id=f"{dataset}-r{rubric_index}-v{variant_index}",
                    counts=np.tile(
                        [[4, 4, 4]],
                        (contract["trace_count_per_variant"], 1),
                    ),
                    arm_ids=tuple(
                        f"a{index:02d}"
                        for index in range(contract["trace_count_per_variant"])
                    ),
                )
                for variant_index in range(contract["variant_count_per_rubric"])
            )
            rubrics.append(
                Layer1RubricData(
                    rubric_id=f"{dataset}-r{rubric_index}",
                    rubric_order=len(rubrics),
                    dataset=dataset,
                    expectation_count=contract["expectation_count"],
                    variants=variants,
                )
            )
    design = Layer1Dataset(rubrics=tuple(rubrics), dataset_levels=datasets)
    simulated = simulate_layer1_dataset(
        design,
        hyperparameters,
        rng=np.random.default_rng(
            derive_child_seed("diagnostic_resampling", "layer1-synthetic-recovery")
        ),
    )
    recovered = method_of_moments_start(simulated).hyperparameters

    assert contract["calibration"]["single_realization_is_gating"] is False
    assert contract["calibration"]["replicate_count"] == 12
    assert np.isfinite(recovered.phi)
    assert np.isfinite(recovered.mu0).all()
    assert np.linalg.eigvalsh(recovered.sigma_within).min() > 0
    assert np.linalg.eigvalsh(recovered.sigma_between).min() > 0


def test_frozen_synthetic_decision_references_replay_within_tolerance() -> None:
    contract = json.loads(SYNTHETIC_CONTRACT.read_text(encoding="utf-8"))
    rng = np.random.default_rng(contract["reference_probability_seed"])
    draw_count = 50_000
    rubric_means = rng.multivariate_normal(
        np.asarray(contract["mu0"]),
        np.asarray(contract["sigma_between"]),
        size=draw_count,
    )
    values = np.empty((draw_count, contract["variant_count_per_rubric"]))
    for variant_index in range(contract["variant_count_per_rubric"]):
        eta = rubric_means + rng.multivariate_normal(
            np.zeros(2),
            np.asarray(contract["sigma_within"]),
            size=draw_count,
        )
        values[:, variant_index] = np.asarray(
            [
                (probability := inverse_alr(current))[0] + 0.5 * probability[2]
                for current in eta
            ]
        )
    references = contract["reference_dataset_event_probabilities"]
    tolerance = contract["acceptance"]["decision_probability_max_abs_error"]

    for floor, suffix in ((0.75, "0_75"), (0.60, "0_60"), (0.50, "0_50")):
        assert (
            abs(
                np.mean(np.min(values, axis=1) >= floor)
                - references[f"minimum_ge_{suffix}"]
            )
            <= tolerance
        )
        assert (
            abs(
                np.mean(np.count_nonzero(values >= floor, axis=1) >= 3)
                - references[f"three_of_four_ge_{suffix}"]
            )
            <= tolerance
        )
