"""Continuous equivalence protection for Layer 1 performance refactors."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from es_index_explorer.question_analysis.layer1_execution import (
    Layer1ExecutionConfig,
    TaskReference,
    WorkUnit,
    WorkUnitKind,
    execute_work_units,
)
from es_index_explorer.question_analysis.layer1_model import (
    Layer1Dataset,
    Layer1Hyperparameters,
    Layer1RubricData,
    Layer1VariantData,
    MarginalEvaluationPayload,
    _variant_likelihood_derivatives,
    alr_from_counts,
    build_hyperparameter_context,
    dirichlet_multinomial_logpmf,
    evaluate_marginal_work_unit,
    inverse_alr,
    inverse_alr_batch,
    laplace_log_marginal,
    method_of_moments_start,
    pack_hyperparameters,
    rubric_log_density,
    rubric_log_density_batch,
    solve_arrowhead_curvature,
)
from es_index_explorer.question_analysis.layer1_packed import (
    compress_pfu_counts,
    expand_pfu_counts,
    pack_layer1_dataset,
)
from tests.oracles.layer1.reference import (
    ReferenceRubric,
    reference_alr_from_counts,
    reference_dirichlet_multinomial_logpmf,
    reference_inverse_alr,
    reference_laplace_log_marginal,
    reference_ordered_sum,
    reference_rubric_v2_draws,
    reference_variant_derivatives,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SOURCE = PROJECT_ROOT / "tests" / "oracles" / "layer1" / "reference.py"
REFERENCE_HASH = PROJECT_ROOT / "tests" / "oracles" / "layer1" / "reference.sha256"
KERNEL_RTOL = 1e-10
KERNEL_ATOL = 1e-12


@pytest.mark.parametrize(
    "counts",
    (
        np.array([2, 3, 5]),
        np.array([0, 3, 5]),
        np.array([2, 0, 5]),
        np.array([2, 3, 0]),
    ),
)
def test_alr_and_inverse_match_independent_reference(counts: np.ndarray) -> None:
    actual, _ = alr_from_counts(counts)
    expected = reference_alr_from_counts(counts)

    np.testing.assert_allclose(actual, expected, rtol=KERNEL_RTOL, atol=KERNEL_ATOL)
    np.testing.assert_allclose(
        inverse_alr(actual),
        reference_inverse_alr(expected),
        rtol=KERNEL_RTOL,
        atol=KERNEL_ATOL,
    )


@pytest.mark.parametrize("seed", range(8))
def test_likelihood_and_derivatives_match_serial_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        12,
        rng.dirichlet(np.array([2.0, 3.0, 1.5])),
        size=28,
    )
    eta = rng.normal(size=2)
    phi = float(rng.uniform(0.2, 100.0))

    expected = reference_variant_derivatives(counts, eta, phi)
    actual = _variant_likelihood_derivatives(counts, eta, phi)

    assert dirichlet_multinomial_logpmf(counts, eta, phi) == pytest.approx(
        reference_dirichlet_multinomial_logpmf(counts, eta, phi),
        rel=KERNEL_RTOL,
        abs=KERNEL_ATOL,
    )
    assert actual[0] == pytest.approx(expected[0], rel=KERNEL_RTOL, abs=KERNEL_ATOL)
    np.testing.assert_allclose(
        actual[1], expected[1], rtol=KERNEL_RTOL, atol=KERNEL_ATOL
    )
    np.testing.assert_allclose(
        actual[2], expected[2], rtol=KERNEL_RTOL, atol=KERNEL_ATOL
    )


def test_ordered_sum_reference_is_sensitive_to_order() -> None:
    values = (1e16, 1.0, -1e16)

    assert reference_ordered_sum(values) == 0.0
    assert reference_ordered_sum(tuple(reversed(values))) == 0.0
    assert reference_ordered_sum((1e16, -1e16, 1.0)) == 1.0


def test_reference_source_is_independent_and_matches_checksum() -> None:
    source = REFERENCE_SOURCE.read_bytes()
    expected_hash = REFERENCE_HASH.read_text(encoding="utf-8").strip()

    assert b"es_index_explorer.question_analysis" not in source
    assert hashlib.sha256(source).hexdigest() == expected_hash


def test_compressed_patterns_preserve_multiplicity_constants_and_derivatives() -> None:
    counts = np.array(
        [
            [0, 2, 1],
            [2, 1, 0],
            [0, 2, 1],
            [1, 1, 1],
            [2, 1, 0],
        ]
    )
    compressed = compress_pfu_counts(counts)
    eta = np.array([0.2, -0.3])
    phi = 7.0

    assert compressed.trace_count == len(counts)
    assert len(compressed.patterns) == 3
    np.testing.assert_array_equal(
        expand_pfu_counts(compressed),
        np.repeat(
            compressed.patterns,
            compressed.multiplicities,
            axis=0,
        ),
    )
    expected = reference_variant_derivatives(counts, eta, phi)
    actual = _variant_likelihood_derivatives(compressed, eta, phi)
    assert actual[0] == pytest.approx(expected[0], rel=KERNEL_RTOL, abs=KERNEL_ATOL)
    np.testing.assert_allclose(
        actual[1], expected[1], rtol=KERNEL_RTOL, atol=KERNEL_ATOL
    )
    np.testing.assert_allclose(
        actual[2], expected[2], rtol=KERNEL_RTOL, atol=KERNEL_ATOL
    )


def test_packed_indices_reconstruct_original_hierarchy_rows() -> None:
    variants = (
        Layer1VariantData(
            variant_id="v1",
            counts=np.array([[1, 2, 0], [2, 1, 0]]),
            arm_ids=("a1", "a2"),
        ),
        Layer1VariantData(
            variant_id="v2",
            counts=np.array([[0, 1, 2]]),
            arm_ids=("a1",),
        ),
    )
    data = Layer1Dataset(
        rubrics=(
            Layer1RubricData(
                rubric_id="r1",
                rubric_order=1,
                dataset="emc2_set1",
                expectation_count=3,
                variants=variants,
            ),
        ),
        dataset_levels=("emc2_set1",),
    )

    packed = pack_layer1_dataset(data)

    assert packed.rubric_ids == ("r1",)
    assert packed.variant_ids == ("v1", "v2")
    np.testing.assert_array_equal(packed.variant_offsets, [0, 2, 3])
    np.testing.assert_array_equal(packed.rubric_variant_offsets, [0, 2])
    np.testing.assert_array_equal(
        packed.counts,
        np.vstack([variant.counts for variant in variants]),
    )


def test_batched_rubric_log_density_matches_scalar_path() -> None:
    rng = np.random.default_rng(91)
    rubric = Layer1RubricData(
        rubric_id="r1",
        rubric_order=1,
        dataset="emc2_set1",
        expectation_count=6,
        variants=(
            Layer1VariantData(
                variant_id="v1",
                counts=rng.multinomial(6, [0.5, 0.3, 0.2], size=8),
                arm_ids=tuple(f"a{index}" for index in range(8)),
            ),
            Layer1VariantData(
                variant_id="v2",
                counts=rng.multinomial(6, [0.4, 0.35, 0.25], size=8),
                arm_ids=tuple(f"a{index}" for index in range(8)),
            ),
        ),
    )
    hyperparameters = Layer1Hyperparameters(
        phi=12.0,
        mu0=np.array([0.2, -0.3]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.array([[0.3, 0.04], [0.04, 0.2]]),
        sigma_between=np.array([[0.4, -0.02], [-0.02, 0.25]]),
    )
    context = build_hyperparameter_context(hyperparameters)
    latent = rng.normal(size=(20, 6))
    expected = np.asarray(
        [rubric_log_density(rubric, context, value) for value in latent]
    )

    actual = rubric_log_density_batch(rubric, context, latent)

    np.testing.assert_allclose(actual, expected, rtol=KERNEL_RTOL, atol=KERNEL_ATOL)


@pytest.mark.parametrize("worker_count", (1, 2))
def test_process_marginal_evaluation_matches_direct_serial_result(
    worker_count: int,
) -> None:
    rng = np.random.default_rng(92)
    rubric = Layer1RubricData(
        rubric_id="r1",
        rubric_order=1,
        dataset="emc2_set1",
        expectation_count=6,
        variants=(
            Layer1VariantData(
                variant_id="v1",
                counts=rng.multinomial(6, [0.5, 0.3, 0.2], size=8),
                arm_ids=tuple(f"a{index}" for index in range(8)),
            ),
            Layer1VariantData(
                variant_id="v2",
                counts=rng.multinomial(6, [0.4, 0.35, 0.25], size=8),
                arm_ids=tuple(f"a{index}" for index in range(8)),
            ),
        ),
    )
    data = Layer1Dataset(rubrics=(rubric,), dataset_levels=("emc2_set1",))
    hyperparameters = Layer1Hyperparameters(
        phi=12.0,
        mu0=np.array([0.2, -0.3]),
        dataset_offsets={"emc2_set1": np.zeros(2)},
        sigma_within=np.array([[0.3, 0.04], [0.04, 0.2]]),
        sigma_between=np.array([[0.4, -0.02], [-0.02, 0.25]]),
    )
    vector = pack_hyperparameters(hyperparameters, data.dataset_levels)
    payload = MarginalEvaluationPayload(
        data=data,
        latent_starts={"r1": np.array([0.2, -0.3, 0.1, -0.2, 0.3, -0.4])},
        vector=vector,
    )
    unit = WorkUnit(
        task_id="marginal",
        kind=WorkUnitKind.OBJECTIVE_GRADIENT_PERTURBATION,
        payload=payload,
    )
    expected = evaluate_marginal_work_unit(unit)

    actual = execute_work_units(
        TaskReference(
            module=evaluate_marginal_work_unit.__module__,
            function=evaluate_marginal_work_unit.__name__,
        ),
        (unit,),
        config=Layer1ExecutionConfig(worker_mode="process", worker_count=worker_count),
    )[0]

    assert actual.succeeded
    assert isinstance(actual.value, (int, float, np.floating))
    assert isinstance(expected, (int, float, np.floating))
    assert float(actual.value) == pytest.approx(
        float(expected), rel=KERNEL_RTOL, abs=KERNEL_ATOL
    )


def test_batched_inverse_alr_and_rubric_v2_match_serial_reference() -> None:
    eta = np.random.default_rng(111).normal(size=(40, 8, 2))
    probabilities = inverse_alr_batch(eta)
    actual = probabilities[..., 0] + 0.5 * probabilities[..., 2]

    expected = reference_rubric_v2_draws(eta)

    np.testing.assert_allclose(actual, expected, rtol=KERNEL_RTOL, atol=KERNEL_ATOL)


@pytest.mark.parametrize("variant_count", range(1, 9))
def test_arrowhead_solve_matches_dense_reference(variant_count: int) -> None:
    rng = np.random.default_rng(1_000 + variant_count)
    dimension = 2 * (variant_count + 1)
    curvature = np.zeros((dimension, dimension))
    curvature[:2, :2] = np.eye(2) * (4.0 + variant_count)
    for variant_index in range(variant_count):
        variant_slice = slice(2 + 2 * variant_index, 4 + 2 * variant_index)
        block = np.array([[2.0, 0.1], [0.1, 1.5]]) + np.eye(2) * (variant_index / 10)
        cross = np.array([[-0.2, 0.05], [0.03, -0.15]])
        curvature[variant_slice, variant_slice] = block
        curvature[:2, variant_slice] = cross
        curvature[variant_slice, :2] = cross.T
    right_hand_side = rng.normal(size=dimension)

    expected = np.linalg.solve(curvature, right_hand_side)
    actual = solve_arrowhead_curvature(curvature, right_hand_side, variant_count)

    np.testing.assert_allclose(actual, expected, rtol=KERNEL_RTOL, atol=KERNEL_ATOL)


def test_full_laplace_objective_matches_independent_dense_reference() -> None:
    rng = np.random.default_rng(155)
    rubrics = tuple(
        Layer1RubricData(
            rubric_id=f"r{rubric_index}",
            rubric_order=rubric_index,
            dataset="emc2_set1",
            expectation_count=6,
            variants=tuple(
                Layer1VariantData(
                    variant_id=f"r{rubric_index}-v{variant_index}",
                    counts=rng.multinomial(
                        6,
                        [
                            0.48 - 0.03 * variant_index,
                            0.32,
                            0.20 + 0.03 * variant_index,
                        ],
                        size=8,
                    ),
                    arm_ids=tuple(f"a{index}" for index in range(8)),
                )
                for variant_index in range(3)
            ),
        )
        for rubric_index in range(4)
    )
    data = Layer1Dataset(rubrics=rubrics, dataset_levels=("emc2_set1",))
    moments = method_of_moments_start(data)
    vector = pack_hyperparameters(moments.hyperparameters, data.dataset_levels)
    reference_rubrics = tuple(
        ReferenceRubric(
            rubric_id=rubric.rubric_id,
            dataset=rubric.dataset,
            variant_counts=tuple(variant.counts for variant in rubric.variants),
        )
        for rubric in rubrics
    )

    actual = laplace_log_marginal(
        data,
        moments.hyperparameters,
        moments.rubric_latent_starts,
    )[0]
    expected = reference_laplace_log_marginal(
        vector,
        data.dataset_levels,
        reference_rubrics,
        dict(moments.rubric_latent_starts),
    )

    assert actual == pytest.approx(expected, rel=KERNEL_RTOL, abs=KERNEL_ATOL)
