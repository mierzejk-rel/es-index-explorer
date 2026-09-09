"""Tests for the immutable R fixture and offline Python oracle gate."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from es_index_explorer.question_analysis import statistical_oracle as oracle_module
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.statistical_oracle import (
    CONTRACT_FILE,
    COVARIANCE_REFERENCE_FILE,
    COVARIANCE_REFERENCE_SOURCE,
    INPUT_FILE,
    OUTPUT_FILE,
    WEIGHTS_FILE,
    OracleContract,
    RCovarianceReference,
    ROracleOutput,
    _numerical_agreement,
    build_f6_linear_fixture,
    compute_python_linear_oracle,
    verify_fixture_provenance,
    verify_production_covariance,
    verify_r_oracle_fixture,
)

pytestmark = pytest.mark.unit


def test_committed_r_fixture_has_frozen_contract_and_versions() -> None:
    contract = OracleContract.model_validate_json(CONTRACT_FILE.read_text())
    output = ROracleOutput.model_validate_json(OUTPUT_FILE.read_text())
    weights = pd.read_csv(WEIGHTS_FILE)

    assert contract.bootstrap_replicates == 9_999
    assert contract.r_oracle_seed == 801891564970050136
    assert contract.seed_words_signed == (186704929, 913048152)
    assert contract.restriction_column == "token_count"
    assert contract.design_columns[:2] == ("intercept", "token_count")
    assert len(contract.design_columns) == 64
    assert len(contract.arm_order) == 28
    assert weights.shape == (28, 9_999)
    assert set(weights.to_numpy().ravel()) == {-1, 1}
    assert output.fwildclusterboot_version == "0.14.3"
    assert output.r_version.startswith("R version 4.4.3")
    assert output.invalid_t_count == 223
    assert output.schema_version == 2
    assert output.call.B == contract.bootstrap_replicates
    assert output.call.conf_int is False


def test_python_linear_special_case_reproduces_r_reference() -> None:
    verification = verify_r_oracle_fixture()

    assert verification.passed
    assert verification.p_f_absolute_difference == 0.0
    assert verification.raw_W_obs_relative_difference < 1e-12
    assert verification.python_invalid_t_count == verification.r_invalid_t_count == 223
    assert verification.oracle_scope == "linear_f6_external_raw_and_independent_r_psd"
    assert verification.schema_version == 4
    assert verification.external_p_value_convention == (
        "fwildclusterboot_valid_only_strict_no_plus_one"
    )
    assert (
        verification.production_sampled_p_value_convention
        == "replenish_to_B_then_plus_one"
    )
    assert verification.production_enumerated_p_value_convention == (
        "full_support_bracket_no_plus_one"
    )


def test_native_r_and_production_p_value_conventions_are_intentionally_distinct() -> (
    None
):
    contract = OracleContract.model_validate_json(CONTRACT_FILE.read_text())
    native_output = ROracleOutput.model_validate_json(OUTPUT_FILE.read_text())
    replay = compute_python_linear_oracle(
        pd.read_csv(INPUT_FILE),
        contract,
        pd.read_csv(WEIGHTS_FILE).to_numpy(dtype=float),
    )
    exceedances = int(np.count_nonzero(replay.bootstrap_wald >= replay.W_obs))
    production_formula_without_replenishment = (1 + exceedances) / (
        contract.bootstrap_replicates + 1
    )

    assert replay.invalid_t_count == native_output.invalid_t_count == 223
    assert replay.p_f == native_output.p_f
    assert production_formula_without_replenishment == pytest.approx(0.8681)
    assert abs(production_formula_without_replenishment - native_output.p_f) > 1e-4


def test_production_cgm_psd_and_wald_match_independent_r_reference() -> None:
    contract = OracleContract.model_validate_json(CONTRACT_FILE.read_text())
    r_output = ROracleOutput.model_validate_json(OUTPUT_FILE.read_text())
    reference = RCovarianceReference.model_validate_json(
        COVARIANCE_REFERENCE_FILE.read_text()
    )

    verification = verify_production_covariance(
        pd.read_csv(INPUT_FILE),
        contract,
        r_output,
        reference,
    )

    assert verification.passed
    assert verification.cluster_counts_match
    assert verification.projection_flags_match
    assert verification.symmetry_passed
    assert verification.projected_psd_passed
    assert verification.external_raw_wald.passed
    assert verification.production_raw_wald.passed
    assert verification.production_projected_wald.passed
    assert verification.rubric_meat.maximum_tolerance_ratio < 1.0
    assert verification.rubric_meat.maximum_reference_magnitude > 100_000
    assert all(
        agreement.passed
        for _field_name, agreement in verification
        if hasattr(agreement, "passed")
    )
    assert verification.materially_indefinite
    assert verification.projection_applied
    assert verification.relative_projection_shift == pytest.approx(0.10885542293567473)
    assert verification.python_raw_W_obs != pytest.approx(
        verification.python_projected_W_obs,
        rel=1e-3,
    )


def test_covariance_reference_has_complete_f6_contract() -> None:
    reference = RCovarianceReference.model_validate_json(
        COVARIANCE_REFERENCE_FILE.read_text()
    )

    assert reference.row_count == 36_623
    assert reference.parameter_count == 64
    assert reference.cluster_counts.model_dump() == {
        "rubric": 63,
        "arm": 28,
        "intersection": 1_764,
    }
    assert len(reference.bread) == reference.parameter_count
    assert len(reference.projected_covariance) == reference.parameter_count
    assert min(reference.eigenvalues_before) < -reference.psd_tolerance
    assert min(reference.eigenvalues_after) == 0.0


def test_fixture_provenance_covers_every_reference_file() -> None:
    provenance = verify_fixture_provenance()

    assert provenance.docker_platform == "linux/amd64"
    assert provenance.docker_image == "simplemode-fwildclusterboot:0.14.3"
    assert provenance.docker_image_id.startswith("sha256:")
    assert provenance.fwildclusterboot_remote_sha == (
        "336bb574eba169ac0183317f01d0564791d8122f"
    )
    assert len(provenance.covariance_reference_source_sha256) == 64
    assert len(provenance.covariance_reference_output_sha256) == 64


def test_canonical_linear_input_rebuilds_identically() -> None:
    project_root = Path(__file__).resolve().parents[3]
    analysis_root = (
        project_root / "artifacts" / "question_analysis" / "simplemode-v1-segment5"
    )
    rebuilt, contract = build_f6_linear_fixture(analysis_root)
    committed = pd.read_csv(INPUT_FILE)

    pd.testing.assert_frame_equal(rebuilt, committed, check_dtype=False)
    committed_contract = json.loads(CONTRACT_FILE.read_text())
    assert contract.model_dump(mode="json") == committed_contract


def test_fixture_builder_requires_both_source_tables(tmp_path: Path) -> None:
    with pytest.raises(MalformedInputError, match="requires criterion and feature"):
        build_f6_linear_fixture(tmp_path)


def test_python_oracle_rejects_misaligned_weight_fixture() -> None:
    contract = OracleContract.model_validate_json(CONTRACT_FILE.read_text())

    with pytest.raises(MalformedInputError, match="wrong shape"):
        compute_python_linear_oracle(
            pd.read_csv(INPUT_FILE),
            contract,
            pd.read_csv(WEIGHTS_FILE).to_numpy(dtype=float)[:, :-1],
        )


@pytest.mark.parametrize(
    "reference_update",
    (
        {"raw_W_obs": 1.0},
        {"projected_W_obs": 1.0},
    ),
)
def test_raw_or_projected_covariance_mismatch_blocks_production_gate(
    reference_update: dict[str, float],
) -> None:
    contract = OracleContract.model_validate_json(CONTRACT_FILE.read_text())
    r_output = ROracleOutput.model_validate_json(OUTPUT_FILE.read_text())
    reference = RCovarianceReference.model_validate_json(
        COVARIANCE_REFERENCE_FILE.read_text()
    ).model_copy(update=reference_update)

    verification = verify_production_covariance(
        pd.read_csv(INPUT_FILE),
        contract,
        r_output,
        reference,
    )

    assert not verification.passed
    if "raw_W_obs" in reference_update:
        assert not verification.external_raw_wald.passed
        assert not verification.production_raw_wald.passed
    else:
        assert not verification.production_projected_wald.passed


def test_covariance_reference_rejects_non_psd_projection() -> None:
    payload = json.loads(COVARIANCE_REFERENCE_FILE.read_text())
    payload["projected_covariance"][0][0] = -1.0

    with pytest.raises(ValueError, match="not positive semidefinite"):
        RCovarianceReference.model_validate(payload)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("dimensions", "dimensions are inconsistent"),
        ("matrix_shape", "matrices are not square and aligned"),
        ("finite", "values must be finite"),
        ("symmetric", "matrices must be symmetric"),
    ),
)
def test_covariance_reference_rejects_invalid_numerical_contract(
    case: str,
    message: str,
) -> None:
    payload = json.loads(COVARIANCE_REFERENCE_FILE.read_text())
    if case == "dimensions":
        payload["design_columns"].pop()
    elif case == "matrix_shape":
        payload["bread"].pop()
    elif case == "finite":
        payload["coefficients"][0] = float("nan")
    else:
        payload["bread"][0][1] += 1.0

    with pytest.raises(ValueError, match=message):
        RCovarianceReference.model_validate(payload)


def test_production_gate_rejects_reference_contract_mismatch() -> None:
    contract = OracleContract.model_validate_json(CONTRACT_FILE.read_text())
    r_output = ROracleOutput.model_validate_json(OUTPUT_FILE.read_text())
    reference = RCovarianceReference.model_validate_json(
        COVARIANCE_REFERENCE_FILE.read_text()
    ).model_copy(update={"row_count": contract.row_count + 1})

    with pytest.raises(GateFailureError, match="does not match the F6 contract"):
        verify_production_covariance(
            pd.read_csv(INPUT_FILE),
            contract,
            r_output,
            reference,
        )


def test_numerical_agreement_rejects_shape_mismatch() -> None:
    agreement = _numerical_agreement(
        [1.0],
        [1.0, 2.0],
        relative_tolerance=1e-8,
        absolute_tolerance=1e-10,
    )

    assert not agreement.passed
    assert agreement.maximum_absolute_difference == float("inf")
    assert agreement.maximum_reference_magnitude == float("inf")
    assert agreement.maximum_tolerance_ratio == float("inf")


@pytest.mark.parametrize(
    ("actual", "expected", "maximum_tolerance_ratio", "passed"),
    (
        ([1.0], [1.0], 0.0, True),
        ([5e-11], [0.0], 0.5, True),
        ([1_000_000.005], [1_000_000.0], 0.499999995, True),
        ([2e-10], [0.0], 2.0, False),
    ),
)
def test_numerical_agreement_reports_scale_aware_tolerance_ratio(
    actual: list[float],
    expected: list[float],
    maximum_tolerance_ratio: float,
    passed: bool,
) -> None:
    agreement = _numerical_agreement(
        actual,
        expected,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-10,
    )

    assert agreement.maximum_tolerance_ratio == pytest.approx(maximum_tolerance_ratio)
    assert agreement.passed is passed


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    (
        (("call", "conf_int"), True),
        (("call", "B"), 10_000),
        (("call", "ssc", "cluster.df"), "min"),
    ),
)
def test_r_oracle_output_rejects_call_contract_drift(
    field_path: tuple[str, ...],
    invalid_value: object,
) -> None:
    payload = json.loads(OUTPUT_FILE.read_text())
    target = payload
    for key in field_path[:-1]:
        target = cast(dict[str, object], target[key])
    target[field_path[-1]] = invalid_value

    with pytest.raises((ValidationError, ValueError)):
        ROracleOutput.model_validate(payload)


def test_r_oracle_output_rejects_missing_confidence_interval_setting() -> None:
    payload = json.loads(OUTPUT_FILE.read_text())
    payload["call"].pop("conf_int")

    with pytest.raises(ValidationError):
        ROracleOutput.model_validate(payload)


def test_provenance_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied: dict[str, Path] = {}
    for name, source in (
        ("PROVENANCE_FILE", oracle_module.PROVENANCE_FILE),
        ("INPUT_FILE", INPUT_FILE),
        ("CONTRACT_FILE", CONTRACT_FILE),
        ("OUTPUT_FILE", OUTPUT_FILE),
        ("COVARIANCE_REFERENCE_FILE", COVARIANCE_REFERENCE_FILE),
        ("WEIGHTS_FILE", WEIGHTS_FILE),
    ):
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        monkeypatch.setattr(oracle_module, name, target)
        copied[name] = target
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    for filename in (
        "Dockerfile",
        ".dockerignore",
        "renv.lock",
        "run_reference.R",
        "covariance_reference.R",
    ):
        source = oracle_module.ORACLE_ROOT / filename
        (oracle_root / filename).write_bytes(source.read_bytes())
    monkeypatch.setattr(oracle_module, "ORACLE_ROOT", oracle_root)
    monkeypatch.setattr(
        oracle_module,
        "COVARIANCE_REFERENCE_SOURCE",
        oracle_root / COVARIANCE_REFERENCE_SOURCE.name,
    )
    copied["INPUT_FILE"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(GateFailureError, match="provenance mismatch"):
        oracle_module.verify_fixture_provenance()
