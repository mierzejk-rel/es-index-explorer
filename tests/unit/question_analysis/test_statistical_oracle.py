"""Tests for the immutable R fixture and offline Python oracle gate."""

import json
from pathlib import Path

import pandas as pd
import pytest

from es_index_explorer.question_analysis import statistical_oracle as oracle_module
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.statistical_oracle import (
    CONTRACT_FILE,
    INPUT_FILE,
    OUTPUT_FILE,
    WEIGHTS_FILE,
    OracleContract,
    ROracleOutput,
    build_f6_linear_fixture,
    compute_python_linear_oracle,
    verify_fixture_provenance,
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


def test_python_linear_special_case_reproduces_r_reference() -> None:
    verification = verify_r_oracle_fixture()

    assert verification.passed
    assert verification.p_f_absolute_difference == 0.0
    assert verification.W_obs_relative_difference < 1e-12
    assert verification.python_invalid_t_count == verification.r_invalid_t_count == 223
    assert verification.oracle_scope == "linear_f6_special_case_only"


def test_fixture_provenance_covers_every_reference_file() -> None:
    provenance = verify_fixture_provenance()

    assert provenance.docker_platform == "linux/amd64"
    assert provenance.docker_image == "simplemode-fwildclusterboot:0.14.3"
    assert provenance.docker_image_id.startswith("sha256:")
    assert provenance.fwildclusterboot_remote_sha == (
        "336bb574eba169ac0183317f01d0564791d8122f"
    )


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


def test_provenance_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied: dict[str, Path] = {}
    for name, source in (
        ("PROVENANCE_FILE", oracle_module.PROVENANCE_FILE),
        ("INPUT_FILE", INPUT_FILE),
        ("CONTRACT_FILE", CONTRACT_FILE),
        ("OUTPUT_FILE", OUTPUT_FILE),
        ("WEIGHTS_FILE", WEIGHTS_FILE),
    ):
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        monkeypatch.setattr(oracle_module, name, target)
        copied[name] = target
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    for filename in ("Dockerfile", ".dockerignore", "renv.lock", "run_reference.R"):
        source = oracle_module.ORACLE_ROOT / filename
        (oracle_root / filename).write_bytes(source.read_bytes())
    monkeypatch.setattr(oracle_module, "ORACLE_ROOT", oracle_root)
    copied["INPUT_FILE"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(GateFailureError, match="provenance mismatch"):
        oracle_module.verify_fixture_provenance()
