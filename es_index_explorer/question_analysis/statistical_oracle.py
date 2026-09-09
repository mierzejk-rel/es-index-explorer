"""Frozen R-reference generation and offline Python oracle verification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, model_validator

from es_index_explorer.question_analysis.cluster_covariance import (
    ClusterCovariance,
    joint_wald_statistic,
    three_term_cluster_covariance,
)
from es_index_explorer.question_analysis.contracts import WorkflowCommand
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
    NumericalError,
)
from es_index_explorer.question_analysis.glm_primitives import (
    linear_bread,
    linear_score_rows,
    solve_unrestricted_linear,
)
from es_index_explorer.question_analysis.seeds import derive_stream_seed
from es_index_explorer.question_analysis.storage import (
    canonical_json_bytes,
    sha256_file,
)
from es_index_explorer.question_analysis.workspace import (
    AnalysisWorkspace,
    StepExecutionResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORACLE_ROOT = PROJECT_ROOT / "tests" / "oracles" / "fwildclusterboot"
FIXTURE_ROOT = ORACLE_ROOT / "fixtures"
INPUT_FILE = FIXTURE_ROOT / "f6-linear-input.csv"
CONTRACT_FILE = FIXTURE_ROOT / "f6-linear-contract.json"
OUTPUT_FILE = FIXTURE_ROOT / "f6-linear-r-output.json"
COVARIANCE_REFERENCE_FILE = FIXTURE_ROOT / "f6-linear-r-covariance-reference.json"
COVARIANCE_REFERENCE_SOURCE = ORACLE_ROOT / "covariance_reference.R"
WEIGHTS_FILE = FIXTURE_ROOT / "f6-linear-rademacher-weights.csv"
PROVENANCE_FILE = FIXTURE_ROOT / "f6-linear-provenance.json"
P_VALUE_TOLERANCE = 1e-4
WALD_RELATIVE_TOLERANCE = 1e-6
WALD_ABSOLUTE_TOLERANCE = 1e-12
MATRIX_RELATIVE_TOLERANCE = 1e-8
MATRIX_ABSOLUTE_TOLERANCE = 1e-10
SYMMETRY_ABSOLUTE_TOLERANCE = 1e-12
PSD_EIGENVALUE_TOLERANCE = 1e-10


class OracleContract(BaseModel):
    """Define the frozen F6 linear reference input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    row_count: int = Field(gt=0)
    design_columns: tuple[str, ...]
    restriction_column: str
    rubric_reference_order: int
    arm_order: tuple[str, ...]
    bootstrap_replicates: int = 9_999
    master_seed: int = 20
    r_oracle_seed: int
    seed_words_signed: tuple[int, int]


class ROracleSmallSampleCorrection(BaseModel):
    """Validate the frozen native-R small-sample correction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adj: Literal[False] = False
    fixef_K: Literal["none"] = Field(default="none", alias="fixef.K")
    cluster_adj: Literal[True] = Field(default=True, alias="cluster.adj")
    cluster_df: Literal["conventional"] = Field(
        default="conventional",
        alias="cluster.df",
    )


class ROracleCall(BaseModel):
    """Validate every frozen argument of the native-R oracle call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clustid: tuple[Literal["rubric"], Literal["arm"]]
    bootcluster: Literal["arm"]
    B: int = Field(gt=0)
    type: Literal["rademacher"]
    impose_null: Literal[True]
    engine: Literal["R"]
    sampling: Literal["dqrng"]
    conf_int: Literal[False]
    ssc: ROracleSmallSampleCorrection


class ROracleOutput(BaseModel):
    """Validate the immutable R output fixture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    p_f: float = Field(ge=0, le=1)
    t_stat: float
    W_obs: float = Field(ge=0)
    bootstrap_replicates: int = Field(gt=0)
    arm_count: int = Field(gt=1)
    invalid_t_count: int = Field(ge=0)
    r_version: str
    fwildclusterboot_version: str
    dqrng_version: str
    call: ROracleCall

    @model_validator(mode="after")
    def validate_call_contract(self) -> "ROracleOutput":
        """Require the recorded call to match the reported bootstrap run."""
        if self.call.B != self.bootstrap_replicates:
            raise ValueError("R oracle call B does not match bootstrap_replicates")
        if self.invalid_t_count > self.bootstrap_replicates:
            raise ValueError("R oracle invalid count exceeds bootstrap_replicates")
        return self


class RClusterCounts(BaseModel):
    """Validate realised cluster counts in the independent R reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rubric: int = Field(gt=1)
    arm: int = Field(gt=1)
    intersection: int = Field(gt=1)


class RMeatReference(BaseModel):
    """Validate the four independent R cluster-meat matrices."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rubric: tuple[tuple[float, ...], ...]
    arm: tuple[tuple[float, ...], ...]
    intersection: tuple[tuple[float, ...], ...]
    combined: tuple[tuple[float, ...], ...]


class RCovarianceReference(BaseModel):
    """Validate the independent base-R CGM, PSD, and Wald reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    row_count: int = Field(gt=0)
    parameter_count: int = Field(gt=0)
    design_columns: tuple[str, ...]
    restriction_column: str
    psd_tolerance: float = Field(ge=0)
    cluster_counts: RClusterCounts
    coefficients: tuple[float, ...]
    bread: tuple[tuple[float, ...], ...]
    meat: RMeatReference
    unprojected_covariance: tuple[tuple[float, ...], ...]
    eigenvalues_before: tuple[float, ...]
    eigenvalues_after: tuple[float, ...]
    projected_covariance: tuple[tuple[float, ...], ...]
    materially_indefinite: bool
    projection_applied: bool
    raw_W_obs: float = Field(ge=0)
    projected_W_obs: float = Field(ge=0)
    relative_projection_shift: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_numerical_contract(self) -> "RCovarianceReference":
        """Require complete, finite, symmetric, and PSD reference objects."""
        width = self.parameter_count
        if (
            len(self.design_columns) != width
            or len(self.coefficients) != width
            or len(self.eigenvalues_before) != width
            or len(self.eigenvalues_after) != width
            or self.restriction_column not in self.design_columns
        ):
            raise ValueError("R covariance reference dimensions are inconsistent")
        matrices = (
            self.bread,
            self.meat.rubric,
            self.meat.arm,
            self.meat.intersection,
            self.meat.combined,
            self.unprojected_covariance,
            self.projected_covariance,
        )
        arrays = tuple(np.asarray(matrix, dtype=float) for matrix in matrices)
        vectors = (
            np.asarray(self.coefficients, dtype=float),
            np.asarray(self.eigenvalues_before, dtype=float),
            np.asarray(self.eigenvalues_after, dtype=float),
        )
        if any(array.shape != (width, width) for array in arrays):
            raise ValueError(
                "R covariance reference matrices are not square and aligned"
            )
        if not all(np.isfinite(value).all() for value in (*arrays, *vectors)):
            raise ValueError("R covariance reference values must be finite")
        if not all(
            np.allclose(array, array.T, rtol=0.0, atol=SYMMETRY_ABSOLUTE_TOLERANCE)
            for array in arrays
        ):
            raise ValueError("R covariance reference matrices must be symmetric")
        projected_minimum = float(np.linalg.eigvalsh(arrays[-1]).min())
        if projected_minimum < -PSD_EIGENVALUE_TOLERANCE:
            raise ValueError("R projected covariance is not positive semidefinite")
        return self


class OracleProvenance(BaseModel):
    """Record immutable identities for every R reference component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 2
    docker_platform: str
    docker_image: str
    docker_image_id: str
    dockerfile_sha256: str
    dockerignore_sha256: str
    renv_lock_sha256: str
    runner_sha256: str
    covariance_reference_source_sha256: str
    input_sha256: str
    contract_sha256: str
    output_sha256: str
    covariance_reference_output_sha256: str
    weights_sha256: str
    source_analysis_root: str
    fwildclusterboot_source_sha256: str
    fwildclusterboot_remote_sha: str


class NumericalAgreement(BaseModel):
    """Persist one numerical oracle comparison and its tolerances."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_absolute_difference: float = Field(ge=0)
    maximum_reference_magnitude: float = Field(ge=0)
    maximum_tolerance_ratio: float = Field(ge=0)
    relative_tolerance: float = Field(ge=0)
    absolute_tolerance: float = Field(ge=0)
    passed: bool


class ProductionCovarianceVerification(BaseModel):
    """Persist production-versus-R CGM, PSD, and Wald comparisons."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_counts_match: bool
    projection_flags_match: bool
    symmetry_passed: bool
    projected_psd_passed: bool
    coefficients: NumericalAgreement
    bread: NumericalAgreement
    rubric_meat: NumericalAgreement
    arm_meat: NumericalAgreement
    intersection_meat: NumericalAgreement
    combined_meat: NumericalAgreement
    unprojected_covariance: NumericalAgreement
    eigenvalues_before: NumericalAgreement
    eigenvalues_after: NumericalAgreement
    projected_covariance: NumericalAgreement
    external_raw_wald: NumericalAgreement
    production_raw_wald: NumericalAgreement
    production_projected_wald: NumericalAgreement
    r_raw_W_obs: float = Field(ge=0)
    python_raw_W_obs: float = Field(ge=0)
    r_projected_W_obs: float = Field(ge=0)
    python_projected_W_obs: float = Field(ge=0)
    relative_projection_shift: float = Field(ge=0)
    materially_indefinite: bool
    projection_applied: bool
    passed: bool


class OracleVerification(BaseModel):
    """Persist the offline Python/R oracle comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 4
    passed: bool
    python_p_f: float
    r_p_f: float
    p_f_absolute_difference: float
    p_f_tolerance: float
    python_parallel_raw_W_obs: float
    fwildclusterboot_raw_W_obs: float
    raw_W_obs_relative_difference: float
    raw_W_obs_relative_tolerance: float
    python_invalid_t_count: int
    r_invalid_t_count: int
    production_covariance: ProductionCovarianceVerification
    fixture_hashes: dict[str, str]
    oracle_scope: str = "linear_f6_external_raw_and_independent_r_psd"
    external_p_value_convention: Literal[
        "fwildclusterboot_valid_only_strict_no_plus_one"
    ] = "fwildclusterboot_valid_only_strict_no_plus_one"
    production_sampled_p_value_convention: Literal["replenish_to_B_then_plus_one"] = (
        "replenish_to_B_then_plus_one"
    )
    production_enumerated_p_value_convention: Literal[
        "full_support_bracket_no_plus_one"
    ] = "full_support_bracket_no_plus_one"


@dataclass(frozen=True, slots=True)
class PythonOracleResult:
    """Store the Python linear-special-case reference result."""

    p_f: float
    W_obs: float
    bootstrap_wald: np.ndarray
    invalid_t_count: int


@dataclass(frozen=True, slots=True)
class ProductionLinearCovarianceResult:
    """Store the production linear CGM, PSD, and Wald calculation."""

    coefficients: np.ndarray
    bread: np.ndarray
    covariance: ClusterCovariance
    raw_wald: float
    projected_wald: float


def _signed_seed_words(seed: int) -> tuple[int, int]:
    words = ((seed >> 32) & 0xFFFFFFFF, seed & 0xFFFFFFFF)
    high = words[0] - 2**32 if words[0] >= 2**31 else words[0]
    low = words[1] - 2**32 if words[1] >= 2**31 else words[1]
    return high, low


def build_f6_linear_fixture(
    analysis_root: Path,
) -> tuple[pd.DataFrame, OracleContract]:
    """Build the frozen resolved-only F6 linear-reduction input."""
    tables = analysis_root / "tables"
    criterion_path = tables / "criterion_table.parquet"
    features_path = tables / "features_deterministic.parquet"
    if not criterion_path.is_file() or not features_path.is_file():
        raise MalformedInputError(
            "Oracle fixture generation requires criterion and feature tables"
        )
    criterion = pd.read_parquet(criterion_path)
    features = pd.read_parquet(features_path)
    questions = features.loc[
        features["item_type"].eq("question"),
        ["variant_id", "token_count"],
    ].copy()
    if (
        questions["variant_id"].duplicated().any()
        or questions["token_count"].isna().any()
    ):
        raise MalformedInputError("F6 question features must be unique and complete")
    rows = criterion.loc[
        criterion["eligible"].astype(bool) & criterion["state"].isin(("PASS", "FAIL")),
        [
            "criterion_observation_id",
            "variant_id",
            "rubric_id",
            "rubric_order",
            "arm_id",
            "state",
        ],
    ].merge(questions, on="variant_id", how="left", validate="many_to_one")
    if rows["token_count"].isna().any() or rows.empty:
        raise MalformedInputError(
            "F6 linear reduction has missing token counts or no rows"
        )
    rows = rows.sort_values("criterion_observation_id", kind="stable").reset_index(
        drop=True
    )
    reference_order = int(rows["rubric_order"].min())
    rubric_orders = sorted(int(value) for value in rows["rubric_order"].unique())
    design_columns = ["intercept", "token_count"]
    output = pd.DataFrame(
        {
            "y": rows["state"].eq("PASS").astype(int),
            "rubric": rows["rubric_id"].astype(str),
            "arm": rows["arm_id"].astype(str),
            "intercept": 1.0,
            "token_count": rows["token_count"].astype(float),
        }
    )
    for rubric_order in rubric_orders:
        if rubric_order == reference_order:
            continue
        column = f"rubric_{rubric_order:02d}"
        output[column] = rows["rubric_order"].eq(rubric_order).astype(float)
        design_columns.append(column)
    if np.linalg.matrix_rank(output[design_columns].to_numpy(dtype=float)) != len(
        design_columns
    ):
        raise GateFailureError("F6 linear oracle design is rank deficient")
    # collapse::GRP assigns bootstrap-cluster group IDs in sorted label order.
    arm_order = tuple(sorted(output["arm"].astype(str).unique()))
    seed = derive_stream_seed("r_oracle")
    contract = OracleContract(
        row_count=len(output),
        design_columns=tuple(design_columns),
        restriction_column="token_count",
        rubric_reference_order=reference_order,
        arm_order=arm_order,
        r_oracle_seed=seed,
        seed_words_signed=_signed_seed_words(seed),
    )
    return output, contract


def fixture_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize an oracle input frame deterministically."""
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
        "utf-8"
    )


def _restriction(contract: OracleContract) -> np.ndarray:
    restriction = np.zeros((1, len(contract.design_columns)), dtype=float)
    restriction[0, contract.design_columns.index(contract.restriction_column)] = 1.0
    return restriction


def compute_python_linear_oracle(
    input_frame: pd.DataFrame,
    contract: OracleContract,
    weight_matrix: np.ndarray,
) -> PythonOracleResult:
    """Replay native R's valid-only strict p-value on its persisted sign schedule.

    This function reproduces ``fwildclusterboot`` and intentionally does not
    implement production WCR replenishment or the sampled ``+1`` convention.
    """
    design = input_frame[list(contract.design_columns)].to_numpy(dtype=float)
    response = input_frame["y"].to_numpy(dtype=float)
    rubric = input_frame["rubric"].astype(str).tolist()
    arm = input_frame["arm"].astype(str).tolist()
    restriction = _restriction(contract)
    unrestricted = solve_unrestricted_linear(design, response)
    bread = linear_bread(design)
    try:
        restriction_direction = np.linalg.solve(bread.T, restriction[0])
    except np.linalg.LinAlgError as error:
        raise NumericalError("Linear oracle bread is singular") from error
    if weight_matrix.shape != (
        len(contract.arm_order),
        contract.bootstrap_replicates,
    ):
        raise MalformedInputError("Rademacher weight fixture has the wrong shape")
    arm_indices = {arm_id: index for index, arm_id in enumerate(contract.arm_order)}
    try:
        row_arm_indices = np.asarray([arm_indices[value] for value in arm], dtype=int)
    except KeyError as error:
        raise MalformedInputError("Oracle input contains an unknown arm") from error
    inverse_bread = np.linalg.solve(bread, np.eye(bread.shape[0]))
    x_a_r = design @ restriction_direction
    beta_hat = unrestricted.coefficients
    restriction_variance = float((restriction @ inverse_bread @ restriction.T)[0, 0])
    q_vector = -x_a_r / restriction_variance
    residual = response - design @ beta_hat
    p_vector = residual - q_vector * float((restriction @ beta_hat)[0])
    x_a_r_x = x_a_r[:, None] * design
    x_a_r_p = x_a_r * p_vector
    arm_count = len(contract.arm_order)
    p1 = np.zeros((arm_count, design.shape[1]), dtype=float)
    np.add.at(p1, row_arm_indices, design * p_vector[:, None])
    p2_boot = np.zeros((len(response), arm_count), dtype=float)
    p2_boot[np.arange(len(response)), row_arm_indices] = x_a_r_p
    all_weights = np.column_stack((np.ones(arm_count, dtype=float), weight_matrix))
    variance_terms = np.zeros(contract.bootstrap_replicates + 1, dtype=float)
    cluster_specs: tuple[tuple[list[object], float], ...] = (
        (rubric, 1.0),
        (arm, 1.0),
        (list(zip(rubric, arm, strict=True)), -1.0),
    )
    for labels, sign in cluster_specs:
        unique_labels = sorted(set(labels), key=repr)
        label_indices = {label: index for index, label in enumerate(unique_labels)}
        codes = np.asarray([label_indices[label] for label in labels], dtype=int)
        cluster_count = len(unique_labels)
        sx = np.zeros((cluster_count, design.shape[1]), dtype=float)
        np.add.at(sx, codes, x_a_r_x)
        p2 = np.zeros((cluster_count, arm_count), dtype=float)
        np.add.at(p2, codes, p2_boot)
        p_all = p2 - (sx @ inverse_bread) @ p1.T
        projected = p_all @ all_weights
        correction = sign * cluster_count / (cluster_count - 1)
        variance_terms += correction * np.square(projected).sum(axis=0)
    numerators = (
        np.bincount(row_arm_indices, weights=x_a_r_p, minlength=arm_count) @ all_weights
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        t_statistics = numerators / np.sqrt(variance_terms)
    observed_wald = float(t_statistics[0] ** 2)
    valid_t = t_statistics[1:][np.isfinite(t_statistics[1:])]
    bootstrap_wald = np.square(valid_t)
    p_value = float(np.mean(np.abs(valid_t) > abs(t_statistics[0])))
    return PythonOracleResult(
        p_value,
        observed_wald,
        bootstrap_wald,
        contract.bootstrap_replicates - len(valid_t),
    )


def compute_production_linear_covariance(
    input_frame: pd.DataFrame,
    contract: OracleContract,
) -> ProductionLinearCovarianceResult:
    """Compute the F6 linear covariance through the production CGM and PSD path."""
    design = input_frame[list(contract.design_columns)].to_numpy(dtype=float)
    response = input_frame["y"].to_numpy(dtype=float)
    rubric = input_frame["rubric"].astype(str).tolist()
    arm = input_frame["arm"].astype(str).tolist()
    restriction = _restriction(contract)
    unrestricted = solve_unrestricted_linear(design, response)
    bread = linear_bread(design)
    scores = linear_score_rows(design, response, unrestricted.coefficients)
    covariance = three_term_cluster_covariance(bread, scores, rubric, arm)
    raw_wald = joint_wald_statistic(
        unrestricted.coefficients,
        covariance.unprojected_covariance,
        restriction,
    ).value
    projected_wald = joint_wald_statistic(
        unrestricted.coefficients,
        covariance.covariance,
        restriction,
    ).value
    return ProductionLinearCovarianceResult(
        coefficients=unrestricted.coefficients,
        bread=bread,
        covariance=covariance,
        raw_wald=raw_wald,
        projected_wald=projected_wald,
    )


def _numerical_agreement(
    actual: ArrayLike,
    expected: ArrayLike,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> NumericalAgreement:
    actual_values = np.asarray(actual, dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    if actual_values.shape != expected_values.shape:
        return NumericalAgreement(
            maximum_absolute_difference=float("inf"),
            maximum_reference_magnitude=float("inf"),
            maximum_tolerance_ratio=float("inf"),
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            passed=False,
        )
    differences = np.abs(actual_values - expected_values)
    tolerance_limits = absolute_tolerance + relative_tolerance * np.abs(expected_values)
    with np.errstate(invalid="ignore", divide="ignore"):
        tolerance_ratios = np.where(
            tolerance_limits > 0,
            differences / tolerance_limits,
            np.where(differences == 0, 0.0, float("inf")),
        )
    return NumericalAgreement(
        maximum_absolute_difference=float(np.max(differences, initial=0.0)),
        maximum_reference_magnitude=float(np.max(np.abs(expected_values), initial=0.0)),
        maximum_tolerance_ratio=float(np.max(tolerance_ratios, initial=0.0)),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        passed=bool(
            np.allclose(
                actual_values,
                expected_values,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        ),
    )


def verify_production_covariance(
    input_frame: pd.DataFrame,
    contract: OracleContract,
    r_output: ROracleOutput,
    reference: RCovarianceReference,
) -> ProductionCovarianceVerification:
    if (
        reference.row_count != contract.row_count
        or reference.parameter_count != len(contract.design_columns)
        or reference.design_columns != contract.design_columns
        or reference.restriction_column != contract.restriction_column
        or reference.psd_tolerance != PSD_EIGENVALUE_TOLERANCE
    ):
        raise GateFailureError("R covariance reference does not match the F6 contract")
    production = compute_production_linear_covariance(input_frame, contract)
    covariance = production.covariance
    meat = covariance.meat
    psd = covariance.psd
    matrix_arguments = {
        "relative_tolerance": MATRIX_RELATIVE_TOLERANCE,
        "absolute_tolerance": MATRIX_ABSOLUTE_TOLERANCE,
    }
    wald_arguments = {
        "relative_tolerance": WALD_RELATIVE_TOLERANCE,
        "absolute_tolerance": WALD_ABSOLUTE_TOLERANCE,
    }
    agreements = {
        "coefficients": _numerical_agreement(
            production.coefficients, reference.coefficients, **matrix_arguments
        ),
        "bread": _numerical_agreement(
            production.bread, reference.bread, **matrix_arguments
        ),
        "rubric_meat": _numerical_agreement(
            meat.rubric, reference.meat.rubric, **matrix_arguments
        ),
        "arm_meat": _numerical_agreement(
            meat.arm, reference.meat.arm, **matrix_arguments
        ),
        "intersection_meat": _numerical_agreement(
            meat.intersection, reference.meat.intersection, **matrix_arguments
        ),
        "combined_meat": _numerical_agreement(
            meat.combined, reference.meat.combined, **matrix_arguments
        ),
        "unprojected_covariance": _numerical_agreement(
            covariance.unprojected_covariance,
            reference.unprojected_covariance,
            **matrix_arguments,
        ),
        "eigenvalues_before": _numerical_agreement(
            psd.eigenvalues_before,
            reference.eigenvalues_before,
            **matrix_arguments,
        ),
        "eigenvalues_after": _numerical_agreement(
            psd.eigenvalues_after,
            reference.eigenvalues_after,
            **matrix_arguments,
        ),
        "projected_covariance": _numerical_agreement(
            covariance.covariance,
            reference.projected_covariance,
            **matrix_arguments,
        ),
        "external_raw_wald": _numerical_agreement(
            reference.raw_W_obs, r_output.W_obs, **wald_arguments
        ),
        "production_raw_wald": _numerical_agreement(
            production.raw_wald, reference.raw_W_obs, **wald_arguments
        ),
        "production_projected_wald": _numerical_agreement(
            production.projected_wald,
            reference.projected_W_obs,
            **wald_arguments,
        ),
    }
    cluster_counts_match = (
        meat.rubric_cluster_count == reference.cluster_counts.rubric
        and meat.arm_cluster_count == reference.cluster_counts.arm
        and meat.intersection_cluster_count == reference.cluster_counts.intersection
    )
    projection_flags_match = (
        psd.materially_indefinite == reference.materially_indefinite
        and psd.projection_applied == reference.projection_applied
    )
    symmetry_passed = all(
        np.allclose(matrix, matrix.T, rtol=0.0, atol=SYMMETRY_ABSOLUTE_TOLERANCE)
        for matrix in (
            covariance.unprojected_covariance,
            covariance.covariance,
        )
    )
    projected_psd_passed = bool(
        np.linalg.eigvalsh(covariance.covariance).min() >= -PSD_EIGENVALUE_TOLERANCE
    )
    passed = (
        cluster_counts_match
        and projection_flags_match
        and symmetry_passed
        and projected_psd_passed
        and all(agreement.passed for agreement in agreements.values())
    )
    return ProductionCovarianceVerification(
        cluster_counts_match=cluster_counts_match,
        projection_flags_match=projection_flags_match,
        symmetry_passed=symmetry_passed,
        projected_psd_passed=projected_psd_passed,
        **agreements,
        r_raw_W_obs=reference.raw_W_obs,
        python_raw_W_obs=production.raw_wald,
        r_projected_W_obs=reference.projected_W_obs,
        python_projected_W_obs=production.projected_wald,
        relative_projection_shift=reference.relative_projection_shift,
        materially_indefinite=psd.materially_indefinite,
        projection_applied=psd.projection_applied,
        passed=passed,
    )


def _read_model[T: BaseModel](path: Path, model: type[T]) -> T:
    if not path.is_file():
        raise MalformedInputError(f"Required oracle fixture is missing: {path.name}")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise MalformedInputError(f"Invalid oracle fixture: {path.name}") from error


def verify_fixture_provenance() -> OracleProvenance:
    """Verify every immutable R fixture hash against its provenance."""
    provenance = _read_model(PROVENANCE_FILE, OracleProvenance)
    expected = {
        ORACLE_ROOT / "Dockerfile": provenance.dockerfile_sha256,
        ORACLE_ROOT / ".dockerignore": provenance.dockerignore_sha256,
        ORACLE_ROOT / "renv.lock": provenance.renv_lock_sha256,
        ORACLE_ROOT / "run_reference.R": provenance.runner_sha256,
        COVARIANCE_REFERENCE_SOURCE: provenance.covariance_reference_source_sha256,
        INPUT_FILE: provenance.input_sha256,
        CONTRACT_FILE: provenance.contract_sha256,
        OUTPUT_FILE: provenance.output_sha256,
        COVARIANCE_REFERENCE_FILE: provenance.covariance_reference_output_sha256,
        WEIGHTS_FILE: provenance.weights_sha256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise GateFailureError(f"R oracle fixture provenance mismatch: {path.name}")
    return provenance


def verify_r_oracle_fixture() -> OracleVerification:
    """Compare the Python linear special case with the immutable R fixture."""
    verify_fixture_provenance()
    contract = _read_model(CONTRACT_FILE, OracleContract)
    r_output = _read_model(OUTPUT_FILE, ROracleOutput)
    covariance_reference = _read_model(COVARIANCE_REFERENCE_FILE, RCovarianceReference)
    input_frame = pd.read_csv(INPUT_FILE)
    weights = pd.read_csv(WEIGHTS_FILE).to_numpy(dtype=float)
    python = compute_python_linear_oracle(input_frame, contract, weights)
    production_covariance = verify_production_covariance(
        input_frame,
        contract,
        r_output,
        covariance_reference,
    )
    p_difference = abs(python.p_f - r_output.p_f)
    wald_denominator = abs(r_output.W_obs)
    wald_difference = (
        abs(python.W_obs - r_output.W_obs) / wald_denominator
        if wald_denominator > 0
        else (0.0 if python.W_obs == r_output.W_obs else float("inf"))
    )
    passed = (
        p_difference <= P_VALUE_TOLERANCE
        and wald_difference <= WALD_RELATIVE_TOLERANCE
        and python.invalid_t_count == r_output.invalid_t_count
        and production_covariance.passed
    )
    verification = OracleVerification(
        passed=passed,
        python_p_f=python.p_f,
        r_p_f=r_output.p_f,
        p_f_absolute_difference=p_difference,
        p_f_tolerance=P_VALUE_TOLERANCE,
        python_parallel_raw_W_obs=python.W_obs,
        fwildclusterboot_raw_W_obs=r_output.W_obs,
        raw_W_obs_relative_difference=wald_difference,
        raw_W_obs_relative_tolerance=WALD_RELATIVE_TOLERANCE,
        python_invalid_t_count=python.invalid_t_count,
        r_invalid_t_count=r_output.invalid_t_count,
        production_covariance=production_covariance,
        fixture_hashes={
            path.name: sha256_file(path)
            for path in (
                INPUT_FILE,
                CONTRACT_FILE,
                OUTPUT_FILE,
                COVARIANCE_REFERENCE_FILE,
                WEIGHTS_FILE,
            )
        },
    )
    if not passed:
        raise GateFailureError(
            "Python linear special case does not reproduce the frozen R oracle"
        )
    return verification


def run_oracle(workspace: AnalysisWorkspace) -> StepExecutionResult:
    """Run the offline statistical oracle gate and persist its evidence."""
    if (workspace.root / "tables" / "criterion_table.parquet").is_file() and (
        workspace.root / "tables" / "features_deterministic.parquet"
    ).is_file():
        rebuilt_input, rebuilt_contract = build_f6_linear_fixture(workspace.root)
        if (
            fixture_csv_bytes(rebuilt_input) != INPUT_FILE.read_bytes()
            or canonical_json_bytes(rebuilt_contract) != CONTRACT_FILE.read_bytes()
        ):
            raise GateFailureError(
                "Current analysis tables do not reproduce the frozen R oracle input"
            )
    verification = verify_r_oracle_fixture()
    artifact = workspace.store.write_json(
        "statistics/r_oracle_verification.json",
        verification,
        created_by=WorkflowCommand.ORACLE,
    )
    return StepExecutionResult(artifacts=(artifact,))
