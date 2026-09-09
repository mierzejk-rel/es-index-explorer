"""Two-way cluster meat, covariance, PSD, and Wald primitives."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from es_index_explorer.question_analysis.errors import (
    MalformedInputError,
    NumericalError,
    SingularRestrictionCovarianceError,
)

PSD_EIGENVALUE_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class ClusterMeat:
    """Store the three finite-sample-corrected cluster meat terms."""

    rubric: np.ndarray
    arm: np.ndarray
    intersection: np.ndarray
    combined: np.ndarray
    rubric_cluster_count: int
    arm_cluster_count: int
    intersection_cluster_count: int


@dataclass(frozen=True, slots=True)
class PsdProjection:
    """Store a symmetric positive-semidefinite projection and diagnostics."""

    symmetrized_input: np.ndarray
    matrix: np.ndarray
    eigenvalues_before: np.ndarray
    eigenvalues_after: np.ndarray
    materially_indefinite: bool
    projection_applied: bool


@dataclass(frozen=True, slots=True)
class ClusterCovariance:
    """Store a two-way covariance and its component diagnostics."""

    unprojected_covariance: np.ndarray
    covariance: np.ndarray
    meat: ClusterMeat
    psd: PsdProjection


@dataclass(frozen=True, slots=True)
class WaldStatistic:
    """Store a computable joint Wald statistic."""

    value: float
    restriction_covariance: np.ndarray
    restriction_rank: int


def _cluster_outer_sum(
    scores: np.ndarray, labels: Sequence[object]
) -> tuple[np.ndarray, int]:
    groups: dict[object, np.ndarray] = {}
    for row, label in zip(scores, labels, strict=True):
        if label is None:
            raise MalformedInputError("Cluster labels cannot be null")
        current = groups.get(label)
        groups[label] = row.copy() if current is None else current + row
    if len(groups) <= 1:
        raise NumericalError(
            "Cluster covariance requires at least two realised clusters"
        )
    outer_sum = sum(
        (np.outer(cluster_sum, cluster_sum) for cluster_sum in groups.values()),
        start=np.zeros((scores.shape[1], scores.shape[1]), dtype=float),
    )
    return outer_sum, len(groups)


def three_term_cluster_meat(
    score_rows: np.ndarray,
    rubric_clusters: Sequence[object],
    arm_clusters: Sequence[object],
) -> ClusterMeat:
    """Construct the finite-sample-corrected CGM add-add-subtract meat."""
    scores = np.asarray(score_rows, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0 or not np.isfinite(scores).all():
        raise MalformedInputError("Score rows must be a non-empty finite matrix")
    if len(rubric_clusters) != len(scores) or len(arm_clusters) != len(scores):
        raise MalformedInputError("Cluster labels must align with score rows")
    rubric_outer, rubric_count = _cluster_outer_sum(scores, rubric_clusters)
    arm_outer, arm_count = _cluster_outer_sum(scores, arm_clusters)
    intersections = tuple(zip(rubric_clusters, arm_clusters, strict=True))
    intersection_outer, intersection_count = _cluster_outer_sum(scores, intersections)
    rubric = rubric_count / (rubric_count - 1) * rubric_outer
    arm = arm_count / (arm_count - 1) * arm_outer
    intersection = intersection_count / (intersection_count - 1) * intersection_outer
    combined = rubric + arm - intersection
    return ClusterMeat(
        rubric=rubric,
        arm=arm,
        intersection=intersection,
        combined=combined,
        rubric_cluster_count=rubric_count,
        arm_cluster_count=arm_count,
        intersection_cluster_count=intersection_count,
    )


def project_psd(
    matrix: np.ndarray,
    *,
    tolerance: float = PSD_EIGENVALUE_TOLERANCE,
) -> PsdProjection:
    """Symmetrize and project a matrix onto the PSD cone."""
    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise MalformedInputError("PSD projection requires a finite square matrix")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise MalformedInputError(
            "PSD projection tolerance must be finite and non-negative"
        )
    symmetric = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.where(eigenvalues <= tolerance, 0.0, eigenvalues)
    projected = (eigenvectors * clipped) @ eigenvectors.T
    projected = (projected + projected.T) / 2.0
    materially_indefinite = bool(np.any(eigenvalues < -tolerance))
    projection_applied = bool(
        not np.array_equal(values, symmetric) or np.any(clipped != eigenvalues)
    )
    return PsdProjection(
        symmetrized_input=symmetric,
        matrix=projected,
        eigenvalues_before=eigenvalues,
        eigenvalues_after=clipped,
        materially_indefinite=materially_indefinite,
        projection_applied=projection_applied,
    )


def three_term_cluster_covariance(
    bread: np.ndarray,
    score_rows: np.ndarray,
    rubric_clusters: Sequence[object],
    arm_clusters: Sequence[object],
) -> ClusterCovariance:
    """Build and PSD-project the three-term two-way sandwich covariance."""
    information = np.asarray(bread, dtype=float)
    scores = np.asarray(score_rows, dtype=float)
    if (
        information.ndim != 2
        or information.shape[0] != information.shape[1]
        or information.shape[0] != scores.shape[1]
        or not np.isfinite(information).all()
    ):
        raise MalformedInputError("Bread must be finite, square, and score-aligned")
    meat = three_term_cluster_meat(scores, rubric_clusters, arm_clusters)
    try:
        left = np.linalg.solve(information, meat.combined)
        covariance = np.linalg.solve(information.T, left.T).T
    except np.linalg.LinAlgError as error:
        raise NumericalError("Cluster covariance bread is singular") from error
    psd = project_psd(covariance)
    return ClusterCovariance(
        unprojected_covariance=psd.symmetrized_input,
        covariance=psd.matrix,
        meat=meat,
        psd=psd,
    )


def joint_wald_statistic(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    restriction: np.ndarray,
    *,
    tolerance: float = PSD_EIGENVALUE_TOLERANCE,
) -> WaldStatistic:
    """Compute a joint Wald statistic or fail on a singular restricted covariance."""
    beta = np.asarray(coefficients, dtype=float)
    variance = np.asarray(covariance, dtype=float)
    constraints = np.asarray(restriction, dtype=float)
    if (
        beta.ndim != 1
        or variance.shape != (len(beta), len(beta))
        or constraints.ndim != 2
        or constraints.shape[1] != len(beta)
        or constraints.shape[0] == 0
        or not all(np.isfinite(value).all() for value in (beta, variance, constraints))
    ):
        raise MalformedInputError("Wald inputs have incompatible dimensions")
    restricted_covariance = constraints @ variance @ constraints.T
    rank = int(np.linalg.matrix_rank(restricted_covariance, tol=tolerance))
    if rank != constraints.shape[0]:
        raise SingularRestrictionCovarianceError(
            "Restricted covariance is singular after PSD projection"
        )
    contrast = constraints @ beta
    try:
        value = float(contrast.T @ np.linalg.solve(restricted_covariance, contrast))
    except np.linalg.LinAlgError as error:
        raise NumericalError("Restricted covariance is non-computable") from error
    if not np.isfinite(value):
        raise NumericalError("Wald statistic is non-finite")
    return WaldStatistic(value, restricted_covariance, rank)
