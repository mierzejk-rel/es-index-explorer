"""Stacked co-primary score alignment and block construction."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.seeds import rng_for


@dataclass(frozen=True, slots=True)
class StackedScoreSystem:
    """Store zero-padded co-primary scores on their union row index."""

    row_ids: tuple[str, ...]
    score_rows: np.ndarray
    rubric_clusters: tuple[object, ...]
    arm_clusters: tuple[object, ...]
    resolved_parameter_count: int
    conservative_parameter_count: int


def validate_common_design_columns(
    declared_columns: Sequence[str],
    resolved_columns: Sequence[str],
    conservative_columns: Sequence[str],
) -> tuple[str, ...]:
    """Require both co-primary blocks to retain one declared column set."""
    declared = tuple(declared_columns)
    if not declared or len(set(declared)) != len(declared):
        raise MalformedInputError(
            "Declared design columns must be non-empty and unique"
        )
    if tuple(resolved_columns) != declared or tuple(conservative_columns) != declared:
        raise MalformedInputError(
            "Resolved and conservative blocks must retain the declared columns"
        )
    return declared


def _score_map(
    row_ids: Sequence[str], scores: np.ndarray, *, label: str
) -> dict[str, np.ndarray]:
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or len(row_ids) != matrix.shape[0]:
        raise MalformedInputError(f"{label} scores are not aligned with row IDs")
    if len(set(row_ids)) != len(row_ids):
        raise MalformedInputError(f"{label} row IDs must be unique")
    if not np.isfinite(matrix).all():
        raise MalformedInputError(f"{label} scores must be finite")
    return {row_id: row.copy() for row_id, row in zip(row_ids, matrix, strict=True)}


def stack_score_rows(
    resolved_row_ids: Sequence[str],
    resolved_scores: np.ndarray,
    conservative_row_ids: Sequence[str],
    conservative_scores: np.ndarray,
    rubric_by_row: Mapping[str, object],
    arm_by_row: Mapping[str, object],
) -> StackedScoreSystem:
    """Align co-primary score blocks on a deterministic union row index."""
    resolved = _score_map(resolved_row_ids, resolved_scores, label="Resolved")
    conservative = _score_map(
        conservative_row_ids, conservative_scores, label="Conservative"
    )
    resolved_width = np.asarray(resolved_scores).shape[1]
    conservative_width = np.asarray(conservative_scores).shape[1]
    union = tuple(sorted(set(resolved) | set(conservative)))
    missing_cluster_rows = [
        row_id
        for row_id in union
        if row_id not in rubric_by_row or row_id not in arm_by_row
    ]
    if missing_cluster_rows:
        raise MalformedInputError("Every stacked row must have rubric and arm clusters")
    zeros_resolved = np.zeros(resolved_width, dtype=float)
    zeros_conservative = np.zeros(conservative_width, dtype=float)
    rows = np.vstack(
        [
            np.concatenate(
                (
                    resolved.get(row_id, zeros_resolved),
                    conservative.get(row_id, zeros_conservative),
                )
            )
            for row_id in union
        ]
    )
    return StackedScoreSystem(
        row_ids=union,
        score_rows=rows,
        rubric_clusters=tuple(rubric_by_row[row_id] for row_id in union),
        arm_clusters=tuple(arm_by_row[row_id] for row_id in union),
        resolved_parameter_count=resolved_width,
        conservative_parameter_count=conservative_width,
    )


def stacked_bread(
    resolved_bread: np.ndarray, conservative_bread: np.ndarray
) -> np.ndarray:
    """Construct the block-diagonal bread for two co-primary outcomes."""
    resolved = np.asarray(resolved_bread, dtype=float)
    conservative = np.asarray(conservative_bread, dtype=float)
    for label, matrix in (
        ("Resolved", resolved),
        ("Conservative", conservative),
    ):
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()
        ):
            raise MalformedInputError(f"{label} bread must be finite and square")
    return np.asarray(block_diag(resolved, conservative), dtype=float)


def stacked_restriction(
    resolved_restriction: np.ndarray, conservative_restriction: np.ndarray
) -> np.ndarray:
    """Construct the block-diagonal joint restriction matrix."""
    resolved = np.asarray(resolved_restriction, dtype=float)
    conservative = np.asarray(conservative_restriction, dtype=float)
    for label, matrix in (
        ("Resolved", resolved),
        ("Conservative", conservative),
    ):
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise MalformedInputError(f"{label} restriction must be a finite matrix")
    return np.asarray(block_diag(resolved, conservative), dtype=float)


def apply_arm_weights(
    score_rows: np.ndarray,
    arm_clusters: Sequence[object],
    weights_by_arm: Mapping[object, float],
) -> np.ndarray:
    """Apply one shared arm weight to every stacked score row."""
    scores = np.asarray(score_rows, dtype=float)
    if scores.ndim != 2 or len(arm_clusters) != len(scores):
        raise MalformedInputError("Arm clusters must align with stacked score rows")
    try:
        weights = np.asarray(
            [weights_by_arm[cluster] for cluster in arm_clusters], dtype=float
        )
    except KeyError as error:
        raise MalformedInputError(
            "Missing bootstrap weight for a realised arm"
        ) from error
    if not np.isfinite(weights).all():
        raise MalformedInputError("Bootstrap weights must be finite")
    return scores * weights[:, None]


def generate_offdiagonal_oracle_scores() -> StackedScoreSystem:
    """Generate the frozen 400-row correlated-score oracle fixture."""
    rng = rng_for("oracle_offdiag")
    scores = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=np.array([[1.0, 0.5], [0.5, 1.0]]),
        size=400,
    )
    row_ids = tuple(f"oracle-{index:03d}" for index in range(400))
    rubric = ("rubric-1",) * 200 + ("rubric-2",) * 200
    arm = ("arm-1",) * 100 + ("arm-2",) * 100 + ("arm-1",) * 100 + ("arm-2",) * 100
    return stack_score_rows(
        row_ids,
        scores[:, [0]],
        row_ids,
        scores[:, [1]],
        dict(zip(row_ids, rubric, strict=True)),
        dict(zip(row_ids, arm, strict=True)),
    )
