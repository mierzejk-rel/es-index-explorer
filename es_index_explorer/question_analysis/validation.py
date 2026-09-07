"""Outcome-blind weighted feature validation and unlock evidence."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from es_index_explorer.question_analysis.contracts import (
    COGNITIVE_PROCESS_LEVELS,
    GOLD_VALIDATION_FEATURES,
    VALIDATION_BOOTSTRAP_REPLICATES,
    AnnotatorKind,
    ArtifactMetadata,
    FeatureValidationStatus,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.dsl_gate import (
    DslGateEvidence,
    evaluate_dsl_gate,
    incomplete_dsl_evidence,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.seeds import derive_stream_seed
from es_index_explorer.question_analysis.storage import (
    atomic_write_bytes,
    canonical_json_bytes,
    versioned_frame,
)
from es_index_explorer.question_analysis.workspace import (
    AnalysisWorkspace,
    StepExecutionResult,
)

FEATURE_VALIDATION_TABLE = "tables/feature_validation.parquet"
VALIDATED_FEATURES_TABLE = "tables/validated_features.parquet"
DSL_GATE_ARTIFACT = "dsl_gate.json"
VALIDATION_UNLOCK_ARTIFACT = "validation_unlock.json"
UNRESOLVED_VALIDATION_ARTIFACT = "unresolved_validation_items.json"
FEATURE_VALIDATION_REPORT = "partial_reports/04-feature-validation.md"

FEATURE_KINDS: Mapping[str, str] = {
    "qdmr_step_count": "numeric",
    "hop_structure": "nominal",
    "exhaustivity_requirement": "nominal",
    "negative_conclusiveness": "binary",
    "referring_form_type": "nominal",
    "answer_locality": "binary",
    "recall_orientation": "binary",
}
FEATURE_SUBSTANTIVE_LEVELS: Mapping[str, tuple[object, ...]] = {
    "hop_structure": ("atomic", "bridge", "comparison", "intersection"),
    "exhaustivity_requirement": (
        "mention_some",
        "weakly_exhaustive",
        "mention_all",
    ),
    "negative_conclusiveness": (False, True),
    "referring_form_type": ("full_name_form", "alias_or_handle", "email_address"),
    "answer_locality": ("single_passage", "cross_document_aggregation"),
    "recall_orientation": ("precision_oriented", "recall_oriented"),
}
STATUS_RANK = {
    FeatureValidationStatus.NOT_VALIDATED: 0,
    FeatureValidationStatus.VALIDATED_WITH_LIMITATIONS: 1,
    FeatureValidationStatus.VALIDATED: 2,
}


class FeatureDecision(BaseModel):
    """Record one write-once human feature-validation decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    order: int = Field(ge=1)
    feature: str
    dossier_sha256: str = Field(min_length=64, max_length=64)
    status: FeatureValidationStatus
    material_failure_mode: bool
    justification: str = Field(min_length=1)
    git_commit: str = Field(min_length=7)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Hold one weighted validity metric and matched baseline."""

    value: float
    baseline: float
    class_counts: dict[str, int]


def ht_confusion_totals(
    gold: Sequence[object],
    predicted: Sequence[object],
    inclusion_probabilities: Sequence[float],
    levels: Sequence[object],
) -> np.ndarray:
    """Compute Horvitz-Thompson confusion-cell population totals."""
    if not (
        len(gold) == len(predicted) == len(inclusion_probabilities) and len(gold) > 0
    ):
        raise MalformedInputError("HT confusion inputs must be non-empty and aligned")
    level_index = {level: index for index, level in enumerate(levels)}
    matrix = np.zeros((len(levels), len(levels)), dtype=float)
    for actual, forecast, probability in zip(
        gold, predicted, inclusion_probabilities, strict=True
    ):
        if actual not in level_index or forecast not in level_index:
            continue
        if not np.isfinite(probability) or probability <= 0 or probability > 1:
            raise MalformedInputError("Inclusion probabilities must lie in (0, 1]")
        matrix[level_index[actual], level_index[forecast]] += 1.0 / probability
    return matrix


def balanced_accuracy(matrix: np.ndarray) -> float:
    """Compute binary balanced accuracy from a 2-by-2 confusion matrix."""
    if matrix.shape != (2, 2):
        raise MalformedInputError("Balanced accuracy requires a 2-by-2 matrix")
    row_totals = matrix.sum(axis=1)
    if np.any(row_totals == 0):
        return float("nan")
    recalls = np.diag(matrix) / row_totals
    return float(recalls.mean())


def macro_f1(matrix: np.ndarray) -> float:
    """Compute macro-F1 from a square confusion matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MalformedInputError("Macro-F1 requires a square confusion matrix")
    scores: list[float] = []
    for index in range(matrix.shape[0]):
        true_positive = matrix[index, index]
        denominator = (
            2 * true_positive
            + matrix[:, index].sum()
            - true_positive
            + matrix[index, :].sum()
            - true_positive
        )
        scores.append(
            0.0 if denominator == 0 else float(2 * true_positive / denominator)
        )
    return float(np.mean(scores))


def quadratically_weighted_kappa(matrix: np.ndarray) -> float:
    """Compute quadratic weighted kappa from an ordinal confusion matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MalformedInputError("Weighted kappa requires a square confusion matrix")
    size = matrix.shape[0]
    if size < 2 or matrix.sum() == 0:
        return float("nan")
    weights = np.fromfunction(
        lambda left, right: ((left - right) / (size - 1)) ** 2,
        (size, size),
    )
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / matrix.sum()
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0:
        return float("nan")
    observed_disagreement = float((weights * matrix).sum())
    return 1.0 - observed_disagreement / expected_disagreement


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Compute a deterministic lower weighted median."""
    if len(values) != len(weights) or not values:
        raise MalformedInputError(
            "Weighted median inputs must be non-empty and aligned"
        )
    ordered = sorted(zip(values, weights, strict=True), key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        raise MalformedInputError("Weighted median requires positive total weight")
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= total / 2:
            return float(value)
    return float(ordered[-1][0])


def mae_skill(
    gold: Sequence[float],
    predicted: Sequence[float],
    inclusion_probabilities: Sequence[float],
) -> tuple[float, float]:
    """Compute HT-weighted MAE skill and its naive weighted-median MAE."""
    if not (
        len(gold) == len(predicted) == len(inclusion_probabilities) and len(gold) > 0
    ):
        raise MalformedInputError("MAE-skill inputs must be non-empty and aligned")
    weights = [1.0 / probability for probability in inclusion_probabilities]
    median = weighted_median(gold, weights)
    total_weight = sum(weights)
    model_mae = (
        sum(
            weight * abs(actual - forecast)
            for actual, forecast, weight in zip(gold, predicted, weights, strict=True)
        )
        / total_weight
    )
    naive_mae = (
        sum(
            weight * abs(actual - median)
            for actual, weight in zip(gold, weights, strict=True)
        )
        / total_weight
    )
    if naive_mae == 0:
        return float("nan"), naive_mae
    return 1.0 - model_mae / naive_mae, naive_mae


def krippendorff_alpha(
    ratings: Sequence[Sequence[object | None]],
    *,
    level: str,
    ordinal_levels: Sequence[object] | None = None,
) -> float:
    """Compute Krippendorff alpha for nominal, ordinal, or interval ratings."""
    pair_distances: list[float] = []
    observed_values: list[object] = []
    for unit in ratings:
        values = [value for value in unit if value is not None]
        observed_values.extend(values)
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                pair_distances.append(
                    _alpha_distance(left, right, level, ordinal_levels)
                )
    if not pair_distances or len(observed_values) < 2:
        return float("nan")
    observed_disagreement = float(np.mean(pair_distances))
    expected_distances = [
        _alpha_distance(left, right, level, ordinal_levels)
        for left_index, left in enumerate(observed_values)
        for right_index, right in enumerate(observed_values)
        if left_index != right_index
    ]
    expected_disagreement = float(np.mean(expected_distances))
    if expected_disagreement == 0:
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def stratified_bootstrap_interval(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    rng: np.random.Generator,
    replicates: int = VALIDATION_BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Compute a deterministic within-stratum percentile interval."""
    if replicates < 1:
        raise MalformedInputError("Bootstrap replicate count must be positive")
    if frame.empty or "stratum_id" not in frame:
        raise MalformedInputError("Bootstrap input requires non-empty stratum_id rows")
    groups = [
        group.reset_index(drop=True)
        for _, group in frame.groupby("stratum_id", sort=True)
    ]
    values: list[float] = []
    for _ in range(replicates):
        sampled_groups = [
            group.iloc[rng.integers(0, len(group), size=len(group))] for group in groups
        ]
        value = metric(pd.concat(sampled_groups, ignore_index=True))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    lower, upper = np.quantile(values, (0.025, 0.975))
    return float(lower), float(upper)


def build_feature_validation(
    gold_labels: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    bootstrap_replicates: int = VALIDATION_BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Build per-model weighted validation summaries and feature dossiers."""
    prediction_rows = _prediction_rows(gold_labels, normalized)
    summaries: list[dict[str, object]] = []
    dossiers: dict[str, dict[str, object]] = {}
    for feature in GOLD_VALIDATION_FEATURES:
        feature_rows = prediction_rows.loc[prediction_rows["feature"].eq(feature)]
        if feature_rows.empty:
            raise GateFailureError(f"Human gold contains no rows for {feature}")
        model_summaries: list[dict[str, object]] = []
        for model_id, model_rows in feature_rows.groupby("model_id", sort=True):
            point = _metric_summary(feature, model_rows)
            rng = _validation_rng(feature, str(model_id), "assigned_metric")
            lower, upper = stratified_bootstrap_interval(
                model_rows,
                lambda sample, current_feature=feature: (
                    _metric_summary(current_feature, sample).value
                ),
                rng=rng,
                replicates=bootstrap_replicates,
            )
            row: dict[str, object] = {
                "feature": feature,
                "model_id": str(model_id),
                "metric": _metric_name(feature),
                "metric_value": point.value,
                "metric_interval_lower": lower,
                "metric_interval_upper": upper,
                "matched_naive_baseline": point.baseline,
                "class_counts_json": json.dumps(point.class_counts, sort_keys=True),
                "metric_rule_passed": bool(
                    np.isfinite(lower) and lower > point.baseline
                ),
            }
            summaries.append(row)
            model_summaries.append(row)

        alpha_value, alpha_lower, alpha_upper = _feature_alpha(
            feature,
            feature_rows,
            bootstrap_replicates,
        )
        coverage_passed = _coverage_passed(feature, feature_rows)
        both_models_passed = len(model_summaries) == 2 and all(
            bool(summary["metric_rule_passed"]) for summary in model_summaries
        )
        structural_passed = (
            both_models_passed and np.isfinite(alpha_lower) and alpha_lower > 0
        )
        degenerate_gold = any(
            not np.isfinite(cast(float, summary["metric_value"]))
            or not np.isfinite(cast(float, summary["metric_interval_lower"]))
            or not np.isfinite(cast(float, summary["metric_interval_upper"]))
            for summary in model_summaries
        ) or not np.isfinite(alpha_value)
        dossier = {
            "schema_version": 1,
            "feature": feature,
            "model_summaries": model_summaries,
            "both_models_metric_rule_passed": both_models_passed,
            "krippendorff_alpha": alpha_value,
            "krippendorff_alpha_interval": [alpha_lower, alpha_upper],
            "alpha_rule_passed": bool(np.isfinite(alpha_lower) and alpha_lower > 0),
            "minimum_class_coverage_passed": coverage_passed,
            "structural_rules_passed": structural_passed,
            "metric_undefined_degenerate_gold": degenerate_gold,
            "weakest_metric_value": min(
                cast(float, summary["metric_value"]) for summary in model_summaries
            ),
            "widest_metric_interval_width": max(
                cast(float, summary["metric_interval_upper"])
                - cast(float, summary["metric_interval_lower"])
                for summary in model_summaries
            ),
            "outcome_columns_loaded": [],
        }
        dossiers[feature] = dossier
    return pd.DataFrame(summaries), dossiers


def run_validate_features(
    workspace: AnalysisWorkspace,
    decisions_dir: Path,
) -> StepExecutionResult:
    """Persist feature-validation evidence and authorize only qualifying human gold."""
    gold_labels = pd.read_parquet(
        workspace.store.path_for("tables/gold_labels.parquet")
    ).drop(columns="artifact_schema_version")
    if set(gold_labels["annotator_kind"]) != {AnnotatorKind.HUMAN}:
        raise GateFailureError("Only qualifying human gold can authorize validation")
    initial_completed = pd.to_datetime(gold_labels["initial_completed_at"], utc=True)
    recode_completed = pd.to_datetime(gold_labels["recode_completed_at"], utc=True)
    if (recode_completed.min() - initial_completed.max()).days < 14:
        raise GateFailureError(
            "Delayed human re-code does not satisfy the 14-day minimum"
        )
    normalized = pd.read_parquet(
        workspace.store.path_for("tables/annotations_normalized.parquet")
    ).drop(columns="artifact_schema_version")
    validation, dossiers = build_feature_validation(gold_labels, normalized)
    degenerate_features = sorted(
        feature
        for feature, dossier in dossiers.items()
        if bool(dossier["metric_undefined_degenerate_gold"])
    )
    if degenerate_features:
        raise GateFailureError(
            "METRIC_UNDEFINED_DEGENERATE_GOLD requires supplementation: "
            f"{degenerate_features}"
        )
    decisions = _load_decisions(decisions_dir, dossiers)
    dsl_evidence_path = decisions_dir / "dsl_evidence.json"
    if dsl_evidence_path.is_file():
        try:
            dsl_evidence = DslGateEvidence.model_validate_json(
                dsl_evidence_path.read_text(encoding="utf-8")
            )
        except ValueError as error:
            raise MalformedInputError("Invalid DSL evidence") from error
    else:
        dsl_evidence = incomplete_dsl_evidence()
    dsl_gate = evaluate_dsl_gate(dsl_evidence)
    validated = _validated_feature_rows(decisions, dossiers, dsl_gate)
    unresolved = {
        "schema_version": 1,
        "items": [
            {
                "code": "INDEPENDENT_HUMAN_EXPERT_VERIFICATION_FOLLOWUP",
                "blocking": False,
                "reason": "The initial single human coder is not an independent expert panel.",
            }
        ],
    }
    unlock = {
        "schema_version": 1,
        "authorized": True,
        "qualifying_initial_human_gold": True,
        "qualifying_delayed_human_recode": bool(
            gold_labels["human_recode_value_json"].notna().any()
        ),
        "minimum_delay_days": 14,
        "degenerate_gold_blocking": False,
        "required_feature_decisions_complete": True,
        "dsl_status": dsl_gate["status"],
        "provisional_llm_can_unlock": False,
    }
    if not unlock["qualifying_delayed_human_recode"]:
        raise GateFailureError("Qualifying delayed human re-code is incomplete")

    artifacts: list[ArtifactMetadata] = []
    for feature, dossier in dossiers.items():
        artifacts.append(
            workspace.store.write_json(
                f"statistics/feature_validation/{feature}.json",
                dossier,
                created_by=WorkflowCommand.VALIDATE_FEATURES,
            )
        )
    artifacts.extend(
        [
            workspace.store.write_parquet(
                FEATURE_VALIDATION_TABLE,
                versioned_frame(validation),
                created_by=WorkflowCommand.VALIDATE_FEATURES,
            ),
            workspace.store.write_parquet(
                VALIDATED_FEATURES_TABLE,
                versioned_frame(validated),
                created_by=WorkflowCommand.VALIDATE_FEATURES,
            ),
            workspace.store.write_json(
                DSL_GATE_ARTIFACT,
                dsl_gate,
                created_by=WorkflowCommand.VALIDATE_FEATURES,
            ),
            workspace.store.write_json(
                UNRESOLVED_VALIDATION_ARTIFACT,
                unresolved,
                created_by=WorkflowCommand.VALIDATE_FEATURES,
            ),
            workspace.store.write_json(
                VALIDATION_UNLOCK_ARTIFACT,
                unlock,
                created_by=WorkflowCommand.VALIDATE_FEATURES,
            ),
            workspace.store.write_bytes(
                FEATURE_VALIDATION_REPORT,
                _render_validation_report(validated, dsl_gate, unresolved).encode(
                    "utf-8"
                ),
                created_by=WorkflowCommand.VALIDATE_FEATURES,
            ),
        ]
    )
    return StepExecutionResult(
        artifacts=tuple(artifacts),
        authorize_outcome_modeling=True,
    )


def emit_next_feature_dossier(
    workspace: AnalysisWorkspace,
    decisions_dir: Path,
    *,
    bootstrap_replicates: int = VALIDATION_BOOTSTRAP_REPLICATES,
) -> Path:
    """Expose only the next seeded-order dossier after prior decisions exist."""
    gold_labels = pd.read_parquet(
        workspace.store.path_for("tables/gold_labels.parquet")
    ).drop(columns="artifact_schema_version")
    normalized = pd.read_parquet(
        workspace.store.path_for("tables/annotations_normalized.parquet")
    ).drop(columns="artifact_schema_version")
    _, dossiers = build_feature_validation(
        gold_labels,
        normalized,
        bootstrap_replicates=bootstrap_replicates,
    )
    decisions_dir.mkdir(parents=True, exist_ok=True)
    for order, feature in enumerate(_feature_decision_order(), start=1):
        decision_path = decisions_dir / f"{order:02d}-{feature}.json"
        dossier_path = decisions_dir / f"{order:02d}-{feature}-dossier.json"
        dossier_bytes = canonical_json_bytes(dossiers[feature])
        dossier_sha = sha256(dossier_bytes).hexdigest()
        if decision_path.is_file():
            try:
                decision = FeatureDecision.model_validate_json(
                    decision_path.read_text(encoding="utf-8")
                )
            except ValueError as error:
                raise MalformedInputError(
                    f"Invalid feature decision: {decision_path.name}"
                ) from error
            if (
                decision.order != order
                or decision.feature != feature
                or decision.dossier_sha256 != dossier_sha
            ):
                raise GateFailureError(
                    f"Feature decision provenance mismatch: {decision_path.name}"
                )
            continue
        if dossier_path.exists():
            if dossier_path.read_bytes() != dossier_bytes:
                raise GateFailureError(
                    f"Pending dossier conflicts with current evidence: {dossier_path.name}"
                )
        else:
            atomic_write_bytes(dossier_path, dossier_bytes)
        return dossier_path
    raise GateFailureError("Every required feature decision is already present")


def _metric_summary(feature: str, frame: pd.DataFrame) -> MetricSummary:
    kind = FEATURE_KINDS[feature]
    valid = frame.loc[frame["human_value"].notna() & frame["model_value"].notna()]
    if valid.empty:
        return MetricSummary(float("nan"), 0.0, {})
    probabilities = valid["inclusion_probability"].astype(float).tolist()
    counts = {
        str(label): int(count)
        for label, count in valid["human_value"].value_counts().sort_index().items()
    }
    if kind == "numeric":
        value, _ = mae_skill(
            [float(value) for value in valid["human_value"]],
            [float(value) for value in valid["model_value"]],
            probabilities,
        )
        return MetricSummary(value, 0.0, counts)
    levels = FEATURE_SUBSTANTIVE_LEVELS[feature]
    substantive = valid.loc[valid["human_value"].isin(levels)]
    matrix = ht_confusion_totals(
        substantive["human_value"].tolist(),
        substantive["model_value"].tolist(),
        substantive["inclusion_probability"].astype(float).tolist(),
        levels,
    )
    if kind == "binary":
        return MetricSummary(balanced_accuracy(matrix), 0.5, counts)
    majority_index = int(np.argmax(matrix.sum(axis=1)))
    majority_matrix = np.zeros_like(matrix)
    majority_matrix[:, majority_index] = matrix.sum(axis=1)
    return MetricSummary(
        macro_f1(matrix),
        macro_f1(majority_matrix),
        counts,
    )


def _prediction_rows(
    gold_labels: pd.DataFrame,
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "item_id",
        "feature",
        "human_gold_value_json",
        "stratum_id",
        "inclusion_probability",
    }
    if missing := sorted(required - set(gold_labels.columns)):
        raise MalformedInputError(f"Gold labels are missing columns: {missing}")
    model_ids = sorted(normalized["model_id"].astype(str).unique())
    if len(model_ids) != 2:
        raise GateFailureError(
            "Validation requires exactly two frozen annotator models"
        )
    rows: list[dict[str, object]] = []
    normalized_by_key = {
        (str(row["item_id"]), str(row["model_id"])): row
        for row in normalized.to_dict(orient="records")
    }
    for label in gold_labels.to_dict(orient="records"):
        feature = str(label["feature"])
        if feature not in GOLD_VALIDATION_FEATURES:
            continue
        human_value = json.loads(str(label["human_gold_value_json"]))
        for model_id in model_ids:
            annotation = normalized_by_key.get((str(label["item_id"]), model_id))
            if annotation is None:
                raise GateFailureError("Gold item is missing a model annotation")
            rows.append(
                {
                    "item_id": str(label["item_id"]),
                    "feature": feature,
                    "model_id": model_id,
                    "model_value": _annotation_value(annotation, feature),
                    "human_value": human_value,
                    "stratum_id": str(label["stratum_id"]),
                    "inclusion_probability": float(label["inclusion_probability"]),
                }
            )
    return pd.DataFrame(rows)


def _annotation_value(annotation: Mapping[str, object], feature: str) -> object:
    value = annotation.get(feature)
    if value is not None and not bool(pd.isna(value)):
        return value
    if feature == "referring_form_type":
        return annotation.get("referring_form_missingness")
    if feature == "answer_locality":
        return annotation.get("answer_locality_missingness")
    return None


def _feature_alpha(
    feature: str,
    rows: pd.DataFrame,
    bootstrap_replicates: int,
) -> tuple[float, float, float]:
    pivot = rows.pivot(
        index=["item_id", "stratum_id"], columns="model_id", values="model_value"
    )
    ratings = pivot.values.tolist()
    kind = FEATURE_KINDS[feature]
    level = "interval" if kind == "numeric" else "nominal"
    ordinal_levels = (
        COGNITIVE_PROCESS_LEVELS if feature == "cognitive_process_level" else None
    )
    point = krippendorff_alpha(ratings, level=level, ordinal_levels=ordinal_levels)
    alpha_frame = pivot.reset_index()
    rng = _validation_rng(feature, "both_models", "alpha")

    def metric(sample: pd.DataFrame) -> float:
        return krippendorff_alpha(
            sample.drop(columns=["item_id", "stratum_id"]).values.tolist(),
            level=level,
            ordinal_levels=ordinal_levels,
        )

    lower, upper = stratified_bootstrap_interval(
        alpha_frame,
        metric,
        rng=rng,
        replicates=bootstrap_replicates,
    )
    return point, lower, upper


def _alpha_distance(
    left: object,
    right: object,
    level: str,
    ordinal_levels: Sequence[object] | None,
) -> float:
    if level == "nominal":
        return float(left != right)
    if level == "interval":
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise MalformedInputError("Interval alpha requires numeric ratings")
        return (float(left) - float(right)) ** 2
    if level == "ordinal":
        if ordinal_levels is None:
            raise MalformedInputError("Ordinal alpha requires explicit levels")
        return float((ordinal_levels.index(left) - ordinal_levels.index(right)) ** 2)
    raise MalformedInputError(f"Unsupported alpha measurement level: {level}")


def _coverage_passed(feature: str, rows: pd.DataFrame) -> bool:
    if feature == "qdmr_step_count":
        return int(rows["human_value"].notna().sum() / 2) >= 12
    levels = FEATURE_SUBSTANTIVE_LEVELS[feature]
    unique_items = rows.drop_duplicates("item_id")
    counts = unique_items["human_value"].value_counts()
    return all(int(counts.get(level, 0)) >= 12 for level in levels)


def _metric_name(feature: str) -> str:
    return {
        "numeric": "ht_weighted_mae_skill",
        "binary": "ht_weighted_balanced_accuracy",
        "nominal": "ht_weighted_macro_f1",
    }[FEATURE_KINDS[feature]]


def _validation_rng(
    feature: str,
    model_id: str,
    purpose: str,
) -> np.random.Generator:
    parent = derive_stream_seed("gold_sampling")
    digest = sha256(f"{parent}:{feature}:{model_id}:{purpose}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="big"))


def _load_decisions(
    decisions_dir: Path,
    dossiers: Mapping[str, Mapping[str, object]],
) -> tuple[FeatureDecision, ...]:
    expected_order = _feature_decision_order()
    decisions: list[FeatureDecision] = []
    for order, feature in enumerate(expected_order, start=1):
        path = decisions_dir / f"{order:02d}-{feature}.json"
        if not path.is_file():
            raise GateFailureError(f"Missing feature decision: {path.name}")
        try:
            decision = FeatureDecision.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError as error:
            raise MalformedInputError(
                f"Invalid feature decision: {path.name}"
            ) from error
        expected_sha = sha256(canonical_json_bytes(dossiers[feature])).hexdigest()
        if (
            decision.order != order
            or decision.feature != feature
            or decision.dossier_sha256 != expected_sha
        ):
            raise GateFailureError(f"Feature decision provenance mismatch: {path.name}")
        ceiling = _status_ceiling(dossiers[feature], decision.material_failure_mode)
        if STATUS_RANK[decision.status] > STATUS_RANK[ceiling]:
            raise GateFailureError(
                f"Feature decision exceeds frozen status ceiling: {feature}"
            )
        decisions.append(decision)
    return tuple(decisions)


def _feature_decision_order() -> tuple[str, ...]:
    rng = np.random.default_rng(derive_stream_seed("annotation_shuffle"))
    values = np.array(GOLD_VALIDATION_FEATURES, dtype=object)
    return tuple(str(value) for value in values[rng.permutation(len(values))])


def _status_ceiling(
    dossier: Mapping[str, object],
    material_failure_mode: bool,
) -> FeatureValidationStatus:
    if not bool(dossier["structural_rules_passed"]):
        return FeatureValidationStatus.NOT_VALIDATED
    if material_failure_mode or not bool(dossier["minimum_class_coverage_passed"]):
        return FeatureValidationStatus.VALIDATED_WITH_LIMITATIONS
    return FeatureValidationStatus.VALIDATED


def _validated_feature_rows(
    decisions: Sequence[FeatureDecision],
    dossiers: Mapping[str, Mapping[str, object]],
    dsl_gate: Mapping[str, object],
) -> pd.DataFrame:
    family_modes = dsl_gate["family_inference_mode"]
    if not isinstance(family_modes, dict):
        raise MalformedInputError("DSL family inference mapping is invalid")
    family_by_feature = {
        "qdmr_step_count": "F1",
        "hop_structure": "F1",
        "exhaustivity_requirement": "F2",
        "negative_conclusiveness": "F2",
        "referring_form_type": "F5",
        "answer_locality": "F8",
        "recall_orientation": "F9",
    }
    return pd.DataFrame(
        [
            {
                "feature": decision.feature,
                "validation_status": decision.status,
                "decision_order": decision.order,
                "dossier_sha256": decision.dossier_sha256,
                "material_failure_mode": decision.material_failure_mode,
                "justification": decision.justification,
                "decision_git_commit": decision.git_commit,
                "structural_rules_passed": bool(
                    dossiers[decision.feature]["structural_rules_passed"]
                ),
                "inference_mode": (
                    "excluded_confirmatory"
                    if decision.status is FeatureValidationStatus.NOT_VALIDATED
                    else family_modes[family_by_feature[decision.feature]]
                ),
            }
            for decision in decisions
        ]
    )


def _render_validation_report(
    validated: pd.DataFrame,
    dsl_gate: Mapping[str, object],
    unresolved: Mapping[str, object],
) -> str:
    lines = [
        "# Partial report 04 — Human-gold feature validation",
        "",
        f"DSL gate: **{dsl_gate['status']}**.",
        "",
        "## Feature decisions",
        "",
    ]
    lines.extend(
        f"- `{row['feature']}`: `{row['validation_status']}`; inference `{row['inference_mode']}`."
        for row in validated.to_dict(orient="records")
    )
    unresolved_items = unresolved.get("items")
    if not isinstance(unresolved_items, list):
        raise MalformedInputError("Unresolved validation items are invalid")
    lines.extend(
        [
            "",
            "## Residual validation items",
            "",
            *[
                f"- `{item['code']}` (blocking={str(item['blocking']).lower()}): {item['reason']}"
                for item in unresolved_items
                if isinstance(item, dict)
            ],
            "",
            "No outcome table or downstream coefficient estimate was loaded.",
        ]
    )
    return "\n".join(lines) + "\n"
