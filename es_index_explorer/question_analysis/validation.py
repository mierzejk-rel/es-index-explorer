"""Outcome-blind weighted feature validation and unlock evidence."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from es_index_explorer.question_analysis.contracts import (
    COGNITIVE_PROCESS_LEVELS,
    GOLD_EXPLORATORY_EVIDENCE_FEATURES,
    GOLD_VALIDATION_FEATURES,
    QDMR_APPLICABILITY_LEVELS,
    QDMR_OPERATOR_INVENTORY,
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
    sha256_file,
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
EXPLORATORY_EVIDENCE_TABLE = "tables/exploratory_p4_evidence.parquet"
EXPLORATORY_EVIDENCE_REPORT = "partial_reports/05-exploratory-p4-evidence.md"

FEATURE_KINDS: Mapping[str, str] = {
    "qdmr_step_count": "numeric",
    "hop_structure": "nominal",
    "exhaustivity_requirement": "nominal",
    "negative_conclusiveness": "binary",
    "referring_form_type": "nominal",
    "answer_locality": "binary",
    "recall_orientation": "binary",
    "presupposition_load": "binary",
    "cognitive_process_level": "ordinal",
    "demand_type": "nominal",
    "specificity": "binary",
    "qdmr_applicability": "nominal",
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
    "presupposition_load": (False, True),
    "cognitive_process_level": COGNITIVE_PROCESS_LEVELS,
    "demand_type": (
        "verbatim_citation",
        "entity_identification",
        "relational_claim",
        "temporal_ordering",
        "quantification",
        "evaluative_synthesis",
    ),
    "specificity": ("specific", "broad"),
    "qdmr_applicability": QDMR_APPLICABILITY_LEVELS,
}
FEATURE_ALL_LEVELS: Mapping[str, tuple[object, ...]] = {
    **FEATURE_SUBSTANTIVE_LEVELS,
    "referring_form_type": (
        "full_name_form",
        "alias_or_handle",
        "email_address",
        "no_focal_referent",
    ),
    "answer_locality": (
        "single_passage",
        "cross_document_aggregation",
        "not_classifiable_binary_locality",
    ),
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
    material_failure_pattern_ids: tuple[str, ...] = ()
    class_interval_material_limitation: bool = False
    material_class_interval_ids: tuple[str, ...] = ()
    class_interval_justification: str | None = None
    justification: str = Field(min_length=1)
    git_commit: str = Field(min_length=7)

    @model_validator(mode="after")
    def validate_limitation_evidence(self) -> "FeatureDecision":
        """Require explicit references for qualitative limitations."""
        if self.material_failure_mode and not self.material_failure_pattern_ids:
            raise ValueError("Material failure decisions require failure-pattern IDs")
        if self.class_interval_material_limitation and (
            not self.material_class_interval_ids
            or not self.class_interval_justification
        ):
            raise ValueError(
                "Material class-interval limitations require interval IDs and justification"
            )
        return self


class IntervalEvidence(BaseModel):
    """Store a deterministic percentile interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float
    upper: float


class ClassMetricEvidence(BaseModel):
    """Store one class's weighted precision and recall evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_json: str
    raw_gold_count: int = Field(ge=0)
    ht_gold_total: float = Field(ge=0)
    precision: float
    precision_interval: IntervalEvidence
    recall: float
    recall_interval: IntervalEvidence


class ConfusionEvidence(BaseModel):
    """Store raw and HT confusion cells with explicit label axes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_levels_json: tuple[str, ...]
    forecast_levels_json: tuple[str, ...]
    raw_cells: tuple[tuple[int, ...], ...]
    ht_cells: tuple[tuple[float, ...], ...]


class FailurePatternEvidence(BaseModel):
    """Store a deterministic observed validation failure pattern."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern_id: str
    actual_json: str
    forecast_json: str
    raw_count: int = Field(ge=1)
    ht_total: float = Field(gt=0)
    example_item_ids: tuple[str, ...]


class ModelValidationEvidence(BaseModel):
    """Store complete model-versus-human evidence for one feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    metric: str
    metric_value: float
    metric_interval: IntervalEvidence
    matched_naive_baseline: float
    metric_rule_passed: bool
    confusion: ConfusionEvidence | None
    class_metrics: tuple[ClassMetricEvidence, ...]
    numeric_error_summary: dict[str, float] | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    scored_item_count: int = Field(ge=0)
    gold_missingness_count: int = Field(ge=0)
    forecast_missingness_error_count: int = Field(ge=0)
    failure_patterns: tuple[FailurePatternEvidence, ...]


class HumanTestRetestEvidence(BaseModel):
    """Store scale-matched initial-versus-recode human evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_count: int = Field(ge=1)
    metric: str
    metric_value: float
    metric_interval: IntervalEvidence
    metric_defined: bool
    matched_naive_baseline: float
    krippendorff_alpha: float
    krippendorff_alpha_interval: IntervalEvidence
    alpha_defined: bool
    measurement_level: Literal["nominal", "ordinal", "interval"]
    weight: Literal["inverse_joint_gold_recode_inclusion_probability"]


class FeatureValidationDossier(BaseModel):
    """Store the complete immutable evidence dossier for one feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    feature: str
    feature_kind: str
    corpus_prevalence_by_model: dict[str, dict[str, int]]
    model_summaries: tuple[ModelValidationEvidence, ...]
    both_models_metric_rule_passed: bool
    krippendorff_alpha: float
    krippendorff_alpha_interval: IntervalEvidence
    alpha_measurement_level: Literal["nominal", "interval"]
    alpha_rule_passed: bool
    human_test_retest: HumanTestRetestEvidence
    minimum_class_coverage_passed: bool
    structural_rules_passed: bool
    metric_undefined_degenerate_gold: bool
    weakest_model_id: str
    weakest_metric_value: float
    widest_model_interval_id: str
    widest_metric_interval_width: float
    widest_class_interval_id: str | None
    widest_class_interval_width: float | None
    failure_patterns: tuple[FailurePatternEvidence, ...]
    gold_sample_sha256: str
    gold_labels_sha256: str
    annotation_manifest_sha256: str
    codebook_sha256: str
    outcome_columns_loaded: tuple[str, ...] = ()


class DescriptiveMetricEvidence(BaseModel):
    """Store one non-gating descriptive metric and interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    value: float | None
    interval: IntervalEvidence | None
    baseline: float | None = None
    defined: bool


class ExploratoryModelEvidence(BaseModel):
    """Store one model's descriptive human-gold evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    metrics: tuple[DescriptiveMetricEvidence, ...]
    confusion: ConfusionEvidence | None = None
    class_metrics: tuple[ClassMetricEvidence, ...] = ()
    sensitivity: float | None = None
    specificity: float | None = None
    scored_item_count: int = Field(ge=0)
    example_item_ids: tuple[str, ...]


class ExploratoryFeatureDossier(BaseModel):
    """Store non-gating evidence for one exploratory P4 field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    feature: str
    evidence_kind: str
    non_gating: Literal[True] = True
    availability: Literal["AVAILABLE", "UNAVAILABLE"]
    corpus_prevalence_by_model: dict[str, dict[str, int]]
    operator_evidence_by_model: dict[str, dict[str, dict[str, int]]] | None = None
    model_summaries: tuple[ExploratoryModelEvidence, ...]
    inter_model_metrics: tuple[DescriptiveMetricEvidence, ...]
    human_test_retest_metrics: tuple[DescriptiveMetricEvidence, ...]
    gold_sample_sha256: str
    gold_labels_sha256: str
    annotation_manifest_sha256: str
    codebook_sha256: str
    outcome_columns_loaded: tuple[str, ...] = ()


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
    forecast_levels: Sequence[object] | None = None,
) -> np.ndarray:
    """Compute Horvitz-Thompson confusion-cell population totals."""
    if not (
        len(gold) == len(predicted) == len(inclusion_probabilities) and len(gold) > 0
    ):
        raise MalformedInputError("HT confusion inputs must be non-empty and aligned")
    gold_index = {level: index for index, level in enumerate(levels)}
    forecast_inventory = tuple(forecast_levels or levels)
    forecast_index = {level: index for index, level in enumerate(forecast_inventory)}
    matrix = np.zeros((len(levels), len(forecast_inventory)), dtype=float)
    for actual, forecast, probability in zip(
        gold, predicted, inclusion_probabilities, strict=True
    ):
        if actual not in gold_index or forecast not in forecast_index:
            raise MalformedInputError(
                "Confusion values must belong to their explicit label inventories"
            )
        if not np.isfinite(probability) or probability <= 0 or probability > 1:
            raise MalformedInputError("Inclusion probabilities must lie in (0, 1]")
        matrix[gold_index[actual], forecast_index[forecast]] += 1.0 / probability
    return matrix


def balanced_accuracy(matrix: np.ndarray) -> float:
    """Compute binary balanced accuracy, retaining forecast-only error columns."""
    if matrix.ndim != 2 or matrix.shape[0] != 2 or matrix.shape[1] < 2:
        raise MalformedInputError(
            "Balanced accuracy requires two gold rows and at least two forecast columns"
        )
    row_totals = matrix.sum(axis=1)
    if np.any(row_totals == 0):
        return float("nan")
    recalls = np.diag(matrix) / row_totals
    return float(recalls.mean())


def macro_f1(matrix: np.ndarray) -> float:
    """Compute macro-F1 while retaining forecast-only error columns."""
    if matrix.ndim != 2 or matrix.shape[1] < matrix.shape[0]:
        raise MalformedInputError(
            "Macro-F1 requires every gold class to have a forecast column"
        )
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
    unit_weights: Sequence[float] | None = None,
) -> float:
    """Compute survey-weighted Krippendorff alpha from coincidences."""
    if unit_weights is not None and len(unit_weights) != len(ratings):
        raise MalformedInputError("Alpha weights must align with rating units")
    weights = list(unit_weights or [1.0] * len(ratings))
    if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
        raise MalformedInputError("Alpha weights must be finite and positive")
    eligible: list[tuple[list[object], float]] = []
    for unit, unit_weight in zip(ratings, weights, strict=True):
        values = [
            value for value in unit if value is not None and not bool(pd.isna(value))
        ]
        if len(values) >= 2:
            eligible.append((values, unit_weight))
    if not eligible:
        return float("nan")
    observed_values = [value for values, _ in eligible for value in values]
    categories = _alpha_categories(observed_values, level, ordinal_levels)
    category_index = {category: index for index, category in enumerate(categories)}
    coincidence = np.zeros((len(categories), len(categories)), dtype=float)
    for values, unit_weight in eligible:
        counts = {
            category: values.count(category)
            for category in categories
            if category in values
        }
        denominator = len(values) - 1
        for left, left_count in counts.items():
            for right, right_count in counts.items():
                pair_count = (
                    left_count * (left_count - 1)
                    if left == right
                    else left_count * right_count
                )
                coincidence[category_index[left], category_index[right]] += (
                    unit_weight * pair_count / denominator
                )
    marginals = coincidence.sum(axis=1)
    total = float(marginals.sum())
    if total <= 1:
        return float("nan")
    distance = _alpha_distance_matrix(
        categories,
        level=level,
        marginals=marginals,
    )
    observed_disagreement = float((coincidence * distance).sum() / total)
    expected = np.outer(marginals, marginals) / (total - 1)
    diagonal = marginals * (marginals - 1) / (total - 1)
    np.fill_diagonal(expected, diagonal)
    expected_disagreement = float((expected * distance).sum() / total)
    if expected_disagreement == 0:
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def _alpha_categories(
    values: Sequence[object],
    level: str,
    ordinal_levels: Sequence[object] | None,
) -> tuple[object, ...]:
    if level == "ordinal":
        if ordinal_levels is None:
            raise MalformedInputError("Ordinal alpha requires explicit levels")
        if unknown := [value for value in values if value not in ordinal_levels]:
            raise MalformedInputError(f"Unknown ordinal alpha values: {unknown}")
        return tuple(ordinal_levels)
    if level == "interval":
        return tuple(sorted({_alpha_number(value) for value in values}))
    if level == "nominal":
        return tuple(sorted(set(values), key=_json_label))
    raise MalformedInputError(f"Unsupported alpha measurement level: {level}")


def _alpha_distance_matrix(
    categories: Sequence[object],
    *,
    level: str,
    marginals: np.ndarray,
) -> np.ndarray:
    size = len(categories)
    distances = np.zeros((size, size), dtype=float)
    for left in range(size):
        for right in range(left + 1, size):
            if level == "nominal":
                value = 1.0
            elif level == "interval":
                value = (
                    _alpha_number(categories[left]) - _alpha_number(categories[right])
                ) ** 2
            elif level == "ordinal":
                lower, upper = sorted((left, right))
                cumulative = float(marginals[lower : upper + 1].sum())
                cumulative -= float((marginals[lower] + marginals[upper]) / 2)
                value = cumulative**2
            else:
                raise MalformedInputError(
                    f"Unsupported alpha measurement level: {level}"
                )
            distances[left, right] = value
            distances[right, left] = value
    return distances


def _alpha_number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise MalformedInputError("Interval alpha requires numeric ratings")
    return float(value)


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
    evidence_hashes: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Build per-model weighted validation summaries and feature dossiers."""
    prediction_rows = _prediction_rows(gold_labels, normalized)
    summaries: list[dict[str, object]] = []
    dossiers: dict[str, dict[str, object]] = {}
    hashes = {
        "gold_sample_sha256": "in-memory",
        "gold_labels_sha256": "in-memory",
        "annotation_manifest_sha256": "in-memory",
        "codebook_sha256": "in-memory",
        **(evidence_hashes or {}),
    }
    for feature in GOLD_VALIDATION_FEATURES:
        feature_rows = prediction_rows.loc[prediction_rows["feature"].eq(feature)]
        if feature_rows.empty:
            raise GateFailureError(f"Human gold contains no rows for {feature}")
        model_evidence: list[ModelValidationEvidence] = []
        for model_id, model_rows in feature_rows.groupby("model_id", sort=True):
            evidence = _model_validation_evidence(
                feature,
                str(model_id),
                model_rows,
                bootstrap_replicates,
            )
            model_evidence.append(evidence)
            row: dict[str, object] = {
                "feature": feature,
                "model_id": str(model_id),
                "metric": evidence.metric,
                "metric_value": evidence.metric_value,
                "metric_interval_lower": evidence.metric_interval.lower,
                "metric_interval_upper": evidence.metric_interval.upper,
                "matched_naive_baseline": evidence.matched_naive_baseline,
                "class_counts_json": json.dumps(
                    {
                        item.label_json: item.raw_gold_count
                        for item in evidence.class_metrics
                    },
                    sort_keys=True,
                ),
                "metric_rule_passed": evidence.metric_rule_passed,
                "scored_item_count": evidence.scored_item_count,
                "gold_missingness_count": evidence.gold_missingness_count,
                "forecast_missingness_error_count": (
                    evidence.forecast_missingness_error_count
                ),
            }
            summaries.append(row)

        alpha_value, alpha_lower, alpha_upper = _feature_alpha(
            feature,
            feature_rows,
            bootstrap_replicates,
        )
        test_retest = _human_test_retest(
            feature,
            gold_labels,
            bootstrap_replicates,
        )
        coverage_passed = _coverage_passed(feature, feature_rows)
        both_models_passed = len(model_evidence) == 2 and all(
            evidence.metric_rule_passed for evidence in model_evidence
        )
        structural_passed = (
            both_models_passed and np.isfinite(alpha_lower) and alpha_lower > 0
        )
        degenerate_gold = any(
            not np.isfinite(evidence.metric_value)
            or not np.isfinite(evidence.metric_interval.lower)
            or not np.isfinite(evidence.metric_interval.upper)
            for evidence in model_evidence
        ) or not np.isfinite(alpha_value)
        weakest = min(model_evidence, key=lambda evidence: evidence.metric_value)
        widest = max(
            model_evidence,
            key=lambda evidence: (
                evidence.metric_interval.upper - evidence.metric_interval.lower
            ),
        )
        class_intervals = [
            (
                f"{evidence.model_id}:{class_metric.label_json}:{metric_name}",
                interval.upper - interval.lower,
            )
            for evidence in model_evidence
            for class_metric in evidence.class_metrics
            for metric_name, interval in (
                ("precision", class_metric.precision_interval),
                ("recall", class_metric.recall_interval),
            )
            if np.isfinite(interval.lower) and np.isfinite(interval.upper)
        ]
        widest_class = max(class_intervals, key=lambda item: item[1], default=None)
        failure_patterns = tuple(
            pattern
            for evidence in model_evidence
            for pattern in evidence.failure_patterns
        )
        dossier_model = FeatureValidationDossier(
            feature=feature,
            feature_kind=FEATURE_KINDS[feature],
            corpus_prevalence_by_model=_corpus_prevalence(feature, normalized),
            model_summaries=tuple(model_evidence),
            both_models_metric_rule_passed=both_models_passed,
            krippendorff_alpha=alpha_value,
            krippendorff_alpha_interval=IntervalEvidence(
                lower=alpha_lower, upper=alpha_upper
            ),
            alpha_measurement_level=(
                "interval" if FEATURE_KINDS[feature] == "numeric" else "nominal"
            ),
            alpha_rule_passed=bool(np.isfinite(alpha_lower) and alpha_lower > 0),
            human_test_retest=test_retest,
            minimum_class_coverage_passed=coverage_passed,
            structural_rules_passed=structural_passed,
            metric_undefined_degenerate_gold=degenerate_gold,
            weakest_model_id=weakest.model_id,
            weakest_metric_value=weakest.metric_value,
            widest_model_interval_id=widest.model_id,
            widest_metric_interval_width=(
                widest.metric_interval.upper - widest.metric_interval.lower
            ),
            widest_class_interval_id=widest_class[0] if widest_class else None,
            widest_class_interval_width=widest_class[1] if widest_class else None,
            failure_patterns=failure_patterns,
            gold_sample_sha256=hashes["gold_sample_sha256"],
            gold_labels_sha256=hashes["gold_labels_sha256"],
            annotation_manifest_sha256=hashes["annotation_manifest_sha256"],
            codebook_sha256=hashes["codebook_sha256"],
        )
        dossiers[feature] = dossier_model.model_dump(mode="json")
    return pd.DataFrame(summaries), dossiers


def build_exploratory_p4_evidence(
    gold_labels: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    bootstrap_replicates: int = VALIDATION_BOOTSTRAP_REPLICATES,
    evidence_hashes: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Build comprehensive non-gating evidence for exploratory P4 fields."""
    prediction_rows = _prediction_rows_for_features(
        gold_labels,
        normalized,
        frozenset(GOLD_EXPLORATORY_EVIDENCE_FEATURES),
    )
    hashes = {
        "gold_sample_sha256": "in-memory",
        "gold_labels_sha256": "in-memory",
        "annotation_manifest_sha256": "in-memory",
        "codebook_sha256": "in-memory",
        **(evidence_hashes or {}),
    }
    table_rows: list[dict[str, object]] = []
    dossiers: dict[str, dict[str, object]] = {}
    for feature in GOLD_EXPLORATORY_EVIDENCE_FEATURES:
        rows = prediction_rows.loc[prediction_rows["feature"].eq(feature)]
        if rows.empty:
            dossier = ExploratoryFeatureDossier(
                feature=feature,
                evidence_kind=_exploratory_kind(feature),
                availability="UNAVAILABLE",
                corpus_prevalence_by_model=_corpus_prevalence(feature, normalized),
                operator_evidence_by_model=None,
                model_summaries=(),
                inter_model_metrics=(),
                human_test_retest_metrics=(),
                gold_sample_sha256=hashes["gold_sample_sha256"],
                gold_labels_sha256=hashes["gold_labels_sha256"],
                annotation_manifest_sha256=hashes["annotation_manifest_sha256"],
                codebook_sha256=hashes["codebook_sha256"],
            )
            dossiers[feature] = dossier.model_dump(mode="json")
            continue
        model_summaries = tuple(
            _exploratory_model_evidence(
                feature,
                str(model_id),
                model_rows,
                bootstrap_replicates,
            )
            for model_id, model_rows in rows.groupby("model_id", sort=True)
        )
        inter_model = _exploratory_inter_model_metrics(
            feature, rows, bootstrap_replicates
        )
        test_retest = _exploratory_human_test_retest_metrics(
            feature, gold_labels, bootstrap_replicates
        )
        dossier = ExploratoryFeatureDossier(
            feature=feature,
            evidence_kind=_exploratory_kind(feature),
            availability="AVAILABLE",
            corpus_prevalence_by_model=_corpus_prevalence(feature, normalized),
            operator_evidence_by_model=(
                _operator_evidence(rows) if feature == "qdmr_operator_set" else None
            ),
            model_summaries=model_summaries,
            inter_model_metrics=inter_model,
            human_test_retest_metrics=test_retest,
            gold_sample_sha256=hashes["gold_sample_sha256"],
            gold_labels_sha256=hashes["gold_labels_sha256"],
            annotation_manifest_sha256=hashes["annotation_manifest_sha256"],
            codebook_sha256=hashes["codebook_sha256"],
        )
        dossiers[feature] = dossier.model_dump(mode="json")
        for summary in model_summaries:
            for metric in summary.metrics:
                table_rows.append(
                    {
                        "feature": feature,
                        "model_id": summary.model_id,
                        "metric": metric.metric,
                        "metric_value": metric.value,
                        "metric_interval_lower": (
                            metric.interval.lower if metric.interval else None
                        ),
                        "metric_interval_upper": (
                            metric.interval.upper if metric.interval else None
                        ),
                        "defined": metric.defined,
                        "non_gating": True,
                    }
                )
    return pd.DataFrame(table_rows), dossiers


def run_validate_features(
    workspace: AnalysisWorkspace,
    decisions_dir: Path,
    *,
    bootstrap_replicates: int = VALIDATION_BOOTSTRAP_REPLICATES,
) -> StepExecutionResult:
    """Persist feature-validation evidence and authorize only qualifying human gold."""
    gold_labels = pd.read_parquet(
        workspace.store.path_for("tables/gold_labels.parquet")
    ).drop(columns="artifact_schema_version")
    if set(gold_labels["annotator_kind"]) != {AnnotatorKind.HUMAN}:
        raise GateFailureError("Only qualifying human gold can authorize validation")
    normalized = pd.read_parquet(
        workspace.store.path_for("tables/annotations_normalized.parquet")
    ).drop(columns="artifact_schema_version")
    validation, dossiers = build_feature_validation(
        gold_labels,
        normalized,
        bootstrap_replicates=bootstrap_replicates,
        evidence_hashes=_validation_evidence_hashes(workspace),
    )
    exploratory_table, exploratory_dossiers = build_exploratory_p4_evidence(
        gold_labels,
        normalized,
        bootstrap_replicates=bootstrap_replicates,
        evidence_hashes=_validation_evidence_hashes(workspace),
    )
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
    for feature, dossier in exploratory_dossiers.items():
        artifacts.append(
            workspace.store.write_json(
                f"statistics/exploratory_p4_evidence/{feature}.json",
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
            workspace.store.write_parquet(
                EXPLORATORY_EVIDENCE_TABLE,
                versioned_frame(exploratory_table),
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
            workspace.store.write_bytes(
                EXPLORATORY_EVIDENCE_REPORT,
                _render_exploratory_report(exploratory_dossiers).encode("utf-8"),
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
        evidence_hashes=_validation_evidence_hashes(workspace),
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


def _exploratory_kind(feature: str) -> str:
    if feature == "qdmr_operator_set":
        return "set"
    if feature == "qdmr_normalized_question":
        return "faithfulness_text"
    return FEATURE_KINDS[feature]


def _exploratory_model_evidence(
    feature: str,
    model_id: str,
    frame: pd.DataFrame,
    bootstrap_replicates: int,
) -> ExploratoryModelEvidence:
    if feature in {"qdmr_operator_set", "qdmr_normalized_question"}:
        metrics = _special_descriptive_metrics(
            feature,
            frame,
            model_id=model_id,
            purpose="human_gold",
            bootstrap_replicates=bootstrap_replicates,
        )
        valid = frame.loc[frame["human_value"].notna() & frame["model_value"].notna()]
        mismatches = valid.loc[
            [
                _special_scores(feature, actual, forecast)[0] < 1.0
                for actual, forecast in zip(
                    valid["human_value"], valid["model_value"], strict=True
                )
            ]
        ]
        return ExploratoryModelEvidence(
            model_id=model_id,
            metrics=metrics,
            scored_item_count=len(valid),
            example_item_ids=tuple(
                sorted(mismatches["item_id"].astype(str).unique())[:5]
            ),
        )
    evidence = _model_validation_evidence(
        feature, model_id, frame, bootstrap_replicates
    )
    metric = DescriptiveMetricEvidence(
        metric=evidence.metric,
        value=evidence.metric_value,
        interval=evidence.metric_interval,
        baseline=evidence.matched_naive_baseline,
        defined=bool(
            np.isfinite(evidence.metric_value)
            and np.isfinite(evidence.metric_interval.lower)
            and np.isfinite(evidence.metric_interval.upper)
        ),
    )
    return ExploratoryModelEvidence(
        model_id=model_id,
        metrics=(metric,),
        confusion=evidence.confusion,
        class_metrics=evidence.class_metrics,
        sensitivity=evidence.sensitivity,
        specificity=evidence.specificity,
        scored_item_count=evidence.scored_item_count,
        example_item_ids=tuple(
            sorted(
                {
                    item_id
                    for pattern in evidence.failure_patterns
                    for item_id in pattern.example_item_ids
                }
            )[:5]
        ),
    )


def _special_descriptive_metrics(
    feature: str,
    frame: pd.DataFrame,
    *,
    model_id: str,
    purpose: str,
    bootstrap_replicates: int,
) -> tuple[DescriptiveMetricEvidence, ...]:
    valid = frame.loc[frame["human_value"].notna() & frame["model_value"].notna()]
    metric_names = (
        ("ht_weighted_mean_jaccard", "ht_weighted_exact_set_match")
        if feature == "qdmr_operator_set"
        else ("ht_weighted_token_f1", "ht_weighted_exact_text_match")
    )
    if valid.empty:
        return tuple(
            DescriptiveMetricEvidence(
                metric=name,
                value=None,
                interval=None,
                defined=False,
            )
            for name in metric_names
        )
    result: list[DescriptiveMetricEvidence] = []
    for metric_index, metric_name in enumerate(metric_names):

        def metric(sample: pd.DataFrame, index: int = metric_index) -> float:
            scores = [
                _special_scores(feature, actual, forecast)[index]
                for actual, forecast in zip(
                    sample["human_value"], sample["model_value"], strict=True
                )
            ]
            return _ht_weighted_mean(
                scores,
                sample["inclusion_probability"].astype(float).tolist(),
            )

        point = metric(valid)
        lower, upper = stratified_bootstrap_interval(
            valid,
            metric,
            rng=_validation_rng(feature, model_id, f"{purpose}:{metric_name}"),
            replicates=bootstrap_replicates,
        )
        result.append(
            DescriptiveMetricEvidence(
                metric=metric_name,
                value=point,
                interval=IntervalEvidence(lower=lower, upper=upper),
                defined=bool(
                    np.isfinite(point) and np.isfinite(lower) and np.isfinite(upper)
                ),
            )
        )
    return tuple(result)


def _special_scores(
    feature: str, actual: object, forecast: object
) -> tuple[float, float]:
    if feature == "qdmr_operator_set":
        actual_set = set(cast(Sequence[str], actual))
        forecast_set = set(cast(Sequence[str], forecast))
        union = actual_set | forecast_set
        jaccard = 1.0 if not union else len(actual_set & forecast_set) / len(union)
        return float(jaccard), float(actual_set == forecast_set)
    actual_text = str(actual)
    forecast_text = str(forecast)
    actual_tokens = actual_text.casefold().split()
    forecast_tokens = forecast_text.casefold().split()
    actual_counts: dict[str, int] = {}
    forecast_counts: dict[str, int] = {}
    for token in actual_tokens:
        actual_counts[token] = actual_counts.get(token, 0) + 1
    for token in forecast_tokens:
        forecast_counts[token] = forecast_counts.get(token, 0) + 1
    overlap = sum(
        min(count, forecast_counts.get(token, 0))
        for token, count in actual_counts.items()
    )
    denominator = len(actual_tokens) + len(forecast_tokens)
    token_f1 = 1.0 if denominator == 0 else 2 * overlap / denominator
    return float(token_f1), float(actual_text == forecast_text)


def _operator_evidence(
    rows: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int]]]:
    evidence: dict[str, dict[str, dict[str, int]]] = {}
    for model_id, model_rows in rows.groupby("model_id", sort=True):
        per_operator: dict[str, dict[str, int]] = {}
        for operator in QDMR_OPERATOR_INVENTORY:
            prevalence = 0
            false_positive = 0
            false_negative = 0
            for row in model_rows.to_dict(orient="records"):
                human = set(cast(Sequence[str], row["human_value"] or []))
                model = set(cast(Sequence[str], row["model_value"] or []))
                prevalence += int(operator in model)
                false_positive += int(operator in model and operator not in human)
                false_negative += int(operator in human and operator not in model)
            per_operator[operator] = {
                "model_prevalence": prevalence,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        evidence[str(model_id)] = per_operator
    return evidence


def _ht_weighted_mean(
    values: Sequence[float], inclusion_probabilities: Sequence[float]
) -> float:
    if len(values) != len(inclusion_probabilities) or not values:
        raise MalformedInputError(
            "Weighted descriptive inputs must be non-empty and aligned"
        )
    probabilities = np.asarray(inclusion_probabilities, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities <= 0) | (probabilities > 1)
    ):
        raise MalformedInputError("Inclusion probabilities must lie in (0, 1]")
    weights = np.reciprocal(probabilities)
    return float(np.dot(weights, np.asarray(values, dtype=float)) / weights.sum())


def _exploratory_inter_model_metrics(
    feature: str,
    rows: pd.DataFrame,
    bootstrap_replicates: int,
) -> tuple[DescriptiveMetricEvidence, ...]:
    if feature not in {"qdmr_operator_set", "qdmr_normalized_question"}:
        alpha, lower, upper = _feature_alpha(feature, rows, bootstrap_replicates)
        return (
            DescriptiveMetricEvidence(
                metric=f"ht_weighted_{_alpha_level(feature)}_alpha",
                value=alpha if np.isfinite(alpha) else None,
                interval=(
                    IntervalEvidence(lower=lower, upper=upper)
                    if np.isfinite(lower) and np.isfinite(upper)
                    else None
                ),
                defined=bool(
                    np.isfinite(alpha) and np.isfinite(lower) and np.isfinite(upper)
                ),
            ),
        )
    paired = _paired_model_frame(rows)
    return _special_descriptive_metrics(
        feature,
        paired,
        model_id="both_models",
        purpose="inter_model",
        bootstrap_replicates=bootstrap_replicates,
    )


def _paired_model_frame(rows: pd.DataFrame) -> pd.DataFrame:
    model_ids = sorted(rows["model_id"].astype(str).unique())
    if len(model_ids) != 2:
        raise GateFailureError("Descriptive evidence requires exactly two models")
    records: list[dict[str, object]] = []
    for _, group in rows.groupby(["item_id", "stratum_id"], sort=True):
        by_model = {
            str(row["model_id"]): row["model_value"]
            for row in group.to_dict(orient="records")
        }
        first = by_model.get(model_ids[0])
        second = by_model.get(model_ids[1])
        records.append(
            {
                "item_id": str(group["item_id"].iloc[0]),
                "stratum_id": str(group["stratum_id"].iloc[0]),
                "human_value": first,
                "model_value": second,
                "inclusion_probability": float(group["inclusion_probability"].iloc[0]),
            }
        )
    return pd.DataFrame(records)


def _exploratory_human_test_retest_metrics(
    feature: str,
    gold_labels: pd.DataFrame,
    bootstrap_replicates: int,
) -> tuple[DescriptiveMetricEvidence, ...]:
    selected = gold_labels.loc[
        gold_labels["feature"].eq(feature)
        & gold_labels["human_recode_value_json"].notna()
    ].copy()
    if selected.empty:
        return (
            DescriptiveMetricEvidence(
                metric="human_test_retest_unavailable",
                value=None,
                interval=None,
                defined=False,
            ),
        )
    selected["human_value"] = selected["human_gold_value_json"].map(
        lambda value: json.loads(str(value))
    )
    selected["model_value"] = selected["human_recode_value_json"].map(
        lambda value: json.loads(str(value))
    )
    selected["inclusion_probability"] = selected[
        "recode_joint_inclusion_probability"
    ].astype(float)
    if feature in {"qdmr_operator_set", "qdmr_normalized_question"}:
        return _special_descriptive_metrics(
            feature,
            selected,
            model_id="human_recode",
            purpose="test_retest",
            bootstrap_replicates=bootstrap_replicates,
        )
    evidence = _human_test_retest(feature, gold_labels, bootstrap_replicates)
    return (
        DescriptiveMetricEvidence(
            metric=evidence.metric,
            value=evidence.metric_value if evidence.metric_defined else None,
            interval=evidence.metric_interval if evidence.metric_defined else None,
            baseline=evidence.matched_naive_baseline,
            defined=evidence.metric_defined,
        ),
        DescriptiveMetricEvidence(
            metric=f"ht_weighted_{evidence.measurement_level}_alpha",
            value=(evidence.krippendorff_alpha if evidence.alpha_defined else None),
            interval=(
                evidence.krippendorff_alpha_interval if evidence.alpha_defined else None
            ),
            defined=evidence.alpha_defined,
        ),
    )


def _model_validation_evidence(
    feature: str,
    model_id: str,
    frame: pd.DataFrame,
    bootstrap_replicates: int,
) -> ModelValidationEvidence:
    point = _metric_summary(feature, frame)
    lower, upper = stratified_bootstrap_interval(
        frame,
        lambda sample: _metric_summary(feature, sample).value,
        rng=_validation_rng(feature, model_id, "assigned_metric"),
        replicates=bootstrap_replicates,
    )
    metric_interval = IntervalEvidence(lower=lower, upper=upper)
    if FEATURE_KINDS[feature] == "numeric":
        if (frame["human_value"].notna() & frame["model_value"].isna()).any():
            raise GateFailureError(
                f"Model {model_id} has null {feature} predictions on numeric gold"
            )
        valid = frame.loc[frame["human_value"].notna() & frame["model_value"].notna()]
        probabilities = valid["inclusion_probability"].astype(float).tolist()
        weights = np.reciprocal(np.asarray(probabilities, dtype=float))
        errors = np.abs(
            valid["human_value"].astype(float).to_numpy()
            - valid["model_value"].astype(float).to_numpy()
        )
        total_weight = float(weights.sum())
        weighted_mae = float(np.dot(weights, errors) / total_weight)
        weighted_rmse = float(np.sqrt(np.dot(weights, errors**2) / total_weight))
        return ModelValidationEvidence(
            model_id=model_id,
            metric=_metric_name(feature),
            metric_value=point.value,
            metric_interval=metric_interval,
            matched_naive_baseline=point.baseline,
            metric_rule_passed=bool(np.isfinite(lower) and lower > point.baseline),
            confusion=None,
            class_metrics=(),
            numeric_error_summary={
                "ht_weighted_mae": weighted_mae,
                "ht_weighted_rmse": weighted_rmse,
                "maximum_absolute_error": float(errors.max()),
            },
            scored_item_count=len(valid),
            gold_missingness_count=int(frame["human_value"].isna().sum()),
            forecast_missingness_error_count=0,
            failure_patterns=_failure_patterns(feature, model_id, valid),
        )

    substantive_levels = FEATURE_SUBSTANTIVE_LEVELS[feature]
    all_levels = FEATURE_ALL_LEVELS[feature]
    if unknown := sorted(
        {
            str(value)
            for value in frame["model_value"].dropna()
            if value not in all_levels
        }
    ):
        raise GateFailureError(
            f"Model {model_id} has unknown {feature} labels: {unknown}"
        )
    if frame.loc[frame["human_value"].isin(all_levels), "model_value"].isna().any():
        raise GateFailureError(
            f"Model {model_id} has null {feature} predictions on classified gold"
        )
    substantive = frame.loc[frame["human_value"].isin(substantive_levels)].copy()
    scored_raw = ht_confusion_totals(
        substantive["human_value"].tolist(),
        substantive["model_value"].tolist(),
        [1.0] * len(substantive),
        substantive_levels,
        all_levels,
    )
    scored_ht = ht_confusion_totals(
        substantive["human_value"].tolist(),
        substantive["model_value"].tolist(),
        substantive["inclusion_probability"].astype(float).tolist(),
        substantive_levels,
        all_levels,
    )
    full = frame.loc[
        frame["human_value"].isin(all_levels) & frame["model_value"].isin(all_levels)
    ]
    full_raw = ht_confusion_totals(
        full["human_value"].tolist(),
        full["model_value"].tolist(),
        [1.0] * len(full),
        all_levels,
        all_levels,
    )
    full_ht = ht_confusion_totals(
        full["human_value"].tolist(),
        full["model_value"].tolist(),
        full["inclusion_probability"].astype(float).tolist(),
        all_levels,
        all_levels,
    )
    class_metrics = tuple(
        _class_metric_evidence(
            feature,
            model_id,
            substantive,
            scored_raw,
            scored_ht,
            class_index,
            label,
            bootstrap_replicates,
        )
        for class_index, label in enumerate(substantive_levels)
    )
    missingness_levels = set(all_levels) - set(substantive_levels)
    forecast_missingness_count = int(
        substantive["model_value"].isin(missingness_levels).sum()
    )
    return ModelValidationEvidence(
        model_id=model_id,
        metric=_metric_name(feature),
        metric_value=point.value,
        metric_interval=metric_interval,
        matched_naive_baseline=point.baseline,
        metric_rule_passed=bool(np.isfinite(lower) and lower > point.baseline),
        confusion=ConfusionEvidence(
            gold_levels_json=tuple(_json_label(level) for level in all_levels),
            forecast_levels_json=tuple(_json_label(level) for level in all_levels),
            raw_cells=tuple(
                tuple(int(value) for value in row) for row in full_raw.tolist()
            ),
            ht_cells=tuple(
                tuple(float(value) for value in row) for row in full_ht.tolist()
            ),
        ),
        class_metrics=class_metrics,
        sensitivity=class_metrics[1].recall
        if FEATURE_KINDS[feature] == "binary"
        else None,
        specificity=class_metrics[0].recall
        if FEATURE_KINDS[feature] == "binary"
        else None,
        scored_item_count=len(substantive),
        gold_missingness_count=int(frame["human_value"].isin(missingness_levels).sum()),
        forecast_missingness_error_count=forecast_missingness_count,
        failure_patterns=_failure_patterns(feature, model_id, substantive),
    )


def _class_metric_evidence(
    feature: str,
    model_id: str,
    frame: pd.DataFrame,
    raw_matrix: np.ndarray,
    ht_matrix: np.ndarray,
    class_index: int,
    label: object,
    bootstrap_replicates: int,
) -> ClassMetricEvidence:
    def precision(matrix: np.ndarray) -> float:
        denominator = float(matrix[:, class_index].sum())
        return (
            float("nan")
            if denominator == 0
            else float(matrix[class_index, class_index] / denominator)
        )

    def recall(matrix: np.ndarray) -> float:
        denominator = float(matrix[class_index, :].sum())
        return (
            float("nan")
            if denominator == 0
            else float(matrix[class_index, class_index] / denominator)
        )

    def matrix_for(sample: pd.DataFrame) -> np.ndarray:
        return ht_confusion_totals(
            sample["human_value"].tolist(),
            sample["model_value"].tolist(),
            sample["inclusion_probability"].astype(float).tolist(),
            FEATURE_SUBSTANTIVE_LEVELS[feature],
            FEATURE_ALL_LEVELS[feature],
        )

    precision_lower, precision_upper = stratified_bootstrap_interval(
        frame,
        lambda sample: precision(matrix_for(sample)),
        rng=_validation_rng(feature, model_id, f"class:{_json_label(label)}:precision"),
        replicates=bootstrap_replicates,
    )
    recall_lower, recall_upper = stratified_bootstrap_interval(
        frame,
        lambda sample: recall(matrix_for(sample)),
        rng=_validation_rng(feature, model_id, f"class:{_json_label(label)}:recall"),
        replicates=bootstrap_replicates,
    )
    return ClassMetricEvidence(
        label_json=_json_label(label),
        raw_gold_count=int(raw_matrix[class_index, :].sum()),
        ht_gold_total=float(ht_matrix[class_index, :].sum()),
        precision=precision(ht_matrix),
        precision_interval=IntervalEvidence(
            lower=precision_lower, upper=precision_upper
        ),
        recall=recall(ht_matrix),
        recall_interval=IntervalEvidence(lower=recall_lower, upper=recall_upper),
    )


def _failure_patterns(
    feature: str,
    model_id: str,
    frame: pd.DataFrame,
) -> tuple[FailurePatternEvidence, ...]:
    failures = frame.loc[frame["human_value"] != frame["model_value"]].copy()
    if failures.empty:
        return ()
    failures["_actual_json"] = failures["human_value"].map(_json_label)
    failures["_forecast_json"] = failures["model_value"].map(_json_label)
    patterns: list[FailurePatternEvidence] = []
    for (actual_json, forecast_json), group in failures.groupby(
        ["_actual_json", "_forecast_json"], sort=True
    ):
        probabilities = group["inclusion_probability"].astype(float)
        pattern_key = f"{feature}\0{model_id}\0{actual_json}\0{forecast_json}"
        patterns.append(
            FailurePatternEvidence(
                pattern_id=f"pattern_{sha256(pattern_key.encode()).hexdigest()[:16]}",
                actual_json=str(actual_json),
                forecast_json=str(forecast_json),
                raw_count=len(group),
                ht_total=float(np.reciprocal(probabilities).sum()),
                example_item_ids=tuple(
                    sorted(group["item_id"].astype(str).unique())[:5]
                ),
            )
        )
    return tuple(patterns)


def _json_label(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


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
    all_levels = FEATURE_ALL_LEVELS[feature]
    substantive = valid.loc[valid["human_value"].isin(levels)]
    matrix = ht_confusion_totals(
        substantive["human_value"].tolist(),
        substantive["model_value"].tolist(),
        substantive["inclusion_probability"].astype(float).tolist(),
        levels,
        all_levels,
    )
    if kind == "binary":
        return MetricSummary(balanced_accuracy(matrix), 0.5, counts)
    if kind == "ordinal":
        return MetricSummary(quadratically_weighted_kappa(matrix), 0.0, counts)
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
    return _prediction_rows_for_features(
        gold_labels,
        normalized,
        frozenset(GOLD_VALIDATION_FEATURES),
    )


def _prediction_rows_for_features(
    gold_labels: pd.DataFrame,
    normalized: pd.DataFrame,
    features: frozenset[str],
) -> pd.DataFrame:
    """Build aligned model-versus-human rows for a closed feature inventory."""
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
        if feature not in features:
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
    return pd.DataFrame(
        rows,
        columns=[
            "item_id",
            "feature",
            "model_id",
            "model_value",
            "human_value",
            "stratum_id",
            "inclusion_probability",
        ],
    )


def _annotation_value(annotation: Mapping[str, object], feature: str) -> object:
    value = annotation.get(feature)
    if isinstance(value, (list, tuple, np.ndarray)):
        return sorted(str(item) for item in value)
    if value is not None and not bool(pd.isna(value)):
        return value
    if feature == "referring_form_type":
        return annotation.get("referring_form_missingness")
    if feature == "answer_locality":
        return annotation.get("answer_locality_missingness")
    return None


def _corpus_prevalence(
    feature: str, normalized: pd.DataFrame
) -> dict[str, dict[str, int]]:
    prevalence: dict[str, dict[str, int]] = {}
    for model_id, rows in normalized.groupby("model_id", sort=True):
        counts: dict[str, int] = {}
        for annotation in rows.to_dict(orient="records"):
            value = _annotation_value(annotation, feature)
            label = "<NULL>" if value is None else _json_label(value)
            counts[label] = counts.get(label, 0) + 1
        prevalence[str(model_id)] = dict(sorted(counts.items()))
    return prevalence


def _human_test_retest(
    feature: str,
    gold_labels: pd.DataFrame,
    bootstrap_replicates: int,
) -> HumanTestRetestEvidence:
    required = {
        "feature",
        "human_gold_value_json",
        "human_recode_value_json",
        "recode_joint_inclusion_probability",
        "stratum_id",
        "item_id",
    }
    if missing := sorted(required - set(gold_labels.columns)):
        raise MalformedInputError(
            f"Gold labels are missing re-code evidence columns: {missing}"
        )
    selected = gold_labels.loc[
        gold_labels["feature"].eq(feature)
        & gold_labels["human_recode_value_json"].notna()
    ].copy()
    if selected.empty:
        raise GateFailureError(f"Human re-code contains no rows for {feature}")
    selected["human_value"] = selected["human_gold_value_json"].map(
        lambda value: json.loads(str(value))
    )
    selected["model_value"] = selected["human_recode_value_json"].map(
        lambda value: json.loads(str(value))
    )
    selected["inclusion_probability"] = selected[
        "recode_joint_inclusion_probability"
    ].astype(float)
    point = _metric_summary(feature, selected)
    metric_lower, metric_upper = stratified_bootstrap_interval(
        selected,
        lambda sample: _metric_summary(feature, sample).value,
        rng=_validation_rng(feature, "human_recode", "assigned_metric"),
        replicates=bootstrap_replicates,
    )
    level = _alpha_level(feature)
    ordinal_levels = (
        COGNITIVE_PROCESS_LEVELS if feature == "cognitive_process_level" else None
    )

    def alpha_metric(sample: pd.DataFrame) -> float:
        probabilities = sample["recode_joint_inclusion_probability"].astype(float)
        return krippendorff_alpha(
            sample[["human_value", "model_value"]].values.tolist(),
            level=level,
            ordinal_levels=ordinal_levels,
            unit_weights=np.reciprocal(probabilities).tolist(),
        )

    alpha_value = alpha_metric(selected)
    alpha_lower, alpha_upper = stratified_bootstrap_interval(
        selected,
        alpha_metric,
        rng=_validation_rng(feature, "human_recode", "alpha"),
        replicates=bootstrap_replicates,
    )
    return HumanTestRetestEvidence(
        item_count=len(selected),
        metric=_metric_name(feature),
        metric_value=point.value,
        metric_interval=IntervalEvidence(lower=metric_lower, upper=metric_upper),
        metric_defined=bool(
            np.isfinite(point.value)
            and np.isfinite(metric_lower)
            and np.isfinite(metric_upper)
        ),
        matched_naive_baseline=point.baseline,
        krippendorff_alpha=alpha_value,
        krippendorff_alpha_interval=IntervalEvidence(
            lower=alpha_lower, upper=alpha_upper
        ),
        alpha_defined=bool(
            np.isfinite(alpha_value)
            and np.isfinite(alpha_lower)
            and np.isfinite(alpha_upper)
        ),
        measurement_level=level,
        weight="inverse_joint_gold_recode_inclusion_probability",
    )


def _feature_alpha(
    feature: str,
    rows: pd.DataFrame,
    bootstrap_replicates: int,
) -> tuple[float, float, float]:
    pivot = rows.pivot(
        index=["item_id", "stratum_id"], columns="model_id", values="model_value"
    )
    ratings = pivot.values.tolist()
    level = _alpha_level(feature)
    ordinal_levels = (
        COGNITIVE_PROCESS_LEVELS if feature == "cognitive_process_level" else None
    )
    probabilities = (
        rows.groupby(["item_id", "stratum_id"], sort=True)["inclusion_probability"]
        .first()
        .reindex(pivot.index)
    )
    point = krippendorff_alpha(
        ratings,
        level=level,
        ordinal_levels=ordinal_levels,
        unit_weights=np.reciprocal(probabilities.astype(float)).tolist(),
    )
    alpha_frame = pivot.reset_index()
    alpha_frame["inclusion_probability"] = probabilities.to_numpy()
    rng = _validation_rng(feature, "both_models", "alpha")

    def metric(sample: pd.DataFrame) -> float:
        rating_columns = [
            column
            for column in sample.columns
            if column not in {"item_id", "stratum_id", "inclusion_probability"}
        ]
        return krippendorff_alpha(
            sample[rating_columns].values.tolist(),
            level=level,
            ordinal_levels=ordinal_levels,
            unit_weights=np.reciprocal(
                sample["inclusion_probability"].astype(float)
            ).tolist(),
        )

    lower, upper = stratified_bootstrap_interval(
        alpha_frame,
        metric,
        rng=rng,
        replicates=bootstrap_replicates,
    )
    return point, lower, upper


def _alpha_level(feature: str) -> Literal["nominal", "ordinal", "interval"]:
    kind = FEATURE_KINDS[feature]
    if kind == "numeric":
        return "interval"
    if kind == "ordinal":
        return "ordinal"
    return "nominal"


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
        "ordinal": "ht_weighted_quadratic_kappa",
    }[FEATURE_KINDS[feature]]


def _validation_rng(
    feature: str,
    model_id: str,
    purpose: str,
) -> np.random.Generator:
    parent = derive_stream_seed("gold_sampling")
    digest = sha256(f"{parent}:{feature}:{model_id}:{purpose}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="big"))


def _locked_codebook_sha256(workspace: AnalysisWorkspace) -> str:
    fingerprint = workspace.load_manifest().resources.get("segment3_codebook")
    if fingerprint is None:
        raise GateFailureError("Analysis lock is missing the Stage 5 codebook")
    return fingerprint.sha256


def _validation_evidence_hashes(
    workspace: AnalysisWorkspace,
) -> dict[str, str]:
    paths = {
        "gold_sample_sha256": "tables/gold_sample.parquet",
        "gold_labels_sha256": "tables/gold_labels.parquet",
        "annotation_manifest_sha256": "annotation_manifest.json",
    }
    hashes = {
        name: sha256_file(workspace.store.path_for(path))
        for name, path in paths.items()
    }
    hashes["codebook_sha256"] = _locked_codebook_sha256(workspace)
    return hashes


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
        pattern_rows = cast(
            list[dict[str, object]], dossiers[feature]["failure_patterns"]
        )
        available_pattern_ids = {str(pattern["pattern_id"]) for pattern in pattern_rows}
        if not set(decision.material_failure_pattern_ids).issubset(
            available_pattern_ids
        ):
            raise GateFailureError(
                f"Feature decision cites unknown failure patterns: {feature}"
            )
        model_rows = cast(list[dict[str, object]], dossiers[feature]["model_summaries"])
        available_class_interval_ids = {
            f"{model['model_id']}:{class_metric['label_json']}:{metric_name}"
            for model in model_rows
            for class_metric in cast(list[dict[str, object]], model["class_metrics"])
            for metric_name in ("precision", "recall")
        }
        if not set(decision.material_class_interval_ids).issubset(
            available_class_interval_ids
        ):
            raise GateFailureError(
                f"Feature decision cites unknown class intervals: {feature}"
            )
        ceiling = _status_ceiling(
            dossiers[feature],
            decision.material_failure_mode,
            decision.class_interval_material_limitation,
        )
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
    class_interval_material_limitation: bool = False,
) -> FeatureValidationStatus:
    if not bool(dossier["structural_rules_passed"]):
        return FeatureValidationStatus.NOT_VALIDATED
    if (
        material_failure_mode
        or class_interval_material_limitation
        or not bool(dossier["minimum_class_coverage_passed"])
    ):
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
                "material_failure_pattern_ids_json": json.dumps(
                    decision.material_failure_pattern_ids
                ),
                "class_interval_material_limitation": (
                    decision.class_interval_material_limitation
                ),
                "material_class_interval_ids_json": json.dumps(
                    decision.material_class_interval_ids
                ),
                "class_interval_justification": (decision.class_interval_justification),
                "justification": decision.justification,
                "decision_git_commit": decision.git_commit,
                "structural_rules_passed": bool(
                    dossiers[decision.feature]["structural_rules_passed"]
                ),
                "weakest_model_id": dossiers[decision.feature]["weakest_model_id"],
                "weakest_metric_value": dossiers[decision.feature][
                    "weakest_metric_value"
                ],
                "widest_model_interval_id": dossiers[decision.feature][
                    "widest_model_interval_id"
                ],
                "widest_metric_interval_width": dossiers[decision.feature][
                    "widest_metric_interval_width"
                ],
                "widest_class_interval_id": dossiers[decision.feature][
                    "widest_class_interval_id"
                ],
                "widest_class_interval_width": dossiers[decision.feature][
                    "widest_class_interval_width"
                ],
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
        (
            f"- `{row['feature']}`: `{row['validation_status']}`; inference "
            f"`{row['inference_mode']}`; weakest `{row['weakest_model_id']}` "
            f"({row['weakest_metric_value']:.4f}); widest model interval "
            f"`{row['widest_model_interval_id']}` "
            f"(width {row['widest_metric_interval_width']:.4f})."
        )
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


def _render_exploratory_report(
    dossiers: Mapping[str, Mapping[str, object]],
) -> str:
    lines = [
        "# Partial report 05 — Exploratory P4 evidence",
        "",
        (
            "This evidence is descriptive and non-gating. It cannot change feature "
            "statuses, family inference modes, the DSL gate, or outcome-modelling unlock."
        ),
        "",
    ]
    for feature in GOLD_EXPLORATORY_EVIDENCE_FEATURES:
        dossier = dossiers[feature]
        lines.append(
            f"- `{feature}`: `{dossier['availability']}`; kind "
            f"`{dossier['evidence_kind']}`; non_gating=true."
        )
    lines.extend(["", "No outcome table or downstream estimate was loaded.", ""])
    return "\n".join(lines)
