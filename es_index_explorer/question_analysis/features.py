"""Segment 3 deterministic feature extraction."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol, cast

import pandas as pd

from es_index_explorer.question_analysis.contracts import (
    OUTCOME_FIELD_DENYLIST,
    ArtifactMetadata,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    GateFailureError,
    MalformedInputError,
)
from es_index_explorer.question_analysis.linguistics import (
    EntitySpan,
    ParsedSentence,
    extract_linguistic_features,
)
from es_index_explorer.question_analysis.nlp_resources import (
    DEFAULT_STANZA_MODEL_DIR,
    SpacyNer,
    StanzaParser,
    build_parser_resource_manifest,
    download_stanza_resources,
)
from es_index_explorer.question_analysis.storage import (
    canonical_json_bytes,
    versioned_frame,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

CATALOGUE_ARTIFACTS = {
    "rubrics": "tables/rubric_catalogue.parquet",
    "variants": "tables/variant_catalogue.parquet",
    "expectations": "tables/expectation_catalogue.parquet",
}
EXPECTED_ITEM_COUNTS = {"question": 263, "expectation": 314}
type LiteralItemType = Literal["question", "expectation"]
ENTITY_COLUMNS = [
    "item_id",
    "entity_index",
    "text",
    "label",
    "start_char",
    "end_char",
]
TOKEN_COLUMNS = [
    "item_id",
    "sentence_index",
    "word_id",
    "text",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "start_char",
    "end_char",
]


class StanzaParserLike(Protocol):
    def parse(self, text: str) -> tuple[ParsedSentence, ...]: ...


class SpacyNerLike(Protocol):
    def parse(self, text: str) -> tuple[EntitySpan, ...]: ...


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Runtime locations for deterministic extraction."""

    stanza_model_dir: Path = DEFAULT_STANZA_MODEL_DIR
    download_resources: bool = False


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """Segment 3 tables and verification products."""

    features: pd.DataFrame
    stanza_tokens: pd.DataFrame
    spacy_entities: pd.DataFrame
    parser_resource_manifest: dict[str, object]
    verification: dict[str, object]
    report: str


def run_features(
    workspace: AnalysisWorkspace, config: FeatureConfig | None = None
) -> tuple[ArtifactMetadata, ...]:
    """Run Segment 3 and persist deterministic linguistic artifacts."""
    config = config or FeatureConfig()
    if config.download_resources:
        download_stanza_resources(config.stanza_model_dir)

    catalogues = _load_catalogues(workspace)
    parser_manifest = build_parser_resource_manifest(config.stanza_model_dir)
    result = build_features(
        rubrics=catalogues["rubrics"],
        variants=catalogues["variants"],
        expectations=catalogues["expectations"],
        stanza_parser=StanzaParser.load(config.stanza_model_dir),
        spacy_ner=SpacyNer.load(),
        parser_resource_manifest=parser_manifest,
    )
    artifacts = (
        workspace.store.write_parquet(
            "tables/features_deterministic.parquet",
            versioned_frame(result.features),
            created_by=WorkflowCommand.FEATURES,
        ),
        workspace.store.write_parquet(
            "tables/stanza_tokens.parquet",
            versioned_frame(result.stanza_tokens),
            created_by=WorkflowCommand.FEATURES,
        ),
        workspace.store.write_parquet(
            "tables/spacy_entities.parquet",
            versioned_frame(result.spacy_entities),
            created_by=WorkflowCommand.FEATURES,
        ),
        workspace.store.write_json(
            "parser_resource_manifest.json",
            result.parser_resource_manifest,
            created_by=WorkflowCommand.FEATURES,
        ),
        workspace.store.write_json(
            "feature_extraction_verification.json",
            result.verification,
            created_by=WorkflowCommand.FEATURES,
        ),
        workspace.store.write_bytes(
            "partial_reports/02-deterministic-features.md",
            result.report.encode("utf-8"),
            created_by=WorkflowCommand.FEATURES,
        ),
    )
    if not bool(result.verification["passed"]):
        raise GateFailureError(
            f"Deterministic feature verification failed: "
            f"{result.verification['blocking_failures']}"
        )
    return artifacts


def build_features(
    *,
    rubrics: pd.DataFrame,
    variants: pd.DataFrame,
    expectations: pd.DataFrame,
    stanza_parser: StanzaParserLike,
    spacy_ner: SpacyNerLike,
    parser_resource_manifest: Mapping[str, object],
) -> FeatureResult:
    """Build deterministic features without loading outcomes."""
    text_items = build_text_items(rubrics, variants, expectations)
    parser_manifest = dict(parser_resource_manifest)
    parser_manifest_sha256 = sha256(canonical_json_bytes(parser_manifest)).hexdigest()
    feature_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []

    for item in text_items.to_dict(orient="records"):
        item_id = str(item["item_id"])
        source_text = str(item["source_text"])
        item_type = cast(str, item["item_type"])
        sentences = stanza_parser.parse(source_text)
        entities = spacy_ner.parse(source_text)
        if item_type not in EXPECTED_ITEM_COUNTS:
            raise MalformedInputError(f"Unknown deterministic item type: {item_type}")
        linguistic = extract_linguistic_features(
            cast(LiteralItemType, item_type),
            source_text,
            sentences,
            entities,
        )

        missingness = dict(linguistic.missingness)
        if item_type == "question":
            missingness["expectation_document_count"] = "not_applicable_to_question"
        feature_row = dict(item)
        feature_row.update(asdict(linguistic))
        feature_row["missingness_reasons"] = json.dumps(
            missingness, sort_keys=True, separators=(",", ":")
        )
        feature_row.pop("missingness")
        feature_row["parser_manifest_sha256"] = parser_manifest_sha256
        feature_rows.append(feature_row)

        for sentence in sentences:
            for word in sentence.words:
                token_rows.append(
                    {
                        "item_id": item_id,
                        "sentence_index": word.sentence_index,
                        "word_id": word.word_id,
                        "text": word.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "feats": "|".join(
                            f"{key}={value}" for key, value in word.feats
                        ),
                        "head": word.head,
                        "deprel": word.deprel,
                        "start_char": word.start_char,
                        "end_char": word.end_char,
                    }
                )
        entity_rows.extend(
            {
                "item_id": item_id,
                "entity_index": entity.entity_index,
                "text": entity.text,
                "label": entity.label,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
            }
            for entity in entities
        )

    features = pd.DataFrame(feature_rows).sort_values(
        "item_order", kind="stable"
    ).reset_index(drop=True)
    stanza_tokens = pd.DataFrame(token_rows, columns=TOKEN_COLUMNS).sort_values(
        ["item_id", "sentence_index", "word_id"], kind="stable"
    ).reset_index(drop=True)
    spacy_entities = pd.DataFrame(entity_rows, columns=ENTITY_COLUMNS).sort_values(
        ["item_id", "entity_index"], kind="stable"
    ).reset_index(drop=True)
    verification = _verify_features(
        features, stanza_tokens, parser_manifest_sha256
    )
    return FeatureResult(
        features=features,
        stanza_tokens=stanza_tokens,
        spacy_entities=spacy_entities,
        parser_resource_manifest=parser_manifest,
        verification=verification,
        report=_render_report(verification, parser_manifest),
    )


def build_text_items(
    rubrics: pd.DataFrame,
    variants: pd.DataFrame,
    expectations: pd.DataFrame,
) -> pd.DataFrame:
    """Create the stable 577-item outcome-blind text population."""
    _reject_outcome_columns(rubrics, variants, expectations)
    rubric_fields = rubrics[
        ["rubric_id", "expectation_count", "variant_count"]
    ].copy()
    if rubric_fields["rubric_id"].duplicated().any():
        raise GateFailureError("Rubric catalogue has duplicate rubric_id values")

    question_rows = variants[
        [
            "variant_id",
            "rubric_id",
            "rubric_order",
            "eval_dataset",
            "source_path",
            "variant_index",
            "question",
            "expectation_count",
            "variant_count",
            "use_cases",
        ]
    ].copy()
    question_rows.rename(
        columns={"variant_id": "item_id", "question": "source_text"}, inplace=True
    )
    question_rows["item_type"] = "question"
    question_rows["variant_id"] = question_rows["item_id"]
    question_rows["expectation_id"] = None
    question_rows["expectation_index"] = pd.NA
    question_rows["expectation_document_count"] = pd.NA

    expectation_rows = expectations[
        [
            "expectation_id",
            "rubric_id",
            "rubric_order",
            "eval_dataset",
            "source_path",
            "expectation_index",
            "description",
            "expectation_document_count",
            "use_cases",
        ]
    ].merge(
        rubric_fields,
        on="rubric_id",
        how="left",
        validate="many_to_one",
    )
    expectation_rows.rename(
        columns={"expectation_id": "item_id", "description": "source_text"},
        inplace=True,
    )
    expectation_rows["item_type"] = "expectation"
    expectation_rows["expectation_id"] = expectation_rows["item_id"]
    expectation_rows["variant_id"] = None
    expectation_rows["variant_index"] = pd.NA

    columns = [
        "item_id",
        "item_type",
        "eval_dataset",
        "rubric_id",
        "rubric_order",
        "source_path",
        "variant_id",
        "expectation_id",
        "variant_index",
        "expectation_index",
        "source_text",
        "expectation_count",
        "variant_count",
        "expectation_document_count",
        "use_cases",
    ]
    result = pd.concat(
        [question_rows[columns], expectation_rows[columns]], ignore_index=True
    )
    result.insert(0, "item_order", range(len(result)))
    result["text_sha256"] = [
        sha256(str(text).encode("utf-8")).hexdigest()
        for text in result["source_text"]
    ]
    result["use_cases"] = result["use_cases"].map(_normalize_string_list)
    if result["item_id"].duplicated().any():
        raise GateFailureError("Deterministic text population has duplicate item_id values")
    if result["source_text"].map(lambda value: not str(value).strip()).any():
        raise GateFailureError("Deterministic text population contains empty source text")
    return result


def _load_catalogues(workspace: AnalysisWorkspace) -> dict[str, pd.DataFrame]:
    state = workspace.load_state()
    catalogues: dict[str, pd.DataFrame] = {}
    for name, relative_path in CATALOGUE_ARTIFACTS.items():
        metadata = state.artifacts.get(relative_path)
        if metadata is None or metadata.created_by is not WorkflowCommand.JOIN:
            raise MalformedInputError(
                f"Required Segment 2 catalogue is not registered: {relative_path}"
            )
        frame = pd.read_parquet(workspace.store.path_for(relative_path))
        if "artifact_schema_version" not in frame:
            raise MalformedInputError(
                f"Catalogue lacks artifact_schema_version: {relative_path}"
            )
        catalogues[name] = frame.drop(columns="artifact_schema_version")
    return catalogues


def _reject_outcome_columns(*frames: pd.DataFrame) -> None:
    forbidden = set(OUTCOME_FIELD_DENYLIST) | {
        "criterion_observation_id",
        "trace_id",
        "run_id",
        "eligible",
        "logged_rubric_v2",
        "recomputed_ordinal_grade",
    }
    present = sorted(
        forbidden.intersection(
            column.casefold() for frame in frames for column in frame.columns
        )
    )
    if present:
        raise GateFailureError(
            f"Outcome-blind feature inputs contain forbidden columns: {present}"
        )


def _verify_features(
    features: pd.DataFrame,
    stanza_tokens: pd.DataFrame,
    parser_manifest_sha256: str,
) -> dict[str, object]:
    item_counts = features["item_type"].value_counts().sort_index().to_dict()
    blocking_failures: list[str] = []
    if item_counts != EXPECTED_ITEM_COUNTS:
        blocking_failures.append("item_population")
    if features["item_id"].duplicated().any():
        blocking_failures.append("item_identity")
    mandatory = [
        "token_count",
        "dependency_tree_depth",
        "clause_count",
        "coordination_count",
        "named_entity_count",
        "temporal_expression_present",
    ]
    mandatory_missing = {
        column: int(features[column].isna().sum()) for column in mandatory
    }
    if any(mandatory_missing.values()):
        blocking_failures.append("mandatory_feature_missingness")
    question_clause_missing = int(
        features.loc[features["item_type"].eq("question"), "clause_type"]
        .isna()
        .sum()
    )
    if question_clause_missing:
        blocking_failures.append("question_clause_type")
    token_item_count = int(stanza_tokens["item_id"].nunique())
    if token_item_count != len(features):
        blocking_failures.append("parse_archive_coverage")

    missingness = {
        column: int(features[column].isna().sum())
        for column in (
            "expectation_document_count",
            "mean_dependency_length",
            "subordinate_clause_ratio",
            "complex_nominals_per_clause",
            "clause_type",
        )
    }
    return {
        "schema_version": 1,
        "passed": not blocking_failures,
        "blocking_failures": blocking_failures,
        "item_counts": item_counts,
        "total_items": len(features),
        "unique_item_count": int(features["item_id"].nunique()),
        "stanza_token_rows": len(stanza_tokens),
        "stanza_item_count": token_item_count,
        "mandatory_missingness": mandatory_missing,
        "expected_missingness": missingness,
        "question_clause_type_missing_count": question_clause_missing,
        "parser_manifest_sha256": parser_manifest_sha256,
        "outcome_columns_loaded": [],
        "numeric_distributions": {
            column: _numeric_distribution(features[column])
            for column in (
                "expectation_count",
                "variant_count",
                "expectation_document_count",
                "token_count",
                "dependency_tree_depth",
                "mean_dependency_length",
                "clause_count",
                "subordinate_clause_ratio",
                "complex_nominals_per_clause",
                "coordination_count",
                "named_entity_count",
            )
        },
        "clause_type_counts": {
            str(key): int(value)
            for key, value in features["clause_type"]
            .dropna()
            .value_counts()
            .sort_index()
            .items()
        },
        "temporal_expression_counts": {
            str(key).lower(): int(value)
            for key, value in features["temporal_expression_present"]
            .value_counts()
            .sort_index()
            .items()
        },
    }


def _numeric_distribution(series: pd.Series) -> dict[str, int | float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _render_report(
    verification: Mapping[str, object],
    parser_manifest: Mapping[str, object],
) -> str:
    item_counts = cast(dict[str, int], verification["item_counts"])
    missingness = cast(dict[str, int], verification["expected_missingness"])
    numeric = cast(
        dict[str, dict[str, int | float | None]],
        verification["numeric_distributions"],
    )
    stanza = cast(dict[str, object], parser_manifest["stanza"])
    spacy = cast(dict[str, object], parser_manifest["spacy"])
    lines = [
        "# Partial report 02 — Deterministic features",
        "",
        f"Feature verification: **{'PASS' if verification['passed'] else 'FAIL'}**.",
        "",
        "## Population and provenance",
        "",
        f"- Questions: {item_counts.get('question', 0)}.",
        f"- Expectation descriptions: {item_counts.get('expectation', 0)}.",
        f"- Total items: {verification['total_items']}.",
        (
            f"- Stanza: {stanza['distribution_version']} "
            f"({verification['stanza_token_rows']} archived words)."
        ),
        (
            f"- spaCy: {spacy['distribution_version']}; "
            f"`{spacy['model']}` {spacy['model_version']} (NER only)."
        ),
        f"- Parser manifest SHA-256: `{verification['parser_manifest_sha256']}`.",
        "- Outcome columns loaded: none.",
        "",
        "## Coverage and missingness",
        "",
    ]
    lines.extend(
        f"- `{feature}`: {count} missing."
        for feature, count in missingness.items()
    )
    lines.extend(["", "## Deterministic distributions", ""])
    for feature, values in numeric.items():
        lines.append(
            f"- `{feature}`: n={values['count']}, min={_format_number(values['min'])}, "
            f"median={_format_number(values['median'])}, "
            f"max={_format_number(values['max'])}."
        )
    lines.extend(
        [
            "",
            "## Clause type",
            "",
        ]
    )
    clause_counts = cast(dict[str, int], verification["clause_type_counts"])
    lines.extend(
        f"- `{label}`: {count}." for label, count in clause_counts.items()
    )
    lines.extend(
        [
            "",
            (
                "`coordination_count` is retained as deterministic exploratory P2 and "
                "is not part of a confirmatory restriction."
            ),
            "",
            "No feature-outcome association was computed.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_number(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _normalize_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        raise MalformedInputError(f"Invalid use_cases value: {value!r}")
    values = list(value)
    if not all(isinstance(item, str) for item in values):
        raise MalformedInputError(f"Invalid use_cases value: {value!r}")
    return cast(list[str], values)
