"""Integration tests for deterministic Segment 3 extraction."""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from es_index_explorer.question_analysis.contracts import WorkflowCommand
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.features import (
    FeatureConfig,
    FeatureResult,
    build_features,
    run_features,
)
from es_index_explorer.question_analysis.join import run_join
from es_index_explorer.question_analysis.nlp_resources import (
    DEFAULT_STANZA_MODEL_DIR,
    SpacyNer,
    StanzaParser,
    build_parser_resource_manifest,
    verify_stanza_resources,
)
from es_index_explorer.question_analysis.storage import ArtifactStore, versioned_frame
from es_index_explorer.question_analysis.workspace import (
    DEFAULT_ANALYSIS_ROOT,
    AnalysisWorkspace,
)

pytestmark = pytest.mark.integration
ARCHIVED_ANALYSIS_ROOT = DEFAULT_ANALYSIS_ROOT.parent / "simplemode-v1-postreview-p2"


def _models_available() -> bool:
    try:
        verify_stanza_resources(DEFAULT_STANZA_MODEL_DIR)
    except MalformedInputError:
        return False
    return True


@pytest.fixture(scope="module")
def result() -> FeatureResult:
    if not _models_available():
        pytest.skip("Pinned Stanza models are not installed")
    tables = DEFAULT_ANALYSIS_ROOT / "tables"
    frames = {
        name: pd.read_parquet(tables / f"{name}_catalogue.parquet").drop(
            columns="artifact_schema_version"
        )
        for name in ("rubric", "variant", "expectation")
    }
    return build_features(
        rubrics=frames["rubric"],
        variants=frames["variant"],
        expectations=frames["expectation"],
        stanza_parser=StanzaParser.load(DEFAULT_STANZA_MODEL_DIR),
        spacy_ner=SpacyNer.load(),
        parser_resource_manifest=build_parser_resource_manifest(
            DEFAULT_STANZA_MODEL_DIR
        ),
    )


def test_full_deterministic_population_passes(result: FeatureResult) -> None:
    assert result.verification["passed"]
    assert result.verification["item_counts"] == {
        "expectation": 314,
        "question": 263,
    }
    assert len(result.features) == 577
    assert result.features["item_id"].is_unique
    assert result.verification["mandatory_missingness"] == {
        "clause_count": 0,
        "coordination_count": 0,
        "dependency_tree_depth": 0,
        "named_entity_count": 0,
        "temporal_expression_present": 0,
        "token_count": 0,
    }
    question_rows = result.features["item_type"].eq("question")
    assert result.features.loc[question_rows, "clause_type"].notna().all()
    assert result.features.loc[~question_rows, "clause_type"].isna().all()
    questions = result.features.loc[question_rows]
    assert questions["clause_type"].value_counts().to_dict() == {
        "open_interrogative": 143,
        "directive_imperative": 70,
        "closed_interrogative": 49,
        "declarative_request": 1,
    }
    assert questions.loc[questions["clause_count"].eq(0), "source_text"].tolist() == [
        "Emails with Dr. Argoff?"
    ]
    assert result.features["clause_count"].eq(0).sum() == 54
    assert result.features["subordinate_clause_ratio"].isna().sum() == 54
    assert result.features["complex_nominals_per_clause"].isna().sum() == 54
    assert result.verification["unexpected_clause_types"] == []
    assert result.verification["invalid_subordinate_ratio_count"] == 0
    assert result.verification["invalid_complex_nominal_ratio_count"] == 0
    assert result.verification["applicability_failures"] == {}
    assert result.verification["parser_manifest_mismatch_count"] == 0
    assert result.stanza_tokens["item_id"].nunique() == 577
    stanza_manifest = result.parser_resource_manifest["stanza"]
    assert isinstance(stanza_manifest, dict)
    assert stanza_manifest["resource_catalogue"]["sha256"] == (
        "4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c"
    )


def test_use_cases_parquet_schema_and_arrow_cells_are_lists() -> None:
    path = DEFAULT_ANALYSIS_ROOT / "tables" / "features_deterministic.parquet"
    field = pq.read_schema(path).field("use_cases")
    arrow_backed = pd.read_parquet(path, dtype_backend="pyarrow")

    assert pa.types.is_list(field.type)
    assert field.type.value_type == pa.string()
    assert isinstance(arrow_backed.loc[0, "use_cases"], list)
    assert all(
        isinstance(use_cases, list) for use_cases in arrow_backed["use_cases"]
    )


def test_canonical_source_whitespace_and_hash_are_preserved(
    result: FeatureResult,
) -> None:
    whitespace_rows = result.features[
        result.features["source_text"].map(lambda text: text != text.strip())
    ]
    expected_texts = {
        " At what point did Wallace diverge from Belford?",
        " Identified Belford's suspicious activities and potential involvement in the cyberattack.",
    }

    assert set(whitespace_rows["source_text"]) == expected_texts
    assert whitespace_rows["text_sha256"].tolist() == [
        sha256(text.encode("utf-8")).hexdigest()
        for text in whitespace_rows["source_text"]
    ]


def test_only_parser_provenance_changes_from_p2_archive(
    result: FeatureResult,
) -> None:
    archived_tables = ARCHIVED_ANALYSIS_ROOT / "tables"
    archived_features = pd.read_parquet(
        archived_tables / "features_deterministic.parquet"
    ).drop(columns="artifact_schema_version")
    archived_tokens = pd.read_parquet(
        archived_tables / "stanza_tokens.parquet"
    ).drop(columns="artifact_schema_version")
    archived_entities = pd.read_parquet(
        archived_tables / "spacy_entities.parquet"
    ).drop(columns="artifact_schema_version")

    pd.testing.assert_frame_equal(result.stanza_tokens, archived_tokens)
    pd.testing.assert_frame_equal(result.spacy_entities, archived_entities)

    changed_columns = {"parser_manifest_sha256"}
    unchanged_columns = [
        column for column in result.features if column not in changed_columns
    ]
    pd.testing.assert_frame_equal(
        result.features[unchanged_columns],
        archived_features[unchanged_columns],
    )

    assert result.features["parser_manifest_sha256"].nunique() == 1
    assert (
        result.features["parser_manifest_sha256"].iloc[0]
        != archived_features["parser_manifest_sha256"].iloc[0]
    )
    current_manifest = json.loads(
        (DEFAULT_ANALYSIS_ROOT / "parser_resource_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    archived_manifest = json.loads(
        (ARCHIVED_ANALYSIS_ROOT / "parser_resource_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert current_manifest["stanza"]["distribution_version"] == "1.14.0"
    assert current_manifest["spacy"]["distribution_version"] == "3.8.14"
    assert (
        current_manifest["stanza"]["model_files"]
        == archived_manifest["stanza"]["model_files"]
    )
    assert (
        current_manifest["spacy"]["model_files"]
        == archived_manifest["spacy"]["model_files"]
    )


def test_feature_artifact_bytes_are_stable(
    result: FeatureResult, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path)
    frames = {
        "features.parquet": result.features,
        "tokens.parquet": result.stanza_tokens,
        "entities.parquet": result.spacy_entities,
    }

    for name, frame in frames.items():
        first = store.write_parquet(
            f"first/{name}",
            versioned_frame(frame),
            created_by=WorkflowCommand.FEATURES,
        )
        second = store.write_parquet(
            f"second/{name}",
            versioned_frame(frame),
            created_by=WorkflowCommand.FEATURES,
        )
        assert first.sha256 == second.sha256
        assert first.size_bytes == second.size_bytes


def test_partial_report_contains_no_outcome_summary(result: FeatureResult) -> None:
    assert "Outcome columns loaded: none." in result.report
    assert "No feature-outcome association was computed." in result.report
    for forbidden in ("RubricV2", "ordinal grade", "pass rate", "tier"):
        assert forbidden not in result.report


def test_fresh_feature_runs_have_identical_artifact_hashes(tmp_path: Path) -> None:
    artifact_hashes: list[dict[str, str]] = []
    for name in ("first", "second"):
        workspace = AnalysisWorkspace.initialize(tmp_path / name)
        assert workspace.run_step(WorkflowCommand.JOIN, run_join)
        assert workspace.run_step(
            WorkflowCommand.FEATURES,
            lambda current: run_features(
                current,
                FeatureConfig(stanza_model_dir=DEFAULT_STANZA_MODEL_DIR),
            ),
        )
        state = workspace.load_state()
        artifact_hashes.append(
            {
                path: metadata.sha256
                for path, metadata in state.artifacts.items()
                if metadata.created_by is WorkflowCommand.FEATURES
            }
        )
        assert state.steps[WorkflowCommand.ANNOTATE_EMIT].status.value == "pending"
        assert not state.outcome_modeling_unlocked

    assert artifact_hashes[0] == artifact_hashes[1]
