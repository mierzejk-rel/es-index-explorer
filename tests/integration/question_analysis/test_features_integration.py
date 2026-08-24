"""Integration tests for deterministic Segment 3 extraction."""

from pathlib import Path

import pandas as pd
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
    assert result.stanza_tokens["item_id"].nunique() == 577
    stanza_manifest = result.parser_resource_manifest["stanza"]
    assert isinstance(stanza_manifest, dict)
    assert stanza_manifest["resource_catalogue"]["sha256"] == (
        "4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c"
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
