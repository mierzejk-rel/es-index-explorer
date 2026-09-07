"""Tests for Segment 3 population and outcome-blind extraction."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from es_index_explorer.question_analysis.contracts import (
    FailureKind,
    StepStatus,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import GateFailureError
from es_index_explorer.question_analysis.features import (
    build_features,
    build_text_items,
)
from es_index_explorer.question_analysis.linguistics import (
    EntitySpan,
    ParsedSentence,
    ParsedWord,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

pytestmark = pytest.mark.unit


class FakeStanzaParser:
    def parse(self, text: str) -> tuple[ParsedSentence, ...]:
        first = text.split()[0]
        return (
            ParsedSentence(
                sentence_index=0,
                text=text,
                words=(
                    ParsedWord(
                        sentence_index=0,
                        word_id=1,
                        text=first,
                        lemma=first.casefold(),
                        upos="VERB",
                        xpos="",
                        feats=(("Mood", "Imp"),),
                        head=0,
                        deprel="root",
                        start_char=0,
                        end_char=len(first),
                    ),
                ),
            ),
        )


class FakeSpacyNer:
    def parse(self, text: str) -> tuple[EntitySpan, ...]:
        if "January 2012" not in text:
            return ()
        start = text.index("January 2012")
        return (
            EntitySpan(
                entity_index=0,
                text="January 2012",
                label="DATE",
                start_char=start,
                end_char=start + len("January 2012"),
            ),
        )


class FailingStanzaParser:
    def parse(self, text: str) -> tuple[ParsedSentence, ...]:
        raise RuntimeError(text)


class EmptyStanzaParser:
    def parse(self, text: str) -> tuple[ParsedSentence, ...]:
        del text
        return ()


class InvalidTreeStanzaParser:
    def parse(self, text: str) -> tuple[ParsedSentence, ...]:
        return (
            ParsedSentence(
                sentence_index=0,
                text=text,
                words=(
                    ParsedWord(
                        sentence_index=0,
                        word_id=1,
                        text="Invalid",
                        lemma="invalid",
                        upos="NOUN",
                        xpos="",
                        feats=(),
                        head=2,
                        deprel="dep",
                        start_char=0,
                        end_char=7,
                    ),
                    ParsedWord(
                        sentence_index=0,
                        word_id=2,
                        text="tree",
                        lemma="tree",
                        upos="NOUN",
                        xpos="",
                        feats=(),
                        head=1,
                        deprel="dep",
                        start_char=8,
                        end_char=12,
                    ),
                ),
            ),
        )


class FailingSpacyNer:
    def parse(self, text: str) -> tuple[EntitySpan, ...]:
        raise RuntimeError(text)


def test_text_population_preserves_grains_order_and_p1_values() -> None:
    rubrics, variants, expectations = _catalogues()

    result = build_text_items(rubrics, variants, expectations)

    assert result["item_id"].tolist() == ["variant-1", "expectation-1"]
    assert result["item_type"].tolist() == ["question", "expectation"]
    assert result["item_order"].tolist() == [0, 1]
    assert result["expectation_count"].tolist() == [1, 1]
    assert result["variant_count"].tolist() == [1, 1]
    assert result.iloc[1]["expectation_document_count"] == 2
    assert result["use_cases"].tolist() == [
        ["communications_analysis"],
        ["communications_analysis"],
    ]
    assert result["text_sha256"].str.len().eq(64).all()


def test_text_population_preserves_canonical_whitespace_and_hash() -> None:
    rubrics, variants, expectations = _catalogues()
    variants.loc[0, "question"] = " Question with source whitespace? "
    expectations.loc[0, "description"] = " Expectation with source whitespace. "

    result = build_text_items(rubrics, variants, expectations)

    assert result["source_text"].tolist() == [
        " Question with source whitespace? ",
        " Expectation with source whitespace. ",
    ]
    assert result["text_sha256"].tolist() == [
        sha256(text.encode("utf-8")).hexdigest()
        for text in result["source_text"]
    ]


def test_feature_build_is_deterministic_and_records_applicability() -> None:
    rubrics, variants, expectations = _catalogues()
    arguments = {
        "rubrics": rubrics,
        "variants": variants,
        "expectations": expectations,
        "stanza_parser": FakeStanzaParser(),
        "spacy_ner": FakeSpacyNer(),
        "parser_resource_manifest": {
            "schema_version": 1,
            "stanza": {"distribution_version": "test"},
            "spacy": {
                "distribution_version": "test",
                "model": "test",
                "model_version": "test",
            },
        },
    }

    first = build_features(**arguments)
    second = build_features(**arguments)

    pd.testing.assert_frame_equal(first.features, second.features)
    pd.testing.assert_frame_equal(first.stanza_tokens, second.stanza_tokens)
    assert first.verification == second.verification
    assert first.verification["unexpected_clause_types"] == []
    assert first.verification["invalid_subordinate_ratio_count"] == 0
    assert first.verification["invalid_complex_nominal_ratio_count"] == 0
    assert first.verification["applicability_failures"] == {}
    assert first.verification["parser_manifest_mismatch_count"] == 0
    assert first.features["temporal_expression_present"].tolist() == [True, False]
    assert first.features.iloc[0]["clause_type"] == "directive_imperative"
    assert pd.isna(first.features.iloc[1]["clause_type"])
    assert (
        '"expectation_document_count":"not_applicable_to_question"'
        in first.features.iloc[0]["missingness_reasons"]
    )
    assert (
        '"clause_type":"not_applicable_to_expectation"'
        in first.features.iloc[1]["missingness_reasons"]
    )


def test_outcome_column_in_catalogue_is_rejected() -> None:
    rubrics, variants, expectations = _catalogues()
    variants["rubric_v2"] = 1.0

    with pytest.raises(GateFailureError, match="forbidden columns"):
        build_text_items(rubrics, variants, expectations)


@pytest.mark.parametrize(
    ("stanza_parser", "spacy_ner", "expected_stage"),
    (
        (FailingStanzaParser(), FakeSpacyNer(), "stanza_parser_failed"),
        (EmptyStanzaParser(), FakeSpacyNer(), "stanza_empty_parse"),
        (FakeStanzaParser(), FailingSpacyNer(), "spacy_ner_failed"),
        (
            InvalidTreeStanzaParser(),
            FakeSpacyNer(),
            "deterministic_rule_failed",
        ),
    ),
)
def test_component_failure_is_a_bounded_gate_failure(
    stanza_parser: object,
    spacy_ner: object,
    expected_stage: str,
) -> None:
    rubrics, variants, expectations = _catalogues()

    with pytest.raises(GateFailureError) as captured:
        build_features(
            rubrics=rubrics,
            variants=variants,
            expectations=expectations,
            stanza_parser=stanza_parser,  # type: ignore[arg-type]
            spacy_ner=spacy_ner,  # type: ignore[arg-type]
            parser_resource_manifest={"schema_version": 1},
        )

    assert str(captured.value) == f"{expected_stage} item_id=variant-1"
    assert "List events from January 2012." not in str(captured.value)


def test_component_failure_is_persisted_in_workflow_state(tmp_path: Path) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# Specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)

    def complete_join(current: AnalysisWorkspace) -> tuple[()]:
        del current
        return ()

    workspace.run_step(WorkflowCommand.JOIN, complete_join)
    rubrics, variants, expectations = _catalogues()

    def fail_features(current: AnalysisWorkspace) -> tuple[()]:
        del current
        build_features(
            rubrics=rubrics,
            variants=variants,
            expectations=expectations,
            stanza_parser=FailingStanzaParser(),
            spacy_ner=FakeSpacyNer(),
            parser_resource_manifest={"schema_version": 1},
        )
        return ()

    with pytest.raises(GateFailureError):
        workspace.run_step(WorkflowCommand.FEATURES, fail_features)

    state = workspace.load_state()
    failure = state.steps[WorkflowCommand.FEATURES].failure
    assert state.steps[WorkflowCommand.FEATURES].status is StepStatus.FAILED
    assert failure is not None
    assert failure.kind is FailureKind.GATE_FAILURE
    assert failure.message == "stanza_parser_failed item_id=variant-1"
    assert not state.artifacts


def test_feature_module_names_only_catalogue_input_paths() -> None:
    source = (
        Path(__file__).parents[3]
        / "es_index_explorer"
        / "question_analysis"
        / "features.py"
    ).read_text(encoding="utf-8")

    assert "tables/trace_pfu_table.parquet" not in source
    assert "tables/criterion_table.parquet" not in source
    assert {
        "tables/rubric_catalogue.parquet",
        "tables/variant_catalogue.parquet",
        "tables/expectation_catalogue.parquet",
    } <= set(source.split('"'))


def _catalogues() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rubrics = pd.DataFrame(
        [
            {
                "rubric_id": "rubric-1",
                "expectation_count": 1,
                "variant_count": 1,
            }
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "variant_id": "variant-1",
                "rubric_id": "rubric-1",
                "rubric_order": 0,
                "eval_dataset": "emc2_set1",
                "source_path": "set_1/001.rubric.toml",
                "variant_index": 0,
                "question": "List events from January 2012.",
                "expectation_count": 1,
                "variant_count": 1,
                "use_cases": ["communications_analysis"],
            }
        ]
    )
    expectations = pd.DataFrame(
        [
            {
                "expectation_id": "expectation-1",
                "rubric_id": "rubric-1",
                "rubric_order": 0,
                "eval_dataset": "emc2_set1",
                "source_path": "set_1/001.rubric.toml",
                "expectation_index": 0,
                "description": "Identify the Phantom Phreak alias and email address.",
                "expectation_document_count": 2,
                "use_cases": ["communications_analysis"],
            }
        ]
    )
    return rubrics, variants, expectations
