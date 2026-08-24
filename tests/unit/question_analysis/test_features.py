"""Tests for Segment 3 population and outcome-blind extraction."""

from pathlib import Path

import pandas as pd
import pytest

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
