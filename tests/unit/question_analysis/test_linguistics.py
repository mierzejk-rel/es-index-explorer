"""Tests for deterministic UD and semantic rules."""

import pytest

from es_index_explorer.question_analysis.linguistics import (
    EntitySpan,
    ParsedSentence,
    ParsedWord,
    classify_clause_type,
    extract_linguistic_features,
)

pytestmark = pytest.mark.unit


def _sentence(text: str, *words: ParsedWord) -> ParsedSentence:
    return ParsedSentence(sentence_index=0, text=text, words=tuple(words))


def _word(
    word_id: int,
    text: str,
    upos: str,
    head: int,
    deprel: str,
    **feats: str,
) -> ParsedWord:
    return ParsedWord(
        sentence_index=0,
        word_id=word_id,
        text=text,
        lemma=text.casefold(),
        upos=upos,
        xpos="",
        feats=tuple(sorted(feats.items())),
        head=head,
        deprel=deprel,
        start_char=None,
        end_char=None,
    )


@pytest.mark.parametrize(
    ("sentence", "expected"),
    (
        (
            _sentence(
                "Which documents support the claim?",
                _word(1, "Which", "DET", 2, "det", PronType="Int"),
                _word(2, "documents", "NOUN", 3, "nsubj"),
                _word(3, "support", "VERB", 0, "root", VerbForm="Fin"),
            ),
            "open_interrogative",
        ),
        (
            _sentence(
                "Did Belford send an email?",
                _word(1, "Did", "AUX", 3, "aux"),
                _word(2, "Belford", "PROPN", 3, "nsubj"),
                _word(3, "send", "VERB", 0, "root", VerbForm="Fin"),
            ),
            "closed_interrogative",
        ),
        (
            _sentence(
                "List the supporting documents.",
                _word(1, "List", "VERB", 0, "root", Mood="Imp", VerbForm="Fin"),
                _word(2, "documents", "NOUN", 1, "obj"),
            ),
            "directive_imperative",
        ),
        (
            _sentence(
                "I would like a summary.",
                _word(1, "I", "PRON", 3, "nsubj"),
                _word(2, "would", "AUX", 3, "aux"),
                _word(3, "like", "VERB", 0, "root", VerbForm="Fin"),
            ),
            "declarative_request",
        ),
    ),
)
def test_clause_type_rules(
    sentence: ParsedSentence,
    expected: str,
) -> None:
    assert classify_clause_type(sentence) == expected


def test_ud_metrics_cover_subordination_coordination_and_complex_nominals() -> None:
    sentence = _sentence(
        "The senior analysts reviewed records and summarized findings because evidence emerged.",
        _word(1, "The", "DET", 3, "det"),
        _word(2, "senior", "ADJ", 3, "amod"),
        _word(3, "analysts", "NOUN", 4, "nsubj"),
        _word(4, "reviewed", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
        _word(5, "records", "NOUN", 4, "obj"),
        _word(6, "summarized", "VERB", 4, "conj", Mood="Ind", VerbForm="Fin"),
        _word(7, "findings", "NOUN", 6, "obj"),
        _word(8, "evidence", "NOUN", 9, "nsubj"),
        _word(9, "emerged", "VERB", 4, "advcl", Mood="Ind", VerbForm="Fin"),
    )
    entities = (
        EntitySpan(
            entity_index=0,
            text="January 2012",
            label="DATE",
            start_char=0,
            end_char=12,
        ),
    )

    result = extract_linguistic_features(
        "question", sentence.text, (sentence,), entities
    )

    assert result.token_count == 9
    assert result.dependency_tree_depth == 2
    assert result.mean_dependency_length == pytest.approx(1.75)
    assert result.clause_count == 3
    assert result.subordinate_clause_ratio == pytest.approx(1 / 3)
    assert result.complex_nominals_per_clause == pytest.approx(1 / 3)
    assert result.coordination_count == 1
    assert result.named_entity_count == 1
    assert result.temporal_expression_present


def test_fragment_records_clause_ratio_missingness() -> None:
    sentence = _sentence(
        "Emails with Dr. Argoff?",
        _word(1, "Emails", "NOUN", 0, "root"),
        _word(2, "Argoff", "PROPN", 1, "nmod"),
    )

    result = extract_linguistic_features(
        "expectation", sentence.text, (sentence,), ()
    )

    assert result.clause_count == 0
    assert result.subordinate_clause_ratio is None
    assert result.complex_nominals_per_clause is None
    assert result.clause_type is None
    assert dict(result.missingness) == {
        "clause_type": "not_applicable_to_expectation",
        "complex_nominals_per_clause": "no_clause",
        "subordinate_clause_ratio": "no_clause",
    }


def test_alias_and_email_are_counted_only_when_spacy_returns_entities() -> None:
    sentence = _sentence(
        "List messages from Phantom Phreak at phantom@example.com.",
        _word(1, "List", "VERB", 0, "root", Mood="Imp", VerbForm="Fin"),
        _word(2, "messages", "NOUN", 1, "obj"),
        _word(3, "Phantom", "PROPN", 1, "obl"),
        _word(4, "Phreak", "PROPN", 3, "flat"),
        _word(5, "phantom@example.com", "X", 1, "obl"),
    )
    entities = (
        EntitySpan(
            entity_index=0,
            text="Phantom Phreak",
            label="PERSON",
            start_char=19,
            end_char=33,
        ),
    )

    result = extract_linguistic_features(
        "question", sentence.text, (sentence,), entities
    )

    assert result.named_entity_count == 1
    assert not result.temporal_expression_present
    assert result.clause_type == "directive_imperative"
