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


def _sentence(
    text: str, *words: ParsedWord, sentence_index: int = 0
) -> ParsedSentence:
    return ParsedSentence(sentence_index=sentence_index, text=text, words=tuple(words))


def _word(
    word_id: int,
    text: str,
    upos: str,
    head: int,
    deprel: str,
    sentence_index: int = 0,
    start_char: int | None = None,
    end_char: int | None = None,
    **feats: str,
) -> ParsedWord:
    return ParsedWord(
        sentence_index=sentence_index,
        word_id=word_id,
        text=text,
        lemma=text.casefold(),
        upos=upos,
        xpos="",
        feats=tuple(sorted(feats.items())),
        head=head,
        deprel=deprel,
        start_char=start_char,
        end_char=end_char,
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
    assert classify_clause_type(sentence.text, (sentence,)) == expected


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


def test_finite_matrix_copula_counts_as_clause() -> None:
    sentence = _sentence(
        "Who is Elmo?",
        _word(1, "Who", "PRON", 0, "root", PronType="Int"),
        _word(2, "is", "AUX", 1, "cop", Mood="Ind", VerbForm="Fin"),
        _word(3, "Elmo", "PROPN", 1, "nsubj"),
    )

    result = extract_linguistic_features(
        "question", sentence.text, (sentence,), ()
    )

    assert result.clause_count == 1
    assert result.subordinate_clause_ratio == 0
    assert result.complex_nominals_per_clause == 0
    assert "subordinate_clause_ratio" not in dict(result.missingness)


def test_subordinate_copula_inherits_predicate_relation() -> None:
    sentence = _sentence(
        "She was happy because he was ready.",
        _word(1, "She", "PRON", 3, "nsubj"),
        _word(2, "was", "AUX", 3, "cop", Mood="Ind", VerbForm="Fin"),
        _word(3, "happy", "ADJ", 0, "root"),
        _word(4, "he", "PRON", 6, "nsubj"),
        _word(5, "was", "AUX", 6, "cop", Mood="Ind", VerbForm="Fin"),
        _word(6, "ready", "ADJ", 3, "advcl"),
    )

    result = extract_linguistic_features(
        "expectation", sentence.text, (sentence,), ()
    )

    assert result.clause_count == 2
    assert result.subordinate_clause_ratio == pytest.approx(0.5)


def test_copula_does_not_duplicate_existing_verbal_clause() -> None:
    sentence = _sentence(
        "What is happening?",
        _word(1, "What", "PRON", 3, "nsubj", PronType="Int"),
        _word(2, "is", "AUX", 3, "cop", Mood="Ind", VerbForm="Fin"),
        _word(3, "happening", "VERB", 0, "root", VerbForm="Part"),
    )

    result = extract_linguistic_features(
        "question", sentence.text, (sentence,), ()
    )

    assert result.clause_count == 1
    assert result.subordinate_clause_ratio == 0


def test_nonfinite_copula_does_not_create_clause() -> None:
    sentence = _sentence(
        "Being ready.",
        _word(1, "Being", "AUX", 2, "cop", VerbForm="Ger"),
        _word(2, "ready", "ADJ", 0, "root"),
    )

    result = extract_linguistic_features(
        "expectation", sentence.text, (sentence,), ()
    )

    assert result.clause_count == 0
    assert result.subordinate_clause_ratio is None
    assert dict(result.missingness)["subordinate_clause_ratio"] == "no_clause"


def test_auxiliary_root_is_not_duplicated() -> None:
    sentence = _sentence(
        "Was Belford present?",
        _word(1, "Was", "AUX", 0, "root", Mood="Ind", VerbForm="Fin"),
        _word(2, "Belford", "PROPN", 1, "nsubj"),
        _word(3, "present", "ADJ", 1, "xcomp"),
    )

    result = extract_linguistic_features(
        "question", sentence.text, (sentence,), ()
    )

    assert result.clause_count == 1
    assert result.clause_type == "closed_interrogative"


def test_email_boundary_is_coalesced_for_polar_clause_type() -> None:
    source_text = "Did eugene.Belford@example.com plan it?"
    sentences = (
        _sentence(
            "Did eugene.",
            _word(
                1,
                "Did",
                "AUX",
                0,
                "root",
                start_char=0,
                end_char=3,
                Mood="Ind",
                VerbForm="Fin",
            ),
            _word(
                2,
                "eugene.",
                "NOUN",
                1,
                "obj",
                start_char=4,
                end_char=11,
            ),
        ),
        _sentence(
            "Belford@example.com plan it?",
            _word(
                1,
                "Belford@example.com",
                "PROPN",
                2,
                "nsubj",
                sentence_index=1,
                start_char=11,
                end_char=30,
            ),
            _word(
                2,
                "plan",
                "VERB",
                0,
                "root",
                sentence_index=1,
                start_char=31,
                end_char=35,
            ),
            _word(
                3,
                "?",
                "PUNCT",
                2,
                "punct",
                sentence_index=1,
                start_char=38,
                end_char=39,
            ),
            sentence_index=1,
        ),
    )

    assert classify_clause_type(source_text, sentences) == "closed_interrogative"


def test_email_boundary_preserves_wh_precedence() -> None:
    source_text = "Who emailed eugene.Belford@example.com?"
    sentences = (
        _sentence(
            "Who emailed eugene.",
            _word(
                1,
                "Who",
                "PRON",
                2,
                "nsubj",
                start_char=0,
                end_char=3,
                PronType="Int",
            ),
            _word(
                2,
                "emailed",
                "VERB",
                0,
                "root",
                start_char=4,
                end_char=11,
            ),
            _word(
                3,
                "eugene.",
                "NOUN",
                2,
                "obj",
                start_char=12,
                end_char=19,
            ),
        ),
        _sentence(
            "Belford@example.com?",
            _word(
                1,
                "Belford@example.com",
                "PROPN",
                0,
                "root",
                sentence_index=1,
                start_char=19,
                end_char=38,
            ),
            _word(
                2,
                "?",
                "PUNCT",
                1,
                "punct",
                sentence_index=1,
                start_char=38,
                end_char=39,
            ),
            sentence_index=1,
        ),
    )

    assert classify_clause_type(source_text, sentences) == "open_interrogative"


def test_genuine_second_sentence_does_not_change_clause_type() -> None:
    source_text = "Tell me this. Is it clear?"
    sentences = (
        _sentence(
            "Tell me this.",
            _word(
                1,
                "Tell",
                "VERB",
                0,
                "root",
                start_char=0,
                end_char=4,
                Mood="Imp",
                VerbForm="Fin",
            ),
            _word(2, ".", "PUNCT", 1, "punct", start_char=12, end_char=13),
        ),
        _sentence(
            "Is it clear?",
            _word(
                1,
                "Is",
                "AUX",
                3,
                "cop",
                sentence_index=1,
                start_char=14,
                end_char=16,
                Mood="Ind",
                VerbForm="Fin",
            ),
            _word(
                2,
                "it",
                "PRON",
                3,
                "nsubj",
                sentence_index=1,
                start_char=17,
                end_char=19,
            ),
            _word(
                3,
                "clear",
                "ADJ",
                0,
                "root",
                sentence_index=1,
                start_char=20,
                end_char=25,
            ),
            sentence_index=1,
        ),
    )

    assert classify_clause_type(source_text, sentences) == "directive_imperative"


def test_nominal_list_parser_error_is_repaired() -> None:
    sentence = _sentence(
        "List documents supporting the claim.",
        _word(1, "List", "NOUN", 2, "compound"),
        _word(2, "documents", "NOUN", 0, "root"),
        _word(3, "supporting", "VERB", 2, "acl", VerbForm="Ger"),
    )

    assert (
        classify_clause_type(sentence.text, (sentence,))
        == "directive_imperative"
    )


def test_nominal_list_repair_requires_subjectless_nominal_root() -> None:
    sentence = _sentence(
        "List documents exist.",
        _word(1, "List", "NOUN", 2, "compound"),
        _word(2, "documents", "NOUN", 3, "nsubj"),
        _word(3, "exist", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "declarative_request"


def test_matrix_imperative_precedes_embedded_wh() -> None:
    sentence = _sentence(
        "List documents showing what happened?",
        _word(1, "List", "VERB", 0, "root", Mood="Imp", VerbForm="Fin"),
        _word(2, "documents", "NOUN", 1, "obj"),
        _word(3, "what", "PRON", 4, "nsubj", PronType="Int"),
        _word(4, "happened", "VERB", 2, "acl", Mood="Ind", VerbForm="Fin"),
    )

    assert (
        classify_clause_type(sentence.text, (sentence,))
        == "directive_imperative"
    )


@pytest.mark.parametrize("root_text", ("List", "Find", "Explain", "Tell"))
def test_observed_request_roots_precede_embedded_interrogative(
    root_text: str,
) -> None:
    sentence = _sentence(
        f"{root_text} me what happened.",
        _word(1, root_text, "VERB", 0, "root", Mood="Imp", VerbForm="Fin"),
        _word(2, "me", "PRON", 1, "iobj"),
        _word(3, "what", "PRON", 4, "nsubj", PronType="Int"),
        _word(4, "happened", "VERB", 1, "ccomp", Mood="Ind", VerbForm="Fin"),
    )

    assert (
        classify_clause_type(sentence.text, (sentence,))
        == "directive_imperative"
    )


def test_direct_matrix_interrogative_is_open() -> None:
    sentence = _sentence(
        "What happened?",
        _word(1, "What", "PRON", 2, "nsubj", PronType="Int"),
        _word(2, "happened", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "open_interrogative"


def test_preposition_fronted_matrix_interrogative_is_open() -> None:
    sentence = _sentence(
        "With whom did Belford communicate?",
        _word(1, "With", "ADP", 2, "case"),
        _word(2, "whom", "PRON", 4, "obl", PronType="Int"),
        _word(3, "Belford", "PROPN", 4, "nsubj"),
        _word(4, "communicate", "VERB", 0, "root", VerbForm="Inf"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "open_interrogative"


def test_adjunct_fronted_matrix_interrogative_is_open() -> None:
    sentence = _sentence(
        "After the attack, what actions followed?",
        _word(1, "After", "ADP", 3, "case"),
        _word(2, "the", "DET", 3, "det"),
        _word(3, "attack", "NOUN", 6, "obl"),
        _word(4, "what", "DET", 5, "det", PronType="Int"),
        _word(5, "actions", "NOUN", 6, "nsubj"),
        _word(6, "followed", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "open_interrogative"


def test_declarative_matrix_with_embedded_interrogative_is_not_open() -> None:
    sentence = _sentence(
        "I wonder who left.",
        _word(1, "I", "PRON", 2, "nsubj"),
        _word(2, "wonder", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
        _word(3, "who", "PRON", 4, "nsubj", PronType="Int"),
        _word(4, "left", "VERB", 2, "ccomp", Mood="Ind", VerbForm="Fin"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "declarative_request"


def test_coordinated_matrix_interrogative_is_open() -> None:
    sentence = _sentence(
        "Who arrived and what happened?",
        _word(1, "Who", "PRON", 2, "nsubj", PronType="Int"),
        _word(2, "arrived", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
        _word(3, "what", "PRON", 4, "nsubj", PronType="Int"),
        _word(4, "happened", "VERB", 2, "conj", Mood="Ind", VerbForm="Fin"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "open_interrogative"


def test_subject_auxiliary_inversion_is_closed_without_punctuation() -> None:
    sentence = _sentence(
        "Did Belford communicate",
        _word(1, "Did", "AUX", 3, "aux", Mood="Ind", VerbForm="Fin"),
        _word(2, "Belford", "PROPN", 3, "nsubj"),
        _word(3, "communicate", "VERB", 0, "root", VerbForm="Inf"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "closed_interrogative"


def test_non_matrix_auxiliary_does_not_imply_inversion() -> None:
    sentence = _sentence(
        "Being ready matters.",
        _word(1, "Being", "AUX", 2, "cop", VerbForm="Ger"),
        _word(2, "ready", "ADJ", 3, "csubj"),
        _word(3, "matters", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "declarative_request"


def test_auxiliary_with_non_matrix_relation_does_not_imply_inversion() -> None:
    sentence = _sentence(
        "Being teams work.",
        _word(1, "Being", "AUX", 3, "mark", VerbForm="Ger"),
        _word(2, "teams", "NOUN", 3, "nsubj"),
        _word(3, "work", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "declarative_request"


@pytest.mark.parametrize(
    "words",
    (
        (
            _word(1, "Who", "PRON", 2, "obj", PronType="Int"),
            _word(2, "left", "VERB", 1, "dep", Mood="Ind", VerbForm="Fin"),
        ),
        (
            _word(1, "Who", "PRON", 9, "obj", PronType="Int"),
            _word(2, "left", "VERB", 0, "root", Mood="Ind", VerbForm="Fin"),
        ),
    ),
)
def test_invalid_interrogative_dependency_path_is_rejected(
    words: tuple[ParsedWord, ...],
) -> None:
    sentence = _sentence("Who left.", *words)

    with pytest.raises(ValueError, match="Invalid interrogative dependency path"):
        classify_clause_type(sentence.text, (sentence,))


def test_initial_wh_surface_form_does_not_require_pron_type() -> None:
    sentence = _sentence(
        "How to proceed",
        _word(1, "How", "ADV", 3, "advmod"),
        _word(2, "to", "PART", 3, "mark"),
        _word(3, "proceed", "VERB", 0, "root", VerbForm="Inf"),
    )

    assert classify_clause_type(sentence.text, (sentence,)) == "open_interrogative"


def test_clause_classification_skips_empty_leading_sentence() -> None:
    empty = _sentence("", sentence_index=0)
    question = _sentence(
        "Did it happen?",
        _word(1, "Did", "AUX", 3, "aux", sentence_index=1),
        _word(2, "it", "PRON", 3, "nsubj", sentence_index=1),
        _word(
            3,
            "happen",
            "VERB",
            0,
            "root",
            sentence_index=1,
            VerbForm="Fin",
        ),
        sentence_index=1,
    )

    assert (
        classify_clause_type(question.text, (empty, question))
        == "closed_interrogative"
    )


def test_invalid_or_empty_parse_is_rejected() -> None:
    with pytest.raises(ValueError, match="parsed, non-empty text"):
        extract_linguistic_features("question", "", (), ())
    with pytest.raises(ValueError, match="at least one sentence"):
        classify_clause_type("Question?", ())
    with pytest.raises(ValueError, match="non-empty sentence"):
        classify_clause_type("Question?", (_sentence(""),))


def test_invalid_copular_dependency_is_rejected() -> None:
    sentence = _sentence(
        "Is ready.",
        _word(1, "Is", "AUX", 9, "cop", Mood="Ind", VerbForm="Fin"),
        _word(2, "ready", "ADJ", 0, "root"),
    )

    with pytest.raises(ValueError, match="Invalid copular dependency"):
        extract_linguistic_features(
            "expectation", sentence.text, (sentence,), ()
        )


def test_single_root_has_no_dependency_length() -> None:
    sentence = _sentence("Evidence", _word(1, "Evidence", "NOUN", 0, "root"))

    result = extract_linguistic_features(
        "expectation", sentence.text, (sentence,), ()
    )

    assert result.mean_dependency_length is None
    assert dict(result.missingness)["mean_dependency_length"] == "no_dependencies"


def test_email_boundary_without_offsets_is_not_guessed() -> None:
    source_text = "Did eugene.Belford@example.com plan it?"
    sentences = (
        _sentence(
            "Did eugene.",
            _word(1, "Did", "AUX", 0, "root", Mood="Ind", VerbForm="Fin"),
            _word(2, "eugene.", "NOUN", 1, "obj"),
        ),
        _sentence(
            "Belford@example.com plan it?",
            _word(
                1,
                "Belford@example.com",
                "PROPN",
                2,
                "nsubj",
                sentence_index=1,
            ),
            _word(2, "plan", "VERB", 0, "root", sentence_index=1),
            sentence_index=1,
        ),
    )

    assert (
        classify_clause_type(source_text, sentences) == "directive_imperative"
    )


@pytest.mark.parametrize(
    "sentence",
    (
        _sentence(".", _word(1, ".", "PUNCT", 0, "root")),
        _sentence("Evidence", _word(1, "Evidence", "NOUN", 9, "dep")),
    ),
)
def test_unclassifiable_non_question_sentence_uses_remainder(
    sentence: ParsedSentence,
) -> None:
    assert classify_clause_type(sentence.text, (sentence,)) == "declarative_request"


def test_invalid_dependency_tree_is_rejected() -> None:
    sentence = _sentence(
        "Cycle",
        _word(1, "Cycle", "NOUN", 2, "dep"),
        _word(2, "here", "ADV", 1, "advmod"),
    )

    with pytest.raises(ValueError, match="Invalid dependency tree"):
        extract_linguistic_features(
            "expectation", sentence.text, (sentence,), ()
        )
