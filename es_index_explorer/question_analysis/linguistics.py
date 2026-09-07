"""Pure deterministic linguistic feature rules."""

import re
from dataclasses import dataclass
from typing import Literal

ItemType = Literal["question", "expectation"]
ClauseType = Literal[
    "open_interrogative",
    "closed_interrogative",
    "directive_imperative",
    "declarative_request",
]

PUNCTUATION_UPOS = "PUNCT"
CLAUSAL_RELATIONS = frozenset({"advcl", "acl", "ccomp", "xcomp", "csubj", "parataxis"})
SUBORDINATE_RELATIONS = frozenset({"advcl", "acl", "ccomp", "xcomp", "csubj"})
COMPLEX_NOMINAL_RELATIONS = frozenset(
    {"amod", "appos", "compound", "nmod", "nummod", "acl"}
)
SUBJECT_RELATIONS = frozenset({"nsubj", "csubj", "expl"})
WH_FORMS = frozenset(
    {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"}
)
EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}"
)


@dataclass(frozen=True, slots=True)
class ParsedWord:
    """One Stanza UD word."""

    sentence_index: int
    word_id: int
    text: str
    lemma: str
    upos: str
    xpos: str
    feats: tuple[tuple[str, str], ...]
    head: int
    deprel: str
    start_char: int | None
    end_char: int | None

    @property
    def base_relation(self) -> str:
        return self.deprel.partition(":")[0]

    def feature(self, name: str) -> str | None:
        return next((value for key, value in self.feats if key == name), None)


@dataclass(frozen=True, slots=True)
class ParsedSentence:
    """One sentence and its UD words."""

    sentence_index: int
    text: str
    words: tuple[ParsedWord, ...]


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """One spaCy NER span."""

    entity_index: int
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class _Clause:
    predicate_id: int
    effective_relation: str


@dataclass(frozen=True, slots=True)
class LinguisticFeatures:
    """P2/P3 values and explicit deterministic missingness."""

    token_count: int
    dependency_tree_depth: int
    mean_dependency_length: float | None
    clause_count: int
    subordinate_clause_ratio: float | None
    complex_nominals_per_clause: float | None
    coordination_count: int
    named_entity_count: int
    temporal_expression_present: bool
    clause_type: ClauseType | None
    missingness: tuple[tuple[str, str], ...]


def extract_linguistic_features(
    item_type: ItemType,
    source_text: str,
    sentences: tuple[ParsedSentence, ...],
    entities: tuple[EntitySpan, ...],
) -> LinguisticFeatures:
    """Compute the frozen P2/P3 features."""
    if not source_text.strip() or not sentences:
        raise ValueError(
            "Deterministic linguistic extraction requires parsed, non-empty text"
        )

    words = tuple(word for sentence in sentences for word in sentence.words)
    content_words = tuple(word for word in words if word.upos != PUNCTUATION_UPOS)
    dependency_lengths = tuple(
        abs(word.word_id - word.head) for word in content_words if word.head != 0
    )
    clauses = tuple(
        clause for sentence in sentences for clause in _clauses(sentence.words)
    )
    clause_count = len(clauses)
    subordinate_count = sum(
        clause.effective_relation in SUBORDINATE_RELATIONS for clause in clauses
    )
    complex_nominal_count = sum(
        _is_complex_nominal(word, sentence.words)
        for sentence in sentences
        for word in sentence.words
    )
    missingness: list[tuple[str, str]] = []
    if dependency_lengths:
        mean_dependency_length = sum(dependency_lengths) / len(dependency_lengths)
    else:
        mean_dependency_length = None
        missingness.append(("mean_dependency_length", "no_dependencies"))
    if clause_count:
        subordinate_clause_ratio = subordinate_count / clause_count
        complex_nominals_per_clause = complex_nominal_count / clause_count
    else:
        subordinate_clause_ratio = None
        complex_nominals_per_clause = None
        missingness.extend(
            (
                ("subordinate_clause_ratio", "no_clause"),
                ("complex_nominals_per_clause", "no_clause"),
            )
        )

    if item_type == "question":
        clause_type: ClauseType | None = classify_clause_type(source_text, sentences)
    else:
        clause_type = None
        missingness.append(("clause_type", "not_applicable_to_expectation"))

    return LinguisticFeatures(
        token_count=len(content_words),
        dependency_tree_depth=max(
            (_tree_depth(sentence.words) for sentence in sentences), default=0
        ),
        mean_dependency_length=mean_dependency_length,
        clause_count=clause_count,
        subordinate_clause_ratio=subordinate_clause_ratio,
        complex_nominals_per_clause=complex_nominals_per_clause,
        coordination_count=sum(
            word.base_relation == "conj" and word.upos != PUNCTUATION_UPOS
            for word in words
        ),
        named_entity_count=len(entities),
        temporal_expression_present=any(
            entity.label in {"DATE", "TIME"} for entity in entities
        ),
        clause_type=clause_type,
        missingness=tuple(sorted(missingness)),
    )


def classify_clause_type(
    source_text: str, sentences: tuple[ParsedSentence, ...]
) -> ClauseType:
    """Apply the frozen first-logical-sentence clause-type precedence."""
    sentence = _first_logical_sentence(source_text, sentences)
    content_words = tuple(
        word for word in sentence.words if word.upos != PUNCTUATION_UPOS
    )
    root = next((word for word in content_words if word.head == 0), None)
    if _is_matrix_imperative(content_words, root):
        return "directive_imperative"
    if any(
        word.feature("PronType") == "Int"
        and _is_matrix_interrogative(word, content_words)
        for word in content_words
    ):
        return "open_interrogative"
    if content_words and content_words[0].text.casefold() in WH_FORMS:
        return "open_interrogative"
    if sentence.text.rstrip().endswith("?") or _has_subject_auxiliary_inversion(
        content_words, root
    ):
        return "closed_interrogative"

    if (
        root is not None
        and root.upos in {"VERB", "AUX"}
        and not _has_subject(root, content_words)
    ):
        return "directive_imperative"
    return "declarative_request"


def _is_matrix_imperative(
    content_words: tuple[ParsedWord, ...], root: ParsedWord | None
) -> bool:
    return (
        root is not None
        and root.feature("Mood") == "Imp"
        or _is_nominal_list_imperative(content_words, root)
    )


def _is_matrix_interrogative(
    interrogative: ParsedWord, content_words: tuple[ParsedWord, ...]
) -> bool:
    by_key = {(word.sentence_index, word.word_id): word for word in content_words}
    seen: set[tuple[int, int]] = set()
    current = interrogative
    while current.head:
        key = (current.sentence_index, current.word_id)
        if key in seen:
            raise ValueError("Invalid interrogative dependency path")
        if current.base_relation in SUBORDINATE_RELATIONS:
            return False
        seen.add(key)
        parent = by_key.get((current.sentence_index, current.head))
        if parent is None:
            raise ValueError("Invalid interrogative dependency path")
        current = parent
    return current.base_relation not in SUBORDINATE_RELATIONS


def _has_subject_auxiliary_inversion(
    content_words: tuple[ParsedWord, ...], root: ParsedWord | None
) -> bool:
    if not content_words or root is None:
        return False
    auxiliary = content_words[0]
    if auxiliary.upos != "AUX":
        return False
    if auxiliary.head == 0:
        predicate = auxiliary
    elif auxiliary.base_relation in {"aux", "cop"}:
        predicate = next(
            (
                word
                for word in content_words
                if word.sentence_index == auxiliary.sentence_index
                and word.word_id == auxiliary.head
                and word.head == 0
            ),
            None,
        )
        if predicate is None:
            return False
    else:
        return False
    return _has_subject(predicate, content_words)


def _has_subject(predicate: ParsedWord, content_words: tuple[ParsedWord, ...]) -> bool:
    return any(
        word.sentence_index == predicate.sentence_index
        and word.head == predicate.word_id
        and word.base_relation in SUBJECT_RELATIONS
        for word in content_words
    )


def _clauses(words: tuple[ParsedWord, ...]) -> tuple[_Clause, ...]:
    by_id = {word.word_id: word for word in words}
    clauses = {
        word.word_id: _Clause(
            predicate_id=word.word_id,
            effective_relation=word.base_relation,
        )
        for word in words
        if _is_standard_clause_head(word)
    }
    for copula in words:
        if not _is_finite_copula(copula):
            continue
        predicate = by_id.get(copula.head)
        if predicate is None:
            raise ValueError("Invalid copular dependency")
        clauses.setdefault(
            predicate.word_id,
            _Clause(
                predicate_id=predicate.word_id,
                effective_relation=predicate.base_relation,
            ),
        )
    return tuple(clauses.values())


def _is_standard_clause_head(word: ParsedWord) -> bool:
    if word.upos not in {"VERB", "AUX"}:
        return False
    if word.head == 0:
        return True
    if word.base_relation in CLAUSAL_RELATIONS:
        return True
    return word.base_relation == "conj" and (
        word.feature("VerbForm") == "Fin" or word.feature("Mood") is not None
    )


def _is_finite_copula(word: ParsedWord) -> bool:
    return (
        word.upos == "AUX"
        and word.base_relation == "cop"
        and (word.feature("VerbForm") == "Fin" or word.feature("Mood") is not None)
    )


def _first_logical_sentence(
    source_text: str, sentences: tuple[ParsedSentence, ...]
) -> ParsedSentence:
    if not sentences:
        raise ValueError("Clause classification requires at least one sentence")
    first_index = next(
        (
            index
            for index, sentence in enumerate(sentences)
            if any(word.text.strip() for word in sentence.words)
        ),
        None,
    )
    if first_index is None:
        raise ValueError("Clause classification requires a non-empty sentence")

    logical_sentences = [sentences[first_index]]
    email_spans = tuple(match.span() for match in EMAIL_PATTERN.finditer(source_text))
    for sentence in sentences[first_index + 1 :]:
        if not _boundary_is_inside_email(logical_sentences[-1], sentence, email_spans):
            break
        logical_sentences.append(sentence)

    words = tuple(
        word
        for logical_sentence in logical_sentences
        for word in logical_sentence.words
    )
    starts = [word.start_char for word in words if word.start_char is not None]
    ends = [word.end_char for word in words if word.end_char is not None]
    text = (
        source_text[min(starts) : max(ends)]
        if starts and ends
        else " ".join(sentence.text for sentence in logical_sentences)
    )
    return ParsedSentence(
        sentence_index=logical_sentences[0].sentence_index,
        text=text,
        words=words,
    )


def _boundary_is_inside_email(
    left: ParsedSentence,
    right: ParsedSentence,
    email_spans: tuple[tuple[int, int], ...],
) -> bool:
    left_ends = [word.end_char for word in left.words if word.end_char is not None]
    right_starts = [
        word.start_char for word in right.words if word.start_char is not None
    ]
    if not left_ends or not right_starts:
        return False
    left_end = max(left_ends)
    right_start = min(right_starts)
    return any(
        span_start < left_end <= right_start < span_end
        for span_start, span_end in email_spans
    )


def _is_nominal_list_imperative(
    content_words: tuple[ParsedWord, ...], root: ParsedWord | None
) -> bool:
    if not content_words or root is None:
        return False
    first = content_words[0]
    if not (
        first.lemma.casefold() == "list"
        and first.upos == "NOUN"
        and first.base_relation == "compound"
        and first.head == root.word_id
        and root.upos in {"NOUN", "PROPN"}
    ):
        return False
    return not any(
        word.head == root.word_id and word.base_relation in SUBJECT_RELATIONS
        for word in content_words
    )


def _is_complex_nominal(
    word: ParsedWord, sentence_words: tuple[ParsedWord, ...]
) -> bool:
    if word.upos not in {"NOUN", "PROPN", "PRON"}:
        return False
    return any(
        child.head == word.word_id and child.base_relation in COMPLEX_NOMINAL_RELATIONS
        for child in sentence_words
    )


def _tree_depth(words: tuple[ParsedWord, ...]) -> int:
    by_id = {word.word_id: word for word in words}
    maximum = 0
    for word in words:
        seen: set[int] = set()
        current = word
        depth = 0
        while current.head:
            if current.word_id in seen or current.head not in by_id:
                raise ValueError("Invalid dependency tree")
            seen.add(current.word_id)
            current = by_id[current.head]
            depth += 1
        maximum = max(maximum, depth)
    return maximum
