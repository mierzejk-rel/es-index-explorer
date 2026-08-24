"""Pure deterministic linguistic feature rules."""

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
CLAUSAL_RELATIONS = frozenset(
    {"advcl", "acl", "ccomp", "xcomp", "csubj", "parataxis"}
)
SUBORDINATE_RELATIONS = frozenset({"advcl", "acl", "ccomp", "xcomp", "csubj"})
COMPLEX_NOMINAL_RELATIONS = frozenset(
    {"amod", "appos", "compound", "nmod", "nummod", "acl"}
)
SUBJECT_RELATIONS = frozenset({"nsubj", "csubj", "expl"})
WH_FORMS = frozenset(
    {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"}
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
        raise ValueError("Deterministic linguistic extraction requires parsed, non-empty text")

    words = tuple(word for sentence in sentences for word in sentence.words)
    content_words = tuple(word for word in words if word.upos != PUNCTUATION_UPOS)
    dependency_lengths = tuple(
        abs(word.word_id - word.head)
        for word in content_words
        if word.head != 0
    )
    clause_heads = tuple(
        word
        for sentence in sentences
        for word in sentence.words
        if _is_clause_head(word)
    )
    clause_count = len(clause_heads)
    subordinate_count = sum(
        word.base_relation in SUBORDINATE_RELATIONS for word in clause_heads
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
        clause_type: ClauseType | None = classify_clause_type(sentences[0])
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


def classify_clause_type(sentence: ParsedSentence) -> ClauseType:
    """Apply the frozen first-sentence clause-type precedence."""
    content_words = tuple(
        word for word in sentence.words if word.upos != PUNCTUATION_UPOS
    )
    if any(word.feature("PronType") == "Int" for word in content_words):
        return "open_interrogative"
    if content_words and content_words[0].text.casefold() in WH_FORMS:
        return "open_interrogative"
    if sentence.text.rstrip().endswith("?"):
        return "closed_interrogative"

    root = next((word for word in content_words if word.head == 0), None)
    if root is not None and root.upos in {"VERB", "AUX"}:
        has_subject = any(
            word.head == root.word_id and word.base_relation in SUBJECT_RELATIONS
            for word in content_words
        )
        if root.feature("Mood") == "Imp" or not has_subject:
            return "directive_imperative"
    return "declarative_request"


def _is_clause_head(word: ParsedWord) -> bool:
    if word.upos not in {"VERB", "AUX"}:
        return False
    if word.head == 0:
        return True
    if word.base_relation in CLAUSAL_RELATIONS:
        return True
    return word.base_relation == "conj" and (
        word.feature("VerbForm") == "Fin" or word.feature("Mood") is not None
    )


def _is_complex_nominal(
    word: ParsedWord, sentence_words: tuple[ParsedWord, ...]
) -> bool:
    if word.upos not in {"NOUN", "PROPN", "PRON"}:
        return False
    return any(
        child.head == word.word_id
        and child.base_relation in COMPLEX_NOMINAL_RELATIONS
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
