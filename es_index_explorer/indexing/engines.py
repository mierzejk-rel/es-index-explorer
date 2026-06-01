"""Concrete engine adapters for the chunker (Tokenizer / SentenceEngine / ClauseEngine).

Heavy optional dependencies are imported lazily inside ``__init__`` so that importing
this module (and the pure ``PunctuationClauseEngine`` / ``HuggingFaceTokenizer``) does
not require ``wtpsplit`` or ``spacy``.
"""

import logging
from typing import Any

from es_index_explorer.indexing.chunking import (
    CharOffset,
    SentenceSpan,
    Priority,
    ClauseBoundary, ClauseEngine,
)

_CLAUSE_PRIORITY: dict[str, Priority] = {
    ")": Priority.PARENTHETICAL,
    ";": Priority.SEMICOLON,
    ",": Priority.COMMA,
}

# transformers' lazy `*_fast` image-processing aliases log a WARNING on every attribute
# access (see transformers/__init__.py). skops (a wtpsplit dependency) probes every module
# in ``sys.modules`` for scipy attributes during import, touching all those aliases and
# emitting hundreds of irrelevant "Accessing `...`" lines. We drop only those records.
_ALIAS_WARNING_PREFIX = "Accessing `"
_alias_filter_installed = False


class _TransformersAliasWarningFilter(logging.Filter):
    """Drops transformers' ``*_fast`` deprecation-alias access warnings (noise only)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_ALIAS_WARNING_PREFIX)


def silence_transformers_alias_warnings() -> None:
    """Install a filter that suppresses transformers' alias-access WARNING spam.

    Idempotent and surgical: only records starting with ``Accessing ``` are dropped, so
    genuine transformers warnings are preserved. Must run before ``wtpsplit``/``skops`` import.
    """

    global _alias_filter_installed
    if _alias_filter_installed:
        return
    logging.getLogger("transformers").addFilter(_TransformersAliasWarningFilter())
    _alias_filter_installed = True


class HuggingFaceTokenizer:
    """Adapts a HuggingFace *fast* tokenizer to the ``Tokenizer`` protocol."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tokenizer = hf_tokenizer

    def token_offsets(self, text: str) -> list[CharOffset]:
        encoded = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        # Drop zero-width entries that some tokenizers emit for control/special pieces.
        return [CharOffset(int(start), int(end)) for start, end in encoded["offset_mapping"] if end > start]


class PunctuationClauseEngine(ClauseEngine):
    """Lean, dependency-free clause engine: scans for ``)`` / ``;`` / ``,``.

    A comma surrounded by digits (e.g. ``1,000``) is not treated as a clause
    boundary (digit-guard). Boundaries are placed immediately after the mark.
    """

    def clause_boundaries(self, text: str, start: int, end: int) -> list[ClauseBoundary]:
        boundaries: list[ClauseBoundary] = []
        upper = min(end, len(text))
        for i in range(max(start, 0), upper):
            char = text[i]
            priority = _CLAUSE_PRIORITY.get(char)
            if priority is None:
                continue
            if char == ",":
                left = text[i - 1] if i > 0 else ""
                right = text[i + 1] if i + 1 < len(text) else ""
                if left.isdigit() and right.isdigit():
                    continue
            pos = i + 1
            if start < pos < end:
                boundaries.append(ClauseBoundary(char_pos=pos, priority=priority))
        return boundaries


class SpacyClauseEngine:
    """Morphosyntactic clause engine using a spaCy English model (default Tier-A engine).

    spaCy tokenization decides what counts as a clause-marking ``)``/``;``/``,``; a
    comma between numbers is skipped (digit-guard). Boundaries are placed after the mark.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        import spacy  # lazy: only needed when this engine is actually used

        self._nlp = spacy.load(model_name)

    def clause_boundaries(self, text: str, start: int, end: int) -> list[ClauseBoundary]:
        segment = text[start:end]
        doc = self._nlp(segment)
        boundaries: list[ClauseBoundary] = []
        for token in doc:
            priority = _CLAUSE_PRIORITY.get(token.text)
            if priority is None:
                continue
            if token.text == ",":
                left = doc[token.i - 1] if token.i > 0 else None
                right = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if (left is not None and left.like_num) and (right is not None and right.like_num):
                    continue
            pos = start + token.idx + len(token.text)
            if start < pos < end:
                boundaries.append(ClauseBoundary(char_pos=pos, priority=priority))
        return boundaries


class SatSentenceEngine:
    """Sentence engine backed by wtpsplit's Segment any Text (SaT) model."""

    def __init__(self, model_name: str = "sat-12l-sm", device: str = "") -> None:
        silence_transformers_alias_warnings()  # before wtpsplit/skops walk transformers' aliases
        from wtpsplit import SaT  # lazy: pulls torch

        self._sat = SaT(model_name)
        # SaT loads on CPU by default; only move it for an actual accelerator.
        if device and device.lower() != "cpu":
            self._sat.to(device)

    def sentence_spans(self, text: str) -> list[SentenceSpan]:
        spans: list[SentenceSpan] = []
        cursor = 0
        for sentence in self._sat.split(text):
            stripped = sentence.strip()
            if not stripped:
                continue
            idx = text.find(stripped, cursor)
            if idx < 0:
                continue
            spans.append(SentenceSpan(idx, idx + len(stripped)))
            cursor = idx + len(stripped)
        return spans
