"""Concrete engine adapters for the chunker (Tokenizer / SentenceEngine / ClauseEngine).

Heavy optional dependencies are imported lazily inside ``__init__`` so that importing
this module (and the pure ``PunctuationClauseEngine`` / ``HuggingFaceTokenizer``) does
not require ``wtpsplit`` or ``spacy``.
"""

import logging
import warnings
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

# transformers logs two known-benign warnings in our startup path:
# 1) lazy `*_fast` image-processing alias access ("Accessing `...`")
# 2) `use_return_dict` deprecation triggered by wtpsplit reading model config
# We suppress only these exact prefixes and let all other transformers warnings through.
_SUPPRESSED_TRANSFORMERS_WARNING_PREFIXES = (
    "Accessing `",
    "`use_return_dict` is deprecated",
)
_alias_filter_installed = False


class _TransformersAliasWarningFilter(logging.Filter):
    """Drops selected known-benign transformers warning lines (noise only)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_SUPPRESSED_TRANSFORMERS_WARNING_PREFIXES)


def silence_transformers_alias_warnings() -> None:
    """Install a targeted filter for known-benign transformers WARNING spam.

    Idempotent and surgical: only warning messages starting with a small explicit
    allowlist are dropped, so genuine transformers warnings are preserved. Must run
    before ``wtpsplit``/``skops`` import.

    The filter is attached both to the ``transformers`` logger (for records it emits
    directly, e.g. the ``*_fast`` alias spam) and to that logger's handler. The handler
    is essential: the ``use_return_dict`` deprecation is emitted by the child logger
    ``transformers.configuration_utils`` and only the parent's handler sees it on
    propagation, so a parent-logger filter alone would miss it.
    """

    global _alias_filter_installed
    if _alias_filter_installed:
        return
    log_filter = _TransformersAliasWarningFilter()
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.addFilter(log_filter)
    # Ensure transformers' default handler exists, then filter at the handler level so
    # records propagated from child loggers are covered too.
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.get_logger("transformers")
    except ImportError:
        pass
    for handler in transformers_logger.handlers:
        handler.addFilter(log_filter)
    _alias_filter_installed = True


# The legacy single-file ``docopt`` module (pulled in transitively by the model stack) uses
# unescaped regex strings, so the byte-compiler emits ``invalid escape sequence`` SyntaxWarnings
# the first time it is compiled. These surface or vanish purely based on whether a valid
# ``docopt`` .pyc is cached, which is why they come and go between runs.
_docopt_filter_installed = False


def silence_docopt_syntax_warnings() -> None:
    """Suppress ``docopt``'s ``invalid escape sequence`` ``SyntaxWarning``s.

    Targeted by category + message only. A ``module=`` filter does NOT work here: compile-time
    syntax warnings are not attributed to the importing module's ``__name__``, so matching on
    ``module`` silently fails (verified empirically). Must run before ``docopt`` is compiled
    (i.e. before the wtpsplit/sentence-transformers import that pulls it in).
    """

    global _docopt_filter_installed
    if _docopt_filter_installed:
        return
    warnings.filterwarnings("ignore", message=r"invalid escape sequence", category=SyntaxWarning)
    _docopt_filter_installed = True


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
        silence_docopt_syntax_warnings()  # before wtpsplit pulls in (and compiles) docopt
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
