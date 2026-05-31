"""Shared fakes and builders for the chunker test suite.

These are hand-written fakes (no mocking library): a whitespace `FakeTokenizer`
(one token == one whitespace-delimited word, so token counts are word counts), a
scripted `FakeSentenceEngine`, and a scripted `FakeClauseEngine`. They let the
chunking algorithm be tested deterministically without SaT/spaCy/torch.
"""

from dataclasses import dataclass

import pytest

from es_index_explorer.indexing.chunking import ChunkParams, ClauseBoundary


class FakeTokenizer:
    """Whitespace tokenizer: each run of non-space characters is one token."""

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i].isspace():
                i += 1
                continue
            start = i
            while i < n and not text[i].isspace():
                i += 1
            offsets.append((start, i))
        return offsets


class FakeSentenceEngine:
    """Returns scripted sentence char spans (independent of the text argument)."""

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        self._spans = list(spans)

    def sentence_spans(self, text: str) -> list[tuple[int, int]]:
        return list(self._spans)


class FakeClauseEngine:
    """Returns scripted clause boundaries that fall strictly within ``(start, end)``."""

    def __init__(self, boundaries: list[ClauseBoundary] | None = None) -> None:
        self._boundaries = list(boundaries or [])

    def clause_boundaries(self, text: str, start: int, end: int) -> list[ClauseBoundary]:
        return [b for b in self._boundaries if start < b.char_pos < end]


@dataclass(frozen=True)
class Document:
    """A built test document with word and sentence char offsets."""

    text: str
    word_offsets: list[tuple[int, int]]   # char span of each word/token
    sentence_spans: list[tuple[int, int]]  # char span of each sentence

    def token_char_start(self, token_index: int) -> int:
        """Char offset where token ``token_index`` begins (or len(text) at the end)."""
        if token_index >= len(self.word_offsets):
            return len(self.text)
        return self.word_offsets[token_index][0]


def build_document(sentence_word_counts: list[int]) -> Document:
    """Build a space-joined document of fixed-width words with known sentence spans.

    Each token is a 4-char word (``wNNN``); sentences are consecutive runs of words.
    Token index == word index, so token boundaries align to word starts.
    """

    total = sum(sentence_word_counts)
    words = [f"w{i:03d}" for i in range(total)]
    text = " ".join(words)

    word_offsets: list[tuple[int, int]] = []
    pos = 0
    for word in words:
        word_offsets.append((pos, pos + len(word)))
        pos += len(word) + 1  # trailing space

    sentence_spans: list[tuple[int, int]] = []
    wi = 0
    for count in sentence_word_counts:
        if count <= 0:
            continue
        start_char = word_offsets[wi][0]
        end_char = word_offsets[wi + count - 1][1]
        sentence_spans.append((start_char, end_char))
        wi += count
    return Document(text=text, word_offsets=word_offsets, sentence_spans=sentence_spans)


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def small_params() -> ChunkParams:
    """Small geometry that keeps multi-chunk scenarios easy to reason about.

    Note unique_floor >= overlap_max so middle chunks are always longer than the
    overlap window (keeps the leading-overlap-is-a-suffix invariant well-defined).
    """

    return ChunkParams(
        unique_target=10,
        unique_floor=8,
        overlap_target=4,
        overlap_min=2,
        overlap_max=6,
        max_content_tokens=18,
        fallback_window=10,
        fallback_overlap=3,
    )
