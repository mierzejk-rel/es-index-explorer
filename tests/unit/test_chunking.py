"""TDD spec for the sentence-first SemanticChunker (report 06 sections 6.1-6.4).

Written before the implementation: these tests define the contract. They use the
hand-written fakes from conftest (whitespace tokenizer == word tokens, scripted
sentence/clause engines) and need no SaT/spaCy/torch.
"""

import pytest

from es_index_explorer.indexing.chunking import (
    CharOffset,
    ChunkParams,
    ClauseBoundary,
    SemanticChunker,
    SentenceSpan,
)

from tests.conftest import (
    Document,
    FakeClauseEngine,
    FakeSentenceEngine,
    FakeTokenizer,
    build_document,
)

pytestmark = pytest.mark.unit


def _chunk(
    doc: Document,
    params: ChunkParams,
    *,
    sentences: list[SentenceSpan] | None = None,
    clauses: list[ClauseBoundary] | None = None,
) -> list:
    tokenizer = FakeTokenizer()
    sentence_engine = FakeSentenceEngine(doc.sentence_spans if sentences is None else sentences)
    clause_engine = FakeClauseEngine(clauses or [])
    return SemanticChunker(tokenizer, sentence_engine, clause_engine, params).chunk(doc.text)


def _token_offsets(text: str) -> list[CharOffset]:
    return FakeTokenizer().token_offsets(text)


def _unique_token_indices(doc: Document, span) -> list[int]:
    offsets = _token_offsets(doc.text)
    content_start_char = span.char_start + span.leading_overlap_chars
    return [
        j
        for j, (start, end) in enumerate(offsets)
        if span.char_start <= start and end <= span.char_end and start >= content_start_char
    ]


def _all_token_indices(doc: Document, span) -> list[int]:
    offsets = _token_offsets(doc.text)
    return [j for j, (start, end) in enumerate(offsets) if span.char_start <= start and end <= span.char_end]


def assert_invariants(doc: Document, chunks: list, params: ChunkParams) -> None:
    """Properties that must hold for any valid chunking of a non-empty doc."""

    assert chunks, "expected at least one chunk for a non-empty document"
    n_tokens = len(_token_offsets(doc.text))

    for i, span in enumerate(chunks):
        # chunk_index is sequential
        assert span.chunk_index == i
        # text is an exact substring of the original
        assert span.text == doc.text[span.char_start : span.char_end]
        # token_count matches the covered tokens and respects the HARD cap
        assert span.token_count == len(_all_token_indices(doc, span))
        assert span.token_count <= params.max_content_tokens
        # every chunk carries unique content beyond its overlap (no pure-overlap chunk)
        assert len(span.text) > span.leading_overlap_chars
        assert _unique_token_indices(doc, span), "chunk has no unique tokens"

    # first chunk has no leading overlap
    assert chunks[0].leading_overlap_chars == 0

    # leading overlap is char-exact: head of chunk i == tail of chunk i-1
    for prev, cur in zip(chunks, chunks[1:]):
        length = cur.leading_overlap_chars
        if length > 0:
            assert cur.text[:length] == prev.text[-length:]

    # contiguity / no loss: unique token ranges, in order, reconstruct [0, n)
    covered: list[int] = []
    for span in chunks:
        covered.extend(_unique_token_indices(doc, span))
    assert covered == list(range(n_tokens))


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #

def test_empty_document_yields_no_chunks(small_params: ChunkParams) -> None:
    doc = build_document([0])  # -> empty text
    assert doc.text == ""
    assert _chunk(doc, small_params) == []


def test_single_short_sentence_is_one_chunk(small_params: ChunkParams) -> None:
    doc = build_document([5])  # below unique_floor (8) -> single chunk to EOF
    chunks = _chunk(doc, small_params)
    assert len(chunks) == 1
    assert chunks[0].leading_overlap_chars == 0
    assert_invariants(doc, chunks, small_params)


def test_two_chunks(small_params: ChunkParams) -> None:
    doc = build_document([10, 6])  # sentence boundaries at tokens 10 and 16
    chunks = _chunk(doc, small_params)
    assert len(chunks) == 2
    # chunk 0: [0, 10) on the sentence boundary closest to unique_target (10)
    assert chunks[0].char_start == doc.token_char_start(0)
    assert chunks[0].char_end == _token_offsets(doc.text)[9].char_end
    assert chunks[0].leading_overlap_chars == 0
    # chunk 1: ends at EOF and carries overlap from chunk 0
    assert chunks[1].char_end == len(doc.text)
    assert chunks[1].leading_overlap_chars > 0
    assert_invariants(doc, chunks, small_params)


def test_many_chunks_first_middle_last(small_params: ChunkParams) -> None:
    doc = build_document([5] * 8)  # 40 tokens, boundaries every 5
    chunks = _chunk(doc, small_params)
    assert len(chunks) >= 3
    assert chunks[0].leading_overlap_chars == 0
    assert all(c.leading_overlap_chars > 0 for c in chunks[1:])
    # last chunk reaches the end of the document
    assert chunks[-1].char_end == len(doc.text)
    assert_invariants(doc, chunks, small_params)


# --------------------------------------------------------------------------- #
# Priorities: clause (4-2) and word (1) fallbacks for an over-long sentence
# --------------------------------------------------------------------------- #

def test_overlong_sentence_word_fallback_when_no_clause(small_params: ChunkParams) -> None:
    doc = build_document([30])  # one sentence, longer than max_content_tokens (18)
    chunks = _chunk(doc, small_params)  # no clause boundaries -> word cut at the target
    # first chunk cut at unique_target (10) tokens via word fallback
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == _token_offsets(doc.text)[9].char_end  # tokens [0, 10)
    assert_invariants(doc, chunks, small_params)


def test_overlong_sentence_prefers_higher_priority_clause(small_params: ChunkParams) -> None:
    doc = build_document([30])
    # A semicolon (priority 3) at token 12 and a comma (priority 2) at token 8.
    clauses = [
        ClauseBoundary(char_pos=doc.token_char_start(12), priority=3),
        ClauseBoundary(char_pos=doc.token_char_start(8), priority=2),
    ]
    chunks = _chunk(doc, small_params, clauses=clauses)
    # priority dominates distance-to-target: cut before token 12 (semicolon), not token 8
    assert chunks[0].char_end == _token_offsets(doc.text)[11].char_end  # tokens [0, 12)
    assert_invariants(doc, chunks, small_params)


# --------------------------------------------------------------------------- #
# Tier B: no sentence structure -> legacy non-semantic sliding window
# --------------------------------------------------------------------------- #

def test_no_sentences_uses_legacy_sliding_window(small_params: ChunkParams) -> None:
    doc = build_document([20])  # 20 tokens
    chunks = _chunk(doc, small_params, sentences=[])  # Tier B
    # window=10, overlap=3, stride=7 -> [0,10), [7,17), [14,20)
    starts = [c.char_start for c in chunks]
    ends = [c.char_end for c in chunks]
    offsets = _token_offsets(doc.text)
    assert starts[0] == offsets[0].char_start
    assert ends[0] == offsets[9].char_end
    assert starts[1] == offsets[7].char_start
    assert ends[-1] == len(doc.text)
    assert chunks[0].leading_overlap_chars == 0
    assert all(c.leading_overlap_chars > 0 for c in chunks[1:])
    assert_invariants(doc, chunks, small_params)


# --------------------------------------------------------------------------- #
# Edge cases and global properties
# --------------------------------------------------------------------------- #

def test_overlap_clamped_to_window(small_params: ChunkParams) -> None:
    doc = build_document([5] * 8)
    chunks = _chunk(doc, small_params)
    for cur in chunks[1:]:
        overlap_tokens = len(_all_token_indices(doc, cur)) - len(_unique_token_indices(doc, cur))
        assert overlap_tokens <= small_params.overlap_max


def test_determinism(small_params: ChunkParams) -> None:
    doc = build_document([5] * 8)
    first = _chunk(doc, small_params)
    second = _chunk(doc, small_params)
    assert first == second


def test_cap_respected_for_overlong_sentence(small_params: ChunkParams) -> None:
    doc = build_document([50])  # one very long sentence
    chunks = _chunk(doc, small_params)
    assert len(chunks) >= 3
    for span in chunks:
        assert span.token_count <= small_params.max_content_tokens
    assert_invariants(doc, chunks, small_params)
