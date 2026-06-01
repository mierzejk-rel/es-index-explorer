"""Semantic, sentence-first chunker (pure algorithm + engine protocols).

The chunker packs whole sentences into chunks with a sentence-aligned leading
overlap. Lower-priority boundaries (clause/word) are used only as a fallback when
a single sentence does not fit the length budget. The algorithm depends only on
injected collaborators (a tokenizer, a sentence engine, a clause engine), so it is
unit-testable with hand-written fakes and free of heavy ML dependencies.

See ``reports/06-document-indexing-and-semantic-chunking.md`` sections 6.1-6.4.
"""

import bisect
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Protocol, runtime_checkable


class Priority(IntEnum):
    """Boundary priority hierarchy (most -> least preferred), report 06 section 6.1."""

    WORD = 1
    COMMA = 2
    SEMICOLON = 3
    PARENTHETICAL = 4
    SENTENCE = 5


@dataclass(frozen=True)
class ChunkSpan:
    """A single produced chunk, expressed over the original document text.

    Attributes
    ----------
    chunk_index : int
        0-based sequential index of the chunk within the document.
    text : str
        Exact substring of the original text covered by this chunk
        (leading overlap + new/unique content).
    token_count : int
        Number of tokenizer tokens spanning the whole chunk (overlap + unique).
    leading_overlap_chars : int
        Number of leading characters duplicated from the previous chunk; 0 for
        the first chunk. Enables tokenizer-free concatenation of contiguous chunks.
    char_start : int
        Inclusive character offset of the chunk in the original text.
    char_end : int
        Exclusive character offset of the chunk in the original text.
    """

    chunk_index: int
    text: str
    token_count: int
    leading_overlap_chars: int
    char_start: int
    char_end: int


class WindowTriple(NamedTuple):
    """Chunk window over token indices.

    Attributes
    ----------
    overlap_start : int
        Inclusive token index where the chunk starts (may include overlap).
    content_start : int
        Inclusive token index where new/unique content starts.
    cut : int
        Exclusive token index where the chunk ends.
    """

    overlap_start: int
    content_start: int
    cut: int


@dataclass(frozen=True)
class ClauseBoundary:
    """A candidate sub-sentence cut position with its priority.

    Attributes
    ----------
    char_pos : int
        Character offset in the original text at which a cut may occur (the start
        of the next clause/segment).
    priority : Priority
        One of ``Priority.PARENTHETICAL``, ``Priority.SEMICOLON``, or
        ``Priority.COMMA``.
    """

    char_pos: int
    priority: Priority


@dataclass(frozen=True)
class ChunkParams:
    """Chunk geometry. All values are soft except ``max_content_tokens``.

    Token counts are in the injected tokenizer's tokens (e5 subword tokens in
    production).
    """

    unique_target: int = 400
    unique_floor: int = 360
    overlap_target: int = 80
    overlap_min: int = 40
    overlap_max: int = 120
    max_content_tokens: int = 505
    fallback_window: int = 500
    fallback_overlap: int = 100


@runtime_checkable
class Tokenizer(Protocol):
    """Token<->character alignment for a text."""

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Return ``(char_start, char_end)`` for each token of ``text``.

        Special/model tokens must be excluded; the returned list defines both the
        token count (its length) and the char span of each token, in order.
        """
        ...


@runtime_checkable
class SentenceEngine(Protocol):
    """Sentence boundary detector (priority 5)."""

    def sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """Return ordered ``(char_start, char_end)`` spans of sentences in ``text``.

        An empty list signals "no sentence structure" and triggers the chunker's
        whole-document non-semantic fallback (Tier B).
        """
        ...


@runtime_checkable
class ClauseEngine(Protocol):
    """Sub-sentence clause boundary detector (priorities 4-2, Tier-A fallback)."""

    def clause_boundaries(self, text: str, start: int, end: int) -> list[ClauseBoundary]:
        """Return candidate clause cut boundaries within ``[start, end)`` of ``text``.

        Only used when a single sentence is too long to fit the budget. Returns
        boundaries with priority 4 (parenthetical), 3 (semicolon), or 2 (comma);
        an empty list means the chunker falls back to a word boundary (priority 1).
        """
        ...


class SemanticChunker:
    """Sentence-first chunker with sentence-aligned overlap and clause/word fallback.

    Parameters
    ----------
    tokenizer : Tokenizer
        Provides token<->char offsets used for counting and slicing.
    sentence_engine : SentenceEngine
        Produces sentence spans (priority 5); the default unit of both the unique
        content and the leading overlap.
    clause_engine : ClauseEngine
        Produces clause boundaries (priorities 4-2) used only inside an over-long
        sentence (Tier A).
    params : ChunkParams
        Chunk geometry (soft targets + the hard ``max_content_tokens`` cap).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        sentence_engine: SentenceEngine,
        clause_engine: ClauseEngine,
        params: ChunkParams,
    ) -> None:
        self._tokenizer = tokenizer
        self._sentence_engine = sentence_engine
        self._clause_engine = clause_engine
        self._params = params

    def chunk(self, text: str) -> list[ChunkSpan]:
        """Split ``text`` into overlapping chunks (see module docstring).

        Returns
        -------
        list[ChunkSpan]
            Chunks in document order. Empty when ``text`` has no tokens.
        """

        offsets = self._tokenizer.token_offsets(text)
        n = len(offsets)
        if n == 0:
            return []

        sentence_spans = self._sentence_engine.sentence_spans(text)
        token_starts = [start for start, _ in offsets]

        if not sentence_spans:
            triples = self._legacy_windows(n)
        else:
            bounds = self._sentence_boundaries(sentence_spans, token_starts, n)
            triples = self._semantic_windows(text, offsets, token_starts, bounds, n)

        return self._build_spans(text, offsets, triples)

    # -- semantic (sentence-first) path ------------------------------------- #

    def _semantic_windows(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        token_starts: list[int],
        bounds: list[int],
        n: int,
    ) -> list[WindowTriple]:
        """Produce (overlap_start, content_start, cut) token triples."""

        triples: list[WindowTriple] = []
        content_start = 0
        prev_overlap_start = 0
        first = True
        while content_start < n:
            if first:
                overlap_start = 0
            else:
                overlap_start = self._choose_overlap_start(
                    text, offsets, token_starts, bounds, content_start, prev_overlap_start
                )
            cut = self._choose_cut(text, offsets, token_starts, bounds, content_start, overlap_start, n)
            triples.append(WindowTriple(overlap_start, content_start, cut))
            prev_overlap_start = overlap_start
            content_start = cut
            first = False
        return triples

    def _choose_overlap_start(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        token_starts: list[int],
        bounds: list[int],
        content_start: int,
        prev_overlap_start: int,
    ) -> int:
        params = self._params
        lo = max(0, content_start - params.overlap_max)
        hi = content_start - params.overlap_min
        chosen: int | None = None
        if hi >= lo and hi >= 0:
            sentence_candidates = [b for b in bounds if lo <= b <= hi and b < content_start]
            if sentence_candidates:
                chosen = min(
                    sentence_candidates,
                    key=lambda b: (abs((content_start - b) - params.overlap_target), -b),
                )
            else:
                clause_candidates = self._clause_token_candidates(
                    text, token_starts, offsets[lo][0], offsets[content_start][0], lo, hi
                )
                if clause_candidates:
                    chosen = min(
                        clause_candidates,
                        key=lambda item: (-item[1], abs((content_start - item[0]) - params.overlap_target), -item[0]),
                    )[0]
                else:
                    chosen = min(max(content_start - params.overlap_target, lo), hi)
        if chosen is None:
            chosen = 0
        # Keep the overlap region a suffix of the previous chunk (char-exact dedup).
        assert chosen is not None
        chosen: int = max(chosen, prev_overlap_start)
        if chosen >= content_start:
            chosen = max(prev_overlap_start, content_start - 1)
        return max(0, chosen)

    def _choose_cut(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        token_starts: list[int],
        bounds: list[int],
        content_start: int,
        overlap_start: int,
        n: int,
    ) -> int:
        params = self._params
        cap_hi = min(overlap_start + params.max_content_tokens, n)
        if cap_hi <= content_start:
            cap_hi = min(content_start + 1, n)

        feasible = [b for b in bounds if content_start < b <= cap_hi]
        ge_floor = [b for b in feasible if (b - content_start) >= params.unique_floor]
        if ge_floor:
            return min(ge_floor, key=lambda b: (abs((b - content_start) - params.unique_target), -b))
        if feasible and max(feasible) == n:
            return n

        # Tier A: cut inside an over-long sentence using clause (4-2) then word (1).
        target_tok = min(content_start + params.unique_target, cap_hi)
        char_hi = offsets[cap_hi][0] if cap_hi < n else len(text)
        clause_candidates = self._clause_token_candidates(
            text, token_starts, offsets[content_start][0], char_hi, content_start + 1, cap_hi
        )
        if clause_candidates:
            best = min(
                clause_candidates,
                key=lambda item: (-item[1], abs(item[0] - target_tok), -item[0]),
            )
            return best[0]
        return max(target_tok, content_start + 1)

    def _clause_token_candidates(
        self,
        text: str,
        token_starts: list[int],
        char_lo: int,
        char_hi: int,
        token_lo: int,
        token_hi: int,
    ) -> list[tuple[int, Priority]]:
        """Return (token_index, priority) clause candidates within the bounds."""

        out: list[tuple[int, Priority]] = []
        for boundary in self._clause_engine.clause_boundaries(text, char_lo, char_hi):
            token_index = bisect.bisect_left(token_starts, boundary.char_pos)
            if token_lo <= token_index <= token_hi:
                out.append((token_index, boundary.priority))
        return out

    @staticmethod
    def _sentence_boundaries(
        sentence_spans: list[tuple[int, int]], token_starts: list[int], n: int
    ) -> list[int]:
        bounds: set[int] = {0, n}
        for char_start, char_end in sentence_spans:
            bounds.add(bisect.bisect_left(token_starts, char_start))
            bounds.add(bisect.bisect_left(token_starts, char_end))
        return sorted(b for b in bounds if 0 <= b <= n)

    # -- Tier B (no sentence structure): legacy sliding window -------------- #

    def _legacy_windows(self, n: int) -> list[WindowTriple]:
        params = self._params
        window = params.fallback_window
        overlap = min(params.fallback_overlap, window - 1)
        stride = max(1, window - overlap)
        triples: list[WindowTriple] = []
        k = 0
        while True:
            overlap_start = k * stride
            if overlap_start >= n:
                break
            cut = min(overlap_start + window, n)
            content_start = overlap_start if k == 0 else overlap_start + overlap
            if content_start >= cut:
                break  # no new content -> drop pure-overlap tail
            triples.append(WindowTriple(overlap_start, content_start, cut))
            if cut >= n:
                break
            k += 1
        return triples

    # -- span assembly ------------------------------------------------------ #

    @staticmethod
    def _build_spans(
        text: str,
        offsets: list[tuple[int, int]],
        triples: list[WindowTriple],
    ) -> list[ChunkSpan]:
        spans: list[ChunkSpan] = []
        index = 0
        for window in triples:
            overlap_start = window.overlap_start
            content_start = window.content_start
            cut = window.cut
            if cut <= content_start:
                continue  # safeguard: never emit a pure-overlap chunk
            char_start = offsets[overlap_start][0]
            char_end = offsets[cut - 1][1]
            # The overlap region is tokens [overlap_start, content_start); its char length
            # ends at the END of the last overlap token (excluding the separator that follows),
            # so the head of this chunk equals the tail of the previous chunk char-for-char.
            if content_start > overlap_start:
                leading_overlap_chars = offsets[content_start - 1][1] - char_start
            else:
                leading_overlap_chars = 0
            spans.append(
                ChunkSpan(
                    chunk_index=index,
                    text=text[char_start:char_end],
                    token_count=cut - overlap_start,
                    leading_overlap_chars=leading_overlap_chars,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            index += 1
        return spans
