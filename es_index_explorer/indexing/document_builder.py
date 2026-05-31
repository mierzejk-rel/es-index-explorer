"""Declarative mapping from RelativityDocument to the nested Elasticsearch document.

The `MAPPING` dict is the single source of truth for the ES `_source` body. This
module is light to import (no ML/ES dependencies).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..relativity.models import RelativityDocument

# Outcome of attempting to index one document (report 06 section 8.2).
Outcome = Literal["created", "overwritten", "conflict", "read_failed", "index_failed"]
Stage = Literal["read", "index"]


@dataclass(frozen=True)
class Chunk:
    """A nested chunk ready to be written to Elasticsearch."""

    chunk_index: int
    text: str
    token_count: int
    leading_overlap_chars: int
    embedding: list[float]


@dataclass
class IndexDocument:
    """Index-ready view over a RelativityDocument (holds a reference; does not copy).

    Attributes
    ----------
    source : RelativityDocument
        The original document POJO.
    chunks : list[Chunk]
        Embedded chunks (produced by the chunker + embedder).
    subset_ids : list[str]
        Multi-tenancy scoping ids for this run.
    full_token_count : int
        Full-document token count (pre-chunking), computed once with the e5 tokenizer.
    """

    source: RelativityDocument
    chunks: list[Chunk]
    subset_ids: list[str]
    full_token_count: int

    @property
    def byte_size(self) -> int:
        """UTF-8 byte length of the extracted text (report 05 section 6.1)."""
        return len(self.source.extracted_text.encode("utf-8"))

    @property
    def char_count(self) -> int:
        """Unicode code-point length of the extracted text (report 05 section 6.4)."""
        return len(self.source.extracted_text)

    @property
    def token_count(self) -> int:
        """Full-document token count, pre-chunking (report 05 section 6.2)."""
        return self.full_token_count

    @property
    def chunk_count(self) -> int:
        """Number of chunks (report 05 section 6.3)."""
        return len(self.chunks)


@dataclass(frozen=True)
class DocumentResult:
    """Per-document outcome recorded for progress/resume and error reporting."""

    artifact_id: int | None
    outcome: Outcome
    stage: Stage | None = None
    error_type: str | None = None
    error_message: str | None = None


def _chunk_to_dict(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "leading_overlap_chars": chunk.leading_overlap_chars,
        "embedding": chunk.embedding,
    }


def _primary_date_time(doc: IndexDocument) -> str | None:
    value = doc.source.primary_date_time
    return value.isoformat() if value is not None else None


# Single source of truth: ES field name -> extractor over an IndexDocument.
# title_semantic/summary_semantic/topic_semantic are filled by ES via copy_to and
# are intentionally not set by the client (report 05 section 4.5).
MAPPING: dict[str, Callable[[IndexDocument], object]] = {
    "document_artifact_id": lambda d: d.source.artifact_id,
    "control_number": lambda d: d.source.control_number,
    "title": lambda d: d.source.title,
    "summary": lambda d: d.source.summary,
    "topic": lambda d: d.source.topic,
    "primary_date_time": _primary_date_time,
    "email_from": lambda d: d.source.email_from,
    "email_to": lambda d: d.source.email_to,
    "email_cc": lambda d: d.source.email_cc,
    "email_bcc": lambda d: d.source.email_bcc,
    "byte_size": lambda d: d.byte_size,
    "token_count": lambda d: d.token_count,
    "char_count": lambda d: d.char_count,
    "chunk_count": lambda d: d.chunk_count,
    "subset_ids": lambda d: d.subset_ids,
    "chunks": lambda d: [_chunk_to_dict(c) for c in d.chunks],
}


def build_source(doc: IndexDocument) -> dict[str, object]:
    """Build the Elasticsearch `_source` body from `MAPPING`, dropping null values."""

    source: dict[str, object] = {}
    for field_name, extract in MAPPING.items():
        value = extract(doc)
        if value is not None:
            source[field_name] = value
    return source


def to_action(index_name: str, doc: IndexDocument, *, overwrite: bool) -> dict[str, object]:
    """Build a `streaming_bulk` action for one document.

    Uses ``create`` by default (Elasticsearch returns 409 if the id already exists,
    so an existing document is never clobbered) and ``index`` when ``overwrite`` is set.
    """

    return {
        "_op_type": "index" if overwrite else "create",
        "_index": index_name,
        "_id": str(doc.source.artifact_id),
        "_source": build_source(doc),
    }
