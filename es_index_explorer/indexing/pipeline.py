"""The single indexing flow shared by both entry points.

`IndexingPipeline.index_documents` takes already-read `RelativityDocument`s, chunks
and embeds them, builds the nested `_source`, and bulk-writes to Elasticsearch,
returning one `DocumentResult` per document.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from elasticsearch import Elasticsearch

from ..client import get_client
from ..config import Config
from ..relativity.models import RelativityDocument
from .chunking import ChunkParams, ClauseEngine, SemanticChunker, SentenceEngine
from .document_builder import Chunk, DocumentResult, IndexDocument, to_action
from .embedding import E5Embedder, SparseEmbedder, configure_hf_offline
from .engines import PunctuationClauseEngine, SatSentenceEngine, SpacyClauseEngine
from .writer import bulk_index, disabled_refresh


class IndexingPipeline:
    """Chunk -> embed -> build nested source -> bulk write. Loads models once."""

    def __init__(self, config: Config, *, overwrite: bool = False, index_name: str | None = None) -> None:
        resolved_index = (index_name or config.elasticsearch.index_name or "").strip()
        if not resolved_index:
            raise ValueError(
                "Index name must be set via --index-name or elasticsearch.index_name to index documents."
            )

        self._config = config
        self._index_name = resolved_index
        self._overwrite = overwrite
        self._client: Elasticsearch = get_client(config)

        indexing = config.indexing
        configure_hf_offline(indexing)
        self._embedder = E5Embedder(indexing)
        self._sparse_embedder = SparseEmbedder(indexing)
        max_content = indexing.max_content_tokens or self._embedder.max_content_tokens()
        params = ChunkParams(
            unique_target=indexing.chunk_unique_target,
            unique_floor=indexing.chunk_unique_floor,
            overlap_target=indexing.overlap_target,
            overlap_min=indexing.overlap_min,
            overlap_max=indexing.overlap_max,
            max_content_tokens=max_content,
            fallback_window=indexing.fallback_window,
            fallback_overlap=indexing.fallback_overlap,
        )
        self._chunker = SemanticChunker(
            self._embedder.tokenizer,
            self._build_sentence_engine(),
            self._build_clause_engine(),
            params,
        )
        subset_id = config.relativity.subset_id
        self._subset_ids = [subset_id] if subset_id else []

    @contextmanager
    def refresh_disabled(self) -> Iterator[None]:
        """Disable index refresh for the duration of a full ingest run."""
        with disabled_refresh(self._client, self._index_name):
            yield

    def index_documents(self, docs: list[RelativityDocument]) -> list[DocumentResult]:
        """Chunk+embed+write a batch of documents; one result per document."""

        actions: list[dict[str, object]] = []
        pre_failures: list[DocumentResult] = []
        for doc in docs:
            try:
                index_doc = self._build_index_document(doc)
            except Exception as exc:  # noqa: BLE001 - report as an index-stage failure
                pre_failures.append(
                    DocumentResult(
                        artifact_id=doc.artifact_id,
                        outcome="index_failed",
                        stage="index",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue
            actions.append(to_action(self._index_name, index_doc, overwrite=self._overwrite))

        bulk_results: list[DocumentResult] = []
        if actions:
            bulk_results = bulk_index(
                self._client,
                actions,
                overwrite=self._overwrite,
                chunk_size=self._config.indexing.bulk_docs_per_request,
                max_retries=self._config.indexing.bulk_max_retries,
            )
        return pre_failures + bulk_results

    def _build_index_document(self, doc: RelativityDocument) -> IndexDocument:
        spans = self._chunker.chunk(doc.extracted_text)
        vectors = self._embedder.encode_passages([span.text for span in spans])
        chunks = [
            Chunk(
                chunk_index=span.chunk_index,
                text=span.text,
                byte_size=len(span.text.encode("utf-8")),
                char_count=len(span.text),
                token_count=span.token_count,
                leading_overlap_chars=span.leading_overlap_chars,
                embedding=vector,
            )
            for span, vector in zip(spans, vectors)
        ]
        title_sparse = self._sparse_embedder.encode(doc.title, field_name="title") if doc.title else None
        summary_sparse = self._sparse_embedder.encode(doc.summary, field_name="summary") if doc.summary else None
        topic_sparse = self._sparse_embedder.encode(doc.topic, field_name="topic") if doc.topic else None
        return IndexDocument(
            source=doc,
            chunks=chunks,
            subset_ids=self._subset_ids,
            full_token_count=self._embedder.count_tokens(doc.extracted_text),
            title_sparse=title_sparse,
            summary_sparse=summary_sparse,
            topic_sparse=topic_sparse,
        )

    def _build_sentence_engine(self) -> SentenceEngine:
        indexing = self._config.indexing
        if indexing.sentence_engine == "sat":
            return SatSentenceEngine(indexing.sat_model, device=indexing.device)
        raise NotImplementedError(
            f"sentence_engine '{indexing.sentence_engine}' is not implemented; use 'sat'."
        )

    def _build_clause_engine(self) -> ClauseEngine:
        indexing = self._config.indexing
        if indexing.clause_engine == "spacy":
            return SpacyClauseEngine(indexing.spacy_model)
        return PunctuationClauseEngine()
