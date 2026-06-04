"""The single ingest loop shared by both read sources.

`IngestEngine.run` drives a `DocumentSource`, and:
- records read failures (stage="read") to the progress log;
- when a sink is present (i.e. `--index`), indexes each batch and records the index
  outcome (created/overwritten/conflict/index_failed);
- when no sink is present (dry run), a successfully read document counts as complete;
- emits a `ProgressSnapshot` as each document reaches its final state, so 100% is only
  reported once every document has been processed end-to-end.

Fresh/resume/retry are expressed purely as the query condition, so both sources behave
identically. Progress logging only happens when a `progress_log` is supplied (i.e. only
when indexing).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Literal

from ..indexing.document_builder import DocumentResult
from ..relativity.conditions import Cond, field
from ..relativity.models import RelativityDocument
from ..relativity.progress import ProgressLog, record_index_result
from ..relativity.tui import ProgressSnapshot
from .sources import DocumentSource, ReadItem

Mode = Literal["fresh", "resume", "retry"]
DocumentSink = Callable[[list[RelativityDocument]], list[DocumentResult]]

_ARTIFACT_ID = "Artifact ID"
_RETRY_CHUNK_SIZE = 1000


def compose_condition(
    *, saved_search_id: int | None, resume_after: int, retry_ids: Sequence[int] | None
) -> Cond | None:
    """Build the OM query condition from saved-search scoping plus resume/retry.

    - saved search (if configured) is always ANDed in;
    - retry restricts to an explicit Artifact ID list;
    - otherwise resume restricts to Artifact ID greater than the last processed id.
    """

    condition: Cond | None = None
    if saved_search_id is not None:
        condition = field(_ARTIFACT_ID).in_saved_search(saved_search_id)
    if retry_ids is not None:
        retry_cond = field(_ARTIFACT_ID).in_(list(retry_ids))
        condition = retry_cond if condition is None else condition & retry_cond
    elif resume_after > 0:
        resume_cond = field(_ARTIFACT_ID).gt(resume_after)
        condition = resume_cond if condition is None else condition & resume_cond
    return condition


def _chunks(ids: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(ids), size):
        yield list(ids[start : start + size])


class IngestEngine:
    """Read (and optionally index) documents, recording progress uniformly."""

    def __init__(
        self,
        *,
        source: DocumentSource,
        saved_search_id: int | None,
        batch_size: int,
        on_progress: Callable[[ProgressSnapshot], None],
        on_batch_complete: Callable[[ProgressSnapshot], None] | None = None,
        progress_log: ProgressLog | None = None,
        sink: DocumentSink | None = None,
        limit: int | None = None,
    ) -> None:
        self._source = source
        self._saved_search_id = saved_search_id
        self._batch_size = batch_size
        self._on_progress = on_progress
        self._on_batch_complete = on_batch_complete
        self._progress_log = progress_log
        self._sink = sink
        self._limit = limit
        self._processed = 0
        self._ok = 0
        self._failed = 0
        self._last_error: str | None = None
        self._last_artifact_id: int | None = None

    def run(
        self, *, mode: Mode, resume_after: int = 0, retry_ids: Sequence[int] | None = None
    ) -> None:
        """Process documents for the given mode (fresh/resume/retry)."""

        self._reset_counters()
        # Never fetch/build more documents than the limit; with a small --limit a full
        # batch would otherwise be read and processed only to be discarded by _consume.
        read_batch_size = min(self._batch_size, self._limit) if self._limit else self._batch_size
        if mode == "retry":
            ids = list(retry_ids or [])
            total = len(ids) if self._limit is None else min(len(ids), self._limit)
            for chunk in _chunks(ids, _RETRY_CHUNK_SIZE):
                condition = compose_condition(
                    saved_search_id=self._saved_search_id, resume_after=0, retry_ids=chunk
                )
                _, batches = self._source.read(condition=condition, batch_size=read_batch_size)
                if self._consume(batches, total, phase="retry"):
                    return
            return

        condition = compose_condition(
            saved_search_id=self._saved_search_id, resume_after=resume_after, retry_ids=None
        )
        total, batches = self._source.read(condition=condition, batch_size=read_batch_size)
        effective_total = total if self._limit is None else min(total, self._limit)
        self._consume(batches, effective_total, phase="run")

    def _reset_counters(self) -> None:
        self._processed = 0
        self._ok = 0
        self._failed = 0
        self._last_error = None
        self._last_artifact_id = None

    def _consume(self, batches: Iterator[list[ReadItem]], total: int, *, phase: str) -> bool:
        """Process batches; return True if the document limit was reached."""

        for batch in batches:
            pending: list[RelativityDocument] = []
            limit_hit = False
            for item in batch:
                if self._limit is not None and self._processed >= self._limit:
                    limit_hit = True
                    break
                self._processed += 1
                self._last_artifact_id = item.artifact_id
                if item.error is not None:
                    self._record_read_failure(item, phase)
                    self._emit(total)
                elif self._sink is not None and item.document is not None:
                    pending.append(item.document)
                else:
                    # Dry run: a successful read is the final state.
                    self._ok += 1
                    self._emit(total)
            if self._sink is not None and pending:
                for result in self._sink(pending):
                    self._record_index_outcome(result, phase)
                    self._emit(total)
            if self._on_batch_complete is not None:
                self._on_batch_complete(self._snapshot(total))
            if limit_hit or (self._limit is not None and self._processed >= self._limit):
                return True
        return False

    def _record_read_failure(self, item: ReadItem, phase: str) -> None:
        self._failed += 1
        self._last_error = item.error
        if self._progress_log is not None:
            self._progress_log.record_error(
                item.artifact_id,
                item.error or "",
                phase=phase,
                stage="read",
                error_type=item.error_type,
            )

    def _record_index_outcome(self, result: DocumentResult, phase: str) -> None:
        if self._progress_log is not None:
            record_index_result(self._progress_log, result, phase=phase)
        if result.outcome in ("created", "overwritten"):
            self._ok += 1
        else:
            self._failed += 1
            self._last_error = result.error_message or result.outcome
        if result.artifact_id is not None:
            self._last_artifact_id = result.artifact_id

    def _emit(self, total: int) -> None:
        self._on_progress(self._snapshot(total))

    def _snapshot(self, total: int) -> ProgressSnapshot:
        return ProgressSnapshot(
            processed=self._processed,
            total=total,
            ok_count=self._ok,
            failed_count=self._failed,
            last_artifact_id=self._last_artifact_id,
            last_error=self._last_error,
        )
