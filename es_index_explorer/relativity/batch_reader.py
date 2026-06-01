"""QuerySlim-based batch importer with resume and retry support."""

from __future__ import annotations

from typing import Callable, Iterable

from pydantic import ValidationError

from ..config import Config, RelativityFieldSelector
from ..indexing.document_builder import DocumentResult
from .auth import get_authenticated_session
from .client import RelativityClient
from .conditions import Cond, field
from .document_factory import build_relativity_document, relativity_field_map
from .fluent import ARTIFACT_ID_KEY, QueryBuilder
from .models import RelativityDocument
from .normalize import DocumentReadError
from .object_manager_models import (
    QuerySlimResponse,
    R1_OBJECT_MANAGER_TRUNCATE_TOKEN,
)
from .progress import ImportState, ProgressLog
from .tui import ProgressSnapshot

DocumentSink = Callable[[list[RelativityDocument]], list[DocumentResult]]


class BatchImporter:
    def __init__(
        self,
        config: Config,
        progress_log: ProgressLog,
        *,
        on_progress: Callable[[ProgressSnapshot], None] | None = None,
        sink: DocumentSink | None = None,
        limit: int | None = None,
    ) -> None:
        self._config = config
        self._progress_log = progress_log
        self._client = RelativityClient(
            config.relativity.host, config.relativity.workspace_id, progress_log_session(config)
        )
        self._on_progress = on_progress or (lambda _: None)
        self._sink = sink
        self._limit = limit
        self._last_error: str | None = None

    def run(self, *, resume_after: int) -> ImportState:
        builder, field_names = self._build_query_builder()
        condition = self._compose_condition(resume_after)
        if condition is not None:
            builder = builder.where(condition)
        builder = builder.sort_by("Artifact ID", direction="Ascending")
        builder = builder.max_text_length(self._config.relativity.max_text_length)

        response = self._execute_page(builder, start=0)
        total = response.TotalCount
        effective_total = total if self._limit is None else min(total, self._limit)

        if response.TotalCount == 0:
            return self._progress_log.load()

        long_text_columns = self._long_text_columns(response, field_names)

        processed = 0
        ok_count = 0
        failed_count = 0
        # OM QuerySlim uses 1-based start; the initial fetch used start=0 which OM treats as start=1
        # (positions 1..batch_size). The next page must begin at position batch_size+1, so we seed
        # current_start=1 so the first loop increment yields 1+batch_size=batch_size+1.
        current_start = 1
        batch_size = self._config.relativity.batch_size

        while response.Objects:
            processed, ok_count, failed_count = self._process_response(
                response,
                field_names,
                long_text_columns,
                processed,
                ok_count,
                failed_count,
                effective_total,
            )
            if self._limit is not None and processed >= self._limit:
                break
            current_start += batch_size
            response = self._execute_page(builder, start=current_start)

        return self._progress_log.load()

    def retry_failed(self, failed_ids: list[int]) -> None:
        if self._limit is not None:
            failed_ids = failed_ids[: self._limit]
        if not failed_ids:
            return
        builder, field_names = self._build_query_builder()
        builder = builder.max_text_length(self._config.relativity.max_text_length)
        chunks = self._chunk_ids(failed_ids, size=1000)
        total = len(failed_ids)
        processed = 0
        ok_count = 0
        failed_count = len(failed_ids)
        for chunk in chunks:
            condition = field("Artifact ID").in_(chunk)
            saved_search_id = self._config.relativity.saved_search_id
            if saved_search_id is not None:
                condition = condition & field("Artifact ID").in_saved_search(saved_search_id)
            response = self._execute_page(builder.where(condition), start=0)
            long_text_columns = self._long_text_columns(response, field_names)
            if response.Objects:
                processed, ok_count, failed_count = self._process_response(
                    response,
                    field_names,
                    long_text_columns,
                    processed,
                    ok_count,
                    failed_count,
                    total,
                    phase="retry",
                )

    def _build_query_builder(self) -> tuple["QueryBuilder", list[str]]:
        builder = self._client.query_object_manager().from_documents()
        field_map = self._field_map()
        builder = builder.select(**{key: value for key, value in field_map.items()})
        return builder, list(field_map.keys())

    def _field_map(self) -> dict[str, RelativityFieldSelector]:
        return relativity_field_map(self._config.relativity.fields)

    def _compose_condition(self, resume_after: int) -> Cond | None:
        saved_search_id = self._config.relativity.saved_search_id
        condition = None
        if saved_search_id is not None:
            condition = field("Artifact ID").in_saved_search(saved_search_id)
        if resume_after > 0:
            resume_cond = field("Artifact ID").gt(resume_after)
            condition = resume_cond if condition is None else condition & resume_cond
        if saved_search_id is not None:
            self._validate_saved_search(condition)
        return condition

    def _validate_saved_search(self, condition: Cond | None) -> None:
        builder, _ = self._build_query_builder()
        if condition is not None:
            builder = builder.where(condition)
        builder = builder.page(0, 1).max_text_length(self._config.relativity.max_text_length)
        builder.execute_raw()

    def _execute_page(self, builder: "QueryBuilder", *, start: int) -> QuerySlimResponse:
        return builder.page(start, self._config.relativity.batch_size).execute_raw()

    @staticmethod
    def _long_text_columns(
            response: QuerySlimResponse,
            field_names: list[str],
    ) -> list[tuple[int, int, str]]:
        if len(response.Fields) != len(field_names):
            raise ValueError("QuerySlim field count does not match selected field names.")
        long_text_columns = []
        for idx, field_info in enumerate(response.Fields):
            if field_info.FieldType == "LongText":
                long_text_columns.append((idx, field_info.ArtifactID, field_names[idx]))
        return long_text_columns

    def _process_response(
        self,
        response: QuerySlimResponse,
        field_names: list[str],
        long_text_columns: list[tuple[int, int, str]],
        processed: int,
        ok_count: int,
        failed_count: int,
        total: int,
        *,
        phase: str = "run",
    ) -> tuple[int, int, int]:
        pending: list[RelativityDocument] = []
        for obj in response.Objects:
            if self._limit is not None and processed >= self._limit:
                break
            row: dict[str, object] = {ARTIFACT_ID_KEY: obj.ArtifactID}
            for name, value in zip(field_names, obj.Values):
                row[name] = value

            last_artifact_id = int(obj.ArtifactID)
            processed += 1
            try:
                row = self._resolve_long_text(
                    artifact_id=last_artifact_id,
                    row=row,
                    long_text_columns=long_text_columns,
                )
                doc = build_relativity_document(artifact_id=last_artifact_id, row=row)
            except (ValidationError, ValueError, KeyError, DocumentReadError) as exc:
                self._last_error = str(exc)
                # noinspection PyTypeChecker
                self._progress_log.record_error(
                    last_artifact_id, self._last_error, phase=phase, stage="read",
                    error_type=type(exc).__name__,
                )
                failed_count += 1
                self._emit(processed, total, ok_count, failed_count, last_artifact_id)
                continue

            if self._sink is None:
                # Read-only mode (no indexing): a successful read counts as ok.
                self._progress_log.record_ok(last_artifact_id, phase=phase, outcome="read")
                ok_count += 1
                self._emit(processed, total, ok_count, failed_count, last_artifact_id)
            else:
                pending.append(doc)

        if self._sink is not None and pending:
            for result in self._sink(pending):
                ok_count, failed_count = self._record_index_result(result, phase, ok_count, failed_count)
                self._emit(processed, total, ok_count, failed_count, result.artifact_id)

        return processed, ok_count, failed_count

    def _record_index_result(
        self, result: DocumentResult, phase: str, ok_count: int, failed_count: int
    ) -> tuple[int, int]:
        if result.artifact_id is None:
            return ok_count, failed_count
        if result.outcome in ("created", "overwritten"):
            self._progress_log.record_ok(result.artifact_id, phase=phase, outcome=result.outcome)
            return ok_count + 1, failed_count
        self._last_error = result.error_message or result.outcome
        # noinspection PyTypeChecker
        self._progress_log.record_error(
            result.artifact_id, self._last_error, phase=phase,
            stage=result.stage or "index", error_type=result.error_type,
        )
        return ok_count, failed_count + 1

    def _emit(
        self, processed: int, total: int, ok_count: int, failed_count: int, last_artifact_id: int | None
    ) -> None:
        self._on_progress(
            ProgressSnapshot(
                processed=processed,
                total=total,
                ok_count=ok_count,
                failed_count=failed_count,
                last_artifact_id=last_artifact_id,
                last_error=self._last_error,
            )
        )

    def _resolve_long_text(
        self,
        *,
        artifact_id: int,
        row: dict[str, object],
        long_text_columns: list[tuple[int, int, str]],
    ) -> dict[str, object]:
        for idx, field_artifact_id, field_name in long_text_columns:
            value = row.get(field_name)
            if not (isinstance(value, str) and value == R1_OBJECT_MANAGER_TRUNCATE_TOKEN):
                continue
            streamed = self._client.object_manager.stream_long_text(
                artifact_id, field_artifact_id
            )
            if field_name == "extracted_text":
                if streamed == R1_OBJECT_MANAGER_TRUNCATE_TOKEN:
                    raise DocumentReadError(
                        "Truncation token still present after StreamLongText."
                    )
                row[field_name] = streamed
            else:
                row[field_name] = streamed
        return row

    @staticmethod
    def _chunk_ids(ids: list[int], size: int) -> Iterable[list[int]]:
        for idx in range(0, len(ids), size):
            yield ids[idx : idx + size]


def progress_log_session(config: Config):
    return get_authenticated_session(config).session
