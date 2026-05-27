"""QuerySlim-based batch importer with resume and retry support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from pydantic import ValidationError

from ..config import Config, RelativityFieldSelector, is_field_configured
from .auth import get_authenticated_session
from .client import RelativityClient
from .conditions import Cond, field
from .fluent import ARTIFACT_ID_KEY, QueryBuilder
from .models import RelativityDocument
from .normalize import (
    DocumentReadError,
    ensure_required_field,
    normalize_str,
    normalize_str_list,
    normalize_summary_topic,
)
from .object_manager_models import (
    QuerySlimResponse,
    R1_OBJECT_MANAGER_TRUNCATE_TOKEN,
)
from .progress import ImportState, ProgressLog

@dataclass
class ProgressSnapshot:
    processed: int
    total: int
    ok_count: int
    failed_count: int
    last_artifact_id: int | None
    last_error: str | None


class BatchImporter:
    def __init__(
        self,
        config: Config,
        progress_log: ProgressLog,
        *,
        on_progress: Callable[[ProgressSnapshot], None] | None = None,
    ) -> None:
        self._config = config
        self._progress_log = progress_log
        self._client = RelativityClient(
            config.relativity.host, config.relativity.workspace_id, progress_log_session(config)
        )
        self._on_progress = on_progress or (lambda _: None)
        self._last_error: str | None = None

    def run(self, *, resume_after: int, state: ImportState) -> ImportState:
        builder, field_names = self._build_query_builder()
        condition = self._compose_condition(resume_after)
        if condition is not None:
            builder = builder.where(condition)
        builder = builder.sort_by("Artifact ID", direction="Ascending")
        builder = builder.max_text_length(self._config.relativity.max_text_length)

        response = self._execute_page(builder, start=0)
        total = state.ok_count + len(state.failed) + response.TotalCount
        long_text_columns = self._long_text_columns(response, field_names)

        processed = state.ok_count + len(state.failed)
        ok_count = state.ok_count
        failed_count = len(state.failed)
        current_start = 0
        batch_size = self._config.relativity.batch_size

        while response.Objects:
            processed, ok_count, failed_count = self._process_response(
                response,
                field_names,
                long_text_columns,
                processed,
                ok_count,
                failed_count,
                total,
            )
            current_start += batch_size
            response = self._execute_page(builder, start=current_start)

        return self._progress_log.load()

    def retry_failed(self, failed_ids: list[int]) -> None:
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
        fields = self._config.relativity.fields
        field_map = {
            "extracted_text": fields.extracted_text,
            "control_number": fields.control_number,
            "primary_date_time": fields.primary_date_time,
            "email_from": fields.email_from,
            "email_to": fields.email_to,
            "email_cc": fields.email_cc,
            "email_bcc": fields.email_bcc,
            "summary": fields.summary,
            "topic": fields.topic,
        }
        return {key: value for key, value in field_map.items() if is_field_configured(value)}

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
        for obj in response.Objects:
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
                extracted_text = ensure_required_field(
                    normalize_str(row.get("extracted_text")), "extracted_text"
                )
                control_number = ensure_required_field(
                    normalize_str(row.get("control_number")), "control_number"
                )
                summary, topic = normalize_summary_topic(
                    row.get("summary"),
                    row.get("topic"),
                )
                # noinspection PyTypeChecker
                RelativityDocument(
                    artifact_id=last_artifact_id,
                    control_number=control_number,
                    extracted_text=extracted_text,
                    primary_date_time=row.get("primary_date_time"),
                    email_from=normalize_str(row.get("email_from")),
                    email_to=normalize_str_list(row.get("email_to")),
                    email_cc=normalize_str_list(row.get("email_cc")),
                    email_bcc=normalize_str_list(row.get("email_bcc")),
                    summary=summary,
                    topic=topic,
                )
                self._progress_log.record_ok(last_artifact_id, phase=phase)
                ok_count += 1
            except (ValidationError, ValueError, KeyError, DocumentReadError) as exc:
                error = str(exc)
                self._last_error = error
                self._progress_log.record_error(last_artifact_id, error, phase=phase)
                failed_count += 1

            snapshot = ProgressSnapshot(
                processed=processed,
                total=total,
                ok_count=ok_count,
                failed_count=failed_count,
                last_artifact_id=last_artifact_id,
                last_error=self._last_error,
            )
            self._on_progress(snapshot)

        # TODO: preprocess batch documents and save to ES index.
        return processed, ok_count, failed_count

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
