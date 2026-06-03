"""Document sources for the unified ingest engine.

A `DocumentSource` fetches RelativityOne documents in ascending Artifact ID order,
honoring an optional condition (saved-search scoping, resume, retry), and yields them
in batches of `ReadItem` (each is either a built document or a per-document read error).
The two implementations differ only in the Object Manager mechanism used:

- `QuerySlimSource`: stateless offset paging (robust to slow consumers; the default).
- `ExportSource`: a stateful export run/cursor (efficient bulk read; long text streamed).

Both build documents via the shared `document_factory`, so field mapping and row -> POJO
normalization live in one place.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable

from pydantic import ValidationError

from ..config import Config
from ..relativity.auth import get_authenticated_session
from ..relativity.client import RelativityClient
from ..relativity.conditions import Cond
from ..relativity.document_factory import build_relativity_document, relativity_field_map
from ..relativity.fluent import (
    ARTIFACT_ID_KEY,
    QueryBuilder,
    resolve_field_selectors,
    resolve_truncated_long_text,
)
from ..relativity.models import RelativityDocument
from ..relativity.normalize import DocumentFetchError, DocumentReadError
from ..relativity.object_manager_models import QuerySlimResponse

# Exceptions that mean "this single document could not be read/built" (vs a fatal error).
_READ_EXCEPTIONS = (ValidationError, ValueError, KeyError, DocumentReadError, DocumentFetchError)

_SORT_FIELD = "Artifact ID"


@dataclass(frozen=True)
class ReadItem:
    """One read attempt for a single document.

    Exactly one of `document` / `error` is populated. A read failure carries the
    error message and exception type so the engine can record it (stage="read").
    """

    artifact_id: int
    document: RelativityDocument | None = None
    error: str | None = None
    error_type: str | None = None


@runtime_checkable
class DocumentSource(Protocol):
    """Fetches documents in ascending Artifact ID order, honoring a condition."""

    def read(
        self, *, condition: Cond | None, batch_size: int
    ) -> tuple[int, Iterator[list[ReadItem]]]:
        """Return ``(total_count, batches)`` for the given condition.

        `total_count` is the number of documents matching the condition (used for
        progress denominators). `batches` yields lists of `ReadItem` in ascending
        Artifact ID order, each up to `batch_size` long.
        """
        ...


class SourceKind(StrEnum):
    """Object Manager read mechanism for document ingestion."""

    QUERYSLIM = "queryslim"
    EXPORT = "export"


def build_source(config: Config, source_kind: SourceKind) -> DocumentSource:
    """Construct the requested document source ("queryslim" or "export")."""

    client = _client(config)
    raw_field_map = relativity_field_map(config.relativity.fields)
    resolved_field_map = resolve_field_selectors(client.object_manager, raw_field_map)

    if source_kind is SourceKind.QUERYSLIM:
        return QuerySlimSource(config, client, resolved_field_map)
    if source_kind is SourceKind.EXPORT:
        return ExportSource(config, client, resolved_field_map)
    raise ValueError(f"Unknown source '{source_kind}'; expected one of {list(SourceKind)}.")


def _client(config: Config) -> RelativityClient:
    session = get_authenticated_session(config).session
    return RelativityClient(config.relativity.host, config.relativity.workspace_id, session)


class QuerySlimSource:
    """Read documents via Object Manager QuerySlim with stateless offset paging."""

    def __init__(self, config: Config, client: RelativityClient, field_map: dict[str, str]) -> None:
        self._config = config
        self._client = client
        self._field_map = field_map

    def read(
        self, *, condition: Cond | None, batch_size: int
    ) -> tuple[int, Iterator[list[ReadItem]]]:
        builder, field_names = self._build_query_builder()
        if condition is not None:
            builder = builder.where(condition)
        builder = builder.sort_by(_SORT_FIELD, direction="Ascending")
        builder = builder.max_text_length(self._config.relativity.max_text_length)

        first = builder.page(0, batch_size).execute_raw()
        total = first.TotalCount
        long_text_field_ids = self._long_text_field_ids(first, field_names)

        def _iter() -> Iterator[list[ReadItem]]:
            response = first
            # OM QuerySlim uses 1-based start; start=0 returns positions 1..batch_size, so the
            # next page must begin at batch_size+1. Seed current_start=1 so the first increment
            # yields 1+batch_size.
            current_start = 1
            while response.Objects:
                yield self._to_read_items(response, field_names, long_text_field_ids)
                current_start += batch_size
                response = builder.page(current_start, batch_size).execute_raw()

        return total, _iter()

    def _build_query_builder(self) -> tuple[QueryBuilder, list[str]]:
        builder = self._client.query_object_manager().from_documents()
        builder = builder.select(**self._field_map)
        return builder, list(self._field_map.keys())

    def _to_read_items(
        self,
        response: QuerySlimResponse,
        field_names: list[str],
        long_text_field_ids: dict[str, int],
    ) -> list[ReadItem]:
        items: list[ReadItem] = []
        for obj in response.Objects:
            row: dict[str, object] = {ARTIFACT_ID_KEY: obj.ArtifactID}
            for name, value in zip(field_names, obj.Values):
                row[name] = value
            artifact_id = int(obj.ArtifactID)
            try:
                resolve_truncated_long_text(
                    self._client.object_manager,
                    object_artifact_id=artifact_id,
                    row=row,
                    field_ids=long_text_field_ids,
                )
                document = build_relativity_document(artifact_id=artifact_id, row=row)
            except _READ_EXCEPTIONS as exc:
                items.append(
                    ReadItem(artifact_id=artifact_id, error=str(exc), error_type=type(exc).__name__)
                )
                continue
            items.append(ReadItem(artifact_id=artifact_id, document=document))
        return items

    @staticmethod
    def _long_text_field_ids(
        response: QuerySlimResponse, field_names: list[str]
    ) -> dict[str, int]:
        if len(response.Fields) != len(field_names):
            raise ValueError("QuerySlim field count does not match selected field names.")
        field_ids: dict[str, int] = {}
        for idx, field_info in enumerate(response.Fields):
            if field_info.FieldType == "LongText":
                field_ids[field_names[idx]] = field_info.ArtifactID
        return field_ids


class ExportSource:
    """Read documents via the Object Manager export run (cursor) with long text streamed."""

    def __init__(self, config: Config, client: RelativityClient, field_map: dict[str, str]) -> None:
        self._config = config
        self._client = client
        self._field_map = field_map

    def read(
        self, *, condition: Cond | None, batch_size: int
    ) -> tuple[int, Iterator[list[ReadItem]]]:
        builder = self._client.query_object_manager().from_documents().select(**self._field_map)
        if condition is not None:
            builder = builder.where(condition)
        builder = builder.sort_by(_SORT_FIELD, direction="Ascending")
        field_names = list(self._field_map.keys())

        total, long_text_field_ids, rows = builder.export_with_total(
            batch_size=batch_size, field_names=field_names
        )

        def _iter() -> Iterator[list[ReadItem]]:
            batch: list[ReadItem] = []
            for row in rows:
                artifact_id = cast(int, row.get(ARTIFACT_ID_KEY))
                row_dict = dict(row)
                try:
                    resolve_truncated_long_text(
                        self._client.object_manager,
                        object_artifact_id=artifact_id,
                        row=row_dict,
                        field_ids=long_text_field_ids,
                    )
                    document = build_relativity_document(artifact_id=artifact_id, row=row_dict)
                    batch.append(ReadItem(artifact_id=artifact_id, document=document))
                except _READ_EXCEPTIONS as exc:
                    batch.append(
                        ReadItem(
                            artifact_id=artifact_id, error=str(exc), error_type=type(exc).__name__
                        )
                    )
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        return total, _iter()
