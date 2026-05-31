"""Read documents from a RelativityOne workspace."""

from __future__ import annotations

import logging
from typing import Callable, cast

from pydantic import ValidationError

from ..config import Config
from .auth import get_authenticated_session
from .client import RelativityClient
from .document_factory import build_relativity_document, relativity_field_map
from .fluent import ARTIFACT_ID_KEY
from .models import FailedDocument, ReadResult, RelativityDocument
from .normalize import DocumentReadError
from .tui import ProgressSnapshot

logger = logging.getLogger(__name__)


def read_documents(
    config: Config,
    *,
    limit: int | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
) -> ReadResult:
    """Read RelativityOne documents using the Object Manager export API."""

    auth = get_authenticated_session(config)
    client = RelativityClient(config.relativity.host, config.relativity.workspace_id, auth.session)

    field_map = relativity_field_map(config.relativity.fields)

    builder = client.query_object_manager().from_documents().select(**field_map)

    if config.relativity.saved_search_id is not None:
        builder = builder.from_saved_search(config.relativity.saved_search_id)

    documents: list[RelativityDocument] = []
    failures: list[FailedDocument] = []
    field_names = list(field_map.keys())
    total, rows = builder.export_with_total(field_names=field_names)
    processed = 0
    ok_count = 0
    failed_count = 0
    last_error: str | None = None
    if on_progress is not None and total == 0:
        on_progress(
            ProgressSnapshot(
                processed=0,
                total=0,
                ok_count=0,
                failed_count=0,
                last_artifact_id=None,
                last_error=None,
            )
        )

    for row in rows:
        processed += 1
        # ARTIFACT_ID_KEY is always int in export rows; typed as RelativityScalar from dict[str, RelativityScalar].
        artifact_id = cast(int, row.get(ARTIFACT_ID_KEY))
        try:
            doc = build_relativity_document(artifact_id=artifact_id, row=dict(row))
            documents.append(doc)
            ok_count += 1
        except (ValidationError, ValueError, KeyError, DocumentReadError) as exc:
            last_error = str(exc)
            logger.warning("Skipping artifact %s: %s", artifact_id, exc)
            failures.append(
                FailedDocument(
                    artifact_id=artifact_id,
                    raw_row=dict(row),
                    error=last_error,
                )
            )
            failed_count += 1
        if on_progress is not None:
            on_progress(
                ProgressSnapshot(
                    processed=processed,
                    total=total,
                    ok_count=ok_count,
                    failed_count=failed_count,
                    last_artifact_id=artifact_id,
                    last_error=last_error,
                )
            )
        if limit is not None and len(documents) >= limit:
            break

    return ReadResult(documents=documents, failures=failures)
