"""Read documents from a RelativityOne workspace."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import logging
from typing import cast

from pydantic import ValidationError

from ..config import Config, RelativityFieldSelector, is_field_configured
from .auth import get_authenticated_session
from .client import RelativityClient
from .fluent import ARTIFACT_ID_KEY
from .models import FailedDocument, ReadResult, RelativityDocument
from .normalize import (
    ensure_required_field,
    normalize_str,
    normalize_str_list,
    normalize_summary_topic,
)

logger = logging.getLogger(__name__)


def _iter_fields(
    fields: dict[str, RelativityFieldSelector],
) -> Iterable[tuple[str, RelativityFieldSelector]]:
    return ((key, value) for key, value in fields.items() if is_field_configured(value))


def read_documents(config: Config, *, limit: int | None = None) -> ReadResult:
    """Read RelativityOne documents using the Object Manager export API."""

    auth = get_authenticated_session(config)
    client = RelativityClient(config.relativity.host, config.relativity.workspace_id, auth.session)

    fields = config.relativity.fields
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

    builder = client.query_object_manager().from_documents().select(
        **{key: value for key, value in _iter_fields(field_map)}
    )

    if config.relativity.saved_search_id is not None:
        builder = builder.from_saved_search(config.relativity.saved_search_id)

    documents: list[RelativityDocument] = []
    failures: list[FailedDocument] = []
    field_names = [key for key, value in _iter_fields(field_map)]
    for row in builder.export(field_names=field_names):
        extracted_text = ensure_required_field(
            normalize_str(row.get("extracted_text")), "extracted_text"
        )
        control_number = ensure_required_field(
            normalize_str(row.get("control_number")), "control_number"
        )
        try:
            summary, topic = normalize_summary_topic(
                row.get("summary"),
                row.get("topic"),
            )
            doc = RelativityDocument(
                # ARTIFACT_ID_KEY is always an int (set from obj.ArtifactID); typed as RelativityScalar because row is dict[str, RelativityScalar].
                artifact_id=cast(int, row[ARTIFACT_ID_KEY]),
                control_number=control_number,
                extracted_text=extracted_text,
                # OM returns str (ISO date) or None; cast to datetime | None so Pydantic's BeforeValidator (_prep_datetime) coerces at runtime.
                primary_date_time=cast(datetime | None, row.get("primary_date_time")),
                email_from=normalize_str(row.get("email_from")),
                email_to=normalize_str_list(row.get("email_to")),
                email_cc=normalize_str_list(row.get("email_cc")),
                email_bcc=normalize_str_list(row.get("email_bcc")),
                summary=summary,
                topic=topic,
            )
            documents.append(doc)
        except (ValidationError, ValueError, KeyError) as exc:
            # ARTIFACT_ID_KEY is always int; typed as RelativityScalar | None from .get() on dict[str, RelativityScalar].
            artifact_id = cast(int, row.get(ARTIFACT_ID_KEY))
            logger.warning("Skipping artifact %s: %s", artifact_id, exc)
            failures.append(
                FailedDocument(
                    artifact_id=artifact_id,
                    raw_row=dict(row),
                    error=str(exc),
                )
            )
        if limit is not None and len(documents) >= limit:
            break

    return ReadResult(documents=documents, failures=failures)
