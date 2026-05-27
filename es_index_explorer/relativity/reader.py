"""Read documents from a RelativityOne workspace."""

from __future__ import annotations

from collections.abc import Iterable
import logging

from pydantic import ValidationError

from ..config import Config, RelativityFieldSelector, is_field_configured
from .auth import get_authenticated_session
from .client import RelativityClient
from .models import FailedDocument, ReadResult, RelativityDocument

logger = logging.getLogger(__name__)


class DocumentReadError(RuntimeError):
    """Raised when document retrieval fails."""


def _normalize_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_str_list(value: object | None) -> list[str] | None:
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    normalized = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized or None


def _normalize_datetime_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_POOR_QUALITY_TOPIC = "Poor Quality Extracted Text"


def _normalize_summary_topic(
    raw_summary: object | None,
    raw_topic: object | None,
) -> tuple[str | None, str | None]:
    summary = _normalize_str(raw_summary)
    topic = _normalize_str(raw_topic)
    if summary is None and topic is None:
        return None, None
    if topic == _POOR_QUALITY_TOPIC and summary is None:
        return "", ""
    return summary, topic


def _ensure_required_field(value: str | None, field_name: str) -> str:
    if value is None:
        raise DocumentReadError(f"Missing required field '{field_name}'.")
    return value


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

    if config.relativity.saved_search_id > 0:
        builder = builder.from_saved_search(config.relativity.saved_search_id)

    documents: list[RelativityDocument] = []
    failures: list[FailedDocument] = []
    field_names = [key for key, value in _iter_fields(field_map)]
    for row in builder.export(field_names=field_names):
        extracted_text = _ensure_required_field(
            _normalize_str(row.get("extracted_text")), "extracted_text"
        )
        control_number = _ensure_required_field(
            _normalize_str(row.get("control_number")), "control_number"
        )
        try:
            summary, topic = _normalize_summary_topic(
                row.get("summary"),
                row.get("topic"),
            )
            doc = RelativityDocument(
                artifact_id=int(row["artifact_id"]),
                control_number=control_number,
                extracted_text=extracted_text,
                primary_date_time=_normalize_datetime_str(row.get("primary_date_time")),
                email_from=_normalize_str(row.get("email_from")),
                email_to=_normalize_str_list(row.get("email_to")),
                email_cc=_normalize_str_list(row.get("email_cc")),
                email_bcc=_normalize_str_list(row.get("email_bcc")),
                summary=summary,
                topic=topic,
            )
            documents.append(doc)
        except (ValidationError, ValueError, KeyError) as exc:
            artifact_value = row.get("artifact_id")
            artifact_id = None
            if artifact_value is not None:
                try:
                    artifact_id = int(artifact_value)
                except (TypeError, ValueError):
                    artifact_id = None
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
