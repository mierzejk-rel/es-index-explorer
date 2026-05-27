"""Read documents from a RelativityOne workspace."""

from __future__ import annotations

from collections.abc import Iterable

from ..config import Config, RelativityFieldSelector, is_field_configured
from .auth import get_authenticated_session
from .client import RelativityClient
from .models import RelativityDocument


class DocumentReadError(RuntimeError):
    """Raised when document retrieval fails."""


def _normalize_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_str_list(value: object | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _ensure_required_field(value: str | None, field_name: str) -> str:
    if value is None:
        raise DocumentReadError(f"Missing required field '{field_name}'.")
    return value


def _iter_fields(
    fields: dict[str, RelativityFieldSelector],
) -> Iterable[tuple[str, RelativityFieldSelector]]:
    return ((key, value) for key, value in fields.items() if is_field_configured(value))


def read_documents(config: Config, *, limit: int | None = None) -> list[RelativityDocument]:
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
    field_names = [key for key, value in _iter_fields(field_map)]
    for row in builder.export(field_names=field_names):
        extracted_text = _ensure_required_field(
            _normalize_str(row.get("extracted_text")), "extracted_text"
        )
        control_number = _ensure_required_field(
            _normalize_str(row.get("control_number")), "control_number"
        )
        doc = RelativityDocument(
            artifact_id=int(row["artifact_id"]),
            control_number=control_number,
            extracted_text=extracted_text,
            primary_date_time=_normalize_str(row.get("primary_date_time")),
            email_from=_normalize_str(row.get("email_from")),
            email_to=_normalize_str_list(row.get("email_to")),
            email_cc=_normalize_str_list(row.get("email_cc")),
            email_bcc=_normalize_str_list(row.get("email_bcc")),
            summary=_normalize_str(row.get("summary")),
            topic=_normalize_str(row.get("topic")),
        )
        documents.append(doc)
        if limit is not None and len(documents) >= limit:
            break

    return documents
