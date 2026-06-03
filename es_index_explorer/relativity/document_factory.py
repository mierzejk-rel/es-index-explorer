"""Shared construction of `RelativityDocument` from an Object Manager row.

Both the one-shot reader and the resumable batch importer use these helpers so the
field mapping (including ``title``) and the row -> POJO normalization live in one place.
"""

from datetime import datetime
from typing import cast

from ..config import RelativityFieldMappingConfig, RelativityFieldSelector, is_field_configured
from .models import RelativityDocument
from .normalize import (
    ensure_required_field,
    normalize_float,
    normalize_str,
    normalize_str_list,
    normalize_summary_topic,
)


def relativity_field_map(
    fields: RelativityFieldMappingConfig,
) -> dict[str, RelativityFieldSelector]:
    """Return the configured field selectors keyed by POJO field name.

    Only fields that are actually configured are included. The keys are the row
    keys produced by the reader/importer and consumed by `build_relativity_document`.
    """

    mapping: dict[str, RelativityFieldSelector] = {
        "extracted_text": fields.extracted_text,
        "control_number": fields.control_number,
        "title": fields.title,
        "primary_date_time": fields.primary_date_time,
        "email_from": fields.email_from,
        "email_to": fields.email_to,
        "email_cc": fields.email_cc,
        "email_bcc": fields.email_bcc,
        "summary": fields.summary,
        "topic": fields.topic,
        "extracted_text_size_kb": fields.extracted_text_size_kb,
    }
    return {key: value for key, value in mapping.items() if is_field_configured(value)}


def build_relativity_document(*, artifact_id: int, row: dict[str, object]) -> RelativityDocument:
    """Build a validated `RelativityDocument` from a normalized OM row.

    Parameters
    ----------
    artifact_id : int
        The Relativity document ArtifactId.
    row : dict[str, object]
        Field values keyed by POJO field name (see `relativity_field_map`).

    Raises
    ------
    DocumentReadError
        If a required field (`extracted_text`, `control_number`) is missing.
    pydantic.ValidationError
        If the row cannot be coerced into a valid document.
    """

    extracted_text = ensure_required_field(normalize_str(row.get("extracted_text")), "extracted_text")
    control_number = ensure_required_field(normalize_str(row.get("control_number")), "control_number")
    summary, topic = normalize_summary_topic(row.get("summary"), row.get("topic"))
    return RelativityDocument(
        artifact_id=artifact_id,
        control_number=control_number,
        extracted_text=extracted_text,
        title=normalize_str(row.get("title")),
        # OM returns str (ISO date) or None; cast so Pydantic's BeforeValidator coerces at runtime.
        primary_date_time=cast(datetime | None, row.get("primary_date_time")),
        email_from=normalize_str(row.get("email_from")),
        email_to=normalize_str_list(row.get("email_to")),
        email_cc=normalize_str_list(row.get("email_cc")),
        email_bcc=normalize_str_list(row.get("email_bcc")),
        summary=summary,
        topic=topic,
        extracted_text_size_kb=normalize_float(row.get("extracted_text_size_kb")),
    )
