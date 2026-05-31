"""Document models for RelativityOne exports."""

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, BeforeValidator


def _prep_datetime(value: object | None) -> str | datetime | None:
    match value:
        case datetime() as dt:
            return dt
        case str(text):
            return text.strip() or None
        case _:  # case None:
            return None


class RelativityDocument(BaseModel):
    """Represent a RelativityOne document before chunking."""

    artifact_id: int
    control_number: str
    extracted_text: str
    title: str | None = None
    primary_date_time: Annotated[datetime | None, BeforeValidator(_prep_datetime)] = None
    email_from: str | None = None
    email_to: list[str] | None = None
    email_cc: list[str] | None = None
    email_bcc: list[str] | None = None
    summary: str | None = None
    topic: str | None = None


@dataclass(frozen=True)
class FailedDocument:
    artifact_id: int | None
    raw_row: dict
    error: str


@dataclass(frozen=True)
class ReadResult:
    documents: list[RelativityDocument]
    failures: list[FailedDocument]
