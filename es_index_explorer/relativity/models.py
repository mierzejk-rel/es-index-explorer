"""Document models for RelativityOne exports."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel


class RelativityDocument(BaseModel):
    """Represent a RelativityOne document before chunking."""

    artifact_id: int
    control_number: str
    extracted_text: str
    primary_date_time: datetime | None = None
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
