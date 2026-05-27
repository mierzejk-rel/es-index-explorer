"""Document models for RelativityOne exports."""

from pydantic import BaseModel


class RelativityDocument(BaseModel):
    """Represent a RelativityOne document before chunking."""

    artifact_id: int
    control_number: str
    extracted_text: str
    primary_date_time: str | None = None
    email_from: str | None = None
    email_to: list[str] | None = None
    email_cc: list[str] | None = None
    email_bcc: list[str] | None = None
    summary: str | None = None
    topic: str | None = None
