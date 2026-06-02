"""Pydantic models for Object Manager query/export APIs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, field_serializer, model_validator

from .base_models import HiddenInputBaseModel

R1_OBJECT_MANAGER_TRUNCATE_TOKEN = "#KCURA99DF2F0FEB88420388879F1282A55760#"
class LongTextBehavior(Enum):
    """Long text retrieval mode for Object Manager queries."""

    DEFAULT = "Default"
    TOKENIZED = "Tokenized"


class SingleChoice(HiddenInputBaseModel):
    """Single-choice field value."""

    Name: str
    Guids: list[UUID]
    ArtifactID: int


LongText = str
RelativityScalar = LongText | int | bool | float | SingleChoice | None
RelativityType = Literal[
    "WholeNumber",
    "Decimal",
    "FixedLengthText",
    "LongText",
    "Date",
    "YesNo",
    "SingleChoice",
    "MultipleChoice",
    "MultipleObject",
]


class RelativityField(HiddenInputBaseModel):
    ArtifactID: int
    FieldCategory: str
    FieldType: RelativityType
    Guids: list[str]
    Name: str
    ViewFieldID: int


class ObjectSlim(HiddenInputBaseModel):
    """Slim object shape returned by queryslim."""

    ArtifactID: int
    Values: list[RelativityScalar | list[RelativityScalar]]


class ObjectType(HiddenInputBaseModel):
    artifact_id: int | None = Field(default=None, alias="ArtifactID")
    name: str | None = Field(default=None, alias="Name")
    guids: list[str] = Field(default_factory=list, alias="Guids")
    artifact_type_id: int | None = Field(default=None, alias="ArtifactTypeID")

    model_config = {"populate_by_name": True}


class QuerySlimResponse(HiddenInputBaseModel):
    TotalCount: int
    ResultCount: int
    CurrentStartIndex: int
    Objects: list[ObjectSlim]
    Fields: list[RelativityField]
    ObjectType: ObjectType


class RelativityIdentifier(HiddenInputBaseModel):
    artifact_id: int | None = Field(default=None, alias="ArtifactID")
    guid: UUID | None = Field(default=None, alias="Guid")
    name: str | None = Field(default=None, alias="Name")

    @model_validator(mode="before")
    @classmethod
    def convert_scalar_inputs(cls, value: Any) -> Any:  # pragma: no cover - pydantic hook
        match value:
            # bool is a subclass of int; do not map it to ArtifactID
            case int(artifact_id) if not isinstance(artifact_id, bool):
                return {"ArtifactID": artifact_id}
            case UUID() as guid:
                return {"Guid": guid}
            case str(name):
                return {"Name": name}
            case _:
                return value

    @field_serializer("guid")
    def serialize_guid(self, value: UUID | None) -> str | None:
        return str(value) if value is not None else None

    @staticmethod
    def build_identifier(identifier: int | UUID | str) -> "RelativityIdentifier":
        match identifier:
            # Guard against bool being matched by int pattern
            case int(artifact_id) if not isinstance(artifact_id, bool):
                return RelativityIdentifier(ArtifactID=artifact_id)
            case UUID() as guid:
                return RelativityIdentifier(Guid=guid)
            case str(name):
                return RelativityIdentifier(Name=name)
            case _:
                raise TypeError(f"Invalid identifier type: {type(identifier).__name__}")

    model_config = {"populate_by_name": True}


class QueryRequestParams(HiddenInputBaseModel):
    object_type: ObjectType = Field(..., alias="ObjectType")
    fields: list[RelativityIdentifier] = Field(..., alias="Fields")
    condition: str | None = Field(default=None, alias="Condition")
    sorts: list[dict[str, object]] | None = Field(default=None, alias="Sorts")
    long_text_behavior: LongTextBehavior = Field(
        default=LongTextBehavior.TOKENIZED, alias="LongTextBehavior"
    )


class QueryRequest(HiddenInputBaseModel):
    query_request: QueryRequestParams = Field(..., alias="QueryRequest")
    start: int = Field(...)

    model_config = {"populate_by_name": True}


class FieldData(HiddenInputBaseModel):
    artifact_id: int = Field(..., alias="ArtifactID")
    field_type: RelativityType = Field(..., alias="FieldType")
    name: str = Field(..., alias="Name")


class InitializeExportResponse(HiddenInputBaseModel):
    run_id: UUID = Field(..., alias="RunID")
    record_count: int = Field(..., alias="RecordCount")
    field_data: list[FieldData] = Field(..., alias="FieldData")
