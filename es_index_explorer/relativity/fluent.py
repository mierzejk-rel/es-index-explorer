"""Fluent query builder for Relativity Object Manager."""

from __future__ import annotations

from typing import Generator, Literal, cast
from uuid import UUID

from .conditions import Cond, field
from .object_manager import ObjectManagerAPI
from .object_manager_models import (
    ObjectType,
    QueryRequest,
    QueryRequestParams,
    QuerySlimResponse,
    RelativityIdentifier,
    RelativityScalar,
    RelativityType,
    R1_OBJECT_MANAGER_TRUNCATE_TOKEN,
)


ARTIFACT_ID_KEY: str = "artifact_id"


class QueryResult:
    """QuerySlim response adapter."""

    def __init__(self, response: QuerySlimResponse, rename: dict[str, str]) -> None:
        self._resp = response
        self._rename = rename

    def to_dicts(self) -> list[dict[str, RelativityScalar | list[RelativityScalar]]]:
        data: list[dict[str, RelativityScalar | list[RelativityScalar]]] = []
        for obj in self._resp.Objects:
            row: dict[str, RelativityScalar | list[RelativityScalar]] = {
                ARTIFACT_ID_KEY: obj.ArtifactID
            }
            for f, v in zip(self._resp.Fields, obj.Values):
                key = self._rename.get(str(f.ArtifactID), f.Name)
                row[key] = v
            data.append(row)
        return data

    def to_scalar(self) -> RelativityScalar:
        if self._resp.TotalCount != 1:
            raise ValueError(
                f"to_scalar only valid for a single row; TotalCount={self._resp.TotalCount}"
            )
        values = self._resp.Objects[0].Values
        if len(values) != 1 or isinstance(values[0], list):
            raise ValueError("to_scalar requires exactly one selected scalar field")
        value = values[0]
        if isinstance(value, list):
            raise ValueError("to_scalar requires exactly one selected scalar field")
        return value


class QueryBuilder:
    """Fluent builder for Object Manager queryslim and export."""

    def __init__(self, api: ObjectManagerAPI) -> None:
        self.api = api
        self._object_type: dict[str, object] = {}
        self._fields: list[RelativityIdentifier] = []
        self._field_names: list[str] = []
        self._condition: str | None = None
        self._sorts: list[dict[str, object]] = []
        self._start: int = 0
        self._length: int = 100
        self._long_text_behavior: str = "Tokenized"
        self._max_text_length: int = 100_000

    def object_type_id(self, artifact_type_id: int) -> "QueryBuilder":
        self._object_type = {"ArtifactTypeID": artifact_type_id}
        return self

    def object_type_guid(self, guid: UUID) -> "QueryBuilder":
        self._object_type = {"Guid": str(guid)}
        return self

    def from_documents(self) -> "QueryBuilder":
        return self.object_type_id(10)

    def select(self, *names: str, **fields: int | UUID | str) -> "QueryBuilder":
        if names and fields:
            raise ValueError(
                "select accepts either positional names or keyword field map, not both"
            )
        if names:
            if not all(isinstance(n, str) for n in names):
                raise TypeError("positional select requires all string field names")
            self._field_names = list(names)
            self._fields = [RelativityIdentifier.build_identifier(n) for n in names]
            return self

        self._field_names = list(fields)
        self._fields = [
            RelativityIdentifier.build_identifier(v) for v in fields.values()
        ]
        return self

    def where(self, cond: Cond | str) -> "QueryBuilder":
        self._condition = cond.expr if isinstance(cond, Cond) else cond
        return self

    def from_saved_search(self, saved_search_id: int) -> "QueryBuilder":
        return self.where(field("Artifact ID").in_saved_search(saved_search_id))

    def sort_by(
        self,
        *field_identifiers: UUID | int | str,
        direction: Literal["Ascending", "Descending"] = "Ascending",
    ) -> "QueryBuilder":
        self._sorts = [
            {
                "FieldIdentifier": (
                    RelativityIdentifier.build_identifier(fid).model_dump(
                        by_alias=True, exclude_none=True
                    )
                ),
                "Order": 0,
                "Direction": direction,
            }
            for fid in field_identifiers
        ]
        return self

    def page(self, start: int, length: int) -> "QueryBuilder":
        self._start = start
        self._length = length
        return self

    def long_text(
        self, behavior: Literal["Default", "Tokenized"] = "Tokenized"
    ) -> "QueryBuilder":
        self._long_text_behavior = behavior
        return self

    def max_text_length(self, length: int) -> "QueryBuilder":
        self._max_text_length = length
        return self

    def build(self) -> dict[str, object]:
        fields = [v.model_dump(exclude_none=True, by_alias=True) for v in self._fields]
        inner: dict[str, object] = {
            "ObjectType": self._object_type,
            "fields": fields,
            "MaxCharactersForLongTextValues": self._max_text_length,
            "LongTextBehavior": self._long_text_behavior,
        }
        request: dict[str, object] = {
            "request": inner,
            "start": self._start,
            "length": self._length,
        }
        if isinstance(self._condition, dict):
            inner.update(self._condition)
        elif isinstance(self._condition, str):
            inner["condition"] = self._condition
        if self._sorts:
            inner["Sorts"] = self._sorts
        return request

    def execute(self) -> QueryResult:
        response = self.api.query_slim(self.build())
        rename: dict[str, str] = {}
        if self._field_names:
            for f, name in zip(response.Fields, self._field_names):
                rename[str(f.ArtifactID)] = name
        return QueryResult(response, rename)

    def execute_raw(self) -> QuerySlimResponse:
        return self.api.query_slim(self.build())

    def export(
        self, *, batch_size: int = 100, field_names: list[str] | None = None
    ) -> Generator[dict[str, RelativityScalar], None, None]:
        _, rows = self.export_with_total(batch_size=batch_size, field_names=field_names)
        return rows

    def export_with_total(
        self, *, batch_size: int = 100, field_names: list[str] | None = None
    ) -> tuple[int, Generator[dict[str, RelativityScalar], None, None]]:
        run_id, total_count, r1_schema = self._initialize_export(field_names)
        return total_count, self._export_rows(
            run_id=run_id, batch_size=batch_size, r1_schema=r1_schema
        )

    def _initialize_export(
        self, field_names: list[str] | None
    ) -> tuple[UUID, int, dict[str, tuple[int, RelativityType]]]:
        object_type = (
            ObjectType(**self._object_type)
            if self._object_type
            else ObjectType(ArtifactTypeID=10)
        )
        qr = QueryRequest(
            QueryRequest=QueryRequestParams(
                ObjectType=object_type,
                Fields=[f for f in self._fields],
                Condition=self._condition if isinstance(self._condition, str) else None,
                Sorts=self._sorts if self._sorts else None,
            ),
            start=0,
        )

        init = self.api.export_initialize(qr)

        if field_names is None:
            resolved_field_names = [str(fd.name) for fd in init.field_data]
        elif len(field_names) != len(init.field_data):
            raise ValueError(
                "Number of field names must match the export session field list."
            )
        else:
            resolved_field_names = field_names

        r1_schema: dict[str, tuple[int, RelativityType]] = {
            ARTIFACT_ID_KEY: (0, "WholeNumber")
        }
        r1_schema |= {
            key: (fd.artifact_id, fd.field_type)
            for key, fd in zip(resolved_field_names, init.field_data)
        }
        return init.run_id, init.record_count, r1_schema

    def _export_rows(
        self,
        *,
        run_id: UUID,
        batch_size: int,
        r1_schema: dict[str, tuple[int, RelativityType]],
    ) -> Generator[dict[str, RelativityScalar], None, None]:
        should_continue = True
        while should_continue:
            block = self.api.export_retrieve_next(run_id, batch_size)
            if not block:
                should_continue = False
                continue
            rows: list[dict[str, RelativityScalar]] = [
                dict(
                    zip(
                        r1_schema.keys(),
                        # line["Values"] is list at runtime; typed as object because export_retrieve_next returns dict[str, object].
                        [line["ArtifactID"]] + cast(list[object], line["Values"]),
                    )
                )
                for line in block
            ]

            for row in rows:
                for key, value in row.items():
                    if (
                        isinstance(value, str)
                        and value == R1_OBJECT_MANAGER_TRUNCATE_TOKEN
                    ):
                        field_id = r1_schema[key][0]
                        row[key] = self.api.stream_long_text(
                            # ARTIFACT_ID_KEY is always an int set from obj.ArtifactID; typed as RelativityScalar because row is dict[str, RelativityScalar].
                            cast(int, row[ARTIFACT_ID_KEY]),
                            field_id)
                yield row
