"""Object Manager API wrapper."""

from __future__ import annotations

import json
from uuid import UUID

from .base import BaseAPIClient
from .object_manager_models import (
    InitializeExportResponse,
    QueryRequest,
    QuerySlimResponse,
)


class ObjectManagerAPI(BaseAPIClient):
    """Low-level Object Manager API wrapper."""

    @property
    def _ws_base(self) -> str:
        return f"/Relativity.REST/api/Relativity.ObjectManager/v1/workspace/{self._ws}/object"

    def query_slim(self, body: dict[str, object]) -> QuerySlimResponse:
        res = self._t.post(f"{self._ws_base}/queryslim", json=body)
        res.raise_for_status()
        return QuerySlimResponse.model_validate_json(res.text)

    def export_initialize(self, request: QueryRequest) -> InitializeExportResponse:
        json_str = request.model_dump_json(by_alias=True, exclude_none=True)
        json_obj = json.loads(json_str)
        res = self._t.post(f"{self._ws_base}/initializeexport", json=json_obj)
        res.raise_for_status()
        return InitializeExportResponse.model_validate(res.json())

    def export_retrieve_next(
        self, run_id: UUID, batch_size: int
    ) -> list[dict[str, object]]:
        res = self._t.post(
            f"{self._ws_base}/RetrieveNextResultsBlockFromExport",
            json={"runID": str(run_id), "batchSize": batch_size},
        )
        res.raise_for_status()
        if res.text == "":
            return []
        return list(res.json())

    def stream_long_text(self, object_artifact_id: int, field_id: int | UUID) -> str:
        if isinstance(field_id, int):
            body = {
                "exportObject": {"ArtifactID": object_artifact_id},
                "longTextField": {"ArtifactID": field_id},
            }
        else:
            body = {
                "exportObject": {"ArtifactID": object_artifact_id},
                "longTextField": {"Guid": str(field_id)},
            }
        res = self._t.post(f"{self._ws_base}/StreamLongText", json=body)
        res.raise_for_status()
        return str(res.text)
