"""Object Manager API wrapper."""

from __future__ import annotations

import json
import time
from uuid import UUID

import requests

from .base import BaseAPIClient
from .normalize import DocumentFetchError
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
        match field_id:
            case int(afid):
                body = {
                    "exportObject": {"ArtifactID": object_artifact_id},
                    "longTextField": {"ArtifactID": afid},
                }
            case UUID() as guid:
                body = {
                    "exportObject": {"ArtifactID": object_artifact_id},
                    "longTextField": {"Guid": str(guid)},
                }
            case _:
                raise TypeError(
                    f"field_id must be int or UUID, got {type(field_id).__name__}"
                )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            res = self._t.post(f"{self._ws_base}/StreamLongText", json=body)
            try:
                res.raise_for_status()
                return str(res.text)
            except requests.HTTPError as exc:
                is_retryable_503 = exc.response is not None and exc.response.status_code == 503
                if not is_retryable_503:
                    raise
                if attempt >= max_attempts:
                    raise DocumentFetchError(
                        f"StreamLongText failed after {max_attempts} attempts "
                        f"(object {object_artifact_id}, field {field_id})."
                    ) from exc
                time.sleep(2.5)

        raise RuntimeError("Unreachable retry loop termination.")
