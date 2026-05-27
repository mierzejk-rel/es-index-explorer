"""JSONL progress log for batch imports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ImportState:
    last_processed_id: int
    ok_count: int
    failed: dict[int, str]


class ProgressLog:
    def __init__(
        self,
        *,
        host: str,
        workspace_id: int,
        saved_search_id: int | None,
        directory: Path,
    ) -> None:
        self._path = self._build_path(host, workspace_id, saved_search_id, directory)
        self._handle = None

    @property
    def path(self) -> Path:
        return self._path

    def reset(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def load(self) -> ImportState:
        last_processed_id = 0
        ok_count = 0
        status_by_id: dict[int, str] = {}
        error_by_id: dict[int, str] = {}
        if not self._path.exists():
            return ImportState(last_processed_id, ok_count, {})
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                artifact_id = int(record["artifact_id"])
                status = str(record["status"])
                phase = str(record.get("phase", "run"))
                if phase == "run":
                    last_processed_id = max(last_processed_id, artifact_id)
                status_by_id[artifact_id] = status
                if status == "error":
                    error_by_id[artifact_id] = str(record.get("error", ""))
                elif artifact_id in error_by_id:
                    del error_by_id[artifact_id]
        ok_count = sum(1 for status in status_by_id.values() if status == "ok")
        return ImportState(last_processed_id, ok_count, error_by_id)

    def record_ok(self, artifact_id: int, *, phase: str = "run") -> None:
        self._append({"artifact_id": artifact_id, "status": "ok", "phase": phase})

    def record_error(self, artifact_id: int, error: str, *, phase: str = "run") -> None:
        self._append(
            {"artifact_id": artifact_id, "status": "error", "error": error, "phase": phase}
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _append(self, record: dict[str, object]) -> None:
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    @staticmethod
    def _build_path(
        host: str,
        workspace_id: int,
        saved_search_id: int | None,
        directory: Path,
    ) -> Path:
        suffix = "all" if saved_search_id is None else str(saved_search_id)
        filename = f"import_{host}_{workspace_id}_{suffix}.jsonl"
        return directory / filename
