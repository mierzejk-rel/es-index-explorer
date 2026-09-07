"""Atomic persistence and SHA-256 helpers."""

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from es_index_explorer.question_analysis.contracts import (
    OUTCOME_ARTIFACT_SCHEMAS,
    PROTECTED_RECOMMENDATION_ARTIFACT_NAMES,
    SCHEMA_VERSION,
    ArtifactMetadata,
    FileFingerprint,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import MalformedInputError

if TYPE_CHECKING:
    import pandas as pd

ARTIFACT_DIRECTORIES = (
    "tables",
    "draws",
    "annotations",
    "gold",
    "statistics",
    "figures",
    "partial_reports",
    "logs",
)


def _validate_protected_parquet_schema(
    relative_path: str | Path, columns: Iterable[object]
) -> None:
    artifact_name = Path(relative_path).name
    expected = OUTCOME_ARTIFACT_SCHEMAS.get(artifact_name)
    if expected is None:
        if artifact_name in PROTECTED_RECOMMENDATION_ARTIFACT_NAMES:
            raise MalformedInputError(
                f"Protected outcome artifact schema is not registered: {artifact_name}"
            )
        return
    expected_columns = ("artifact_schema_version", *expected)
    actual_columns = tuple(str(column) for column in columns)
    if actual_columns != expected_columns:
        raise MalformedInputError(
            f"Protected outcome artifact schema mismatch: {artifact_name}"
        )


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path, *, label: str | None = None) -> FileFingerprint:
    """Create an immutable identity for an existing file."""
    if not path.is_file():
        raise MalformedInputError(f"Input file does not exist: {path}")
    return FileFingerprint(
        path=label or path.as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def versioned_frame(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Add the shared artifact schema version to a table."""
    versioned = frame.copy()
    versioned.insert(0, "artifact_schema_version", SCHEMA_VERSION)
    return versioned


def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    """Serialize JSON deterministically with one trailing newline."""
    payload = (
        value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    )
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace a file atomically after flushing its contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class ArtifactStore:
    """Safe access to one analysis root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def prepare(self) -> None:
        """Create the frozen artifact directory layout."""
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ARTIFACT_DIRECTORIES:
            (self.root / directory).mkdir(exist_ok=True)

    def path_for(self, relative_path: str | Path) -> Path:
        """Resolve a root-relative path and reject traversal."""
        relative = Path(relative_path)
        if relative.is_absolute():
            raise MalformedInputError(f"Artifact path must be relative: {relative}")
        resolved = (self.root / relative).resolve()
        if not resolved.is_relative_to(self.root):
            raise MalformedInputError(
                f"Artifact path escapes analysis root: {relative}"
            )
        return resolved

    def write_bytes(
        self,
        relative_path: str | Path,
        data: bytes,
        *,
        created_by: WorkflowCommand,
        now: datetime | None = None,
    ) -> ArtifactMetadata:
        """Atomically write an output and return its recorded identity."""
        target = self.path_for(relative_path)
        atomic_write_bytes(target, data)
        return ArtifactMetadata(
            path=target.relative_to(self.root).as_posix(),
            sha256=sha256(data).hexdigest(),
            size_bytes=len(data),
            created_by=created_by,
            created_at=now or datetime.now(UTC),
        )

    def write_json(
        self,
        relative_path: str | Path,
        value: BaseModel | Mapping[str, object],
        *,
        created_by: WorkflowCommand,
        now: datetime | None = None,
    ) -> ArtifactMetadata:
        """Atomically write canonical JSON and return its recorded identity."""
        return self.write_bytes(
            relative_path,
            canonical_json_bytes(value),
            created_by=created_by,
            now=now,
        )

    def write_parquet(
        self,
        relative_path: str | Path,
        frame: "pd.DataFrame",
        *,
        created_by: WorkflowCommand,
        now: datetime | None = None,
    ) -> ArtifactMetadata:
        """Write a deterministic, index-free Parquet artifact."""
        _validate_protected_parquet_schema(relative_path, frame.columns)
        buffer = BytesIO()
        frame.to_parquet(
            buffer,
            engine="pyarrow",
            index=False,
            compression="zstd",
        )
        return self.write_bytes(
            relative_path,
            buffer.getvalue(),
            created_by=created_by,
            now=now,
        )

    def read_model[T: BaseModel](
        self, relative_path: str | Path, model_type: type[T]
    ) -> T:
        """Read and validate a persisted JSON model."""
        path = self.path_for(relative_path)
        if not path.is_file():
            raise MalformedInputError(f"Required artifact does not exist: {path}")
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MalformedInputError(f"Invalid {path.name}: {error}") from error
