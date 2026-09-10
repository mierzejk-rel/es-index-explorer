"""Deterministic, tamper-evident storage contracts for Layer 1 checkpoints."""

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Annotated, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from es_index_explorer.question_analysis.contracts import FailureKind
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.storage import (
    atomic_write_bytes,
    canonical_json_bytes,
)

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_DIRECTORY = Path("checkpoints") / "fit-layer1"
CHECKPOINT_MANIFEST_NAME = "manifest.json"

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class Layer1CheckpointStatus(StrEnum):
    """Describe the durable lifecycle state of a Layer 1 checkpoint."""

    IN_PROGRESS = "in_progress"
    INTERRUPTED = "interrupted"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    FAILED = "failed"


class Layer1ShardArea(StrEnum):
    """Separate required scientific state from optional evaluation caches."""

    STATE = "state"
    CACHE = "cache"


class Layer1ShardFormat(StrEnum):
    """Identify the only serialization formats accepted for checkpoint shards."""

    JSON = "json"
    NPY = "npy"
    PARQUET = "parquet"


class Layer1TaskStatus(StrEnum):
    """Describe ordered computation and admission states for a checkpoint task."""

    COMPUTED_AVAILABLE = "COMPUTED_AVAILABLE"
    ADMITTED_RETAINED = "ADMITTED_RETAINED"
    REJECTED = "REJECTED"


class Layer1CheckpointCompatibilityIdentity(BaseModel):
    """Capture only fields that determine exact replay compatibility.

    Operational scheduling choices intentionally do not appear in this model.
    Consequently, executor and worker-count changes cannot alter its canonical
    compatibility hash.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=CHECKPOINT_SCHEMA_VERSION, ge=1)
    specification_sha256: Sha256Hex
    input_sha256s: dict[str, Sha256Hex] = Field(min_length=1)
    resource_sha256s: dict[str, Sha256Hex] = Field(default_factory=dict)
    numerical_contract_version: NonEmptyStr
    statistical_run_config: dict[str, JsonValue]
    seed_contract: dict[str, JsonValue]
    numerical_backend: NonEmptyStr
    architecture: NonEmptyStr
    float_abi: NonEmptyStr
    python_version: NonEmptyStr
    numpy_version: NonEmptyStr
    scipy_version: NonEmptyStr
    blas_lapack_identity: NonEmptyStr

    @field_validator("statistical_run_config", "seed_contract")
    @classmethod
    def _validate_json_fields(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _require_finite_json(value)
        return value


class Layer1ExecutionProvenance(BaseModel):
    """Record operational execution details excluded from compatibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_model: NonEmptyStr
    os_details: NonEmptyStr
    executor_type: NonEmptyStr
    worker_count: int = Field(ge=1)
    max_in_flight: int = Field(ge=1)
    progress_interval_seconds: float = Field(gt=0)
    auto_tuning_samples: dict[str, JsonValue] = Field(default_factory=dict)
    timing_observations: dict[str, JsonValue] = Field(default_factory=dict)
    memory_observations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "auto_tuning_samples",
        "timing_observations",
        "memory_observations",
    )
    @classmethod
    def _validate_json_fields(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _require_finite_json(value)
        return value


class Layer1TaskFailure(BaseModel):
    """Persist the stable category and concrete type of a rejected task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_kind: FailureKind
    failure_type: NonEmptyStr
    message: NonEmptyStr


class Layer1TaskLedgerEntry(BaseModel):
    """Persist one stable task identity and its ordered-admission state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: NonEmptyStr
    admission_group: NonEmptyStr
    admission_index: int = Field(ge=0)
    status: Layer1TaskStatus
    failure: Layer1TaskFailure | None = None

    @model_validator(mode="after")
    def _validate_failure_contract(self) -> Self:
        if self.status is Layer1TaskStatus.REJECTED and self.failure is None:
            raise ValueError("Rejected tasks require a typed failure.")
        if self.status is not Layer1TaskStatus.REJECTED and self.failure is not None:
            raise ValueError("Only rejected tasks may carry a failure.")
        return self


class Layer1CheckpointShard(BaseModel):
    """Describe one complete, content-addressed checkpoint shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: NonEmptyStr
    task_id: NonEmptyStr
    format: Layer1ShardFormat
    content_sha256: Sha256Hex
    size_bytes: int = Field(ge=0)
    compatibility_sha256: Sha256Hex
    npy_dtype: str | None = None
    npy_shape: tuple[int, ...] | None = None
    parquet_columns: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_format_metadata(self) -> Self:
        relative = Path(self.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Checkpoint shard paths must be root-relative.")
        if not relative.parts or relative.parts[0] not in {
            Layer1ShardArea.STATE.value,
            Layer1ShardArea.CACHE.value,
        }:
            raise ValueError("Checkpoint shards must be stored under state/ or cache/.")
        expected_suffix = f".{self.format.value}"
        if relative.suffix != expected_suffix:
            raise ValueError(f"{self.format.value} shards require {expected_suffix}.")
        if self.format is Layer1ShardFormat.NPY:
            if self.npy_dtype is None or self.npy_shape is None:
                raise ValueError("NPY shards require dtype and shape metadata.")
            if any(dimension < 0 for dimension in self.npy_shape):
                raise ValueError("NPY shard dimensions must be non-negative.")
            if self.parquet_columns is not None:
                raise ValueError("NPY shards cannot carry Parquet metadata.")
        elif self.format is Layer1ShardFormat.PARQUET:
            if self.parquet_columns is None:
                raise ValueError("Parquet shards require ordered column metadata.")
            if self.npy_dtype is not None or self.npy_shape is not None:
                raise ValueError("Parquet shards cannot carry NPY metadata.")
        elif any(
            value is not None
            for value in (self.npy_dtype, self.npy_shape, self.parquet_columns)
        ):
            raise ValueError("JSON shards cannot carry array or table metadata.")
        return self


class Layer1CheckpointManifest(BaseModel):
    """Persist compatibility, provenance, task ledger, and complete shard identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=CHECKPOINT_SCHEMA_VERSION, ge=1)
    compatibility_identity: Layer1CheckpointCompatibilityIdentity
    compatibility_sha256: Sha256Hex
    execution_provenance: Layer1ExecutionProvenance
    execution_provenance_sha256: Sha256Hex
    status: Layer1CheckpointStatus
    scientific_state_shards: tuple[Layer1CheckpointShard, ...] = ()
    optional_cache_shards: tuple[Layer1CheckpointShard, ...] = ()
    task_ledger: tuple[Layer1TaskLedgerEntry, ...] = ()
    created_at: datetime
    updated_at: datetime
    interruption_exit_code: int | None = None

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if self.compatibility_sha256 != compatibility_sha256(
            self.compatibility_identity
        ):
            raise ValueError("Compatibility hash does not match its identity.")
        if self.execution_provenance_sha256 != execution_provenance_sha256(
            self.execution_provenance
        ):
            raise ValueError("Execution provenance hash does not match its record.")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("Checkpoint timestamps must be timezone-aware.")
        if self.updated_at < self.created_at:
            raise ValueError("Checkpoint updated_at precedes created_at.")
        if self.status is Layer1CheckpointStatus.INTERRUPTED:
            if self.interruption_exit_code not in {130, 143}:
                raise ValueError(
                    "Interrupted checkpoints require exit code 130 or 143."
                )
        elif self.interruption_exit_code is not None:
            raise ValueError("Only interrupted checkpoints may carry an exit code.")
        self._validate_shard_collections()
        self._validate_ledger()
        return self

    def _validate_shard_collections(self) -> None:
        all_shards = self.scientific_state_shards + self.optional_cache_shards
        paths = [shard.path for shard in all_shards]
        if len(paths) != len(set(paths)):
            raise ValueError("Checkpoint shard paths must be unique.")
        for shard in self.scientific_state_shards:
            if Path(shard.path).parts[0] != Layer1ShardArea.STATE.value:
                raise ValueError("Scientific state shards must be stored under state/.")
            if shard.compatibility_sha256 != self.compatibility_sha256:
                raise ValueError("Scientific shard compatibility hash does not match.")
        for shard in self.optional_cache_shards:
            if Path(shard.path).parts[0] != Layer1ShardArea.CACHE.value:
                raise ValueError("Optional cache shards must be stored under cache/.")
            if shard.compatibility_sha256 != self.compatibility_sha256:
                raise ValueError("Cache shard compatibility hash does not match.")

    def _validate_ledger(self) -> None:
        task_ids = [entry.task_id for entry in self.task_ledger]
        positions = [
            (entry.admission_group, entry.admission_index) for entry in self.task_ledger
        ]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Checkpoint task IDs must be unique.")
        if len(positions) != len(set(positions)):
            raise ValueError(
                "Task admission positions must be unique within each group."
            )


@dataclass(frozen=True, slots=True)
class Layer1LoadedCheckpoint:
    """Hold verified required state and reusable optional cache payloads."""

    manifest: Layer1CheckpointManifest
    scientific_state: dict[str, bytes]
    optional_cache: dict[str, bytes]


class Layer1CheckpointSession:
    """Manage one compatible mutable-by-replacement checkpoint manifest."""

    def __init__(
        self,
        checkpoint_root: Path,
        manifest: Layer1CheckpointManifest,
    ) -> None:
        self.root = checkpoint_root
        self.manifest = manifest

    @classmethod
    def open(
        cls,
        analysis_root: Path,
        identity: Layer1CheckpointCompatibilityIdentity,
        provenance: Layer1ExecutionProvenance,
        *,
        resume: bool,
        fresh: bool,
    ) -> "Layer1CheckpointSession":
        """Open, create, or explicitly restart one Layer 1 checkpoint."""
        checkpoint_root = layer1_checkpoint_root(analysis_root)
        if fresh and checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
        manifest_path = checkpoint_root / CHECKPOINT_MANIFEST_NAME
        if manifest_path.is_file():
            if not resume:
                raise MalformedInputError(
                    "Layer 1 checkpoint exists; use --resume or --fresh"
                )
            manifest = read_checkpoint_manifest(
                checkpoint_root,
                expected_identity=identity,
            ).model_copy(
                update={
                    "execution_provenance": provenance,
                    "execution_provenance_sha256": execution_provenance_sha256(
                        provenance
                    ),
                    "status": Layer1CheckpointStatus.IN_PROGRESS,
                    "interruption_exit_code": None,
                    "updated_at": datetime.now(UTC),
                }
            )
        else:
            manifest = new_checkpoint_manifest(identity, provenance)
        return cls(
            checkpoint_root,
            write_checkpoint_manifest(checkpoint_root, manifest),
        )

    def commit_json_state(
        self,
        relative_name: str | Path,
        value: BaseModel | dict[str, object],
        *,
        task_id: str,
    ) -> Layer1CheckpointShard:
        """Commit one required canonical JSON scientific-state shard."""
        shard = write_json_checkpoint_shard(
            self.root,
            relative_name,
            value,
            area=Layer1ShardArea.STATE,
            task_id=task_id,
            compatibility_sha256_value=self.manifest.compatibility_sha256,
        )
        self._replace_shard(shard, scientific=True)
        return shard

    def commit_npy_state(
        self,
        relative_name: str | Path,
        array: np.ndarray,
        *,
        task_id: str,
    ) -> Layer1CheckpointShard:
        """Commit one required deterministic NPY scientific-state shard."""
        shard = write_npy_checkpoint_shard(
            self.root,
            relative_name,
            array,
            area=Layer1ShardArea.STATE,
            task_id=task_id,
            compatibility_sha256_value=self.manifest.compatibility_sha256,
        )
        self._replace_shard(shard, scientific=True)
        return shard

    def commit_json_cache(
        self,
        relative_name: str | Path,
        value: BaseModel | dict[str, object],
        *,
        task_id: str,
    ) -> Layer1CheckpointShard:
        """Commit one optional exact cache shard."""
        shard = write_json_checkpoint_shard(
            self.root,
            relative_name,
            value,
            area=Layer1ShardArea.CACHE,
            task_id=task_id,
            compatibility_sha256_value=self.manifest.compatibility_sha256,
        )
        self._replace_shard(shard, scientific=False)
        return shard

    def read_json_state(self, relative_name: str | Path) -> dict[str, object] | None:
        """Read one verified JSON state shard, returning None when absent."""
        path = (Path(Layer1ShardArea.STATE.value) / relative_name).as_posix()
        matching = [
            shard
            for shard in self.manifest.scientific_state_shards
            if shard.path == path
        ]
        if not matching:
            return None
        payload = read_scientific_checkpoint_shard(
            self.root,
            matching[0],
            expected_compatibility_sha256=self.manifest.compatibility_sha256,
        )
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise MalformedInputError("Layer 1 JSON state shard must contain an object")
        return parsed

    def read_npy_state(self, relative_name: str | Path) -> np.ndarray | None:
        """Read one verified NPY state shard, returning None when absent."""
        path = (Path(Layer1ShardArea.STATE.value) / relative_name).as_posix()
        matching = [
            shard
            for shard in self.manifest.scientific_state_shards
            if shard.path == path
        ]
        if not matching:
            return None
        payload = read_scientific_checkpoint_shard(
            self.root,
            matching[0],
            expected_compatibility_sha256=self.manifest.compatibility_sha256,
        )
        loaded = np.load(BytesIO(payload), allow_pickle=False)
        if not isinstance(loaded, np.ndarray):
            raise MalformedInputError("Layer 1 NPY state shard is invalid")
        return loaded

    def upsert_task(self, entry: Layer1TaskLedgerEntry) -> None:
        """Insert or replace one stable task-ledger entry."""
        ledger = tuple(
            current
            for current in self.manifest.task_ledger
            if current.task_id != entry.task_id
        ) + (entry,)
        self._persist(task_ledger=ledger)

    def mark_interrupted(self, exit_code: int) -> None:
        """Mark this session interrupted after committed units are flushed."""
        self.manifest = mark_checkpoint_interrupted(
            self.root,
            exit_code=exit_code,
        )

    def mark_completed(self) -> None:
        """Mark all checkpoint scientific work completed before promotion."""
        self._persist(status=Layer1CheckpointStatus.COMPLETED)

    def mark_failed(self) -> None:
        """Mark checkpoint execution failed without promoting artifacts."""
        self._persist(status=Layer1CheckpointStatus.FAILED)

    def _replace_shard(
        self,
        shard: Layer1CheckpointShard,
        *,
        scientific: bool,
    ) -> None:
        field_name = (
            "scientific_state_shards" if scientific else "optional_cache_shards"
        )
        existing = getattr(self.manifest, field_name)
        updated = tuple(
            current for current in existing if current.path != shard.path
        ) + (shard,)
        self._persist(**{field_name: updated})

    def _persist(self, **updates: object) -> None:
        updates["updated_at"] = datetime.now(UTC)
        payload = {**self.manifest.model_dump(), **updates}
        self.manifest = write_checkpoint_manifest(
            self.root,
            Layer1CheckpointManifest.model_validate(payload),
        )


def layer1_checkpoint_root(analysis_root: Path) -> Path:
    """Return the internal Layer 1 checkpoint root for an analysis workspace."""
    return analysis_root / CHECKPOINT_DIRECTORY


def compatibility_sha256(identity: Layer1CheckpointCompatibilityIdentity) -> str:
    """Hash only the canonical result-determining compatibility identity."""
    return sha256(canonical_json_bytes(identity)).hexdigest()


def execution_provenance_sha256(provenance: Layer1ExecutionProvenance) -> str:
    """Hash operational provenance without affecting checkpoint compatibility."""
    return sha256(canonical_json_bytes(provenance)).hexdigest()


def new_checkpoint_manifest(
    identity: Layer1CheckpointCompatibilityIdentity,
    provenance: Layer1ExecutionProvenance,
    *,
    now: datetime | None = None,
) -> Layer1CheckpointManifest:
    """Create an empty in-progress checkpoint manifest."""
    timestamp = now or datetime.now(UTC)
    return Layer1CheckpointManifest(
        compatibility_identity=identity,
        compatibility_sha256=compatibility_sha256(identity),
        execution_provenance=provenance,
        execution_provenance_sha256=execution_provenance_sha256(provenance),
        status=Layer1CheckpointStatus.IN_PROGRESS,
        created_at=timestamp,
        updated_at=timestamp,
    )


def canonical_npy_bytes(array: np.ndarray) -> bytes:
    """Encode a non-object array as deterministic C-order NPY bytes."""
    source = np.asarray(array)
    if source.dtype.hasobject:
        raise MalformedInputError("Checkpoint NPY shards cannot contain objects")
    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(source), allow_pickle=False)
    return buffer.getvalue()


def write_checkpoint_shard(
    checkpoint_root: Path,
    relative_name: str | Path,
    payload: bytes,
    *,
    area: Layer1ShardArea,
    task_id: str,
    compatibility_sha256_value: str,
    shard_format: Layer1ShardFormat,
    npy_dtype: str | None = None,
    npy_shape: tuple[int, ...] | None = None,
    parquet_columns: tuple[str, ...] | None = None,
) -> Layer1CheckpointShard:
    """Atomically write one validated complete shard and return exact metadata."""
    relative_path = _area_relative_path(area, relative_name)
    shard = Layer1CheckpointShard(
        path=relative_path.as_posix(),
        task_id=task_id,
        format=shard_format,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        compatibility_sha256=compatibility_sha256_value,
        npy_dtype=npy_dtype,
        npy_shape=npy_shape,
        parquet_columns=parquet_columns,
    )
    _validate_shard_payload(payload, shard)
    atomic_write_bytes(
        _resolve_checkpoint_path(checkpoint_root, relative_path), payload
    )
    return shard


def write_json_checkpoint_shard(
    checkpoint_root: Path,
    relative_name: str | Path,
    value: BaseModel | dict[str, object],
    *,
    area: Layer1ShardArea,
    task_id: str,
    compatibility_sha256_value: str,
) -> Layer1CheckpointShard:
    """Atomically write one canonical JSON checkpoint shard."""
    return write_checkpoint_shard(
        checkpoint_root,
        relative_name,
        canonical_json_bytes(value),
        area=area,
        task_id=task_id,
        compatibility_sha256_value=compatibility_sha256_value,
        shard_format=Layer1ShardFormat.JSON,
    )


def write_npy_checkpoint_shard(
    checkpoint_root: Path,
    relative_name: str | Path,
    array: np.ndarray,
    *,
    area: Layer1ShardArea,
    task_id: str,
    compatibility_sha256_value: str,
) -> Layer1CheckpointShard:
    """Atomically write one deterministic, pickle-free NPY checkpoint shard."""
    canonical_array = np.ascontiguousarray(np.asarray(array))
    payload = canonical_npy_bytes(canonical_array)
    return write_checkpoint_shard(
        checkpoint_root,
        relative_name,
        payload,
        area=area,
        task_id=task_id,
        compatibility_sha256_value=compatibility_sha256_value,
        shard_format=Layer1ShardFormat.NPY,
        npy_dtype=canonical_array.dtype.str,
        npy_shape=canonical_array.shape,
    )


def write_parquet_checkpoint_shard(
    checkpoint_root: Path,
    relative_name: str | Path,
    payload: bytes,
    *,
    area: Layer1ShardArea,
    task_id: str,
    compatibility_sha256_value: str,
    columns: tuple[str, ...],
) -> Layer1CheckpointShard:
    """Atomically write deterministic Parquet bytes with ordered schema metadata."""
    return write_checkpoint_shard(
        checkpoint_root,
        relative_name,
        payload,
        area=area,
        task_id=task_id,
        compatibility_sha256_value=compatibility_sha256_value,
        shard_format=Layer1ShardFormat.PARQUET,
        parquet_columns=columns,
    )


def read_scientific_checkpoint_shard(
    checkpoint_root: Path,
    shard: Layer1CheckpointShard,
    *,
    expected_compatibility_sha256: str,
) -> bytes:
    """Read and exactly verify a required scientific-state shard."""
    if Path(shard.path).parts[0] != Layer1ShardArea.STATE.value:
        raise MalformedInputError("Required checkpoint shard is outside state/")
    if shard.compatibility_sha256 != expected_compatibility_sha256:
        raise MalformedInputError("Required checkpoint shard is incompatible")
    return _read_verified_shard(checkpoint_root, shard)


def read_optional_cache_checkpoint_shard(
    checkpoint_root: Path,
    shard: Layer1CheckpointShard,
    *,
    expected_compatibility_sha256: str,
) -> bytes | None:
    """Read an optional cache shard, returning None when it must be recomputed."""
    try:
        if Path(shard.path).parts[0] != Layer1ShardArea.CACHE.value:
            raise MalformedInputError("Optional checkpoint shard is outside cache/")
        if shard.compatibility_sha256 != expected_compatibility_sha256:
            raise MalformedInputError("Optional checkpoint shard is incompatible")
        return _read_verified_shard(checkpoint_root, shard)
    except MalformedInputError:
        return None


def write_checkpoint_manifest(
    checkpoint_root: Path,
    manifest: Layer1CheckpointManifest,
) -> Layer1CheckpointManifest:
    """Verify referenced shards and atomically persist a canonical manifest.

    Missing or corrupt cache entries are omitted because they are accelerators.
    Missing or corrupt scientific-state entries fail closed.
    """
    for shard in manifest.scientific_state_shards:
        read_scientific_checkpoint_shard(
            checkpoint_root,
            shard,
            expected_compatibility_sha256=manifest.compatibility_sha256,
        )
    available_cache = tuple(
        shard
        for shard in manifest.optional_cache_shards
        if read_optional_cache_checkpoint_shard(
            checkpoint_root,
            shard,
            expected_compatibility_sha256=manifest.compatibility_sha256,
        )
        is not None
    )
    persisted_manifest = Layer1CheckpointManifest.model_validate(
        {
            **manifest.model_dump(),
            "optional_cache_shards": available_cache,
        }
    )
    atomic_write_bytes(
        checkpoint_root / CHECKPOINT_MANIFEST_NAME,
        canonical_json_bytes(persisted_manifest),
    )
    return persisted_manifest


def read_checkpoint_manifest(
    checkpoint_root: Path,
    *,
    expected_identity: Layer1CheckpointCompatibilityIdentity | None = None,
) -> Layer1CheckpointManifest:
    """Read a canonical manifest and fail closed on compatibility mismatch."""
    path = checkpoint_root / CHECKPOINT_MANIFEST_NAME
    try:
        payload = path.read_bytes()
        manifest = Layer1CheckpointManifest.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise MalformedInputError(
            f"Invalid Layer 1 checkpoint manifest: {error}"
        ) from error
    if payload != canonical_json_bytes(manifest):
        raise MalformedInputError("Layer 1 checkpoint manifest is not canonical JSON")
    if expected_identity is not None:
        expected_sha256 = compatibility_sha256(expected_identity)
        if (
            manifest.compatibility_sha256 != expected_sha256
            or manifest.compatibility_identity != expected_identity
        ):
            raise MalformedInputError(
                "Layer 1 checkpoint is incompatible with this environment"
            )
    return manifest


def load_compatible_checkpoint(
    checkpoint_root: Path,
    expected_identity: Layer1CheckpointCompatibilityIdentity,
) -> Layer1LoadedCheckpoint:
    """Load verified scientific state and only reusable optional cache entries."""
    manifest = read_checkpoint_manifest(
        checkpoint_root,
        expected_identity=expected_identity,
    )
    scientific_state = {
        shard.path: read_scientific_checkpoint_shard(
            checkpoint_root,
            shard,
            expected_compatibility_sha256=manifest.compatibility_sha256,
        )
        for shard in manifest.scientific_state_shards
    }
    optional_cache: dict[str, bytes] = {}
    for shard in manifest.optional_cache_shards:
        payload = read_optional_cache_checkpoint_shard(
            checkpoint_root,
            shard,
            expected_compatibility_sha256=manifest.compatibility_sha256,
        )
        if payload is not None:
            optional_cache[shard.path] = payload
    return Layer1LoadedCheckpoint(
        manifest=manifest,
        scientific_state=scientific_state,
        optional_cache=optional_cache,
    )


def admit_retained_task(
    ledger: tuple[Layer1TaskLedgerEntry, ...],
    task_id: str,
) -> tuple[Layer1TaskLedgerEntry, ...]:
    """Admit a computed task only after every earlier group task is resolved."""
    matching = [entry for entry in ledger if entry.task_id == task_id]
    if len(matching) != 1:
        raise MalformedInputError(f"Task ledger must contain exactly one {task_id!r}")
    target = matching[0]
    if target.status is not Layer1TaskStatus.COMPUTED_AVAILABLE:
        raise MalformedInputError(f"Task {task_id!r} is not available for admission")
    unresolved_predecessors = [
        entry.task_id
        for entry in ledger
        if entry.admission_group == target.admission_group
        and entry.admission_index < target.admission_index
        and entry.status
        not in {Layer1TaskStatus.ADMITTED_RETAINED, Layer1TaskStatus.REJECTED}
    ]
    if unresolved_predecessors:
        raise MalformedInputError(
            f"Task {task_id!r} has unresolved predecessors: {unresolved_predecessors}"
        )
    admitted = target.model_copy(update={"status": Layer1TaskStatus.ADMITTED_RETAINED})
    return tuple(admitted if entry.task_id == task_id else entry for entry in ledger)


def mark_checkpoint_interrupted(
    checkpoint_root: Path,
    *,
    exit_code: int,
    now: datetime | None = None,
) -> Layer1CheckpointManifest:
    """Atomically mark a checkpoint interrupted with conventional signal exit status."""
    if exit_code not in {130, 143}:
        raise MalformedInputError("Interrupted checkpoint exit code must be 130 or 143")
    manifest = read_checkpoint_manifest(checkpoint_root)
    interrupted = Layer1CheckpointManifest.model_validate(
        {
            **manifest.model_dump(),
            "status": Layer1CheckpointStatus.INTERRUPTED,
            "updated_at": now or datetime.now(UTC),
            "interruption_exit_code": exit_code,
        }
    )
    return write_checkpoint_manifest(checkpoint_root, interrupted)


def _require_finite_json(value: JsonValue) -> None:
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Checkpoint JSON values must be finite.")
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item)


def _area_relative_path(
    area: Layer1ShardArea,
    relative_name: str | Path,
) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise MalformedInputError("Checkpoint shard name must be a safe relative path")
    return Path(area.value) / relative


def _resolve_checkpoint_path(checkpoint_root: Path, relative_path: Path) -> Path:
    root = checkpoint_root.resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise MalformedInputError("Checkpoint shard path escapes the checkpoint root")
    return resolved


def _read_verified_shard(
    checkpoint_root: Path,
    shard: Layer1CheckpointShard,
) -> bytes:
    path = _resolve_checkpoint_path(checkpoint_root, Path(shard.path))
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise MalformedInputError(
            f"Checkpoint shard is unavailable: {shard.path}"
        ) from error
    if len(payload) != shard.size_bytes:
        raise MalformedInputError(f"Checkpoint shard size mismatch: {shard.path}")
    if sha256(payload).hexdigest() != shard.content_sha256:
        raise MalformedInputError(
            f"Checkpoint shard content hash mismatch: {shard.path}"
        )
    _validate_shard_payload(payload, shard)
    return payload


def _validate_shard_payload(
    payload: bytes,
    shard: Layer1CheckpointShard,
) -> None:
    if shard.format is Layer1ShardFormat.JSON:
        try:
            parsed = json.loads(
                payload,
                parse_constant=_reject_non_finite_json_constant,
            )
            canonical = (
                json.dumps(parsed, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
        except (UnicodeDecodeError, ValueError, TypeError) as error:
            raise MalformedInputError("Checkpoint JSON shard is invalid") from error
        if payload != canonical:
            raise MalformedInputError("Checkpoint JSON shard is not canonical")
        return
    if shard.format is Layer1ShardFormat.NPY:
        try:
            loaded = np.load(BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise MalformedInputError("Checkpoint NPY shard is invalid") from error
        if not isinstance(loaded, np.ndarray) or loaded.dtype.hasobject:
            raise MalformedInputError("Checkpoint NPY shard is not a pickle-free array")
        if loaded.dtype.str != shard.npy_dtype or loaded.shape != shard.npy_shape:
            raise MalformedInputError(
                "Checkpoint NPY shard metadata does not match content"
            )
        return
    if len(payload) < 8 or payload[:4] != b"PAR1" or payload[-4:] != b"PAR1":
        raise MalformedInputError("Checkpoint Parquet shard has invalid framing")


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not permitted: {value}")
