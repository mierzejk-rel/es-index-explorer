"""Focused tests for deterministic Layer 1 checkpoint persistence."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from es_index_explorer.question_analysis.contracts import FailureKind
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.layer1_checkpoint import (
    Layer1CheckpointCompatibilityIdentity,
    Layer1CheckpointManifest,
    Layer1CheckpointSession,
    Layer1CheckpointShard,
    Layer1CheckpointStatus,
    Layer1ExecutionProvenance,
    Layer1ShardArea,
    Layer1ShardFormat,
    Layer1TaskFailure,
    Layer1TaskLedgerEntry,
    Layer1TaskStatus,
    admit_retained_task,
    canonical_npy_bytes,
    compatibility_sha256,
    layer1_checkpoint_root,
    load_compatible_checkpoint,
    mark_checkpoint_interrupted,
    new_checkpoint_manifest,
    read_checkpoint_manifest,
    write_checkpoint_manifest,
    write_json_checkpoint_shard,
    write_npy_checkpoint_shard,
)
from es_index_explorer.question_analysis.storage import atomic_write_bytes

pytestmark = pytest.mark.unit

CREATED_AT = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)


def _identity() -> Layer1CheckpointCompatibilityIdentity:
    return Layer1CheckpointCompatibilityIdentity(
        specification_sha256="1" * 64,
        input_sha256s={"trace_pfu_table.parquet": "2" * 64},
        resource_sha256s={"oracle.json": "3" * 64},
        numerical_contract_version="layer1-v1",
        statistical_run_config={
            "adaptive_depths": [500, 1000, 2000, 4000],
            "inner_draw_count": 40,
        },
        seed_contract={"master_seed": 20, "derivation": "sha256-child-v1"},
        numerical_backend="adaptive-finite-difference",
        architecture="arm64",
        float_abi="IEEE-754-binary64-little-endian",
        python_version="3.12.11",
        numpy_version="2.4.6",
        scipy_version="1.17.1",
        blas_lapack_identity="Apple Accelerate 14.0",
    )


def _provenance(worker_count: int) -> Layer1ExecutionProvenance:
    return Layer1ExecutionProvenance(
        cpu_model="Apple M4 Max",
        os_details="Darwin arm64",
        executor_type="process",
        worker_count=worker_count,
        max_in_flight=worker_count * 2,
        progress_interval_seconds=5.0,
        auto_tuning_samples={"rubric": [1.2, 0.7]},
    )


def _manifest_with_shards(
    checkpoint_root: Path,
    *,
    include_cache: bool,
) -> tuple[Layer1CheckpointCompatibilityIdentity, Layer1CheckpointManifest]:
    identity = _identity()
    manifest = new_checkpoint_manifest(identity, _provenance(4), now=CREATED_AT)
    state_shard = write_json_checkpoint_shard(
        checkpoint_root,
        "moments.json",
        {"schema_version": 1, "phi": 2.5},
        area=Layer1ShardArea.STATE,
        task_id="moments",
        compatibility_sha256_value=manifest.compatibility_sha256,
    )
    cache_shards = ()
    if include_cache:
        cache_shards = (
            write_json_checkpoint_shard(
                checkpoint_root,
                "objective-cache.json",
                {"schema_version": 1, "objective": 12.0},
                area=Layer1ShardArea.CACHE,
                task_id="objective:primary",
                compatibility_sha256_value=manifest.compatibility_sha256,
            ),
        )
    manifest = Layer1CheckpointManifest.model_validate(
        {
            **manifest.model_dump(),
            "scientific_state_shards": (state_shard,),
            "optional_cache_shards": cache_shards,
        }
    )
    return identity, write_checkpoint_manifest(checkpoint_root, manifest)


def test_compatibility_identity_ignores_worker_and_executor_configuration() -> None:
    identity = _identity()
    sequential = new_checkpoint_manifest(
        identity,
        _provenance(1).model_copy(update={"executor_type": "sequential"}),
        now=CREATED_AT,
    )
    process = new_checkpoint_manifest(identity, _provenance(8), now=CREATED_AT)

    assert sequential.compatibility_sha256 == process.compatibility_sha256
    assert sequential.execution_provenance_sha256 != process.execution_provenance_sha256
    assert {
        "worker_count",
        "executor_type",
        "max_in_flight",
        "progress_interval_seconds",
    }.isdisjoint(Layer1CheckpointCompatibilityIdentity.model_fields)


def test_environment_mismatch_fails_closed(tmp_path: Path) -> None:
    checkpoint_root = layer1_checkpoint_root(tmp_path)
    identity = _identity()
    write_checkpoint_manifest(
        checkpoint_root,
        new_checkpoint_manifest(identity, _provenance(4), now=CREATED_AT),
    )
    incompatible = identity.model_copy(update={"numpy_version": "2.5.0"})

    with pytest.raises(MalformedInputError, match="incompatible with this environment"):
        read_checkpoint_manifest(checkpoint_root, expected_identity=incompatible)


def test_scientific_state_tamper_is_a_malformed_input(
    tmp_path: Path,
) -> None:
    checkpoint_root = layer1_checkpoint_root(tmp_path)
    identity, manifest = _manifest_with_shards(checkpoint_root, include_cache=False)
    state_path = checkpoint_root / manifest.scientific_state_shards[0].path
    atomic_write_bytes(state_path, b'{"phi": 9.0}\n')

    with pytest.raises(MalformedInputError, match="shard (size|content hash) mismatch"):
        load_compatible_checkpoint(checkpoint_root, identity)


@pytest.mark.parametrize("damage", ("missing", "tampered"))
def test_optional_cache_loss_or_tamper_requests_recomputation(
    tmp_path: Path,
    damage: str,
) -> None:
    checkpoint_root = layer1_checkpoint_root(tmp_path)
    identity, manifest = _manifest_with_shards(checkpoint_root, include_cache=True)
    cache_path = checkpoint_root / manifest.optional_cache_shards[0].path
    if damage == "missing":
        cache_path.unlink()
    else:
        atomic_write_bytes(cache_path, b'{"objective": 99.0}\n')

    loaded = load_compatible_checkpoint(checkpoint_root, identity)

    assert set(loaded.scientific_state) == {"state/moments.json"}
    assert loaded.optional_cache == {}


def test_task_ledger_admits_only_after_predecessors_are_resolved() -> None:
    rejected = Layer1TaskLedgerEntry(
        task_id="outer:1",
        admission_group="global-outer",
        admission_index=1,
        status=Layer1TaskStatus.REJECTED,
        failure=Layer1TaskFailure(
            failure_kind=FailureKind.NUMERICAL,
            failure_type="Layer1ModeError",
            message="Conditional mode did not converge.",
        ),
    )
    second = Layer1TaskLedgerEntry(
        task_id="outer:2",
        admission_group="global-outer",
        admission_index=2,
        status=Layer1TaskStatus.COMPUTED_AVAILABLE,
    )
    third = Layer1TaskLedgerEntry(
        task_id="outer:3",
        admission_group="global-outer",
        admission_index=3,
        status=Layer1TaskStatus.COMPUTED_AVAILABLE,
    )
    ledger = (rejected, second, third)

    with pytest.raises(MalformedInputError, match="unresolved predecessors"):
        admit_retained_task(ledger, "outer:3")

    ledger = admit_retained_task(ledger, "outer:2")
    ledger = admit_retained_task(ledger, "outer:3")

    assert [entry.status for entry in ledger] == [
        Layer1TaskStatus.REJECTED,
        Layer1TaskStatus.ADMITTED_RETAINED,
        Layer1TaskStatus.ADMITTED_RETAINED,
    ]
    assert ledger[0].failure is not None
    assert ledger[0].failure.failure_type == "Layer1ModeError"


def test_npy_shard_is_pickle_free_and_carries_exact_layout(
    tmp_path: Path,
) -> None:
    checkpoint_root = layer1_checkpoint_root(tmp_path)
    identity = _identity()
    compatibility = compatibility_sha256(identity)
    array = np.arange(12, dtype=np.float64).reshape(3, 4)

    shard = write_npy_checkpoint_shard(
        checkpoint_root,
        "outer-0001.npy",
        array,
        area=Layer1ShardArea.STATE,
        task_id="outer:1",
        compatibility_sha256_value=compatibility,
    )

    assert shard.npy_dtype == array.dtype.str
    assert shard.npy_shape == (3, 4)
    assert canonical_npy_bytes(array) == (checkpoint_root / shard.path).read_bytes()
    with pytest.raises(MalformedInputError, match="cannot contain objects"):
        canonical_npy_bytes(np.array([object()], dtype=object))


def test_mark_interrupted_persists_conventional_status(
    tmp_path: Path,
) -> None:
    checkpoint_root = layer1_checkpoint_root(tmp_path)
    assert checkpoint_root == tmp_path / "checkpoints" / "fit-layer1"
    write_checkpoint_manifest(
        checkpoint_root,
        new_checkpoint_manifest(_identity(), _provenance(4), now=CREATED_AT),
    )

    interrupted = mark_checkpoint_interrupted(
        checkpoint_root,
        exit_code=130,
        now=datetime(2026, 9, 10, 10, 5, tzinfo=UTC),
    )

    assert interrupted.status is Layer1CheckpointStatus.INTERRUPTED
    assert interrupted.interruption_exit_code == 130
    assert read_checkpoint_manifest(checkpoint_root) == interrupted


def test_checkpoint_session_resumes_across_worker_count_and_reads_state(
    tmp_path: Path,
) -> None:
    identity = _identity()
    session = Layer1CheckpointSession.open(
        tmp_path,
        identity,
        _provenance(2),
        resume=True,
        fresh=False,
    )
    session.commit_json_state(
        "primary/start_1_0.json",
        {"schema_version": 1, "objective": 12.5},
        task_id="primary:start:1.0",
    )

    resumed = Layer1CheckpointSession.open(
        tmp_path,
        identity,
        _provenance(8),
        resume=True,
        fresh=False,
    )

    assert resumed.read_json_state("primary/start_1_0.json") == {
        "schema_version": 1,
        "objective": 12.5,
    }
    assert resumed.manifest.execution_provenance.worker_count == 8
    assert (
        resumed.manifest.compatibility_sha256 == session.manifest.compatibility_sha256
    )


def test_checkpoint_session_fresh_discards_only_layer1_checkpoint(
    tmp_path: Path,
) -> None:
    session = Layer1CheckpointSession.open(
        tmp_path,
        _identity(),
        _provenance(2),
        resume=True,
        fresh=False,
    )
    session.commit_json_state(
        "moments.json",
        {"schema_version": 1},
        task_id="moments",
    )
    unrelated = tmp_path / "tables" / "input.parquet"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"preserved")

    restarted = Layer1CheckpointSession.open(
        tmp_path,
        _identity(),
        _provenance(4),
        resume=False,
        fresh=True,
    )

    assert restarted.read_json_state("moments.json") is None
    assert unrelated.read_bytes() == b"preserved"


def test_task_ledger_requires_failure_only_for_rejected_status() -> None:
    with pytest.raises(ValidationError, match="require a typed failure"):
        Layer1TaskLedgerEntry(
            task_id="rejected",
            admission_group="group",
            admission_index=1,
            status=Layer1TaskStatus.REJECTED,
        )
    with pytest.raises(ValidationError, match="Only rejected"):
        Layer1TaskLedgerEntry(
            task_id="available",
            admission_group="group",
            admission_index=1,
            status=Layer1TaskStatus.COMPUTED_AVAILABLE,
            failure=Layer1TaskFailure(
                failure_kind=FailureKind.NUMERICAL,
                failure_type="NumericalError",
                message="failure",
            ),
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"path": "/absolute.json"},
        {"path": "../escape.json"},
        {"path": "other/value.json"},
        {"path": "state/value.npy", "format": Layer1ShardFormat.NPY},
        {
            "path": "state/value.json",
            "format": Layer1ShardFormat.JSON,
            "npy_dtype": "<f8",
            "npy_shape": (1,),
        },
    ),
)
def test_checkpoint_shard_rejects_unsafe_or_incomplete_metadata(
    updates: dict[str, object],
) -> None:
    payload = {
        "path": "state/value.json",
        "task_id": "task",
        "format": Layer1ShardFormat.JSON,
        "content_sha256": "1" * 64,
        "size_bytes": 1,
        "compatibility_sha256": "2" * 64,
        **updates,
    }

    with pytest.raises(ValidationError):
        Layer1CheckpointShard.model_validate(payload)


def test_checkpoint_session_requires_explicit_resume_or_fresh(
    tmp_path: Path,
) -> None:
    identity = _identity()
    Layer1CheckpointSession.open(
        tmp_path,
        identity,
        _provenance(1),
        resume=True,
        fresh=False,
    )

    with pytest.raises(MalformedInputError, match="--resume or --fresh"):
        Layer1CheckpointSession.open(
            tmp_path,
            identity,
            _provenance(1),
            resume=False,
            fresh=False,
        )


def test_interrupt_rejects_non_signal_exit_code(tmp_path: Path) -> None:
    session = Layer1CheckpointSession.open(
        tmp_path,
        _identity(),
        _provenance(1),
        resume=True,
        fresh=False,
    )

    with pytest.raises(MalformedInputError, match="130 or 143"):
        session.mark_interrupted(1)
