"""Tests for deterministic migration of legacy annotation response pairs."""

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from cursor_sdk import AgentOptions
from cursor_sdk.types import SDKModel

from es_index_explorer.question_analysis.annotations import (
    CursorSdkLike,
    ResolvedModel,
    _annotation_manifest,
    _archive_file,
    _archive_v1_preflight,
    _migrate_annotation_sources,
    _read_json_mapping,
    _require_exact_migration_files,
    _run_or_resume_batch,
    _verify_raw_response,
)
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

pytestmark = pytest.mark.unit


class NoCallSdk:
    """Fail if a valid legacy pair unexpectedly invokes an annotation model."""

    def list_models(self, api_key: str) -> Sequence[SDKModel]:
        raise AssertionError(f"Unexpected model listing with key length {len(api_key)}")

    def prompt(self, message: str, options: AgentOptions) -> object:
        raise AssertionError(
            f"Unexpected model call with message length {len(message)} and {options!r}"
        )


def _model() -> ResolvedModel:
    return ResolvedModel(
        "GPT-5.6 Sol",
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        (),
    )


def _batch() -> dict[str, object]:
    return {
        "schema_version": 1,
        "batch_id": "batch-001",
        "items": [
            {
                "item_id": "source-question-001",
                "item_type": "question",
                "source_text": "Who?",
                "text_sha256": "a" * 64,
            }
        ],
        "item_ids_sha256": "b" * 64,
    }


def _response(*, item_id: str = "item-001") -> str:
    return json.dumps(
        {
            "item_id": item_id,
            "exhaustivity_requirement": "mention_some",
            "negative_conclusiveness": False,
            "presupposition_load": False,
            "qdmr_applicability": "not_applicable",
            "qdmr_step_count": None,
            "qdmr_operator_set": None,
            "qdmr_normalized_question": None,
            "hop_structure": None,
            "referring_form_type": None,
            "referring_form_missingness": "no_focal_referent",
            "recall_orientation": "precision_oriented",
            "cognitive_process_level": "remember",
        },
        sort_keys=True,
    )


def _attempt(response: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_id": "gpt-5.6-sol",
        "batch_id": "batch-001",
        "batch_hash": "b" * 64,
        "agent_id": "agent-1",
        "run_id": "run-1",
        "status": "finished",
        "created_at": "2026-09-07T00:00:00Z",
        "duration_ms": 1,
        "usage": None,
        "response_sha256": sha256(response.encode("utf-8")).hexdigest(),
        "attempts": 1,
    }
    payload.update(overrides)
    return payload


def _write_legacy_pair(
    tmp_path: Path,
    response: str,
    attempt: Mapping[str, object],
) -> tuple[Path, Path]:
    raw_path = tmp_path / "batch-001.jsonl"
    attempt_path = tmp_path / "batch-001.json"
    raw_path.write_text(response, encoding="utf-8")
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
    return raw_path, attempt_path


def test_verified_legacy_response_returns_original_attempt(tmp_path: Path) -> None:
    response = _response()
    attempt = _attempt(response)
    raw_path, attempt_path = _write_legacy_pair(tmp_path, response, attempt)

    verified = _verify_raw_response(
        raw_path, attempt_path, _model(), "batch-001", _batch()
    )

    assert verified == attempt


@pytest.mark.parametrize(
    ("attempt_overrides", "error_match"),
    [
        ({"batch_hash": "wrong"}, "metadata mismatch"),
        ({"response_sha256": "0" * 64}, "metadata mismatch"),
    ],
)
def test_legacy_response_rejects_metadata_mismatch(
    attempt_overrides: dict[str, object],
    error_match: str,
    tmp_path: Path,
) -> None:
    response = _response()
    raw_path, attempt_path = _write_legacy_pair(
        tmp_path, response, _attempt(response, **attempt_overrides)
    )

    with pytest.raises(MalformedInputError, match=error_match):
        _verify_raw_response(raw_path, attempt_path, _model(), "batch-001", _batch())


@pytest.mark.parametrize(
    "response",
    [
        pytest.param("not-json", id="malformed-jsonl"),
        pytest.param(_response(item_id="item-999"), id="wrong-local-id"),
    ],
)
def test_legacy_response_rejects_invalid_batch_payload(
    response: str, tmp_path: Path
) -> None:
    raw_path, attempt_path = _write_legacy_pair(tmp_path, response, _attempt(response))

    with pytest.raises(MalformedInputError):
        _verify_raw_response(raw_path, attempt_path, _model(), "batch-001", _batch())


def test_legacy_resume_rejects_incomplete_pair_without_model_call(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)
    raw_path = workspace.store.path_for("annotations/raw/gpt-5.6-sol/batch-001.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(_response(), encoding="utf-8")

    with pytest.raises(MalformedInputError, match="Incomplete legacy"):
        _run_or_resume_batch(
            workspace,
            cast(CursorSdkLike, NoCallSdk()),
            "unused-key",
            _model(),
            "batch-001",
            _batch(),
        )

    assert not workspace.store.path_for(
        "annotations/envelopes/gpt-5.6-sol/batch-001.json"
    ).exists()


def test_resume_only_rejects_missing_pair_without_api_key_or_sdk(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)

    with pytest.raises(MalformedInputError, match="Resume-only annotation"):
        _run_or_resume_batch(
            workspace,
            None,
            "",
            _model(),
            "batch-001",
            _batch(),
            allow_model_calls=False,
        )


def test_migration_file_inventory_rejects_unexpected_source_file(
    tmp_path: Path,
) -> None:
    expected = Path("annotations/raw/gpt-5.6-sol/batch-001.jsonl")
    expected_path = tmp_path / expected
    expected_path.parent.mkdir(parents=True)
    expected_path.write_text(_response(), encoding="utf-8")

    _require_exact_migration_files(tmp_path, "annotations/raw", {expected})
    (expected_path.parent / "unexpected.jsonl").write_text(
        _response(), encoding="utf-8"
    )

    with pytest.raises(MalformedInputError, match="frozen model/batch inventory"):
        _require_exact_migration_files(tmp_path, "annotations/raw", {expected})


def test_migration_requires_resume_only_and_independent_complete_source(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)

    with pytest.raises(MalformedInputError, match="requires resume-only"):
        _migrate_annotation_sources(
            workspace, tmp_path / "source", {}, resume_only=False
        )
    with pytest.raises(MalformedInputError, match="must be independent"):
        _migrate_annotation_sources(workspace, workspace.root, {}, resume_only=True)
    with pytest.raises(MalformedInputError, match="source is incomplete"):
        _migrate_annotation_sources(
            workspace, tmp_path / "source", {}, resume_only=True
        )


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param("not-json", id="malformed"),
        pytest.param("[]", id="not-object"),
    ],
)
def test_read_json_mapping_rejects_invalid_objects(
    contents: str, tmp_path: Path
) -> None:
    path = tmp_path / "value.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(MalformedInputError, match="Invalid test value"):
        _read_json_mapping(path, "test value")


def test_v1_preflight_is_archived_with_legacy_response_files(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "analysis", specification)
    preflight = workspace.store.path_for("annotation_preflight_manifest.json")
    raw = workspace.store.path_for("annotations/raw/claude-opus-5/batch-001.jsonl")
    attempt = workspace.store.path_for(
        "annotations/attempts/claude-opus-5/batch-001.json"
    )
    raw.parent.mkdir(parents=True)
    attempt.parent.mkdir(parents=True)
    preflight.write_text('{"schema_version": 1}', encoding="utf-8")
    raw.write_text("legacy response", encoding="utf-8")
    attempt.write_text('{"legacy": true}', encoding="utf-8")

    _archive_v1_preflight(workspace)

    superseded = workspace.store.path_for("annotations/superseded/v1")
    migration = json.loads((superseded / "migration.json").read_text(encoding="utf-8"))
    assert migration["superseded_paths"] == [
        "annotations/raw/claude-opus-5/batch-001.jsonl",
        "annotations/attempts/claude-opus-5/batch-001.json",
    ]
    assert (superseded / "annotation_preflight_manifest.json").read_text(
        encoding="utf-8"
    ) == '{"schema_version": 1}'
    assert not preflight.exists()
    assert not raw.exists()
    assert not attempt.exists()


def test_archive_file_rejects_conflicting_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "archive" / "destination.json"
    source.write_text("source", encoding="utf-8")
    destination.parent.mkdir()
    destination.write_text("different", encoding="utf-8")

    with pytest.raises(MalformedInputError, match="Superseded annotation artifact"):
        _archive_file(source, destination)


def _write_minimal_migration_control_files(
    source: Path,
    preflight: Mapping[str, object],
    *,
    manifest_overrides: Mapping[str, object] | None = None,
) -> None:
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    (source / "state.json").write_text("{}", encoding="utf-8")
    (source / "annotation_preflight_manifest.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )
    manifest = {**preflight, "runs": []}
    if manifest_overrides is not None:
        manifest.update(manifest_overrides)
    (source / "annotation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_migration_rejects_preflight_or_manifest_contract_drift(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    batch = _batch()
    batches = {"batch-001": batch}
    preflight = _annotation_manifest((_model(),), batches)

    preflight_source = tmp_path / "preflight-source"
    _write_minimal_migration_control_files(preflight_source, preflight)
    preflight_workspace = AnalysisWorkspace.initialize(
        tmp_path / "preflight-destination", specification
    )
    with pytest.raises(MalformedInputError, match="preflight conflicts"):
        _migrate_annotation_sources(
            preflight_workspace,
            preflight_source,
            {"batch-001": {**batch, "item_ids_sha256": "c" * 64}},
            resume_only=True,
        )

    manifest_source = tmp_path / "manifest-source"
    _write_minimal_migration_control_files(
        manifest_source,
        preflight,
        manifest_overrides={"prompt_sha256": "0" * 64},
    )
    manifest_workspace = AnalysisWorkspace.initialize(
        tmp_path / "manifest-destination", specification
    )
    with pytest.raises(MalformedInputError, match="manifest conflicts"):
        _migrate_annotation_sources(
            manifest_workspace,
            manifest_source,
            batches,
            resume_only=True,
        )


def test_migration_requires_frozen_fifty_model_batch_pairs(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "specification.md"
    specification.write_text("# specification\n", encoding="utf-8")
    workspace = AnalysisWorkspace.initialize(tmp_path / "destination", specification)
    batches = {"batch-001": _batch()}
    preflight = _annotation_manifest((_model(),), batches)
    source = tmp_path / "source"
    _write_minimal_migration_control_files(source, preflight)

    with pytest.raises(MalformedInputError, match="frozen 50"):
        _migrate_annotation_sources(workspace, source, batches, resume_only=True)
