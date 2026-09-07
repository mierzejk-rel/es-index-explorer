"""Tests for frozen dependency and external-runtime boundaries."""

import hashlib
import json
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

from es_index_explorer.question_analysis import nlp_resources as nlp_resources_module
from es_index_explorer.question_analysis.contracts import (
    ANNOTATION_BATCH_SIZE,
    ANNOTATOR_MODELS,
    GOLD_SAMPLE_PER_STRATUM,
    OUTCOME_FIELD_DENYLIST,
)
from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.nlp_resources import (
    _spacy_model_files,
    _spacy_model_tree_sha256,
    download_stanza_resources,
    load_spacy_manifest,
    load_stanza_manifest,
    verify_spacy_resources,
    verify_stanza_resources,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_dependency_groups_are_separate() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        groups = tomllib.load(handle)["dependency-groups"]

    assert {"analysis", "analysis-test", "annotation"} <= groups.keys()
    assert any(
        dependency.startswith("cursor-sdk") for dependency in groups["annotation"]
    )
    python_dependencies = (
        dependency
        for dependencies in groups.values()
        for dependency in dependencies
        if isinstance(dependency, str)
    )
    assert not any(
        "fwildclusterboot" in dependency.lower() for dependency in python_dependencies
    )


def test_nlp_distribution_dependencies_are_exactly_pinned() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    assert "spacy==3.8.14" in project["project"]["dependencies"]
    assert "stanza==1.14.0" in project["dependency-groups"]["analysis"]

    with (PROJECT_ROOT / "uv.lock").open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    versions = {
        package["name"]: package["version"]
        for package in packages
        if package["name"] in {"spacy", "stanza"}
    }
    assert versions == {"spacy": "3.8.14", "stanza": "1.14.0"}


def test_stanza_resource_catalogue_is_versioned_and_hashed() -> None:
    path = (
        PROJECT_ROOT
        / "es_index_explorer"
        / "question_analysis"
        / "resources"
        / "stanza-en-resource-manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["stanza_version"] == version("stanza")
    assert len(manifest["resources_sha256"]) == 64
    assert manifest["processors"] == ["tokenize", "mwt", "pos", "lemma", "depparse"]
    assert set(manifest["selected_models"]) == {
        "backward_charlm",
        "depparse",
        "forward_charlm",
        "lemma",
        "mwt",
        "pos",
        "pretrain",
        "tokenize",
    }
    assert all(
        len(model["md5"]) == 32 for model in manifest["selected_models"].values()
    )


def test_stanza_runtime_catalogue_checksum_is_enforced(tmp_path: Path) -> None:
    (tmp_path / "resources.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        MalformedInputError, match="resource catalogue checksum mismatch"
    ):
        verify_stanza_resources(tmp_path, load_stanza_manifest())


def test_stanza_runtime_catalogue_is_required(tmp_path: Path) -> None:
    with pytest.raises(MalformedInputError, match="Missing Stanza resource"):
        verify_stanza_resources(tmp_path, load_stanza_manifest())


def test_stanza_runtime_requires_every_selected_model(tmp_path: Path) -> None:
    resources = b"{}"
    (tmp_path / "resources.json").write_bytes(resources)
    manifest = load_stanza_manifest().model_copy(
        update={"resources_sha256": hashlib.sha256(resources).hexdigest()}
    )

    with pytest.raises(MalformedInputError, match="Missing Stanza model"):
        verify_stanza_resources(tmp_path, manifest)


def test_stanza_runtime_rejects_changed_selected_model(tmp_path: Path) -> None:
    resources = b"{}"
    (tmp_path / "resources.json").write_bytes(resources)
    model_bytes = b"model"
    selected_models = {
        processor: pin.model_copy(
            update={"md5": hashlib.md5(model_bytes).hexdigest()}
        )
        for processor, pin in load_stanza_manifest().selected_models.items()
    }
    manifest = load_stanza_manifest().model_copy(
        update={
            "resources_sha256": hashlib.sha256(resources).hexdigest(),
            "selected_models": selected_models,
        }
    )
    for processor, pin in selected_models.items():
        path = tmp_path / manifest.language / processor / f"{pin.package}.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(model_bytes)
    changed = next(iter(selected_models))
    changed_pin = selected_models[changed]
    (tmp_path / manifest.language / changed / f"{changed_pin.package}.pt").write_bytes(
        b"changed"
    )

    with pytest.raises(MalformedInputError, match="checksum mismatch"):
        verify_stanza_resources(tmp_path, manifest)


def test_stanza_distribution_version_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = load_stanza_manifest().model_copy(
        update={"stanza_version": "0.0.0"}
    )

    with pytest.raises(MalformedInputError, match="stanza version"):
        verify_stanza_resources(tmp_path, manifest)


def test_stanza_download_uses_selected_processor_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    verifications: list[tuple[Path, object]] = []

    def fake_download(**kwargs: object) -> None:
        calls.append(kwargs)

    def fake_verify(model_dir: Path, manifest: object) -> tuple[Path, ...]:
        verifications.append((model_dir, manifest))
        return ()

    monkeypatch.setattr(nlp_resources_module.stanza, "download", fake_download)
    monkeypatch.setattr(
        nlp_resources_module, "verify_stanza_resources", fake_verify
    )

    download_stanza_resources(tmp_path)

    manifest = load_stanza_manifest()
    assert calls == [
        {
            "lang": "en",
            "model_dir": tmp_path.as_posix(),
            "processors": {
                processor: manifest.selected_models[processor].package
                for processor in manifest.processors
            },
            "package": None,
            "resources_url": manifest.resources_url.rsplit("/", maxsplit=1)[0],
            "resources_version": "1.14.0",
            "verbose": False,
        }
    ]
    assert verifications == [(tmp_path, manifest)]
    assert set(manifest.selected_models) == {
        *manifest.processors,
        "pretrain",
        "forward_charlm",
        "backward_charlm",
    }


def test_spacy_ner_model_is_versioned_and_hashed() -> None:
    path = (
        PROJECT_ROOT
        / "es_index_explorer"
        / "question_analysis"
        / "resources"
        / "spacy-en-resource-manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["model"] == "en_core_web_sm"
    assert manifest["model_version"] == version("en-core-web-sm")
    assert manifest["schema_version"] == 3
    assert manifest["spacy_version"] == version("spacy")
    assert manifest["pipeline_component"] == "ner"
    assert len(manifest["wheel_sha256"]) == 64
    assert manifest["model_file_count"] == 28
    assert len(manifest["model_tree_sha256"]) == 64


def test_spacy_wheel_hash_matches_uv_lock() -> None:
    with (PROJECT_ROOT / "uv.lock").open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    locked = next(
        package for package in packages if package["name"] == "en-core-web-sm"
    )
    manifest = load_spacy_manifest()

    assert locked["version"] == manifest.model_version
    assert locked["source"]["url"] == manifest.wheel_url
    assert locked["wheels"] == [
        {"url": manifest.wheel_url, "hash": f"sha256:{manifest.wheel_sha256}"}
    ]


def test_installed_spacy_model_tree_matches_manifest() -> None:
    package_root, files = verify_spacy_resources()
    manifest = load_spacy_manifest()

    assert len(files) == manifest.model_file_count
    assert _spacy_model_tree_sha256(files, package_root) == (
        manifest.model_tree_sha256
    )


def test_spacy_model_tree_rejects_missing_extra_and_changed_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "model" / "a.bin"
    second = tmp_path / "model" / "nested" / "b.bin"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    cache = tmp_path / "model" / "__pycache__" / "ignored.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"ignored")
    files = _spacy_model_files(tmp_path / "model")
    assert [path.relative_to(tmp_path / "model").as_posix() for path in files] == [
        "a.bin",
        "nested/b.bin",
    ]
    manifest = load_spacy_manifest().model_copy(
        update={
            "model_file_count": len(files),
            "model_tree_sha256": _spacy_model_tree_sha256(
                files, tmp_path / "model"
            ),
        }
    )

    assert verify_spacy_resources(tmp_path / "model", manifest)[1] == files

    extra = tmp_path / "model" / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(MalformedInputError, match="file count"):
        verify_spacy_resources(tmp_path / "model", manifest)
    extra.unlink()

    second.unlink()
    with pytest.raises(MalformedInputError, match="file count"):
        verify_spacy_resources(tmp_path / "model", manifest)
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(b"changed")

    with pytest.raises(MalformedInputError, match="tree checksum mismatch"):
        verify_spacy_resources(tmp_path / "model", manifest)


def test_spacy_model_version_mismatch_is_rejected() -> None:
    manifest = load_spacy_manifest().model_copy(
        update={"model_version": "0.0.0"}
    )

    with pytest.raises(MalformedInputError, match="model version"):
        verify_spacy_resources(manifest=manifest)


def test_spacy_distribution_version_mismatch_is_rejected() -> None:
    manifest = load_spacy_manifest().model_copy(
        update={"spacy_version": "0.0.0"}
    )

    with pytest.raises(MalformedInputError, match="spacy version"):
        verify_spacy_resources(manifest=manifest)


def test_r_oracle_base_image_is_digest_pinned() -> None:
    dockerfile = (
        PROJECT_ROOT / "tests" / "oracles" / "fwildclusterboot" / "Dockerfile"
    ).read_text(encoding="utf-8")

    first_line = dockerfile.splitlines()[0]
    assert first_line.startswith("FROM rocker/r-ver:4.4.3@sha256:")
    assert len(first_line.rsplit(":", maxsplit=1)[-1]) == 64


def test_annotation_constants_match_locked_specification() -> None:
    assert ANNOTATION_BATCH_SIZE == 24
    assert GOLD_SAMPLE_PER_STRATUM == 12
    assert ANNOTATOR_MODELS == (
        "Claude Opus 5 (high thinking)",
        "GPT-5.6 Sol",
    )
    assert OUTCOME_FIELD_DENYLIST == {
        "p",
        "f",
        "u",
        "state",
        "error_override",
        "rubric_v2",
        "pass_rate",
        "grade",
        "ordinal_grade",
        "tier",
    }


def test_question_analysis_has_no_r1_evals_import() -> None:
    package = PROJECT_ROOT / "es_index_explorer" / "question_analysis"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.rglob("*.py")
    )

    assert "import r1_evals" not in source
    assert "from r1_evals" not in source


def test_question_analysis_does_not_use_first_input_only_catalogue_helpers() -> None:
    package = PROJECT_ROOT / "es_index_explorer" / "question_analysis"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.rglob("*.py")
    )

    assert "mlflow_analysis.rubric_analysis" not in source
    assert "import rubric_analysis" not in source
    for helper_name in ("_rubric_question", "_build_catalogue", "analyze_rubrics"):
        assert helper_name not in source
