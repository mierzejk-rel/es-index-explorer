"""Tests for frozen dependency and external-runtime boundaries."""

import json
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

from es_index_explorer.question_analysis.contracts import (
    ANNOTATION_BATCH_SIZE,
    ANNOTATOR_MODELS,
    GOLD_SAMPLE_PER_STRATUM,
    OUTCOME_FIELD_DENYLIST,
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
