"""Tests for the question-suitability command shell."""

import json
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest

from es_index_explorer.question_analysis.cli import build_parser, main
from es_index_explorer.question_analysis.contracts import (
    ArtifactMetadata,
    ExitCode,
    WorkflowCommand,
)
from es_index_explorer.question_analysis.errors import (
    AnalysisError,
    GateFailureError,
    MalformedInputError,
    NumericalError,
)
from es_index_explorer.question_analysis.workspace import AnalysisWorkspace

pytestmark = pytest.mark.unit


def _specification(tmp_path: Path) -> Path:
    path = tmp_path / "specification.md"
    path.write_text("# Locked specification\n", encoding="utf-8")
    return path


def test_help_lists_all_fourteen_commands() -> None:
    help_text = build_parser().format_help()

    for command in WorkflowCommand:
        assert command.value in help_text


def test_features_parser_accepts_explicit_resource_setup(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        [
            "features",
            "--stanza-model-dir",
            str(tmp_path / "stanza"),
            "--download-resources",
        ]
    )

    assert arguments.stanza_model_dir == tmp_path / "stanza"
    assert arguments.download_resources


def test_annotate_run_parser_accepts_fail_closed_migration_options(
    tmp_path: Path,
) -> None:
    source = tmp_path / "historical"

    arguments = build_parser().parse_args(
        [
            "annotate-run",
            "--migrate-from-root",
            str(source),
            "--resume-only",
        ]
    )

    assert arguments.migrate_from_root == source
    assert arguments.resume_only is True


def test_annotation_migration_requires_resume_only(tmp_path: Path) -> None:
    error_output = StringIO()

    exit_code = main(
        [
            "--analysis-root",
            str(tmp_path / "analysis"),
            "--specification",
            str(_specification(tmp_path)),
            "annotate-run",
            "--migrate-from-root",
            str(tmp_path / "historical"),
        ],
        stderr=error_output,
    )

    assert exit_code == ExitCode.MALFORMED_INPUT
    assert "requires --resume-only" in error_output.getvalue()


def test_gold_ingest_and_validation_parsers_require_manual_gate_inputs(
    tmp_path: Path,
) -> None:
    gold_arguments = build_parser().parse_args(
        [
            "gold-ingest",
            "--adjudication-csv",
            str(tmp_path / "initial.csv"),
            "--recode-csv",
            str(tmp_path / "recode.csv"),
            "--provenance-json",
            str(tmp_path / "provenance.json"),
        ]
    )
    validation_arguments = build_parser().parse_args(
        [
            "validate-features",
            "--decisions-dir",
            str(tmp_path / "decisions"),
        ]
    )

    assert gold_arguments.adjudication_csv == tmp_path / "initial.csv"
    assert gold_arguments.recode_csv == tmp_path / "recode.csv"
    assert gold_arguments.provenance_json == tmp_path / "provenance.json"
    assert validation_arguments.decisions_dir == tmp_path / "decisions"


def test_status_is_read_only_for_uninitialized_root(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    output = StringIO()

    exit_code = main(
        ["--analysis-root", str(root), "status", "--json"],
        stdout=output,
    )

    assert exit_code == 0
    assert json.loads(output.getvalue())["status"] == "not_initialized"
    assert not root.exists()


def test_unimplemented_command_fails_closed(tmp_path: Path) -> None:
    error_output = StringIO()

    exit_code = main(
        [
            "--analysis-root",
            str(tmp_path / "analysis"),
            "--specification",
            str(_specification(tmp_path)),
            "oracle",
        ],
        stderr=error_output,
    )

    assert exit_code == ExitCode.PREREQUISITE
    assert "no implementation registered" in error_output.getvalue()


def test_malformed_arguments_use_locked_exit_code() -> None:
    error_output = StringIO()

    exit_code = main(["not-a-command"], stderr=error_output)

    assert exit_code == ExitCode.MALFORMED_INPUT
    assert "invalid choice" in error_output.getvalue()


@pytest.mark.parametrize("command", ("gold-adjudicate", "fit"))
def test_superseded_commands_are_not_aliases(command: str) -> None:
    error_output = StringIO()

    exit_code = main([command], stderr=error_output)

    assert exit_code == ExitCode.MALFORMED_INPUT
    assert "invalid choice" in error_output.getvalue()


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    (
        (MalformedInputError, ExitCode.MALFORMED_INPUT),
        (GateFailureError, ExitCode.GATE_FAILURE),
        (NumericalError, ExitCode.NUMERICAL),
    ),
)
def test_handler_errors_use_locked_exit_categories(
    tmp_path: Path,
    error_type: type[AnalysisError],
    expected_code: ExitCode,
) -> None:
    def fail(workspace: AnalysisWorkspace) -> None:
        del workspace
        raise error_type("controlled failure")

    exit_code = main(
        [
            "--analysis-root",
            str(tmp_path / error_type.__name__),
            "--specification",
            str(_specification(tmp_path)),
            "join",
        ],
        handlers={WorkflowCommand.JOIN: fail},
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert exit_code == expected_code


def test_cli_handler_completes_step_and_status_reports_it(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    specification = _specification(tmp_path)

    def join_handler(workspace: AnalysisWorkspace) -> Iterator[ArtifactMetadata]:
        yield workspace.store.write_json(
            "tables/join.json",
            {"complete": True},
            created_by=WorkflowCommand.JOIN,
        )

    exit_code = main(
        [
            "--analysis-root",
            str(root),
            "--specification",
            str(specification),
            "join",
        ],
        handlers={WorkflowCommand.JOIN: join_handler},
        stdout=StringIO(),
    )
    status_output = StringIO()
    status_exit = main(
        ["--analysis-root", str(root), "status", "--json"],
        stdout=status_output,
    )

    assert exit_code == 0
    assert status_exit == 0
    assert json.loads(status_output.getvalue())["steps"]["join"] == {
        "attempts": 1,
        "last_failure": None,
        "status": "completed",
    }
