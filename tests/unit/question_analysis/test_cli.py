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
    initial_arguments = build_parser().parse_args(
        [
            "gold-ingest-initial",
            "--adjudication-csv",
            str(tmp_path / "initial.csv"),
            "--provenance-json",
            str(tmp_path / "initial-provenance.json"),
        ]
    )
    gold_arguments = build_parser().parse_args(
        [
            "gold-ingest",
            "--recode-csv",
            str(tmp_path / "recode.csv"),
            "--provenance-json",
            str(tmp_path / "provenance.json"),
        ]
    )
    provisional_arguments = build_parser().parse_args(
        [
            "gold-ingest-provisional",
            "--recode-csv",
            str(tmp_path / "provisional.csv"),
            "--provenance-json",
            str(tmp_path / "provisional.json"),
            "--raw-response",
            str(tmp_path / "raw.bin"),
        ]
    )
    validation_arguments = build_parser().parse_args(
        [
            "validate-features",
            "--decisions-dir",
            str(tmp_path / "decisions"),
        ]
    )

    assert initial_arguments.adjudication_csv == tmp_path / "initial.csv"
    assert initial_arguments.provenance_json == tmp_path / "initial-provenance.json"
    assert gold_arguments.recode_csv == tmp_path / "recode.csv"
    assert gold_arguments.provenance_json == tmp_path / "provenance.json"
    assert provisional_arguments.recode_csv == tmp_path / "provisional.csv"
    assert provisional_arguments.provenance_json == tmp_path / "provisional.json"
    assert provisional_arguments.raw_response == tmp_path / "raw.bin"
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


def test_status_distinguishes_default_working_root_from_layer1_readiness(
    tmp_path: Path,
) -> None:
    root = tmp_path / "analysis"
    specification = _specification(tmp_path)
    AnalysisWorkspace.initialize(root, specification)
    output = StringIO()

    exit_code = main(
        ["--analysis-root", str(root), "status", "--json"],
        stdout=output,
    )

    assert exit_code == 0
    payload = json.loads(output.getvalue())
    assert payload["fit_layer1_runnable"] is False
    assert payload["fit_layer1_blockers"] == [
        "validate-features",
        "oracle",
        "outcome-modeling-unlock",
    ]


def test_oracle_command_runs_offline_and_records_artifact(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    error_output = StringIO()
    output = StringIO()

    exit_code = main(
        [
            "--analysis-root",
            str(root),
            "--specification",
            str(_specification(tmp_path)),
            "oracle",
        ],
        stdout=output,
        stderr=error_output,
    )

    assert exit_code == 0
    assert error_output.getvalue() == ""
    assert "oracle: completed" in output.getvalue()
    artifact = root / "statistics" / "r_oracle_verification.json"
    assert artifact.is_file()
    verification = json.loads(artifact.read_text(encoding="utf-8"))
    assert verification["passed"]
    assert verification["schema_version"] == 4
    assert verification["oracle_scope"] == (
        "linear_f6_external_raw_and_independent_r_psd"
    )
    assert verification["production_covariance"]["passed"]
    assert verification["production_covariance"]["materially_indefinite"]
    assert verification["production_covariance"]["relative_projection_shift"] > 0.10
    assert verification["external_p_value_convention"] == (
        "fwildclusterboot_valid_only_strict_no_plus_one"
    )
    assert verification["production_sampled_p_value_convention"] == (
        "replenish_to_B_then_plus_one"
    )


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
