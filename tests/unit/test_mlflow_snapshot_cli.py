"""Unit tests for the MLflow snapshot command surface."""

import sys

import pytest

from mlflow_snapshot import parse_args

pytestmark = pytest.mark.unit


def test_cli_help_includes_stage_a_arm_analysis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The arm-level Stage A analysis command remains public."""
    monkeypatch.setattr(sys, "argv", ["mlflow_snapshot.py", "--help"])

    with pytest.raises(SystemExit) as error:
        parse_args()

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "analyze-stage-a" in help_text
    assert "analyze-stage-b" in help_text
    assert "analyze-rubrics" in help_text


def test_cli_parses_stage_neutral_rubric_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rubric analysis remains available with its required local inputs."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlflow_snapshot.py",
            "analyze-rubrics",
            "--snapshot",
            "snapshot",
            "--rubric-root",
            "rubrics",
            "--task",
            "task.toml",
        ],
    )

    args = parse_args()

    assert args.command == "analyze-rubrics"
    assert str(args.snapshot) == "snapshot"


def test_cli_parses_stage_a_arm_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage A arm-level analysis accepts snapshot and report arguments."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlflow_snapshot.py",
            "analyze-stage-a",
            "--snapshot",
            "snapshot",
            "--report",
            "report.md",
        ],
    )

    args = parse_args()

    assert args.command == "analyze-stage-a"
    assert str(args.snapshot) == "snapshot"
    assert str(args.report) == "report.md"


def test_cli_parses_stage_b_arm_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage B arm-level analysis accepts snapshot and report arguments."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlflow_snapshot.py",
            "analyze-stage-b",
            "--snapshot",
            "snapshot",
            "--report",
            "report.md",
        ],
    )

    args = parse_args()

    assert args.command == "analyze-stage-b"
    assert str(args.snapshot) == "snapshot"
    assert str(args.report) == "report.md"
