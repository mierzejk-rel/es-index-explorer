"""Command-line shell for the question-suitability workflow."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from es_index_explorer.question_analysis.contracts import WorkflowCommand
from es_index_explorer.question_analysis.errors import (
    AnalysisError,
    MalformedInputError,
    PrerequisiteError,
)
from es_index_explorer.question_analysis.workflow import assert_can_start
from es_index_explorer.question_analysis.workspace import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_SPECIFICATION,
    AnalysisWorkspace,
    StepAction,
)

COMMAND_HELP: dict[WorkflowCommand, str] = {
    WorkflowCommand.JOIN: "Build and verify the criterion/trace data contract.",
    WorkflowCommand.FEATURES: "Extract deterministic P1/P2/P3 features.",
    WorkflowCommand.ANNOTATE_EMIT: "Emit deterministic outcome-blind annotation batches.",
    WorkflowCommand.ANNOTATE_RUN: "Run both frozen Cursor SDK annotators.",
    WorkflowCommand.ANNOTATE_INGEST: "Validate and normalize raw P4 annotations.",
    WorkflowCommand.GOLD_SAMPLE: "Emit the deterministic stratified gold sample.",
    WorkflowCommand.GOLD_INGEST: "Ingest adjudication and delayed blind re-code labels.",
    WorkflowCommand.VALIDATE_FEATURES: "Run feature-validation and DSL gates.",
    WorkflowCommand.ORACLE: "Run the independent R fwildclusterboot reference check.",
    WorkflowCommand.FIT_LAYER1: "Fit Layer 1 and assign rubric/variant tiers.",
    WorkflowCommand.FIT_FAMILIES: "Fit the ten stacked confirmatory families.",
    WorkflowCommand.ROBUSTNESS: "Run secondary and sensitivity analyses.",
    WorkflowCommand.REPORT: "Render persisted results and figures.",
    WorkflowCommand.STATUS: "Show workflow state without modifying it.",
}


class AnalysisArgumentParser(argparse.ArgumentParser):
    """Argument parser using the locked malformed-input exit category."""

    def error(self, message: str) -> NoReturn:
        raise MalformedInputError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen fourteen-command parser."""
    parser = AnalysisArgumentParser(
        description="Run the Simple Mode question-suitability analysis."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help=f"Analysis root (default: {DEFAULT_ANALYSIS_ROOT}).",
    )
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION,
        help=f"Locked research specification (default: {DEFAULT_SPECIFICATION}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in WorkflowCommand:
        command_parser = subparsers.add_parser(
            command.value, help=COMMAND_HELP[command]
        )
        if command is WorkflowCommand.JOIN:
            command_parser.add_argument(
                "--snapshot-dir",
                type=Path,
                default=None,
                help="Schema-v3 MLflow snapshot directory.",
            )
            command_parser.add_argument(
                "--rubric-root",
                type=Path,
                default=None,
                help="Local r1-evals rubric_data directory.",
            )
            command_parser.add_argument(
                "--task",
                type=Path,
                default=None,
                help="Air Assist task TOML containing the use-case taxonomy.",
            )
        elif command is WorkflowCommand.STATUS:
            command_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[WorkflowCommand, StepAction] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one CLI command and return its process exit code."""
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    try:
        args = build_parser().parse_args(argv)
        command = WorkflowCommand(args.command)
        if command is WorkflowCommand.STATUS:
            return _show_status(args.analysis_root, as_json=args.as_json, output=output)
        workspace = AnalysisWorkspace.initialize(args.analysis_root, args.specification)
        handler = (
            handlers.get(command)
            if handlers is not None
            else _production_handler(command, args)
        )
        if handler is None:
            assert_can_start(workspace.load_state(), command)
            raise PrerequisiteError(
                f"{command.value} has no implementation registered in the foundation shell"
            )
        executed = workspace.run_step(command, handler)
        outcome = "completed" if executed else "already complete"
        print(f"{command.value}: {outcome}", file=output)
        return 0
    except AnalysisError as error:
        print(f"error: {error.message}", file=error_output)
        return int(error.exit_code)


def _production_handler(
    command: WorkflowCommand, args: argparse.Namespace
) -> StepAction | None:
    if command is not WorkflowCommand.JOIN:
        return None
    from es_index_explorer.question_analysis.join import JoinConfig, run_join

    defaults = JoinConfig()
    config = JoinConfig(
        snapshot_dir=args.snapshot_dir or defaults.snapshot_dir,
        rubric_root=args.rubric_root or defaults.rubric_root,
        task_path=args.task or defaults.task_path,
    )
    return lambda workspace: run_join(workspace, config)


def _show_status(root: Path, *, as_json: bool, output: TextIO) -> int:
    if not (root / "manifest.json").is_file() or not (root / "state.json").is_file():
        payload = {
            "analysis_root": root.resolve().as_posix(),
            "status": "not_initialized",
        }
        if as_json:
            print(json.dumps(payload, sort_keys=True, indent=2), file=output)
        else:
            print(f"Analysis root: {payload['analysis_root']}", file=output)
            print("Status: not_initialized", file=output)
        return 0

    workspace = AnalysisWorkspace.open_existing(root)
    state = workspace.load_state()
    steps: dict[str, dict[str, str | int | None]] = {}
    for command in WorkflowCommand:
        if command is WorkflowCommand.STATUS:
            continue
        record = state.steps[command]
        steps[command.value] = {
            "status": record.status.value,
            "attempts": record.attempts,
            "last_failure": record.failure.message
            if record.failure is not None
            else None,
        }
    payload = {
        "analysis_root": workspace.root.as_posix(),
        "status": "initialized",
        "outcome_modeling_unlocked": state.outcome_modeling_unlocked,
        "outcome_modeling_unlocked_at": (
            state.outcome_modeling_unlocked_at.isoformat()
            if state.outcome_modeling_unlocked_at is not None
            else None
        ),
        "steps": steps,
    }
    if as_json:
        print(json.dumps(payload, sort_keys=True, indent=2), file=output)
    else:
        print(f"Analysis root: {payload['analysis_root']}", file=output)
        print("Status: initialized", file=output)
        for name, record in steps.items():
            print(
                f"  {name}: {record['status']} (attempts={record['attempts']})",
                file=output,
            )
        unlocked = "yes" if state.outcome_modeling_unlocked else "no"
        print(f"Outcome modeling unlocked: {unlocked}", file=output)
    return 0


def entrypoint() -> None:
    """Run the installed console script."""
    raise SystemExit(main())
