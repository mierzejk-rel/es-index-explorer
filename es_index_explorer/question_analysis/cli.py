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
    WorkflowCommand.GOLD_INGEST_INITIAL: "Checkpoint initial human adjudication.",
    WorkflowCommand.GOLD_RECODE_RELEASE: "Release the delayed blind re-code bundle.",
    WorkflowCommand.GOLD_INGEST_PROVISIONAL: (
        "Persist a non-human provisional re-code sidecar."
    ),
    WorkflowCommand.GOLD_INGEST: "Ingest delayed human re-code labels.",
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
    """Build the frozen workflow command parser."""
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
        elif command is WorkflowCommand.FEATURES:
            command_parser.add_argument(
                "--stanza-model-dir",
                type=Path,
                default=None,
                help="Directory containing the pinned Stanza English models.",
            )
            command_parser.add_argument(
                "--download-resources",
                action="store_true",
                help="Download and verify missing pinned Stanza models before extraction.",
            )
        elif command is WorkflowCommand.ANNOTATE_RUN:
            command_parser.add_argument(
                "--migrate-from-root",
                type=Path,
                default=None,
                help="Historical analysis root containing verified annotation source records.",
            )
            command_parser.add_argument(
                "--resume-only",
                action="store_true",
                help="Reconstruct verified responses without model listing or model calls.",
            )
        elif command is WorkflowCommand.GOLD_INGEST_INITIAL:
            command_parser.add_argument(
                "--adjudication-csv",
                type=Path,
                required=True,
                help="Completed initial human-adjudication CSV.",
            )
            command_parser.add_argument(
                "--provenance-json",
                type=Path,
                required=True,
                help="Strict initial-human provenance JSON.",
            )
        elif command in (
            WorkflowCommand.GOLD_INGEST,
            WorkflowCommand.GOLD_INGEST_PROVISIONAL,
        ):
            command_parser.add_argument(
                "--recode-csv",
                type=Path,
                required=True,
                help="Completed delayed blind-recode CSV.",
            )
            command_parser.add_argument(
                "--provenance-json",
                type=Path,
                required=True,
                help=(
                    "Strict provisional-LLM provenance JSON."
                    if command is WorkflowCommand.GOLD_INGEST_PROVISIONAL
                    else "Strict delayed-human provenance JSON."
                ),
            )
            if command is WorkflowCommand.GOLD_INGEST_PROVISIONAL:
                command_parser.add_argument(
                    "--raw-response",
                    type=Path,
                    required=True,
                    help="Immutable raw frontier-LLM response artifact.",
                )
        elif command is WorkflowCommand.VALIDATE_FEATURES:
            command_parser.add_argument(
                "--decisions-dir",
                type=Path,
                required=True,
                help="Directory containing write-once feature decision JSON files.",
            )
            command_parser.add_argument(
                "--emit-next",
                action="store_true",
                help="Expose only the next seeded-order feature dossier without completing validation.",
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
        if command is WorkflowCommand.VALIDATE_FEATURES and args.emit_next:
            from es_index_explorer.question_analysis.validation import (
                emit_next_feature_dossier,
            )

            assert_can_start(workspace.load_state(), command)
            dossier_path = emit_next_feature_dossier(
                workspace,
                args.decisions_dir,
            )
            print(f"validate-features: emitted {dossier_path}", file=output)
            return 0
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
    if command is WorkflowCommand.JOIN:
        from es_index_explorer.question_analysis.join import JoinConfig, run_join

        defaults = JoinConfig()
        join_config = JoinConfig(
            snapshot_dir=args.snapshot_dir or defaults.snapshot_dir,
            rubric_root=args.rubric_root or defaults.rubric_root,
            task_path=args.task or defaults.task_path,
        )
        return lambda workspace: run_join(workspace, join_config)
    if command is WorkflowCommand.FEATURES:
        from es_index_explorer.question_analysis.features import (
            FeatureConfig,
            run_features,
        )

        defaults = FeatureConfig()
        feature_config = FeatureConfig(
            stanza_model_dir=args.stanza_model_dir or defaults.stanza_model_dir,
            download_resources=args.download_resources,
        )
        return lambda workspace: run_features(workspace, feature_config)
    if command is WorkflowCommand.ANNOTATE_EMIT:
        from es_index_explorer.question_analysis.annotations import run_annotate_emit

        return run_annotate_emit
    if command is WorkflowCommand.ANNOTATE_RUN:
        from es_index_explorer.question_analysis.annotations import run_annotate_run

        if args.migrate_from_root is not None and not args.resume_only:
            raise MalformedInputError(
                "--migrate-from-root requires --resume-only to prevent model calls"
            )
        return lambda workspace: run_annotate_run(
            workspace,
            migration_source_root=args.migrate_from_root,
            resume_only=args.resume_only,
        )
    if command is WorkflowCommand.ANNOTATE_INGEST:
        from es_index_explorer.question_analysis.annotations import run_annotate_ingest

        return run_annotate_ingest
    if command is WorkflowCommand.GOLD_SAMPLE:
        from es_index_explorer.question_analysis.gold import run_gold_sample

        return run_gold_sample
    if command is WorkflowCommand.GOLD_INGEST_INITIAL:
        from es_index_explorer.question_analysis.gold import run_gold_ingest_initial

        return lambda workspace: run_gold_ingest_initial(
            workspace,
            args.adjudication_csv,
            args.provenance_json,
        )
    if command is WorkflowCommand.GOLD_RECODE_RELEASE:
        from es_index_explorer.question_analysis.gold import run_gold_recode_release

        return run_gold_recode_release
    if command is WorkflowCommand.GOLD_INGEST_PROVISIONAL:
        from es_index_explorer.question_analysis.gold import (
            run_gold_ingest_provisional,
        )

        return lambda workspace: run_gold_ingest_provisional(
            workspace,
            args.recode_csv,
            args.provenance_json,
            args.raw_response,
        )
    if command is WorkflowCommand.GOLD_INGEST:
        from es_index_explorer.question_analysis.gold import run_gold_ingest

        return lambda workspace: run_gold_ingest(
            workspace,
            args.recode_csv,
            args.provenance_json,
        )
    if command is WorkflowCommand.VALIDATE_FEATURES:
        from es_index_explorer.question_analysis.validation import run_validate_features

        return lambda workspace: run_validate_features(workspace, args.decisions_dir)
    if command is WorkflowCommand.ORACLE:
        from es_index_explorer.question_analysis.statistical_oracle import run_oracle

        return run_oracle
    return None


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
