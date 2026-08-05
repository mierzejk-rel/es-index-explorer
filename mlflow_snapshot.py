"""CLI for read-only MLflow snapshot export and offline analysis.

Export progress is checkpointed per experiment run under
``<output-dir>/checkpoint-{unix_epoch}/``. Resume uses the same ``--output-dir``
and matching export identity (profile, selector, all-runs, concurrency). Changed
remote runs invalidate only that run's shard; Ctrl+C retains the checkpoint.
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from es_index_explorer.mlflow_analysis.snapshot import (
    DEFAULT_EXPERIMENT_FOLDER,
    DEFAULT_TRACE_FETCH_CONCURRENCY,
    SnapshotSelector,
    analyze_snapshot,
    export_snapshot,
)
from es_index_explorer.mlflow_analysis.stage_a import analyze_stage_a_snapshot

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse export and offline analysis arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export sanitized MLflow snapshots or analyze existing snapshots. "
            "Exports are resumable via per-run checkpoints under the output directory."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help=(
            "Read selected MLflow experiments and write a local snapshot. "
            "Interrupted exports resume from committed run shards when the same "
            "output directory and export identity are reused."
        ),
    )
    _add_selector_arguments(export_parser)
    export_parser.add_argument(
        "--profile",
        required=True,
        help="Explicit Databricks CLI profile used for read-only MLflow access.",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Snapshot directory and checkpoint parent. Defaults to "
            "artifacts/mlflow/snapshot-<UTC timestamp>. Resume requires reusing "
            "the same directory."
        ),
    )
    export_parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Export all finished runs instead of only the latest finished run per experiment.",
    )
    export_parser.add_argument(
        "--trace-fetch-concurrency",
        type=int,
        choices=range(1, DEFAULT_TRACE_FETCH_CONCURRENCY + 1),
        default=DEFAULT_TRACE_FETCH_CONCURRENCY,
        help=(
            "Maximum simultaneous full-trace downloads. Defaults to 10, "
            "matching MLflow's default connection-pool size."
        ),
    )
    export_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from the newest incomplete checkpoint whose CLI identity matches "
            "(default: true). Use --no-resume to start a new session."
        ),
    )
    export_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing matching checkpoints and start a fresh export session.",
    )
    export_parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help=(
            "Resume a specific checkpoint-{epoch} directory under --output-dir. "
            "Mutually exclusive with --fresh."
        ),
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze an existing local snapshot without contacting MLflow.",
    )
    _add_selector_arguments(analyze_parser)
    analyze_parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="Existing snapshot directory created by the export command.",
    )
    stage_a_parser = subparsers.add_parser(
        "analyze-stage-a",
        help=(
            "Produce auditable Stage A comparisons, Pareto sets, retrieval overlap, "
            "and Stage B recommendations without contacting MLflow."
        ),
    )
    stage_a_parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="Completed 54-run Stage A snapshot directory.",
    )
    stage_a_parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports") / "11-stage-a-results-stage-b-redesign.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def _add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive experiment name selection arguments."""
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--experiment-prefix",
        help="Literal MLflow experiment-name prefix.",
    )
    selector.add_argument(
        "--experiment-folder",
        help=(
            "MLflow experiment folder. Defaults to recursive selection under "
            f"{DEFAULT_EXPERIMENT_FOLDER!r}."
        ),
    )
    parser.add_argument(
        "--direct-children",
        action="store_true",
        help="Limit --experiment-folder selection to immediate child experiments.",
    )


def _selector_from_args(args: argparse.Namespace) -> SnapshotSelector:
    """Build and validate a reusable experiment selector."""
    return SnapshotSelector(
        experiment_prefix=args.experiment_prefix,
        experiment_folder=args.experiment_folder,
        direct_children=args.direct_children,
    )


def _default_output_dir() -> Path:
    """Return a timestamped local snapshot directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "mlflow" / f"snapshot-{timestamp}"


def main() -> None:
    """Run export or offline analysis."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("es_index_explorer").setLevel(logging.INFO)
    try:
        args = parse_args()

        if args.command == "export":
            selector = _selector_from_args(args)
            if args.fresh and args.checkpoint_epoch is not None:
                raise SystemExit("error: --fresh and --checkpoint-epoch are mutually exclusive")
            output_dir = args.output_dir or _default_output_dir()
            snapshot_dir = export_snapshot(
                profile=args.profile,
                selector=selector,
                output_dir=output_dir,
                all_runs=args.all_runs,
                trace_fetch_concurrency=args.trace_fetch_concurrency,
                resume=args.resume,
                fresh=args.fresh,
                checkpoint_epoch=args.checkpoint_epoch,
            )
            print(f"Exported sanitized MLflow snapshot to {snapshot_dir}")
            return

        if args.command == "analyze-stage-a":
            analysis_dir = analyze_stage_a_snapshot(
                snapshot_dir=args.snapshot,
                report_path=args.report,
            )
            print(f"Wrote auditable Stage A analysis to {analysis_dir}")
            return

        selector = _selector_from_args(args)
        analysis_dir = analyze_snapshot(snapshot_dir=args.snapshot, selector=selector)
        print(f"Wrote offline analysis to {analysis_dir}")
    except KeyboardInterrupt:
        logger.info(
            "Interrupted; checkpoint retained under the output directory when present. "
            "No completed snapshot manifest was written for this attempt."
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
