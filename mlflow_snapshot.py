"""CLI for read-only MLflow snapshot export and offline analysis."""

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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse export and offline analysis arguments."""
    parser = argparse.ArgumentParser(
        description="Export sanitized MLflow snapshots or analyze existing snapshots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Read selected MLflow experiments and write a local snapshot.",
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
        help="Snapshot directory. Defaults to artifacts/mlflow/snapshot-<UTC timestamp>.",
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
        selector = _selector_from_args(args)

        if args.command == "export":
            output_dir = args.output_dir or _default_output_dir()
            snapshot_dir = export_snapshot(
                profile=args.profile,
                selector=selector,
                output_dir=output_dir,
                all_runs=args.all_runs,
                trace_fetch_concurrency=args.trace_fetch_concurrency,
            )
            print(f"Exported sanitized MLflow snapshot to {snapshot_dir}")
            return

        analysis_dir = analyze_snapshot(snapshot_dir=args.snapshot, selector=selector)
        print(f"Wrote offline analysis to {analysis_dir}")
    except KeyboardInterrupt:
        logger.info("Interrupted; no completed snapshot manifest was written.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
