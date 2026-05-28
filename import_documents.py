"""Batch import documents with resume/retry support."""

import argparse
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

from es_index_explorer.config import load_config
from es_index_explorer.relativity.batch_reader import BatchImporter, ProgressSnapshot
from es_index_explorer.relativity.progress import ProgressLog


class ProgressView:
    def __init__(self, total: int | None) -> None:
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
        )
        self._task_id = self._progress.add_task("Importing", total=total)
        total_value = total or 0
        self._snapshot = ProgressSnapshot(
            processed=0,
            total=total_value,
            ok_count=0,
            failed_count=0,
            last_artifact_id=None,
            last_error=None,
        )

    def mark_complete(self) -> None:
        self._progress.update(
            self._task_id,
            description="Up to date",
            completed=0,
            total=0,
        )

    def update(self, snapshot: ProgressSnapshot) -> None:
        self._snapshot = snapshot
        self._progress.update(
            self._task_id,
            completed=snapshot.processed,
            total=max(snapshot.total, 1),
        )

    def render(self) -> Group:
        table = Table.grid(padding=(0, 2))
        table.add_row(
            f"OK: {self._snapshot.ok_count}",
            f"Failed: {self._snapshot.failed_count}",
            f"Last ID: {self._snapshot.last_artifact_id or '-'}",
        )
        table.add_row(
            "Last error:",
            self._snapshot.last_error or "-",
        )
        return Group(self._progress, table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch import RelativityOne documents.")
    parser.add_argument("--config", default=None, help="Path to config.toml.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for progress log files.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry failed documents from the progress log.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard existing progress log and start from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir).expanduser().resolve()
    progress_log = ProgressLog(
        host=config.relativity.host,
        workspace_id=config.relativity.workspace_id,
        saved_search_id=config.relativity.saved_search_id,
        directory=output_dir,
    )
    if args.fresh:
        progress_log.reset()

    state = progress_log.load()

    if args.retry:
        failed_ids = list(state.failed.keys())
        if not failed_ids:
            print("No failures to retry.")
            return
        view = ProgressView(total=len(failed_ids))
        with Live(view.render(), refresh_per_second=10) as live:
            def _on_progress(snapshot: ProgressSnapshot) -> None:
                view.update(snapshot)
                live.update(view.render())

            importer = BatchImporter(config, progress_log, on_progress=_on_progress)
            importer.retry_failed(failed_ids)
            live.update(view.render())
    else:
        first_snapshot: ProgressSnapshot | None = None

        def _on_progress(snapshot: ProgressSnapshot) -> None:
            nonlocal first_snapshot
            if first_snapshot is None:
                first_snapshot = snapshot
            view.update(snapshot)
            live.update(view.render())

        view = ProgressView(total=None)
        with Live(view.render(), refresh_per_second=10) as live:
            importer = BatchImporter(config, progress_log, on_progress=_on_progress)
            importer.run(resume_after=state.last_processed_id, state=state)
            if first_snapshot is None:
                view.mark_complete()
            live.update(view.render())
        if first_snapshot is None:
            print("Already up to date — no pending documents.")
            progress_log.close()
            return

    progress_log.close()


if __name__ == "__main__":
    main()
