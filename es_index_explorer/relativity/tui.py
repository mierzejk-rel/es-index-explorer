"""Shared terminal progress rendering for Relativity workflows."""

from dataclasses import dataclass

from rich.console import Group
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table


@dataclass
class ProgressSnapshot:
    processed: int
    total: int
    ok_count: int
    failed_count: int
    last_artifact_id: int | None
    last_error: str | None


class ProgressView:
    def __init__(self, total: int | None, *, description: str = "Importing") -> None:
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
        )
        self._task_id = self._progress.add_task(description, total=total)
        self._default_description = description
        total_value = total if total is not None else 0
        self._snapshot = ProgressSnapshot(
            processed=0,
            total=total_value,
            ok_count=0,
            failed_count=0,
            last_artifact_id=None,
            last_error=None,
        )

    def mark_complete(self, *, description: str = "Up to date") -> None:
        self._progress.update(
            self._task_id,
            description=description,
            completed=0,
            total=0,
        )

    def update(self, snapshot: ProgressSnapshot) -> None:
        self._snapshot = snapshot
        self._progress.update(
            self._task_id,
            description=self._default_description,
            completed=snapshot.processed,
            total=snapshot.total,
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
