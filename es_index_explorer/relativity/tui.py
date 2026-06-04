"""Shared terminal progress rendering for Relativity workflows."""

from dataclasses import dataclass

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
        self._description = description
        self._previous_failed_count = 0

    def note_warning(self, message: str) -> None:
        """Print warning immediately so it is visible in non-interactive consoles."""
        print(f"WARNING: {message}", flush=True)

    def update(self, snapshot: ProgressSnapshot) -> None:
        """Print progress and any newly observed errors as plain text lines."""
        if snapshot.failed_count > self._previous_failed_count and snapshot.last_error:
            print(f"ERROR: {snapshot.last_error}", flush=True)

        self._previous_failed_count = snapshot.failed_count
        percentage = (
            f"{(snapshot.processed * 100) // snapshot.total}%"
            if snapshot.total > 0
            else "?%"
        )
        print(
            f"{self._description}: "
            f"[{percentage}] {snapshot.processed}/{snapshot.total} "
            f"ok={snapshot.ok_count} failed={snapshot.failed_count} "
            f"last_id={snapshot.last_artifact_id or '-'}",
            flush=True,
        )
