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
        total_value = total if total is not None else 0
        self._last_snapshot: ProgressSnapshot | None = ProgressSnapshot(
            processed=0,
            total=total_value,
            ok_count=0,
            failed_count=0,
            last_artifact_id=None,
            last_error=None,
        )

    def note_warning(self, message: str) -> None:
        """Print warning immediately so it is visible in non-interactive consoles."""
        print(f"WARNING: {message}", flush=True)

    def update(self, snapshot: ProgressSnapshot) -> None:
        """Handle per-document updates and print errors immediately."""
        if snapshot.failed_count > self._previous_failed_count and snapshot.last_error:
            print(f"ERROR: {snapshot.last_error}", flush=True)

        self._previous_failed_count = snapshot.failed_count
        self._last_snapshot = snapshot

    def flush_batch(self, snapshot: ProgressSnapshot | None = None) -> None:
        """Print one summary status line for the completed batch."""
        current = snapshot or self._last_snapshot
        if current is None:
            return
        percentage = (
            f"{(current.processed * 100) // current.total}%"
            if current.total > 0
            else "?%"
        )
        print(
            f"{self._description}: "
            f"[{percentage}] {current.processed}/{current.total} "
            f"ok={current.ok_count} failed={current.failed_count} "
            f"last_id={current.last_artifact_id or '-'}",
            flush=True,
        )
