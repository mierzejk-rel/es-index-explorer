"""Read RelativityOne documents and optionally index them into Elasticsearch.

One entrypoint, one code path. `--source` selects the Object Manager read mechanism
(`queryslim`, the default, or `export`). With `--index` the documents are chunked,
embedded, and written to Elasticsearch, with a resumable JSONL progress log; without
`--index` it is a dry-run read (verify connectivity/readability only - no models are
loaded and nothing is written).
"""

import argparse
import sys
from pathlib import Path

from rich.live import Live

from es_index_explorer.config import load_config
from es_index_explorer.ingest.engine import IngestEngine, Mode
from es_index_explorer.ingest.sources import build_source
from es_index_explorer.relativity.progress import ProgressLog
from es_index_explorer.relativity.tui import ProgressSnapshot, ProgressView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read RelativityOne documents and optionally index them into Elasticsearch."
    )
    parser.add_argument("--config", default=None, help="Path to config.toml.")
    parser.add_argument(
        "--source",
        choices=["queryslim", "export"],
        default="queryslim",
        help="RelativityOne read mechanism (default: queryslim).",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index documents into Elasticsearch (chunk -> embed -> write). Without it this is a dry-run read.",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Target Elasticsearch index (overrides elasticsearch.index_name from config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Read page/block size (overrides relativity.batch_size; default 50).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="With --index, replace documents that already exist (default: report conflicts).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="With --index, discard the progress log and start from scratch.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="With --index, reprocess failed documents from the progress log.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N documents.")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for the progress log (used when indexing)."
    )
    return parser.parse_args()


def _resolve_indexing(args: argparse.Namespace) -> bool:
    """Decide whether this run writes to Elasticsearch.

    --index always writes. --index-name without --index is ambiguous: prompt in an
    interactive terminal (default No = dry run), and error out when non-interactive.
    """

    if args.index:
        return True
    if args.index_name is not None:
        if sys.stdin.isatty():
            answer = input(
                f"--index-name '{args.index_name}' was given without --index. "
                "Write to this Elasticsearch index? [y/N]: "
            ).strip().lower()
            return answer in ("y", "yes")
        raise SystemExit(
            "--index-name was given without --index in a non-interactive session. "
            "Pass --index to write to Elasticsearch, or omit --index-name for a dry-run read."
        )
    return False


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    indexing = _resolve_indexing(args)

    if not indexing and (args.overwrite or args.fresh or args.retry):
        raise SystemExit("--overwrite/--fresh/--retry require --index (or confirming --index-name).")

    batch_size = args.batch_size if args.batch_size is not None else config.relativity.batch_size
    if batch_size <= 0:
        raise SystemExit("batch size must be a positive integer.")

    source = build_source(config, args.source)

    view = ProgressView(total=None, description="Indexing" if indexing else "Reading")
    last_snapshot: ProgressSnapshot | None = None

    def _on_progress(snapshot: ProgressSnapshot) -> None:
        nonlocal last_snapshot
        last_snapshot = snapshot
        view.update(snapshot)
        live.update(view.render())

    progress_log: ProgressLog | None = None
    try:
        if indexing:
            # Lazy import: only an indexing run loads the embedding/segmentation models.
            from es_index_explorer.indexing.pipeline import IndexingPipeline

            output_dir = Path(args.output_dir).expanduser().resolve()
            progress_log = ProgressLog(
                host=config.relativity.host,
                workspace_id=config.relativity.workspace_id,
                saved_search_id=config.relativity.saved_search_id,
                directory=output_dir,
            )
            pipeline = IndexingPipeline(config, overwrite=args.overwrite, index_name=args.index_name)

            mode: Mode
            resume_after = 0
            retry_ids: list[int] | None = None
            if args.retry:
                retry_ids = list(progress_log.load().failed)
                if not retry_ids:
                    print("No failures to retry.")
                    return
                mode = "retry"
            elif args.fresh:
                progress_log.reset()
                mode = "fresh"
            else:
                mode = "resume"
                resume_after = progress_log.load().last_processed_id

            engine = IngestEngine(
                source=source,
                saved_search_id=config.relativity.saved_search_id,
                batch_size=batch_size,
                on_progress=_on_progress,
                progress_log=progress_log,
                sink=pipeline.index_documents,
                limit=args.limit,
            )
            with pipeline.refresh_disabled(), Live(view.render(), refresh_per_second=10) as live:
                engine.run(mode=mode, resume_after=resume_after, retry_ids=retry_ids)
                live.update(view.render())
        else:
            engine = IngestEngine(
                source=source,
                saved_search_id=config.relativity.saved_search_id,
                batch_size=batch_size,
                on_progress=_on_progress,
                progress_log=None,
                sink=None,
                limit=args.limit,
            )
            with Live(view.render(), refresh_per_second=10) as live:
                engine.run(mode="fresh", resume_after=0, retry_ids=None)
                live.update(view.render())
    finally:
        if progress_log is not None:
            progress_log.close()

    if last_snapshot is None:
        print("No documents to process.")
        return
    verb = "Indexed" if indexing else "Read"
    print(
        f"{verb} {last_snapshot.processed} document(s): "
        f"ok {last_snapshot.ok_count}, failed {last_snapshot.failed_count}."
    )


if __name__ == "__main__":
    main()
