"""Batch import RelativityOne documents into the nested Elasticsearch index.

Reads in resumable batches (sorted by Artifact ID) and, for each batch, chunks +
embeds + bulk-writes to Elasticsearch via the shared IndexingPipeline. Supports
resume, retry of failures, and an explicit --overwrite for existing documents.
"""

import argparse
from pathlib import Path

from rich.live import Live

from es_index_explorer.config import load_config
from es_index_explorer.indexing.pipeline import IndexingPipeline
from es_index_explorer.relativity.batch_reader import BatchImporter
from es_index_explorer.relativity.progress import ProgressLog
from es_index_explorer.relativity.tui import ProgressSnapshot, ProgressView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch import RelativityOne documents into Elasticsearch.")
    parser.add_argument("--config", default=None, help="Path to config.toml.")
    parser.add_argument("--output-dir", default=".", help="Directory for progress log files.")
    parser.add_argument("--retry", action="store_true", help="Retry failed documents from the progress log.")
    parser.add_argument("--fresh", action="store_true", help="Discard existing progress log and start from scratch.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace documents that already exist in the index (default: existing ids are reported as conflicts).",
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

    # Loads the embedding + segmentation models once; the indexer is the batch sink.
    pipeline = IndexingPipeline(config, overwrite=args.overwrite)

    if args.retry:
        failed_ids = list(state.failed.keys())
        if not failed_ids:
            print("No failures to retry.")
            return
        view = ProgressView(total=len(failed_ids), description="Indexing (retry)")

        def _on_progress_retry(snapshot: ProgressSnapshot) -> None:
            view.update(snapshot)
            live.update(view.render())

        importer = BatchImporter(
            config, progress_log, on_progress=_on_progress_retry, sink=pipeline.index_documents
        )
        with pipeline.refresh_disabled(), Live(view.render(), refresh_per_second=10) as live:
            importer.retry_failed(failed_ids)
            live.update(view.render())
        progress_log.close()
        return

    first_snapshot: ProgressSnapshot | None = None

    def _on_progress(snapshot: ProgressSnapshot) -> None:
        nonlocal first_snapshot
        if first_snapshot is None:
            first_snapshot = snapshot
        view.update(snapshot)
        live.update(view.render())

    view = ProgressView(total=None, description="Indexing")
    importer = BatchImporter(config, progress_log, on_progress=_on_progress, sink=pipeline.index_documents)
    with pipeline.refresh_disabled(), Live(view.render(), refresh_per_second=10) as live:
        importer.run(resume_after=state.last_processed_id, state=state)
        if first_snapshot is None:
            view.mark_complete()
        live.update(view.render())

    if first_snapshot is None:
        print("Already up to date — no pending documents.")
    progress_log.close()


if __name__ == "__main__":
    main()
