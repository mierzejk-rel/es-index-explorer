"""CLI for reading RelativityOne documents."""

import argparse
import json
from pathlib import Path

from rich.live import Live

from es_index_explorer.config import load_config
from es_index_explorer.relativity.reader import read_documents
from es_index_explorer.relativity.tui import ProgressSnapshot, ProgressView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read documents from RelativityOne.")
    parser.add_argument("--config", default=None, help="Path to config.toml.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N documents.")
    parser.add_argument(
        "--batch-size",
        dest="limit",
        type=int,
        default=None,
        help="Alias for --limit. For large workspaces with resume support use import_documents.py.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write documents.json to the output directory.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index the read documents into Elasticsearch (one-shot, no resume).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="With --index, replace documents that already exist (default: conflicts are reported).",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output files.")
    return parser.parse_args()


def _index_documents(config, documents, *, overwrite: bool) -> None:
    """Index already-read documents using the shared pipeline (one-shot)."""

    from collections import Counter

    from es_index_explorer.indexing.pipeline import IndexingPipeline

    pipeline = IndexingPipeline(config, overwrite=overwrite)
    with pipeline.refresh_disabled():
        results = pipeline.index_documents(documents)

    counts = Counter(result.outcome for result in results)
    summary = ", ".join(f"{outcome}: {count}" for outcome, count in sorted(counts.items()))
    print(f"Indexing complete ({summary or 'no documents'}).")
    for result in results:
        if result.outcome in ("conflict", "index_failed"):
            print(f"  {result.outcome} {result.artifact_id}: {result.error_type} - {result.error_message}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    view = ProgressView(total=None, description="Reading")
    with Live(view.render(), refresh_per_second=10) as live:
        def _on_progress(snapshot: ProgressSnapshot) -> None:
            view.update(snapshot)
            live.update(view.render())

        result = read_documents(config, limit=args.limit, on_progress=_on_progress)
        live.update(view.render())

    print(f"Retrieved {len(result.documents)} documents.")
    if result.documents:
        preview = ", ".join(doc.control_number for doc in result.documents[:5])
        print(f"First control numbers: {preview}")
    if result.failures:
        print(f"Skipped {len(result.failures)} documents due to validation errors.")

    if args.index and result.documents:
        _index_documents(config, result.documents, overwrite=args.overwrite)

    if args.save:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "documents.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                [doc.model_dump(mode="json") for doc in result.documents],
                handle,
                indent=2,
            )
        print(f"Wrote {path}")
        if result.failures:
            failures_path = output_dir / "documents_failed.json"
            with failures_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "artifact_id": failure.artifact_id,
                            "error": failure.error,
                            "raw_row": failure.raw_row,
                        }
                        for failure in result.failures
                    ],
                    handle,
                    indent=2,
                )
            print(f"Wrote {failures_path}")


if __name__ == "__main__":
    main()
