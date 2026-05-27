"""CLI for reading RelativityOne documents."""

import argparse
import json
from pathlib import Path

from es_index_explorer.config import load_config
from es_index_explorer.relativity.reader import read_documents


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
    parser.add_argument("--output-dir", default=".", help="Directory for output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = read_documents(config, limit=args.limit)

    print(f"Retrieved {len(result.documents)} documents.")
    if result.documents:
        preview = ", ".join(doc.control_number for doc in result.documents[:5])
        print(f"First control numbers: {preview}")
    if result.failures:
        print(f"Skipped {len(result.failures)} documents due to validation errors.")

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
