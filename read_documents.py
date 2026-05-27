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
        "--save",
        action="store_true",
        help="Write documents.json to the output directory.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    documents = read_documents(config, limit=args.limit)

    print(f"Retrieved {len(documents)} documents.")
    if documents:
        preview = ", ".join(doc.control_number for doc in documents[:5])
        print(f"First control numbers: {preview}")

    if args.save:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "documents.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump([doc.model_dump() for doc in documents], handle, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
