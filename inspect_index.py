"""CLI for inspecting an Elasticsearch index."""

import argparse
import json
from pathlib import Path

from es_index_explorer.client import with_auth_retry
from es_index_explorer.config import load_config
from es_index_explorer.inspect import IndexNotFoundError, inspect_index, list_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an Elasticsearch index.")
    parser.add_argument("index_name", nargs="?", help="Elasticsearch index name.")
    parser.add_argument(
        "--config",
        help="Path to config.toml (default: ./config.toml).",
        default=None,
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write <index>.report.md and <index>.schema.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write report files when using --save.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of the text report.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List index names accessible to the current credentials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.list:
        def _list(client):
            return list_indices(client)

        indices = with_auth_retry(config, _list)
        print("\n".join(indices))
        return

    if not args.index_name:
        raise SystemExit("Error: index_name is required unless --list is used.")

    def _run(client):
        return inspect_index(client, args.index_name)

    try:
        result = with_auth_retry(config, _run)
    except IndexNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    if args.json:
        print(json.dumps(result["raw"], indent=2))
    else:
        print(result["text"])

    if args.save:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{args.index_name}.report.md"
        schema_path = output_dir / f"{args.index_name}.schema.json"
        report_path.write_text(result["markdown"], encoding="utf-8")
        schema_path.write_text(json.dumps(result["raw"], indent=2), encoding="utf-8")
        print(f"Saved {report_path} and {schema_path}")


if __name__ == "__main__":
    main()
