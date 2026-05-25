"""CLI for inspecting an Elasticsearch index."""

import argparse
import json
from pathlib import Path

from es_index_explorer.client import with_auth_retry
from es_index_explorer.config import load_config
from es_index_explorer.inspect import inspect_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an Elasticsearch index.")
    parser.add_argument("index_name", help="Elasticsearch index name.")
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
        "--json",
        action="store_true",
        help="Print raw JSON instead of the text report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    def _run(client):
        return inspect_index(client, args.index_name)

    result = with_auth_retry(config, _run)
    if args.json:
        print(json.dumps(result["raw"], indent=2))
    else:
        print(result["text"])

    if args.save:
        report_path = Path(f"{args.index_name}.report.md")
        schema_path = Path(f"{args.index_name}.schema.json")
        report_path.write_text(result["markdown"], encoding="utf-8")
        schema_path.write_text(json.dumps(result["raw"], indent=2), encoding="utf-8")
        print(f"Saved {report_path} and {schema_path}")


if __name__ == "__main__":
    main()
