"""Resume helpers for the aiR Assist ingest pipeline."""

import argparse
from typing import cast

from elasticsearch import Elasticsearch

from es_index_explorer.client import with_auth_retry
from es_index_explorer.config import Config, load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for resume helpers."""

    parser = argparse.ArgumentParser(description="Resume utilities for the aiR Assist ingest pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.toml (default: ./config.toml).")
    parser.add_argument(
        "--index-name",
        default=None,
        help="Target Elasticsearch index (overrides elasticsearch.index_name from config).",
    )
    return parser.parse_args()


def _resolve_index(args: argparse.Namespace, config: Config) -> str:
    """Resolve the target Elasticsearch index name."""

    resolved = (args.index_name or config.elasticsearch.index_name or "").strip()
    if not resolved:
        raise SystemExit("Index name must be set via --index-name or elasticsearch.index_name in config.")
    return resolved


def _get_max_artifact_id(client: Elasticsearch, index_name: str) -> int | None:
    """Return the greatest indexed document artifact id."""

    response = client.search(
        index=index_name,
        size=1,
        sort=[{"document_artifact_id": "desc"}],
        source=["document_artifact_id"],
    )
    hits = cast(list[dict[str, object]], response.get("hits", {}).get("hits", []))
    if not hits:
        return None
    source = cast(dict[str, object], hits[0].get("_source", {}))
    artifact_id = source.get("document_artifact_id")
    if isinstance(artifact_id, int):
        return artifact_id
    if isinstance(artifact_id, str) and artifact_id.isdigit():
        return int(artifact_id)
    raise SystemExit("Top hit is missing a valid integer document_artifact_id.")


def main() -> None:
    """Run the resume utility CLI."""

    args = parse_args()
    config = load_config(args.config)
    index_name = _resolve_index(args, config)
    max_id = cast(int | None, with_auth_retry(config, lambda client: _get_max_artifact_id(client, index_name)))
    if max_id is None:
        print("No documents found in the index.")
    else:
        print(f"Max indexed document_artifact_id: {max_id}")


if __name__ == "__main__":
    main()
