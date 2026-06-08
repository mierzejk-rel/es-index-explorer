"""Resume helpers for the aiR Assist ingest pipeline."""

import argparse
import pickle
from typing import cast

from elasticsearch import Elasticsearch
from jinja2.ext import do

from es_index_explorer.client import with_auth_retry
from es_index_explorer.config import Config, load_config
from es_index_explorer.relativity.auth import get_authenticated_session
from es_index_explorer.relativity.client import RelativityClient
from es_index_explorer.relativity.fluent import field as r1_field


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for resume helpers."""

    parser = argparse.ArgumentParser(description="Resume utilities for the aiR Assist ingest pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.toml (default: ./config.toml).")
    parser.add_argument(
        "--index-name",
        default=None,
        help="Target Elasticsearch index (overrides elasticsearch.index_name from config).",
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--max-id",
        action="store_true",
        help="Print the greatest indexed document_artifact_id.",
    )
    actions.add_argument(
        "--list-ids",
        action="store_true",
        help="Print all indexed document_artifact_id values in ascending order, space-separated.",
    )
    actions.add_argument(
        "--list-r1-ids",
        action="store_true",
        help="Print all RelativityOne document Artifact IDs in ascending order, space-separated.",
    )
    actions.add_argument(
        "--get-id",
        action="store_true",
        help="Placeholder operation for orchestrating all ID-related flows.",
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


def _list_all_artifact_ids(client: Elasticsearch, index_name: str, page_size: int = 1000) -> list[int]:
    """Return all indexed artifact ids sorted ascending."""

    ids: list[int] = []
    search_after: list[object] | None = None

    while True:
        response = client.search(
            index=index_name,
            size=page_size,
            sort=[{"document_artifact_id": "asc"}],
            source=["document_artifact_id"],
            search_after=search_after,
        )
        hits = cast(list[dict[str, object]], response.get("hits", {}).get("hits", []))
        if not hits:
            return ids

        for hit in hits:
            source = cast(dict[str, object], hit.get("_source", {}))
            artifact_id = source.get("document_artifact_id")
            if isinstance(artifact_id, int):
                ids.append(artifact_id)
            elif isinstance(artifact_id, str) and artifact_id.isdigit():
                ids.append(int(artifact_id))
            else:
                raise SystemExit("Encountered a hit without a valid integer document_artifact_id.")

        raw_sort = hits[-1].get("sort")
        if not isinstance(raw_sort, list):
            raise SystemExit("Search results missing sort values required for pagination.")
        search_after = raw_sort


def _list_r1_artifact_ids(config: Config) -> list[int]:
    """Return all RelativityOne document Artifact IDs sorted ascending."""

    session = get_authenticated_session(config).session
    client = RelativityClient(config.relativity.host, config.relativity.workspace_id, session)
    builder = (
        client.query_object_manager()
        .from_documents()
        .select("Artifact ID")
        .sort_by("Artifact ID", direction="Ascending")
    )
    if config.relativity.saved_search_id is not None:
        builder = builder.where(r1_field("Artifact ID").in_saved_search(config.relativity.saved_search_id))

    ids: list[int] = []
    batch_size = config.relativity.batch_size
    response = builder.page(0, batch_size).execute_raw()
    start = 1
    while response.Objects:
        for obj in response.Objects:
            ids.append(int(obj.ArtifactID))
        start += batch_size
        response = builder.page(start, batch_size).execute_raw()
    return ids


def _run_get_id_operation(config: Config, index_name: str) -> None:
    from pathlib import Path
    import pickle

    r1_ids_path = Path(r'r1_ids.pkl')
    if r1_ids_path.is_file():
        with open(r1_ids_path, 'rb') as f:
            r1_ids = pickle.load(f)
    else:
        r1_ids = _list_r1_artifact_ids(config)
        with open(r1_ids_path, 'wb') as f:
            pickle.dump(r1_ids, f)

    r1_ids_len = len(r1_ids)

    es_ids_path = Path(r'es_ids.pkl')
    if es_ids_path.is_file():
        with open(es_ids_path, 'rb') as f:
            es_ids = pickle.load(f)
    else:
        es_ids = cast(list[int],
                      with_auth_retry(
                          config,
                          lambda client: _list_all_artifact_ids(client, index_name))
                      )
        with open(es_ids_path, 'wb') as f:
            pickle.dump(es_ids, f)


    es_ids_len = len(es_ids)
    es_max_id = cast(int | None,
                     with_auth_retry(
                         config,
                         lambda client: _get_max_artifact_id(client, index_name)))

    assert es_ids_len <= r1_ids_len
    r1_i = -1
    missing_ids: list[int] = []

    def next_r1_id(value: int) -> int | None:
        nonlocal r1_i
        r1_value = -1 if r1_i < 0 else r1_ids[r1_i]
        while r1_i < r1_ids_len - 1:
            r1_next = r1_ids[r1_i := r1_i + 1]
            assert r1_value < r1_next
            if r1_next == value:
                return r1_next

            assert r1_next < value
            missing_ids.append(r1_next)
            r1_value = r1_next

        return None
    es_i = 0
    es_value = es_ids[es_i]
    while next_r1_id(es_value) is not None and (es_i := es_i + 1) < es_ids_len:
        es_next = es_ids[es_i]
        assert es_value < es_next
        es_value = es_next

    print(f'Missing ids: {missing_ids}')
    assert es_value == es_max_id
    assert r1_ids[r1_i] == es_value
    print(f'Last Document Artifact ID: {es_value}')



def main() -> None:
    """Run the resume utility CLI."""

    args = parse_args()
    config = load_config(args.config)

    if args.get_id:
        _run_get_id_operation(config, _resolve_index(args, config))
        return

    if args.list_r1_ids:
        ids = _list_r1_artifact_ids(config)
        if not ids:
            print("No documents found in the workspace.")
        else:
            print(" ".join(str(artifact_id) for artifact_id in ids))
        return

    if args.list_ids:
        index_name = _resolve_index(args, config)
        ids = cast(list[int], with_auth_retry(config, lambda client: _list_all_artifact_ids(client, index_name)))
        if not ids:
            print("No documents found in the index.")
        else:
            print(" ".join(str(artifact_id) for artifact_id in ids))
        return

    index_name = _resolve_index(args, config)
    max_id = cast(int | None, with_auth_retry(config, lambda client: _get_max_artifact_id(client, index_name)))
    if max_id is None:
        print("No documents found in the index.")
    else:
        print(f"Max indexed document_artifact_id: {max_id}")


if __name__ == "__main__":
    main()
