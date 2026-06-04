"""CLI for creating and updating declarative Elasticsearch indices."""

import argparse
from pathlib import Path
from typing import cast

from es_index_explorer.config import load_config
from es_index_explorer.index_setup import (
    IndexDefinition,
    IndexDiff,
    IndexSetupError,
    apply_in_place,
    create_index,
    diff_index,
    ensure_inference_endpoint,
    index_exists,
    load_index_definition,
    recreate_index,
    run_with_auth_retry,
)

DEFAULT_INDEX_DEFINITION = Path("es_index_explorer/index_definitions/air_assist_nested.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Set up an Elasticsearch index from a JSON definition.")
    parser.add_argument("index_name", help="Elasticsearch index name.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.toml (default: ./config.toml).",
    )
    parser.add_argument(
        "--index-structure",
        default=str(DEFAULT_INDEX_DEFINITION),
        help="Path to the JSON index definition file.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Apply additive in-place updates when possible.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Allow delete-and-recreate for breaking/static changes (destructive).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation for destructive recreate.",
    )
    return parser.parse_args()


def _confirm_recreate(index_name: str, assume_yes: bool) -> bool:
    """Confirm whether destructive recreate is allowed."""

    if assume_yes:
        return True
    answer = input(
        f"Recreate index '{index_name}'? This deletes existing data before re-creating it. [y/N]: "
    ).strip()
    return answer.lower() in {"y", "yes"}


def _warn_if_non_standard_index_name(index_name: str) -> None:
    """Warn when index name does not follow the expected prefix."""

    if not index_name.startswith("as-"):
        print(f"Warning: '{index_name}' does not start with 'as-'.")


def _ensure_elser_precheck(config_path: str | None, definition: IndexDefinition) -> None:
    """Optionally validate ELSER endpoint when semantic_text fields are present."""

    mappings_text = str(definition.mappings)
    if "semantic_text" not in mappings_text:
        return
    config = load_config(config_path)
    run_with_auth_retry(config, lambda client: ensure_inference_endpoint(client), write=False)


def _load_diff(config_path: str | None, index_name: str, definition: IndexDefinition) -> IndexDiff:
    """Load diff for an existing index."""

    config = load_config(config_path)
    result = run_with_auth_retry(
        config,
        lambda client: diff_index(client, index_name, definition),
        write=False,
    )
    return cast(IndexDiff, result)


def _read_exists(config_path: str | None, index_name: str) -> bool:
    """Return whether the index currently exists."""

    config = load_config(config_path)
    result = run_with_auth_retry(config, lambda client: index_exists(client, index_name), write=False)
    return bool(result)


def _create(config_path: str | None, index_name: str, definition: IndexDefinition) -> None:
    """Create an index from scratch."""

    config = load_config(config_path)
    run_with_auth_retry(config, lambda client: create_index(client, index_name, definition), write=True)
    print(f"Created index '{index_name}'.")


def _apply_update(config_path: str | None, index_name: str, definition: IndexDefinition, diff: IndexDiff) -> None:
    """Apply in-place update to an existing index."""

    _ensure_elser_precheck(config_path, definition)
    config = load_config(config_path)
    run_with_auth_retry(
        config,
        lambda client: apply_in_place(client, index_name, diff, definition),
        write=True,
    )
    print(f"Applied in-place update to '{index_name}'.")


def _recreate(config_path: str | None, index_name: str, definition: IndexDefinition, assume_yes: bool) -> None:
    """Recreate an existing index."""

    _ensure_elser_precheck(config_path, definition)
    config = load_config(config_path)
    run_with_auth_retry(
        config,
        lambda client: recreate_index(
            client,
            index_name,
            definition,
            confirm=lambda target_name: _confirm_recreate(target_name, assume_yes),
        ),
        write=True,
    )
    print(f"Recreated index '{index_name}'.")


def main() -> None:
    """Run index setup workflow."""

    args = parse_args()
    _warn_if_non_standard_index_name(args.index_name)
    definition = load_index_definition(args.index_structure)
    exists = _read_exists(args.config, args.index_name)

    if not args.update and not args.recreate:
        if not exists:
            _create(args.config, args.index_name, definition)
            return
        diff = _load_diff(args.config, args.index_name, definition)
        print(diff.render())
        print("No changes applied. Re-run with --update or --recreate to make changes.")
        return

    if args.update and not args.recreate:
        if not exists:
            print(f"Index '{args.index_name}' does not exist; creating it.")
            _create(args.config, args.index_name, definition)
            return
        diff = _load_diff(args.config, args.index_name, definition)
        print(diff.render())
        if diff.is_empty():
            print("Index already matches the definition.")
            return
        if diff.is_in_place_applicable():
            _apply_update(args.config, args.index_name, definition, diff)
            return
        print("Breaking/static changes detected; no changes applied. Re-run with --recreate.")
        return

    if args.recreate and not args.update:
        if not exists:
            print(f"Index '{args.index_name}' does not exist; creating it.")
            _create(args.config, args.index_name, definition)
            return
        diff = _load_diff(args.config, args.index_name, definition)
        print(diff.render())
        _recreate(args.config, args.index_name, definition, args.yes)
        return

    # args.update and args.recreate
    if not exists:
        print(f"Index '{args.index_name}' does not exist; creating it.")
        _create(args.config, args.index_name, definition)
        return
    diff = _load_diff(args.config, args.index_name, definition)
    print(diff.render())
    if diff.is_empty():
        print("Index already matches the definition.")
        return
    if diff.is_in_place_applicable():
        _apply_update(args.config, args.index_name, definition, diff)
        return
    print("In-place update is not possible; falling back to --recreate flow.")
    _recreate(args.config, args.index_name, definition, args.yes)


if __name__ == "__main__":
    try:
        main()
    except IndexSetupError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
