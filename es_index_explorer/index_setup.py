"""Index setup helpers for declarative Elasticsearch mappings."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from elasticsearch import ApiError, AuthenticationException, Elasticsearch, NotFoundError

from es_index_explorer.auth import get_token
from es_index_explorer.client import get_client, with_auth_retry
from es_index_explorer.config import Config

DYNAMIC_SETTING_KEYS: set[str] = {
    "number_of_replicas",
    "refresh_interval",
    "max_inner_result_window",
    "mapping.nested_objects.limit",
    "mapping.total_fields.limit",
}


@dataclass(frozen=True)
class IndexDefinition:
    """Represent a declarative index definition."""

    settings: dict[str, Any]
    mappings: dict[str, Any]

    def create_body(self) -> dict[str, Any]:
        """Return the full payload used by ``indices.create``."""

        return {
            "settings": self.settings,
            "mappings": self.mappings,
        }


@dataclass(frozen=True)
class IndexDiff:
    """Represent differences between desired and existing index state."""

    additive_fields: list[str]
    breaking_fields: list[str]
    dynamic_settings: dict[str, tuple[Any, Any]]
    static_settings: dict[str, tuple[Any, Any]]
    additive_mapping_payload: dict[str, Any] = field(default_factory=dict, repr=False)
    dynamic_settings_payload: dict[str, Any] = field(default_factory=dict, repr=False)

    def is_empty(self) -> bool:
        """Return ``True`` when no differences were detected."""

        return (
            not self.additive_fields
            and not self.breaking_fields
            and not self.dynamic_settings
            and not self.static_settings
        )

    def is_in_place_applicable(self) -> bool:
        """Return ``True`` when in-place update can be safely applied."""

        return not self.breaking_fields and not self.static_settings

    def render(self) -> str:
        """Render a human-readable summary of all differences."""

        lines: list[str] = []
        if self.is_empty():
            return "No differences found."

        lines.append("Detected index differences:")
        if self.additive_fields:
            lines.append("- Additive fields:")
            lines.extend(f"  - {field_name}" for field_name in self.additive_fields)
        if self.breaking_fields:
            lines.append("- Breaking field changes:")
            lines.extend(f"  - {field_name}" for field_name in self.breaking_fields)
        if self.dynamic_settings:
            lines.append("- Dynamic setting changes:")
            for key, (current, desired) in sorted(self.dynamic_settings.items()):
                lines.append(f"  - {key}: {current!r} -> {desired!r}")
        if self.static_settings:
            lines.append("- Static setting changes:")
            for key, (current, desired) in sorted(self.static_settings.items()):
                lines.append(f"  - {key}: {current!r} -> {desired!r}")
        return "\n".join(lines)


class IndexSetupError(RuntimeError):
    """Raised when index setup operations fail."""


def load_index_definition(path: str | Path) -> IndexDefinition:
    """Load an index definition from a JSON file."""

    definition_path = Path(path)
    if not definition_path.exists():
        raise IndexSetupError(f"Index definition file does not exist: {definition_path}")
    try:
        raw = json.loads(definition_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexSetupError(f"Invalid JSON in index definition file: {definition_path}") from exc
    if not isinstance(raw, dict):
        raise IndexSetupError("Index definition root must be a JSON object.")
    settings = raw.get("settings")
    mappings = raw.get("mappings")
    if not isinstance(settings, dict) or not isinstance(mappings, dict):
        raise IndexSetupError("Index definition must include object fields 'settings' and 'mappings'.")
    return IndexDefinition(settings=settings, mappings=mappings)


def index_exists(client: Elasticsearch, name: str) -> bool:
    """Return whether an index exists and is accessible."""

    return bool(client.indices.exists(index=name))


def format_es_error(exc: Exception) -> str:
    """Format Elasticsearch errors in a consistent human-readable form."""

    if not isinstance(exc, ApiError):
        return str(exc)
    status = exc.status_code
    error_type: str | None = None
    error_reason: str | None = None
    body = exc.body
    if isinstance(body, dict):
        raw_error = body.get("error")
        if isinstance(raw_error, dict):
            error_type = raw_error.get("type")
            error_reason = raw_error.get("reason")
    parts = [f"status={status}"]
    if error_type:
        parts.append(f"type={error_type}")
    if error_reason:
        parts.append(f"reason={error_reason}")
    return ", ".join(parts)


def run_with_auth_retry(config: Config, func: Any, *, write: bool) -> object:
    """Run an Elasticsearch function with CID token refresh on auth failure."""

    if not write:
        return with_auth_retry(config, func)
    try:
        return func(get_client(config))
    except AuthenticationException:
        token = get_token(config, force_refresh=True)
        client = Elasticsearch(config.elasticsearch.hosts, bearer_auth=token)
        return func(client)


def ensure_inference_endpoint(
    client: Elasticsearch,
    inference_id: str = ".elser-2-elasticsearch",
) -> None:
    """Best-effort check that the requested inference endpoint exists."""

    if not hasattr(client, "inference"):
        return
    try:
        client.inference.get(inference_id=inference_id)
    except NotFoundError as exc:
        raise IndexSetupError(
            f"Inference endpoint '{inference_id}' is missing. Create it before setting up the index."
        ) from exc
    except ApiError as exc:
        # Optional pre-check: skip when caller lacks inference privileges.
        if exc.status_code in {401, 403}:
            return
        raise IndexSetupError(f"Unable to validate inference endpoint '{inference_id}': {format_es_error(exc)}") from exc


def create_index(client: Elasticsearch, name: str, definition: IndexDefinition) -> None:
    """Create an index and verify that it matches the declarative definition."""

    created = False
    try:
        client.indices.create(index=name, settings=definition.settings, mappings=definition.mappings)
        created = True
        verification = diff_index(client, name, definition)
        if not verification.is_empty():
            raise IndexSetupError(
                "Index was created but does not match the requested definition.\n"
                f"{verification.render()}"
            )
    except Exception as exc:
        rollback_error: str | None = None
        if created:
            try:
                client.indices.delete(index=name)
            except Exception as delete_exc:  # noqa: BLE001
                rollback_error = format_es_error(delete_exc) if isinstance(delete_exc, Exception) else str(delete_exc)
        if isinstance(exc, IndexSetupError):
            if rollback_error:
                raise IndexSetupError(f"{exc}\nRollback delete failed: {rollback_error}") from exc
            raise
        message = f"Failed to create index '{name}': {format_es_error(exc if isinstance(exc, Exception) else Exception())}"
        if rollback_error:
            message = f"{message}\nRollback delete failed: {rollback_error}"
        raise IndexSetupError(message) from exc


def diff_index(client: Elasticsearch, name: str, definition: IndexDefinition) -> IndexDiff:
    """Compare existing index state to the desired definition."""

    if not index_exists(client, name):
        raise IndexSetupError(f"Index '{name}' does not exist or is not accessible.")

    try:
        mapping_response = client.indices.get_mapping(index=name).body
        settings_response = client.indices.get_settings(index=name).body
    except Exception as exc:  # noqa: BLE001
        raise IndexSetupError(f"Failed to read index state for '{name}': {format_es_error(exc)}") from exc

    mapping_root = mapping_response.get(name, {}).get("mappings", {})
    existing_properties = mapping_root.get("properties", {})
    desired_properties = definition.mappings.get("properties", {})
    if not isinstance(existing_properties, dict) or not isinstance(desired_properties, dict):
        raise IndexSetupError("Both existing and desired mappings must include object 'properties'.")

    existing_leaves = _flatten_leaf_fields(existing_properties)
    desired_leaves = _flatten_leaf_fields(desired_properties)

    additive_fields = sorted(path for path in desired_leaves if path not in existing_leaves)
    additive_mapping_payload = _build_additive_mapping_payload(desired_properties, additive_fields)

    breaking_fields: list[str] = []
    for path in sorted(existing_leaves):
        if path not in desired_leaves:
            breaking_fields.append(f"{path} (removed)")
            continue
        existing_spec = existing_leaves[path]
        desired_spec = desired_leaves[path]
        if _is_breaking_field_change(existing_spec, desired_spec):
            breaking_fields.append(path)

    existing_index_settings = settings_response.get(name, {}).get("settings", {}).get("index", {})
    if not isinstance(existing_index_settings, dict):
        existing_index_settings = {}
    desired_index_settings = _extract_index_settings(definition.settings)
    desired_settings_flat = _flatten_settings(desired_index_settings)
    existing_settings_flat = _flatten_settings(existing_index_settings)

    dynamic_settings: dict[str, tuple[Any, Any]] = {}
    static_settings: dict[str, tuple[Any, Any]] = {}
    dynamic_settings_payload: dict[str, Any] = {}
    for key, desired_value in desired_settings_flat.items():
        if key not in existing_settings_flat:
            continue
        existing_value = existing_settings_flat[key]
        normalized_existing = _normalize_setting_value(existing_value)
        normalized_desired = _normalize_setting_value(desired_value)
        if normalized_existing == normalized_desired:
            continue
        if key in DYNAMIC_SETTING_KEYS:
            dynamic_settings[key] = (existing_value, desired_value)
            dynamic_settings_payload[key] = desired_value
        else:
            static_settings[key] = (existing_value, desired_value)

    return IndexDiff(
        additive_fields=additive_fields,
        breaking_fields=breaking_fields,
        dynamic_settings=dynamic_settings,
        static_settings=static_settings,
        additive_mapping_payload=additive_mapping_payload,
        dynamic_settings_payload=dynamic_settings_payload,
    )


def apply_in_place(client: Elasticsearch, name: str, diff: IndexDiff, definition: IndexDefinition) -> None:
    """Apply additive mapping changes and dynamic settings to an index."""

    if not diff.is_in_place_applicable():
        raise IndexSetupError("In-place update cannot be applied because breaking changes were detected.")

    try:
        if diff.additive_fields:
            client.indices.put_mapping(index=name, body=diff.additive_mapping_payload)
        if diff.dynamic_settings_payload:
            client.indices.put_settings(index=name, settings=diff.dynamic_settings_payload)
    except Exception as exc:  # noqa: BLE001
        raise IndexSetupError(f"Failed to apply in-place update to '{name}': {format_es_error(exc)}") from exc

    verification = diff_index(client, name, definition)
    if not verification.is_empty():
        raise IndexSetupError(
            "In-place update finished but the index still differs from definition.\n"
            f"{verification.render()}"
        )


def recreate_index(
    client: Elasticsearch,
    name: str,
    definition: IndexDefinition,
    *,
    confirm: Any,
) -> None:
    """Recreate an index safely after validating mapping/settings on a temporary index."""

    if not confirm(name):
        raise IndexSetupError("Recreate cancelled by user.")

    temp_name = f"{name}-validate-{uuid4().hex[:8]}"
    temp_created = False
    had_existing_index = index_exists(client, name)

    try:
        client.indices.create(index=temp_name, settings=definition.settings, mappings=definition.mappings)
        temp_created = True
        temp_diff = diff_index(client, temp_name, definition)
        if not temp_diff.is_empty():
            raise IndexSetupError(
                "Definition validation failed on temporary index.\n"
                f"{temp_diff.render()}"
            )
    except Exception as exc:  # noqa: BLE001
        message = (
            f"Definition validation failed on temporary index '{temp_name}': "
            f"{format_es_error(exc if isinstance(exc, Exception) else Exception())}"
        )
        raise IndexSetupError(message) from exc
    finally:
        if temp_created:
            # noinspection PyBroadException
            try:
                client.indices.delete(index=temp_name)
            except Exception:  # noqa: BLE001
                # Best effort cleanup only.
                pass

    if had_existing_index:
        try:
            client.indices.delete(index=name)
        except Exception as exc:  # noqa: BLE001
            raise IndexSetupError(f"Failed to delete existing index '{name}': {format_es_error(exc)}") from exc

    try:
        create_index(client, name, definition)
    except Exception as exc:  # noqa: BLE001
        if had_existing_index:
            raise IndexSetupError(
                f"Recreate failed after deleting existing index '{name}'. "
                f"Old index was dropped and re-create failed: {format_es_error(exc)}"
            ) from exc
        raise


def _extract_index_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Extract index-level settings from a create payload."""

    index_settings = settings.get("index")
    if isinstance(index_settings, dict):
        return index_settings
    return settings


def _flatten_settings(settings: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested settings dictionaries into dotted-key mapping."""

    flattened: dict[str, Any] = {}
    for key, value in settings.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_settings(value, full_key))
            continue
        flattened[full_key] = value
    return flattened


def _flatten_leaf_fields(properties: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    """Flatten mapping leaf fields including multi-fields."""

    flattened: dict[str, dict[str, Any]] = {}
    for field_name, field_info_raw in properties.items():
        full_name = f"{prefix}.{field_name}" if prefix else field_name
        if not isinstance(field_info_raw, dict):
            continue
        nested_properties = field_info_raw.get("properties")
        if isinstance(nested_properties, dict):
            flattened.update(_flatten_leaf_fields(nested_properties, full_name))
            continue
        field_info = dict(field_info_raw)
        multi_fields = field_info.pop("fields", None)
        flattened[full_name] = field_info
        if isinstance(multi_fields, dict):
            for sub_name, sub_info_raw in multi_fields.items():
                if isinstance(sub_info_raw, dict):
                    flattened[f"{full_name}.{sub_name}"] = dict(sub_info_raw)
    return flattened


def _is_breaking_field_change(existing: dict[str, Any], desired: dict[str, Any]) -> bool:
    """Return whether a leaf field definition change is breaking."""

    tracked_keys = {
        "type",
        "dims",
        "dimension",
        "similarity",
        "analyzer",
        "search_analyzer",
        "inference_id",
    }
    for key in tracked_keys:
        existing_value = existing.get(key)
        desired_value = desired.get(key)
        if _normalize_setting_value(existing_value) != _normalize_setting_value(desired_value):
            return True
    desired_index_options = desired.get("index_options")
    if desired_index_options is not None and not _normalized_contains(existing.get("index_options"), desired_index_options):
        return True
    # Elasticsearch always returns copy_to as a list, even when set as a scalar.
    if _normalize_copy_to(existing.get("copy_to")) != _normalize_copy_to(desired.get("copy_to")):
        return True
    return False


def _normalized_contains(existing: Any, desired: Any) -> bool:
    """Return True when normalized desired values exist in normalized existing values."""

    return _contains(_normalize_setting_value(existing), _normalize_setting_value(desired))


def _contains(existing: Any, desired: Any) -> bool:
    """Return True when `existing` recursively contains `desired`."""

    if isinstance(desired, dict):
        if not isinstance(existing, dict):
            return False
        return all(key in existing and _contains(existing[key], value) for key, value in desired.items())
    return existing == desired


def _normalize_copy_to(value: Any) -> list[str]:
    """Coerce a copy_to value into a sorted list for stable comparison."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    return [str(value)]


def _build_additive_mapping_payload(
    desired_properties: dict[str, Any],
    additive_paths: list[str],
) -> dict[str, Any]:
    """Build a ``put_mapping`` payload containing only additive fields."""

    if not additive_paths:
        return {}

    payload: dict[str, Any] = {"properties": {}}
    for path in additive_paths:
        field_definition = _lookup_field_definition(desired_properties, path)
        _insert_field(payload["properties"], path.split("."), field_definition)
    return payload


def _lookup_field_definition(desired_properties: dict[str, Any], path: str) -> dict[str, Any]:
    """Look up a leaf field definition by dotted path."""

    parts = path.split(".")
    current_properties: dict[str, Any] = desired_properties
    for index, part in enumerate(parts):
        raw_field = current_properties.get(part)
        if not isinstance(raw_field, dict):
            raise IndexSetupError(f"Unable to resolve additive mapping path '{path}'.")
        is_last = index == len(parts) - 1
        if is_last:
            return dict(raw_field)
        nested_properties = raw_field.get("properties")
        if isinstance(nested_properties, dict):
            current_properties = nested_properties
            continue
        multi_fields = raw_field.get("fields")
        if isinstance(multi_fields, dict) and index + 1 == len(parts) - 1:
            sub_field = multi_fields.get(parts[index + 1])
            if isinstance(sub_field, dict):
                return dict(sub_field)
        raise IndexSetupError(f"Unable to resolve additive mapping path '{path}'.")
    raise IndexSetupError(f"Unable to resolve additive mapping path '{path}'.")


def _insert_field(current: dict[str, Any], parts: list[str], definition: dict[str, Any]) -> None:
    """Insert a leaf field into a nested ``properties`` tree."""

    part = parts[0]
    if len(parts) == 1:
        current[part] = definition
        return
    node = current.get(part)
    if not isinstance(node, dict):
        node = {"properties": {}}
        current[part] = node
    nested_properties = node.get("properties")
    if not isinstance(nested_properties, dict):
        nested_properties = {}
        node["properties"] = nested_properties
    _insert_field(nested_properties, parts[1:], definition)


def _normalize_setting_value(value: Any) -> Any:
    """Normalize value types for stable setting comparisons."""

    match value:
        case str(text):
            stripped = text.strip()
            return int(stripped) if stripped.isdigit() else stripped
        case dict(mapping):
            return {
                key: _normalize_setting_value(item)
                for key, item in sorted(mapping.items())
            }
        case list(items):
            return [_normalize_setting_value(item) for item in items]
        case _:
            return value
