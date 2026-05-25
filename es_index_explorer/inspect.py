"""Index inspection helpers."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from elasticsearch import Elasticsearch


class IndexNotFoundError(Exception):
    """Raised when the requested index does not exist or is not accessible."""


# noinspection PyClassHasNoInit
@dataclass(frozen=True)
class FieldInfo:
    """Describes a single field mapping."""

    name: str
    field_type: str
    analyzer: str | None
    search_analyzer: str | None
    index: bool | None
    store: bool | None
    dims: int | None
    similarity: str | None
    index_options: dict[str, Any] | None


def _flatten_properties(
    properties: dict[str, Any], prefix: str = ""
) -> list[FieldInfo]:
    fields: list[FieldInfo] = []
    for name, info in properties.items():
        full_name = f"{prefix}{name}" if prefix else name
        field_type = info.get("type", "object")
        if "properties" in info:
            fields.extend(_flatten_properties(info["properties"], prefix=f"{full_name}."))
            continue
        fields.append(
            FieldInfo(
                name=full_name,
                field_type=field_type,
                analyzer=info.get("analyzer"),
                search_analyzer=info.get("search_analyzer"),
                index=info.get("index"),
                store=info.get("store"),
                dims=info.get("dims") or info.get("dimension"),
                similarity=info.get("similarity"),
                index_options=info.get("index_options"),
            )
        )
        for sub_name, sub_info in info.get("fields", {}).items():
            fields.append(
                FieldInfo(
                    name=f"{full_name}.{sub_name}",
                    field_type=sub_info.get("type", "object"),
                    analyzer=sub_info.get("analyzer"),
                    search_analyzer=sub_info.get("search_analyzer"),
                    index=sub_info.get("index"),
                    store=sub_info.get("store"),
                    dims=sub_info.get("dims") or sub_info.get("dimension"),
                    similarity=sub_info.get("similarity"),
                    index_options=sub_info.get("index_options"),
                )
            )
    return fields


def _index_stats(stats: dict[str, Any], index_name: str) -> dict[str, Any]:
    index_stats = stats.get("indices", {}).get(index_name, {}).get("total", {})
    store = index_stats.get("store", {})
    return {
        "size_in_bytes": store.get("size_in_bytes", 0),
        "docs_count": index_stats.get("docs", {}).get("count", 0),
    }


def _extract_analysis(settings: dict[str, Any]) -> dict[str, Any]:
    return settings.get("analysis", {})


def _retrieval_hints(fields: list[FieldInfo]) -> dict[str, list[str]]:
    text_fields = sorted({f.name for f in fields if f.field_type == "text"})
    keyword_fields = sorted({f.name for f in fields if f.field_type == "keyword"})
    vector_fields = sorted({f.name for f in fields if f.field_type == "dense_vector"})
    numeric_fields = sorted(
        {
            f.name
            for f in fields
            if f.field_type in {"integer", "long", "short", "byte", "double", "float"}
        }
    )
    date_fields = sorted({f.name for f in fields if f.field_type == "date"})
    return {
        "text_fields": text_fields,
        "keyword_fields": keyword_fields,
        "vector_fields": vector_fields,
        "numeric_fields": numeric_fields,
        "date_fields": date_fields,
    }


def _add_counts(a: int | None, b: int | None) -> int | None:
    """Merge two shard counts: None means 'not reported', int (including 0) means 'known value'."""
    if a is None and b is None:
        return None
    return (0 if a is None else a) + (0 if b is None else b)


def _maybe_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _collect_field_usage(stats: dict[str, Any]) -> dict[str, dict[str, int | None]]:
    """Aggregate field usage stats across shards.

    A None value means the counter was never reported by ES for that field
    (i.e. not applicable to that field type). An int value of 0 means
    it was reported but not used.
    """
    fields_usage: dict[str, dict[str, int | None]] = {}
    for shard in stats.get("shards", []):
        for field_name, field_stats in shard.get("stats", {}).get("fields", {}).items():
            usage = fields_usage.setdefault(field_name, {})
            inverted = field_stats.get("inverted_index")

            for key in ("any", "stored_fields", "doc_values", "points", "norms", "term_vectors", "knn_vectors"):
                usage[key] = _add_counts(usage.get(key), _maybe_int(field_stats.get(key)))

            for key in ("terms", "postings", "term_frequencies", "positions", "offsets", "payloads", "proximity"):
                shard_val = _maybe_int(inverted.get(key)) if inverted is not None else None
                usage[key] = _add_counts(usage.get(key), shard_val)

    return fields_usage


def list_indices(client: Elasticsearch) -> list[str]:
    """Return all index names accessible to the current credentials."""

    aliases = client.indices.get_alias()
    return sorted(aliases.body.keys())


def inspect_index(client: Elasticsearch, index_name: str) -> dict[str, Any]:
    """Collect index structure and prepare a report."""

    if not client.indices.exists(index=index_name):
        raise IndexNotFoundError(
            f"Index '{index_name}' does not exist or is not accessible with current credentials."
        )

    info = client.indices.get(index=index_name)
    count = client.count(index=index_name)
    stats = client.indices.stats(index=index_name)
    field_usage = client.indices.field_usage_stats(index=index_name)

    index_info = info[index_name]
    mappings = index_info.get("mappings", {})
    settings = index_info.get("settings", {}).get("index", {})
    analysis = _extract_analysis(settings)
    properties = mappings.get("properties", {})
    fields = _flatten_properties(properties)

    # shards/replicas come from settings; no cluster-level privilege needed
    num_shards = settings.get("number_of_shards", "unknown")
    num_replicas = settings.get("number_of_replicas", "unknown")

    stats_summary = _index_stats(stats, index_name)
    size_mb = stats_summary["size_in_bytes"] / 1_000_000

    retrieval = _retrieval_hints(fields)
    index_usage = _collect_field_usage(field_usage.get(index_name, {}))

    field_details: list[str] = []
    for field in fields:
        details = [f"type={field.field_type}"]
        if field.analyzer:
            details.append(f"analyzer={field.analyzer}")
        if field.search_analyzer:
            details.append(f"search_analyzer={field.search_analyzer}")
        if field.index is not None:
            details.append(f"index={field.index}")
        if field.store is not None:
            details.append(f"store={field.store}")
        if field.dims is not None:
            details.append(f"dims={field.dims}")
        if field.similarity:
            details.append(f"similarity={field.similarity}")
        if field.index_options:
            details.append(f"index_options={json.dumps(field.index_options)}")
        field_details.append(f"- {field.name}: {', '.join(details)}")

    usage_lines: list[str] = []
    for field_name, usage in sorted(index_usage.items()):
        # Show counts that are known (not None); flag non-zero ones to highlight actual use.
        known = {k: v for k, v in usage.items() if k != "any" and v is not None}
        if not known:
            continue
        parts = []
        for k, v in sorted(known.items()):
            parts.append(f"{k}={v}" if v == 0 else f"{k}={v}*")
        usage_lines.append(f"- {field_name}: {', '.join(parts)}")

    report_lines = [
        f"Index: {index_name}",
        f"Docs: {count.get('count', 0)}",
        f"Size (MB): {size_mb:.2f}",
        f"Shards: {num_shards} primary / {num_replicas} replica",
        "",
        "Retrieval summary:",
        f"- Text fields (BM25): {', '.join(retrieval['text_fields']) or 'none'}",
        f"- Keyword fields (filters): {', '.join(retrieval['keyword_fields']) or 'none'}",
        f"- Vector fields (kNN): {', '.join(retrieval['vector_fields']) or 'none'}",
        f"- Numeric fields: {', '.join(retrieval['numeric_fields']) or 'none'}",
        f"- Date fields: {', '.join(retrieval['date_fields']) or 'none'}",
        "",
        "Field usage (non-zero counters):",
        *(usage_lines or ["- none"]),
        "",
        "Field details:",
    ]

    report_lines.extend(field_details)

    timestamp = datetime.now(UTC).isoformat()
    markdown = "\n".join(
        [
            f"# Elasticsearch Index Report: {index_name}",
            "",
            f"Generated: `{timestamp}`",
            "",
            "## Overview",
            f"- Documents: `{count.get('count', 0)}`",
            f"- Size (MB): `{size_mb:.2f}`",
            f"- Shards: `{num_shards}` primary / `{num_replicas}` replica",
            "",
            "## Retrieval Hints",
            f"- Text fields (BM25): {', '.join(retrieval['text_fields']) or 'none'}",
            f"- Keyword fields (filters): {', '.join(retrieval['keyword_fields']) or 'none'}",
            f"- Vector fields (kNN): {', '.join(retrieval['vector_fields']) or 'none'}",
            "",
            "## Field Usage",
            "\n".join(usage_lines) if usage_lines else "No field usage stats reported.",
            "",
            "## Analysis Settings",
            "```json",
            json.dumps(analysis, indent=2),
            "```",
            "",
            "## Field Details",
            "\n".join(field_details) if field_details else "No fields found.",
        ]
    )

    return {
        "raw": {
            "info": info.body,
            "count": count.body,
            "stats": stats.body,
            "field_usage": field_usage.body,
        },
        "text": "\n".join(report_lines),
        "markdown": markdown,
    }
