"""Index inspection helpers."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from elasticsearch import Elasticsearch

QNA_EXPECTED_FIELDS = {
    "body",
    "title",
    "documentId",
    "controlNumber",
    "chunkId",
    "subsetIds",
    "embedding",
}


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


def inspect_index(client: Elasticsearch, index_name: str) -> dict[str, Any]:
    """Collect index structure and prepare a report."""

    info = client.indices.get(index=index_name)
    count = client.count(index=index_name)
    stats = client.indices.stats(index=index_name)
    cat = client.cat.indices(index=index_name, format="json")

    index_info = info[index_name]
    mappings = index_info.get("mappings", {})
    settings = index_info.get("settings", {}).get("index", {})
    analysis = _extract_analysis(settings)
    properties = mappings.get("properties", {})
    fields = _flatten_properties(properties)

    stats_summary = _index_stats(stats, index_name)
    size_mb = stats_summary["size_in_bytes"] / 1_000_000

    retrieval = _retrieval_hints(fields)
    present_qna_fields = sorted({f.name for f in fields if f.name in QNA_EXPECTED_FIELDS})
    metadata_fields = sorted({f.name for f in fields if f.name.startswith("metadata.")})

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
        field_details.append(f"- {field.name}: {', '.join(details)}")

    report_lines = [
        f"Index: {index_name}",
        f"Health: {cat[0].get('health') if cat else 'unknown'}",
        f"Status: {cat[0].get('status') if cat else 'unknown'}",
        f"Docs: {count.get('count', 0)}",
        f"Size (MB): {size_mb:.2f}",
        f"Shards: {cat[0].get('pri') if cat else 'unknown'} primary / {cat[0].get('rep') if cat else 'unknown'} replica",
        "",
        "Retrieval summary:",
        f"- Text fields (BM25): {', '.join(retrieval['text_fields']) or 'none'}",
        f"- Keyword fields (filters): {', '.join(retrieval['keyword_fields']) or 'none'}",
        f"- Vector fields (kNN): {', '.join(retrieval['vector_fields']) or 'none'}",
        f"- Numeric fields: {', '.join(retrieval['numeric_fields']) or 'none'}",
        f"- Date fields: {', '.join(retrieval['date_fields']) or 'none'}",
        "",
        "QnA schema alignment:",
        f"- Matching fields: {', '.join(present_qna_fields) or 'none'}",
        f"- Metadata fields: {', '.join(metadata_fields) or 'none'}",
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
            f"- Health: `{cat[0].get('health') if cat else 'unknown'}`",
            f"- Status: `{cat[0].get('status') if cat else 'unknown'}`",
            f"- Documents: `{count.get('count', 0)}`",
            f"- Size (MB): `{size_mb:.2f}`",
            f"- Shards: `{cat[0].get('pri') if cat else 'unknown'}` primary / `{cat[0].get('rep') if cat else 'unknown'}` replica",
            "",
            "## Retrieval Hints",
            f"- Text fields (BM25): {', '.join(retrieval['text_fields']) or 'none'}",
            f"- Keyword fields (filters): {', '.join(retrieval['keyword_fields']) or 'none'}",
            f"- Vector fields (kNN): {', '.join(retrieval['vector_fields']) or 'none'}",
            "",
            "## QnA Schema Alignment",
            f"- Matching fields: {', '.join(present_qna_fields) or 'none'}",
            f"- Metadata fields: {', '.join(metadata_fields) or 'none'}",
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
            "info": info,
            "count": count,
            "stats": stats,
            "cat": cat,
        },
        "text": "\n".join(report_lines),
        "markdown": markdown,
    }
