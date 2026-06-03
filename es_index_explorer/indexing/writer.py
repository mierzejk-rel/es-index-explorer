"""Elasticsearch bulk writing with refresh handling and outcome classification."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from .document_builder import DocumentResult, Outcome


@contextmanager
def disabled_refresh(client: Elasticsearch, index: str) -> Iterator[None]:
    """Disable periodic refresh during a bulk load and restore it afterwards.

    Sets ``index.refresh_interval = -1`` for the duration, then issues one refresh
    and restores the original value (``None`` resets to the cluster default). The
    restore runs in ``finally`` so it always happens, even on error.
    """

    current = client.indices.get_settings(index=index)
    original = current[index]["settings"]["index"].get("refresh_interval")
    client.indices.put_settings(index=index, settings={"index": {"refresh_interval": "-1"}})
    try:
        yield
    finally:
        try:
            client.indices.refresh(index=index)
        finally:
            client.indices.put_settings(
                index=index, settings={"index": {"refresh_interval": original}}
            )


def bulk_index(
    client: Elasticsearch,
    actions: Iterable[dict[str, Any]],
    *,
    overwrite: bool,
    chunk_size: int,
    max_retries: int,
) -> list[DocumentResult]:
    """Bulk-write actions, returning a per-document outcome for each.

    Uses ``streaming_bulk`` with 429 retry. Per the overwrite policy, actions use
    ``create`` by default (409 -> ``conflict`` outcome) or ``index`` when overwriting
    (``result == "updated"`` -> ``overwritten``, else ``created``).
    """

    op_key = "index" if overwrite else "create"
    results: list[DocumentResult] = []
    for ok, info in streaming_bulk(
        client,
        actions,
        chunk_size=chunk_size,
        max_retries=max_retries,
        retry_on_status=(429,),
        raise_on_error=False,
        yield_ok=True,
    ):
        item = info.get(op_key) or next(iter(info.values()))
        artifact_id = _artifact_id(item.get("_id"))
        if ok:
            outcome: Outcome = "overwritten" if item.get("result") == "updated" else "created"
            results.append(DocumentResult(artifact_id=artifact_id, outcome=outcome))
            continue
        status = item.get("status")
        error = item.get("error")
        error_type, error_message = _format_es_error(error)
        if status == 409:
            results.append(
                DocumentResult(
                    artifact_id=artifact_id,
                    outcome="conflict",
                    stage="index",
                    error_type=error_type or "document_exists_conflict",
                    error_message=error_message or "document already exists",
                )
            )
        else:
            results.append(
                DocumentResult(
                    artifact_id=artifact_id,
                    outcome="index_failed",
                    stage="index",
                    error_type=error_type,
                    error_message=error_message,
                )
            )
    return results


def _artifact_id(raw_id: object) -> int | None:
    if isinstance(raw_id, str) and raw_id.isdigit():
        return int(raw_id)
    return None


def _format_es_error(error: object) -> tuple[str | None, str | None]:
    """Flatten an Elasticsearch bulk-item error into ``(error_type, message)``.

    Walks the nested ``caused_by`` chain (and the first ``root_cause``) so the recorded
    message exposes the actual root cause - e.g. an ELSER ``inference_exception`` whose
    real reason is "No ML nodes exist in the cluster" - instead of only the generic
    top-level ``reason``.
    """

    if not isinstance(error, dict):
        # noinspection PyStringConversionWithoutDunderMethod
        return None, (str(error) if error is not None else None)
    error_type = error.get("type")
    segments: list[str] = []
    top_reason = error.get("reason")
    if top_reason:
        segments.append(str(top_reason))
    cause = error.get("caused_by")
    while isinstance(cause, dict):
        cause_type = cause.get("type")
        cause_reason = cause.get("reason") or ""
        label = f"[{cause_type}] " if cause_type else ""
        segments.append(f"caused_by: {label}{cause_reason}".rstrip())
        cause = cause.get("caused_by")
    root_cause = error.get("root_cause")
    if isinstance(root_cause, list) and root_cause and isinstance(root_cause[0], dict):
        root_reason = root_cause[0].get("reason")
        if root_reason and not any(str(root_reason) in segment for segment in segments):
            root_type = root_cause[0].get("type")
            label = f"[{root_type}] " if root_type else ""
            segments.append(f"root_cause: {label}{root_reason}")
    message = " <- ".join(segments) if segments else None
    return (str(error_type) if error_type is not None else None), message
