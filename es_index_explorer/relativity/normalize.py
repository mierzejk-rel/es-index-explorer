"""Normalization helpers for Relativity document fields."""

import warnings


class DocumentReadError(RuntimeError):
    """Raised when required document fields are missing."""


class DocumentFetchError(RuntimeError):
    """Raised when a per-document network fetch fails after retries."""


_POOR_QUALITY_TOPIC = "Poor Quality Extracted Text"


def normalize_str(value: object | None) -> str | None:
    match value:
        case str(text):
            return text.strip() or None
        case [str(text), *rest]:
            if rest:
                warnings.warn(
                    "normalize_str received a sequence with multiple values; using the first item.",
                    stacklevel=2,
                )
            return text.strip() or None
        case _:
            return None


def normalize_str_list(value: object | None) -> list[str] | None:
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    normalized = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized or None


def normalize_summary_topic(
    raw_summary: object | None,
    raw_topic: object | None,
) -> tuple[str | None, str | None]:
    summary = normalize_str(raw_summary)
    topic = normalize_str(raw_topic)
    if summary is None and topic is None:
        return None, None
    if topic == _POOR_QUALITY_TOPIC and summary is None:
        return "", ""
    return summary, topic


def ensure_required_field(value: str | None, field_name: str) -> str:
    if value is None:
        raise DocumentReadError(f"Missing required field '{field_name}'.")
    return value
