"""Configuration loading for es-index-explorer."""

import tomllib
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

# int = Relativity field ArtifactID; str = field display Name
RelativityFieldSelector = int | str


class CidConfig(BaseModel):
    """CID v1 configuration."""

    sts_url: str
    grant_type: str
    scope: str
    client_id: str
    client_secret: str


class TokenCacheConfig(BaseModel):
    """Token cache behavior."""

    buffer_seconds: int
    cache_file: str


class RetryConfig(BaseModel):
    """Retry behavior for network errors."""

    max_attempts: int
    wait_seconds: int


class ElasticsearchConfig(BaseModel):
    """Elasticsearch connection settings."""

    hosts: str | list[str]
    index_name: str = ""


class RelativityAuthConfig(BaseModel):
    """RelativityOne authentication settings."""

    method: Literal["basic", "oauth", "cid"]
    basic_auth: str = ""
    oauth_client_id: str = ""
    oauth_client_secret: str = ""


class RelativityFieldMappingConfig(BaseModel):
    """Relativity field mapping settings."""

    extracted_text: RelativityFieldSelector
    control_number: RelativityFieldSelector
    title: RelativityFieldSelector = ""
    primary_date_time: RelativityFieldSelector = ""
    email_from: RelativityFieldSelector = ""
    email_to: RelativityFieldSelector = ""
    email_cc: RelativityFieldSelector = ""
    email_bcc: RelativityFieldSelector = ""
    summary: RelativityFieldSelector = ""
    topic: RelativityFieldSelector = ""

    @field_validator(
        "extracted_text",
        "control_number",
        "title",
        "primary_date_time",
        "email_from",
        "email_to",
        "email_cc",
        "email_bcc",
        "summary",
        "topic",
    )
    @classmethod
    def validate_field_selector(cls, value: RelativityFieldSelector) -> RelativityFieldSelector:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("Field Artifact ID must be a positive integer.")
            return value
        if not value.strip():
            return ""
        return value

    @model_validator(mode="after")
    def validate_required_fields(self) -> "RelativityFieldMappingConfig":
        for name in ("extracted_text", "control_number"):
            if not is_field_configured(getattr(self, name)):
                raise ValueError(f"relativity.fields.{name} is required.")
        return self


def is_field_configured(value: RelativityFieldSelector) -> bool:
    if isinstance(value, int):
        return value > 0
    return bool(value.strip())


class RelativityConfig(BaseModel):
    """RelativityOne connection settings."""

    host: str
    tenant_id: str
    workspace_id: int
    saved_search_id: int | None = None
    subset_id: str = ""
    batch_size: int = 50
    max_text_length: int = 100_000
    auth: RelativityAuthConfig
    fields: RelativityFieldMappingConfig

    @field_validator("subset_id")
    @classmethod
    def normalize_subset_id(cls, value: str) -> str:
        # Treat missing / empty / whitespace-only as "no subset scoping".
        return value.strip()

    @field_validator("saved_search_id", mode="before")
    @classmethod
    def normalize_saved_search_id(cls, value: object | None) -> int | None:
        match value:
            case None:
                return None
            case str(s) if not s.strip():
                return None
            case str(s):
                return int(s)
            case int(n):
                return n
            case _:
                raise ValueError("saved_search_id must be an int or empty.")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be a positive integer.")
        return value

    @field_validator("max_text_length")
    @classmethod
    def validate_max_text_length(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_text_length must be a positive integer.")
        return value


class IndexingConfig(BaseModel):
    """Settings for the document-indexing pipeline (chunking, embedding, ES write)."""

    # Elasticsearch bulk write
    bulk_docs_per_request: int = 200
    bulk_max_retries: int = 3

    # Embedding model (HuggingFace, no company dependencies)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 96
    passage_prefix: str = "passage: "
    model_path: str = ""  # local model directory; empty = use the hub id
    offline: bool = False  # set HF_HUB_OFFLINE for air-gapped runs
    device: str = "cpu"  # CPU-only deployment (no GPU/MPS/NPU available)

    # Chunk geometry (e5 subword tokens). All SOFT except max_content_tokens.
    chunk_unique_target: int = 400
    chunk_unique_floor: int = 360
    overlap_target: int = 80
    overlap_min: int = 40
    overlap_max: int = 120
    max_content_tokens: int | None = None  # None = compute from the tokenizer at runtime

    # Sentence engine (priority 5)
    sentence_engine: Literal["sat", "blingfire", "sentencex", "pysbd"] = "sat"
    sat_model: str = "sat-12l-sm"

    # Clause engine (Tier-A fallback, priorities 4-2)
    clause_engine: Literal["spacy", "punctuation"] = "spacy"
    spacy_model: str = "en_core_web_sm"

    # Tier-B fallback (no sentence structure): legacy non-semantic sliding window
    fallback_window: int = 500
    fallback_overlap: int = 100

    @field_validator(
        "bulk_docs_per_request",
        "embedding_batch_size",
        "chunk_unique_target",
        "chunk_unique_floor",
        "overlap_target",
        "overlap_min",
        "overlap_max",
        "fallback_window",
        "fallback_overlap",
    )
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("indexing numeric settings must be positive integers.")
        return value

    @model_validator(mode="after")
    def validate_geometry(self) -> "IndexingConfig":
        if not self.overlap_min <= self.overlap_target <= self.overlap_max:
            raise ValueError("Require overlap_min <= overlap_target <= overlap_max.")
        if self.max_content_tokens is not None and self.max_content_tokens <= 0:
            raise ValueError("max_content_tokens must be a positive integer or null.")
        return self


class Config(BaseModel):
    """Full application configuration."""

    cid: CidConfig
    token_cache: TokenCacheConfig
    retry: RetryConfig
    elasticsearch: ElasticsearchConfig
    relativity: RelativityConfig
    indexing: IndexingConfig = IndexingConfig()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.toml"


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a TOML file.

    Parameters
    ----------
    path : str | Path | None
        Path to the config TOML. Defaults to `config.toml` in project root.

    Returns
    -------
    Config
        Parsed and validated configuration object.
    """

    config_path = Path(path) if path is not None else _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Create config.toml from config.example.toml."
        )
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return Config.model_validate(data)
