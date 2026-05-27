"""Configuration loading for es-index-explorer."""

import tomllib
from pathlib import Path

from typing import Literal

from pydantic import BaseModel


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


class RelativityAuthConfig(BaseModel):
    """RelativityOne authentication settings."""

    method: Literal["basic", "oauth", "cid"]
    basic_auth: str = ""
    oauth_client_id: str = ""
    oauth_client_secret: str = ""


class RelativityFieldMappingConfig(BaseModel):
    """Relativity field mapping settings."""

    extracted_text: str
    control_number: str
    primary_date_time: str = ""
    email_from: str = ""
    email_to: str = ""
    email_cc: str = ""
    email_bcc: str = ""
    summary: str = ""
    topic: str = ""


class RelativityConfig(BaseModel):
    """RelativityOne connection settings."""

    host: str
    tenant_id: str
    workspace_id: int
    saved_search_id: int
    subset_id: str
    auth: RelativityAuthConfig
    fields: RelativityFieldMappingConfig


class Config(BaseModel):
    """Full application configuration."""

    cid: CidConfig
    token_cache: TokenCacheConfig
    retry: RetryConfig
    elasticsearch: ElasticsearchConfig
    relativity: RelativityConfig


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
