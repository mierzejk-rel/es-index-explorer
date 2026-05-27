"""CID v1 token retrieval with disk cache."""

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from es_index_explorer.config import Config


class AuthenticationError(RuntimeError):
    """Raised when credentials are invalid or rejected by STS."""


class NetworkError(RuntimeError):
    """Raised when token acquisition fails after retries."""


# noinspection PyClassHasNoInit
@dataclass(frozen=True)
class CachedToken:
    """Represents a cached access token."""

    access_token: str
    expires_at: datetime


def _cache_path(config: Config) -> Path:
    return Path(config.token_cache.cache_file)


def _load_cached_token(config: Config) -> CachedToken | None:
    cache_path = _cache_path(config)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        expires_at = datetime.fromisoformat(data["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return CachedToken(access_token=data["access_token"], expires_at=expires_at)
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _is_token_valid(config: Config, token: CachedToken) -> bool:
    buffer_seconds = config.token_cache.buffer_seconds
    now = datetime.now(UTC)
    return token.expires_at - timedelta(seconds=buffer_seconds) > now


def _write_cached_token(config: Config, access_token: str, expires_in: int) -> None:
    cache_path = _cache_path(config)
    expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)
    payload = {
        "access_token": access_token,
        "expires_at": expires_at.isoformat(),
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_auth_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error") or payload.get("error_description")
        if error:
            return str(error)
    except json.JSONDecodeError:
        pass
    return response.text.strip() or f"STS returned HTTP {response.status_code}"


def _fetch_token(config: Config) -> tuple[str, int]:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-CSRF-Header": "-",
    }
    data = {
        "grant_type": config.cid.grant_type,
        "scope": config.cid.scope,
        "client_id": config.cid.client_id,
        "client_secret": config.cid.client_secret,
    }
    response = requests.post(
        config.cid.sts_url,
        data=data,
        headers=headers,
        timeout=60,
    )
    if response.status_code in (400, 401):
        raise AuthenticationError(_parse_auth_error(response))
    response.raise_for_status()
    payload = response.json()
    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    if not access_token or not expires_in:
        raise AuthenticationError("STS response missing access_token or expires_in.")
    return str(access_token), int(expires_in)


# noinspection PyTypeChecker
def _fetch_token_with_retry(config: Config) -> tuple[str, int]:
    attempts = 0
    while True:
        try:
            return _fetch_token(config)
        except AuthenticationError:
            raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempts < config.retry.max_attempts:
                attempts += 1
                time.sleep(config.retry.wait_seconds)
                continue
            raise NetworkError("Token request failed after retries.") from exc


def get_token(config: Config, force_refresh: bool = False) -> str:
    """Get a CID token with cache and retry.

    Parameters
    ----------
    config : Config
        Loaded application config.
    force_refresh : bool, optional
        When True, bypass cache and fetch a fresh token.

    Returns
    -------
    str
        Access token string.
    """

    if not force_refresh:
        cached = _load_cached_token(config)
        if cached and _is_token_valid(config, cached):
            return cached.access_token

    access_token, expires_in = _fetch_token_with_retry(config)
    _write_cached_token(config, access_token, expires_in)
    return access_token
