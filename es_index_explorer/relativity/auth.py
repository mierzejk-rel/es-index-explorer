"""Authentication helpers for RelativityOne APIs."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import requests

from ..config import Config


class AuthError(RuntimeError):
    """Authentication error for RelativityOne sessions."""


@dataclass(frozen=True)
class RelativityAuthResult:
    """Resolved session + metadata."""

    session: requests.Session
    method: str


def _ensure_basic_auth(value: str) -> str:
    if not value:
        raise AuthError("relativity.auth.basic_auth is required for method=basic.")
    return value


def _build_basic_session(auth_value: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Basic {auth_value}", "X-CSRF-Header": "."}
    )
    return session


def _build_oauth_session(host: str, client_id: str, client_secret: str) -> requests.Session:
    if not client_id or not client_secret:
        raise AuthError(
            "relativity.auth.oauth_client_id and oauth_client_secret are required for method=oauth."
        )
    url = f"https://{host.rstrip('/')}/Relativity/Identity/connect/token"
    response = requests.post(
        url,
        headers={"content-type": "application/x-www-form-urlencoded"},
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "SystemUserInfo",
            "grant_type": "client_credentials",
        },
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise AuthError("Relativity OAuth token response missing access_token.")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "X-CSRF-Header": "."})
    return session


def _build_cid_session(config: Config) -> requests.Session:
    cid = config.cid
    rel = config.relativity
    if not rel.tenant_id:
        raise AuthError("relativity.tenant_id is required for method=cid.")
    response = requests.post(
        cid.sts_url,
        headers={"content-type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": cid.grant_type,
            "scope": cid.scope,
            "client_id": cid.client_id,
            "client_secret": cid.client_secret,
            "environment": rel.tenant_id,
        },
        timeout=60,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise AuthError("CID token response missing access_token.")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "X-CSRF-Header": "."})
    return session


def get_authenticated_session(config: Config) -> RelativityAuthResult:
    """Create a requests.Session for RelativityOne Object Manager."""

    method = config.relativity.auth.method
    if method == "basic":
        auth_value = _ensure_basic_auth(config.relativity.auth.basic_auth)
        return RelativityAuthResult(_build_basic_session(auth_value), method)
    if method == "oauth":
        session = _build_oauth_session(
            config.relativity.host,
            config.relativity.auth.oauth_client_id,
            config.relativity.auth.oauth_client_secret,
        )
        return RelativityAuthResult(session, method)
    if method == "cid":
        return RelativityAuthResult(_build_cid_session(config), method)
    raise AuthError(f"Unknown relativity.auth.method '{method}'.")


def encode_basic_auth(username: str, password: str) -> str:
    """Encode a username/password pair for config.toml."""

    raw = f"{username}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("utf-8")
