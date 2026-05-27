"""Minimal HTTP transport wrapper for Relativity APIs."""

import requests


class Transport:
    """Thin wrapper around a configured requests.Session."""

    def __init__(self, base_url: str, session: requests.Session) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.session.headers.setdefault("X-CSRF-Header", "-")

    def post(self, path: str, json: dict[str, object]) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.post(url, json=json)

    def get(self, path: str) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.get(url)

    def delete(self, path: str) -> requests.Response:
        url = f"{self.base_url}{path}"
        return self.session.delete(url)
