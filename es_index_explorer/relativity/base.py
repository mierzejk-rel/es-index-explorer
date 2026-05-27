"""Shared API client utilities."""

from .transport import Transport


class BaseAPIClient:
    """Base class for Relativity API clients."""

    def __init__(self, transport: Transport, workspace_id: int) -> None:
        self._t = transport
        self._ws = workspace_id
