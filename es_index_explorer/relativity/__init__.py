"""RelativityOne integration helpers."""

from .auth import AuthError, get_authenticated_session
from .client import RelativityClient
from .models import RelativityDocument
from .normalize import DocumentReadError

__all__ = [
    "AuthError",
    "DocumentReadError",
    "RelativityClient",
    "RelativityDocument",
    "get_authenticated_session",
]
