"""RelativityOne integration helpers."""

from .auth import AuthError, get_authenticated_session
from .client import RelativityClient
from .models import RelativityDocument
from .normalize import DocumentReadError
from .reader import read_documents

__all__ = [
    "AuthError",
    "DocumentReadError",
    "RelativityClient",
    "RelativityDocument",
    "get_authenticated_session",
    "read_documents",
]
