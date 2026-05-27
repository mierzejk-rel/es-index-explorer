"""RelativityOne client façade."""

import requests

from .fluent import QueryBuilder
from .object_manager import ObjectManagerAPI
from .transport import Transport


# noinspection HttpUrlsUsage
class RelativityClient:
    """Compose Relativity Object Manager APIs with a session."""

    def __init__(self, host: str, workspace_id: int, session: requests.Session) -> None:
        base = host.rstrip("/")
        if base.startswith("http://") or base.startswith("https://"):
            base_url = base
        else:
            base_url = f"https://{base}"
        self.transport = Transport(base_url, session)
        self.workspace_id = workspace_id
        self.object_manager = ObjectManagerAPI(self.transport, workspace_id)

    def query_object_manager(self) -> QueryBuilder:
        return QueryBuilder(self.object_manager)
