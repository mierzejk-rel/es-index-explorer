"""Elasticsearch client helpers."""

from collections.abc import Callable

from elasticsearch import AuthenticationException, Elasticsearch

from es_index_explorer.auth import get_token
from es_index_explorer.config import Config

def get_client(config: Config) -> Elasticsearch:
    """Create an Elasticsearch client using a CID v1 token."""

    token = get_token(config)
    return Elasticsearch(config.elasticsearch.hosts, bearer_auth=token)


def with_auth_retry(
    config: Config,
    func: Callable[[Elasticsearch], object],
) -> object:
    """Retry once with a fresh token on 401 responses."""

    try:
        return func(get_client(config))
    except AuthenticationException:
        token = get_token(config, force_refresh=True)
        client = Elasticsearch(config.elasticsearch.hosts, bearer_auth=token)
        return func(client)
