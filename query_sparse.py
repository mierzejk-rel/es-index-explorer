"""Query similar documents using the summary sparse vector field."""

from pathlib import Path
from typing import Any, cast

from es_index_explorer.client import with_auth_retry
from es_index_explorer.config import load_config
from es_index_explorer.indexing.embedding import SparseEmbedder, configure_hf_offline

# User-editable variables.
QUERY_TEXT = "Exalgo presence in press articles"
INDEX_NAME = "as-mierzej-mlcdt-a4r-v01"  # Leave empty to use elasticsearch.index_name from config.toml.
TOP_K = 25
CONFIG_PATH: str | Path | None = None  # None = ./config.toml


def _search(
    *,
    config_path: str | Path | None,
    query_text: str,
    index_name_override: str,
    top_k: int,
) -> dict[str, Any]:
    """Encode query text and run a sparse-vector search on summary_sparse."""

    if not query_text.strip():
        raise ValueError("QUERY_TEXT must be non-empty.")
    if top_k <= 0:
        raise ValueError("TOP_K must be a positive integer.")

    config = load_config(config_path)
    index_name = (index_name_override or config.elasticsearch.index_name).strip()
    if not index_name:
        raise ValueError("Set INDEX_NAME or elasticsearch.index_name in config.toml.")

    configure_hf_offline(config.indexing)
    embedder = SparseEmbedder(config.indexing)
    query_vector = embedder.encode(query_text, field_name="query")
    body: dict[str, Any] = {
        "size": top_k,
        "_source": ["document_artifact_id", "summary"],
        "query": {
            "sparse_vector": {
                "field": "summary_sparse",
                "query_vector": query_vector,
            }
        },
    }

    def _run(client: Any) -> Any:
        return client.search(index=index_name, body=body)

    response = with_auth_retry(config, _run)
    return cast(dict[str, Any], response)


def main() -> None:
    """Run the sparse summary query and print artifact IDs plus summaries."""

    response = _search(
        config_path=CONFIG_PATH,
        query_text=QUERY_TEXT,
        index_name_override=INDEX_NAME,
        top_k=TOP_K,
    )
    hits = response.get("hits", {}).get("hits", [])
    if not isinstance(hits, list) or not hits:
        print("No results found.")
        return

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        source = hit.get("_source", {})
        score = hit.get("_score")
        if not isinstance(source, dict):
            source = {}
        artifact_id = source.get("document_artifact_id", "(unknown)")
        summary = source.get("summary", "(no summary)")
        score_label = f"{float(score):.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"[score={score_label}] artifact_id={artifact_id}")
        print(f"  summary: {summary}")
        print()


if __name__ == "__main__":
    main()
