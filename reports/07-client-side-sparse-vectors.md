# 07 - Client-Side Sparse Vectors (migrating off `semantic_text`/ELSER)

## Overview

This report specifies how to replace the three ELSER-backed `semantic_text` metadata fields
(`title_semantic`, `summary_semantic`, `topic_semantic`) with plain `sparse_vector` fields whose
token-weight maps are computed by us on the client, at both index time and query time. It is a
complete, self-contained instruction set: theory and rationale, the encoder, the Elasticsearch
mapping and index-build changes, ingest-time and query-time behavior, the concrete source-code
changes, and a short note on the long-input options we considered.

It builds on (and does not modify) [05-index-structure-design.md](05-index-structure-design.md) and
[06-document-indexing-and-semantic-chunking.md](06-document-indexing-and-semantic-chunking.md). Where
those reports describe the ELSER/`semantic_text` design, this report supersedes only the sparse parent
fields; the dense `chunks.embedding` path, BM25 `text` fields, and everything else are unchanged.

### Why this change

Our Elasticsearch cluster exposes the `.elser-2-elasticsearch` inference endpoint id, but the model is
not deployable (no ML nodes; EIS not connected). Every `semantic_text` ingest therefore fails with an
`inference_exception` whose root cause is `No ML nodes exist in the cluster` (or, on a cold endpoint,
`Could not find trained model`). Because `semantic_text` delegates inference to that endpoint at both
index and query time, no model choice (sparse or dense) can work while the cluster has no inference
engine. We already compute dense chunk vectors on the client (e5) and index them as a plain
`dense_vector`; this report applies the exact same principle to the sparse parent fields: compute the
sparse representation ourselves and index it into a plain `sparse_vector` field, removing the cluster's
inference dependency entirely.

### Scope

- In scope: the three parent metadata fields only (`title`, `summary`, `topic`).
- Unchanged: dense `chunks.embedding` (client-side e5, see report 06), BM25 on the `text` fields,
  derived parent fields, identifiers, retrieval shapes other than the sparse leg.
- Rollout: a new branch (`sparse_vectors`) and new indexes only. Existing `semantic_text` indexes are
  deleted and recreated; there is no in-place mapping migration.

## Background and grounding

ELSER and `semantic_text` implement learned-sparse retrieval: a transformer expands text into a sparse
map of vocabulary-token -> weight, stored in an inverted-index-friendly structure and scored with a
sparse dot product. ELSER belongs to the SPLADE family (BERT WordPiece vocabulary, term expansion,
~512-token input limit). `semantic_text` is a convenience wrapper that, server-side, tokenizes, chunks,
runs the inference endpoint, stores per-chunk sparse vectors as nested objects, and scores a document as
the MAX over its chunks (see 05 §4.5). All of that requires a running inference model in the cluster.

The client-side equivalent removes the wrapper: we run an open SPLADE-family encoder ourselves, produce
the `{token: weight}` map, and index it into a `sparse_vector` field. Elasticsearch then provides the
field type, the `sparse_vector` query, and token pruning - none of which need an inference endpoint.

## Requirements

- R1. Compute the sparse representation on the client, at index time and at query time; no reliance on
  any Elasticsearch inference endpoint.
- R2. Stay close to the ELSER/`semantic_text` paradigm (learned sparse, BERT WordPiece, token pruning),
  acknowledging that no self-hostable model reproduces ELSER's exact weights/scores.
- R3. Use one `sparse_vector` per field. We assume each field's text fits the encoder's token window
  (verified: the longest current summary is 212 tokens vs a 512 limit; `title`/`topic` are shorter). We
  deliberately do not chunk or nest these fields (see §8 for the alternative).
- R4. If any of the three fields exceeds the encoder's token window for a document, that document must
  FAIL, with the failure reported and logged like any other ingest failure.
- R5. Keep the sparse vectors out of `_source` (as we do for `chunks.embedding`).
- R6. Keep `prune: true` and the existing `pruning_config`.
- R7. Make the encoder model and all limits/parameters configurable (we may swap models later); nothing
  hardcoded.
- R8. CPU-only runtime, reusing the existing Hugging Face / PyTorch stack.

## 1. Encoder

- Model: `opensearch-project/opensearch-neural-sparse-encoding-v2-distill` (Apache-2.0). Learned-sparse,
  English, BERT WordPiece vocabulary (30522 dimensions), symmetric (the same encoder is used for both
  documents and queries, like ELSER), ~512-token input limit. It is the closest practical analog to
  ELSER's paradigm among openly self-hostable models; it will not reproduce ELSER's exact scores.
- Runtime: PyTorch CPU via `sentence-transformers` `SparseEncoder`, loaded lazily and mirroring the
  `E5Embedder` pattern in [es_index_explorer/indexing/embedding.py](../es_index_explorer/indexing/embedding.py)
  (lazy import of `sentence_transformers`/`torch`, `device` from config, offline flag honored).
- Tokenizer: the model's built-in WordPiece tokenizer (no separate component). The usable token budget
  is derived from the model at runtime - `max_seq_length` minus the special-token allowance - exactly as
  `E5Embedder.max_content_tokens()` does; it is not hardcoded, and an override is available via config.
- Output: a `dict[str, float]` of token -> weight, ready to index into a `sparse_vector` field. The
  exact `SparseEncoder` call used to obtain the human-readable token-weight map (rather than a raw sparse
  tensor) must be pinned during implementation (see Open Items).
- Symmetry matters here: because we no longer chunk these fields, ingest-time and query-time encoding are
  the same single call, which keeps the query path trivial.

## 2. Index schema (Elasticsearch build configuration)

Replace each `semantic_text` field with a `sparse_vector` field, keep the pruning configuration, drop the
`copy_to` from the text fields and the `semantic_text`-only `chunking_settings`, and exclude the new
fields from `_source`. Recommended rename: `*_semantic` -> `*_sparse` (the fields are no longer
`semantic_text`); confirm at review.

Mapping delta to [es_index_explorer/index_definitions/air_assist_nested.json](../es_index_explorer/index_definitions/air_assist_nested.json):

```json
{
  "mappings": {
    "_source": {
      "enabled": true,
      "excludes": [
        "chunks.embedding",
        "title_sparse",
        "summary_sparse",
        "topic_sparse"
      ]
    },
    "properties": {
      "title":   { "type": "text" },
      "summary": { "type": "text" },
      "topic":   { "type": "text" },

      "title_sparse": {
        "type": "sparse_vector",
        "index_options": {
          "prune": true,
          "pruning_config": { "tokens_freq_ratio_threshold": 5, "tokens_weight_threshold": 0.4 }
        }
      },
      "summary_sparse": {
        "type": "sparse_vector",
        "index_options": {
          "prune": true,
          "pruning_config": { "tokens_freq_ratio_threshold": 5, "tokens_weight_threshold": 0.4 }
        }
      },
      "topic_sparse": {
        "type": "sparse_vector",
        "index_options": {
          "prune": true,
          "pruning_config": { "tokens_freq_ratio_threshold": 5, "tokens_weight_threshold": 0.4 }
        }
      }
    }
  }
}
```

Notes:

- `copy_to` is removed from `title`/`summary`/`topic`; the client now writes the sparse fields directly.
  The `text` fields remain for BM25 (05 §7.1) unchanged.
- `chunking_settings` was a `semantic_text` parameter and has no meaning on a `sparse_vector` field.
- Excluding the sparse fields from `_source` does not affect search: a `sparse_vector` field is indexed
  independently of `_source`, so it remains fully queryable; only the raw map is omitted from retrieved
  documents (same rationale as `chunks.embedding`).
- The pruning thresholds `5` / `0.4` are ELSER v2-calibrated. They remain valid defaults but should be
  re-validated for this encoder (its weight/frequency distribution differs). Keep them configurable.

### Index-build code

Index creation is declarative, so the only build-code change beyond the JSON is in
[es_index_explorer/index_setup.py](../es_index_explorer/index_setup.py): drop (or make conditional) the
ELSER pre-check `ensure_inference_endpoint`, since the index no longer depends on an inference endpoint.
The diff/verify logic already handles `sparse_vector` + `index_options` and tolerates the extra keys ES
echoes back, so no change is needed there.

## 3. Ingestion (code changes: populate the index)

Mirror the dense path. The metadata fields are short, so there is no chunker for them - one encode call
per field.

- New `SparseEmbedder` (sibling of `E5Embedder` in
  [es_index_explorer/indexing/embedding.py](../es_index_explorer/indexing/embedding.py)):
  - Lazily loads the `SparseEncoder` (CPU, `device` from config, offline-aware).
  - `token_budget()` derived from the model (`max_seq_length` minus special tokens), with an optional
    config override.
  - `count_tokens(text) -> int`.
  - `encode(text) -> dict[str, float]` returning the token -> weight map.
- Over-limit handling (R4): before encoding, compare `count_tokens(text)` against `token_budget()`. If a
  field exceeds it, raise a dedicated `SparseInputTooLongError` (carrying the field name and the token
  counts). The pipeline's `_build_index_document` already wraps document building in a try/except that
  records any exception as `outcome="index_failed"`, `stage="index"`,
  `error_type=type(exc).__name__`, `error_message=str(exc)` (see
  [es_index_explorer/indexing/pipeline.py](../es_index_explorer/indexing/pipeline.py) lines ~71-84). So
  the failure is logged through the existing progress/resume path with `error_type="SparseInputTooLongError"`;
  no new logging mechanism is needed.
- `IndexDocument` + `MAPPING` ([es_index_explorer/indexing/document_builder.py](../es_index_explorer/indexing/document_builder.py)):
  add precomputed `title_sparse`/`summary_sparse`/`topic_sparse` maps to `IndexDocument` and matching
  `MAPPING` extractors. Set a field's sparse map only when its source text is non-empty; `build_source`
  already drops `None`, so empty metadata yields no sparse field (and no wasted encode call).
- `IndexingPipeline` ([es_index_explorer/indexing/pipeline.py](../es_index_explorer/indexing/pipeline.py)):
  construct the `SparseEmbedder` in `__init__` (indexing runs only), and in `_build_index_document`
  encode the present metadata fields into maps (or raise on overflow) and attach them to `IndexDocument`.
- `_source` exclusion is handled entirely by the mapping; no code involved.

## 4. Query time

At query time we encode the user's query string with the same `SparseEmbedder` (symmetric model -> the
identical call used at ingest) to obtain a query `{token: weight}` map, then run a `sparse_vector` query
that supplies that map directly via `query_vector` (no inference endpoint, no `semantic` query):

```json
{
  "query": {
    "sparse_vector": {
      "field": "summary_sparse",
      "query_vector": { "<token>": 1.23, "...": 0.0 }
    }
  }
}
```

Repeat / combine across `title_sparse`, `summary_sparse`, `topic_sparse`, and fuse with BM25 (`match` on
the `text` fields) and dense kNN on `chunks.embedding` via an `rrf` or linear retriever, exactly as the
ELSER `semantic` leg was fused before (see 05 §7.3 and §7.4 - the only change is `semantic` ->
`sparse_vector` with a client-provided `query_vector`). Query-side token pruning can be applied via the
`sparse_vector` query's `prune`/`pruning_config` if desired.

Query input over the encoder window: a query string can also exceed the encoder's token window. We have
at least two options, and the choice should be configurable:

- Trim the end so the input fits the window (the behavior `semantic_text` / the default ELSER endpoint
  use today). More forgiving - a long query still returns results.
- Fail the query with a clear error. Stricter and surfaces the problem, but rejects the user's request.

Recommendation: for queries, prefer trimming (resilience; it matches the prior `semantic_text`
behavior), while ingestion fails on overflow (R4) so indexed data is never silently truncated. Make the
query-time policy a setting so it can be flipped per deployment.

Scope: es-index-explorer is an indexer/inspector and has no search module. This report fully specifies
the query-time encoding and DSL, and the `SparseEmbedder` is reusable for query encoding, but executing
the search lives in the retrieval service that owns query handling.

## 5. Configuration

Add to `IndexingConfig` in [es_index_explorer/config.py](../es_index_explorer/config.py), mirroring the
existing dense settings and defaulting sensibly:

- `sparse_model: str = "opensearch-project/opensearch-neural-sparse-encoding-v2-distill"`
- `sparse_model_path: str = ""` (local dir; empty = hub id)
- `sparse_device: str = ""` (empty = reuse `device`)
- `sparse_max_tokens: int | None = None` (None = derive from the model)
- query-side controls (used by the retrieval service): `sparse_query_overflow: Literal["trim", "fail"] = "trim"`,
  and optional query pruning toggles.

The `offline` flag and CPU `device` are reused. The ingest-time overflow policy is fixed to "fail" per
R4 (only the query-time policy is configurable).

## 6. Dependencies

- `sentence-transformers` must be a version that ships `SparseEncoder` (sparse embeddings landed in
  `sentence-transformers` 5.x); pin and verify the minimum during implementation. `torch` (CPU) is
  already present via the pinned CPU index.
- The encoder model is downloaded from Hugging Face (Apache-2.0); honor `HF_HUB_OFFLINE` as `E5Embedder`
  does. No new heavy dependencies.

## 7. Migration and rollout

- Work on the `sparse_vectors` branch.
- New indexes only: delete the old `semantic_text` indexes, recreate from the updated definition with
  `setup_index.py`, then re-ingest with `ingest.py --index`. There is no in-place mapping migration.
- Remove the ELSER inference pre-check from `setup_index.py` (§2).

## 8. Long-input options (what we chose, and the alternative)

The metadata fields are short and we verified they fit the encoder window, so we chose the simplest
design: one `sparse_vector` per field, no chunking, and a hard FAIL (logged) if any field exceeds the
window (R3, R4). This avoids nested mappings and nested queries entirely.

If a future workspace has metadata that does not fit the window and we want to keep it (rather than
fail), the alternative is to chunk and nest, replicating `semantic_text`'s internal behavior. That would
change:

- Index design: each field becomes a `nested` object containing a `sparse_vector` (keep `prune`/
  `pruning_config`); exclude the nested vector from `_source`.
- Index-build code: only the declarative JSON in `air_assist_nested.json` changes; `index_setup.py`
  needs no logic change (its diff already supports nested + `sparse_vector`).
- Ingestion: reuse/extend the existing `SemanticChunker`
  ([es_index_explorer/indexing/chunking.py](../es_index_explorer/indexing/chunking.py)) with a sparse
  profile - sentence-first with one-sentence overlap, fill to the encoder budget, clause/word fallbacks
  for an over-long sentence, and end-trim as the last resort (mirroring report 06's ladder and the ELSER
  default). Encode each chunk and store a list of maps in the nested field.
- Query: wrap the `sparse_vector` query in a `nested` query with `score_mode: max` (per field) so the
  document scores as the MAX over its chunks, then fuse as in §4. (Plain `sparse_vector` fields do not
  provide multi-chunk MAX automatically; the nested wrapper is what enables it.)

This path is documented for the future and intentionally not implemented now.

## 9. Open items for the implementer

- Confirm the field rename `*_semantic` -> `*_sparse` (or keep the old names).
- Pin the `SparseEncoder` API call that yields a `{token: weight}` dict (vs a raw sparse tensor).
- Pin the minimum `sentence-transformers` version and add/verify it in `pyproject.toml`.
- Re-validate `prune`/`pruning_config` thresholds for this encoder (the `5`/`0.4` values are ELSER-tuned).
- Decide where query execution lives and wire the query-time overflow policy (`trim`/`fail`).

## Sources

- OpenSearch neural sparse v2-distill model card and Apache-2.0 license:
  https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v2-distill
- `sparse_vector` field type: https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/sparse-vector
- `sparse_vector` query (with `query_vector` and pruning): https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-sparse-vector-query
- `sentence-transformers` SparseEncoder: https://www.sbert.net/docs/sparse_encoder/usage/usage.html
- SPLADE (learned sparse retrieval) background: https://github.com/naver/splade
- Cross-references: [05-index-structure-design.md](05-index-structure-design.md) §4.5, §7.1, §7.3, §7.4;
  [06-document-indexing-and-semantic-chunking.md](06-document-indexing-and-semantic-chunking.md) for the
  chunker reused in the §8 alternative.
