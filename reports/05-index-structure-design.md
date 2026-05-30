# Custom aiR Assist Elasticsearch Index: Two-Level Parent/Child Design

**Purpose:** Design reference for engineers and AI agents building a new, experiment-ready
Elasticsearch index for the aiR Assist RAG application. It specifies a two-level
parent (document) / child (chunk) structure, compares relationship-modeling patterns and ranks
them against our requirements, defines a fully-explicit mapping, and proves the design is not
inferior to the current production index for any data that can be stored, searched, or retrieved —
while unlocking additional retrieval strategies (dense, sparse/ELSER, hybrid/RRF, reranking).

**Target stack (confirmed):**
- Elasticsearch **9.2**, **Platinum/Enterprise** license.
- Sparse retrieval via **ELSER** deployed in-cluster (`sparse_vector` + `sparse_vector` query).
- Dense chunk embeddings via **`intfloat/multilingual-e5-small`** (384 dims, cosine) — same model as production.
- New index lives under the **`as-*`** prefix (Applied Science / research grant), per
  `01-air-assist-elasticsearch-index.md` §1.

**Reads from:** `00`–`04` reports in this folder and the `es-index-explorer` codebase. Cross-references
to current behavior cite `01-air-assist-elasticsearch-index.md` (current mapping),
`02-index-population-pipeline.md` (ingestion), and `03-retrieval-strategies.md` (search).

**Scope:** Index *structure and configuration* only. Chunking algorithm, ingestion batching, and the
client/setup code are out of scope here and will be designed separately. Decisions deliberately left
open for implementation time are collected in §10.

---

## 1. Requirements and Constraints (restated)

The design is evaluated against these explicit requirements:

| # | Requirement | Source |
|---|---|---|
| R1 | Two-level hierarchy: one parent (document) has 0..N children (chunks); each child has exactly one parent; children have no children. | User |
| R2 | Document-level (global/categorical) attributes live at the parent: title, summary, topic, email from/to/cc/bcc, primary date, derived metrics. | User |
| R3 | Evaluate highly specific child-level queries, enforce parent-level metadata filters, and reconstitute parent context — without performance degradation or index fragmentation. | User |
| R4 | Children are immutable after write; parents are very unlikely to change. | User |
| R5 | Typically <10 children per parent, occasionally up to a few dozen. | User |
| R6 | Be ready for any chunking strategy (fixed window ± overlap, semantic, sentence-boundary) without changing the index structure. | User |
| R7 | Support, on a single index, the full matrix of retrieval strategies: lexical (BM25), dense (kNN), sparse (ELSER), hybrid (RRF/linear), and later reranking (MMR, cross-encoder, late-interaction, LLM). | User |
| R8 | Support both single-call ("do everything in ES, return everything") and multi-step (find docs → fetch chunks → re-query) retrieval. | User |
| R9 | Be ready to return either separate document/chunk objects (joined agent-side) or denormalized chunks-with-metadata. | User |
| R10 | Decide whether the first chunk is a parent field or a child element; avoid redundant data. | User |
| R11 | Not inferior to the current index: every field/search capability present today must be available here, plus more. | User |
| R12 | Derived field `byte_size` to exclude documents (and their chunks) whose content exceeds 5 MB, computed as faithfully as possible to today's behavior. | User |
| R13 | Index-setup code must set all important parameters explicitly — no reliance on cluster-template defaults (unlike today, where vector params come from an external template). | User |

---

## 2. Source Data Model and Parent/Child Semantics

### 2.1 What we read from RelativityOne today

The `es-index-explorer` reader produces `RelativityDocument`
([`es_index_explorer/relativity/models.py`](../es_index_explorer/relativity/models.py)):

| Python field | Type | Required | Role |
|---|---|---|---|
| `artifact_id` | `int` | yes | Relativity document ArtifactId — the stable document primary key |
| `control_number` | `str` | yes | Human-readable Relativity Control Number |
| `extracted_text` | `str` | yes | Full document text; **source of all chunks** |
| `primary_date_time` | `datetime \| None` | no | Date for range filtering |
| `email_from` | `str \| None` | no | Email sender |
| `email_to` | `list[str] \| None` | no | Email recipients (multi-valued) |
| `email_cc` | `list[str] \| None` | no | Email CC (multi-valued) |
| `email_bcc` | `list[str] \| None` | no | Email BCC (multi-valued) |
| `summary` | `str \| None` | no | Document-level summary |
| `topic` | `str \| None` | no | Document-level topic |

### 2.2 Parent vs child mapping of these fields

- **Parent (document)** carries all of the above *except* `extracted_text`, plus derived metrics
  (§6): `byte_size`, `token_count`, `chunk_count`, and the multi-tenancy `subset_ids`.
- **Child (chunk)** carries only what is derived from `extracted_text`: `chunk_index`, `text`,
  `embedding`, `token_count`. Per R6, the chunk schema is intentionally agnostic to *how* the text
  was split.

### 2.3 Fields explicitly in/out of scope

- **`title`** — kept. It exists in the current ES mapping and in qna-service's BM25 `multi_match`
  field list, although it is **never populated today** (`01-air-assist-elasticsearch-index.md` §4.1,
  §8). Its RelativityOne source field is not yet read by `RelativityDocument` (see §10).
- **`metadata.*` dynamic fields** — the production index supports admin-configured dynamic metadata
  under a `metadata` object (`01-...index.md` §4.2). For our experimental index we promote the
  *known* metadata (emails, date, topic, summary, title) to first-class typed parent fields. An
  optional dynamic `metadata` object can be added later if arbitrary per-workspace fields are needed;
  it is not required for parity with the inspected workspace, which had no `metadata.*` fields.

### 2.4 Identifiers

| Identifier | Where | Notes |
|---|---|---|
| `document_artifact_id` | parent | = Relativity ArtifactId (today's `documentId`). Stable, used for security trimming and joins. |
| `control_number` | parent | Display/citation. |
| `chunk_index` | child | 0-based sequential index within the document (today's `chunkId`). |
| ES `_id` (parent) | parent doc | Recommended: the document ArtifactId as a string. With nested chunks there is **one** ES document per Relativity document, so no `{docId}_{chunkId}` composite is needed (contrast today's flat model). |

---

## 3. Relationship-Modeling Patterns: Analysis and Ranking

Four viable patterns were considered. Each is assessed against R1–R13.

### 3.1 Pattern A — Nested objects (RECOMMENDED)

One ES document per Relativity document; chunks stored as a `nested` array (`chunks`). Parent
metadata is stored once at the top level.

**How it satisfies the hard requirements:**
- **Child-level queries + parent-level filters in one query (R3).** In ES 9.2, a `knn` query inside
  a `nested` query supports a `filter` over **both** top-level (parent) metadata and nested (chunk)
  metadata as pre-filters — this mix is **GA in 9.2** specifically. Nested kNN uses
  `score_mode=max` and `inner_hits` returns just the matching chunk(s).
- **Parent context reconstruction without fragmentation (R3).** Parent metadata lives once on the
  parent; chunks are stored in the same Lucene block. Returning the parent returns its context
  directly; no separate "first-chunk" round-trip and no metadata duplicated per chunk.
- **Lifecycle fit (R4, R5).** Nested's main cost — reindexing the whole parent when any child
  changes — is irrelevant here: children are immutable and parents rarely change. With <10 (max a
  few dozen) children, nested stays well within healthy limits (`index.mapping.nested_objects.limit`
  default 10,000).
- **Deletion/filtering semantics (R12).** Excluding a parent (e.g. `byte_size > 5MB`) excludes all
  its chunks automatically — they are the same document.

**Trade-offs / caveats (documented honestly):**
- Nested kNN/BM25 returns **k top-level documents** (each scored by its best passage via
  `inner_hits`), **not** a global ranking of individual chunks. For document-grouped RAG this is
  exactly what we want; for pure *chunk-centric* global ranking experiments, use the flat fallback
  (Pattern B) — see §8.
- Changing one chunk requires reindexing the parent document (acceptable per R4).
- Each nested chunk is internally a hidden Lucene document; this is invisible to queries but counts
  toward segment doc counts.

### 3.2 Pattern B — Flat denormalization (current production model; documented fallback)

One ES document per chunk; parent fields copied onto every chunk (today's model —
`02-index-population-pipeline.md` §5).

- **Pros:** global chunk-level ranking is natural; simplest RRF over chunks; scales trivially across
  shards; no nested/join overhead; this is the proven production shape (guarantees R11 parity).
- **Cons:** parent metadata duplicated on every chunk (storage + update amplification); reconstituting
  the parent and fetching the "first chunk" need extra queries or `collapse` (today qna-service issues
  a *separate* first-chunk query — `03-retrieval-strategies.md` §5.1); parent-level filters are
  evaluated per chunk.
- **Verdict:** Kept as a **documented secondary option** for chunk-centric experiments and as the
  guaranteed-parity baseline. Not the primary recommendation because it does not model R1/R2/R3 as
  cleanly and reintroduces the duplication R10 wants to avoid.

### 3.3 Pattern C — Separate parent and child indices (application-side join)

Parents in one index, chunks in another; join in the agent/client.

- **Pros:** clean separation; independent lifecycle; chunk index can be sharded independently.
- **Cons:** no single-query hybrid spanning both levels; every search needs ≥2 round-trips and a
  client-side join; filter logic duplicated across indices; cross-index consistency burden. This is a
  multi-step-only pattern and cannot satisfy R8's "do everything in one call" branch.
- **Verdict:** Acceptable only for specific multi-step experiments; not a general-purpose base.

### 3.4 Pattern D — Parent/child `join` field (NOT recommended)

A single index with a `join` field and `has_child`/`has_parent` queries.

- **Cons (well-documented by Elastic):** Elastic explicitly recommends denormalization/nested before
  `join`; `has_child`/`has_parent` are typically **5–10× slower** than nested; parent and all children
  **must live on the same shard** (forces routing, risks uneven/"hot" shards = fragmentation); the
  `join` field builds **global ordinals** rebuilt on refresh (latency spikes on writes); only **one
  `join` per index**; and — decisively for us — it does **not compose cleanly with `knn`/RRF
  retrievers**, which are central to R7.
- **Verdict:** Worst fit despite being the literal "parent-child" mechanism. Rejected.

### 3.5 Ranking summary

| Rank | Pattern | R3 child-query + parent-filter | Parent-context / no fragmentation | Hybrid (dense+sparse+BM25+RRF) in one query (R7) | Single-call (R8) | Update/lifecycle fit (R4/R5) | Scales to large corpora | Overall |
|---|---|---|---|---|---|---|---|---|
| **1** | **Nested (A)** | Excellent (9.2 GA mixed pre-filters) | Excellent (one doc, no dup) | Excellent (nested knn/BM25 + parent sparse via retrievers) | Excellent (`inner_hits`) | Excellent (immutable children) | Good (per-parent block; fine at <few dozen chunks) | **Recommended** |
| 2 | Flat (B) | Good (filters per chunk) | Weak (dup metadata; extra first-chunk query) | Good (chunk-only) | Good | Good | Excellent | Fallback / parity baseline |
| 3 | Separate indices (C) | Partial (two queries) | Manual join | No (multi round-trip) | No | Excellent | Excellent | Niche multi-step only |
| 4 | `join` field (D) | Poor (slow has_child) | OK | No (doesn't compose with knn/RRF) | Limited | OK (independent child updates we don't need) | Poor (single-shard constraint) | Rejected |

**Decision:** Adopt **nested objects (A)** as the primary design; keep **flat (B)** documented for
chunk-centric experiments and as the explicit R11 parity baseline.

---

## 4. Recommended Mapping (Nested)

Every parameter is set explicitly per R13. Index settings are included so nothing is inherited
silently from a cluster template (the gap called out in `01-...index.md` §6, §8).

```json
PUT as-air-assist-<workspace>-nested
{
  "settings": {
    "index": {
      "number_of_shards": 1,
      "number_of_replicas": 1,
      "refresh_interval": "1s",
      "mapping": {
        "nested_objects": { "limit": 10000 },
        "total_fields": { "limit": 2000 }
      }
    }
  },
  "mappings": {
    "_source": { "enabled": true },
    "dynamic": "strict",
    "properties": {
      "document_artifact_id": { "type": "long" },
      "control_number":       { "type": "keyword" },

      "title":   { "type": "text", "fields": { "kw": { "type": "keyword", "ignore_above": 1024 } } },
      "summary": { "type": "text" },
      "topic":   { "type": "text", "fields": { "kw": { "type": "keyword", "ignore_above": 1024 } } },

      "primary_date_time": { "type": "date" },
      "email_from": { "type": "keyword" },
      "email_to":   { "type": "keyword" },
      "email_cc":   { "type": "keyword" },
      "email_bcc":  { "type": "keyword" },

      "byte_size":   { "type": "long" },
      "token_count": { "type": "integer" },
      "chunk_count": { "type": "integer" },

      "subset_ids": { "type": "keyword" },

      "title_sparse":   { "type": "sparse_vector" },
      "summary_sparse": { "type": "sparse_vector" },
      "topic_sparse":   { "type": "sparse_vector" },

      "chunks": {
        "type": "nested",
        "properties": {
          "chunk_index": { "type": "integer" },
          "text":        { "type": "text" },
          "token_count": { "type": "integer" },
          "embedding": {
            "type": "dense_vector",
            "dims": 384,
            "index": true,
            "similarity": "cosine",
            "index_options": {
              "type": "bbq_hnsw",
              "m": 16,
              "ef_construction": 100,
              "rescore_vector": { "oversample": 3.0 }
            }
          }
        }
      }
    }
  }
}
```

### 4.1 Field-by-field rationale

| Field | Type | Why this type | Search/retrieval role |
|---|---|---|---|
| `document_artifact_id` | `long` | Relativity ArtifactId; numeric. | Exact filter, security-trim join key, document identity. |
| `control_number` | `keyword` | Exact, non-tokenized. | Display/citation, exact filter, sort/agg via doc_values. |
| `title` | `text` + `.kw` keyword | BM25 on text; keyword sub-field for exact/agg. | BM25 (R7 lexical-on-title), exact filter. Unpopulated today (§10). |
| `summary` | `text` | BM25 full-text. | BM25 (R7 lexical-on-summary). |
| `topic` | `text` + `.kw` keyword | BM25 + exact/facet. | BM25 (R7 lexical-on-topic), facet. |
| `primary_date_time` | `date` | Range queries. | Date range filter (parity with `metadata.primaryDateTime`). |
| `email_from/to/cc/bcc` | `keyword` (multi-valued) | Exact + wildcard per element. | Email participant filters (parity with `metadata.email*`). |
| `byte_size` | `long` | Byte count. | 5 MB include/exclude (R12; §6). |
| `token_count` | `integer` | Token count of full doc. | Analytics, filtering, cost estimation (§6). |
| `chunk_count` | `integer` | Number of chunks. | Cross-index filter/sort/agg convenience (§6). |
| `subset_ids` | `keyword` (multi-valued) | Multi-tenancy scoping. | `term` filter parity with production's always-on subset filter (`03-...md` §4.2). |
| `title_sparse` / `summary_sparse` / `topic_sparse` | `sparse_vector` | Stores ELSER token-weight pairs. | Sparse (ELSER) retrieval on parent metadata (R7 sparse). |
| `chunks` | `nested` | Preserves chunk independence for nested kNN/BM25 with parent pre-filters. | Container for child chunks (R1). |
| `chunks.chunk_index` | `integer` | 0-based order. | First-chunk selection (`== 0`), ordering within doc. |
| `chunks.text` | `text` | BM25 on chunk text. | Lexical chunk retrieval (R7 BM25-on-chunks). |
| `chunks.token_count` | `integer` | Per-chunk tokens. | Diagnostics; not summable to document tokens (§6). |
| `chunks.embedding` | `dense_vector` (384, cosine, `bbq_hnsw`) | Dense ANN per chunk. | Dense/kNN chunk retrieval (R7 dense-on-chunks). |

### 4.2 Parity mapping to the current (flat) index

Confirms R11 — nothing today is lost:

| Current field (`01-...md` §3) | New location | Notes |
|---|---|---|
| `body` (chunk text) | `chunks.text` | Same role; now nested. |
| `embedding` (dense_vector 384/cosine/bbq_hnsw) | `chunks.embedding` | Identical vector config, now nested + **explicitly set** (not template-derived). |
| `documentId` | `document_artifact_id` | Renamed; same value. |
| `chunkId` | `chunks.chunk_index` | Renamed; same semantics. |
| `chunkSize` (UTF-8×2 per chunk) | superseded by `byte_size` (doc-level, §6) + optional `chunks.token_count` | The per-chunk byte stat had no retrieval role; replaced by a faithful doc-level size. |
| `controlNumber` | `control_number` | Same. |
| `subsetIds` | `subset_ids` | Same multi-tenancy role. |
| `title` | `title` | Same (still optional/unpopulated; §10). |
| `documentModifyTime` | (optional add) | Not in `RelativityDocument` today; add as parent `date` if a source is wired. |
| `createdAt` | dropped | Never populated in production (`01-...md` §8). Add only if needed. |
| `metadata.primaryDateTime` | `primary_date_time` | Promoted to first-class typed field. |
| `metadata.emailFrom/To/Cc/Bcc` | `email_from/to/cc/bcc` | Promoted to first-class. |
| `metadata.documentName` | `title` or optional `document_name` | Map to whichever R1 field is chosen (§10). |

### 4.3 Vector index choice for `chunks.embedding`

Default `bbq_hnsw` matches production exactly (`01-...md` §5.2) for behavioral parity and is set
explicitly here. Two documented alternatives:

| Option | When to choose | Notes (ES 9.2) |
|---|---|---|
| `bbq_hnsw` (default) | Parity with production; in-memory HNSW graph, strong recall/latency. | Binary-quantized HNSW + automatic oversampling/rescore. `rescore_vector.oversample: 3.0` mirrors production. |
| `bbq_disk` (DiskBBQ) | Large corpora (hundreds of thousands of docs → millions of chunk vectors) where RAM is the constraint. | New/GA in 9.2. Reads compact quantized clusters from disk; far lower RAM, ~sub-20 ms latency. Tunable via `num_candidates` / `visit_percentage` / `bits`. |
| `int8_hnsw` | If BBQ recall ever proves insufficient and memory allows. | 9.0 default scalar quantization. |

All three keep `dims: 384`, `similarity: cosine`. Switching requires a reindex (vector
`index_options` is fixed at field creation).

### 4.4 ES 9.2 `_source` behavior for dense vectors

In ES 9.2, newly created indices **exclude `dense_vector` fields from `_source` by default** (storage
and indexing-throughput win). We do not need raw chunk vectors echoed in results, so the default is
desirable. If a future strategy needs vectors returned (e.g. client-side MaxSim), override via
explicit `_source` includes or synthetic source. This is documented so the behavior is intentional,
not surprising.

### 4.5 ELSER inference for the `sparse_vector` fields

`title_sparse`/`summary_sparse`/`topic_sparse` store ELSER output (token-weight pairs). Two ways to
populate them:
- **Ingest/inference pipeline** in ES (ELSER inference endpoint) writes the sparse vector, or
- **Precompute in Python** and index the token-weight map directly into the `sparse_vector` field.

At query time, the `sparse_vector` query expands the query text with the same ELSER `inference_id`
(ELSER may be served via a standard ML deployment or, in 9.2, via **ELSER on EIS**). The choice of
deployment does not change the mapping. (If desired later, these three could instead be modeled as
`semantic_text` fields so ES manages inference end-to-end — see §10.)

---

## 5. First-Chunk Handling

**Confirmed correction to a common misconception:** today the first chunk is **stored exactly once** —
chunk 0 is an ordinary chunk document like any other. The "special" behavior is purely at the
**MCP/response level**: qna-service issues a *separate* query (`GetFirstChunksForDocumentsAsync`) to
fetch chunk 0 and attaches it as `FirstChunk`, which is what produces the *response-level* duplication
described in `03-retrieval-strategies.md` §5.1 (the same text can appear under both `first_chunk` and
`retrieved_chunks`). There is **no storage redundancy** today.

**In the nested design:**
- **Default (no redundancy, recommended):** the first chunk is simply the nested element with
  `chunk_index == 0`. Because chunks live inside the parent document, chunk 0 is reachable directly
  from the parent `_source` (or via an `inner_hits`/nested query) with **no separate ES round-trip**.
  The special first-chunk query disappears entirely. Whether to also surface chunk 0 as "context" in
  the response becomes a pure agent/MCP formatting choice — not a storage concern, and easy to
  de-duplicate against the retrieved chunks.
- **Optional, opt-in (explicitly redundant — not in the default mapping):** a denormalized
  `first_chunk_text` copy on the parent, for ultra-cheap context retrieval without touching nested
  data. Because R10 prioritizes avoiding redundant data, this is **excluded by default**; it is
  offered only as a documented optimization with the trade-off stated plainly (it duplicates chunk
  0's text on the parent).

**Recommendation:** Default to "first chunk = `chunks[chunk_index == 0]`", no copy. This satisfies
R10 ("store either in the document or as a separate object, not both") — here it is stored once,
inside the document, as a normal chunk.

---

## 6. Derived Parent-Level Fields

### 6.1 `byte_size` — the 5 MB content filter (R12)

**Goal:** replicate production's `TooLarge` exclusion as closely as possible. That filter keys off the
**raw ADLS file byte size** (`FileSize > 5,242,880` bytes — `02-...md` §2.2, §8), *not* the per-chunk
`chunkSize = UTF8.GetByteCount(chunk) * 2` statistic (which has no retrieval role and over-estimates
non-ASCII by 2–3×).

Since `es-index-explorer` reads `extracted_text` directly (no ADLS file handle), the closest faithful
reproduction of the raw file size is the **actual UTF-8 byte length** of the text:

```python
byte_size = len(extracted_text.encode("utf-8"))
EXCLUDE if byte_size > 5_242_880   # 5 * 1024 * 1024
```

- **Why UTF-8 actual:** text files on ADLS are, by the .NET default and common practice, UTF-8;
  `len(encode("utf-8"))` matches the bytes-on-disk that the production filter measures.
- **Why not `UTF-8 × 2`:** that is only a UTF-16 approximation (equal to true UTF-16 just for ASCII)
  and over-estimates any non-ASCII character — it would wrongly exclude documents.
- **Note (alternative interpretation):** if one instead wanted the .NET *in-memory* string size, that
  is UTF-16 actual = `len(text.encode("utf-16-le"))` (= .NET `string.Length × 2`). We deliberately do
  **not** use this as the default, because the production 5 MB filter is about the file, not the
  in-memory string.
- **Residual uncertainty:** if ADLS files were ever UTF-16-encoded on disk, the true size would differ
  (~2× for ASCII); UTF-8-actual remains the best approximation obtainable from text alone. BOM and
  encoding deltas are negligible against a 5 MB threshold.

**Enforcement is two-layer:**
1. **Ingest-time:** skip oversized documents entirely (they and their chunks never enter the index).
2. **Query-time (defensive):** every retrieval may add `"range": { "byte_size": { "lte": 5242880 } }`
   so even if an oversized document slipped in, it (and its nested chunks) is excluded. Because chunks
   are nested in the parent, excluding the parent excludes its chunks automatically.

### 6.2 `token_count` — full-document tokens (recommended)

Token count of the **entire** `extracted_text` *before* chunking. This is **genuinely non-derivable
from the chunks**: chunking with overlap double-counts boundary tokens, so summing
`chunks[].token_count` over-counts. It is cheap (we already tokenize during chunking) and useful for
analytics, filtering, and embedding/inference cost estimation. **Include it.**

### 6.3 `chunk_count` — number of chunks (recommended)

A single integer derived at index time. For a single returned document it is also available as
`len(_source.chunks)`, but a stored field enables efficient **cross-index** filter/sort/aggregation
(e.g. "documents with 0 chunks", "documents with > N chunks") without nested aggregations. It is a
scalar, not duplicated text, so it does not constitute problematic redundancy. **Include it.**

---

## 7. Retrieval Strategies (Parity + More)

All query shapes below target ES 9.2 and run against the single nested index. They prove R7 and R11.

### 7.1 BM25 — lexical

**On chunk text (nested):**
```json
{
  "query": {
    "nested": {
      "path": "chunks",
      "query": { "match": { "chunks.text": "<query>" } },
      "score_mode": "max",
      "inner_hits": { "size": 5, "name": "matched_chunks" }
    }
  }
}
```

**On parent metadata (title/topic/summary):**
```json
{ "query": { "multi_match": { "query": "<query>", "fields": ["title", "topic", "summary"] } } }
```

These two cover today's `multi_match` on `body`/`title` (`03-...md` §3.2) and extend lexical search to
`topic`/`summary`.

### 7.2 Dense — kNN on chunk vectors (nested)

```json
{
  "query": {
    "nested": {
      "path": "chunks",
      "query": {
        "knn": {
          "field": "chunks.embedding",
          "query_vector": [/* 384-dim, multilingual-e5-small, "query:" prefix */],
          "num_candidates": 250,
          "filter": [
            { "term":  { "subset_ids": "<subset>" } },
            { "range": { "byte_size": { "lte": 5242880 } } },
            { "range": { "primary_date_time": { "gte": "2024-01-01" } } }
          ]
        }
      },
      "score_mode": "max",
      "inner_hits": { "size": 3, "name": "matched_chunks" }
    }
  }
}
```

In 9.2 the `filter` may mix **top-level (parent) metadata** (subset, size, date, email) and **nested
(chunk) metadata** as pre-filters in one `knn` query (the capability that makes nested the right
choice — §3.1). Returns k documents, each with its best passages via `inner_hits`.

### 7.3 Sparse — ELSER on parent metadata

```json
{
  "query": {
    "sparse_vector": {
      "field": "summary_sparse",
      "inference_id": "<elser-endpoint>",
      "query": "<query>"
    }
  }
}
```
Repeat / combine across `title_sparse`, `summary_sparse`, `topic_sparse`. (A per-chunk
`chunks.text_sparse` can be added later for sparse-on-chunks — §10.)

### 7.4 Hybrid — RRF and linear retrievers

**Dense + BM25 on chunks (RRF):**
```json
{
  "retriever": {
    "rrf": {
      "rank_window_size": 100,
      "retrievers": [
        { "standard": { "query": { "nested": {
            "path": "chunks",
            "query": { "match": { "chunks.text": "<query>" } },
            "score_mode": "max",
            "inner_hits": { "name": "bm25_chunks" }
        } } } },
        { "knn": {
            "field": "chunks.embedding",
            "query_vector": [/* 384-dim */],
            "k": 100, "num_candidates": 250,
            "inner_hits": { "name": "knn_chunks" }
        } }
      ]
    }
  }
}
```
This mirrors production's default RRF (`03-...md` §3.1) but over the nested structure.

**Sparse (ELSER) + BM25 on parent metadata (RRF or linear):** swap in `standard` retrievers wrapping a
`sparse_vector` query and a `multi_match`. The **`linear`** retriever is available when weighted
combination with `minmax`/`l2_norm` normalization is preferred over reciprocal-rank fusion (e.g.
"weight kNN 5× BM25").

### 7.5 Reranking roadmap (R7 future)

| Method | ES 9.2 mechanism | Fields needed | Status |
|---|---|---|---|
| MMR (diversity) | Application-side (as today, `03-...md` §3.3): fetch candidates + chunk vectors, greedy MMR. | `chunks.embedding` (or re-embed) | Available now. |
| Cross-encoder rerank | `text_similarity_reranker` retriever (Elastic Rerank `.rerank-v1-elasticsearch` or custom `rerank` inference endpoint). | text field to rerank on (`chunks.text` via inner_hits, or `summary`) | Native, Platinum. |
| Late-interaction (ColBERT-style) | `rank_vectors` field + `maxSimDotProduct` script in a `rescore`/`script_score` second phase. | optional `chunks.late_interaction` (`rank_vectors`) | Experimental in 9.2; reserve field if pursuing. |
| LLM-based rerank | Application/agent-side over returned chunks. | returned `chunks.text` | Available now. |

### 7.6 Coverage matrix — proof of parity-or-better (R11)

| Capability | Current index | New nested index | Parity? |
|---|---|---|---|
| BM25 on chunk text | `body` | `chunks.text` | = |
| BM25 on title | `title` (unpopulated) | `title` (+ `topic`, `summary`) | ≥ (more fields) |
| Dense kNN on chunks | `embedding` | `chunks.embedding` (same config) | = |
| RRF (dense+BM25) | yes | yes (nested) | = |
| BM25 + MMR | yes (app-side) | yes (app-side) | = |
| Sparse / ELSER | **no** | `*_sparse` on title/topic/summary (+ optional chunk sparse) | **+ new** |
| Hybrid sparse+BM25 (parent) | no | `rrf`/`linear` retrievers | + new |
| Cross-encoder / late-interaction rerank | no | `text_similarity_reranker` / `rank_vectors` | + new |
| Subset scoping | `subsetIds` | `subset_ids` | = |
| Date range filter | `metadata.primaryDateTime` | `primary_date_time` | = |
| Email participant filter | `metadata.email*` | `email_from/to/cc/bcc` | = |
| Doc grouping | `documentId` + collapse + separate first-chunk | native (one doc + `inner_hits`) | ≥ (simpler) |
| 5 MB exclusion | ADLS file size at ingest | `byte_size` ingest + query-time | ≥ (also query-time) |

No current capability is lost; several are added.

---

## 8. Read / Return Patterns (R8, R9)

The same index supports both interaction styles with no remapping:

- **Single-call (everything in ES):** one nested query (or `rrf` retriever) returns the top-k parent
  documents; `inner_hits` returns the matched chunks per document; the full chunk list and all parent
  metadata are available from `_source`. The agent receives documents + matched chunks + context in
  one response.
- **Multi-step:** (1) find relevant documents (metadata filter + cheap query, request no/low
  `inner_hits`); (2) fetch all chunks for chosen documents (get parent `_source` or a nested query
  with large `inner_hits`); (3) re-query other chunks with a different strategy. Each step is a normal
  query against the same index.

**Return shape (R9):** the index is agnostic.
- *Separate objects (join agent-side):* return parent fields and `inner_hits` chunks as distinct
  structures; the agent joins by `document_artifact_id`.
- *Denormalized chunks-with-metadata:* the client flattens parent metadata onto each returned chunk
  before handing to the LLM. This is a response-assembly choice, not an index change.

**Flat fallback (Pattern B):** when an experiment needs a **global chunk ranking** (top-k individual
chunks across the whole corpus regardless of parent), use a flat per-chunk index and `collapse` on
`document_artifact_id` to group. Documented so both centric models are available.

---

## 9. Chunking-Agnostic Guarantees (R6)

The chunk schema stores only `chunk_index`, `text`, `token_count`, `embedding`. None of these assume a
particular splitting method. Therefore all of the following work **without any index change**:
- fixed token window **with** overlap (current production: 500 tokens / 100 overlap — `02-...md` §3),
- fixed token window **without** overlap,
- sentence-/separator-boundary or semantic chunking, with or without overlap.

Only the *values* of `chunk_index`/`text`/`token_count`/`embedding` differ between strategies. The
sole field that must not be derived by summing chunks is the document-level `token_count` (§6.2),
precisely because overlap makes chunk token counts non-additive. (A future per-chunk
`chunks.text_sparse` or `chunks.late_interaction` can be added later without disturbing existing data —
§10.)

---

## 10. Open Decisions (to finalize at implementation time)

These are intentionally left open; they do not block the structural design:

1. **`title` source.** Identify the RelativityOne field to populate `title` (and/or map
   `metadata.documentName`). `RelativityDocument` does not read it today, and the live index leaves
   `title` empty.
2. **Dense-on-parent.** Whether to also store a dense vector for `summary` (in addition to ELSER
   sparse) for semantic search over summaries.
3. **Sparse-on-chunks.** Whether to reserve `chunks.text_sparse` (`sparse_vector`) now for ELSER over
   chunk text (zero cost if unused) or add later.
4. **Late-interaction.** Whether to reserve `chunks.late_interaction` (`rank_vectors`) now for
   ColBERT-style reranking (experimental in 9.2) or add later.
5. **Vector index type.** `bbq_hnsw` (parity default) vs `bbq_disk`/DiskBBQ (memory-efficient at
   scale) for `chunks.embedding` (§4.3) — a reindex is required to change.
6. **`subset_ids`.** Keep for multi-tenancy parity (recommended) or drop for a single-purpose
   experimental index.
7. **`semantic_text` alternative.** Whether to model the parent sparse fields as `semantic_text`
   (ES-managed inference + automatic query expansion) instead of explicit `sparse_vector` + inference
   pipeline. Trade-off: less control vs less plumbing.
8. **`documentModifyTime`.** Add a parent `date` if a Relativity source is wired (present in the
   current mapping, absent from `RelativityDocument`).

Settled decisions (recorded for traceability): nested objects as primary pattern; flat as fallback;
first chunk = `chunks[chunk_index == 0]` with no denormalized copy; `byte_size = len(text.encode("utf-8"))`
with a 5,242,880-byte threshold; include `token_count` and `chunk_count`.

---

## 11. Source References

**This folder:**
- `00-...qna-subsetting.report.md` — original subsetting context.
- `01-air-assist-elasticsearch-index.md` — current index mapping, vector config, metadata registry, gaps.
- `02-index-population-pipeline.md` — ingestion, chunking (500/100), embeddings, 5 MB `TooLarge`, identifiers.
- `03-retrieval-strategies.md` — RRF/BM25/MMR strategies, metadata filtering, first-chunk injection, return shapes.
- `04-relativity-object-manager-api.md` — QuerySlim vs Export, conditions, long-text, resume.

**Code:**
- [`es_index_explorer/relativity/models.py`](../es_index_explorer/relativity/models.py) — `RelativityDocument`.
- [`es_index_explorer/relativity/normalize.py`](../es_index_explorer/relativity/normalize.py) — field normalization (incl. summary/topic).
- [`es_index_explorer/config.py`](../es_index_explorer/config.py) — field-mapping configuration.

**Elasticsearch 9.2 documentation (consulted):**
- kNN search — nested kNN, `inner_hits`, mixed top-level + nested pre-filters (GA 9.2): https://www.elastic.co/docs/solutions/search/vector/knn
- `knn` query (inside `nested`, `score_mode=max`): https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query
- `dense_vector` mapping (bbq_hnsw / bbq_disk / int8_hnsw, `_source` exclusion default): https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector.md
- Better Binary Quantization & DiskBBQ (`bbq_disk`, 9.2): https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/bbq
- `sparse_vector` field & query (ELSER): https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-sparse-vector-query
- ELSER ingest/sparse workflows: https://www.elastic.co/docs/solutions/search/vector/dense-versus-sparse-ingest-pipelines
- Linear retriever: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever
- Text similarity reranker retriever: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/text-similarity-reranker-retriever
- `rank_vectors` / late-interaction MaxSim: https://github.com/elastic/elasticsearch/pull/118804 and https://www.elastic.co/search-labs/blog/late-interaction-model-colpali-scale
- `join` field type & limitations: https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/parent-join
- What's new in Elastic 9.2 (DiskBBQ, ELSER on EIS): https://www.elastic.co/blog/whats-new-elastic-9-2-0
