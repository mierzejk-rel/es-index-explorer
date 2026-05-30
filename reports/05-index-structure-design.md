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
open for implementation time are collected in §11.

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
| R12 | Derived field `byte_size` enabling an **optional query-time** filter on documents (and their chunks) whose content exceeds 5 MB (size computed faithfully to today's behavior). Applied at query time, not as an ingest-time exclusion — every document is indexed. | User |
| R13 | Index-setup code must set all important parameters explicitly — no reliance on cluster-template defaults (unlike today, where vector params come from an external template). | User |
| R14 | Retrieval must be able to concatenate a contiguous series of sibling chunks into one larger chunk, de-duplicating the dynamic per-boundary overlap so the shared text appears once (not naive `str + str`). | User |

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
  (§6): `byte_size`, `token_count`, `char_count`, `chunk_count`, and the multi-tenancy `subset_ids`.
- **Child (chunk)** carries only what is derived from `extracted_text`: `chunk_index`, `text`,
  `embedding`, `token_count`, `leading_overlap_chars`. Per R6, the chunk schema is intentionally
  agnostic to *how* the text was split.

### 2.3 Fields explicitly in/out of scope

- **`title`** — kept. It exists in the current ES mapping and in qna-service's BM25 `multi_match`
  field list, although it is **never populated today** (`01-air-assist-elasticsearch-index.md` §4.1,
  §8). Its RelativityOne source field is not yet read by `RelativityDocument` (see §11).
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

### 3.1 Pattern A — Nested objects (recommended default — document-centric; to be confirmed by evaluation)

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
  set to 25,000; the Elasticsearch default is 10,000).
- **Deletion/filtering semantics (R12).** Excluding a parent (e.g. `byte_size > 5MB`) excludes all
  its chunks automatically — they are the same document.

**Trade-offs / caveats (documented honestly):**
- **Ranking unit changes from chunk to document.** A nested `knn`/`nested` query ranks **parent
  documents** — `k`/`size` counts documents, each scored by its best passage (`score_mode: max`) —
  and the matching chunks are returned via `inner_hits`. Chunks remain fully scored and retrievable;
  what changes is that the top-level result enumerates documents, not chunks. See §3.1.1 for the
  paradigm framing and §7.5–§7.6 for how to obtain a chunk-level ranking on the nested index.
- Changing one chunk requires reindexing the parent document (acceptable per R4).
- Each nested chunk is stored as its own hidden Lucene document beneath the parent. These child
  documents are fully searchable (`nested`/`knn` queries) and returnable (`inner_hits`); they are
  simply **not addressable as standalone top-level hits** — a search returns the parent with its
  matching chunks attached. They still count toward Lucene's internal `docs.count`, which is relevant
  for shard sizing and the `nested_objects` limit.

#### 3.1.1 Ranking unit: chunk vs document

This is the one property where the two paradigms genuinely differ, so it is called out explicitly to
avoid ambiguity.

- **Flat / chunk-centric (today):** the chunk *is* the document, so a query produces a **global
  ranking of chunks** (`size: 100` → up to 100 chunks, possibly several from the same document).
  qna-service then trims to the top 25 chunks and groups them by document for display
  (`03-retrieval-strategies.md` §5).
- **Nested / document-centric:** a query produces a ranking of **documents**, each scored by its best
  passage; the matching chunks come back under `inner_hits`. The top-level `size`/`k` counts
  documents, not chunks. Chunks remain fully scored and retrievable — nothing is all-or-nothing per
  document — but the unit the top-level result enumerates is the document.

Neither unit is inherently better for RAG: chunk-centric ranking is the current behaviour, while
document-centric ranking adds grouping, parent pre-filters, and de-duplicated metadata. The nested
index can still produce a chunk-level ranking via a small **client-side flatten**, and a hybrid
(RRF/linear) chunk ranking via two documented approaches. The scoring mechanics, the flatten
procedure, completeness conditions, and the hybrid options all live in §7.5–§7.6; the empirical
choice between paradigms is deferred to evaluation.

### 3.2 Pattern B — Flat denormalization (current production model — chunk-centric paradigm)

One ES document per chunk; parent fields copied onto every chunk (today's model —
`02-index-population-pipeline.md` §5).

- **Pros:** global chunk-level ranking is natural; simplest RRF over chunks; scales trivially across
  shards; no nested/join overhead; this is the proven production shape (guarantees R11 parity).
- **Cons:** parent metadata duplicated on every chunk (storage + update amplification); reconstituting
  the parent and fetching the "first chunk" need extra queries or `collapse` (today qna-service issues
  a *separate* first-chunk query — `03-retrieval-strategies.md` §5.1); parent-level filters are
  evaluated per chunk.
- **Verdict:** A different, **chunk-centric paradigm** with its own strengths (native global chunk
  ranking, trivial sharding, guaranteed R11 parity) — chosen when chunk-level ranking is the priority.
  Not inferior to nested; it simply optimizes for a different unit of retrieval. Which paradigm serves
  the RAG better is an evaluation question (see §3.1.1).

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
| **1** | **Nested (A)** | Excellent (9.2 GA mixed pre-filters) | Excellent (one doc, no dup) | Excellent (nested knn/BM25 + parent sparse via retrievers) | Excellent (`inner_hits`) | Excellent (immutable children) | Good (per-parent block; fine at <few dozen chunks) | **Recommended default (document-centric)** |
| 2 | Flat (B) | Good (filters per chunk) | Weak (dup metadata; extra first-chunk query) | Good (chunk-only) | Good | Good | Excellent | Alternative paradigm (chunk-centric); evaluate |
| 3 | Separate indices (C) | Partial (two queries) | Manual join | No (multi round-trip) | No | Excellent | Excellent | Niche multi-step only |
| 4 | `join` field (D) | Poor (slow has_child) | OK | No (doesn't compose with knn/RRF) | Limited | OK (independent child updates we don't need) | Poor (single-shard constraint) | Rejected |

**Decision:** Start with **nested objects (A)** as the recommended default, because of its
metadata/filtering/grouping/de-duplication benefits and single-query parent-pre-filtered hybrid
search. Treat **flat (B)** as a first-class alternative paradigm whose native chunk-level ranking may
prove better for this RAG. This is an evaluation question, not a verdict of inferiority — both are
legitimate, and the document-centric vs chunk-centric choice will be settled empirically.

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
        "nested_objects": { "limit": 25000 },
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

      "title":   { "type": "text" },
      "summary": { "type": "text" },
      "topic":   { "type": "text" },

      "primary_date_time": { "type": "date" },
      "email_from": { "type": "keyword" },
      "email_to":   { "type": "keyword" },
      "email_cc":   { "type": "keyword" },
      "email_bcc":  { "type": "keyword" },

      "byte_size":   { "type": "long" },
      "token_count": { "type": "integer" },
      "char_count":  { "type": "integer" },
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
          "leading_overlap_chars": { "type": "integer" },
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
| `title` | `text` | BM25 full-text only (no keyword sub-field: exact-match/sort/agg on title not needed). | BM25 (R7 lexical-on-title). Unpopulated today (§11). |
| `summary` | `text` | BM25 full-text. | BM25 (R7 lexical-on-summary). |
| `topic` | `text` | BM25 full-text only (LLM-generated sentence, not categorical → no keyword sub-field). | BM25 (R7 lexical-on-topic). |
| `primary_date_time` | `date` | Range queries. | Date range filter (parity with `metadata.primaryDateTime`). |
| `email_from/to/cc/bcc` | `keyword` (multi-valued) | Exact + wildcard per element. | Email participant filters (parity with `metadata.email*`). |
| `byte_size` | `long` | Byte count. | Optional query-time 5 MB filter (R12; §6). |
| `token_count` | `integer` | Token count of full doc. | Analytics, filtering, cost estimation (§6). |
| `char_count` | `integer` | Character count of full doc (`len(extracted_text)`, Unicode code points). | Encoding- and tokenizer-independent length for filtering/analytics (§6). |
| `chunk_count` | `integer` | Number of chunks. | Cross-index filter/sort/agg convenience (§6). |
| `subset_ids` | `keyword` (multi-valued) | Multi-tenancy scoping. | `term` filter parity with production's always-on subset filter (`03-...md` §4.2). |
| `title_sparse` / `summary_sparse` / `topic_sparse` | `sparse_vector` | Stores ELSER token-weight pairs. | Sparse (ELSER) retrieval on parent metadata (R7 sparse). |
| `chunks` | `nested` | Preserves chunk independence for nested kNN/BM25 with parent pre-filters. | Container for child chunks (R1). |
| `chunks.chunk_index` | `integer` | 0-based order. | First-chunk selection (`== 0`), ordering within doc. |
| `chunks.text` | `text` | BM25 on chunk text. | Lexical chunk retrieval (R7 BM25-on-chunks). |
| `chunks.token_count` | `integer` | Per-chunk tokens. | Diagnostics; not summable to document tokens (§6). |
| `chunks.leading_overlap_chars` | `integer` | Leading characters duplicated from previous chunk; chunk 0 = 0. | Tokenizer-free de-duplication for contiguous chunk concatenation (R14). |
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
| `title` | `title` | Same (still optional/unpopulated; §11). |
| `documentModifyTime` | (optional add) | Not in `RelativityDocument` today; add as parent `date` if a source is wired. |
| `createdAt` | dropped | Never populated in production (`01-...md` §8). Add only if needed. |
| `metadata.primaryDateTime` | `primary_date_time` | Promoted to first-class typed field. |
| `metadata.emailFrom/To/Cc/Bcc` | `email_from/to/cc/bcc` | Promoted to first-class. |
| `metadata.documentName` | `title` or optional `document_name` | Map to whichever R1 field is chosen (§11). |

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
`semantic_text` fields so ES manages inference end-to-end — see §11.)

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

### 6.1 `byte_size` — optional query-time size filter (R12)

**Goal:** reproduce production's `TooLarge` size *measure* faithfully, but apply it as an **optional,
query-time** filter rather than an ingest-time exclusion. Production keys its filter off the **raw ADLS
file byte size** (`FileSize > 5,242,880` bytes — `02-...md` §2.2, §8), *not* the per-chunk
`chunkSize = UTF8.GetByteCount(chunk) * 2` statistic (which has no retrieval role and over-estimates
non-ASCII by 2–3×).

Since `es-index-explorer` reads `extracted_text` directly (no ADLS file handle), the closest faithful
reproduction of the raw file size is the **actual UTF-8 byte length** of the text:

```python
byte_size = len(extracted_text.encode("utf-8"))
# stored on every document; the 5 MB threshold (5_242_880 = 5 * 1024 * 1024) is applied
# optionally at query time, never to exclude at ingest
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

**Enforcement — optional, query-time only:** unlike production (which drops >5 MB documents at ingest
as `TooLarge`), this index **stores every document** and applies the 5 MB rule only when a query opts
in, by adding `"range": { "byte_size": { "lte": 5242880 } }`. Because chunks are nested in the parent,
filtering out the parent removes its chunks automatically. Indexing everything lets us test whether
intentionally skipping large documents costs retrieval/eval parity, and the filter can be toggled (or
re-thresholded) per query.

**Implicit size ceiling:** indexing is still bounded by the maximum permitted number of nested chunks
per document (`index.mapping.nested_objects.limit`, §3.1) — a document that produces more chunks than
the limit is rejected in full and is absent from the index. That cap is effectively a *derived* upper
bound on document size, independent of (and not toggleable like) the optional `byte_size` filter.

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

### 6.4 `char_count` — full-document characters (optional)

Character count of the **entire** `extracted_text`, defined as **`len(extracted_text)`** — i.e.
Python `str` length, which counts Unicode code points (not bytes, not UTF-16 code units, not grapheme
clusters). Like `token_count` it is **not derivable from the chunks** (overlap double-counts boundary
characters) and **not retrievable for free** from Elasticsearch (the parent stores no full-text field,
and `_source` scripting would be slow), so it is computed once at ingest — a single tokenizer-free
integer. It is a *distinct* length lens: it equals `byte_size` only for pure ASCII and diverges for
multi-byte (non-ASCII) content, and is unrelated to the token scale. Useful as an encoding- and
tokenizer-independent document length for filtering and analytics. Chunk-level character count is
deliberately **not** stored: when a chunk is returned its text is already present, so `len(text)` is a
zero-cost client-side computation.

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
`chunks.text_sparse` can be added later for sparse-on-chunks — §11.)

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

### 7.5 Ranking semantics and chunk-level results

Because chunk strategies (§7.1–§7.2, §7.4) run over the `nested` `chunks` field, they rank **parent
documents**; the matching chunks come back under `inner_hits`. This section is the single home for the
scoring mechanics needed to turn that into a chunk-level ranking.

**`score_mode` (how child scores roll up to the parent).** A [`nested` query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-nested-query)
supports `avg` (default), `max`, `sum`, `min`, and `none`. A nested [`knn` query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query)
supports **only `score_mode: max`** — the document is scored by its single nearest passage. Use `max`
for any strategy you intend to flatten to chunks (see completeness below).

**Cross-document chunk comparability.** For a single-signal query, every `inner_hits` chunk `_score`
comes from the same scoring function, so chunk scores are comparable across documents and a global
chunk ranking by `_score` is valid. With `number_of_shards: 1` (our design) BM25 IDF (Inverse Document
Frequency) statistics are global, so chunk scores are directly comparable. The only caveat is
**multi-shard lexical/BM25**: each shard then computes IDF from its own local stats, so identical text
can score differently per shard; if the index is ever sharded, query with `dfs_query_then_fetch`
(DFS, Distributed Frequency Search) to compute global IDF. kNN/cosine carries no corpus statistics and
is unaffected, so dense scores are always comparable.

> Note — *single-signal* vs *hybrid*: a **single-signal** query ranks with exactly one retrieval
> signal (lexical BM25, dense kNN, or sparse alone), so every chunk `_score` comes from one scoring
> function and is directly comparable. A query that combines two or more signals and fuses them
> (e.g. dense + BM25 via RRF/`linear`) is termed **hybrid** in this report (equivalently
> *multi-signal*); its per-signal scores live on different scales and are not directly comparable, so
> a chunk ranking requires explicit fusion — see §7.6. "Hybrid" is preferred here for consistency with
> §7.4/§7.6; "multi-signal" is its literal structural antonym.

**Client-side flatten (single-signal global chunk ranking).**
1. Request enough parents (e.g. `k`/`size` = 100, matching today's `results_from_rrf`) and a large
   enough `inner_hits.size` (at least the largest expected number of relevant chunks in any one
   document).
2. Read every chunk from every parent's `inner_hits`; each carries its own `_score`.
3. Flatten into a single list, sort by `_score`, take the top N (e.g. 25) → "top-N chunks, possibly
   several per document", the shape qna-service feeds the LLM today.

**Completeness.** With `score_mode: max` the flatten is provably complete: any chunk in the global
top-N lives in a document whose max ≥ that chunk, and there are at most N such documents, so the
top-`k` documents by max (with `k ≥ N`) contain every document holding a top-N chunk — provided
`inner_hits.size` is large enough to surface them. The guarantee breaks for `avg`/`sum` (a document
with one strong chunk can rank low), which is the second reason to use `max` when flattening. Worked
example: doc A {0.6, 0.45, 0.4} (max 0.6) and doc B {0.55, 0.5, 0.33} (max 0.55) → documents rank
A > B, but the global top-3 *chunks* are 0.6 (A), 0.55 (B), 0.5 (B), which the flatten recovers as long
as `inner_hits.size ≥ 2`.

**`inner_hits` limits.** `size` defaults to 3; only *matching* chunks are returned; the global cap
`index.max_inner_result_window` (default 100) bounds `from + size`.

**`inner_hits` options** — per [Retrieve inner hits](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrieve-inner-hits#inner-hits-options):
- `from` — offset of the first chunk to fetch per `inner_hits`.
- `size` — maximum chunks returned per `inner_hits` (default 3).
- `sort` — how chunks are sorted per `inner_hits` (default: by `_score`).
- `name` — the key for this inner-hit set in the response (default: the nested path); give each a
  unique name when multiple inner_hits are present.

Inner hits also support per-document features: highlighting, explain, search fields, source filtering,
script fields, doc value fields, versions, and sequence/primary numbers. Performance tip: for nested
inner hits, disabling `_source` and reading `docvalue_fields` instead avoids the relatively expensive
per-hit source extraction.

Reranking (§7.7) consumes whatever this stage emits — documents, or the flattened chunks above — so it
is out of scope here.

### 7.6 Hybrid global chunk ranking: two approaches

A hybrid (RRF/`linear`) query fuses at the **document** level; there is no native fused per-chunk
score. To obtain a global *chunk* ranking under hybrid, use one of two approaches.

**Approach A — single fused call with propagated `inner_hits`.** Give each sub-retriever's nested query
a uniquely-named `inner_hits`; the compound retriever propagates them, computed after fusion on the
final top documents ([RRF inner hits](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion);
[retrievers examples](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/retrievers-examples)).
The client then reads both named sets (one per signal) and fuses per chunk.
- Properties: one round-trip; reuses ES's document fusion; returns document grouping for free.
- Limits / trade-offs: a chunk carries **up to two** scores — it appears in a signal's set only if its
  document made the top set *and* the chunk was within that signal's `inner_hits.size`; fuse on
  **ranks** for RRF (BM25 and cosine scales differ) or on min-max-normalized scores for `linear`; the
  chunk candidate pool is bounded by `rank_window_size` and each signal's `inner_hits.size`.

**Approach B — two single-signal queries + client-side RRF of chunk rankings.** Issue one nested `knn`
query and one nested BM25 query, each with `inner_hits`; flatten each to a per-signal chunk ranking
(comparable within a signal, §7.5), then RRF the two **chunk** rankings client-side.
- Properties: full control over per-signal candidate depth; a clean, well-defined global fused chunk
  ranking; no dependence on the post-fusion top-document set.
- Trade-offs: two round-trips; it is a concrete instance of the §9 multi-step pattern.

**Relation to the current approach.** Today's flat RRF ranks chunks natively in a single query. Both A
and B reproduce a chunk-level fused ranking on the document-centric index with the trade-offs above:
pick **A** for one-call, document-grouped hybrid retrieval; pick **B** for an exact global fused chunk
ranking. Neither makes the document-centric index inferior — they are the documented options for
recovering chunk-level hybrid ranking on top of nested's grouping and parent pre-filters.

### 7.7 Reranking roadmap (R7 future)

| Method | ES 9.2 mechanism | Fields needed | Status |
|---|---|---|---|
| MMR (diversity) | Application-side (as today, `03-...md` §3.3): fetch candidates + chunk vectors, greedy MMR. | `chunks.embedding` (or re-embed) | Available now. |
| Cross-encoder rerank | `text_similarity_reranker` retriever (Elastic Rerank `.rerank-v1-elasticsearch` or custom `rerank` inference endpoint). | text field to rerank on (`chunks.text` via inner_hits, or `summary`) | Native, Platinum. |
| Late-interaction (ColBERT-style) | `rank_vectors` field + `maxSimDotProduct` script in a `rescore`/`script_score` second phase. | optional `chunks.late_interaction` (`rank_vectors`) | Experimental in 9.2; reserve field if pursuing. |
| LLM-based rerank | Application/agent-side over returned chunks. | returned `chunks.text` | Available now. |

### 7.8 Coverage matrix — proof of parity-or-better (R11)

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
| Global chunk-level ranking (single-signal) | native (chunks are documents) | via `inner_hits` + client-side flatten, provably complete with `score_mode: max` (§7.5) | = (flatten) |
| Global chunk-level ranking (hybrid RRF/linear) | native (chunks are documents) | client-side chunk fusion — Approach A or B (§7.6) | = (client fusion) |
| Doc grouping | `documentId` + collapse + separate first-chunk | native (one doc + `inner_hits`) | ≥ (simpler) |
| 5 MB size filter | ADLS file size, mandatory at ingest | `byte_size`, optional query-time filter (index everything) | reframed (opt-in) |
| Chunk concatenation (overlap-dedup) | not available | `leading_overlap_chars` + retrieval merge (optional ES-side script) | + new |

No current capability is lost; several are added.

---

## 8. Chunk Concatenation (Contiguous-Series Merging) — R14

**Requirement recap:** retrieval must be able to concatenate a contiguous series of sibling chunks
into one larger chunk while **de-duplicating overlap** so the shared text appears once, not twice.
Overlap can be fixed or dynamic (token/character count, or semantic/structure-aware), and the
retrieval layer should **not** depend on a tokenizer.

### 8.1 Field materialization: `chunks.leading_overlap_chars`

We store a chunk-local, tokenizer-free overlap value:

```json
"leading_overlap_chars": { "type": "integer" }
```

**Definition:** the number of **leading characters** of this chunk that are identical to the
**trailing characters of the previous sibling** (`chunk_index - 1`). Chunk 0 = `0`.

This value is computed by the chunker at ingest time. It works for any overlap policy (fixed or
semantic) as long as the overlap is a shared substring between adjacent chunks. If a chunking
strategy ever produces non-identical overlaps, it must set this value to `0` (no dedup), which is
safe and explicit.

### 8.2 Concatenation algorithm (retrieval-side, tokenizer-free)

Given a **contiguous** series of chunks (no gaps in `chunk_index`), ascending by `chunk_index`:

```python
text = series[0].text
for chunk in series[1:]:
    text += chunk.text[chunk.leading_overlap_chars:]
```

**Rules:**
- Only **consecutive** `chunk_index` values are merged. Any gap breaks the series into a new
  concatenation segment.
- The first chunk of the series is kept whole.
- The caller assigns a **concatenated chunk id** (e.g., `"3-6"` or the explicit list `[3,4,5,6]`).
  This is an output-layer concern; no index changes are needed for it.

### 8.3 Can Elasticsearch do this concatenation?

**Short answer:** not out of the box for arbitrary runs; best handled in the retrieval module.

**Fact-based analysis:**
- **ES|QL:** not possible. Nested fields are unsupported in ES|QL and are not returned at all, so
  ES|QL cannot iterate chunks to concatenate them ([ES|QL limitations](https://www.elastic.co/docs/reference/query-languages/esql/limitations)).
- **`script_fields` (Painless):** partially possible. A search-time script can read
  `params._source.chunks`, sort by `chunk_index`, and concatenate `text` while stripping
  `leading_overlap_chars`. This works only for the **“all chunks of the document”** case and requires
  loading/parsing `_source` (documented by Elastic as **very slow** per hit —
  [retrieve selected fields](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrieve-selected-fields)).
- **Arbitrary contiguous runs (e.g., only the inner_hits matched by a query):** not supported out of
  the box. ES does not know which run to merge — that comes from `inner_hits` — and static script
  parameters cannot express per-document dynamic runs in a single query.

**Recommendation:** implement concatenation in the retrieval layer (simple, tokenizer-free, handles
dynamic overlap and gaps). Optionally add a Painless `script_fields` helper for full-document
reconstruction, with the `_source` performance caveat.

---

## 9. Read / Return Patterns (R8, R9)

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

**Chunk-centric alternative (Pattern B):** when an experiment wants a **native global chunk ranking**
(top-k individual chunks across the whole corpus regardless of parent) without the client-side flatten
of §7.6, use the flat per-chunk index and `collapse` on `document_artifact_id` to group. Both paradigms
are available; the choice is an evaluation question (§3.1.1), not a fallback.

---

## 10. Chunking-Agnostic Guarantees (R6)

The chunk schema stores only `chunk_index`, `text`, `token_count`, `leading_overlap_chars`, and
`embedding`. None of these assume a particular splitting method. Therefore all of the following work
**without any index change**:
- fixed token window **with** overlap (current production: 500 tokens / 100 overlap — `02-...md` §3),
- fixed token window **without** overlap,
- sentence-/separator-boundary or semantic chunking, with or without overlap.

Only the *values* of `chunk_index`/`text`/`token_count`/`leading_overlap_chars`/`embedding` differ
between strategies. The sole field that must not be derived by summing chunks is the document-level
`token_count` (§6.2), precisely because overlap makes chunk token counts non-additive. (A future
per-chunk `chunks.text_sparse` or `chunks.late_interaction` can be added later without disturbing
existing data — §11.)

---

## 11. Open Decisions (to finalize at implementation time)

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
9. **ES-side full-document reconstruction.** Whether to provide a `script_fields` helper that
   concatenates all chunks server-side (uses `_source`, slow) vs always concatenating in the
   retrieval module (recommended default).
10. **Default hybrid chunk-fusion path.** Approach A (single fused call + propagated `inner_hits`) vs
    Approach B (two single-signal queries + client-side RRF of chunk rankings) from §7.6 — to settle
    during retrieval-module work.
11. **Retrieval paradigm.** Document-centric (nested) vs chunk-centric (flat) is the recommended
    default vs the alternative; the final choice for this RAG is an evaluation outcome (§3.1.1), not a
    structural decision blocked here.

Settled decisions (recorded for traceability): nested objects (document-centric) as the recommended
default, with flat (chunk-centric) retained as a first-class alternative to be chosen by evaluation
(not a fallback); a chunk-level ranking on the nested index is obtained via the client-side flatten
(single-signal, §7.5) and via Approach A or B for hybrid (§7.6), both accepted; first chunk =
`chunks[chunk_index == 0]` with no denormalized copy; `byte_size = len(text.encode("utf-8"))` with a
5,242,880-byte threshold, applied as an optional query-time filter (not an ingest-time exclusion —
every document is indexed); include `token_count`, `char_count` (doc-level, `len(str)` code points), and
`chunk_count`; R14 satisfied via
`leading_overlap_chars` + retrieval-side concatenation.

---

## 12. Source References

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
- `nested` query (`score_mode` options): https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-nested-query
- Retrieve inner hits (`inner_hits` options, limits, features): https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrieve-inner-hits
- Reciprocal rank fusion (inner hits in RRF): https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
- Retrievers examples (computing inner hits from sub-retrievers): https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/retrievers-examples
- `dense_vector` mapping (bbq_hnsw / bbq_disk / int8_hnsw, `_source` exclusion default): https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector.md
- Better Binary Quantization & DiskBBQ (`bbq_disk`, 9.2): https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/bbq
- `sparse_vector` field & query (ELSER): https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-sparse-vector-query
- ELSER ingest/sparse workflows: https://www.elastic.co/docs/solutions/search/vector/dense-versus-sparse-ingest-pipelines
- Linear retriever: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever
- Text similarity reranker retriever: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/text-similarity-reranker-retriever
- `rank_vectors` / late-interaction MaxSim: https://github.com/elastic/elasticsearch/pull/118804 and https://www.elastic.co/search-labs/blog/late-interaction-model-colpali-scale
- `join` field type & limitations: https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/parent-join
- What's new in Elastic 9.2 (DiskBBQ, ELSER on EIS): https://www.elastic.co/blog/whats-new-elastic-9-2-0
- ES|QL limitations (nested unsupported): https://www.elastic.co/docs/reference/query-languages/esql/limitations
- Retrieve selected fields / `script_fields` + `_source` performance: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrieve-selected-fields
- Painless field API / scripting: https://www.elastic.co/docs/explore-analyze/scripting/script-fields-api
