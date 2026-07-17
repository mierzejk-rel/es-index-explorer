# 09 — Retrieval Experiment Design

**Purpose:** Experiment plan for evaluating the impact of Summary, Topic, and Title fields —
together with dense, sparse, and hybrid retrieval strategies — on the aiR Assist RAG agent's
answer quality. This report defines the factor space, proposes a curated experiment matrix,
specifies the tool implementations required in `air-assist-agent`, and describes the evaluation
protocol. It is written for human engineers and AI agents who will implement, execute, and
analyse these experiments.

**Scope:** Retrieval-layer experiments only. The LangGraph computational graph structure is
held constant (v3 `MultihopFreeTextRag`); changes are confined to tool implementations, tool
registrations, system prompt retrieval instructions, and per-experiment TOML configs.

**Prior art in this folder:**

- `08-index-design-and-ingestion.md` — authoritative index design and retrieval query patterns
  (sections 13.1–13.9).
- `03-retrieval-strategies.md` — current production retrieval strategies and qna-service
  architecture.
- `01-air-assist-elasticsearch-index.md` — current production index structure.
- `07-client-side-sparse-vectors.md` — sparse vector migration and query DSL.

---

## 1. Introduction and Motivation

### 1.1 The hypothesis

The current aiR Assist production retrieval pipeline relies on two signals: BM25 on chunk text
(`body`) and dense kNN on chunk embeddings (`embedding`), fused via Elasticsearch RRF. The
`title` field exists in the production index mapping but is never populated; Summary and Topic
fields do not exist at all.

The new nested index (`air_assist_nested.json`) introduces three parent-level metadata fields
— `title`, `summary`, `topic` — each available in both BM25 (`text` type) and learned-sparse
(`sparse_vector` type) form. The central hypothesis is:

> **Enriching the retrieval signal with document-level metadata (title, summary, topic) —
> via BM25, learned-sparse, or hybrid fusion — improves answer quality, retrieval recall,
> and citation accuracy compared to chunk-only retrieval.**

### 1.2 Why metadata fields matter for e-discovery RAG

E-discovery questions frequently target document-level properties that are poorly represented in
any single chunk:

- **"Find documents about the Da Vinci attack"** — a topic-level query where `topic` or
  `summary` directly answers the intent, while chunk-level BM25 requires the exact phrase to
  appear in a passage.
- **"What role did Belford play?"** — an entity-relationship query where `title` ("RE: Belford
  Network Access Review") or `summary` provides immediate context that a 400-token chunk
  extracted from a long email body may not.
- **"Show me compliance documents"** — a category query where `topic` ("pharmaceutical
  compliance") matches directly, while BM25 on chunk text needs the term to appear in the passage.

Learned-sparse representations (SPLADE-family models) add term expansion — semantically related
tokens not present in the original text receive non-zero weights — improving recall on synonym
and paraphrase queries without the vocabulary mismatch that limits BM25 [1][2].

### 1.3 Prior work

Recent benchmarks confirm that hybrid retrieval (BM25 + dense) with RRF fusion is the 2025–2026
production default, improving nDCG@10 by 1.4% over ELSER alone and 18% over BM25 alone on
BEIR [3]. A two-stage pipeline — hybrid retrieval followed by neural reranking — consistently
outperforms single-stage approaches across domains [1][4]. For agentic RAG specifically,
query decomposition yields consistent gains in structured/domain-specific settings but can
degrade precision on open-domain multi-hop benchmarks, suggesting that agentic enhancements must
be applied selectively [5].

Adding sparse (SPLADE/ELSER) as a third signal alongside BM25 and dense is less well studied in
production RAG systems, but Elastic's own benchmarks show that RRF across BM25 + ELSER
achieves robust improvement without tuning [3], and the financial QA benchmark [1] found that
convex combination with α=0.5 (equal BM25/dense weight) outperforms RRF (k=60) at Recall@5
(0.726 vs 0.695), suggesting the linear retriever may be worth testing.

---

## 2. Baseline Characterization

### 2.1 Production agent architecture

```
User question
    │
    ▼
air-assist-agent (LangGraph v3, MultihopFreeTextRag)
    │  LLM selects tool, generates query
    ▼
McpToolProvider → qna-service MCP /v2
    │  injects subsetId, auth headers
    ▼
DocumentProviderV2 → LaunchDarkly strategy selection
    │  RRF / BM25 / BM25+MMR
    ▼
Elasticsearch (flat chunk index: {tenant}-{workspace}-qna-subsetting)
    │  returns ranked chunks
    ▼
Post-processing: security trim → top-25 → first-chunk fetch → group by document (production)
    │
    ▼
GroupedDocumentSearchResult → agent (XML for LLM)
```

Source: `03-retrieval-strategies.md` §1; `rag_agent.py` lines 136–194.

*Note:* the production "first-chunk fetch" (always adding chunk 0 for each returned document)
is preserved in the experiment Tier 0 and Tier 1 as an **output-only decoration** step that
does not affect ranking or fusion. It is not present in Tier 2. See §7.4.

### 2.2 Production tools

| Tool | Parameters | ES strategy |
|---|---|---|
| `GetRelevantDocuments` | `query` (required) | BM25 `multi_match` on `body`, `title` + kNN on `embedding` via RRF |
| `GetRelevantDocumentsWithMetadataFilter` | `query?`, `dateFrom?`, `dateTo?`, `emailParticipants?` | Same + `bool.filter` on `metadata.*` |

Both tools return top 25 grouped chunks. The LLM is instructed to perform 1–3 retrieval
iterations before answering (013.toml system prompt).

Source: `tool_definitions.py`; `03-retrieval-strategies.md` §2.1.

### 2.3 Production index structure (flat, chunk-per-document)

| Field | ES type | Populated | Retrieval role |
|---|---|---|---|
| `body` | `text` | Yes | Primary BM25 |
| `title` | `text` | **No** (mapped but never written) | In `multi_match` but contributes nothing |
| `embedding` | `dense_vector` (384, cosine, bbq_hnsw) | Yes | kNN retrieval |
| `documentId` | `integer` | Yes | Grouping, filtering |
| `chunkId` | `integer` | Yes | Ordering |
| `subsetIds` | `keyword` | Yes | Mandatory filter |
| `metadata.*` | various | Per-workspace | Date/email filters |

**No Summary. No Topic. No sparse vectors. Title is a dead field.**

Chunking: fixed sliding window, 500 tokens / 100 overlap (HF BPE from e5-small tokenizer).

Source: `01-air-assist-elasticsearch-index.md`; `00-*.report.md`.

### 2.4 Production config (013.toml)

| Parameter | Value |
|---|---|
| Model | `gpt-5.1-2025-11-13` |
| `reasoning_effort` | `low` |
| `max_completion_tokens` | 30,000 |
| `response_format.version` | 2 |
| MCP API version | v2 |
| Retrieval iterations | 1–3 (prompt), forced first call, max 5 |
| Results per call | 25 (qna-service `ResultsSentToGpt`) |

Source: `013.toml`.

---

## 3. New Index Capabilities

The new index (`air_assist_nested.json`) is document-centric: one ES document per Relativity
document, with chunks in a `nested` array. Two indices exist, one per workspace (EMC2,
Mallinckrodt), both with identical mapping.

### 3.1 Available retrieval signals

There are two kinds of signals. **Chunk-level signals** return ranked chunks directly via
`inner_hits`; their scores are globally comparable across documents (single shard, global
IDF / absolute cosine). **Document-level signals** return ranked parent documents (0 inner
hits); each document's rank is broadcast identically to every chunk of that document at
client-side fusion time.

| Signal | Kind | Field(s) | Query type | Returns |
|---|---|---|---|---|
| **BM25 on chunks** | Chunk-level | `chunks.text` | `nested` → `match`, `score_mode: max` | Ranked chunks via `inner_hits` |
| **Dense kNN on chunks** | Chunk-level | `chunks.embedding` (384-dim, cosine, bbq_hnsw) | `nested` → `knn`, `score_mode: max` | Ranked chunks via `inner_hits` |
| **BM25 on parent metadata** | Document-level | `topic`, `summary`, `title` (equal weights) | `combined_fields`, 0 inner hits | Ranked documents; rank broadcast to all their chunks |
| **Sparse on parent metadata** | Document-level | `title_sparse`, `summary_sparse`, `topic_sparse` | `bool/should` (sum), 0 inner hits | Ranked documents; rank broadcast to all their chunks |

### 3.2 Fusion mechanisms (ES-native)

| Mechanism | How it works | Tuning knobs |
|---|---|---|
| **RRF** | Rank-based fusion: `RRF(d) = Σ 1/(k + rank_r(d))` | `rank_constant` (default 60), `rank_window_size` |
| **Linear** | Weighted score sum with normalization (minmax/l2) | Per-retriever weight, normalization method |

Both operate as ES retrievers and can fuse 2+ sub-retrievers in a single round-trip.

### 3.3 New parent fields (the core experiment variable)

| Field | Type | Content | Indexing |
|---|---|---|---|
| `title` | `text` | RelativityOne "Unified Title" | BM25 via standard analyzer |
| `summary` | `text` | AI-generated document summary | BM25 via standard analyzer |
| `topic` | `text` | AI-generated topic label | BM25 via standard analyzer |
| `title_sparse` | `sparse_vector` | Client-side learned-sparse token-weight map | `opensearch-neural-sparse-encoding-v2-distill` |
| `summary_sparse` | `sparse_vector` | Same | Same |
| `topic_sparse` | `sparse_vector` | Same | Same |

All sparse fields have index-time token pruning: `prune: true`,
`tokens_freq_ratio_threshold: 5`, `tokens_weight_threshold: 0.4`.

### 3.4 Query examples

**Chunk BM25 signal (nested, score_mode max):**
```json
{
  "size": 100,
  "_source": ["document_artifact_id", "control_number", "title", "summary", "topic"],
  "query": { "bool": {
    "filter": [ /* subset/date/email/size filters */,
      { "nested": { "path": "chunks", "query": { "term": { "chunks.chunk_index": 0 } },
        "inner_hits": { "name": "first_chunk", "size": 1, "_source": { "includes": ["chunks.text"] } } } }
    ],
    "must": [ { "nested": {
      "path": "chunks",
      "query": { "match": { "chunks.text": "<q>" } },
      "score_mode": "max",
      "inner_hits": {
        "name": "ranked_chunks",
        "size": 50,
        "_source": { "includes": ["chunks.chunk_index", "chunks.text", "chunks.leading_overlap_chars"] }
      }
    } } ]
  } }
}
```
`inner_hits.size` = `min(100, 2 × result_count)` (50 for 25-chunk experiments, 100 for 60-chunk).

**Chunk kNN signal (nested, score_mode max):**
```json
{
  "size": 100,
  "_source": ["document_artifact_id", "control_number", "title", "summary", "topic"],
  "query": { "bool": {
    "filter": [ /* subset/date/email/size filters */,
      { "nested": { "path": "chunks", "query": { "term": { "chunks.chunk_index": 0 } },
        "inner_hits": { "name": "first_chunk", "size": 1, "_source": { "includes": ["chunks.text"] } } } }
    ],
    "must": [ { "nested": {
      "path": "chunks",
      "query": { "knn": { "field": "chunks.embedding", "query_vector": [/*384*/], "k": 100, "num_candidates": 250 } },
      "score_mode": "max",
      "inner_hits": {
        "name": "ranked_chunks",
        "size": 50,
        "_source": { "includes": ["chunks.chunk_index", "chunks.text", "chunks.leading_overlap_chars"] }
      }
    } } ]
  } }
}
```

**Document BM25 signal (`combined_fields`, equal weights, 0 inner hits):**
```json
{
  "size": 25,
  "_source": ["document_artifact_id", "control_number", "title", "summary", "topic"],
  "query": { "bool": {
    "filter": [ /* subset/date/email filters */,
      { "nested": { "path": "chunks", "query": { "term": { "chunks.chunk_index": 0 } },
        "inner_hits": { "name": "first_chunk", "size": 1, "_source": { "includes": ["chunks.text"] } } } }
    ],
    "must": [ { "combined_fields": { "query": "<q>", "fields": ["topic", "summary", "title"] } } ]
  } }
}
```
`title` is included in `fields` only when `title_enabled` is true (EMC2); omitted for Mallinckrodt.
All fields use equal weight (`combined_fields` requires field boosts ≥ 1.0; default is 1.0).
The `first_chunk` inner_hits filter is present in **all four** signal queries (both chunk-level and
document-level), not only in document-level queries. For chunk-only experiments (E0, E0-mmr) no
document-level signal queries are issued, so `first_chunk` text must be obtainable from chunk-level
query responses. The filter therefore appears in the `bool.filter` of every signal query sent to
Elasticsearch, ensuring `GroupedChunks.first_chunk` can always be populated regardless of which
signals are active.

**Document sparse signal (`bool/should` sum, 0 inner hits):**
```json
{
  "size": 25,
  "_source": ["document_artifact_id", "control_number", "title", "summary", "topic"],
  "query": { "bool": {
    "filter": [ /* same filters + first_chunk inner_hits clause as above */ ],
    "should": [
      { "sparse_vector": { "field": "topic_sparse",   "query_vector": {"token_a": 1.23} } },
      { "sparse_vector": { "field": "summary_sparse", "query_vector": {"token_a": 1.23} } },
      { "sparse_vector": { "field": "title_sparse",   "query_vector": {"token_a": 1.23} } }
    ],
    "minimum_should_match": 1
  } }
}
```
ES sums the per-field sparse scores into a single document score. `title_sparse` is included
only when `title_enabled` is true.

**Scoped follow-up (chunk signals filtered to metadata_only_docs, Option 1):**
```json
{
  "size": 100,
  "_source": ["document_artifact_id", "control_number", "title", "summary", "topic"],
  "query": { "bool": {
    "filter": [ /* subset/date/email/size filters */,
      { "terms": { "document_artifact_id": [/* metadata_only_docs */] } },
      { "nested": { "path": "chunks", "query": { "term": { "chunks.chunk_index": 0 } },
        "inner_hits": { "name": "first_chunk", "size": 1, "_source": { "includes": ["chunks.text"] } } } }
    ],
    "must": [ { "nested": {
      "path": "chunks",
      "query": { "match": { "chunks.text": "<q>" } },
      "score_mode": "max",
      "inner_hits": { "name": "ranked_chunks", "size": 50, "_source": { "includes": ["chunks.chunk_index", "chunks.text", "chunks.leading_overlap_chars"] } }
    } } ]
  } }
}
```
Issued as two queries (BM25 and kNN), **only when `metadata_only_docs` is non-empty**.
The `terms` filter is placed in `bool.filter` (not `post_filter`) so only the target documents
are scored, not merely filtered after scoring.

**ES-side RRF (E2c — all four sub-retrievers):**
```json
{
  "retriever": { "rrf": { "rank_window_size": 100, "retrievers": [
    { "standard": { "query": { "nested": { "path": "chunks", "query": { "match": { "chunks.text": "<q>" } }, "score_mode": "max", "inner_hits": { "name": "bm25_chunks", "size": 50 } } } } },
    { "knn": { "field": "chunks.embedding", "query_vector": [/*384*/], "k": 100, "num_candidates": 250, "inner_hits": { "name": "knn_chunks", "size": 50 } } },
    { "standard": { "query": { "combined_fields": { "query": "<q>", "fields": ["topic", "summary", "title"] } } } },
    { "standard": { "query": { "bool": { "should": [ { "sparse_vector": { "field": "topic_sparse", "query_vector": {/*...*/} } }, { "sparse_vector": { "field": "summary_sparse", "query_vector": {/*...*/} } } ] } } } }
  ] } }
}
```
ES fuses at the document level. Client reads named `inner_hits` from returned documents.

Full query catalogue: `08-index-design-and-ingestion.md` §13.1–§13.9.

---

## 4. Held-Constant Capabilities

The following capabilities are **not** experiment variables. They are baseline infrastructure
present in every experiment configuration.

### 4.1 Metadata filtering

| Filter | Production (MCP → qna-service) | New tools (direct ES) |
|---|---|---|
| Date range | `dateFrom`/`dateTo` → `metadata.primaryDateTime` range | `primary_date_time` range query |
| Email participants | `emailParticipants` → wildcard on `metadata.email*` | `bool` with `term`/`wildcard` on `email_from`, `email_to`, `email_cc`, `email_bcc` |
| Subset scoping | `subsetId` → `subsetIds` term filter | `subset_ids` term filter |

The new tools must implement equivalent semantics: direction rules (`AtoB`, `BtoA`,
`AnyDirection`), match strategy (`Any`, `All`), presence rules (`Both`, `Either`). These map
directly to `bool` query composition over the index's first-class parent keyword fields.

Source: `tool_definitions.py` `EmailParticipantsFilter`; `03-retrieval-strategies.md` §5.

### 4.2 Virtual file tools

`WriteFile` and `ReadFile` support the `notes.txt` / `signature.txt` workflow in the 013.toml
system prompt. These are unchanged across all experiments.

### 4.3 Agent graph structure

The LangGraph topology is held constant:

```
START → call_llm →[tool_calls?]→ call_tools → call_llm → ... → structured_output → clean_markdown → END
```

Iteration policy: forced tool use on iteration 0; auto on 1–4; none on 5+.

Source: `rag_agent.py` lines 890–925.

---

## 5. Experiment Dimensions

The experiment set is organised into three tiers. Each tier adds capability over the previous
one. Within each tier, individual experiments vary one dimension at a time against a designated
anchor, enabling clean attribution of any quality change.

### 5.1 Tier overview

| Tier | Description | Output format | Reasoning effort |
|---|---|---|---|
| **Tier 0** | Production parity: reproduce current qna-service behaviour on the nested index | Flat chunk list with first-chunk decoration | `low` |
| **Tier 1** | Metadata-enriched retrieval: add document-level signals, keep flat output | Flat chunk list with first-chunk decoration | `low` |
| **Tier 2** | Nested document format: full document groups with concatenated adjacent chunks | Nested doc groups | `low` or `medium` |

First-chunk decoration (adding chunk 0 per returned document to the output, matching qna-service
behaviour) applies to **Tier 0 and Tier 1 only**. It is an output-only step that does not affect
ranking or fusion. Tier 2 uses adjacent-chunk concatenation instead.

The first-chunk support is split across two branches:
- **Root branch (`DSAS-2836/experiments`):** the `first_chunk` named inner_hits filter (size 1,
  `chunk_index = 0`) is present in all document-level signal queries. The root branch reads the
  resulting chunk 0 text and stores it in `GroupedChunks.first_chunk` — the field used by the
  eval scorers and XML serialiser.
- **Dedicated Tier 0/1 plan/branch:** adds the **decoration** step — forcing chunk 0 into each
  group's `retrieved_chunks` list even when it was not selected by fusion. This is the only part
  deferred to that branch.

### 5.2 Variable dimensions

| Dimension | Tier 0 | Tier 1 | Tier 2 |
|---|---|---|---|
| **Retrieval signals** | 2 chunk-level: BM25-chunks + kNN-chunks | 4 signals: 2 chunk-level (BM25-chunks + kNN-chunks) + 2 document-level (BM25-parent + sparse-parent) | 4 signals (always ON): same 2 chunk-level + 2 document-level |
| **Metadata for generation** | OFF | OFF | ON or OFF |
| **Fusion location** | Client-side RRF | Client-side RRF | Client-side RRF or ES-side RRF |
| **Final selection method** | RRF or MMR | RRF or MMR | RRF or MMR |
| **Chunk count** | 25 | 25 | 25 or 60 |
| **Reasoning effort** | `low` | `low` | `low` (E2a-low) or `medium` (E2a-med, E2c, E2d, E2d-nometa, E2e) |
| **Hop strategy** | Multi-hop | Multi-hop | Single-hop (E2a-low, E2e) or Multi-hop |
| **Title in retrieval** | N/A | EMC2: ON / Mallinckrodt: OFF | EMC2: ON / Mallinckrodt: OFF |
| **5 MB size filter** | ON | ON | **OFF** |

### 5.3 Notes on design choices

**Reasoning effort in Tier 2.** Most Tier 2 experiments use `reasoning_effort: medium` because
the nested document format sends significantly more structured information to the LLM (metadata
per document group plus concatenated adjacent chunks), and medium reasoning is expected to be
necessary to fully exploit that richer context. However, E2a-low deliberately uses `low`
reasoning and single-hop — it is the minimum viable Tier 2 configuration, testing whether the
nested structure alone (with no metadata visible to the LLM) adds value over the flat output at
the cheapest operational settings. The E2a-low → E2a-med comparison captures the combined
value of switching to medium reasoning and multi-hop. (Mechanism: see §6.5.5–6.5.6.)

**Metadata retrieval always ON in Tier 2.** Once the nested document structure is adopted,
excluding parent-field signals would waste the indexed metadata. All Tier 2 experiments use
the full four-signal set; the value of metadata retrieval signals is assessed at Tier 1
(E0 vs E1). (Mechanism: see §6.5.3.)

**Metadata for generation tested at both chunk counts.** E2a-low (25 chunks, low reasoning)
has metadata gen OFF, and E2d-nometa (60 chunks, medium reasoning) also has metadata gen OFF.
Together they reveal whether the value of exposing document metadata to the LLM depends on
how much raw chunk context the model already has.

**Title toggle.** The EMC2 workspace has meaningful document titles (email subjects, file
names); the Mallinckrodt corpus has less reliable titles. Title is therefore included as a
retrieval signal for EMC2 but excluded for Mallinckrodt. This is implemented as a per-eval-set
configuration flag (`title_enabled`) passed to the tool at runtime, not as a separate experiment
dimension. When included, `title` contributes with equal weight alongside `summary` and `topic`
in `combined_fields` and its corresponding `title_sparse` field is added to the sparse `bool/should`.

**Client-side vs ES-side fusion.** In the client-side path, four independent ES queries are
issued (two chunk-level with `inner_hits`, two document-level with 0 inner hits), followed by
a conditional scoped follow-up for documents found only by metadata signals. Client-side code
forms a unified chunk pool, assigns ordinal ranks per signal (document-level ranks are broadcast
to every chunk of the matched document; chunks outside a document-level top-`N_doc` receive a
fallback rank of `last_ranked_position + 1`), and fuses via ordinal RRF across all four ranks.
Metadata-only documents — those matched by document-level signals whose chunks were not surfaced
by chunk-level signals — can surface through the conditional scoped follow-up, which is bounded
to `N_doc × inner_hits.size` additional candidates.

In the ES-side path (E2c), a single `rrf` retriever over all four sub-retrievers fuses at the
**document level** (each document is scored by its best inner chunk via `score_mode: max`); the
client reads named `inner_hits` from returned documents without cross-document chunk re-ranking.
ES-side fusion has no MMR variant.

**Large-document-rank broadcast.** Because all chunks of a document receive the same
document-level ordinal rank, a document that is a strong metadata match and has many chunks
may contribute several chunks near the top of the fused ranking, mildly reducing document
diversity in the final result set. This effect is accepted during the experimental phase;
ablation in the follow-up grid can quantify its impact (see §9.6).

**5 MB size filter in Tier 2.** The 5 MB size filter (`workspace_extracted_text_size <= 5,242,880`)
is applied in Tier 0 and Tier 1 to match current production behaviour. It is disabled for all
Tier 2 experiments so that larger documents — which may contain the most relevant context for
complex questions — are not excluded from retrieval. (Mechanism: see §6.5.3.)

**MMR as an alternative selection method.** A parallel branch (E0-mmr, E1-mmr, E2a-med-mmr)
replaces the client-side RRF fusion step with Maximum Marginal Relevance (MMR) selection. This
is an orthogonal dimension that spans tiers: instead of fusing the per-signal ranked lists by
reciprocal rank, MMR encodes each candidate's chunk text client-side using the dense E5 model
(`intfloat/multilingual-e5-small`) with the `"passage: "` prefix to obtain a dense vector.
`chunks.embedding` is excluded from `_source` at index time and cannot be retrieved regardless
of the `_source.includes` projection, so client-side passage encoding is the only viable path.
The query vector is always encoded client-side with the `"query: "` prefix. MMR then greedily
selects chunks that are relevant to the query while penalizing redundancy with already-selected
chunks (`λ·sim(q,d) − (1−λ)·max_s sim(d,s)`, λ = 0.5, mirroring the qna-service production
MMR strategy). The branch isolates the selection method while holding signals, output format,
reasoning, and hop strategy identical to each experiment's RRF counterpart.

**Why the index design drives retrieval architecture.** The following design decisions in this
report are direct consequences of the nested index type chosen in
`08-index-design-and-ingestion.md` (§2). Understanding the constraint is useful for interpreting
the pipeline described in §7.4.

*Nested query mechanics.* The index stores one ES document per Relativity document, with chunks
as a `nested` array. Nested queries therefore rank **documents**, not chunks. Each document is
scored according to its chunk matches via `score_mode`. Available modes: `max` — document score
= its highest-scoring chunk (used throughout this report; appropriate for RAG because a document
is relevant if *any* passage matches); `avg` — average over all matching chunks (rewards
consistently relevant documents); `sum` — sum of all matching chunk scores (rewards documents
with many relevant passages); `min` — document score = its weakest matching chunk (most
conservative); `none` — chunk scores do not propagate to the parent (used for filtering only).
The returned unit is a parent document with `inner_hits` — a per-document, score-ordered, capped
subset of its matching chunks (`inner_hits.size` bounded by the index setting
`max_inner_result_window = 100`, so at most 100 chunks per document per signal reach the
client). There is no native ES mechanism to produce a global cross-document chunk ranking in a
single query; the client must flatten and re-sort `inner_hits` across all returned documents —
provably complete under `score_mode: max` (see `08-index-design-and-ingestion.md` §13.5).
Document-level fields (`title`, `summary`, `topic`, and their sparse counterparts) are queried
directly at the parent level and produce a **document ranking**, not a chunk ranking; their
contribution to chunk-level fusion is mediated by broadcasting each document's rank to all its
chunks, as described in §7.4.

*Alternative index designs (terse; see `08-index-design-and-ingestion.md` §2 for the full
evaluation).* **Flat denormalized** (one ES document per chunk, parent metadata copied onto
every chunk) is the current production model. Chunks are first-class ES documents, so native
global chunk ranking is trivial — no client-side flatten, no broadcast, no inner_hits cap — and
a single `multi_match` or hybrid retriever scores each chunk on both text and metadata in one
round-trip. Adjacent-chunk concatenation is equally straightforward: each chunk document carries
`document_artifact_id`, `chunk_index`, and `leading_overlap_chars`, so the client groups by
document, sorts by index, and concatenates consecutive runs — the same client-side step required
in the nested model, adding no meaningful complexity. The primary cost is metadata duplication
across all chunks (storage and update overhead); document grouping for presentation requires
`collapse` or an application-side step. **`join` field (parent-child):** parents and children
are separate ES documents linked by a declared relation and must be routed to the same shard.
Children update independently. ES provides `has_parent`/`has_child` queries but explicitly
discourages this pattern: it is 5–10x slower than nested at query time, imposes a single-shard
routing constraint, and does not compose with kNN or `rrf` retrievers. Not viable for this use
case.

*Trade-off summary.* The nested model was chosen because it eliminates metadata duplication
(parent fields stored once), keeps chunks atomically co-located with their parent, preserves the
document as the citation and grouping unit (natural for e-discovery), and composes cleanly with
parent-level pre-filters. The cost — global chunk ranking and metadata-to-chunk contribution
both require client-side orchestration — is the direct source of the Phase 1–2–3 pipeline
architecture described in §7.4.

---

## 6. Experiment Matrix

### 6.1 Experiment definitions

**Tier 0 — Baseline reproduction**

**E0** — Production parity on nested index.

Run two chunk-level ES queries independently (nested BM25 on `chunks.text` and nested kNN on
`chunks.embedding`, both `score_mode: max`), flatten the returned `inner_hits` into a global
chunk pool, apply client-side ordinal RRF, take top 25 by fused score. No document-level
metadata signals; no scoped follow-up. Filter `byte_size <= 5 MB`. Chunk 0 of each
represented document is added to the output group as a decoration step (first-chunk algorithm,
Tier 0/1 only; does not affect ranking). Uses the production config (`rag_agent_v3/013.toml`,
version 3.13) with no prompt changes. Output is a flat grouped-chunk list.

*Purpose:* confirm the new direct-ES Python tool reproduces current qna-service behavior.

---

**Tier 1 — Metadata-enriched retrieval (flat output)**

**E1** — Add parent-field retrieval signals, keep flat output.

Same as E0 plus two document-level signals: `combined_fields` on `topic`/`summary`/`title`
(equal weights; title when `title_enabled`) and a sparse-sum `bool/should` on the corresponding
`*_sparse` fields. Four ES queries are issued: two chunk-level (with `inner_hits`) and two
document-level (0 inner hits, top 25 documents each). A conditional scoped follow-up retrieves
chunk-level scores for any documents matched only by the document-level signals. All four
signal ranks are broadcast/assigned per chunk (document-level ranks broadcast to all chunks of
the matched document; fallback rank `last+1` for unmatched documents). Ordinal RRF fuses the
four per-chunk ranks; top 25 chunks are selected. Filter `byte_size <= 5 MB`. Title signals
enabled for EMC2, omitted for Mallinckrodt. Output is a flat chunk list with first-chunk
decoration (Tier 1); the LLM does not see document metadata directly.

*Purpose:* does adding document-level metadata signals improve chunk selection quality?

---

**Tier 2 — Nested document format**

All Tier 2 experiments share a common output format, retrieval configuration, citation scheme,
prompt structure, and code base. See **§6.5** for the complete shared design. What varies per
experiment (metadata gen, reasoning, hops, chunk count, fusion) is noted in each definition
below and summarised in §6.2.

**E2a-low** — Minimum viable Tier 2 configuration.

Retrieval: all four signals (2 chunk-level + 2 document-level), client-side ordinal RRF,
top 25 chunks including conditional scoped follow-up. Generation context: document groups with
metadata fields **omitted** (metadata gen OFF). `reasoning_effort: low`. Single-hop (exactly
one retrieval iteration). Title signals enabled for EMC2, omitted for Mallinckrodt.

*Purpose:* does the nested document structure alone — without metadata visible to the LLM,
without medium reasoning, without multi-hop — beat flat output? Establishes the minimum
nested baseline and enables a clean E1 vs E2a-low comparison (both low reasoning, both
metadata gen OFF, same chunk count; only output format differs — flat vs nested).

**E2a-med** — Full nested anchor.

Same as E2a-low except: metadata gen **ON** (title/summary/topic visible to LLM),
`reasoning_effort: medium`, multi-hop (1–3 iterations). This is the Tier 2 anchor for all
medium-reasoning comparisons.

*Purpose:* first test of the full nested pipeline; anchor for E2c, E2d, E2d-nometa, E2e.
The E2a-low → E2a-med comparison captures the combined value of adding metadata to generation,
upgrading to medium reasoning, and switching from single-hop to multi-hop.

**E2c** — Same as E2a-med but ES-side fusion.

A single ES `rrf` retriever over all four sub-retrievers (nested BM25 chunks, nested kNN
chunks, `combined_fields` parent BM25, sparse-sum parent) fuses at the **document level** (each
document scored by its best inner chunk via `score_mode: max`). The client reads named
`inner_hits` from the returned documents and groups/concatenates. No cross-document global chunk
re-ranking by the client; no scoped follow-up needed (all signals travel together).

*Purpose:* isolates fusion location (client-side global chunk ranking with document-level
metadata broadcast vs ES-side document ranking).

**E2d** — Same as E2a-med but 60 chunks.

`result_count` = 60. All depth parameters scale: `inner_hits.size` = 100 (= `min(100, 2 × 60)`),
`N_doc` = 60. Scoped follow-up uses the same `inner_hits.size` = 100.

*Purpose:* isolates the effect of providing more retrieved context to the LLM.

**E2d-nometa** — Same as E2d but metadata generation OFF.

Document groups are returned without `title`/`summary`/`topic` in the XML; the LLM sees only
chunk text and chunk count is 60.

*Purpose:* at 60 chunks, does metadata gen add value, or does raw passage volume suffice?
Pairs with E2d (one dimension change) and cross-pairs with E2a-low (both metadata gen OFF,
different reasoning/hops/chunk count).

**E2e** — Same as E2d but single-hop with parallel tool calls.

The LLM is constrained to exactly one retrieval iteration (`tool_choice="none"` after
iteration 0). The agent may issue multiple parallel tool calls in that single iteration.
Chunk count: 60, metadata gen ON.

*Purpose:* isolates hop strategy at 60 chunks — does multi-hop reasoning add value beyond
what a single richer retrieval provides?

---

**MMR branch — alternative final selection**

These three experiments form a parallel branch that replaces the client-side ordinal RRF fusion
step with MMR selection (λ = 0.5). The candidate pool is assembled identically to the RRF
counterpart (same four-signal queries, same conditional scoped follow-up, same ordinal rank
assignment). MMR then selects the final top-k over the pooled chunks using embedding similarity,
with candidate vectors encoded client-side from each candidate's `chunks.text` using the E5
model with `"passage: "` prefix (`chunks.embedding` is not retrievable from `_source`; see §7.4).
Everything else is held identical to the named RRF counterpart, making each pair a clean
single-dimension (RRF vs MMR) swap.

**E0-mmr** — E0 with MMR instead of RRF (baseline of the MMR branch).

Two chunk-level signals (BM25-chunks, kNN-chunks) build the candidate pool; MMR selects the
final 25 chunks. Flat output, low reasoning, multi-hop. Identical to E0 otherwise.

*Purpose:* MMR-branch baseline; isolates RRF vs MMR with chunk-only signals.

**E1-mmr** — E1 with MMR instead of RRF.

All four signals (2 chunk-level + 2 document-level) build the candidate pool via the same
pipeline as E1 (including the conditional scoped follow-up); MMR selects the final 25 chunks.
Flat output, low reasoning, multi-hop. Identical to E1 otherwise. Title signals enabled for
EMC2, omitted for Mallinckrodt.

*Purpose:* does MMR's diversity-aware selection beat RRF once parent-metadata signals are added?

**E2a-med-mmr** — E2a-med with MMR instead of RRF.

All four signals build the candidate pool; MMR selects the final 25 chunks. Nested document
output, metadata gen ON, medium reasoning, multi-hop. Identical to the Tier 2 anchor E2a-med
otherwise. E2a-med is chosen as the Tier 2 base because it is the designated anchor (cleanest
single-dimension comparison) and at 25 chunks the selection method has maximum leverage over
what the LLM sees.

*Purpose:* does MMR selection help in the full nested pipeline?

### 6.2 Full experiment table

| ID | Tier | Metadata retrieval | Metadata gen | Fusion location | Chunks | Reasoning | Hop strategy | 5 MB filter |
|---|---|---|---|---|---|---|---|---|
| **E0** | 0 | OFF | OFF | Client-side RRF | 25 | low | Multi-hop | ON |
| **E1** | 1 | ON | OFF | Client-side RRF | 25 | low | Multi-hop | ON |
| **E2a-low** | 2 | ON | **OFF** | Client-side RRF | 25 | **low** | **Single-hop** | **OFF** |
| **E2a-med** | 2 | ON | ON | Client-side RRF | 25 | medium | Multi-hop | **OFF** |
| **E2c** | 2 | ON | ON | **ES-side RRF** | 25 | medium | Multi-hop | **OFF** |
| **E2d** | 2 | ON | ON | Client-side RRF | **60** | medium | Multi-hop | **OFF** |
| **E2d-nometa** | 2 | ON | **OFF** | Client-side RRF | **60** | medium | Multi-hop | **OFF** |
| **E2e** | 2 | ON | ON | Client-side RRF | **60** | medium | **Single-hop** | **OFF** |
| **E0-mmr** | 0 | OFF | OFF | **Client-side MMR** | 25 | low | Multi-hop | ON |
| **E1-mmr** | 1 | ON | OFF | **Client-side MMR** | 25 | low | Multi-hop | ON |
| **E2a-med-mmr** | 2 | ON | ON | **Client-side MMR** | 25 | medium | Multi-hop | **OFF** |

Bold values mark the dimension(s) that differ from E2a-med (Tier 2 anchor), E0 (Tier 0), or
E1 (Tier 1) in the relevant comparison. For the MMR branch, the bold "Client-side MMR" marks
the only difference from each experiment's RRF counterpart (E0, E1, E2a-med).

### 6.3 Comparison map

| Comparison | Dimension isolated | Question answered |
|---|---|---|
| E0 vs E1 | Metadata retrieval signals | Does adding BM25 + sparse on parent fields improve chunk selection? |
| E1 vs E2a-low | Output format (flat → nested) | Does the nested structure alone improve answers? (both low reasoning, metadata gen OFF, same chunk count; hops differ) |
| E2a-low vs E2a-med | Metadata gen + reasoning + hops | What is the full upgrade from minimum to full nested config worth? |
| E2a-med vs E2c | Fusion location | Is client-side global chunk ranking better than ES-side document ranking? |
| E2a-med vs E2d | Chunk count (25 → 60) | Does more retrieved context improve answers? |
| E2d vs E2d-nometa | Metadata for generation at 60 chunks | At 60 chunks, does exposing document metadata to the LLM still add value? |
| E2d vs E2e | Hop strategy at 60 chunks | Does multi-hop reasoning beat single-hop retrieval of 60 chunks? |
| E0 vs E0-mmr | Final selection (RRF vs MMR) | Does MMR beat RRF with chunk-only signals? |
| E1 vs E1-mmr | Final selection (RRF vs MMR) | Does MMR beat RRF with metadata signals (flat output)? |
| E2a-med vs E2a-med-mmr | Final selection (RRF vs MMR) | Does MMR beat RRF in the full nested pipeline? |

### 6.4 Execution order

**Starting point: E1.** E0 is deferred — it may be executed by a colleague, or run last as a
retrospective production-parity check. The evaluation sequence therefore starts at E1.

E1 validates the four-signal flat retriever and establishes the metadata-retrieval baseline.
E2a-low is cheap (low reasoning, single-hop) and runs next — it validates the nested pipeline
quickly before committing to medium-reasoning runs. E2a-med establishes the Tier 2 anchor;
E2c, E2d, E2d-nometa, E2e then run in parallel.

```
E1 → E2a-low → E2a-med → E2c, E2d, E2d-nometa, E2e (parallel)
[E0 deferred: run last, or by a colleague]
```

The MMR branch runs after its RRF counterparts so each pair can be compared directly:

```
E1-mmr → E2a-med-mmr
[E0-mmr deferred alongside E0]
```

### 6.5 Tier 2 — shared design (applies to all Tier 2 experiments)

> **Scope.** Every item in this section applies exclusively to Tier 2 (E2a-low, E2a-med, E2c,
> E2d, E2d-nometa, E2e, E2a-med-mmr). Tier 0 and Tier 1 are unaffected. What varies *between*
> Tier 2 experiments (metadata gen, reasoning effort, hop count, chunk count, fusion location) is
> captured in §6.2 and §8.2.

#### 6.5.1 Output format

Tier 2 returns `<document_group>` XML elements instead of the flat `<grouped_chunks>` used in
Tier 0/1. Each group corresponds to one retrieved document and contains one or more `<chunk>`
elements produced by adjacent-chunk concatenation (see §6.5.2). No first-chunk decoration
is applied in Tier 2 — chunk 0 is included only if it was selected by fusion.

**XML delivery to the LLM (Tier-2 base branch).** The `<document_group>` XML is built by
`retrieve_nested` / `_build_nested_output` inside `air_assist_experiments` and returned as the
`content` text of the `ToolCallResult`. The v3 graph (`_get_documents`) uses this provider text
directly as the tool-call message delivered to the LLM — it does **not** re-serialise via
`GroupedChunks.to_xml()`. As a consequence, `GroupedChunks.to_xml()` on the Tier-2 base branch
has also been simplified: it emits all selected chunks as plain `<chunk>` elements in start-index
order with no `<first_chunk>` block. This ensures the snippet-generation and grounding contexts
(which do call `to_xml()` internally) are also decoration-free and consistent with the LLM-facing
output.

**Metadata gen ON** (E2a-med, E2c, E2d, E2e, E2a-med-mmr): parent fields `title`, `summary`,
`topic` are emitted inside the group (`title` only for indices with `title_enabled = true`,
e.g. EMC2; omitted for Mallinckrodt). `control_number` and `primary_date_time` are never
emitted in the LLM-facing XML (`control_number` is carried in the structured output for eval
scorers; `primary_date_time` is used only for ES filtering).

**Metadata gen OFF** (E2a-low, E2d-nometa): parent fields are omitted from the XML entirely.

Canonical output structure:
```xml
<!-- metadata gen ON -->
<document_group>
  <doc_id>{document_artifact_id}</doc_id>
  <title>{title}</title>       <!-- omitted if title_enabled=false or metadata gen OFF -->
  <summary>{summary}</summary> <!-- omitted if metadata gen OFF -->
  <topic>{topic}</topic>       <!-- omitted if metadata gen OFF -->
  <chunk><chunk_id>1:3</chunk_id><content>{concatenated text}</content></chunk>
  <chunk><chunk_id>5</chunk_id><content>{single chunk text}</content></chunk>
</document_group>
```

#### 6.5.2 Adjacent-chunk concatenation and range chunk IDs

Within each document group, detect runs of consecutive `chunk_index` values. For a run
`[i, i+1, ..., j]`: start with chunk `i`'s full text; for each subsequent chunk strip the
first `leading_overlap_chars` characters before appending (removes duplicated overlap without
losing unique content).

**Chunk ID scheme (Tier 2 only):** a concatenated run is assigned the range-style identifier
`"i:j"` (e.g. `"1:3"`). A single non-concatenated chunk retains its plain integer identifier
(e.g. `"5"`). Because the identifier can be a range string, `Document.chunk_id` and
`Snippet.chunk_id` are **string-typed** in Tier 2 code paths (not integers as in Tier 0/1).

**Citation format (Tier 2 only):** `[doc_id-1:3]` for a concatenated run; `[doc_id-5]` for a
single chunk. In-agent citation-validation regexes and the r1-evals citation scorers must
accept the range form for Tier 2 traces. Tier 0/1 continue to use plain integer chunk IDs and
`[doc_id-N]` citations; those scorers and regexes are unaffected.

#### 6.5.3 Retrieval configuration

All Tier 2 experiments use:
- **All four retrieval signals** (BM25-chunks, kNN-chunks, BM25-parent, sparse-parent) — metadata
  retrieval is always ON. The value of metadata retrieval signals is assessed at Tier 1 (E0 vs E1).
- **5 MB size filter OFF** — larger documents are not excluded from retrieval.
- **`apply_size_filter=False`** passed to `retrieve_nested`; `include_metadata` and
  `result_count` vary per experiment (see §6.2).

#### 6.5.4 Hybrid retrieval and query guidance

Every query string issued by the agent is matched **simultaneously** by all active signals:
lexical BM25 on chunk text, dense kNN on chunk embeddings, lexical `combined_fields` BM25 on
parent metadata, and sparse-vector matching on parent metadata. The tool backend is **not**
keyword-only — production tool docstrings that claim "keyword search (e.g., BM25), NOT semantic
vectors" are incorrect for these experiments.

Consequently, Tier 2 system prompts correct this guidance:
- Inform the model that results combine lexical and semantic matching across the query.
- **Single-hop experiments** (E2a-low, E2e): instruct the model to issue several varied parallel
  queries in the single allowed iteration — mixing keyword-dense and natural-language/semantic
  phrasings maximizes recall across all signal types in one round.
- **Multi-hop experiments**: instruct the model to mix query phrasings within each hop and to
  use subsequent hops for following leads discovered in results, not for switching retrieval
  modality (each hop already covers all modalities).

Tier 0/1 deliberately keep `013.toml` unchanged — the keyword framing is retained as an experimental
control and the retrieval backend matches the production qna-service behaviour the prompt was
written for.

#### 6.5.5 TOML configs and base branch

All Tier 2 TOML configs are placed under
`air_assist_core/registry/configs/DSAS-2836/` and auto-discovered by the registry via
`rglob("*.toml")`. Version numbers 92–93 are used (no collision with existing configs in
`rag_agent_v3/`, range 8–24).

Tier 2 experiment branches share a **common Tier-2 base branch** (`DSAS-2836/tier2-base`) that
sits between the root `DSAS-2836/experiments` and individual experiment branches.

The nested `<document_group>` output (`retrieve_nested`) and the range-style `"i:j"` chunk-ID
rendering already live on **root**. Placing them there is harmless to Tier 0/1 — those tiers use
the flat output path and never invoke the nested retriever — and is warranted because the nested
output is common to all Tier 2 experiments and should be available as early as possible.

`DSAS-2836/tier2-base` adds only the Tier-2-specific changes layered on top of root's nested
retriever: provider routing from flat to nested, string-typed `Document.chunk_id`/`Snippet.chunk_id`
with matching citation-validation regex and r1-evals scorer updates (so citations carry the range
form end-to-end), and the config-driven hop cap. Each experiment branch then adds only its
per-experiment TOML and runner wiring.

#### 6.5.6 Config-driven hop cap (single-hop experiments)

For single-hop experiments (E2a-low, E2e), `tool_choice` is capped at `"none"` after iteration 0
via a TOML-configurable field read by `get_tool_choice()` in `rag_agent.py`. The field defaults
to the current multi-hop behaviour so all other experiments are unaffected. This replaces
per-branch `rag_agent.py` edits — the cap is expressed entirely in the TOML and the base-branch
`rag_agent.py` reads it.

---

## 7. Tool Implementation Design

### 7.1 Architecture: direct ES from Python

New tools query Elasticsearch directly using `elasticsearch-py`, bypassing MCP and qna-service.
This gives full control over query composition, field selection, and fusion strategy.

```
air-assist-agent (LangGraph v3)
    │
    ▼
New tool (Pydantic model + async handler)
    │  routed by ExperimentToolProvider (eval runner selects flat vs nested)
    ▼
elasticsearch-py Elasticsearch client
│  Phase 1 — four concurrent ES queries:
│    ├─ chunk BM25: nested match on chunks.text (score_mode max, inner_hits)
│    ├─ chunk kNN:  nested kNN on chunks.embedding (score_mode max, inner_hits)
│    ├─ doc BM25:   combined_fields over topic/summary/[title] (0 inner hits)
│    └─ doc sparse: bool/should over *_sparse fields (0 inner hits)
│  Phase 2 — conditional scoped follow-up (only if metadata_only_docs non-empty):
│    chunk BM25 + kNN filtered to document_artifact_id ∈ metadata_only_docs
│  Client: union pool → ordinal RRF or MMR → top result_count chunks
│  (text/overlap returned inline in Phase 1/2 inner_hits; MMR encodes candidate passages client-side; no separate fetch phase)
│  or single ES-side rrf retriever (E2c, no Phase 2)
    ▼
Elasticsearch (as-* nested index)
    │
    ▼
Flat chunk list + first-chunk decoration (Tier 0/1)
or Nested doc groups with adjacent-chunk concatenation (Tier 2) → XML for LLM
```

**Client lifecycle:** a single synchronous `Elasticsearch` instance (connection-pooled), created
at agent startup with bearer auth (CID token). Concurrent ES calls are issued via
`asyncio.to_thread`, sharing one client instance safely across threads.

**Query embedding:** the tool handler embeds the query string at call time using cached model
instances:
- Dense: `intfloat/multilingual-e5-small` with `"query: "` prefix → 384-dim vector
- Sparse: `opensearch-neural-sparse-encoding-v2-distill` → `{token: weight}` map

**Passage encoding (MMR branch only):** when `AIR_ASSIST_FUSION=mmr`, all candidates in the
pool are also encoded client-side using the same dense E5 model with the `"passage: "` prefix
(`DenseQueryEncoder.encode_passages`). `chunks.embedding` is excluded from `_source` at index
time and cannot be retrieved, making client-side encoding the only option.

Models are loaded lazily and cached for the process lifetime, mirroring the `E5Embedder` /
`SparseEmbedder` pattern in `es-index-explorer/indexing/embedding.py`.

### 7.2 Tool selection

The retrieval mode for each experiment is selected by which `ExperimentToolProvider` variant
`AirAssistRunner` instantiates in its `__init__` at process start — no environment variable is
required for this. `AirAssistRunner` is the runner class used by `air-assist-evals eval-flow`
(see §9.3). It receives the corpus from the `--dataset` CLI argument, maps it to the
appropriate index key (`emc2` or `mallinckrodt`), constructs `ExperimentToolProvider` with
that index config, and passes it to `create_agent()`.

The existing tool pair (`GetRelevantDocuments` + `GetRelevantDocumentsWithMetadataFilter`) is
presented to the LLM unchanged across all experiments. Behind these tool names, the provider
routes calls to the appropriate Python retriever, bypassing MCP entirely. No changes to
`tool_definitions.py`, `tool_selection.py`, or `rag_agent.py` are required.

| Retrieval mode | Retriever called | Used by |
|---|---|---|
| Flat — 2 chunk-level signals (BM25-chunks + kNN-chunks) | `handle_search_documents` (signals restricted) | E0 |
| Flat — 4 signals (2 chunk-level + 2 document-level) | `handle_search_documents` (default signals) | E1 |
| Nested — client-side RRF | `handle_search_documents_nested` | E2a-low, E2a-med, E2d, E2d-nometa, E2e |
| Nested — ES-side RRF | future `handle_search_documents_nested_es_rrf` | E2c |

**Final selection method (RRF vs MMR).** `AIR_ASSIST_FUSION` (default `rrf`) controls the
fusion step independently of the retrieval mode. Setting `AIR_ASSIST_FUSION=mmr` replaces
client-side RRF with MMR selection. The MMR experiments (E0-mmr, E1-mmr, E2a-med-mmr) reuse
their RRF counterparts' eval runners and configs with this env var added. ES-side fusion
(`nested_docs_es_rrf`) has no MMR variant.

### 7.3 Tool definitions

The existing tool pair (`GetRelevantDocuments` and `GetRelevantDocumentsWithMetadataFilter`)
is reused in all experiments. The LLM-facing interface, tool descriptions, and system prompt
guidance are identical to production. The retrieval strategy (signals, fusion, scope, chunk
count) is controlled by the `ExperimentToolProvider`, invisible to the LLM.

The `ExperimentToolProvider` intercepts calls to these tools by name and routes them to the
Python retriever, returning `GroupedChunks` structured content in the same format as the MCP
layer. Arguments are parsed from the MCP-format dict (e.g. `dateFrom` → `date_from`) and
passed to `handle_search_documents` or `handle_search_documents_nested`. When the flat
retriever is active the tool returns a flat chunk list (Tier 0/1 format); when the nested
retriever is active it returns nested document groups with concatenated adjacent chunks
(Tier 2 format). Metadata fields are included or omitted from the XML based on a
per-experiment flag (not exposed to the LLM as a tool parameter).

### 7.4 Post-retrieval processing

#### Candidate pool construction

Per tool call, retrieval proceeds in up to three phases.

**Phase 1 — four concurrent ES queries.**

1. **Chunk BM25 signal.** Nested `match` on `chunks.text`, `score_mode: max`, `size` = 100
   parent documents, `inner_hits.size` = `min(100, 2 × result_count)` (50 for 25-chunk
   experiments, 100 for 60-chunk experiments; bounded by the index setting
   `max_inner_result_window = 100`). Returns per-document inner hits sorted by descending
   chunk BM25 score. `inner_hits` field selection: `chunks.chunk_index`, `chunks.text`,
   `chunks.leading_overlap_chars`.

2. **Chunk kNN signal.** Nested kNN on `chunks.embedding`, `score_mode: max`, `k` = 100,
   `num_candidates` = 250, same parent `size` and `inner_hits.size` as above. Returns
   per-document inner hits sorted by descending cosine similarity.

3. **Document BM25 signal.** `combined_fields` over `topic`, `summary`, and (when
   `title_enabled`) `title`, all fields with equal weight. **0 inner hits.** Returns top
   `N_doc = result_count` parent documents by lexical score. A first-chunk nested filter
   (`chunk_index = 0`, named `first_chunk`, size 1) is included in `bool.filter` to enable
   the first-chunk decoration step in Tier 0/1 output.

4. **Document sparse signal.** One `bool/should` over `summary_sparse`, `topic_sparse`,
   and (when `title_enabled`) `title_sparse`; ES sums the per-field scores into a single
   document score. **0 inner hits.** Returns top `N_doc = result_count` parent documents
   by sparse score. Same first-chunk filter as signal 3.

All chunk-level inner-hit scores are globally comparable across documents because the index
has `number_of_shards: 1` (global IDF for BM25; cosine similarity is absolute for kNN).
See `08-index-design-and-ingestion.md` §13.5.

From the chunk-level signals, flatten all inner-hit `(document_artifact_id, chunk_index)`
pairs into the initial candidate pool. Let `C` denote the set of documents represented by
at least one chunk-level candidate, and `M` the union of documents from the document-level
top-`N_doc` results.

**Phase 2 — conditional scoped follow-up.**

Compute `metadata_only_docs = M \ C`. If this set is **non-empty**, re-run the same two
chunk-level queries (BM25 and kNN) with an added `terms` filter on
`document_artifact_id ∈ metadata_only_docs`. This recovers real BM25 and kNN chunk scores
for those documents (capped at `inner_hits.size` per document) and merges their chunks into
the global candidate pool. If `metadata_only_docs` is empty, this step is skipped entirely
(no extra round trip).

*Option 2 (optimization, not implemented by default):* a single `bool/should` with two named
nested inner_hits (BM25 and kNN) filtered to `metadata_only_docs`, discarding the parent
score. Per-chunk BM25 and kNN scores are **semantically identical** to Option 1; the
difference is one round trip instead of two. Implementing Option 2 requires additional code
for the combined query construction and two named inner_hits sets.

**Per-chunk ordinal scoring.**

Each pooled chunk `(document_artifact_id, chunk_index)` receives up to four ordinal ranks,
computed by global **dense_rank**: equal scores share the same position and there are no gaps
after a tie group (a group of N tied items at rank 1 is followed by rank 2, not rank N+1).

*Dense_rank vs standard rank.* Standard (competition) ranking also assigns equal ranks to
tied items, but leaves a gap after each tie group: N items tied at rank 1 are followed by
rank N+1, penalising items below any large tie group by pushing their rank number up
proportionally. Dense_rank avoids this gap, so items below a tie group receive a slightly
higher (better) rank and correspondingly higher RRF contribution. For the chunk-level signals
(BM25, kNN) ties on continuous floating-point scores are rare and the choice is practically
immaterial. For the document-level signals, where tied scores are more common (e.g., multiple
documents matching the same sparse tokens with similar weights), dense_rank distributes the
signal's influence more evenly across the ranking and gives metadata signals a slightly
stronger effect in fusion — which is the intended behaviour. The alternative (standard rank)
is more conservative, penalising items after large tie groups more heavily; it can be
substituted as a follow-up ablation parameter if the metadata signal contribution proves
too strong.

- **Chunk BM25 rank:** global ordinal rank across all chunks from Phase 1 and Phase 2 chunk
  BM25 queries, by BM25 score descending. A chunk absent from this signal → no contribution.
- **Chunk kNN rank:** same logic, by kNN score descending.
- **Document BM25 rank:** the rank of the chunk's parent document in the Phase 1
  `combined_fields` result set, **broadcast identically to every chunk of that document**.
  If the document is outside the document BM25 top-`N_doc` → fallback rank = `last_ranked_position + 1`.
- **Document sparse rank:** the rank of the parent document in the Phase 1 sparse-sum
  result set, broadcast to all its chunks. Same fallback logic.

*Note on document-rank broadcast.* Because all chunks of a document receive the same
document-level rank, a document that is a strong metadata match and has many chunks may place
several of them high in the fused ranking, mildly reducing document diversity. This is a
known effect, acceptable during the experimental phase; later ablation can assess its impact.

#### Fusion and selection

**Ordinal RRF (default).** For each candidate chunk, compute
`RRF(chunk) = Σ_r 1 / (k + rank_r(chunk))` using ordinal ranks, standard constant `k = 60`,
summing only over signals that contribute (no-contribution signals are skipped). Take the top
`result_count` chunks by fused score.

**Inline field projection.** `chunks.text` and `chunks.leading_overlap_chars` are included in
the `inner_hits._source` of the Phase 1 and Phase 2 chunk-level queries. There is no separate
fetch phase. Chunk payload per query is bounded: `inner_hits.size` ≤ 100 per document,
100 documents returned, so at most ~10 MB of text in the worst case across all signals —
comfortably within network limits.

#### Flat output (Tier 0, Tier 1)

1. Assemble the candidate pool (Phases 1–2), apply ordinal RRF, take top `result_count` chunks.
2. `chunks.text` and `control_number` are already available from Phase 1/2 `inner_hits._source`
   (inline projection; no separate fetch).
3. Group selected chunks by `document_artifact_id`.
4. **First-chunk decoration (Tier 0/1, output-only).** For each document that has at least one
   fusion-selected chunk (i.e., each document group already present in the output), chunk 0 is
   added to that group even if it was not itself selected by fusion — matching the qna-service
   production behaviour. Documents with no fusion-selected chunks are not represented in the
   output at all and therefore receive no decoration. The `first_chunk` named inner_hits filter
   (size 1, present in all document-level signal queries) supplies chunk 0 text: the root branch
   reads it and populates `GroupedChunks.first_chunk`. The step that **forces** chunk 0 into
   `retrieved_chunks` (making it appear in the output list even when not fusion-selected) is
   implemented on the dedicated Tier 0/1 plan/branch. Chunk 0 is **never a fusion candidate**
   and does not affect ranking in any way.
5. Serialize as flat `<grouped_chunks>` XML elements.

#### Nested output (Tier 2)

1. Assemble the candidate pool (Phases 1–2), apply ordinal RRF, take top `result_count` chunks.
2. `chunks.text`, `chunks.leading_overlap_chars`, `control_number`, and (when metadata gen ON)
   `title`, `summary`, `topic` are already available from Phase 1/2 `inner_hits._source` and
   parent `_source` (inline projection; no separate fetch).
3. Group selected chunks by `document_artifact_id`. No first-chunk decoration in Tier 2.
4. Determine which parent fields appear in the LLM-facing XML per group:
   - **LLM-facing XML** (`<document_group>` elements): carries `title` / `summary` / `topic`
     when metadata for generation is ON (`title` further gated by `title_enabled`: ON for
     EMC2, OFF for Mallinckrodt). When metadata for generation is OFF (E2a-low, E2d-nometa),
     these fields are omitted entirely.
   - **`control_number`** is carried in the structured `GroupedChunks` output alongside the
     XML (used by eval scorers and citation post-processing). It is not emitted in the
     LLM-facing XML.
   - **`primary_date_time`** is used solely for ES-side filtering (a `range` query clause)
     and is never serialized into the XML.
5. **Adjacent chunk concatenation.** Within each document group, detect runs of consecutive
   `chunk_index` values. For a run `[i, i+1, ..., j]`: start with chunk `i`'s full text;
   for each subsequent chunk strip the first `leading_overlap_chars` characters before
   appending (removes duplicated overlap without losing unique content). The chunk ID and
   citation format for the resulting passage are defined in §6.5.2.
6. Serialize as `<document_group>` XML elements (canonical structure in §6.5.1) and return
   the XML as the `ToolCallResult.content` text. The v3 graph delivers this text verbatim as
   the tool-call message to the LLM — see §6.5.1 for the XML delivery mechanism.

#### MMR selection (MMR branch)

When `AIR_ASSIST_FUSION=mmr`, the ordinal RRF fusion step is replaced by Maximum Marginal
Relevance selection:

1. Assemble the same candidate pool (Phases 1–2) as the RRF counterpart.
2. Obtain a dense vector for each candidate chunk. Every candidate originates from a chunk-level
   signal (Phase 1 chunk BM25/kNN or Phase 2 scoped follow-up), so every candidate has
   `chunks.text` available from the inline `inner_hits._source` projection. `chunks.embedding`
   is excluded from `_source` at index time and cannot be retrieved regardless of the
   `_source.includes` projection, making client-side encoding the only viable path. The query
   vector is always encoded client-side with the `"query: "` prefix.

   **Candidate ordering.** Before encoding, the candidate pool is sorted ascending by
   `(document_artifact_id, chunk_index)` — this is the deterministic MMR order used by
   `_fuse_with_mmr` and is not the raw Elasticsearch hit/inner-hit response order.

   **MMR passage-embedding cache.** To avoid redundant encoding across concurrent and sequential
   rubric-variant invocations that share the same Elasticsearch index, each raw candidate vector
   is memoized in a bounded, process-local synchronized LRU cache (`DenseQueryEncoder`) keyed by
   `(index_name, document_artifact_id, chunk_index)`. `index_name` is included so that the
   process-singleton encoder can be safely reused across multiple datasets (e.g. EMC2 and
   Mallinckrodt) in one evaluation run without key collisions. The cache is an in-process
   `OrderedDict`-backed true LRU: every hit promotes the entry to the most-recently-used
   position; every insertion places the new entry at the most-recently-used end; exceeding the
   capacity of 250,000 entries evicts the least-recently-used entry. The cache is empty at
   process start and expires when the evaluation process exits.

   **Batch hit/miss algorithm.** All cache operations for one MMR candidate batch are performed
   under a single `threading.RLock` critical section so that concurrent rubric variants cannot
   duplicate encoding work. Within the lock, the sorted candidate list is partitioned:

   - **Cache hits:** the compact float32 vector is read from the LRU and the entry is promoted
     (marked most-recently-used). The vector is placed at the candidate's original output
     position.
   - **Cache misses:** the key, raw text, and original output position are recorded; no encoding
     occurs yet.

   After partitioning, all miss texts are encoded in one batched `SentenceTransformer.encode`
   call with the `"passage: "` prefix, matching the encoding used before caching was introduced.
   Each resulting vector is stored as a compact `array('f')` (float32), inserted into the LRU,
   and placed at its recorded original output position. After the batch, hits and newly encoded
   misses are merged back index-for-index into the original sorted MMR candidate order. The
   returned sequence is identical to what `encode_passages` would produce for all candidates,
   preserving candidate-to-vector alignment and all MMR tie-breaking exactly.

   The cache does not change the candidate pool, the candidate order, the E5 model, the passage
   prefix, vector normalization, MMR selection, or λ. It is a pure performance optimization that
   has no effect on retrieval or fusion semantics.

3. Run MMR over the **full candidate pool** (MMR is greedy and must score all remaining
   candidates at each step): first select the chunk most similar to the query, then iteratively
   select the chunk maximizing `λ·sim(q,d) − (1−λ)·max_s sim(d,s)` until top-`result_count`
   chunks are chosen. λ = 0.5.
4. Feed the selected chunks into flat (Tier 0/1) or nested (Tier 2) output assembly.
   First-chunk decoration (Tier 0/1 only) applies identically to the RRF path.

#### ES-side RRF (E2c only)

For E2c, a single ES request with an `rrf` retriever over all four sub-retrievers is issued.
Each sub-retriever has its own named `inner_hits`. ES fuses at the **document level** (each
document is scored by its best inner chunk via `score_mode: max`) and the client reads
`inner_hits` from the returned documents — there is no cross-document global chunk re-ranking
by the client. The client groups each returned document's chunks and applies adjacent-chunk
concatenation as in the standard Tier 2 nested path. ES-side fusion has no MMR variant.

### 7.5 Metadata filter mapping

The new tools replicate the current qna-service metadata filter semantics against the nested
index's first-class parent fields:

| Current MCP parameter | New ES query |
|---|---|
| `dateFrom` | `{"range": {"primary_date_time": {"gte": "<dateFrom>"}}}` |
| `dateTo` | `{"range": {"primary_date_time": {"lt": "<dateTo + 1 day>"}}}` |
| `emailParticipants` (AtoB) | `bool.must`: `term` on `email_from` ∩ `term` on `email_to`/`cc`/`bcc` |
| `emailParticipants` (AnyDirection) | `bool.should` with both direction permutations |
| `emailParticipants` (Either) | `bool.should` with side A only OR side B only |
| `subsetId` | `{"term": {"subset_ids": "<subset>"}}` — always applied |

### 7.6 Per-experiment changes required

**Tier 0/1 experiment branches** (E0, E0-mmr, E1, E1-mmr) fork directly from the shared root
branch (`DSAS-2836/experiments`). Root provides all parameterised retriever methods (signals,
chunk count, size filter, fusion method), and also carries the nested `<document_group>` output
path (`retrieve_nested`) and range-style chunk-ID rendering — both of which are shared by all
Tier 2 experiments but are harmless to Tier 0/1 (those tiers use the flat output path only).

**Tier 2 experiment branches** (E2a-low, E2a-med, E2c, E2d, E2d-nometa, E2e, E2a-med-mmr)
fork from the Tier-2 base branch (`DSAS-2836/tier2-base`), which itself forks from root (see
§6.5.5). The base branch adds the Tier-2-specific changes on top of root. Each individual
experiment branch then adds only its per-experiment TOML and runner wiring.

The table below shows what must change in each experiment branch relative to its parent (root
for Tier 0/1; `tier2-base` for Tier 2).

| Dimension | Control mechanism | What changes per experiment branch |
|---|---|---|
| Retrieval signals (2 vs 4) | `signals: list[SignalType]` param on retriever (default: all 4 — 2 chunk-level + 2 document-level) | `tool.py` handler passes `[BM25_CHUNKS, KNN_CHUNKS]` for E0; default (all 4) for E1 and Tier 2 |
| Metadata for generation (ON/OFF) | `include_metadata: bool` param on `retrieve_nested` / `handle_search_documents_nested` (default `False`) — implemented on root branch | `tool.py` handler passes `include_metadata=True` for metadata-ON experiments (E2a-med, E2d, E2e, E2a-med-mmr) |
| Fusion location (ES-side RRF) | Separate `nested_docs_es_rrf` retriever module | `air_assist_experiments` new retriever (E2c branch only) |
| Final selection method (RRF/MMR) | `AIR_ASSIST_FUSION` env var (default `rrf`) | Env var only — no code change |
| Chunk count (25/60) | `result_count: int` method param (default 25) | `tool.py` handler passes override per experiment |
| 5 MB size filter (ON/OFF) | `apply_size_filter: bool` param on `retrieve_flat` / `retrieve_nested` | `tool.py` handler passes `apply_size_filter=False` for all Tier 2 experiments |
| Reasoning effort (low/medium) | TOML `reasoning_effort` field | New TOML file only — no code change |
| Hop strategy (multi/single) | TOML-configurable `max_tool_iterations` field read by `get_tool_choice()` in `rag_agent.py` (see §6.5.6) | TOML only — no per-branch `rag_agent.py` edit; all Tier 2 single-hop experiments set the field in their TOML |
| Title in retrieval | `title_enabled` in per-index config dict | Already implemented on root — no change |
| Tool registration + dispatch | `ExperimentToolProvider` injected by the eval runner (no code change in `air_assist_core`) | Eval runner script selects provider per experiment |

---

## 8. Config and Prompt Variations

### 8.1 TOML configs

**All experiments — including E0 — use the new nested index** (`air_assist_nested.json`
mapping) and the new direct-ES Python tools via `ExperimentToolProvider`. No experiment routes
through qna-service or MCP. E0's purpose is to confirm that the new tools produce results
equivalent to qna-service on the same two signals (BM25-chunks + kNN-chunks), establishing a
baseline before parent-field signals are introduced.

**E0 and E1** (and their MMR variants E0-mmr, E1-mmr) reuse `rag_agent_v3/013.toml`
(version 3.13) with **no changes**. The system prompt, tool descriptions, and agent behaviour
are identical to production. The only difference is the retrieval backend, which is swapped at
the eval runner level: `AirAssistRunner` instantiates `ExperimentToolProvider` instead of
`McpToolProvider` (see §7.2). No new TOML files are needed for Tier 0 or Tier 1.

Tier 2 experiments require new TOML files because the system prompt genuinely changes to
explain the nested document format to the LLM. These are placed under
`air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/configs/DSAS-2836/`
alongside the existing `rag_agent_v3/` folder. The registry discovers them automatically via
`rglob("*.toml")`. Version numbers 92–96 have no collision with any existing config (current
range is 8–24 in `rag_agent_v3/`).

**E2a-low** uses `DSAS-2836/092.toml` (version 3.92) with:
- `reasoning_effort: low`
- System prompt: nested format explanation (metadata gen OFF variant; see §6.5.1) +
  hybrid retrieval + single-hop parallel-query guidance (see §6.5.4)
- Single-hop: `max_tool_iterations = 1` (see §6.5.6)

Each Tier 2 experiment is fully identified by its own TOML file; no CLI switches are needed
for retrieval or generation parameters. The TOML schema is extended with three optional fields
(defaults in parentheses):

- `max_tool_iterations` (`None` — unlimited, i.e. multi-hop by default; see §6.5.6)
- `include_metadata` (`false` — metadata fields omitted from LLM-facing XML)
- `result_count` (`None` — defers to `EXPERIMENT_CONFIG["retrieval"]["result_count"]` = 25)

Retrieval signals and fusion method are not TOML fields; signals default to all four active and
fusion is set via the `AIR_ASSIST_FUSION` env var as before. Safe defaults ensure that
`013.toml`, `092.toml`, and all executed experiments are unaffected.

**`max_completion_tokens` rule:** `60_000` for any experiment with `reasoning_effort = "medium"`
OR `result_count = 60` (logical OR); `30_000` otherwise (Tier 0/1 and E2a-low).

| TOML | Version | Experiments | Metadata gen | Reasoning | Hop policy | Chunks | `max_completion_tokens` |
|---|---|---|---|---|---|---|---|
| `092.toml` | 3.92 | E2a-low (frozen — executed) | OFF | low | Single | 25 | 30 000 |
| `093.toml` | 3.93 | E2a-med, E2c, E2a-med-mmr | ON | medium | Multi | 25 | 60 000 |
| `094.toml` | 3.94 | E2d | ON | medium | Multi | 60 | 60 000 |
| `095.toml` | 3.95 | E2d-nometa | OFF | medium | Multi | 60 | 60 000 |
| `096.toml` | 3.96 | E2e | ON | medium | Single | 60 | 60 000 |

**E2a-med, E2c** use `DSAS-2836/093.toml` (version 3.93) with:
- `include_metadata = true`, `result_count = 25`
- `reasoning_effort = "medium"`, `max_completion_tokens = 60_000`
- System prompt: nested format explanation (metadata gen ON variant; see §6.5.1) +
  hybrid retrieval + multi-hop phrasing-mix guidance (see §6.5.4)
- Multi-hop: `max_tool_iterations` absent (default unlimited)

**E2d** uses `DSAS-2836/094.toml` (version 3.94) with:
- `include_metadata = true`, `result_count = 60`
- Same system prompt as `093.toml`; `reasoning_effort = "medium"`, `max_completion_tokens = 60_000`
- Multi-hop: `max_tool_iterations` absent

**E2d-nometa** uses `DSAS-2836/095.toml` (version 3.95) with:
- `include_metadata = false`, `result_count = 60`
- System prompt: metadata gen OFF variant (same hybrid + multi-hop structure as E2a-med but
  metadata fields omitted from the context-format and metadata-use guidance)
- `reasoning_effort = "medium"`, `max_completion_tokens = 60_000`

**E2e** uses `DSAS-2836/096.toml` (version 3.96) with:
- `include_metadata = true`, `result_count = 60`
- Same system prompt as `093.toml`; `reasoning_effort = "medium"`, `max_completion_tokens = 60_000`
- Single-hop: `max_tool_iterations = 1` (see §6.5.6)

**MMR branch configs.** E0-mmr and E1-mmr add no new TOML files — they reuse
`rag_agent_v3/013.toml` (version 3.13), identical to their RRF counterparts, differing only by
`AIR_ASSIST_FUSION=mmr`. E2a-med-mmr reuses `DSAS-2836/093.toml` (version 3.93) and also
differs only by `AIR_ASSIST_FUSION=mmr`.

### 8.2 Per-experiment configuration — CLI and environment

Experiments are run with `uv run air-assist-evals eval-flow` (see §9.3 for full command
examples). The CLI parameters and environment variables below govern each run:

**CLI parameter `--dataset`** (required)
: Corpus selector. Determines which Elasticsearch index `AirAssistRunner` passes to
  `ExperimentToolProvider` and which Relativity workspace is used in the request context.
  Allowed values: `emc2`, `mallinckrodt`.
  Internally maps `emc2` → index `as-mierzej-emc2-a4r-v01` (title enabled) and
  `mallinckrodt` → index `as-mierzej-mlcdt-a4r-v01` (title disabled).
  Defined in `air_assist_experiments/config.py` (`EXPERIMENT_CONFIG["indices"]`).

**CLI parameter `--rubric-pattern-override`** (required for EMC2 set_1/set_2 separation)
: Narrows the rubric set within the eval-set glob. Use to target a single EMC2 UAT set:
  - `"**/uat/set_1/*.rubric.toml"` — 21 rubrics (EMC2 UAT set_1)
  - `"**/uat/set_2/*.rubric.toml"` — 20 rubrics (EMC2 UAT set_2)
  Omit for Mallinckrodt (all 22 rubrics are in a single folder with no sub-sets).

**CLI parameter `--model-version`** (required)
: Model version string in `<type>.<config>` format. For all Tier 0/1 experiments: `3.13`
  (resolves to `rag_agent_v3/013.toml`). For Tier 2: `3.92` (E2a-low), `3.93` (E2a-med, E2c,
  E2a-med-mmr), `3.94` (E2d), `3.95` (E2d-nometa), `3.96` (E2e).

**`AIR_ASSIST_EXPERIMENT_CID_SECRET`** (required, env)
: CID client secret for authenticating the Elasticsearch bearer token.
  Read by `air_assist_experiments/config.py:get_cid_client_secret()`.

**`AIR_ASSIST_FUSION`** (optional, env, default: `rrf`)
: Selects the final candidate-selection method applied after signal retrieval.
  Allowed values:
  - `rrf` — Reciprocal Rank Fusion (default for all non-MMR experiments).
  - `mmr` — Maximum Marginal Relevance (set for E0-mmr, E1-mmr, E2a-med-mmr).

  Resolved in `air_assist_experiments/retrieval/flat_retriever.py` and
  `nested_retriever.py` via `_get_fusion_method()`.

**Retrieval depth parameters** — stored in `EXPERIMENT_CONFIG["retrieval"]` (root branch,
`air_assist_experiments/config.py`); all are tunable. Depth parameters scale with
`result_count` as shown:

| Parameter | 25-chunk experiments | 60-chunk experiments | Rule |
|---|---|---|---|
| `result_count` | 25 | 60 | per-experiment (see table above) |
| chunk-signal `size` (documents) | 100 | 100 | fixed at 100; `size ≥ result_count` always holds (see §13.5 of `08-index-design-and-ingestion.md`) |
| chunk-signal `inner_hits.size` | 50 | 100 | `min(100, 2 × result_count)` |
| `N_doc` (document-level signals, top-k documents) | 25 | 60 | `= result_count` |
| scoped follow-up `inner_hits.size` | 50 | 100 | `min(100, 2 × result_count)` |

**Ceiling:** `inner_hits.size` cannot exceed `max_inner_result_window = 100` (index setting,
`from + size ≤ 100`). Supporting `result_count > 100` would require raising that index setting
first.

**Fusion parameters:** sparse-parent signal = `bool/should` sum (ES sums per-field scores);
document-level BM25 = `combined_fields` with equal field weights; RRF constant `k = 60`
(ordinal ranks); MMR `λ = 0.5`. E2c uses ES-side `rrf` retriever (no client fusion).

| Experiment | Config (version) | Retrieval mode | Fusion | Chunks | Metadata gen | Reasoning | Hop policy | 5 MB filter |
|---|---|---|---|---|---|---|---|---|
| E0 | `rag_agent_v3/013.toml` (3.13) | Flat — 2 chunk-level signals | Client-side RRF | 25 | OFF | low | Multi | ON |
| E1 | `rag_agent_v3/013.toml` (3.13) | Flat — 4 signals (2 chunk + 2 doc-level) | Client-side RRF | 25 | OFF | low | Multi | ON |
| E2a-low | `DSAS-2836/092.toml` (3.92) | Nested — client-side RRF | Client-side RRF | 25 | OFF | low | Single (capped) | OFF |
| E2a-med | `DSAS-2836/093.toml` (3.93) | Nested — client-side RRF | Client-side RRF | 25 | ON | medium | Multi | OFF |
| E2c | `DSAS-2836/093.toml` (3.93) | Nested — ES-side RRF | ES-side RRF | 25 | ON | medium | Multi | OFF |
| E2d | `DSAS-2836/094.toml` (3.94) | Nested — client-side RRF | Client-side RRF | 60 | ON | medium | Multi | OFF |
| E2d-nometa | `DSAS-2836/095.toml` (3.95) | Nested — client-side RRF | Client-side RRF | 60 | OFF | medium | Multi | OFF |
| E2e | `DSAS-2836/096.toml` (3.96) | Nested — client-side RRF | Client-side RRF | 60 | ON | medium | Single (capped) | OFF |
| E0-mmr | `rag_agent_v3/013.toml` (3.13) | Flat — 2 chunk-level signals | Client-side MMR (`AIR_ASSIST_FUSION=mmr`) | 25 | OFF | low | Multi | ON |
| E1-mmr | `rag_agent_v3/013.toml` (3.13) | Flat — 4 signals (2 chunk + 2 doc-level) | Client-side MMR (`AIR_ASSIST_FUSION=mmr`) | 25 | OFF | low | Multi | ON |
| E2a-med-mmr | `DSAS-2836/093.toml` (3.93) | Nested — client-side RRF | Client-side MMR (`AIR_ASSIST_FUSION=mmr`) | 25 | ON | medium | Multi | OFF |

### 8.3 Prompt changes summary

| Config | Experiment(s) | Changes from 013.toml |
|---|---|---|
| `rag_agent_v3/013.toml` (3.13) | E0, E1, E0-mmr, E1-mmr | **No changes.** System prompt, tool descriptions, and all guidance are identical to production. Retrieval backend is swapped at eval runner level only. Keyword-only framing is retained as an experimental control. |
| `DSAS-2836/092.toml` (3.92) | E2a-low | Nested format explanation, metadata gen OFF variant (see §6.5.1); corrected hybrid retrieval guidance + single-hop parallel-query instruction (see §6.5.4); single-hop cap via config-driven `max_tool_iterations` (see §6.5.6). |
| `DSAS-2836/093.toml` (3.93) | E2a-med, E2c, E2a-med-mmr | Nested format explanation, metadata gen ON variant (see §6.5.1); hybrid retrieval + multi-hop phrasing-mix guidance + metadata-as-hint instruction (see §6.5.4); `reasoning_effort = "medium"`; `include_metadata = true`; `result_count = 25`; `max_completion_tokens = 60_000`. E2a-med-mmr identical + `AIR_ASSIST_FUSION=mmr`. |
| `DSAS-2836/094.toml` (3.94) | E2d | Same prompt as `093.toml`; `include_metadata = true`, `result_count = 60`, `max_completion_tokens = 60_000`. |
| `DSAS-2836/095.toml` (3.95) | E2d-nometa | Metadata gen OFF variant of `093.toml` prompt (metadata context-format and metadata-use guidance omitted); `include_metadata = false`, `result_count = 60`, `max_completion_tokens = 60_000`. |
| `DSAS-2836/096.toml` (3.96) | E2e | Same prompt as `093.toml` + single-hop cap; `include_metadata = true`, `result_count = 60`, `max_completion_tokens = 60_000`; `max_tool_iterations = 1` (see §6.5.6). |

---

## 9. Evaluation Protocol

### 9.1 Evaluation datasets

| Dataset | Workspace | Rubric count | Schema | Source |
|---|---|---|---|---|
| EMC2 UAT set_1 | EMC2 (1030345) | 21 | v7 | Human-authored |
| EMC2 UAT set_2 | EMC2 (1030345) | 20 | v7 | Human-authored |
| Mallinckrodt GA | Mallinckrodt (1034598) | 22 | v7 | Human-authored |

**Total:** 63 rubrics across 3 datasets. Each dataset is evaluated as a separate MLflow
experiment with its own traces and quality report. Each rubric has multiple input variants
(paraphrases) and 2–74 expectations graded PASS/FAIL/UNDETERMINED by an LLM judge.

Rubric paths:
- `r1-evals-new/src/r1_evals/rubrics/rubric_data/air_assist/EMC2/uat/set_1/`
- `r1-evals-new/src/r1_evals/rubrics/rubric_data/air_assist/EMC2/uat/set_2/`
- `r1-evals-new/src/r1_evals/rubrics/rubric_data/air_assist/mallinckrodt/rubrics_for_ga/`

### 9.2 Metrics

| Metric | Source | What it measures |
|---|---|---|
| **Rubric score** | `RubricLLMJudgeScorerV2` | Answer quality: weighted average of expectation PASS/FAIL (0.0–1.0) |
| **Retrieval recall** | Custom scorer on `retrievedDocumentIds` vs rubric `document_ids` | Were the right documents retrieved? |
| **Citation accuracy** | `citation_validation` scorer | Do `[doc_id-chunk_id]` references point to real retrieved chunks? |
| **Citation-in-snippet match** | `citation_in_snippet_matching` scorer | Do cited snippets match the source chunk text? |
| **Latency** | MLflow trace duration | End-to-end response time |

### 9.3 Evaluation workflow

All experiments are run with the `air-assist-evals eval-flow` CLI command. This command
invokes `AirAssistRunner` (from `air_assist_evals/runner.py`) via the r1-evals
`invoke_and_evaluate` function. It handles model invocation, MLflow tracing, rubric scoring,
metrics aggregation, and optional report generation in a single call.

The general template per run is:

```bash
AIR_ASSIST_EXPERIMENT_CID_SECRET=<secret> [AIR_ASSIST_FUSION=mmr] \
uv run air-assist-evals eval-flow --full-evals \
    --dataset <emc2|mallinckrodt> \
    --model-version <3.13|3.92|3.93> \
    --experiment-name "<mlflow_experiment_path>" \
    --generate-report \
    [--rubric-pattern-override "<glob>"] \
    [--max-rubrics <n>]
```

`--max-rubrics 1` limits to one rubric for a quick smoke check.

### 9.4 How to run E1 and E1-mmr

The following table lists the exact command for each E1/E1-mmr run. All six runs must be
executed to cover the three evaluation datasets (EMC2 set_1, EMC2 set_2, Mallinckrodt GA).

| Run | Corpus | `--dataset` | `--rubric-pattern-override` | `AIR_ASSIST_FUSION` |
|---|---|---|---|---|
| E1 — EMC2 set_1 | EMC2 | `emc2` | `**/uat/set_1/*.rubric.toml` | (unset) |
| E1 — EMC2 set_2 | EMC2 | `emc2` | `**/uat/set_2/*.rubric.toml` | (unset) |
| E1 — Mallinckrodt | Mallinckrodt | `mallinckrodt` | (omit) | (unset) |
| E1-mmr — EMC2 set_1 | EMC2 | `emc2` | `**/uat/set_1/*.rubric.toml` | `mmr` |
| E1-mmr — EMC2 set_2 | EMC2 | `emc2` | `**/uat/set_2/*.rubric.toml` | `mmr` |
| E1-mmr — Mallinckrodt | Mallinckrodt | `mallinckrodt` | (omit) | `mmr` |

Example — E1, EMC2 set_1:

```bash
AIR_ASSIST_EXPERIMENT_CID_SECRET=<secret> \
uv run air-assist-evals eval-flow --full-evals \
    --dataset emc2 \
    --model-version 3.13 \
    --experiment-name "DSAS-2836/E1-emc2-set1" \
    --generate-report \
    --rubric-pattern-override "**/uat/set_1/*.rubric.toml"
```

Example — E1-mmr, Mallinckrodt:

```bash
AIR_ASSIST_EXPERIMENT_CID_SECRET=<secret> AIR_ASSIST_FUSION=mmr \
uv run air-assist-evals eval-flow --full-evals \
    --dataset mallinckrodt \
    --model-version 3.13 \
    --experiment-name "DSAS-2836/E1-mmr-mallinckrodt" \
    --generate-report
```

Each call creates a separate MLflow run within the named experiment, enabling per-run and
cross-run comparison.

### 9.5 Statistical analysis

Each of the 3 datasets is evaluated in a separate MLflow experiment with its own traces and
quality report. With 11 experiments per dataset:

| Dataset | Rubrics | Evaluation points (× 11) |
|---|---|---|
| EMC2 UAT set_1 | 21 | 231 |
| EMC2 UAT set_2 | 20 | 220 |
| Mallinckrodt GA | 22 | 242 |
| **Total** | **63** | **693** |

**Paired comparisons:** each rubric is evaluated under every experiment, enabling paired
statistical tests (Wilcoxon signed-rank) between any two experiments within the same dataset.
Paired tests apply within each dataset independently; cross-dataset aggregation is descriptive
only.

**Effect size:** Cohen's d or rank-biserial correlation to quantify practical significance
beyond p-values.

**Multiple comparisons:** Bonferroni or Holm-Bonferroni correction when comparing many
experiment pairs simultaneously within a dataset.

### 9.6 Feature selection methodology

After the initial 11-experiment matrix is evaluated:

1. **Rank experiments** by mean rubric score across all three datasets.
2. **Identify the top configuration(s)** and the dimensions that drove their gains over E0.
3. **Ablation:** for each active dimension in the winning configuration, run a variant with that
   dimension reverted to its E0 baseline. If performance drops significantly, the dimension
   contributes; if not, it can be dropped.
4. **Follow-up grid:** once dominant dimensions are identified, a targeted follow-up grid can
   explore secondary parameters (RRF k, `combined_fields` field weight tuning, document-rank
   broadcast vs per-chunk scoring, chunk concatenation policy, scoped follow-up depth) without
   running the full cross-product.

This produces a Pareto-optimal retrieval configuration balancing quality and complexity.

---

## 10. References

1. Ahuja, K. et al. (2025). "From BM25 to Corrective RAG: Benchmarking Retrieval Strategies
   for Text-and-Table Documents." *arXiv:2604.01733*.
   https://arxiv.org/html/2604.01733v1

2. Formal, T. et al. (2021). "SPLADE: Sparse Lexical and Expansion Model for First Stage
   Ranking." *SIGIR 2021*. https://arxiv.org/abs/2107.05720

3. Elastic. (2024). "Improving Information Retrieval in the Elastic Stack: Hybrid Retrieval."
   Elasticsearch Labs.
   https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid

4. Thulke, D. et al. (2025). "Benchmarking Retrieval Strategies for Biomedical
   Retrieval-Augmented Generation: A Controlled Empirical Study." *arXiv:2605.02520*.
   https://arxiv.org/pdf/2605.02520

5. Agarwal, P. et al. (2026). "Agent-Orchestrated Adaptive RAG: A Comparative Study on
   Structured and Multi-Hop Retrieval." *arXiv:2606.05658*.
   https://arxiv.org/html/2606.05658v1

6. Cormack, G. V. et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual
   Rank Learning methods." *SIGIR 2009*.
   https://dl.acm.org/doi/10.1145/1571941.1572114

7. Elastic. (2025). "Linear Retriever for Hybrid Search: Introduction & Configuration."
   Elasticsearch Labs.
   https://www.elastic.co/search-labs/blog/linear-retriever-hybrid-search

8. Elastic. (2025). "Reciprocal Rank Fusion." Elasticsearch Reference.
   https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion

9. Nguyen, A.T. (2026). "Elasticsearch 9 and Hybrid Search 2026 — BBQ, ELSER, Retrievers API,
   and a Production Search System Architecture."
   https://anhtu.dev/elasticsearch-9-hybrid-search-2026-bbq-elser-retrievers-api-production-search-system-architecture-1073

10. FutureAGI. (2026). "RAG Architecture in 2026: Patterns + Eval."
    https://futureagi.com/blog/rag-architecture-llm-2025/

### Internal references

- `es-index-explorer/reports/08-index-design-and-ingestion.md` — index design, query patterns
  (§13.1–§13.9), coverage matrix (§13.9).
- `es-index-explorer/reports/03-retrieval-strategies.md` — current production retrieval.
- `es-index-explorer/reports/01-air-assist-elasticsearch-index.md` — production index.
- `es-index-explorer/reports/07-client-side-sparse-vectors.md` — sparse migration.
- `air-assist-agent/.../rag_agent.py` — agent graph and tool dispatch.
- `air-assist-agent/.../tool_definitions.py` — current tool schemas.
- `air-assist-agent/.../tool_selection.py` — tool registration.
- `air-assist-agent/.../configs/rag_agent_v3/013.toml` — current agent config.
- `es-index-explorer/es_index_explorer/index_definitions/air_assist_nested.json` — new mapping.
