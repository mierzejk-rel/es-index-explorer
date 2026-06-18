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
Post-processing: security trim → top-25 → first-chunk fetch → group by document
    │
    ▼
GroupedDocumentSearchResult → agent (XML for LLM)
```

Source: `03-retrieval-strategies.md` §1; `rag_agent.py` lines 136–194.

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

| Signal | Field(s) | Query type | Scope |
|---|---|---|---|
| **BM25 on chunks** | `chunks.text` | `nested` → `match` | Chunk |
| **BM25 on parent metadata** | `title`, `summary`, `topic` | `multi_match` | Parent |
| **Dense kNN on chunks** | `chunks.embedding` (384-dim, cosine, bbq_hnsw) | `nested` → `knn` | Chunk |
| **Sparse on parent metadata** | `title_sparse`, `summary_sparse`, `topic_sparse` | `sparse_vector` query | Parent |

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

**BM25 on parent metadata:**
```json
{ "query": { "multi_match": { "query": "<q>", "fields": ["title", "topic", "summary"] } } }
```

**Dense kNN on chunks (nested):**
```json
{
  "query": { "nested": {
    "path": "chunks",
    "query": { "knn": { "field": "chunks.embedding", "query_vector": [/*384*/], "k": 100, "num_candidates": 250 } },
    "score_mode": "max",
    "inner_hits": { "size": 25 }
  } }
}
```

**Sparse on parent metadata:**
```json
{ "query": { "sparse_vector": { "field": "summary_sparse", "query_vector": {"token_a": 1.23, "token_b": 0.45} } } }
```

**Hybrid RRF (BM25 + dense on chunks):**
```json
{
  "retriever": { "rrf": { "rank_window_size": 100, "retrievers": [
    { "standard": { "query": { "nested": { "path": "chunks", "query": { "match": { "chunks.text": "<q>" } }, "score_mode": "max" } } } },
    { "knn": { "field": "chunks.embedding", "query_vector": [/*384*/], "k": 100, "num_candidates": 250 } }
  ] } }
}
```

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
| **Tier 0** | Production parity: reproduce current qna-service behaviour on the nested index | Flat chunk list | `low` |
| **Tier 1** | Metadata-enriched retrieval: add parent-field signals, keep flat output | Flat chunk list | `low` |
| **Tier 2** | Nested document format: full document groups with metadata and concatenated chunks | Nested doc groups | `medium` |

### 5.2 Variable dimensions

| Dimension | Tier 0 | Tier 1 | Tier 2 |
|---|---|---|---|
| **Retrieval signals** | BM25-chunks + kNN-chunks | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent (always ON) |
| **Metadata for generation** | OFF | OFF | ON or OFF |
| **Fusion location** | Client-side RRF | Client-side RRF | Client-side RRF or ES-side RRF |
| **Chunk count** | 25 | 25 | 25 or 60 |
| **Reasoning effort** | `low` | `low` | `medium` |
| **Hop strategy** | Multi-hop | Multi-hop | Multi-hop or Single-hop |
| **Title in retrieval** | N/A | EMC2: ON / Mallinckrodt: OFF | EMC2: ON / Mallinckrodt: OFF |
| **5 MB size filter** | ON | ON | ON |

### 5.3 Notes on design choices

**Reasoning effort shift at Tier 2.** Tier 2 experiments use `reasoning_effort: medium`
throughout rather than making it a variable dimension. The nested document format sends
significantly more structured information to the LLM (metadata per document group plus
concatenated adjacent chunks); medium reasoning is expected to be necessary to fully exploit
that richer context. A clean reasoning-effort comparison is available through E0/E1 (low) vs
E2a (medium) via the E1 → E2a transition, though this comparison also changes the output
format.

**Metadata retrieval always ON in Tier 2.** Once the nested document structure is adopted,
excluding parent-field signals would waste the indexed metadata. All Tier 2 experiments use
the full signal set. The value of metadata retrieval signals is assessed at Tier 1 (E0 vs E1).

**Title toggle.** The EMC2 workspace has meaningful document titles (email subjects, file
names); the Mallinckrodt corpus has less reliable titles. Title is therefore included as a
retrieval signal for EMC2 but excluded for Mallinckrodt. This is implemented as a per-eval-set
configuration flag passed to the tool at runtime, not as a separate experiment dimension.

**Client-side vs ES-side fusion.** When fusion is performed on the client, each signal
(BM25-chunks, kNN-chunks, BM25-parent, sparse-parent) is issued as a separate ES query and the
returned chunk-level results are fused via RRF in Python. When fusion is ES-side, a single ES
request with a nested RRF retriever is issued; ES fuses at the **document level** (each
document is scored by its best matching chunk via `score_mode: max`) and the client reads
chunks from `inner_hits`. ES-side fusion cannot rank chunks across documents globally — that
distinction is explored by E2a vs E2c.

---

## 6. Experiment Matrix

### 6.1 Experiment definitions

**Tier 0 — Baseline reproduction**

**E0** — Production parity on nested index.

Run two nested ES queries independently (BM25 on `chunks.text`, kNN on `chunks.embedding`),
flatten the returned `inner_hits` chunks, apply client-side RRF, take top 25 by score. Filter
`byte_size <= 5 MB`. No parent metadata fields involved in retrieval or generation. Use
`013.toml` unchanged (model config, prompt, tool descriptions). Output is a flat chunk list
in the same XML format as current production.

*Purpose:* confirm the new direct-ES Python tool reproduces current qna-service behavior.

---

**Tier 1 — Metadata-enriched retrieval (flat output)**

**E1** — Add parent-field retrieval signals, keep flat output.

Same as E0 plus: BM25 `multi_match` on `title`/`summary`/`topic` and `sparse_vector` queries
on `title_sparse`/`summary_sparse`/`topic_sparse`. All four signal results (BM25-chunks,
kNN-chunks, BM25-parent, sparse-parent) are issued as separate ES queries and fused
client-side via RRF to produce the top 25 chunks. Title included for EMC2, excluded for
Mallinckrodt. Output remains a flat chunk list; the LLM does not see document metadata
directly.

*Purpose:* does adding BM25 + sparse signals on parent metadata fields improve chunk
selection quality?

---

**Tier 2 — Nested document format (anchor and variants)**

All Tier 2 experiments use the nested document output structure: the tool returns document
groups, each containing the document's `title`, `summary`, `topic`, `control_number`,
`primary_date_time`, and a list of retrieved chunks with adjacent chunks concatenated where
`chunk_index` values are consecutive. The LLM prompt is updated for this format. A new
model TOML config is created for Tier 2 (see §8).

**E2a** — Anchor experiment (nested, full signals, metadata in generation, client-side RRF,
25 chunks, medium reasoning, multi-hop).

Retrieval: all four signals (BM25-chunks, kNN-chunks, BM25-parent, sparse-parent) issued
separately, client-side RRF fusion, top 25 chunks. Generation context: document groups with
`title`/`summary`/`topic` visible to the LLM. `reasoning_effort: medium`. Multi-hop (1–3
iterations, LLM decides). Title toggle: EMC2 ON / Mallinckrodt OFF.

*Purpose:* anchor for all Tier 2 comparisons; first test of the full nested pipeline.

**E2b** — Same as E2a but metadata generation OFF.

Document groups are returned without `title`/`summary`/`topic` in the XML; the LLM sees only
chunk text. Everything else identical to E2a.

*Purpose:* isolates the contribution of metadata fields to generation quality, holding
retrieval fixed.

**E2c** — Same as E2a but ES-side fusion.

A single ES request with a nested RRF retriever fuses all signals. ES returns documents ranked
by their best chunk score; the client reads `inner_hits` and groups. No cross-document chunk
ranking by the client.

*Purpose:* isolates fusion location (client-side global chunk ranking vs ES-side document
ranking).

**E2d** — Same as E2a but 60 chunks.

Retrieval returns 60 chunks instead of 25. Inner-hits sizes scaled accordingly.

*Purpose:* isolates the effect of providing more retrieved context to the LLM.

**E2e** — Same as E2d but single-hop with parallel tool calls.

The LLM is constrained to exactly one retrieval iteration (`tool_choice="none"` after
iteration 0). The agent may issue multiple parallel tool calls in that single iteration.
Chunk count: 60.

*Purpose:* isolates hop strategy at 60 chunks — does multi-hop reasoning add value beyond
what a single richer retrieval provides?

### 6.2 Full experiment table

| ID | Tier | Retrieval signals | Metadata for generation | Fusion location | Chunks | Reasoning | Hop strategy |
|---|---|---|---|---|---|---|---|
| **E0** | 0 | BM25-chunks + kNN-chunks | OFF | Client-side RRF | 25 | low | Multi-hop |
| **E1** | 1 | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | OFF | Client-side RRF | 25 | low | Multi-hop |
| **E2a** | 2 | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | ON | Client-side RRF | 25 | medium | Multi-hop |
| **E2b** | 2 | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | **OFF** | Client-side RRF | 25 | medium | Multi-hop |
| **E2c** | 2 | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | ON | **ES-side RRF** | 25 | medium | Multi-hop |
| **E2d** | 2 | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | ON | Client-side RRF | **60** | medium | Multi-hop |
| **E2e** | 2 | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | ON | Client-side RRF | **60** | medium | **Single-hop** |

Bold values mark the single dimension that differs from the anchor (E2a for Tier 2, E0 for
Tier 1).

### 6.3 Comparison map

Each comparison isolates one dimension; all other dimensions are held constant.

| Comparison | Dimension isolated | Question answered |
|---|---|---|
| E0 vs E1 | Retrieval signals | Does adding BM25 + sparse on parent metadata improve chunk selection? |
| E1 vs E2a | Output format + reasoning effort | Does the nested document format (with metadata visible to LLM) and stronger reasoning improve answers? |
| E2a vs E2b | Metadata for generation | Does exposing title/summary/topic to the LLM improve answers, given metadata-enriched retrieval? |
| E2a vs E2c | Fusion location | Is client-side global chunk ranking better than ES-side document-level ranking? |
| E2a vs E2d | Chunk count | Does providing 60 chunks instead of 25 improve answers? |
| E2d vs E2e | Hop strategy | Does multi-hop reasoning over 60 chunks beat single-hop retrieval of 60 chunks? |

### 6.4 Execution order

Run E0 and E1 first to validate that the new Python tools produce results at parity with
production and confirm that metadata signals add value at the flat-output level. Then run
E2a as the Tier 2 anchor before running E2b–E2e in parallel.

```
E0 → E1 → E2a → E2b, E2c, E2d, E2e (parallel)
```

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
    │  selected by AIR_ASSIST_RETRIEVAL env var
    ▼
elasticsearch-py AsyncElasticsearch client
    │  separate queries per signal (client-side fusion)
    │  or single RRF retriever (ES-side fusion)
    ▼
Elasticsearch (as-* nested index)
    │
    ▼
Post-processing: inner_hits flatten + chunk ranking + grouping + concatenation
    │
    ▼
Flat chunk list (Tier 0/1) or Nested doc groups (Tier 2) → XML for LLM
```

**Client lifecycle:** a single `AsyncElasticsearch` instance, created at agent startup with
bearer auth (CID token), shared across all tool calls within a request.

**Query embedding:** the tool handler embeds the query string at call time using cached model
instances:
- Dense: `intfloat/multilingual-e5-small` with `"query: "` prefix → 384-dim vector
- Sparse: `opensearch-neural-sparse-encoding-v2-distill` → `{token: weight}` map

Models are loaded lazily and cached for the process lifetime, mirroring the `E5Embedder` /
`SparseEmbedder` pattern in `es-index-explorer/indexing/embedding.py`.

### 7.2 Tool selection

The active tool set is selected by the environment variable `AIR_ASSIST_RETRIEVAL`:

| Value | Tool set | Used by |
|---|---|---|
| `flat_baseline` | `SearchDocuments` (flat output) + `WriteFile` + `ReadFile` | E0, E1 |
| `nested_docs` | `SearchDocuments` (nested output) + `WriteFile` + `ReadFile` | E2a–E2e |

Tool selection reads the env var in `tool_selection.py` and registers the appropriate
`_ALL_TOOL_MODELS` tuple. E0 continues to use the current MCP tools (no env var needed; it
runs with the original `013.toml` and the existing `tool_selection.py` unchanged).

### 7.3 Tool definitions

Two new Pydantic tool models replace the current MCP pair:

#### `SearchDocuments` (all experiments except E0)

```python
class SearchDocuments(BaseModel):
    """Search the document corpus using hybrid retrieval.

    Returns top documents ranked by relevance to `query`. Metadata filters
    (dates, email participants) are optional hard constraints. The retrieval
    strategy — signals, fusion, and result count — is configured server-side
    and is not exposed here.
    """
    query: str = Field(..., min_length=1,
        description="Keyword or natural-language query. No boolean operators.")
    date_from: ISODate | None = None
    date_to: ISODate | None = None
    email_participants: EmailParticipantsFilter | None = None
```

This single tool replaces both `GetRelevantDocuments` and
`GetRelevantDocumentsWithMetadataFilter`. The retrieval strategy (signals, fusion, scope,
chunk count) is controlled by the server-side config, invisible to the LLM. This matches
the current contract: the agent describes what it wants to find, not how to find it.

When `flat_baseline` is active, the tool returns a flat chunk list (Tier 0/1 format).
When `nested_docs` is active, it returns nested document groups with metadata and
concatenated adjacent chunks (Tier 2 format).

### 7.4 Post-retrieval processing

#### Flat output (Tier 0, Tier 1)

1. Issue separate ES queries for each active signal.
2. Collect all chunks from `inner_hits` across all queries.
3. Apply client-side RRF: for each chunk `(document_artifact_id, chunk_index)`, sum
   reciprocal ranks across signals. Take top-k by fused score.
4. Serialize as flat `<passage>` XML elements matching the current qna-service format.

#### Nested output (Tier 2)

1. Same multi-signal queries as Tier 1 (client-side RRF) or a single ES RRF retriever
   (ES-side RRF, E2c).
2. Group result chunks by `document_artifact_id`.
3. For each group, attach parent metadata (`title`, `summary`, `topic`, `control_number`,
   `primary_date_time`) from `_source`. If metadata for generation is OFF (E2b), omit
   `title`/`summary`/`topic` from the XML.
4. Within each group, detect runs of consecutive `chunk_index` values and concatenate
   adjacent chunks by removing the leading overlap (using `leading_overlap_chars`).
5. Serialize as `<document_group>` XML elements containing per-document metadata and
   chunk list.

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

---

## 8. Config and Prompt Variations

### 8.1 TOML configs

**E0** uses `013.toml` unchanged (current production config, MCP tools, `reasoning_effort: low`).

**E1** uses a new config (e.g. `014.toml`) identical to `013.toml` except:
- `reasoning_effort` stays `low`
- System prompt updated: `SearchDocuments` replaces the two retrieval tools; remove
  references to "keyword search" / "BM25"; describe as "relevance search"
- `required_tools` updated accordingly

**E2a–E2e** use a new config (e.g. `015.toml`) with:
- `reasoning_effort: medium`
- System prompt updated for nested document format: explain that each result group contains
  document metadata (title, summary, topic) plus extracted passages; instruct the LLM to use
  metadata to understand document context before citing specific passages
- E2e additionally: prompt instructs exactly one retrieval iteration; tool_choice capped at
  `none` after iteration 0

The TOML schema is not extended. Retrieval parameters (signals, chunk count, fusion) are
passed via the `AIR_ASSIST_RETRIEVAL` environment variable and a companion environment-specific
config file read by the tool at startup, not via the TOML.

### 8.2 Per-experiment environment configuration

| Experiment | `AIR_ASSIST_RETRIEVAL` | Fusion | Chunks | Reasoning | Hop policy |
|---|---|---|---|---|---|
| E0 | *(MCP tools, not used)* | ES RRF via qna-service | 25 | low | Multi (013.toml) |
| E1 | `flat_baseline` | Client-side RRF | 25 | low | Multi (014.toml) |
| E2a | `nested_docs` | Client-side RRF | 25 | medium | Multi (015.toml) |
| E2b | `nested_docs` | Client-side RRF | 25 | medium | Multi (015.toml, metadata gen OFF) |
| E2c | `nested_docs_es_rrf` | ES-side RRF | 25 | medium | Multi (015.toml) |
| E2d | `nested_docs` | Client-side RRF | 60 | medium | Multi (015.toml) |
| E2e | `nested_docs` | Client-side RRF | 60 | medium | Single (015.toml, capped) |

### 8.3 Prompt changes summary

| Experiment group | Change from 013.toml |
|---|---|
| E0 | None |
| E1 | Merge two retrieval tools into `SearchDocuments`; remove BM25-specific guidance; describe retrieval as "relevance search" |
| E2a, E2c–E2e | E1 changes + add nested document format explanation: "Each result contains document-level metadata (title, summary, topic) followed by the most relevant passages. Use the metadata to orient your understanding before citing passages." |
| E2b | Same as E2a group but metadata fields are omitted from the XML; no prompt change needed for the format difference |
| E2e | Add: "Perform exactly one retrieval call per turn. Issue all necessary queries simultaneously using parallel tool calls." |

---

## 9. Evaluation Protocol

### 9.1 Evaluation datasets

| Dataset | Workspace | Rubric count | Schema | Source |
|---|---|---|---|---|
| EMC2 UAT set_1 | EMC2 (1030345) | 21 | v7 | Human-authored |
| Mallinckrodt GA | Mallinckrodt (1034598) | 22 | v7 | Human-authored |
| Mallinckrodt synthetic | Mallinckrodt (1034598) | 34 | v1 | AI-generated (Vals.ai) |

**Total:** 77 rubrics. Each rubric has multiple input variants (paraphrases) and 2–74
expectations graded PASS/FAIL/UNDETERMINED by an LLM judge.

Rubric paths:
- `r1-evals-new/src/r1_evals/rubrics/rubric_data/air_assist/EMC2/uat/set_1/`
- `r1-evals-new/src/r1_evals/rubrics/rubric_data/air_assist/mallinckrodt/rubrics_for_ga/`
- `r1-evals-new/src/r1_evals/rubrics/rubric_data/air_assist/mallinckrodt/synthetic_valsai_key_doc/`

### 9.2 Metrics

| Metric | Source | What it measures |
|---|---|---|
| **Rubric score** | `RubricLLMJudgeScorerV2` | Answer quality: weighted average of expectation PASS/FAIL (0.0–1.0) |
| **Retrieval recall** | Custom scorer on `retrievedDocumentIds` vs rubric `document_ids` | Were the right documents retrieved? |
| **Citation accuracy** | `citation_validation` scorer | Do `[doc_id-chunk_id]` references point to real retrieved chunks? |
| **Citation-in-snippet match** | `citation_in_snippet_matching` scorer | Do cited snippets match the source chunk text? |
| **Latency** | MLflow trace duration | End-to-end response time |

### 9.3 Evaluation workflow

```
For each experiment E_i:
    1. Load config (TOML)
    2. For each rubric:
       a. Run agent with rubric input
       b. Log MLflow trace (input, output, retrieved docs, search queries)
    3. Score: r1-evals run evaluate --eval-set air_assist_v2
    4. Aggregate: r1-evals run aggregate
    5. Report: r1-evals run rubric-dashboard-v2
```

Each experiment is a separate MLflow run within a shared experiment. This enables per-experiment
and cross-experiment comparison.

### 9.4 Statistical analysis

With 77 rubrics per experiment and 7 experiments, we have 539 total evaluation points.

**Paired comparisons:** each rubric is evaluated under every experiment, enabling paired
statistical tests (Wilcoxon signed-rank) between any two experiments on the same rubric set.

**Effect size:** Cohen's d or rank-biserial correlation to quantify practical significance
beyond p-values.

**Multiple comparisons:** Bonferroni or Holm-Bonferroni correction when comparing many
experiment pairs simultaneously.

### 9.5 Feature selection methodology

After the initial 7-experiment matrix is evaluated:

1. **Rank experiments** by mean rubric score across both workspaces.
2. **Identify the top configuration(s)** and the dimensions that drove their gains over E0.
3. **Ablation:** for each active dimension in the winning configuration, run a variant with that
   dimension reverted to its E0 baseline. If performance drops significantly, the dimension
   contributes; if not, it can be dropped.
4. **Follow-up grid:** once dominant dimensions are identified, a targeted follow-up grid can
   explore secondary parameters (field boosts, RRF k, chunk concatenation policy) without
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
