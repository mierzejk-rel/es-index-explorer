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

## 5. Experiment Factor Space

Six variable factors span the experimental design. Each factor has discrete levels.

### 5.1 Factor taxonomy

#### F1: Retrieval signal

| Level | Signals used | Query shape |
|---|---|---|
| `BM25` | BM25 on `chunks.text` only | nested `match` |
| `Dense` | kNN on `chunks.embedding` only | nested `knn` |
| `Sparse` | Sparse on `title_sparse` + `summary_sparse` + `topic_sparse` | `sparse_vector` queries (3 fields) |
| `BM25+Dense` | BM25 on chunks + kNN on chunks | Two nested sub-retrievers |
| `BM25+Sparse` | BM25 on chunks + sparse on parent metadata | Mixed nested + top-level |
| `BM25+Dense+Sparse` | All three signals | Three sub-retrievers |

#### F2: Fusion method

| Level | Mechanism | When applicable |
|---|---|---|
| `None` | Single signal, no fusion | F1 is single-signal |
| `RRF-60` | ES RRF retriever, `rank_constant=60` | F1 is multi-signal |
| `RRF-10` | ES RRF retriever, `rank_constant=10` | F1 is multi-signal |
| `Linear-equal` | ES linear retriever, equal weights, minmax normalization | F1 is multi-signal |

The literature suggests RRF k=60 is a robust default [3][6], while k=10 emphasises top-ranked
documents more aggressively — which the financial QA benchmark found optimal for RRF [1]. Linear
with equal weights and minmax normalization slightly outperforms RRF on some benchmarks (Recall@5
0.726 vs 0.695 [1]) but requires normalization and is harder to tune. We include both for
comparison.

#### F3: Retrieval scope

| Level | What is retrieved |
|---|---|
| `Chunks` | Only chunk-level signals (BM25 on `chunks.text`, kNN on `chunks.embedding`) |
| `Parent` | Only parent-level signals (BM25 on `title`/`summary`/`topic`, sparse on `*_sparse`) |
| `Chunks+Parent` | Both chunk-level and parent-level signals combined |

#### F4: Field boosting (parent BM25)

| Level | `multi_match.fields` configuration |
|---|---|
| `None` | No parent BM25 in the retriever |
| `Equal` | `["title", "summary", "topic"]` — equal weight |
| `Boosted` | `["title^2", "summary^3", "topic^2"]` — summary weighted highest |

Boosting is relevant only when parent BM25 is included (F3 = `Parent` or `Chunks+Parent`). The
boost values are initial estimates motivated by the hypothesis that `summary` is the richest
metadata signal, followed by `title` and `topic`. These can be refined via grid search in a
follow-up optimisation phase.

Elasticsearch also supports boosting in `sparse_vector` queries via a `boost` parameter on each
field. This is a secondary tuning knob documented in `08-index-design-and-ingestion.md` §13.3,
available for follow-up optimisation but not included as a primary factor to keep the initial
matrix tractable.

#### F5: Agent hop strategy

| Level | Behaviour |
|---|---|
| `SingleHop` | Agent makes exactly 1 retrieval call per iteration (LLM still loops 1–3 times) |
| `MultiHopBaseline` | Current 013.toml prompt: 1–3 retrieval iterations, LLM decides when to stop |
| `MultiHopMetadataFirst` | Hop 1: parent-only retrieval (sparse + BM25 on title/summary/topic); Hop 2: chunk-level retrieval restricted to documents from Hop 1 |

`MultiHopMetadataFirst` is the novel strategy. The intuition: use cheap, high-recall
parent-metadata queries to identify candidate documents, then use expensive dense+BM25 chunk
queries on the narrowed set for passage extraction. This is analogous to the two-stage
retrieve-then-rerank pattern [1][4], but with the first stage operating on document metadata
rather than passages.

#### F6: Result count

| Level | Chunks returned to the LLM |
|---|---|
| `Top-10` | 10 |
| `Top-25` | 25 (current default) |
| `Top-50` | 50 |

### 5.2 Combinatorial analysis

The full factor space has 7 × 4 × 3 × 3 × 3 × 3 = **2,268** combinations, many of which are
invalid (e.g. fusion without multi-signal, boosting without parent scope). The curated matrix
below selects ~15 valid, informative experiments that cover all factors and ideas.

---

## 6. Experiment Matrix

### 6.1 Curated experiment set

Each row specifies the factor levels and the primary question it answers.

| ID | Name | F1: Signal | F2: Fusion | F3: Scope | F4: Boost | F5: Hops | F6: k | Primary question |
|---|---|---|---|---|---|---|---|---|
| **E0** | Baseline | BM25+Dense (via MCP) | RRF-60 | Chunks | None | MultiHopBaseline | 25 | Reproduce current agent performance |
| **E1** | BM25-nested | BM25 | None | Chunks | None | MultiHopBaseline | 25 | Parity: new index BM25 vs production BM25 |
| **E2** | Dense-only | Dense | None | Chunks | None | MultiHopBaseline | 25 | Dense kNN ceiling on new index |
| **E3** | Sparse-only | Sparse | None | Parent | None | MultiHopBaseline | 25 | Sparse parent metadata retrieval ceiling |
| **E4** | Hybrid-chunks | BM25+Dense | RRF-60 | Chunks | None | MultiHopBaseline | 25 | 2026 default hybrid on new nested index |
| **E5** | Hybrid+parent-BM25 | BM25+Dense | RRF-60 | Chunks+Parent | Equal | MultiHopBaseline | 25 | Does adding parent BM25 help? |
| **E6** | Hybrid+parent-boosted | BM25+Dense | RRF-60 | Chunks+Parent | Boosted | MultiHopBaseline | 25 | Does field boosting improve over equal weight? |
| **E7** | Three-signal-RRF | BM25+Dense+Sparse | RRF-60 | Chunks+Parent | Equal | MultiHopBaseline | 25 | Does adding sparse as a third signal help? |
| **E8** | Three-signal-linear | BM25+Dense+Sparse | Linear-equal | Chunks+Parent | Equal | MultiHopBaseline | 25 | Linear vs RRF for three-signal fusion |
| **E9** | MetadataFirst-hop | BM25+Dense+Sparse | RRF-60 | Chunks+Parent | Equal | MultiHopMetadataFirst | 25 | Does metadata-first two-phase retrieval help? |
| **E10** | SingleHop-full | BM25+Dense+Sparse | RRF-60 | Chunks+Parent | Equal | SingleHop | 25 | Can a single rich retrieval call match multi-hop? |
| **E11** | RRF-k10 | BM25+Dense | RRF-10 | Chunks | None | MultiHopBaseline | 25 | Lower k emphasises top ranks — better or worse? |
| **E12** | Top-50 | BM25+Dense+Sparse | RRF-60 | Chunks+Parent | Equal | MultiHopBaseline | 50 | More context to the LLM — does it help or hurt? |
| **E13** | Top-10 | BM25+Dense+Sparse | RRF-60 | Chunks+Parent | Equal | MultiHopBaseline | 10 | Less context to the LLM — precision over recall? |

### 6.2 Experiment dependencies and ordering

```
E0 (baseline)
 ├── E1 (BM25 parity check)
 ├── E2 (dense ceiling)
 ├── E3 (sparse ceiling)
 └── E4 (hybrid-chunks baseline for new index)
      ├── E5, E6 (parent BM25 ablation)
      ├── E11 (RRF k ablation)
      └── E7 (three-signal)
           ├── E8 (linear vs RRF)
           ├── E9 (metadata-first hops)
           ├── E10 (single-hop)
           ├── E12 (top-50)
           └── E13 (top-10)
```

**Recommended execution order:** E0, E1, E4, E2, E3, then E5–E13 in parallel. E0 establishes
the baseline; E1 and E4 validate the new index is at least at parity; the remaining experiments
test incremental improvements.

### 6.3 Factor coverage verification

| Factor | Levels covered | By experiments |
|---|---|---|
| F1: Signal | BM25, Dense, Sparse, BM25+Dense, BM25+Dense+Sparse | E1, E2, E3, E4, E7 |
| F2: Fusion | None, RRF-60, RRF-10, Linear-equal | E1/E2/E3, E4/E5–E10/E12/E13, E11, E8 |
| F3: Scope | Chunks, Parent, Chunks+Parent | E1/E2/E4/E11, E3, E5–E10/E12/E13 |
| F4: Boost | None, Equal, Boosted | E0–E4/E11, E5/E7–E10/E12/E13, E6 |
| F5: Hops | SingleHop, MultiHopBaseline, MultiHopMetadataFirst | E10, E0–E8/E11–E13, E9 |
| F6: Count | 10, 25, 50 | E13, E0–E11, E12 |

All factor levels are represented.

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
    │  constructs ES query from config + tool args
    ▼
elasticsearch-py AsyncElasticsearch client
    │
    ▼
Elasticsearch (as-* nested index)
    │
    ▼
Post-processing: inner_hits flatten → chunk ranking → group by document
    │
    ▼
GroupedChunks → XML for LLM
```

**Client lifecycle:** a single `AsyncElasticsearch` instance, created at pipeline startup with
bearer auth (CID token), shared across tool calls within a request. Connection parameters
(hosts, auth) come from TOML config.

**Query embedding:** the tool handler embeds the query string using the same models as ingest:
- Dense: `intfloat/multilingual-e5-small` with `"query: "` prefix → 384-dim vector
- Sparse: `opensearch-neural-sparse-encoding-v2-distill` → `{token: weight}` map

Model instances are loaded lazily and cached for the process lifetime, mirroring the
`E5Embedder` / `SparseEmbedder` pattern in `es-index-explorer/indexing/embedding.py`.

### 7.2 Tool definitions

Three new Pydantic tool models, registered in place of the current MCP-based tools:

#### `SearchDocuments` (primary retrieval tool)

```python
class SearchDocuments(BaseModel):
    """Search the document corpus using hybrid retrieval.

    Returns top documents ranked by relevance to `query`, using the configured
    retrieval strategy (BM25, dense, sparse, or hybrid fusion). Supports
    optional metadata filters for dates, email participants, and subset scoping.
    """
    query: str = Field(..., min_length=1)
    date_from: ISODate | None = None
    date_to: ISODate | None = None
    email_participants: EmailParticipantsFilter | None = None
```

This tool replaces both `GetRelevantDocuments` and `GetRelevantDocumentsWithMetadataFilter`.
The retrieval strategy (signals, fusion, scope, boosting, result count) is **not** exposed to
the LLM — it is configured per-experiment in TOML. The tool always accepts metadata filters;
when none are provided, only the subset filter is applied.

**Rationale for merging:** the current two-tool split exists because qna-service has separate
code paths. With direct ES queries, a single tool with optional filter parameters is simpler
and avoids the LLM needing to choose between tools.

#### `SearchByMetadata` (multi-hop hop 1)

```python
class SearchByMetadata(BaseModel):
    """Lightweight search on document metadata (title, summary, topic).

    Returns document IDs and metadata — not chunk content — for use as a
    first-pass filter before detailed chunk retrieval.
    """
    query: str = Field(..., min_length=1)
    date_from: ISODate | None = None
    date_to: ISODate | None = None
    email_participants: EmailParticipantsFilter | None = None
```

Returns parent-level results only (no `inner_hits`). Used in `MultiHopMetadataFirst` (E9) as
hop 1 to identify candidate documents before chunk-level follow-up.

#### `SearchChunksInDocuments` (multi-hop hop 2)

```python
class SearchChunksInDocuments(BaseModel):
    """Retrieve chunks from specific documents.

    Given a set of document IDs (from a prior metadata search), retrieve the
    most relevant chunks using dense and/or BM25 search within those documents.
    """
    query: str = Field(..., min_length=1)
    document_ids: list[int] = Field(..., min_length=1)
```

Issues a chunk-level nested query with a `terms` filter on `document_artifact_id` restricting
to the hop-1 result set.

### 7.3 Tool registration per experiment

| Experiment | Registered tools |
|---|---|
| E0 | Current MCP tools (unchanged) |
| E1–E8, E10–E13 | `SearchDocuments` + `WriteFile` + `ReadFile` |
| E9 | `SearchByMetadata` + `SearchChunksInDocuments` + `WriteFile` + `ReadFile` |

The registration happens in `tool_selection.py` where `_ALL_TOOL_MODELS` is defined. For
experiments, a config flag selects which tool set to register.

### 7.4 Post-retrieval processing

The tool handler must replicate qna-service's post-processing:

1. **Inner-hits flatten:** Extract all chunks from `inner_hits`, flatten across documents,
   sort by `_score`, take top-k.
2. **Document grouping:** Group chunks by `document_artifact_id`, attach parent metadata
   (`control_number`, `title`, `summary`, `topic`, `primary_date_time`).
3. **First-chunk injection:** For each document, include `chunk_index=0` text as context
   header (currently done via a separate ES query in qna-service; in the nested index,
   chunk 0 is available in `_source.chunks[0]` or via a targeted `inner_hits` filter).
4. **XML serialization:** Produce the same `GroupedChunks` → XML format the LLM expects.

### 7.5 Metadata filter mapping

| Current MCP field | New ES query |
|---|---|
| `dateFrom` | `{"range": {"primary_date_time": {"gte": "<dateFrom>"}}}` |
| `dateTo` | `{"range": {"primary_date_time": {"lt": "<dateTo + 1 day>"}}}` |
| `emailParticipants.participantsA` (AtoB) | `bool.must` with `term` on `email_from` (side A) and `term` on `email_to`/`email_cc`/`email_bcc` (side B) |
| `emailParticipants` (AnyDirection) | `bool.should` with both direction permutations |
| `emailParticipants` (Either) | `bool.should` with side A only OR side B only |
| `subsetId` | `{"term": {"subset_ids": "<subset>"}}` |

The semantic mapping is equivalent to `03-retrieval-strategies.md` §5; the implementation
shifts from qna-service C# to Python `elasticsearch-py` query builders.

---

## 8. Config and Prompt Variations

### 8.1 TOML config structure

Each experiment is defined by a TOML config extending 013.toml:

```toml
[metadata]
version = 14           # or 15, 16, ...
agent_version = 3
experiment_id = "E4"   # maps to experiment matrix row

[elasticsearch]
hosts = "https://..."
index_name = "as-mierzej-mlcdt-a4r-v01"      # workspace-specific
subset_id = "97848311-d89e-426e-8386-e48b7b1fec39"

[retrieval]
signals = ["bm25_chunks", "dense_chunks"]     # F1
fusion = "rrf"                                 # F2
rrf_rank_constant = 60
scope = "chunks"                               # F3
parent_bm25_fields = []                        # F4
parent_bm25_boost = {}
result_count = 25                              # F6
rank_window_size = 100
knn_k = 100
knn_num_candidates = 250
inner_hits_size = 25

[retrieval.sparse]
model = "opensearch-project/opensearch-neural-sparse-encoding-v2-distill"
model_path = ""
device = "cpu"
query_overflow = "trim"

[retrieval.dense]
model = "intfloat/multilingual-e5-small"
model_path = ""
device = "cpu"
query_prefix = "query: "

[openai_request]
model = "gpt-5.1-2025-11-13"
reasoning_effort = "low"
seed = 20
max_completion_tokens = 30_000
```

### 8.2 Signal-to-config mapping

| Experiment | `signals` | `fusion` | `scope` | `parent_bm25_fields` | `parent_bm25_boost` |
|---|---|---|---|---|---|
| E1 | `["bm25_chunks"]` | `none` | `chunks` | — | — |
| E2 | `["dense_chunks"]` | `none` | `chunks` | — | — |
| E3 | `["sparse_parent"]` | `none` | `parent` | — | — |
| E4 | `["bm25_chunks", "dense_chunks"]` | `rrf` | `chunks` | — | — |
| E5 | `["bm25_chunks", "dense_chunks"]` | `rrf` | `chunks_parent` | `["title", "summary", "topic"]` | `{}` |
| E6 | `["bm25_chunks", "dense_chunks"]` | `rrf` | `chunks_parent` | `["title", "summary", "topic"]` | `{"title": 2, "summary": 3, "topic": 2}` |
| E7 | `["bm25_chunks", "dense_chunks", "sparse_parent"]` | `rrf` | `chunks_parent` | `["title", "summary", "topic"]` | `{}` |
| E8 | `["bm25_chunks", "dense_chunks", "sparse_parent"]` | `linear` | `chunks_parent` | `["title", "summary", "topic"]` | `{}` |
| E11 | `["bm25_chunks", "dense_chunks"]` | `rrf` | `chunks` | — | — |

E11 differs from E4 only in `rrf_rank_constant = 10`.

### 8.3 Prompt variations

The 013.toml system prompt is largely reusable. Per-experiment changes:

| Experiment | Prompt change |
|---|---|
| E0 | No change (current prompt, current tools) |
| E1–E8, E10–E13 | Replace tool descriptions: `SearchDocuments` replaces `GetRelevantDocuments` + `GetRelevantDocumentsWithMetadataFilter`. Merge tool use instructions into one. Remove references to "keyword search" / "BM25 only" — describe as "relevance search" (the backend strategy is transparent to the LLM). |
| E9 | Two-tool prompt: instruct LLM to call `SearchByMetadata` first for candidate documents, then `SearchChunksInDocuments` for passage extraction. Add workflow step: "First search by topic/summary to identify relevant documents, then retrieve detailed passages from those documents." |
| E10 | Prompt instructs exactly 1 retrieval call per iteration (no follow-up in same turn). |

### 8.4 Hop strategy implementation

| Strategy | Implementation |
|---|---|
| `MultiHopBaseline` | Current iteration policy unchanged. LLM decides 1–3 iterations. |
| `SingleHop` | Set `tool_choice="none"` after iteration 0 (or cap `max_iterations=1` in graph). |
| `MultiHopMetadataFirst` | Register `SearchByMetadata` + `SearchChunksInDocuments`. Prompt guides LLM to use metadata tool first, chunk tool second. Iteration policy: `required` for iterations 0 and 1, `auto` thereafter. |

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

With 77 rubrics per experiment and 14 experiments, we have 1,078 total evaluation points.

**Paired comparisons:** each rubric is evaluated under every experiment, enabling paired
statistical tests (Wilcoxon signed-rank) between any two experiments on the same rubric set.

**Effect size:** Cohen's d or rank-biserial correlation to quantify practical significance
beyond p-values.

**Multiple comparisons:** Bonferroni or Holm-Bonferroni correction when comparing many
experiment pairs simultaneously.

### 9.5 Feature selection methodology

After the initial 14-experiment matrix is evaluated:

1. **Rank experiments** by mean rubric score across both workspaces.
2. **Identify top-3 configurations** and their shared factor levels.
3. **Ablation:** for each factor in the top configuration, run a variant with that factor
   removed/changed to its baseline level. If performance drops significantly, the factor
   contributes; if not, it can be dropped.
4. **Forward selection:** starting from the best single-signal experiment (E1, E2, or E3),
   greedily add factors that improve performance most, stopping when marginal gain < threshold.

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
