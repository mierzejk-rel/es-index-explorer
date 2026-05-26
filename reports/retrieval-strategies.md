# aiR Assist: Retrieval Strategies and Search Execution

**Purpose:** Reference for engineers and AI agents on how the aiR Assist RAG agent searches the
Elasticsearch index, what strategies are used, how results are processed, and what qna-service
endpoints are involved. Provides enough detail to replicate search behavior in Python against a
custom ES index.

---

## 1. Architecture: Agent → qna-service → Elasticsearch

The agent (`air-assist-agent`) does not query Elasticsearch directly. It calls qna-service MCP
tools, which orchestrate ES search, permission filtering, and result grouping.

```
User question
    │
    ▼
air-assist-agent (LangGraph v3)
    │  LLM decides which tool to call and generates the query
    ▼
McpToolProvider → qna-service MCP /v2
    │  injects subsetId, user auth headers
    ▼
DocumentProviderV2 → RetrievalStrategySettingsProvider (LaunchDarkly)
    │  selects RRF / BM25 / BM25+MMR
    ▼
RetrievalService → ElasticsearchClientWrapper
    │  dispatches to RrfSearchStrategy or Bm25SearchStrategy
    ▼
Elasticsearch
    │  returns ranked chunks
    ▼
Post-processing: security trim → top-K → first-chunk fetch → group by document
    │
    ▼
GroupedDocumentSearchResult → agent
    │
    ▼
Agent: normalize passages → merge across hops → XML for LLM → citations
```

---

## 2. MCP Tools: What the Agent Calls

### 2.1 Current Tools (V2, production)

The v3 agent defines two retrieval tools as Pydantic models in
[`tool_definitions.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/tool_definitions.py).
The LLM selects between them based on the user's question and system prompt guidance.

| Agent Tool Name | MCP Tool Name | Parameters | When Used |
|---|---|---|---|
| `GetRelevantDocuments` | `get_relevant_documents` | `query` (required) | General keyword/semantic search |
| `GetRelevantDocumentsWithMetadataFilter` | `get_relevant_documents_with_metadata_filter` | `query?`, `dateFrom?`, `dateTo?`, `emailParticipants?` | Date-scoped or email participant filtering |

Both tools receive `subsetId` injected by
[`McpToolProvider`](../../air-assist-agent/src/air_assist_agent/mcp/tool_provider.py) (line 246)
— hidden from the LLM's tool schema.

**Metadata filter tool — email participants shape:**

```python
{
    "participantsA": ["alice@example.com"],
    "participantsB": ["bob@example.com"],     # optional
    "direction": "AToB",                       # AToB | BToA | AnyDirection
    "matchStrategy": "Any",                    # Any | All
    "presenceRule": "Both"                     # Both | EitherSide
}
```

Source: [`EmailParticipantsFilter`](../../repos/qna-service/Source/Relativity.QnA.Domain/Models/EmailParticipantsFilter.cs)

**Metadata fallback:** If the metadata filter returns zero documents, the agent falls back to a
keyword-only search by expanding dates/emails into the query string. See
[`to_keyword_search_args()`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/tool_definitions.py)
(lines 257–287) and [`rag_agent.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/rag_agent.py) (lines 149–155).

### 2.2 Legacy / V1 Tools (deprecated)

| Tool | Version | Status |
|---|---|---|
| `get_relevant_documents` (flat chunks) | V1 | Superseded by V2 grouped response |
| `get_relevant_documents_v2` | Legacy (unversioned) | Same as V2 but returns only `.Documents` |
| `get_relevant_documents_with_include_exclude` | V1/V2 | Exposed but **not used** by current v3 agent tools |
| `get_relevant_documents_with_include_exclude_v2` | Legacy | Same |
| `filter_documents_using_metadata` | V1/Legacy | Object Manager-based, not ES search. Returns artifact IDs only. |
| `get_document_content` | All versions | DataGrid + ADLS read, not ES search |

MCP version routing: agent configs set `mcp_api_version = "v2"` in TOML. The MCP URL becomes
`{QNA_SERVICE_MCP_URL}/v2`. See
[`McpToolsetRegistry`](../../repos/qna-service/Source/Relativity.QnA.API/Mcp/McpToolsetRegistry.cs)
for version dispatch.

### 2.3 Tool Selection Logic

The LLM chooses the tool. The agent controls which tools are **available** to the LLM:

- If `UseMetadata` capability is `false` or absent → `GetRelevantDocumentsWithMetadataFilter` is
  excluded from the tool list
- Source: [`tool_selection.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/tool_selection.py)
  (lines 49–57)
- `UseMetadata` comes from the `rel-tool-capabilities` HTTP header, parsed in
  [`http_request_context.py`](../../air-assist-agent/src/air_assist_agent/services/models/http_request_context.py)

Iteration policy (v3 ReAct loop):
- Iteration 0 → `tool_choice="required"` (must search)
- Iterations 1–4 → `tool_choice="auto"` (may search again or answer)
- Iteration 5+ → `tool_choice="none"` (must answer)

---

## 3. Elasticsearch Search Strategies

Strategy selection happens entirely in qna-service, controlled by LaunchDarkly flag
`qna-service.retrieval.strategy.configuration`. The agent has no control over which ES strategy
runs.

### 3.1 RRF — Reciprocal Rank Fusion (default)

Hybrid search combining BM25 text search and kNN vector search, fused by Elasticsearch's RRF
retriever.

**ES query shape:**

```json
{
    "size": 100,
    "retriever": {
        "rrf": {
            "rank_window_size": 100,
            "retrievers": [
                {
                    "standard": {
                        "query": {
                            "multi_match": {
                                "fields": ["body", "title"],
                                "query": "<user query>"
                            }
                        }
                    }
                },
                {
                    "knn": {
                        "field": "embedding",
                        "k": 100,
                        "num_candidates": 250,
                        "query_vector": [/* 384-dim float array */]
                    }
                }
            ]
        }
    }
}
```

**Index fields used:**

| Field | Role in RRF |
|---|---|
| `body` | BM25 text scoring (standard analyzer) |
| `title` | BM25 text scoring (standard analyzer) — but never populated, so contributes nothing |
| `embedding` | kNN cosine similarity via `bbq_hnsw` index |
| `subsetIds` | Always filtered (term match on subset GUID) |
| `documentId` | Include/exclude filtering when specified |
| `metadata.*` | Date range and email participant filters when metadata tool is used |

**Query embedding:** qna-service generates the query embedding before calling ES, using the
same model as indexing (`intfloat/multilingual-e5-small`). Source:
[`RetrievalService.RetrieveWithRrfAsync()`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/RetrievalService.cs)

**Three ES query paths in RrfSearchStrategy** (depending on filters):

| Condition | Path | Implementation |
|---|---|---|
| No subset, no metadata, no include/exclude | Template-based: deserialize JSON from LD flag with `%QUERY%`/`%EMBEDDINGS%` placeholders | [`RrfSearchStrategy.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/RrfSearchStrategy.cs) lines 43–69 |
| Subset set, no include/exclude | Bool `multi_match` + parallel `Knn` with filter | Same file, lines 99–129 |
| Include/exclude or metadata filters | Fluent RRF with `Standard` + `Knn` retrievers, both filtered | Same file, lines 188–206 |

Source: [`RrfSearchStrategy.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/RrfSearchStrategy.cs),
[`RetrievalStrategyDefaults.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyDefaults.cs)

### 3.2 BM25 — Keyword Only

Text-only search, no embeddings. Used when the LD flag selects `Bm25Search`.

**ES query shape:**

```json
{
    "size": 100,
    "query": {
        "bool": {
            "filter": [{"term": {"subsetIds": "<subset GUID>"}}],
            "must": [{
                "multi_match": {
                    "fields": ["body", "title"],
                    "query": "<user query>"
                }
            }]
        }
    }
}
```

**Index fields used:** `body`, `title` (BM25 scoring), `subsetIds` (filter), `documentId`
(include/exclude), `metadata.*` (date/email filters).

Source: [`Bm25SearchStrategy.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/Bm25SearchStrategy.cs)

### 3.3 BM25 + MMR — Keyword Search with Diversity Reranking

Same BM25 ES query as above, followed by application-level MMR reranking in qna-service.

**Flow:**
1. BM25 search against ES (same as 3.2)
2. Embed the query + batch-embed all retrieved chunk bodies
3. Apply MMR: greedily select chunks maximizing `λ·sim(query, chunk) − (1−λ)·max(sim(chunk, already_selected))`
4. Take top-K from MMR-ranked list

**Parameters:**

| Parameter | Default |
|---|---|
| `mmr_lambda` | 0.5 |
| `mmr_top_k` | 25 |
| `embedding_batch_size` | 25 |

Source: [`RetrievalService.RetrieveWithBm25MmrAsync()`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/RetrievalService.cs)
(lines 82–163),
[`MaximumMarginalRelevanceSelector.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/MaximumMarginalRelevanceSelector.cs)

### 3.4 Strategy Configuration (LaunchDarkly)

Flag: `qna-service.retrieval.strategy.configuration`

JSON payload shape:

```json
{
    "strategy": "RrfSearch",
    "rrf_search_settings": {
        "results_from_rrf": 100,
        "rank_window": 100,
        "num_candidates": 250,
        "results_sent_to_gpt": 25
    }
}
```

Source: [`RetrievalStrategySettingsProvider.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/RetrievalStrategySettingsProvider.cs),
[`RetrievalStrategyFlagPayload.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyFlagPayload.cs)

| Strategy Key | ES Class | App-Layer Post-Processing |
|---|---|---|
| `RrfSearch` | `RrfSearchStrategy` | None (fusion in ES) |
| `Bm25Search` | `Bm25SearchStrategy` | None |
| `Bm25SearchWithMmr` | `Bm25SearchStrategy` | MMR reranking |

### Strategy Summary

| Strategy | Text Search | Vector Search | Fusion | Post-ES Reranking | Fields Queried |
|---|---|---|---|---|---|
| **RRF** (default) | `multi_match` on `body`, `title` | `knn` on `embedding` (k=100, candidates=250) | ES RRF (rank_window=100) | None | `body`, `title`, `embedding`, `subsetIds`, `documentId`, `metadata.*` |
| **BM25** | `multi_match` on `body`, `title` | None | None | None | `body`, `title`, `subsetIds`, `documentId`, `metadata.*` |
| **BM25+MMR** | `multi_match` on `body`, `title` | None (ES), then embed post-fetch | None (ES) | MMR (λ=0.5, topK=25) | `body`, `title`, `subsetIds`, `documentId`, `metadata.*` |

---

## 4. Metadata Filtering

### 4.1 Filter Types

Implemented in
[`ElasticMetadataFilterBuilder.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticMetadataFilterBuilder.cs):

| Filter Type | ES Query | Operates On |
|---|---|---|
| String `Contains` | `wildcard` (case-insensitive) | `metadata.{fieldKey}` |
| String `Equals` | `term` | `metadata.{fieldKey}` |
| Date `GreaterThanOrEqual` | `range` (gte) | `metadata.{fieldKey}` |
| Date `LessThan` | `range` (lt) | `metadata.{fieldKey}` |
| Date `LessThanOrEqual` | `range` (lte) | `metadata.{fieldKey}` |
| Email participants | Compound `bool` with `wildcard` queries | `metadata.emailFrom`, `emailTo`, `emailCc`, `emailBcc` |

For email fields, ES `wildcard` queries match against each element of a multi-valued `keyword`
array independently (e.g., `metadata.emailTo: ["alice@example.com", "bob@example.com"]` — a
wildcard for `*alice*` matches). Documents where the field is `null` (non-email documents) are
naturally excluded. See `air-assist-elasticsearch-index.md` §4.2.4 for the full registry of
metadata fields, multi-value semantics, and null handling.

### 4.2 Subset Filter (always applied)

Every search scopes to a subset via a `term` filter on `subsetIds`:

```csharp
filters.Add(q => q.Term(t => t.Field("subsetIds").Value(subsetId.ToString())));
```

Source: [`ElasticQueryFilterBuilder.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticQueryFilterBuilder.cs)

### 4.3 ES-Native vs Object Manager Routing

Controlled by LD flag `qna-service.metadata-filtering.use-elasticsearch`:

| Condition | Path | Behavior |
|---|---|---|
| LD flag on, has query | ES-native: filters embedded in RRF/BM25 query | Full ES search with metadata filters |
| LD flag on, no query | ES-native: `GetFirstChunksByMetadataFilterAsync` | Filter-only, returns first chunk per matching doc |
| LD flag off, has query | OM hybrid: OM resolves doc IDs → ES with `IncludedDocumentList` | Two-step: permission-aware OM first, then ES |
| LD flag off, no query | OM only: doc IDs → `GetFirstChunksForDocumentsAsync` | No ES search, just chunk retrieval |

Source: [`DocumentProviderV2.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs) (lines 501–526)

---

## 5. Post-Search Processing (qna-service)

After ES returns ranked chunks, qna-service applies these steps in order:

| Step | What | Where |
|---|---|---|
| 1. Capture `retrievedDocumentIds` | All unique doc IDs from ES result (pre-filter) | `RetrievalService` |
| 2. Security trim | Object Manager `GetUserDocumentArtifactIds` — keep only docs the user can access | [`DocumentProviderV2.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs) lines 120–151 |
| 3. Top-K trim | `.Take(config.ResultsSentToGpt)` (default 25 chunks) | Same file, lines 153–155 |
| 4. First-chunk fetch | Separate ES query for chunk 0 per document — see §5.1 below | `GetFirstChunksByDocumentId()` line 428 |
| 5. Group by document | Chunks grouped by `documentId`, sorted by first appearance in ES results; within group sorted by `chunkId`. First chunk attached as a separate property. | `GroupChunksByDocument()` lines 305–329 |
| 6. Return | `GroupedDocumentSearchResult { Documents, RetrievedDocumentIds }` | MCP response |

### 5.1 First-Chunk Injection

The first chunk (chunk 0) of each document is **not part of the ranked search results**. It is
fetched via a **separate ES query** and attached as a distinct `FirstChunk` property on each
document group. This ensures the LLM always has the opening context of every returned document,
even when only middle or later chunks matched the search query.

**How it works:**

1. After top-K trim, `GetFirstChunksByDocumentId()` (line 428 of `DocumentProviderV2.cs`)
   collects all unique `documentId` values from the surviving chunks.
2. It calls `ElasticsearchClientWrapper.GetFirstChunksForDocumentsAsync()` — this queries ES
   for documents matching those IDs where `chunkId == 0 OR chunkId does not exist` (the
   "does not exist" clause handles legacy documents that predate the `chunkId` field).
3. Returns `Dictionary<int, string>` mapping each `documentId` to the body text of chunk 0.
4. `GroupChunksByDocument()` (line 305) assembles the final `GroupedDocumentChunks`:
   - `FirstChunk` = chunk 0 body text from the dictionary (line 324)
   - `RetrievedChunks` = the ranked chunks that actually matched the search, sorted by `chunkId`

**Result structure per document:**

```json
{
    "docId": 1045678,
    "controlNumber": "DOC-001",
    "firstChunk": "Opening paragraph of the document...",
    "retrievedChunks": [
        {"id": 3, "documentId": 1045678, "controlNumber": "DOC-001", "content": "...matched chunk 3..."},
        {"id": 7, "documentId": 1045678, "controlNumber": "DOC-001", "content": "...matched chunk 7..."}
    ]
}
```

**`FirstChunk` and `RetrievedChunks` are independent — chunk 0 is _always_ fetched separately
regardless of whether it was already in the search results. There is no deduplication at the
qna-service level.** If chunk 0 also matched the search, the same text appears in both places.

**What this means for the LLM:** When chunk 0 matched the search, the agent sends the same
content twice in the XML — once under `<first_chunk>` (context) and once under
`<retrieved_chunks>` with `<chunk_id>0</chunk_id>` (citable evidence):

```xml
<grouped_chunks>
    <doc_id>1045678</doc_id>
    <first_chunk>Opening paragraph...</first_chunk>
    <retrieved_chunks>
        <retrieved_chunk>
            <chunk_id>0</chunk_id>
            <content>Opening paragraph...</content>   <!-- same text as first_chunk -->
        </retrieved_chunk>
        <retrieved_chunk>
            <chunk_id>3</chunk_id>
            <content>Another matched chunk...</content>
        </retrieved_chunk>
    </retrieved_chunks>
</grouped_chunks>
```

The duplication is intentional. The system prompt instructs the LLM that `first_chunk` provides
context but is **not a candidate for citation**: *"The passage provided in the first_chunk is not
a candidate for being referenced in the final response"* (TOML config). The LLM should only cite
from `retrieved_chunks`. However, if the LLM does cite `[doc_id:0]`, the citation validation
system resolves it against `first_chunk` content (when non-empty). Source:
[`citation_validation.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/citation_validation.py) lines 377–379,
[`question_answer_v3.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/schemas/question_answer_v3.py) lines 144–151.

If the ES lookup for first chunks fails, the operation degrades gracefully — `FirstChunk` is set
to an empty string and the document group is still returned with its matched chunks.

Source: [`DocumentProviderV2.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs)
lines 157, 284, 324, 428–455;
[`ElasticsearchClientWrapper.GetFirstChunksForDocumentsAsync()`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/ElasticsearchClientWrapper.cs)
lines 156–231

**Telemetry logged:**
- `ChunksRetrievedCount` — from ES before any filtering
- `ChunksUserCanAccessCount` — after security trim
- `ChunksToSendToRagCount` — after top-K

---

## 6. Post-Search Processing (agent-side)

After receiving `GroupedDocumentSearchResult` from qna-service, the v3 agent applies:

| Step | What | Source |
|---|---|---|
| 1. Parse response | `GroupedChunks.model_validate(doc)` from MCP `structured_content["documents"]` | [`rag_agent.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/rag_agent.py) lines 147–148 |
| 2. Normalize passages | Collapse whitespace, strip text in each chunk | [`passage_normalization.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/passage_normalization.py) |
| 3. Accumulate across hops | Extend document list across multiple retrieval iterations (ReAct loop) | `rag_agent.py` lines 473–505 |
| 4. Merge same-doc chunks | Deduplicate chunks by `chunk_id` when same document found in multiple hops | [`chunks.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/chunks.py) |
| 5. Format as XML | `GroupedChunks.to_xml()` — each document becomes an XML passage element for the LLM tool message | [`document.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/schemas/document.py) |
| 6. LLM generates answer | System prompt + XML passages + conversation history → answer with citations | `rag_agent.py` |
| 7. Citation validation | Extract `[doc_id:chunk_id]` citations, validate against retrieved docs, normalize | [`citation_validation.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/citation_validation.py) |
| 8. Markdown cleanup | Clean up LLM output formatting | `rag_agent.py` lines 766–789 |

The agent tracks two ID sets for reporting:
- `retrieved_doc_ids` — raw ES doc IDs before MMR/top-K (from `retrievedDocumentIds` in MCP response)
- `relevant_doc_ids` — doc IDs of documents actually shown to the LLM (post-filtering)

---

## 7. Query Rewriting

**There is no pre-search query rewrite step in the current pipeline.** A v1 query rewrite module
existed but was removed (commit `a61e1a0`).

The LLM itself acts as the query rewriter: in the v3 ReAct loop, the model reads the user's
question and conversation history, then generates the `query` argument for the retrieval tool
call. This is an implicit rewrite — the model may rephrase, decompose, or focus the query based
on context.

The only "rewrite" that persists is qna-service's `contentRewrite` field on conversation
messages, used when loading conversation history in
[`rag_agent_service_v2.py`](../../air-assist-agent/src/air_assist_agent/services/rag_agent_service_v2.py) (line 138).

---

## 8. Security Trimming

Security trimming is **not implemented in the agent or in Elasticsearch**. It is an
application-level permission check in qna-service.

**How it works:**
1. ES returns chunks (up to `results_from_rrf` or `max_results` — typically 100)
2. qna-service extracts unique `documentId` values
3. Calls Object Manager with user credentials: `'Artifact ID' IN [docIds]`
4. OM returns only the document IDs the authenticated user has permission to see
5. Chunks whose `documentId` is not in the accessible set are dropped

**Source:**
[`DocumentProviderV2.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs) lines 120–151,
[`ObjectManagerClient.GetUserDocumentArtifactIds()`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ObjectManager/ObjectManagerClient.cs) lines 51–71

---

## 9. Data Returned to the Agent

The MCP response is a `GroupedDocumentSearchResult`:

```json
{
    "documents": [
        {
            "documentId": 1045678,
            "controlNumber": "DOC-001",
            "firstChunk": { "id": 0, "content": "..." },
            "chunks": [
                { "id": 0, "documentId": 1045678, "controlNumber": "DOC-001", "content": "chunk text..." },
                { "id": 3, "documentId": 1045678, "controlNumber": "DOC-001", "content": "another chunk..." }
            ]
        }
    ],
    "retrievedDocumentIds": [1045678, 1045679, 1045680, ...]
}
```

**What each field means:**
- `documents` — post-security-trim, post-top-K, grouped by document. These are what the LLM sees.
- `retrievedDocumentIds` — all doc IDs from the initial ES result, before any filtering. Used for
  analytics and eval metrics.
- `firstChunk` — chunk 0 of each document, fetched separately. Provides document-level context
  even if chunk 0 wasn't in the search results.
- `chunks` — the ranked chunks that matched the search, sorted by `chunkId` within each document group.

---

## 10. Configuration Defaults Summary

| Parameter | Default | Controlled By | Affects |
|---|---|---|---|
| Retrieval strategy | `RrfSearch` | LD `qna-service.retrieval.strategy.configuration` | Which ES query shape runs |
| `results_from_rrf` / `max_results` | 100 | LD flag payload | Max chunks from ES |
| `results_sent_to_gpt` | 25 | LD flag payload | Chunks sent to agent after security trim |
| `rank_window` | 100 | LD flag payload | RRF rank window size |
| `num_candidates` | 250 | LD flag payload | kNN candidate pool |
| `mmr_lambda` | 0.5 | LD flag payload | MMR diversity vs relevance tradeoff |
| `mmr_top_k` | 25 | LD flag payload | Chunks selected by MMR |
| `UseMetadata` | per-request | `rel-tool-capabilities` header | Whether metadata filter tool is available to LLM |
| ES metadata filtering | per-tenant | LD `qna-service.metadata-filtering.use-elasticsearch` | ES-native vs OM hybrid path |
| MCP API version | `v2` | Agent TOML config `mcp_api_version` | Which qna-service toolset is used |

---

## 11. Implications for Custom Index Design

To support the same retrieval strategies against a custom ES index, the index must provide:

| Capability | Required Fields | Required Configuration |
|---|---|---|
| BM25 text search | `body` (type `text`) | Standard analyzer (default) |
| kNN vector search (for RRF) | `embedding` (type `dense_vector`, dims=384, similarity=cosine) | `bbq_hnsw` or equivalent ANN index |
| Subset scoping | `subsetIds` (type `keyword`, multi-valued) | Term filter support |
| Document grouping | `documentId` (type `integer`) | Collapse / terms filter |
| Chunk ordering | `chunkId` (type `integer`) | Sort within document group |
| First-chunk retrieval | `chunkId` field with value `0` for first chunk | Term filter `chunkId == 0` |
| Metadata date filtering | `metadata.primaryDateTime` (type `date`) | Range filter |
| Metadata email filtering | `metadata.emailFrom/To/Cc/Bcc` (type `keyword`) | Wildcard filter |
| Security trimming | `documentId` mapping to Relativity ArtifactId | Object Manager permission check (external) |

Fields present in the index but not used by any current retrieval strategy:
- `title` — in BM25 `multi_match` fields list but never populated, so never contributes to scoring
- `chunkSize` — stored only, not queried
- `controlNumber` — returned in results for display, not queried
- `documentModifyTime` — stored only, not queried
- `createdAt` — stored only, never populated, not queried

---

## 12. Source Code Reference

| Topic | File |
|---|---|
| Agent tool definitions (v3) | [`air-assist-agent/.../v3/tool_definitions.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/tool_definitions.py) |
| Agent tool selection | [`air-assist-agent/.../v3/tool_selection.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/tool_selection.py) |
| Agent RAG graph (v3) | [`air-assist-agent/.../v3/rag_agent.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/rag_agent.py) |
| Agent MCP call | [`air-assist-agent/.../mcp/tool_provider.py`](../../air-assist-agent/src/air_assist_agent/mcp/tool_provider.py) |
| Agent passage normalization | [`air-assist-agent/.../passage_normalization.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/passage_normalization.py) |
| Agent chunk merging | [`air-assist-agent/.../v3/chunks.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/chunks.py) |
| Agent citation validation | [`air-assist-agent/.../v3/citation_validation.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/citation_validation.py) |
| Agent document model | [`air-assist-agent/.../schemas/document.py`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/schemas/document.py) |
| qna-service MCP V2 tools | [`qna-service/.../Mcp/Tools/V2/DocumentProviderTool.cs`](../../repos/qna-service/Source/Relativity.QnA.API/Mcp/Tools/V2/DocumentProviderTool.cs) |
| qna-service retrieval orchestration | [`qna-service/.../Services/RetrievalService.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/RetrievalService.cs) |
| qna-service document provider V2 | [`qna-service/.../Services/DocumentProviderV2.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs) |
| qna-service strategy config | [`qna-service/.../Services/RetrievalStrategySettingsProvider.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/RetrievalStrategySettingsProvider.cs) |
| RRF strategy implementation | [`qna-service/.../Strategies/RrfSearchStrategy.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/RrfSearchStrategy.cs) |
| BM25 strategy implementation | [`qna-service/.../Strategies/Bm25SearchStrategy.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/Bm25SearchStrategy.cs) |
| MMR selector | [`qna-service/.../Services/MaximumMarginalRelevanceSelector.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Services/MaximumMarginalRelevanceSelector.cs) |
| Metadata filter builder | [`qna-service/.../Filters/ElasticMetadataFilterBuilder.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticMetadataFilterBuilder.cs) |
| Query filter builder | [`qna-service/.../Filters/ElasticQueryFilterBuilder.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticQueryFilterBuilder.cs) |
| LD flag keys | [`qna-service/.../Constants/LaunchDarklyFeatureFlagKeys.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Constants/LaunchDarklyFeatureFlagKeys.cs) |
| Strategy config docs | [`qna-service/docs/retrieval-strategy-configuration.md`](../../repos/qna-service/docs/retrieval-strategy-configuration.md) |
| MCP versioning docs | [`qna-service/docs/mcp.md`](../../repos/qna-service/docs/mcp.md) |
