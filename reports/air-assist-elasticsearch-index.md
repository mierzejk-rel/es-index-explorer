# aiR Assist Elasticsearch Index: Structure and Configuration

**Purpose:** Reference document for engineers and AI agents working on aiR Assist Elasticsearch
index design, expansion, and custom index creation.

**Primary live index inspected:** `51b1e1f7-f607-4967-a439-cf9babafab2c-1030345-qna-subsetting`
(tenant `51b1e1f7-f607-4967-a439-cf9babafab2c`, workspace `1030345`)

Generated from: `es-index-explorer` inspection tool + source code in `embedding-service`,
`qna-service`, `elasticsearch-infra`, and `air-assist-hub` repositories.

---

## 1. Index Naming Convention

Three naming patterns exist, in different states of use:

### Pattern 1: Production / regression — `{tenantId}-{workspaceId}-qna-subsetting` (current)

The active pattern for real tenant indices in regression and production environments. One index
per workspace per tenant. This is what `embedding-service` creates today and what `qna-service`
searches against.

Defined in
[`embedding-service/.../Common/IndexNames.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Common/IndexNames.cs):

```csharp
// Current — subsetting index used in regression and production
public static string IndexNameForSubsetting(TenantId tenantGuid, WorkspaceId workspaceId)
    => $"{tenantGuid}-{workspaceId}-qna-subsetting";
```

Mirrored in
[`qna-service/.../Common/IndexNames.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Common/IndexNames.cs).

Covered by the ES permission grant:
```json
{ "names": ["*-qna-*"], "privileges": ["all", "read"] }
```

The live index inspected for this report (`51b1e1f7-f607-4967-a439-cf9babafab2c-1030345-qna-subsetting`)
follows this pattern and is representative of the production schema.

### Pattern 2: Applied Science / research — `as-*` (current, separate purpose)

Used exclusively by the Applied Science team for development, research, and internal testing
indices. Not used in production tenant environments. Examples from the live cluster: `as-exp-emc2`,
`as-rubrics-*`, `as-rag-*`, `as-exp-msmarco-*`.

Covered by a separate, broader ES permission grant:
```json
{ "names": ["as-*"], "privileges": ["all", "create", "create_index", "manage", "monitor", "read", "write"] }
```

This is the correct prefix for any new custom index created outside of the production indexing
pipeline (e.g. for experiments or a custom aiR Assist variant).

### Pattern 3: `{tenantId}-qna-{workspaceId}` (deprecated)

The original per-workspace index pattern, used before the concept of subsetting was introduced.
Replaced by Pattern 1 once subsetting became the standard. Still defined in the codebase for
reference but no longer created for new workspaces:

```csharp
// DEPRECATED — pre-subsetting main index
public static string IndexName(TenantId tenantGuid, int workspaceId)
    => $"{tenantGuid}-qna-{workspaceId}";
```

---

## 2. Index Creation: Who and How

**The `embedding-service` creates the index on demand** — lazily, the first time a workspace is
indexed. See
[`embedding-service/.../Elasticsearch/IndexManagementService.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Elasticsearch/IndexManagementService.cs):

```csharp
public async Task<Result<string>> CreateSubsettingIndexIfNotExists(
    TenantId tenantId, WorkspaceId workspaceId, CancellationToken cancellationToken)
{
    string indexName = IndexNames.IndexNameForSubsetting(tenantId, workspaceId);
    // ... checks Exists, creates if missing
    Result createResult = await elasticSearchClientWrapper.CreateIndex(tenantId, indexName);
}
```

This is called by Temporal workflow activities during the indexing pipeline (see
[`ConsumerActivities.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Activities/ConsumerActivities.cs)).

**`qna-service` does not create indices.** It only performs search/retrieval against
already-existing indices. See
[`ElasticsearchClientWrapper.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/ElasticsearchClientWrapper.cs)
— contains only search and retrieval methods, no `CreateIndex`.

---

## 3. Base Mapping: Hardcoded at Index Creation

The base field mapping is **hardcoded** in
[`embedding-service/.../Elasticsearch/ElasticsearchClientWrapper.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Elasticsearch/ElasticsearchClientWrapper.cs),
lines 105–157:

```csharp
TypeMapping mappings = new()
{
    Properties = new Properties
    {
        { "embedding",          new DenseVectorProperty() },
        { "documentId",         new IntegerNumberProperty() },
        { "documentModifyTime", new DateProperty() },
        { "createdAt",          new DateProperty() },
        { "body",               new TextProperty() },
        { "title",              new TextProperty() },
        { "controlNumber",      new KeywordProperty() },
        { "subsetIds",          new KeywordProperty() },
        { "chunkId",            new IntegerNumberProperty() },
        { "chunkSize",          new IntegerNumberProperty() }
    }
};
```

This is the **only** code that defines the base schema. No template files, no YAML, no external
schema registry.

---

## 4. Field Reference

### 4.1 Core Fields (always present, set at index time)

| Field | ES Type | Source in Code | Search/Retrieval Role |
|-------|---------|---------------|----------------------|
| `body` | `text` | Chunk text, from [`IndexingDocument.ToElasticDocuments()`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Models/IndexingDocument.cs) | **BM25 full-text search.** Primary text field queried by `multi_match` in qna-service. Default analyzer (standard). No custom analyzer configured at index creation. |
| `title` | `text` | Set in `ElasticDocument.cs` model but **not assigned** in `ToElasticDocuments()` — always empty in practice | Included in qna-service BM25 `multi_match` queries (`fields: ["body", "title"]`) but effectively a no-op for retrieval today. |
| `embedding` | `dense_vector` | `_embeddings[i]` — float array from model inference | **kNN / vector search.** Used in RRF hybrid retrieval by qna-service. See section 5 for vector configuration details. |
| `documentId` | `integer` | `DocumentId` from Relativity workspace | **Numeric equality/range filter and document collapse.** Used in qna-service to group chunks by document and to check which documents are indexed. |
| `chunkId` | `integer` | `_chunks[i].Index` (0-based position within document) | Identifies which chunk within a document. Used to retrieve only the first chunk (`chunkId == 0`) for document-level operations. |
| `chunkSize` | `integer` | `Encoding.UTF8.GetByteCount(chunkText) * 2` (UTF-16 approximation) | Stored for subset size statistics. Not directly used in retrieval queries. |
| `controlNumber` | `keyword` | `ControlNumber` from Relativity workspace | **Exact-match keyword filter.** Stored in doc_values for sorting/aggregation. Not tokenized, not used for full-text search. |
| `subsetIds` | `keyword` | List of subset GUIDs appended to each chunk | **Keyword filter — critical for multi-tenancy.** Every query in qna-service filters on `subsetIds` to scope results to a specific saved search subset. Supports multiple values per chunk. |
| `documentModifyTime` | `date` | `ModifyTime` from Relativity workspace metadata | Document last-modified timestamp. Stored as date. Not actively queried in current retrieval strategies but available for filtering. |
| `createdAt` | `date` | Set in `ElasticDocument` model but **not assigned** in `ToElasticDocuments()` | Present in the mapping and visible in live index. Not populated during indexing — always default/zero value in practice. |

### 4.2 Dynamic Metadata Fields (`metadata.*`)

> **Not present in the inspected index.** The mapping of workspace `1030345`
> (`51b1e1f7-...-1030345-qna-subsetting`) contains no `metadata.*` fields — confirmed by both
> the REST mapping query and the `es-index-explorer` inspection. This workspace either has no
> Relativity fields configured for metadata mapping, or the `EnsureMetadataFieldMappingsAsync`
> activity has not been invoked for it. The description below documents the capability as it
> exists in the code.

### 4.2.1 Metadata Field Registry

The set of supported metadata keys is **hardcoded** in embedding-service as a static registry.
These are NOT extracted from document text. They are explicitly configured per workspace by an
administrator who maps each metadata key to a specific Relativity workspace field by its
`FieldArtifactId`.

Source: [`MetadataFieldDefinitionProvider.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Services/MetadataFieldDefinitionProvider.cs),
[`MetadataFieldKeys.cs`](../../embedding-service/Source/Relativity.Embedding.Domain/Constants/MetadataFieldKeys.cs)

**Complete metadata field registry:**

| Metadata Key | Display Name | Required | Compatible R1 Field Types | ES Mapping Type | ES Field Path | Purpose |
|---|---|---|---|---|---|---|
| `primaryDateTime` | Primary Date/Time | Yes | `Date` | `date` | `metadata.primaryDateTime` | Date range filtering |
| `emailFrom` | Email From | Yes | `FixedLengthText`, `LongText` | `keyword` | `metadata.emailFrom` | Email sender filtering |
| `emailTo` | Email To | Yes | `FixedLengthText`, `LongText` | `keyword` | `metadata.emailTo` | Email recipient filtering |
| `emailCc` | Email CC | No | `FixedLengthText`, `LongText` | `keyword` | `metadata.emailCc` | Email CC filtering |
| `emailBcc` | Email BCC | No | `FixedLengthText`, `LongText` | `keyword` | `metadata.emailBcc` | Email BCC filtering |
| `documentName` | Document Name | No | `FixedLengthText`, `LongText` | `keyword` | `metadata.documentName` | Claire UI display (gated by LaunchDarkly) |

"Required" means the field must be mapped before indexing can proceed for that workspace.

### 4.2.2 Metadata Lifecycle

The metadata fields are added to the ES index **after** the base index is created, via a separate
`PUT /{index}/_mapping` call. The full lifecycle:

1. **Configuration** — Admin calls `POST /workspaces/{id}/metadata-mappings` on embedding-service
   with entries like `[{metadataMappingKey: "emailFrom", fieldArtifactId: 12345}, ...]`. The
   `FieldArtifactId` is the Relativity workspace field ID to read from.
   Source: [`MetadataMappingEntry.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Models/MetadataMapping/MetadataMappingEntry.cs)
2. **Validation** — embedding-service checks that the R1 field type is compatible with the
   metadata key (e.g., `emailFrom` requires `FixedLengthText` or `LongText`).
   Source: [`MetadataFieldKey.EnsureCompatibleWith()`](../../embedding-service/Source/Relativity.Embedding.Domain/ValueObjects/MetadataFieldKey.cs)
3. **Storage** — Mapping saved to Postgres (survives across indexing runs).
4. **ES schema update** — On the next indexing run, `ApplyMetadataFieldSchemaToIndexAsync` calls
   `EnsureMetadataFieldMappingsAsync()` which issues `PUT /{index}/_mapping` to add the
   `metadata.*` fields to the ES index.
   Source: [`ElasticsearchClientWrapper.EnsureMetadataFieldMappingsAsync()`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Elasticsearch/ElasticsearchClientWrapper.cs) (lines 159–207)
5. **Population** — During indexing, Object Manager is queried for those R1 fields per document.
   Values are stored under `metadata.*` on every chunk of each document.

If no metadata mappings are configured for a workspace, no `metadata.*` fields exist in the ES
index and no `PUT /_mapping` is issued.

### 4.2.3 Relativity Field Type to ES Property Type Mapping

```csharp
return parsed switch
{
    RelativityFieldType.FixedLengthText or RelativityFieldType.LongText  => new KeywordProperty(),
    RelativityFieldType.Date                                              => new DateProperty(),
    RelativityFieldType.WholeNumber                                       => new LongNumberProperty(),
    RelativityFieldType.Decimal or RelativityFieldType.Currency           => new DoubleNumberProperty(),
    RelativityFieldType.YesNo                                             => new BooleanProperty(),
    RelativityFieldType.SingleChoice or RelativityFieldType.MultipleChoice => new KeywordProperty(),
    RelativityFieldType.SingleObject or RelativityFieldType.MultipleObject => new KeywordProperty(),
    RelativityFieldType.User                                              => new KeywordProperty(),
    RelativityFieldType.File                                              => new KeywordProperty(),
    _ /* fallback */                                                      => new KeywordProperty(),
};
```

All metadata fields are nested under a top-level `metadata` object:
`metadata.{metadata_key}`. The `ElasticDocument` model stores them as
`Dictionary<string, object?> Metadata` — see
[`ElasticDocument.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Models/Elasticsearch/ElasticDocument.cs).

### 4.2.4 Multi-Value and Null Handling

Metadata values are stored per document (all chunks share the same values). The value handling:

| Condition | ES Value | Example |
|---|---|---|
| Single value | Scalar string | `"metadata": {"emailFrom": "alice@example.com"}` |
| Multiple values | JSON array of strings | `"metadata": {"emailTo": ["alice@example.com", "bob@example.com"]}` |
| Field configured but no value for this document | `null` | `"metadata": {"emailFrom": null}` |
| Field not configured for this workspace | Key absent entirely | No `metadata` object in document |

Source: [`IndexingDocument.BuildElasticMetadata()`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Models/IndexingDocument.cs)
(lines 276–298)

For email fields (`emailFrom`, `emailTo`, `emailCc`, `emailBcc`), multiple values come from the
Relativity field value — they arrive as separate list entries from Object Manager, not as a
semicolon-delimited string. Each array element is independently queryable by ES `keyword`
`term` or `wildcard` queries. A `null` value means the document is not an email (or the field
has no value) — ES `wildcard` queries will not match `null`, so non-email documents are
naturally excluded from email-participant filter results.

### 4.2.5 Agent Awareness of Metadata Fields

The agent does NOT know whether metadata fields exist for a given workspace index. The
`GetRelevantDocumentsWithMetadataFilter` tool is always available (when the `UseMetadata`
capability header is enabled). If the agent sends email participant filters for a workspace with
no metadata mappings, qna-service queries ES targeting `metadata.emailFrom` etc. — which don't
exist in the index — and gets zero matches. The agent's fallback logic then converts the failed
metadata search into a keyword-only search. See
[`tool_definitions.py` `to_keyword_search_args()`](../../air-assist-agent/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/tool_definitions.py)
(lines 257–287).

Metadata fields are used for **filtering** in qna-service's `DocumentProviderV2`, constructed by
[`ElasticMetadataFilterBuilder.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticMetadataFilterBuilder.cs).
See `retrieval-strategies.md` §4 for query-level details.

---

## 5. The `embedding` Field and Vector Index Configuration

### 5.1 What embedding-service configures

The service creates `embedding` as a bare `DenseVectorProperty()` — no `dims`, no `similarity`,
no `index_options` specified in code:

```csharp
{ "embedding", new DenseVectorProperty() }
```

### 5.2 What the live index actually shows

The live index (inspected via `es-index-explorer`) shows:

```json
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
```

### 5.3 Where these settings come from

**These settings are NOT configured in `embedding-service` source code.** They are most likely
applied by an Elasticsearch **index template** configured at the cluster level, managed by the
`elasticsearch-infra` repository
([`helm/eck-stack/`](../../elasticsearch-infra/helm/eck-stack/)). No template files defining
these settings were found in the currently available repos. This is owned by the
**Search Infrastructure** team.

The embedding model in use is `intfloat/multilingual-e5-small`, which produces 384-dimensional
vectors — matching `dims: 384`. This is confirmed in
[`air-assist-hub/architecture/embedding-service.md`](../../air-assist-hub/architecture/embedding-service.md)
and the `EmbeddingConfiguration` section of `embedding-service` config.

### 5.4 Implications for retrieval

| Setting | Value | Implication |
|---------|-------|-------------|
| `similarity` | `cosine` | Scores normalized embeddings by cosine similarity. The embedding model output should be L2-normalized for this to be well-calibrated. |
| `index_options.type` | `bbq_hnsw` | Binary Quantized HNSW — a quantization scheme that compresses float32 vectors to binary for the ANN graph traversal, then uses `rescore_vector` for re-ranking. Faster search, smaller memory footprint than plain HNSW. |
| `index_options.m` | `16` | Number of bidirectional links per HNSW node. Controls graph connectivity. Standard production value (higher = better recall, higher memory). |
| `index_options.ef_construction` | `100` | Candidates considered during graph construction. Higher = better graph quality, slower indexing. |
| `rescore_vector.oversample` | `3.0` | After initial ANN retrieval, re-score `3x` more candidates using full-precision vectors. Restores recall lost to binary quantization. |

---

## 6. Index Settings (Non-Mapping)

From live index inspection:

| Setting | Value | Source |
|---------|-------|--------|
| `number_of_shards` | `1` | **Likely cluster default or index template** — not set in `embedding-service` `CreateIndex` call. |
| `number_of_replicas` | `1` | **Likely cluster default or index template** — not set by embedding-service. |
| `routing.allocation.include._tier_preference` | `data_content` | Standard Elasticsearch ILM tier routing — **cluster default**. |
| Analysis settings | None (empty) | No custom analyzers, tokenizers, or filters are configured. Both `body` and `title` use Elasticsearch's **default standard analyzer** (lowercase + standard tokenizer). |

---

## 7. How Search Works Against This Index

### 7.1 Retrieval strategies in qna-service

The retrieval strategies are defined in
[`qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/).

**RRF (Reciprocal Rank Fusion) — primary strategy:**

```json
{
  "rank_window_size": 100,
  "retrievers": [
    {
      "standard": {
        "query": { "multi_match": { "fields": ["body", "title"], "query": "%QUERY%" } }
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
```

Source: [`RetrievalStrategyDefaults.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyDefaults.cs)

**BM25-only strategy:** Uses `multi_match` on `["body", "title"]` without vector search.

**Subset scope filter (always applied):**

Every query is scoped to a specific subset via a `terms` filter on `subsetIds`:

```csharp
QueryDescriptor<ElasticDocument> subsetFilter = new QueryDescriptor<ElasticDocument>()
    .Term(t => t.Field("subsetIds").Value(subsetId));
```

Source: [`ElasticsearchClientWrapper.cs` in qna-service](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/ElasticsearchClientWrapper.cs)

**Document collapse:** Results are collapsed by `documentId` to deduplicate chunks from the same
document.

### 7.2 Metadata filtering

When the user provides metadata filters (e.g. `ControlNumber = "DOC-001"`), qna-service builds
ES bool `filter` clauses from `metadata.*` fields. See
[`ElasticMetadataFilterBuilder.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticMetadataFilterBuilder.cs).
These are keyword equality/range filters — not full-text search.

---

## 8. Document Shape Indexed

Each chunk written to the index is an instance of
[`ElasticDocument`](../../embedding-service/Source/Relativity.Embedding.Application/Models/Elasticsearch/ElasticDocument.cs).
The document ID in Elasticsearch is `{documentId}_{chunkId}` (e.g. `1055572_0`,
`1055572_1`). This means multiple ES documents exist per Relativity document — one per chunk.

The populated fields at write time (from
[`IndexingDocument.ToElasticDocuments()`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Models/IndexingDocument.cs)):

```csharp
yield return new ElasticDocument
{
    DocumentId      = DocumentId,
    Body            = chunkText,
    ChunkId         = _chunks[i].Index,                    // 0-based
    ChunkSize       = Encoding.UTF8.GetByteCount(chunkText) * 2,  // UTF-16 approx
    Embedding       = _embeddings?[i],                     // null if embedding failed
    ControlNumber   = ControlNumber ?? string.Empty,
    SubsetIds       = /* existing subsets + current */,
    DocumentModifyTime = ModifyTime,
    Metadata        = /* workspace metadata fields */,
    // NOTE: Title and CreatedAt are NOT set here — always default/empty
};
```

**Chunking:** HuggingFace BPE tokenizer (`intfloat/multilingual-e5-small`), max 500 tokens per
chunk, 100-token overlap.

---

## 9. Field Usage in Practice (from live `_field_usage_stats`)

From the aggregated field usage statistics on the inspected index:

| Field | Access Patterns Observed | Interpretation |
|-------|--------------------------|----------------|
| `body` | inverted_index (terms, postings, term_frequencies), norms | Actively used for BM25 text scoring |
| `subsetIds` | inverted_index (terms, postings) | Actively used for keyword filtering (subset scope) |
| `documentId` | points (numeric range) | Actively used for numeric equality/range queries and collapse |
| `chunkId` | doc_values, points | Used for sorting/filtering (e.g. fetching only chunk 0) |
| `controlNumber` | doc_values | Used for sorting/aggregation but not term matching |
| `title` | **not in usage stats** | Not queried in the tracked period — effectively unused |
| `embedding` | knn_vectors = 0 on observed shards | No kNN queries observed in the tracked window |
| `chunkSize`, `createdAt`, `documentModifyTime` | **not in usage stats** | Not queried |

The absence of `embedding` kNN usage likely reflects the time window of tracking rather than the
field not being used — RRF hybrid search is the primary retrieval strategy in production.

---

## 10. Known Gaps and Caveats

| Topic | Status |
|-------|--------|
| `bbq_hnsw` / `dims: 384` / `similarity: cosine` on `embedding` | **Not set by application code.** Applied by cluster-level index template in `elasticsearch-infra`. Template definition not found in available repos. |
| `number_of_shards`, `number_of_replicas`, tier routing | **Not set by application code.** ES cluster defaults or index template. |
| `title` field | Mapped as `text` but **never populated** during indexing. Included in BM25 queries but contributes nothing. |
| `createdAt` field | Mapped as `date` but **never assigned** in `ToElasticDocuments()`. Always zero/default value. |
| No custom analyzers | Both `body` and `title` use ES **standard analyzer** (lowercase tokenizer). No language-specific or custom analysis configuration. |
| Index template location | Likely in `elasticsearch-infra` cluster configuration (ECK Helm charts). Not found in code search — would need to inspect the live cluster's `GET /_index_template/*` or contact Search Infrastructure team. |

---

## 11. Key Source Code References

| Topic | File |
|-------|------|
| Index naming | [`embedding-service/.../Common/IndexNames.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Common/IndexNames.cs) |
| Base mapping (hardcoded) | [`embedding-service/.../Elasticsearch/ElasticsearchClientWrapper.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Elasticsearch/ElasticsearchClientWrapper.cs) (L105–157) |
| Metadata field mapping | Same file, L159–229 (`EnsureMetadataFieldMappingsAsync`) |
| Relativity → ES type mapping | Same file, L209–229 (`MapRelativityFieldTypeToEsProperty`) |
| Index lifecycle (create-if-missing) | [`embedding-service/.../Elasticsearch/IndexManagementService.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Elasticsearch/IndexManagementService.cs) |
| Document model | [`embedding-service/.../Models/Elasticsearch/ElasticDocument.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Models/Elasticsearch/ElasticDocument.cs) |
| Document construction (what gets populated) | [`embedding-service/.../Temporalio/Models/IndexingDocument.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Models/IndexingDocument.cs) (`ToElasticDocuments()`) |
| RRF query template | [`qna-service/.../ElasticSearch/RetrievalStrategyDefaults.cs`](../../repos/qna-service/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyDefaults.cs) |
| Subset scope filter | [`qna-service/.../ElasticSearch/ElasticsearchClientWrapper.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/ElasticsearchClientWrapper.cs) |
| Metadata filter building | [`qna-service/.../ElasticSearch/Filters/ElasticMetadataFilterBuilder.cs`](../../repos/qna-service/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticMetadataFilterBuilder.cs) |
| Architecture overview | [`air-assist-hub/architecture/embedding-service.md`](../../air-assist-hub/architecture/embedding-service.md) |
| ES cluster management | [`elasticsearch-infra/helm/eck-stack/`](../../elasticsearch-infra/helm/eck-stack/) |
| Test index mapping (minimal, no embedding) | [`qna-service/.../ElasticTestDataManager.cs`](../../repos/qna-service/Source/Relativity.QnA.API.NUnit.Integration/TestsInfrastructure/ElasticTestDataManager.cs) |
