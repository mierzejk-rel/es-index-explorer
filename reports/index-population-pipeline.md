# aiR Assist: How the Elasticsearch Index Gets Populated

**Purpose:** Detailed reference for engineers and AI agents on how RelativityOne workspace
documents are read, transformed into chunks, enriched with embeddings, and written to the
Elasticsearch index analyzed in `air-assist-elasticsearch-index.md`. Intended to provide enough
detail to replicate the pipeline in Python using `es-index-explorer`.

---

## 1. Pipeline Overview

The production indexing pipeline lives in `embedding-service` (.NET/Temporal). Documents flow
through these stages:

```
RelativityOne Workspace
    │
    ├── Object Manager API → document metadata (ArtifactId, ControlNumber, FileName, ...)
    ├── DataGrid API        → extracted text file path + file size on ADLS
    │
    ▼
ADLS (Azure Data Lake Storage)
    │  read extracted text by file path
    ▼
Chunker (HuggingFace BPE tokenizer, sliding window)
    │  500 tokens/chunk, 100-token overlap
    ▼
Embedding Model (intfloat/multilingual-e5-small via R1 Model Gateway)
    │  384-dimensional float vectors
    ▼
Elasticsearch Bulk Index
    one ES document per chunk: {documentId}_{chunkId}
```

Orchestration is via Temporal: `IndexSubsetWorkflow` (parent) produces document batches,
`ConsumerWorkflow` (child, one per batch) handles download → chunk → embed → index.

---

## 2. Reading RelativityOne Data

### 2.1 Object Manager API

The producer activities fetch document metadata from the RelativityOne Object Manager.
Two access patterns are used depending on the source type:

**Saved Search** (most common):
- `InitializeExportAsync` → `RetrieveNextResultsBlockAsync` (paginated export session)
- Fallback: `QuerySlim` with pagination (`start`, `length=1000`)

**Case Home** (explicit document IDs):
- `GetSetOfDocumentsFromWorkspace` with specific ArtifactIds

Source: [`DocumentsProvider.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Services/Indexing/DocumentsProvider.cs)

Fields requested from Object Manager:

| OM Field Name | Maps To | Notes |
|---|---|---|
| `Artifact ID` | `documentId` (int) | The Relativity document ArtifactId — this becomes the ES `documentId` field |
| `Relativity Text Identifier` | `controlNumber` (string) | The document Control Number; mapped in [`R1DocumentDtoMapper.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/ObjectManager/R1DocumentDtoMapper.cs) |
| `File Icon` | `fileName` | Used for logging/tracking |
| `Relativity Native Type` | `relativityNativeType` | Carried through pipeline but NOT written to ES |
| `Unified Title` | `unifiedTitle` | Carried through pipeline but NOT written to ES |
| `System Created By` | `systemCreatedBy` | Carried through pipeline but NOT written to ES |
| `System Created On` | `systemCreatedOn` | Carried through pipeline but NOT written to ES |
| Workspace metadata fields | `metadata.*` | Dynamically configured per workspace |

### 2.2 DataGrid API

After fetching OM metadata, each batch is joined with DataGrid to get:

| DataGrid Field | Purpose |
|---|---|
| `FilePath` | ADLS path to the extracted text file (prefixed with `file:`, stripped before use) |
| `FileSize` | Used for LPT bin-packing into consumer batches and for TooLarge filtering (>5MB) |
| `DocumentArtifactID` | Join key with OM results |

Source: [`DocumentsProvider.JoinWithDataGridAsync()`](../../embedding-service/Source/Relativity.Embedding.Application/Services/Indexing/DocumentsProvider.cs)
(lines 297–329)

Documents excluded at this stage (never reach chunking):
- Not found in Object Manager → status `NotInObjectManager`
- Not found in DataGrid → status `NotInDataGrid`
- No FilePath in DataGrid → status `MissingFilePath`

### 2.3 ADLS (Extracted Text Download)

The actual document text is read from Azure Data Lake Storage using the `FilePath` from DataGrid.
This happens in the consumer workflow, not the producer.

Source: [`StorageAccessWrapper.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Storage/StorageAccessWrapper.cs)

Two calls per document:
1. `GetFileMetadataAsync(textPath)` → file size (for TooLarge check) + last modified time (→ `documentModifyTime`)
2. `ReadAllTextAsync(textPath)` → full extracted text string (→ becomes `body` after chunking)

**The extracted text IS the source of the ES `body` field** — confirmed. The text is read whole,
then chunked into overlapping windows.

### 2.4 Python Equivalent: `r1-api-client` in `gpt-candidate-experiments`

For replicating this in Python, the
[`r1-api-client`](../../gpt-candidate-experiments/packages/r1_api_client/) package provides the
same Object Manager access patterns:

**Core fields** (always needed):

```python
# Fluent API — saved search export with Extracted Text + Control Number
documents = (
    relone_client.query_object_manager()
    .from_documents()
    .select("Extracted Text", "Control Number")
    .from_saved_search(saved_search_id)
    .export()
)
# Each document: {"artifact_id": int, "Extracted Text": str, "Control Number": str}
```

Or by field artifact ID (more portable):

```python
# Default field artifact IDs for these standard fields
EXTRACTED_TEXT_FIELD_ID = 1003668
CONTROL_NUMBER_FIELD_ID = 1003667

documents = om_client.export_saved_search_polars(
    field_ids=[CONTROL_NUMBER_FIELD_ID, EXTRACTED_TEXT_FIELD_ID],
    saved_search_id=saved_search_id,
    field_names=["control_num", "extracted_text"],
)
```

Long text streaming (for documents with truncated extracted text):

```python
full_text = om_client.stream_long_text_field(
    rdo_artifact_id=document_artifact_id,
    field_id=extracted_text_field_artifact_id
)
```

**Metadata fields** (optional, per workspace configuration):

To replicate the production metadata behavior in Python, read the additional R1 fields that
correspond to the metadata registry (see `air-assist-elasticsearch-index.md` §4.2.1 for the
full registry). The workspace admin provides the R1 `FieldArtifactId` for each metadata key.
Include them in the same export call:

```python
# Example: workspace has emailFrom mapped to R1 field 1234567,
# emailTo to 1234568, primaryDateTime to 1234569
METADATA_FIELD_MAP = {
    "emailFrom": 1234567,       # R1 FixedLengthText or LongText field
    "emailTo": 1234568,         # R1 FixedLengthText or LongText field
    "primaryDateTime": 1234569, # R1 Date field
}

all_field_ids = [CONTROL_NUMBER_FIELD_ID, EXTRACTED_TEXT_FIELD_ID] + list(METADATA_FIELD_MAP.values())
all_field_names = ["control_num", "extracted_text"] + list(METADATA_FIELD_MAP.keys())

documents = om_client.export_saved_search_polars(
    field_ids=all_field_ids,
    saved_search_id=saved_search_id,
    field_names=all_field_names,
)
```

Metadata values are document-level (shared across all chunks). For multi-valued fields (e.g.,
multiple email recipients), the R1 Object Manager returns them as separate list entries. Store
as a JSON array in ES. For documents where a metadata field has no value (e.g., non-email
documents), store `null` — this ensures the ES field key is present but non-email documents are
excluded from email-participant filter queries. See `air-assist-elasticsearch-index.md` §4.2.4
for the full value handling rules.

Key files in `gpt-candidate-experiments`:
- Client: [`r1_api_client/relone_client.py`](../../gpt-candidate-experiments/packages/r1_api_client/src/r1_api_client/relone_client.py)
- Fluent query: [`fluent/query.py`](../../gpt-candidate-experiments/packages/r1_api_client/src/r1_api_client/clients/object_manager/fluent/query.py)
- Legacy OM: [`legacy.py`](../../gpt-candidate-experiments/packages/r1_api_client/src/r1_api_client/legacy.py)
- Auth: [`auth.py`](../../gpt-candidate-experiments/packages/r1_api_client/src/r1_api_client/auth.py),
  [`auth_cidv2.py`](../../gpt-candidate-experiments/packages/r1_api_client/src/r1_api_client/auth_cidv2.py)
- Models: [`object_manager/models.py`](../../gpt-candidate-experiments/packages/r1_api_client/src/r1_api_client/clients/object_manager/models.py)
- Document protocol: [`r1_protocols/models.py`](../../gpt-candidate-experiments/packages/r1_protocols/src/r1_protocols/models.py)

---

## 3. Chunking

### 3.1 Algorithm

Sliding window over the token sequence of the extracted text:

1. Tokenize the full extracted text using a HuggingFace BPE tokenizer
   (`intfloat/multilingual-e5-small/tokenizer.json`)
2. Compute `stride = max_tokens - overlap` (default: `500 - 100 = 400`)
3. Slide a window of `max_tokens` tokens across the token array at `stride` increments
4. For each window, map token byte offsets back to character offsets and slice the original text
5. Assign `chunkId = chunks.Count` (0-based, sequential, in order of appearance)

Source: [`Chunker.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Embedding/Chunker.cs) (lines 83–124)

```csharp
int stride = maxTokensPerChunk - overlapTokens;
for (int i = 0; i < tokenSpan.Length; i += stride)
{
    int chunkSize = Math.Min(maxTokensPerChunk, tokenSpan.Length - i);
    // map byte offsets → char offsets, slice text
    chunks.Add(new TextChunk(
        DocumentId: documentId,
        Index: chunks.Count,    // 0, 1, 2, ...
        Text: slicedText,
        NumTokens: chunkSize));
}
```

### 3.2 Configuration Defaults

| Parameter | Default | Configured In |
|---|---|---|
| Tokenizer | HuggingFace BPE (`intfloat/multilingual-e5-small/tokenizer.json`) | [`Models.cs`](../../embedding-service/Source/Relativity.Embedding.SharedResources/Models/Models.cs) |
| Max tokens per chunk | 500 | [`appsettings.json`](../../embedding-service/Source/Relativity.Embedding.API/appsettings.json) `EmbeddingConfiguration.EnabledModels` |
| Overlap tokens | 100 | Same |
| Stride | 400 (computed: 500 - 100) | Computed at runtime |

### 3.3 Overlap Explained

With 500 max tokens and 100 overlap, chunks share the last 100 tokens with the next chunk:

```
Tokens:  [0 ─────────── 499] [400 ─────────── 899] [800 ─────────── 1199] ...
Chunk 0: [0..499]
Chunk 1:          [400..899]    ← 100 tokens overlap with chunk 0
Chunk 2:                    [800..1199]  ← 100 tokens overlap with chunk 1
```

This ensures that information at chunk boundaries is not lost during retrieval.

### 3.4 Python Equivalent

The `r1_chunking` package in `air_research` provides the same algorithm via
`SemanticKernelChunker`:

```python
from r1_chunking import build_chunker

chunker = build_chunker({
    "method": "semantic-kernel",
    "min_chunk_tokens": 350,
    "max_chunk_tokens": 500,
    "overlap": 100,
    "model": "intfloat/multilingual-e5-small",
})
chunks = chunker.chunk(document_text)
# Each chunk: Chunk(chunk_id=0, start=..., end=..., text="...")
```

Key files:
- [`r1_chunking/chunkers.py`](../../air_research/packages/r1_chunking/src/r1_chunking/chunkers.py)
- [`r1_chunking/chunk.py`](../../air_research/packages/r1_chunking/src/r1_chunking/chunk.py)

---

## 4. Embedding Generation

### 4.1 Model

| Property | Value |
|---|---|
| Model | `intfloat/multilingual-e5-small` |
| Variation | `passage` (prefix `"passage: "` prepended to each chunk text) |
| Dimensions | 384 |
| Similarity | `cosine` (configured at ES cluster level, not in this service) |

### 4.2 API

Embeddings are generated via the R1 Model Gateway (inference service):

```
POST model-inference-index/v1/embeddings
```

Source: [`RelativityModelInferenceClient.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Embedding/RelativityModelInferenceClient.cs)

### 4.3 Batching

| Parameter | Default | Notes |
|---|---|---|
| Inference batch size | 96 chunks per request | When R1 Model Gateway is enabled (LaunchDarkly) |
| Max concurrent requests | 32 | Parallel embedding API calls |
| Fallback batch size | 1 chunk per request | Legacy path without Model Gateway |

All chunks across all documents in a sub-batch (~100 estimated chunks) are flattened into a
single list, split into batches of 96, and embedded in parallel. If any chunk fails embedding,
the entire document is marked `EmbeddingFailed`.

### 4.4 Text-Search-Only Mode

A LaunchDarkly flag (`air-assist-should-use-text-search-strategy`) can skip embedding entirely.
In this case, chunks are indexed without the `embedding` field — only BM25 text search is
available, no vector/kNN retrieval.

---

## 5. ES Document Construction and Indexing

### 5.1 Document-to-Chunk Mapping

One RelativityOne document produces **multiple ES documents** — one per chunk. The ES document
ID is deterministic: `{documentId}_{chunkId}`.

Source: [`ElasticDocument.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Models/Elasticsearch/ElasticDocument.cs)

```csharp
public string GetElasticDocumentId() => FormatElasticDocumentId(DocumentId, ChunkId);
public static string FormatElasticDocumentId(int documentId, int chunkId) => $"{documentId}_{chunkId}";
```

### 5.2 Field Population

Source: [`IndexingDocument.ToElasticDocuments()`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Models/IndexingDocument.cs)
(lines 247–274)

```csharp
yield return new ElasticDocument
{
    DocumentId        = DocumentId,                              // Relativity ArtifactId
    Body              = chunkText,                               // chunk of extracted text
    ChunkId           = _chunks[i].Index,                        // 0-based sequential
    ChunkSize         = UTF8.GetByteCount(chunkText) * 2,        // approx UTF-16 byte size
    Embedding         = _embeddings?[i],                         // 384-dim float[] or null
    ControlNumber     = ControlNumber ?? string.Empty,           // OM "Relativity Text Identifier"
    SubsetIds         = existingSubsets.Concat([SubsetId]).Distinct().ToList(),
    DocumentModifyTime = ModifyTime,                             // from ADLS file metadata
    Metadata          = metadata,                                // workspace metadata fields
};
```

### 5.3 Version/Update Logic

Before indexing, the consumer checks the existing ES state for each document:

| ES State | Action | Category |
|---|---|---|
| No chunks in ES for this documentId | Full index (chunk + embed + write) | `NewDocument` |
| Chunks exist, ADLS modify time is newer | Delete old chunks, re-chunk, re-embed, re-index | `NewVersion` |
| Chunks exist, same version, subsetId not in subsetIds | Append subsetId via `UpdateByQuery` (no re-chunking) | `AlreadyIndexedAppendToSubset` |
| Chunks exist, same version, subsetId already present | Skip entirely | `NothingChangedKeepInSubset` |

When a document is re-indexed as `NewVersion` and the new version has fewer chunks, orphaned
chunks (higher chunkIds) are explicitly deleted from ES.

Source: [`ConsumerWorkflow.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Workflows/ConsumerWorkflow.cs)
and [`IndexingDocument.SetElasticMetadata()`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Models/IndexingDocument.cs)
(lines 350–382)

---

## 6. Identifier Reference

| Identifier | Type | Origin | Algorithm | Deterministic? | Reversible? |
|---|---|---|---|---|---|
| `documentId` | `int` | Relativity `ArtifactId` from Object Manager | Direct passthrough — no transformation | Yes (same ArtifactId always yields same documentId) | Yes — `documentId` IS the Relativity ArtifactId |
| `chunkId` | `int` | Assigned during chunking | 0-based sequential index in chunk creation order (left-to-right sliding window) | Yes, for same text + same chunker config | Maps to a text window position: chunk N covers tokens `[N*stride .. N*stride + max_tokens]` |
| `controlNumber` | `string` | OM field `"Relativity Text Identifier"` | Direct passthrough | Yes | Yes — it IS the Relativity Control Number |
| `subsetId` | `string` (GUID) | Created by embedding-service when a "document subset" (saved search index) is created | GUID generated at subset creation; stored in Postgres | Stable for the lifetime of the subset | Maps to a Subset entity in embedding-service DB |
| ES `_id` | `string` | Computed from documentId + chunkId | `"{documentId}_{chunkId}"` | Yes | Yes — split on `_` to get both parts |

---

## 7. Summary Table: ES Field Population

| ES Field | ES Type | Source System | Source Field / API | Transformation | Set By |
|---|---|---|---|---|---|
| `documentId` | integer | RelativityOne | Object Manager → `ArtifactId` | None (direct passthrough) | `ToElasticDocuments()` |
| `body` | text | RelativityOne (ADLS) | DataGrid → `FilePath` → ADLS `ReadAllTextAsync` | Extracted text chunked by HF BPE tokenizer (500 tokens, 100 overlap) | `ToElasticDocuments()` |
| `chunkId` | integer | Computed | Chunker output | 0-based sequential index in chunk order | `Chunker.ChunkTextBatch()` → `TextChunk.Index` |
| `chunkSize` | integer | Computed | Chunk text byte length | `UTF8.GetByteCount(chunkText) * 2` (UTF-16 approximation) | `ToElasticDocuments()` |
| `controlNumber` | keyword | RelativityOne | Object Manager → `"Relativity Text Identifier"` | None (direct passthrough) | `ToElasticDocuments()` |
| `embedding` | dense_vector | R1 Model Gateway | `POST model-inference-index/v1/embeddings` | `intfloat/multilingual-e5-small` (passage variation, 384 dims) | `ToElasticDocuments()` |
| `subsetIds` | keyword[] | embedding-service | Subset GUID from job context | Existing ES subsetIds + current subsetId (deduplicated) | `ToElasticDocuments()` or `AppendSubsetIdToDocumentsRecords` |
| `documentModifyTime` | date | ADLS | File metadata → last modified timestamp | None (direct passthrough) | `ToElasticDocuments()` |
| `title` | text | — | — | **Never populated.** Mapped in ES but not set in `ToElasticDocuments()`. | — |
| `createdAt` | date | — | — | **Never populated.** Mapped in ES but not assigned. Defaults to `DateTime` zero. | — |
| `metadata.*` | varies | RelativityOne | Object Manager → workspace metadata fields | Typed conversion per Relativity field type (see index report §4.2) | `BuildElasticMetadata()` in `ToElasticDocuments()` |
| ES `_id` | string | Computed | `documentId` + `chunkId` | `"{documentId}_{chunkId}"` | `ElasticDocument.GetElasticDocumentId()` |

---

## 8. Key Exclusion Filters

Documents may be excluded at various stages before reaching ES:

| Stage | Condition | Status / Outcome |
|---|---|---|
| Object Manager query | Document not returned by OM | `NotInObjectManager` |
| DataGrid join | Document has no DataGrid entry | `NotInDataGrid` |
| DataGrid join | Document has no `FilePath` | `MissingFilePath` |
| ADLS metadata check | File size > 5 MB (5,242,880 bytes) | `TooLarge` |
| ADLS metadata check | File size = 0 bytes | `Empty` |
| ES metadata check | Same version, subsetId already assigned | `NothingChangedKeepInSubset` (skipped) |
| Chunking | Chunker produces 0 chunks | `ChunkingFailed` |
| Embedding | Model Gateway returns error | `EmbeddingFailed` |

---

## 9. Configuration Summary for Python Replication

To replicate this pipeline in `es-index-explorer`, these are the key parameters:

```python
# Chunking
TOKENIZER_MODEL = "intfloat/multilingual-e5-small"
MAX_TOKENS_PER_CHUNK = 500
OVERLAP_TOKENS = 100
STRIDE = MAX_TOKENS_PER_CHUNK - OVERLAP_TOKENS  # 400

# Embedding
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_PREFIX = "passage: "  # prepended to chunk text before embedding

# ES document ID
def es_document_id(document_id: int, chunk_id: int) -> str:
    return f"{document_id}_{chunk_id}"

# Chunk size (as stored in ES)
def chunk_size_bytes(chunk_text: str) -> int:
    return len(chunk_text.encode("utf-8")) * 2  # UTF-16 approximation

# Subset ID assignment
# subsetIds is a list; new subset GUID is appended (deduplicated) to existing list
```

### Existing Python libraries available in workspace

| Capability | Package | Location |
|---|---|---|
| Read RelativityOne documents | `r1-api-client` | [`gpt-candidate-experiments/packages/r1_api_client/`](../../gpt-candidate-experiments/packages/r1_api_client/) |
| Chunking (matching production) | `r1-chunking` | [`air_research/packages/r1_chunking/`](../../air_research/packages/r1_chunking/) |
| Embedding (OpenAI via gateway) | `r1-rate-limiter` | [`air_research/packages/r1_rate_limiter/`](../../air_research/packages/r1_rate_limiter/) |
| Elasticsearch client | `elasticsearch` | Already in `es-index-explorer` |
| CID authentication | Custom (see `es_index_explorer/auth.py`) | `es-index-explorer` |

---

## 10. Source Code Reference

| Topic | File |
|---|---|
| Indexing entry point | [`embedding-service/.../Controllers/v1/IndexingController.cs`](../../embedding-service/Source/Relativity.Embedding.API/Controllers/v1/IndexingController.cs) |
| Parent workflow | [`IndexSubsetWorkflow.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Workflows/IndexSubsetWorkflow.cs) |
| Child workflow | [`ConsumerWorkflow.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Workflows/ConsumerWorkflow.cs) |
| Producer activities | [`ProducerActivities.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Activities/ProducerActivities.cs) |
| Consumer activities | [`ConsumerActivities.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Activities/ConsumerActivities.cs) |
| Document fetch from R1 | [`DocumentsProvider.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Services/Indexing/DocumentsProvider.cs) |
| OM field mapping | [`R1DocumentDtoMapper.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/ObjectManager/R1DocumentDtoMapper.cs) |
| OM field constants | [`R1TenantQueryConstants.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Common/R1TenantQueryConstants.cs) |
| Chunker | [`Chunker.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Embedding/Chunker.cs) |
| Embedding client | [`RelativityModelInferenceClient.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Embedding/RelativityModelInferenceClient.cs) |
| Document → ES mapping | [`IndexingDocument.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Temporalio/Models/IndexingDocument.cs) |
| ES document model | [`ElasticDocument.cs`](../../embedding-service/Source/Relativity.Embedding.Application/Models/Elasticsearch/ElasticDocument.cs) |
| ES bulk indexing | [`ElasticsearchClientWrapper.cs`](../../embedding-service/Source/Relativity.Embedding.Infrastructure/Elasticsearch/ElasticsearchClientWrapper.cs) |
| Embedding config | [`appsettings.json`](../../embedding-service/Source/Relativity.Embedding.API/appsettings.json) |
| Python R1 client | [`r1_api_client/relone_client.py`](../../gpt-candidate-experiments/packages/r1_api_client/src/r1_api_client/relone_client.py) |
| Python chunking | [`r1_chunking/chunkers.py`](../../air_research/packages/r1_chunking/src/r1_chunking/chunkers.py) |
| Python embedding | [`r1_rate_limiter/sync.py`](../../air_research/packages/r1_rate_limiter/src/r1_rate_limiter/sync.py) |
