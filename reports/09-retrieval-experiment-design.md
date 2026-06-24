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
{ "query": { "multi_match": { "query": "<q>", "fields": ["summary", "topic", "title^0.5"] } } }
```
`summary` and `topic` use the default field weight of `1.0`. `title^0.5` is included only when `title_enabled` is true (EMC2); omitted for Mallinckrodt.

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
| **Tier 2** | Nested document format: full document groups with metadata and concatenated chunks | Nested doc groups | `low` or `medium` |

### 5.2 Variable dimensions

| Dimension | Tier 0 | Tier 1 | Tier 2 |
|---|---|---|---|
| **Retrieval signals** | BM25-chunks + kNN-chunks | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent | BM25-chunks + kNN-chunks + BM25-parent + sparse-parent (always ON) |
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
value of switching to medium reasoning and multi-hop.

**Metadata retrieval always ON in Tier 2.** Once the nested document structure is adopted,
excluding parent-field signals would waste the indexed metadata. All Tier 2 experiments use
the full signal set. The value of metadata retrieval signals is assessed at Tier 1 (E0 vs E1).

**Metadata for generation tested at both chunk counts.** E2a-low (25 chunks, low reasoning)
has metadata gen OFF, and E2d-nometa (60 chunks, medium reasoning) also has metadata gen OFF.
Together they reveal whether the value of exposing document metadata to the LLM depends on
how much raw chunk context the model already has.

**Title toggle.** The EMC2 workspace has meaningful document titles (email subjects, file
names); the Mallinckrodt corpus has less reliable titles. Title is therefore included as a
retrieval signal for EMC2 but excluded for Mallinckrodt. This is implemented as a per-eval-set
configuration flag passed to the tool at runtime, not as a separate experiment dimension.
When included, `title` is assigned a field weight of `0.5` in the BM25 `multi_match` query —
half the default weight of `summary` and `topic` — reflecting its lower reliability as a
standalone retrieval signal relative to AI-generated metadata.

**Client-side vs ES-side fusion.** When fusion is performed on the client, each signal
(BM25-chunks, kNN-chunks, BM25-parent, sparse-parent) is issued as a separate ES query and the
returned chunk-level results are fused via RRF in Python. When fusion is ES-side, a single ES
request with a nested RRF retriever is issued; ES fuses at the **document level** (each
document is scored by its best matching chunk via `score_mode: max`) and the client reads
chunks from `inner_hits`. ES-side fusion cannot rank chunks across documents globally — that
distinction is explored by E2a-med vs E2c.

**5 MB size filter in Tier 2.** The 5 MB size filter (`workspace_extracted_text_size <= 5,242,880`)
is applied in Tier 0 and Tier 1 to match current production behaviour. It is disabled for all
Tier 2 experiments so that larger documents — which may contain the most relevant context for
complex multi-hop questions — are not excluded from retrieval.

**MMR as an alternative selection method.** A parallel branch (E0-mmr, E1-mmr, E2a-med-mmr)
replaces the client-side RRF fusion step with Maximum Marginal Relevance (MMR) selection. This
is an orthogonal dimension that spans tiers: instead of fusing the per-signal ranked lists by
reciprocal rank, MMR embeds the candidate chunks and the query with the dense e5 model and
greedily selects chunks that are relevant to the query while penalizing redundancy with
already-selected chunks (`λ·sim(q,d) − (1−λ)·max_s sim(d,s)`, λ = 0.5, mirroring the
qna-service production MMR strategy). The branch isolates the selection method while holding
signals, output format, reasoning, and hop strategy identical to each experiment's RRF
counterpart.

---

## 6. Experiment Matrix

### 6.1 Experiment definitions

**Tier 0 — Baseline reproduction**

**E0** — Production parity on nested index.

Run two nested ES queries independently (BM25 on `chunks.text`, kNN on `chunks.embedding`),
flatten the returned `inner_hits` chunks, apply client-side RRF, take top 25 by score. Filter
`byte_size <= 5 MB`. No parent metadata fields involved in retrieval or generation. Uses
the production config (`rag_agent_v3/013.toml`, version 3.13) with no prompt changes. The
experiment eval runner restricts retrieval to BM25-chunks + kNN-chunks only; the tool pair
presented to the LLM is unchanged. Output is a flat chunk list in the same XML format as
current production.

*Purpose:* confirm the new direct-ES Python tool reproduces current qna-service behavior.

---

**Tier 1 — Metadata-enriched retrieval (flat output)**

**E1** — Add parent-field retrieval signals, keep flat output.

Same as E0 plus: BM25 `multi_match` on `title`/`summary`/`topic` and `sparse_vector` queries
on `title_sparse`/`summary_sparse`/`topic_sparse`. All four signal results (BM25-chunks,
kNN-chunks, BM25-parent, sparse-parent) are issued as separate ES queries and fused
client-side via RRF to produce the top 25 chunks. Filter `byte_size <= 5 MB`. Title included
for EMC2, excluded for Mallinckrodt. Output remains a flat chunk list; the LLM does not see
document metadata directly.

*Purpose:* does adding BM25 + sparse signals on parent metadata fields improve chunk
selection quality?

---

**Tier 2 — Nested document format**

All Tier 2 experiments return document groups: each group contains a list of retrieved
chunks with adjacent chunks concatenated where `chunk_index` values are consecutive
(each concatenated run identified by a range-style chunk ID, e.g. `"1:3"`). In experiments
marked "metadata gen ON", the LLM-facing `<document_group>` XML also carries
`title`, `summary`, and `topic` (`title` only for indices where `title_enabled` is true,
e.g. EMC2). In experiments marked "metadata gen OFF" (`E2a-low`, `E2d-nometa`), those
fields are omitted from the XML. The 5 MB size filter is disabled for all Tier 2 experiments. A new
TOML config is created for Tier 2 (see §8).

**E2a-low** — Minimum viable Tier 2 configuration.

Retrieval: all four signals, client-side RRF, top 25 chunks. Generation context: document
groups with metadata fields **omitted** (metadata gen OFF). `reasoning_effort: low`.
Single-hop (exactly one retrieval iteration). Title toggle: EMC2 ON / Mallinckrodt OFF.

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

A single ES request with a nested RRF retriever fuses all signals. ES returns documents ranked
by their best chunk score; the client reads `inner_hits` and groups. No cross-document chunk
ranking by the client.

*Purpose:* isolates fusion location (client-side global chunk ranking vs ES-side document
ranking).

**E2d** — Same as E2a-med but 60 chunks.

Retrieval returns 60 chunks instead of 25. Inner-hits sizes scaled accordingly.

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

These three experiments form a parallel branch that replaces the client-side RRF fusion step
with MMR selection (λ = 0.5). Each signal fetches a larger candidate pool (up to 100) so MMR
has room to diversify; MMR then selects the final top-k. Everything else is held identical to
the named RRF counterpart, making each pair a clean single-dimension (RRF vs MMR) swap.

**E0-mmr** — E0 with MMR instead of RRF (baseline of the MMR branch).

Two chunk-level signals (BM25-chunks, kNN-chunks) build the candidate pool; MMR selects the
final 25 chunks. Flat output, low reasoning, multi-hop. Identical to E0 otherwise.

*Purpose:* MMR-branch baseline; isolates RRF vs MMR with chunk-only signals.

**E1-mmr** — E1 with MMR instead of RRF.

All four signals (BM25-chunks, kNN-chunks, BM25-parent, sparse-parent) build the candidate
pool; MMR selects the final 25 chunks. Flat output, low reasoning, multi-hop. Identical to E1
otherwise. Title included for EMC2, excluded for Mallinckrodt.

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
| Flat — 2 signals (BM25-chunks + kNN-chunks) | `handle_search_documents` (signals restricted) | E0 |
| Flat — 4 signals (all) | `handle_search_documents` (default signals) | E1 |
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

#### Flat output (Tier 0, Tier 1)

1. Issue separate ES queries for each active signal. The signal set is controlled by the
   `signals: list[SignalType]` parameter on the retriever method — defaults to all four
   signals; E0 passes only `[BM25_CHUNKS, KNN_CHUNKS]`.
2. Collect all chunks from `inner_hits` across all queries.
3. Apply client-side RRF: for each chunk `(document_artifact_id, chunk_index)`, sum
   reciprocal ranks across signals. Take top-k by fused score.
4. Serialize as flat `<passage>` XML elements matching the current qna-service format.

#### Nested output (Tier 2)

1. Same multi-signal queries as Tier 1 (client-side RRF) or a single ES RRF retriever
   (ES-side RRF, E2c).
2. Group result chunks by `document_artifact_id`.
3. For each group, determine which parent fields are included in the LLM-facing XML:
   - **LLM-facing XML** (`<document_group>` elements): carries only `title` / `summary` /
     `topic` when metadata for generation is ON (`title` further gated by `title_enabled`:
     ON for EMC2, OFF for Mallinckrodt). When metadata for generation is OFF (E2a-low,
     E2d-nometa), these fields are omitted entirely.
   - **`control_number`** is carried in the structured `GroupedChunks` output returned
     alongside the XML. This is the channel used by eval scorers and citation
     post-processing (`citation_validation.py`). It is not emitted in the LLM-facing XML.
   - **`primary_date_time`** is used solely for ES-side filtering (a `range` query clause
     in `filters.py`). It is neither extracted into `_source` nor serialized into the XML.
4. **Adjacent chunk concatenation.** `chunk_index` values are 0-based. Within each
   document group, detect runs of consecutive `chunk_index` values. For a run of chunks
   with indices `[i, i+1, ..., j]`:
   - Start with the full text of chunk `i`.
   - For each subsequent chunk `i+1, ..., j`, strip the first `leading_overlap_chars`
     characters from its text before appending (this removes the duplicated overlap
     without losing any unique content).
   - The resulting concatenated passage is assigned the range-style chunk ID `"i:j"`
     (inclusive), e.g. `"1:3"` means chunks 1, 2, and 3 were merged. A single
     non-concatenated chunk retains its plain integer ID (e.g. `"5"`).
   - In the XML, the concatenated chunk appears as a single `<chunk>` element with
     `<chunk_id>1:3</chunk_id>` and the de-duplicated text as `<content>`. The LLM
     cites it as `[doc_id-1:3]`.
5. Serialize as `<document_group>` XML elements containing per-document metadata and
   chunk list.

#### MMR selection (MMR branch)

When `AIR_ASSIST_FUSION=mmr`, the client-side RRF step is replaced by Maximum Marginal
Relevance selection:

1. Issue the same per-signal ES queries as the RRF counterpart and collect the deduplicated
   union of candidate chunks `(document_artifact_id, chunk_index)`. Each signal fetches a
   larger pool (up to 100) so MMR has room to diversify.
2. Embed the query and each candidate chunk's text with the dense e5 model
   (`intfloat/multilingual-e5-small`, `"passage: "` prefix for chunks), reusing the cached
   encoder; these reproduce the index-time chunk vectors.
3. Run MMR with λ = 0.5: first select the chunk most similar to the query, then iteratively
   select the chunk maximizing `λ·sim(q,d) − (1−λ)·max_s sim(d,s)` until top-k are chosen.
4. Feed the selected chunks into the same flat (Tier 0/1) or nested (Tier 2) serialization as
   the RRF path.

This mirrors qna-service's `MaximumMarginalRelevanceSelector` and `Bm25SearchWithMmrSettings`
(λ = 0.5, candidate fetch 100, top-k 25). MMR is additive to the experiment toolkit — a new
`mmr_select` function alongside `rrf_fuse` — so no existing `air_assist_experiments` code is
invalidated; the RRF path is unchanged.

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

All experiment branches fork from the shared root branch (`DSAS-2836/experiments`), which
provides retriever methods with parameterised signals, chunk count, size filter, and fusion
method (RRF/MMR). The table below shows what must change in each experiment branch relative to
that shared root — the root itself remains identical for every experiment.

| Dimension | Control mechanism | What changes per experiment branch |
|---|---|---|
| Retrieval signals (2 vs 4) | `signals: list[SignalType]` param on retriever (default: all 4) | `tool.py` handler sets signal list per experiment |
| Metadata for generation (ON/OFF) | `include_metadata: bool` param on `retrieve_nested` / `handle_search_documents_nested` (default `False`) — implemented on root branch | `tool.py` handler passes `include_metadata=True` for metadata-ON experiments (E2a-med, E2d, E2e, E2a-med-mmr) |
| Fusion location (ES-side RRF) | Separate `nested_docs_es_rrf` retriever module | `air_assist_experiments` new retriever (E2c branch only) |
| Final selection method (RRF/MMR) | `AIR_ASSIST_FUSION` env var (default `rrf`) | Env var only — no code change |
| Chunk count (25/60) | `result_count: int` method param (default 25) | `tool.py` handler passes override per experiment |
| 5 MB size filter (ON/OFF) | `size_limit_bytes: int \| None` method param | `tool.py` handler passes `None` for all Tier 2 experiments |
| Reasoning effort (low/medium) | TOML `reasoning_effort` field | New TOML file only — no code change |
| Hop strategy (multi/single) | System prompt wording + `get_tool_choice()` in `rag_agent.py` | `air_assist_core/src` rag_agent.py (experiment branch) |
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
`rglob("*.toml")`. Version numbers 92–93 have no collision with any existing config (current
range is 8–24 in `rag_agent_v3/`).

**E2a-low** uses `DSAS-2836/092.toml` (version 3.92) with:
- `reasoning_effort: low`
- System prompt updated for nested document format (metadata gen OFF variant — metadata
  fields omitted from XML, so the format explanation does not mention them)
- Single-hop: tool_choice capped at `none` after iteration 0

**E2a-med, E2c, E2d, E2d-nometa, E2e** use `DSAS-2836/093.toml` (version 3.93) with:
- `reasoning_effort: medium`
- System prompt updated for nested document format with metadata visible: "Each result
  contains document-level metadata (title, summary, topic) followed by the most relevant
  passages. Use the metadata to orient your understanding before citing passages."
- E2e additionally: prompt instructs exactly one retrieval iteration; tool_choice capped at
  `none` after iteration 0

The TOML schema is not extended. Retrieval parameters (signals, chunk count, fusion,
metadata-gen flag) are controlled by the `ExperimentToolProvider` and companion Python
configuration, not via the TOML.

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
  (resolves to `rag_agent_v3/013.toml`). For Tier 2: `3.92` or `3.93`.

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

| Experiment | Config (version) | Retrieval mode | Fusion | Chunks | Metadata gen | Reasoning | Hop policy | 5 MB filter |
|---|---|---|---|---|---|---|---|---|
| E0 | `rag_agent_v3/013.toml` (3.13) | Flat — 2 signals | Client-side RRF | 25 | OFF | low | Multi | ON |
| E1 | `rag_agent_v3/013.toml` (3.13) | Flat — 4 signals | Client-side RRF | 25 | OFF | low | Multi | ON |
| E2a-low | `DSAS-2836/092.toml` (3.92) | Nested — client-side RRF | Client-side RRF | 25 | OFF | low | Single (capped) | OFF |
| E2a-med | `DSAS-2836/093.toml` (3.93) | Nested — client-side RRF | Client-side RRF | 25 | ON | medium | Multi | OFF |
| E2c | `DSAS-2836/093.toml` (3.93) | Nested — ES-side RRF | ES-side RRF | 25 | ON | medium | Multi | OFF |
| E2d | `DSAS-2836/093.toml` (3.93) | Nested — client-side RRF | Client-side RRF | 60 | ON | medium | Multi | OFF |
| E2d-nometa | `DSAS-2836/093.toml` (3.93) | Nested — client-side RRF | Client-side RRF | 60 | OFF | medium | Multi | OFF |
| E2e | `DSAS-2836/093.toml` (3.93) | Nested — client-side RRF | Client-side RRF | 60 | ON | medium | Single (capped) | OFF |
| E0-mmr | `rag_agent_v3/013.toml` (3.13) | Flat — 2 signals | Client-side MMR (`AIR_ASSIST_FUSION=mmr`) | 25 | OFF | low | Multi | ON |
| E1-mmr | `rag_agent_v3/013.toml` (3.13) | Flat — 4 signals | Client-side MMR (`AIR_ASSIST_FUSION=mmr`) | 25 | OFF | low | Multi | ON |
| E2a-med-mmr | `DSAS-2836/093.toml` (3.93) | Nested — client-side RRF | Client-side MMR (`AIR_ASSIST_FUSION=mmr`) | 25 | ON | medium | Multi | OFF |

### 8.3 Prompt changes summary

| Config | Experiment(s) | Changes from 013.toml |
|---|---|---|
| `rag_agent_v3/013.toml` (3.13) | E0, E1, E0-mmr, E1-mmr | **No changes.** System prompt, tool descriptions, and all guidance are identical to production. Retrieval backend is swapped at eval runner level only. |
| `DSAS-2836/092.toml` (3.92) | E2a-low | Nested format explanation (no metadata fields mentioned, as they are omitted from XML); single-hop instruction: "Perform exactly one retrieval call per turn. Issue all necessary queries simultaneously." |
| `DSAS-2836/093.toml` (3.93) | E2a-med, E2c, E2d, E2d-nometa | Nested format explanation with metadata visible: "Each result contains document-level metadata (title, summary, topic) followed by the most relevant passages. Use the metadata to orient your understanding before citing passages."; `reasoning_effort = "medium"` |
| `DSAS-2836/093.toml` (3.93) | E2e | Same as above + single-hop instruction |
| `DSAS-2836/093.toml` (3.93) | E2a-med-mmr | Identical to E2a-med (MMR set via `AIR_ASSIST_FUSION=mmr`; no prompt change) |

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
