# 10 — Simple Mode Experiment Design

**Purpose:** Define a latency-first aiR Assist Simple Mode, the `S` experiment suite used to
measure it, and the implementation roadmap needed to execute those experiments. Simple Mode is
intended for questions plausibly answerable from one or a few passages—for example, acronym
expansion, person identification, or a focused evidence question. It is not assumed to be
appropriate for every question; the suite measures where it succeeds and where full mode remains
necessary.

**Scope:** This is an experiment design, not a production routing design. It reuses the existing
EMC2 and Mallinckrodt nested indices, direct-ES experiment tooling, v3 agent, and r1-evals
workflow. It adds no index mapping or ingestion requirement.

**Prior art:**

- `08-index-design-and-ingestion.md` — authoritative nested-index structure and Elasticsearch
  query capabilities.
- `09-retrieval-experiment-design.md` — E0/Tier 2 baselines, evaluation protocol, and dataset
  conventions.
- Elastic Elasticsearch 9.4 retriever/RRF documentation — server-side RRF and nested `inner_hits`.
- GPT-5.1 model/prompting documentation — supported `none` and `low` reasoning efforts.
- MLflow tracing documentation — trace/span timing semantics and manual instrumentation.

---

## 1. Hypothesis and evaluation philosophy

Simple Mode is a deliberately constrained agent intended to trade broad research coverage for
lower retrieval and generation latency. The hypothesis is:

> For a discoverable subset of aiR Assist questions, one retrieval round, a small passage
> context, no generated document metadata, and minimal/no reasoning retain acceptable answer,
> retrieval, and citation quality while materially reducing latency.

The suite does **not** pre-classify questions as simple. It runs all question variations already
used for the three established datasets, then identifies the empirical Simple-compatible cohort
from quality and timing outcomes:

| Dataset | Rubrics | Dataset selector | Rubric scope |
|---|---:|---|---|
| EMC2 UAT set_1 | 21 | `emc2` | `**/uat/set_1/*.rubric.toml` |
| EMC2 UAT set_2 | 20 | `emc2` | `**/uat/set_2/*.rubric.toml` |
| Mallinckrodt GA | 22 | `mallinckrodt` | all GA rubrics |

All existing input variations and repetitions used by prior experiments remain eligible. Results
must be analysed overall and per rubric/use case/question variation: an aggregate score cannot
establish whether individual questions are suitable for a future Simple/Full routing decision.

No experiment report is generated during this suite. The implementation must store all required
raw measurements in MLflow; comparative reporting occurs after the experiments finish.

---

## 2. Baseline and held constants

### 2.1 E0 reference baseline

E0 is the historical quality reference. It uses the direct-ES nested index but reproduces the
production-style two chunk signals: nested BM25 and nested kNN, client-side ordinal RRF, 25 final
chunks, low reasoning, multi-hop, and flat output. It also has important non-parity caveats:
semantic rather than fixed-window chunking, a nested rather than flat index, no qna-service/MCP
path, and different first-chunk handling.

Simple Mode is not an E0 clone. It intentionally changes the hop, context, fusion, output, and
reasoning design to measure a low-latency alternative.

### 2.2 Held constants for all initial S experiments

| Property | Initial Simple Mode value |
|---|---|
| Agent | v3 `MultihopFreeTextRag` |
| LLM | `gpt-5.1-2025-11-13` |
| Retrieval iteration count | Exactly one |
| Hop enforcement | `max_tool_iterations = 1`; `get_tool_choice()` forces `none` after the first tool round |
| LLM tool schema | Retrieval tools only; no `WriteFile` or `ReadFile` |
| Side artifacts | None: no `notes.txt`, `signature.txt`, or virtual file access |
| Retrieval fields | Chunk text / chunk dense embedding only; no title, summary, topic, sparse parent vectors, or parent BM25 |
| Generation metadata | OFF: no title, summary, or topic in LLM context |
| Hard filters | Mandatory `subset_ids`; explicit date/email filters when allowed by call policy |
| Empty filtered result | No fallback generic query |
| Size filter | OFF: search the full existing index; no document is excluded by `workspace_extracted_text_size` |
| First chunk | No query/fetch/decorate algorithm |
| Invocation concurrency | 1 |
| Output format | Flat-concatenated grouped chunks |
| Citation format | `[doc_id-N]` or `[doc_id-i:j]` for a concatenated run |
| Output token budget | 12,000 `max_completion_tokens` |

The one-hop rule is both prompt guidance and a code rule. At iteration 0 the graph requires a tool
call. After that tool round, `max_tool_iterations = 1` makes the next LLM call use
`tool_choice="none"`, so no second retrieval round can occur.

E0 uses 30,000 `max_completion_tokens` because it permits multi-hop, low-reasoning planning and
its structured output can follow a broader retrieval context. Simple Mode instead has one retrieval
round, default `reasoning_effort = "none"`, a small global-context chunk budget, and a 100–600
word answer target. A 12,000-token cap is therefore 60% below E0 while preserving headroom for
the structured response and exact source snippets. An 8,000-token cap would likely work but adds
unnecessary length-finish risk when a 20-chunk answer cites several passages.

### 2.3 Flat-concatenated output

Simple Mode uses **flat** output like E0: one `<grouped_chunks>` element per document, not a Tier
2 `<document_group>` with parent metadata. It does not use first-chunk decoration.

It retains the Tier 2 adjacent-chunk algorithm because it is cheap and avoids duplicating stored
overlap. After global selection, contiguous selected chunk indices `[i, …, j]` from one document
are joined by removing each later chunk’s `leading_overlap_chars`. A one-chunk run retains ID
`"i"`; a multi-chunk run receives ID `"i:j"`. The range is a single flat `<chunk>` and is cited
as `[doc_id-i:j]`.

This requires the existing range-ID citation validation/scoring support. It does not require
metadata generation or nested XML.

---

## 3. Retrieval architecture

### 3.1 Retrieval modes

Initial Simple Mode supports only chunk-level retrieval:

| Mode | Elasticsearch request per retrieval tool call | Query style |
|---|---|---|
| `bm25` | Nested `match` on `chunks.text` | Terse keywords, entities, aliases, exact terminology |
| `dense` | Nested kNN on `chunks.embedding` | Natural-language information need |
| `hybrid_es_rrf` | One server-side RRF request combining nested BM25 and nested kNN | Blended, keyword-oriented, semantic, or mixed phrasing by call budget |

Dense/hybrid calls compute only the E5 **query** embedding client-side, using the existing query
prefix; BM25-only calls do not. This local query-embedding time is part of measured retrieval
latency. Simple Mode never calculates client-side candidate/passage embeddings: it performs no
MMR, client-side reranking, or passage-vector caching. Sparse query encoding, parent BM25, sparse
parent signals, client RRF, and client MMR are also absent from the initial suite.

### 3.2 Elasticsearch 9.4 server-side RRF feasibility

Server-side RRF is technically supported by the existing mapping:

- `chunks` is a nested field.
- `chunks.text` is searchable text.
- `chunks.embedding` is an indexed 384-dimension cosine dense vector.
- Elasticsearch RRF accepts two or more child retrievers and propagates uniquely named nested
  `inner_hits` from sub-retrievers to final RRF parent hits.

The hybrid request must use two `standard` retrievers, each wrapping a nested query:

1. nested BM25 `match` on `chunks.text` with `inner_hits.name = "bm25_chunks"`;
2. nested kNN on `chunks.embedding` with `inner_hits.name = "knn_chunks"`.

The direct kNN retriever does not itself accept the required `inner_hits`; the nested-standard
wrapper is therefore mandatory. The RRF request uses `rank_window_size >= size`, a fixed
`rank_constant`, and a request `size` equal to `simple_per_call_fetch_count`.

### 3.3 Parent-ranking limitation and extraction

With the nested index, RRF ranks parent documents rather than a global chunk list. Inner hits are
computed only for the final RRF parent hits. Therefore each hybrid call must extract chunks
deterministically without client reranking:

1. traverse RRF parent hits in server rank order;
2. consume the uniquely named child inner-hit lists in fixed child order (BM25 then kNN), each in
   returned rank order;
3. deduplicate `(document_artifact_id, chunk_index)`;
4. stop at `simple_per_call_fetch_count` chunks.

This is an extraction policy, not a fusion/reranking stage.

### 3.4 ES RRF integration smoke test

Before running any S experiment, run a credentials-gated live integration test against the deployed
Elasticsearch 9.4 cluster. The test:

1. builds the two-child `standard`/nested RRF payload with nested BM25 and nested kNN, uniquely
   named inner hits, and the mandatory `subset_ids` filter;
2. executes the request once against each EMC2 and Mallinckrodt experiment index;
3. asserts a successful response, parent hits, both named inner-hit sections, and usable
   `chunk_index`/`text` fields;
4. runs the deterministic extraction routine and asserts at most `simple_per_call_fetch_count`
   unique chunk identities in the documented order, without a first-chunk query or fetch;
5. reports only structural counts and identifiers—never document text.

The mapping and Elasticsearch documentation prove API capability, but only this live request proves
the exact nested-kNN-in-standard-retriever syntax, named-inner-hit propagation, and response shape
for the deployed cluster.

### 3.5 First-chunk removal

Simple Mode query builders must omit `_first_chunk_filter_clause()` entirely. This avoids both the
extra nested inner-hit work and the E0/Tier 0 first-chunk output behavior. Chunk 0 appears only
when it is returned/selected by the active retrieval mode.

---

## 4. One-hop tool-call policy

The experiment config supplies a requested call count of 1, 2, or 3. The foundation model is
prompted to make that many retrieval calls, but the implementation accepts and executes every
retrieval call it actually emits in the sole tool round. MLflow stores:

- requested call count;
- actual emitted retrieval-call count;
- generic versus metadata-filtered call counts;
- deviation between requested and actual counts.

All calls from the one tool round execute concurrently. The current v3 `_call_tools` loop executes
them sequentially, so Simple Mode needs a dedicated concurrent path that preserves emitted-call
ordinal for later merge and trace labels.

### 4.1 Query style

| Retrieval mode | 1 requested call | 2 requested calls | 3 requested calls |
|---|---|---|---|
| BM25 | Compact keyword query | Distinct keyword/entity angles | Three distinct keyword/alias/aspect angles |
| Dense | One natural-language information need | Distinct semantic formulations | Three semantic/relationship/timeframe formulations |
| Hybrid ES RRF | Blended exact-entity + natural-language query | Keyword-oriented + semantic-oriented query | Keyword-oriented + semantic-oriented + mixed alternative angle |

The prompt must avoid near-duplicate calls, boolean syntax, invented email addresses, and claims
that the agent can conduct a second retrieval round.

### 4.2 Date/email filter policy

Date and email participant fields are allowed only as hard filters:

| Mode / requested calls | Metadata-filter tool availability |
|---|---|
| Any mode, 1 call | Not exposed; generic retrieval only |
| BM25 or dense, 2–3 calls | At most one filter call, only when user supplied explicit date/email values |
| Hybrid, 2 calls | Not exposed |
| Hybrid, 3 calls | At most one filter call, only when user supplied explicit date/email values |

All remaining calls are generic. An empty filtered result stays empty; it does not trigger an
unfiltered fallback query.

---

## 5. Cross-call merge policies

One retrieval call already returns a ranked, per-call passage list. Multiple calls need a global
merge after all tool calls complete. Raw `_score` values cannot be compared across distinct query
strings: BM25 scores, dense similarities, and per-query RRF scores are query-relative.

### 5.1 `round_robin` — fixed global context budget

Consume rank 1 from every actual call in emitted-call order, then rank 2 from every call, and so
on. Skip duplicate chunk identities until exactly `simple_global_context_chunk_count` unique
chunks are selected, or every list is exhausted.

This policy is deterministic, gives every query an equal opportunity to contribute, does not
compare query-relative scores, and performs no client reranking.

### 5.2 `current_union` — literal current control

Use current `merge_chunks()` behavior: preserve first-seen document order, deduplicate chunks,
sort chunks inside each document by start index, and retain the full unique union up to each
call's `simple_per_call_fetch_count` cap. Context can grow to roughly
`actual_call_count × simple_per_call_fetch_count`.

The implementation logs both configured counts, the actual unique chunk count, and the estimated
prompt-context size. This arm is an intentional control, not a clean merge-only comparison: it
changes both merge policy and context volume relative to `round_robin`.

---

## 6. S experiment suite

### 6.1 Naming

Simple Mode experiments use `S` instead of `E`. IDs encode stage and dimensions:

`S-<stage>-<mode>-c<requested_calls>-<merge>-f<per_call_fetch>-g<global_context>-r<reasoning>`

`f<per_call_fetch>` is the `simple_per_call_fetch_count` value used by every retrieval call.
`g<global_context>` is the `simple_global_context_chunk_count` limit applied after `round_robin`
cross-call merging. `g` is omitted from `current_union` arms because that policy retains the full
per-call-capped union without a global budget.

`rr` is rank-preserving round-robin, not reciprocal-rank fusion.

Examples:

- `S-A-bm25-c1-rr-f20-g10-rnone`
- `S-A-hybrid-c3-rr-f20-g10-rnone`
- `S-B-hybrid-c3-union-f20-rnone`
- `S-C-hybrid-c3-rr-f20-g20-rnone`
- `S-D-hybrid-c3-rr-f20-g15-rlow`

### 6.2 Full experiment table

Stage A fixes `simple_per_call_fetch_count = 20` for all arms. The global context budget
(`simple_global_context_chunk_count`) is selected per individual experiment and is shown in the
`g` part of each arm's ID; typical choices are `g10`, `g15`, or `g20` and are chosen at run time
rather than pre-declared as a mandatory full cross-product. Stage B arms use `current_union` and
carry no `g` segment because that policy does not apply a global selection cap.

| Stage | ID pattern | Retrieval | Requested calls | Merge | Per-call fetch | Global context | Reasoning | Purpose |
|---|---|---|---:|---|---:|---|---|---|
| A | `S-A-bm25-c{1,2,3}-rr-f20-g{chosen}-rnone` | BM25 only | 1, 2, 3 | round-robin | 20 | chosen per run | none | Lexical/call-count screen |
| A | `S-A-dense-c{1,2,3}-rr-f20-g{chosen}-rnone` | dense only | 1, 2, 3 | round-robin | 20 | chosen per run | none | Semantic/call-count screen |
| A | `S-A-hybrid-c{1,2,3}-rr-f20-g{chosen}-rnone` | ES RRF BM25+dense | 1, 2, 3 | round-robin | 20 | chosen per run | none | Hybrid/call-count screen |
| B | `S-B-<selected>-union-f20-rnone` | selected Stage A setup | selected | current union | 20 per call | full union | none | Context-volume control |
| C | `S-C-<selected>-rr-f20-g20-rnone` | selected setup | selected | round-robin | 20 | 20 | none | Context-depth comparison vs Stage A g-value |
| D | `S-D-<selected>-rr-f20-g{chosen}-rlow` | selected setup | selected | round-robin | 20 | same as a completed `rnone` counterpart | low | Reasoning at chosen global context |

All rows hold constant: one code-enforced retrieval round, generated metadata OFF, no parent
metadata ranking, no first chunk, flat-concatenated output, date/email filters only under the
policy above, 5 MiB filter OFF, and invocation concurrency 1.

### 6.3 Comparison map

| Comparison | Dimension isolated | Interpretation |
|---|---|---|
| `S-A-bm25-cN` vs `S-A-dense-cN` | lexical vs dense | Query-representation/retrieval-mode effect |
| `S-A-dense-cN` vs `S-A-hybrid-cN` | dense vs server-side hybrid RRF | Value of combining chunk lexical+dense signals |
| `S-A-<mode>-c1/c2/c3` | requested call count | Value of query diversity; actual count recorded separately |
| `S-A/B selected rr` vs `union` | merge plus context volume | Intentionally confounded control |
| `S-A gX` vs `S-C g20` | global context budget at fixed f20 | Value of more passages at fixed merge policy |
| `S-D gX none vs low` | reasoning effort at any selected global context `gX` | Value of GPT-5.1 reasoning tokens |
| E0 vs selected S arm | full baseline vs Simple Mode | Quality/latency trade-off; multi-dimensional comparison |

### 6.4 Execution order and stop/go gates

Execute from lowest expected latency to highest:

1. Stage A one-call BM25, dense, hybrid (at chosen g-value);
2. Stage A two-call BM25, dense, hybrid;
3. Stage A three-call BM25, dense, hybrid;
4. Stage B current-union control for Stage A Pareto candidate(s);
5. Stage C g20 comparison for selected candidate(s);
6. Stage D `low` arms for any selected global context, each paired with an already completed
   like-for-like `rnone` arm from Stage A or Stage C.

Stage D can contain as many selected `g` values as needed. Each `low` arm requires a like-for-like
`none` counterpart (from Stage A or C), isolating reasoning effort without replacing a screening
arm or conflating effort with context budget. GPT-5.1 supports custom tool calling for both `none`
and `low`; the chosen effort remains constant across all LLM calls in one run.

Advance a candidate only if it has acceptable rubric/citation/retrieval quality relative to E0 and
demonstrates a latency benefit at concurrency 1. Retain the complete Stage A matrix even when a
candidate fails: failures define the empirical boundary of Simple Mode.

### 6.5 Deferred dimensions

The first suite intentionally excludes:

- parent metadata retrieval and metadata generation;
- client MMR;
- client cross-call RRF;
- dynamic reasoning effort between planning and answer calls;
- a production Simple/Full router;
- a new index or reindex;
- persistent caching;
- invocation concurrency above 1.

Follow-up experiments may add generated metadata to retrieval and/or generation after the initial
no-metadata suite is complete.

---

## 7. Agent/config implementation design

### 7.1 v3 model configurations

Simple Mode configs use the `rag_agent_v3` namespace, beginning at `061.toml` / model version
`3.61`; subsequent arms allocate `062.toml`, `063.toml`, and so on. They do not reuse the
DSAS-2836 `092`–`095` namespace.

Each TOML derives from E0 `013.toml`. It retains evidence, citation, legal-language, error, and
structured-response requirements but removes multi-hop research and note/signature instructions.
It explains flat-concatenated range chunks, no generated metadata, the requested call count,
allowed query styles, and one-hop final-answer behavior.

Every initial config includes:

```toml
simple_mode = true
requested_retrieval_calls = 1          # 1, 2, or 3 by arm
simple_retrieval_mode = "bm25"         # "dense" or "hybrid_es_rrf"
simple_merge_policy = "round_robin"    # or "current_union"
simple_per_call_fetch_count = 20       # candidates returned by each retrieval call; fixed at 20 for all Stage A arms
simple_global_context_chunk_count = 10 # unique chunks selected after round_robin merging; chosen per experiment (e.g. 10, 15, 20)
include_metadata = false
max_tool_iterations = 1
reasoning_effort = "none"              # Stage D comparison uses "low"
max_completion_tokens = 12_000
```

`simple_per_call_fetch_count` controls the ES request `size` / per-call extraction cap; it feeds
the merge step. `simple_global_context_chunk_count` is the maximum unique chunks delivered to the
LLM after `round_robin` selection and adjacent-chunk concatenation; it is not used by
`current_union` arms. Neither replaces the existing `result_count` field, which continues to
control legacy E/Tier 2 retrievers.

`ModelConfig` gains typed, validated Simple Mode fields rather than untyped TOML access. Existing
E/TOMLs retain their current behavior through defaults.

### 7.2 Tool allowlist and one-hop execution

Simple Mode has a config-driven tool allowlist. It exposes:

- generic retrieval in every arm;
- metadata-filter retrieval only in arms allowed by the policy in section 4.2.

`WriteFile` and `ReadFile` are absent from the OpenAI tool schema, not merely unmentioned in the
prompt. The graph still performs current answer cleaning, citation normalization, snippet
extraction/repair, and structured output after the single retrieval round.

All retrieval calls emitted by the model during that round run concurrently. Each receives a tool
ordinal and configuration attributes. Once all are complete, Simple Mode applies the configured
cross-call merge policy, concatenates adjacent chunks, emits flat grouped XML, and invokes final
answer generation.

### 7.3 Simple retrieval adapter

Add a dedicated Simple provider/retriever rather than altering established E0/Tier 2 behavior. It:

1. issues chunk-only BM25, dense, or ES-RRF retrieval with ES `size = simple_per_call_fetch_count`;
2. disables the first-chunk query;
3. returns per-call ranked chunk lists capped at `simple_per_call_fetch_count`;
4. merges lists with `round_robin` (capping at `simple_global_context_chunk_count`) or `current_union`;
5. concatenates adjacent selected chunks into range-ID flat chunks;
6. returns `GroupedChunks`, flat XML, retrieved document IDs, and final relevant document IDs.

No candidate passage MMR encoding is present in this path.

---

## 8. Latency and MLflow observability

### 8.1 Timing layers

| Metric | Definition | Includes | Excludes |
|---|---|---|---|
| End-to-end invocation | Root r1-evals trace: prompt accepted to final structured response | all agent work after invocation semaphore acquisition | scoring, scorer LLM calls, pre-semaphore queue time |
| Observed operation latency | Child LLM/tool span wall duration | token waits, retries, backoff, all internal work | scoring |
| Successful-attempt latency | Explicit child-span attribute for final successful external request | only final API/ES request attempt | token wait, prior failures, retry backoff, pauses |

### 8.2 Required per-trace attributes

Every trace stores:

- requested/actual generic and metadata-filter tool-call counts;
- retrieval mode, merge policy, configured per-call fetch count (`simple_per_call_fetch_count`),
  configured global context budget (`simple_global_context_chunk_count`), actual selected chunk
  count, and actual serialized context size;
- invocation concurrency (`1`);
- Simple Mode config/model version;
- total root trace duration.

Every LLM span stores an operation label:

- `simple.query_plan`;
- `simple.answer_generation`;
- `simple.structured_output`;
- `simple.snippet_repair` when used.

Every retrieval span stores:

- `simple.retrieval_generic` or `simple.retrieval_metadata_filter`;
- tool ordinal;
- signal mode;
- configured per-call fetch count and actual returned count;
- ES request count and attempt count;
- observed duration and successful-attempt duration.

Dense/hybrid retrieval additionally records a child `EMBEDDING` span and
`simple.query_embedding_duration_ms` for the local E5 query embedding. This duration contributes
to observed retrieval-call latency; it is not an external successful-attempt metric.

#### Span attribute key contract

The attribute keys below are the literal strings consumed by
`r1_evals.mlflow_logging.log_simple_span_time_percentiles` in `r1-evals-new`. Because
`air_assist_core` intentionally has no dependency on `r1_evals` (production package,
dependency-minimization rule), these strings cannot be shared via import. Any future
instrumentation in `LlmModel.complete` or the Simple retrieval spans must reproduce them
character-for-character; a mismatch causes `log_simple_span_time_percentiles` to silently
aggregate zero metrics rather than raise an error.

| Attribute key | Set by | Canonical values / notes |
|---|---|---|
| `simple.operation` | Every Simple Mode LLM and retrieval span | See operation labels listed above |
| `llm.successful_attempt_latency_ms` | LLM spans (from `r1_rate_limiter` timing hook via `LlmModel.complete`) | Wall time of the final successful `await func(...)` attempt only, in ms |
| `simple.es_success_duration_ms` | Retrieval spans | Wall time of the final successful ES request attempt, in ms |
| `simple.query_embedding_duration_ms` | Dense/hybrid retrieval spans | Local E5 query-embedding wall time, in ms; not a successful-attempt metric |

#### Run-level metric naming convention

`log_simple_span_time_percentiles` aggregates span attributes into run-level MLflow metrics. The
operation label value (e.g. `simple.query_plan`) is normalised to lowercase with
non-alphanumeric characters replaced by `_`, and the redundant leading `simple_` prefix is
stripped before the outer `simple_` prefix is applied, so each metric is prefixed with a single
`simple_`:

| Metric key pattern | Source | Example |
|---|---|---|
| `simple_<operation>_observed_duration_ms_p{N}` | Span wall time (`end_time_ns - start_time_ns`) | `simple_query_plan_observed_duration_ms_p50` |
| `simple_<operation>_successful_attempt_ms_p{N}` | `llm.successful_attempt_latency_ms` attribute | `simple_query_plan_successful_attempt_ms_p95` |
| `simple_<operation>_es_success_ms_p{N}` | `simple.es_success_duration_ms` attribute | `simple_retrieval_generic_es_success_ms_p90` |
| `simple_query_embedding_duration_ms_p{N}` | `simple.query_embedding_duration_ms` attribute | `simple_query_embedding_duration_ms_p50` |

`{N}` is one of `50`, `90`, or `95`. The embedding metric is not operation-keyed because it is
recorded on a child `EMBEDDING` span, not on the retrieval operation span itself. All other
metrics produce one row per distinct `simple.operation` value present in the traces.

### 8.3 Retry-aware LLM measurement

The generic rate limiter remains independent from MLflow. In `r1_rate_limiter`, measure immediately
around each underlying `await func(...)` API attempt. On success, store final-attempt duration and
attempt count in context-local state so concurrent calls cannot overwrite each other. OpenAI SDK
retries remain disabled. `LlmModel.complete` reads the final-attempt value and sets it plus the
operation label on the active MLflow LLM span.

Failed attempts and retry waits remain visible through observed child-span and root-trace
durations but are absent from successful-attempt latency.

### 8.4 Retry-aware retrieval measurement

Create one MLflow TOOL/RETRIEVER span per concurrently executed Simple retrieval call. Instrument
the ES transient/auth retry helpers to retain final-success attempt duration and attempt count,
while keeping total tool duration as a separate attribute. Hybrid ES RRF is one ES request per
tool call; BM25/dense are also one request. If a technical fallback needs multiple internal ES
requests, record every successful request and the call critical-path maximum.

### 8.5 MLflow data and later analysis

Store raw per-variation measurements in traces and aggregate run-level p50/p90/p95 metrics for
the three timing layers. Store quality, retrieval, citation, retry/failure, requested/actual
call-count, both configured depth values, actual selected chunks, and serialized context-size
data. Do not generate experiment reports during S execution; reporting and Pareto analysis happen
after all configured S experiments complete.

---

## 9. Evaluation protocol

Reuse current answer-quality, retrieval, citation-format, reference, and range-citation scorers.
Before execution, audit every enabled scorer that consumes chunk IDs—especially citation-to-snippet
matching—to ensure `"i:j"` range IDs remain accepted for flat-concatenated output.

All S arms use the identical dataset partitions, rubric variations, seed, scorer configuration,
and invocation concurrency. Compare timing only within one dataset and concurrency setting.

MLflow must retain enough information for later:

- quality and citation validity by rubric/use case;
- retrieval recall/precision;
- full simple-config identity (including both `simple_per_call_fetch_count` and
  `simple_global_context_chunk_count`);
- requested vs actual calls;
- actual chunks/context delivered to the LLM;
- all three latency layers by operation;
- retry/failure rates separate from happy-path metrics.

---

## 10. Branch and project plan

| Project | Branch/base | Future purpose |
|---|---|---|
| `es-index-explorer` | Stay on `experiments` at `2496796` | Add this report only; no new branch |
| `air-assist-agent` | `DSAS-2836/simple-mode-base` from `DSAS-2836/tier2-base` at `89e65180` | Shared Simple graph/provider/retriever/config/tracing implementation |
| `r1_rate_limiter` | `DSAS-2836/simple-mode-timing` from local `main` `bda88cd904e81fdc164088f0c63d55331b4048c2` | Generic context-local successful-attempt timing hook |
| `r1-evals-new` | `DSAS-2836/simple-mode-evals` from `2a5cad9e387e12062b598bf8b96453fd3b4c9133`, then cherry-pick required range-citation changes | Range scorer audit/fixes and optional child-span timing aggregation |

The air-assist-agent Simple base retains direct ES, range citations, and existing experiment
infrastructure. It is the required common root for strict tool allowlisting, code-enforced
single-hop, concurrent retrieval-call execution, chunk-only ES retrieval, server-side RRF
extraction, flat-concatenated output, the no-first-chunk path, and timing spans. Its original main
base is `fa50a3198d16971056f578c48e239f0209b43810` (2026-06-18).

After this shared implementation exists, individual S experiment branches normally add only their
v3 model TOML/config values. S experiments are therefore TOML-configurable after the common root
is complete, but they are not TOML-only today.

The r1-evals original DSAS base is `2a5cad9e387e12062b598bf8b96453fd3b4c9133` (2026-06-18).
The es-index experiments base is `b4a96446f1b932ecbad4aa3a0f7a284587d6eb47` (2026-06-04).
`r1_rate_limiter` was not on a DSAS experiment branch; its current local main base is
`bda88cd…` from 2026-04-24.

No Elasticsearch mapping, index, or ingestion branch/change is planned.

---

## 11. Implementation gates and non-goals

Before the first S run, verify:

1. ES-side RRF nested BM25/kNN returns uniquely named, usable inner hits.
2. Simple extraction preserves a deterministic per-call order without client reranking, capped at `simple_per_call_fetch_count`.
3. Retrieval calls emitted in one tool round execute concurrently.
4. The first-chunk query/fetch/decorate path is absent.
5. Parent metadata fields/signals are absent from ranking and generation XML.
6. File tools are absent from the Simple Mode OpenAI tool schema.
7. The root trace excludes scoring and successful-attempt timing excludes waits/retries.
8. Flat-concatenated range citations pass all enabled scorers.
9. A multi-call `round_robin` run with `simple_per_call_fetch_count = 20` and
   `simple_global_context_chunk_count = 10` delivers at most 10 unique chunks to the LLM, with
   each call's ranked candidate list capped independently at 20.

Non-goals:

- production Simple/Full routing;
- generated metadata retrieval or generation;
- new indices or reindexing;
- persistent cache;
- dynamic reasoning effort in one invocation;
- client MMR or client cross-call RRF in the initial suite;
- experiment-report generation during execution;
- merging experimental changes into production main.

---

## 12. References

- [Elasticsearch RRF retriever](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
  and [retriever examples](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/retrievers-examples):
  server-side RRF, nested standard retrievers, and propagated inner hits.
- [GPT-5.1 model documentation](https://developers.openai.com/api/docs/models/gpt-5.1) and
  [prompting guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-1_prompting_guide):
  `none`, `low`, `medium`, and `high` reasoning effort; tool calling with `none`.
- [MLflow spans](https://mlflow.org/docs/latest/genai/concepts/span.md) and
  [manual tracing](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing.md):
  trace root spans, manual child spans, span attributes, and timing fields.
- `08-index-design-and-ingestion.md` and `09-retrieval-experiment-design.md`.
