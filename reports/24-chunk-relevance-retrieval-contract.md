# Chunk Relevance in Retrieval: Current Operating Model and Proposed Contract

## TL;DR

- Current qna MCP retrieval preserves an implicit ranked chunk order only until the service maps,
  caps, groups, and reorders the results. It **returns neither explicit chunk rank / score** nor the
  retrieval pipeline that produced the order.
- Retrieval strategy, MMR settings, candidate limits, and the final ranked‑chunk budget are
  resolved server‑side through LaunchDarkly. **Callers** can constrain document scope but **cannot
  select or inspect the retrieval algorithm for an individual call**.
- Q&A, Simple Mode (Fast Lookup), Deep Research, memo drafting, and future skill consumers have
  different latency, breadth, diversity, reranking, and context‑budget needs. **One hidden
  configuration cannot optimize all of them**.
- **The proposed contract returns** a canonical flat, **best‑first `chunks[]` list with integer rank**,
  optional typed numeric score, transparent applied‑pipeline metadata, caller‑selected profile,
  finite caller‑selected chunk budget, and capability‑dependent top‑down continuation.

## 1. Motivation

Retrieval tools increasingly serve as shared infrastructure rather than implementation details of
one aiR Assist flow. **A consumer that receives chunks relevant to a request benefits from knowing
why those chunks were selected, how they were ordered, and which limits shaped the result**.
Explicit relevance information lets a consumer:

- select a smaller top‑N context without guessing from a grouped payload;
- fuse or rerank results from concurrent tool calls;
- choose a latency, recall, or diversity profile appropriate to its task;
- diagnose retrieval failures and compare retrieval configurations;
- preserve evidence provenance for answer and memo generation;
- evaluate ranked retrieval reproducibly.

The author's remembrance of the project history is that the core contract predates the current
agentic usage and was shaped around a narrower retrieval setup. This write‑up revisits the contract
because the consumer set and use cases have expanded, not because the original design was
necessarily defective for its original purpose.

### 1.1 Consumers and design pressure

The documented dependency graph includes direct callers, evidence‑processing workflows, and
products that consume derived outputs:

**Retrieval backend -> MCP retrieval tools -> `air-assist-agent` tools and skills -> Q&A, Simple
Mode (Fast Lookup), Deep Research, and memo drafting -> artifacts, citations, and scopes -> aiR
Assist and ClaiR surfaces. Final drafted memos flow to `case-strategy-air-service` for persistence.**

The roles are not equivalent:

- **Direct retrieval orchestrator:** `air-assist-agent` selects and invokes retrieval tools. Its
  Q&A and drafting skills process the returned evidence.
- **Evidence‑processing consumers:** answer synthesis and memo drafting consume retrieved chunks
  and citations. The memo‑drafting skill runs in the agent harness; `case-strategy-air-service`
  persists and serves the finished memo rather than selecting source chunks
  ([a4CS MCP](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1742962810/aiR+for+Case+Strategy+MCP)).
- **Product and UI consumers:** aiR Assist and ClaiR consume derived artifacts, document IDs,
  filters, scopes, and citations. `claire-service` hydrates document lists; the a4CS viewer package
  loads saved memos and reports citation clicks
  ([ClaiR Search and Q&A flow](https://relativity-oda.atlassian.net/wiki/spaces/~712020f52569fec67e4b74bfc314f9d70e4636/pages/1782677688/Clair+Search+Q+A+Flow),
  [memo viewer contract](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1932198083/a4CS+Memo+Viewer+Contract+for+ClaiR+ADR)).
- **Future consumers:** the shared MCP and skill direction allows other authorized agents and
  surfaces to reuse the same retrieval capability. Internal platform guidance explicitly treats
  tools as shared access and skills as task‑specific operating knowledge
  ([tool, skill, and agent decision framework](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1909031054/Decision+Tree+for+Building+Tools+Skills+Agents+or+New+Surface+Areas+in+Relativity)).

These capabilities have different maturity. Current `qna-service` MCP V2 / V3 retrieval is implemented.
The shared `embedding-service` MCP V4 tools are documented direction at the pinned evidence
baseline. Deep Research is a product / research direction, and Simple Mode is being implemented
(as of the date of writing this document). The consumer argument therefore combines current direct
and indirect dependencies with actively developed and proposed uses; it does not present every use
as deployed.

### 1.2 Current control limitation

Current callers have useful scoping controls, but **not retrieval‑algorithm controls**. The model can
supply query terms and metadata filters. Trusted infrastructure injects subset and focus scope.
Neither can choose BM25, MMR, hybrid score combination, or RRF for an individual call. The caller
also **cannot request a ranked chunk budget**. Strategy, candidate‑pool, MMR, and final ranked‑chunk
limits are resolved server‑side through LaunchDarkly.

The response discloses neither an explicit chunk score / rank nor the effective retrieval pipeline.
Consequently, callers cannot reliably interpret the relevance ordering that survives the service,
even though they can constrain which documents are eligible.

The planned MCP V4 work does not appear to change this picture. Its engineering plan is thorough
on non‑functional requirements such as workspace isolation, security, performance,
observability, cancellation, avoiding N+1 calls, parser compatibility, and rollout control. It
does not appear to identify relevance, rank, or caller retrieval intent as explicit functional
requirements, which leaves the retrieval capability itself narrower than the surrounding
engineering suggests
([MCP V4 engineering plan](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1792245775/Engineering+Plan+MCP+V4+Clair+Search+Q+A+AA+V2)).

### 1.3 Evidence that one configuration is insufficient

The case for caller‑aware retrieval is supported by several independent observations:

1. **Task objectives conflict.** Standard Q&A needs a bounded balance of relevant and diverse
   evidence. Simple Mode prioritizes latency, precision, and a small context budget for factual
   lookup, with a documented direction of direct answers with citations in less than 30 seconds
   ([Simple Mode](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1698595003/Simple+Fast+mode+-+Intro)).
   Deep Research prioritizes breadth and recall, potentially across hundreds of documents, and
   accepts longer execution
   ([Deep Research](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1640797199/Deep+Research+-+Intro)).
   One candidate count, final chunk count, fusion policy, and reranker cannot optimize all three.
2. **MMR settings are dataset‑dependent.** The MMR lambda controls relevance versus diversity and
   may need to differ even for consumers in the same LD targeting scope. Experiments found
   inconsistent effects from changing `mmr_lambda`. Increasing `results_sent_to_gpt` sometimes
   improved quality while increasing token usage. The study measured tokens, not retrieval latency
   ([MMR configuration study](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1736540182/2026-07-14+Retrieval+Configuration+Study+MMR+Delta+Document+Count)).
3. **Corpus shape changes vector trade‑offs.** Slack research from early September 2026 reports
   that embedding model and dimensionality effects reversed between long and short‑document corpora. This is
   research evidence, not a fleet‑wide production guarantee
   ([Slack discussion](https://kcura-pd.slack.com/archives/C05PKQYTZTM/p1788349409928209)).
4. **Post‑retrieval needs differ.** Q&A, Simple Mode, Deep Research, and memo drafting may choose
   different context budgets, diversity policies, reranking, or cross‑call fusion. Memo drafting
   also needs sufficiently broad, defensible, citation‑grounded evidence for long‑form synthesis.
   **They cannot make these choices safely from a response that has lost global chunk rank**.
   A consumer with rank and an interpretable score may also decide whether marginal results justify
   requesting a subsequent page, particularly for Deep Research. Score alone cannot establish
   whether another page exists, so this also requires explicit continuation metadata.
5. **Document Search and chunk Q&A are different contracts.** Search returns document sets,
   exact counts, pagination, filters, and deterministic document‑level sort order. Q&A retrieves
   and ranks evidence chunks
   ([ClaiR Search V3 mental model](https://relativity-oda.atlassian.net/wiki/spaces/~712020f52569fec67e4b74bfc314f9d70e4636/pages/1774420235/ClaiR+Search+V3+Tool+Design+Mental+Model)).

Simple Mode provides a concrete example. Simple Mode research selected a fixed‑budget,
rank‑preserving round‑robin across per‑call chunk lists
([Simple Mode experiment design](https://github.com/mierzejk-rel/es-index-explorer/blob/81905343d0dd3e6bf0708733ed9ba99c4073e696/reports/10-simple-mode-experiment-design.md#L250-L257)).
The agent implementation could not recover that ordering from the grouped MCP payload, because it
contained neither a global chunk rank nor score
([PR #623 review comment](https://github.com/relativityone/air-assist-agent/pull/623#issuecomment-5698164026)).
The resulting design decision was to use a complete deduplicated union without a global
relevance‑based cap. This write‑up does not analyze later Simple Mode branches.

### 1.4 Information Retrieval (IR) evaluation

Ranked retrieval evaluation is another consumer of explicit rank and identity. Metrics such as
[Discounted Cumulative Gain (DCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain)
and [Normalised Discounted Cumulative Gain (NDCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)
combine system result positions with externally supplied relevance judgments. The tool's `_score`,
similarity, or MMR score is not ground‑truth relevance, and retaining rank alone does not make
an evaluation valid. However, stable chunk identity, actual returned rank, and the applied retrieval
pipeline support reproducible and attributable evaluation when the evaluated corpus/index
generation and other relevant configuration are fixed
([Elasticsearch ranking evaluation](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/search-rank-eval)).

Chunk‑level judgments also depend on chunk lineage. Chunk identifiers are stable only within one
chunk/index generation, so re‑chunking or replacing them invalidates existing judgments unless an
explicit migration or lineage mechanism maps old chunks to new ones.

## 2. Scope and terminology

### 2.1 Evidence baseline

Current implementation claims in this report are pinned to:

- `qna-service` local `main` at
  [`15ee362c`](https://github.com/relativityone/qna-service/tree/15ee362c9e4470db7a774a431f5a42e37e14d577);
- `air-assist-agent` local `main` at
  [`87e422bf`](https://github.com/relativityone/air-assist-agent/tree/87e422bf05d00c7a0926c256538b1941d19ef515);
- `embedding-service` local `main` at
  [`1e310137`](https://github.com/relativityone/embedding-service/tree/1e31013745d51969836163caf1156197a27e5ae5);
- [PR #623](https://github.com/relativityone/air-assist-agent/pull/623) context at
  [`9491f0b6`](https://github.com/relativityone/air-assist-agent/tree/9491f0b6b6a1bf22f8d6dd158d8e4291543a0177).

Here, "local `main`" means the `main` branch as cloned on 17 September 2026. Those revisions may
trail upstream, and implementation‑status statements hold as of them. In particular, `GetSearchCapabilities`, `SearchDocuments`, and MCP V4
`GetRelevantDocuments` are absent from the pinned `embedding-service` revision. They are treated as
documented direction, not current implementation. No repository refresh or re‑pin was used.

### 2.2 Status and version terminology

The report distinguishes:

- **Current implementation:** behavior verified in the pinned source code.
- **Documented direction:** a Confluence contract or engineering plan, not necessarily implemented.
- **Observed research evidence:** experiment or investigation results with their stated scope.
- **Reasoned consequence:** a deduction from verified implementation, not an observed incident.
- **Proposed design:** the author's recommendation, not an agreed owner‑team decision.

**Agent v4** means the `air-assist-agent` `DEEP_AGENT` graph. **MCP V4** means the planned tool
contract/version. The qualifiers are used consistently because capitalization alone is too easy
to misread.

### 2.3 Retrieval terms

In the current flat Elasticsearch index, the smallest indexed search result is a chunk. It is an
Elasticsearch document, but not a parent Relativity document. A parent Relativity document may
have many chunks.

- **Score:** a numeric value computed by a retrieval or reranking stage. Its scale and direction
  depend on the algorithm.
- **Similarity:** a numeric relationship between query and candidate representations, such as
  cosine similarity. It is not automatically a calibrated relevance probability.
- **Rank:** an ordinal position within one logical result set. In the proposed contract, rank 1
  is best, and ranks continue across continuation pages of the same result set.
- **Ordering:** the sequence in which results are returned. It can represent relevance, an
  explicit field sort, or only a deterministic tie‑break.
- **Fusion:** combination of outputs or scores from multiple retrievers, such as Reciprocal Rank
  Fusion (RRF) or weighted score addition.
- **Reranking:** reordering an existing candidate set using an additional algorithm such as
  Maximum Marginal Relevance (MMR).
- **Filter‑only retrieval:** selection by hard constraints without a relevance‑producing query.

The proposed contract is backend‑ and index‑layout‑agnostic. Elasticsearch provides the current
implementation and evidence, but the future retrieval system may use another engine or storage
design.

## 3. Current retrieval strategies and available signals

**Status: current `qna-service` implementation unless marked as documented direction.**

The current `qna-service` exposes three selectable strategy values:
`Bm25Search`, `Bm25SearchWithMmr`, and `RrfSearch`
([strategy enum](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyType.cs)).
The selection is not an MCP argument. It is resolved through LaunchDarkly. If the flag is
unavailable, disabled, empty, or invalid, the settings provider falls back to `RrfSearch`
([settings provider](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/RetrievalStrategySettingsProvider.cs)).

Filter‑only retrieval is covered separately because it is a non‑ranked path rather than one of
the selectable strategies.

### 3.1 BM25

`Bm25Search` sends a `multi_match` query over `body` and `title`, with eligibility constraints in
filter context
([BM25 strategy](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/Bm25SearchStrategy.cs)).
Elasticsearch returns a positive floating‑point `_score` and, without another sort, orders hits
by decreasing score
([full‑text relevance](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/full-text-search.html)).

BM25 `_score` is useful for ordering results within the same query and corpus state. It is not a
probability, and ratios or differences should not be interpreted as calibrated relevance. It
should not be compared directly across different queries.

Available at the Elasticsearch response boundary:

- numeric BM25 `_score`;
- ordinal hit rank from response position.

Preserved by `qna-service` before grouping:

- ordinal order only.

Exposed by the MCP result:

- **no explicit score or chunk rank**.

### 3.2 BM25 followed by MMR

`Bm25SearchWithMmr` uses BM25 to obtain a candidate pool, removes chunks with empty content,
embeds the query and every remaining candidate body, and applies Maximum Marginal Relevance
([retrieval service](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/RetrievalService.cs#L82-L163)).

MMR first selects the candidate with highest query cosine similarity. Later selections maximize a
combination of query similarity and a penalty for similarity to already selected chunks
([MMR selector](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/MaximumMarginalRelevanceSelector.cs#L49-L106)).
Its output order is therefore an MMR selection order, not pure descending query relevance.

The weighting parameter `mmr_lambda` controls the relevance‑versus‑diversity balance for all
selections after the first. Values are accepted in the inclusive range `[0, 1]`: `0` emphasizes
diversity and `1` emphasizes query similarity. The pinned code default is `0.5`; the
`qna-service.retrieval.strategy.configuration` LD payload can override it
([MMR defaults](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyDefaults.cs#L25-L32),
[settings merge and validation](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/RetrievalStrategySettingsProvider.cs#L108-L152)).

The implementation computes query similarity, candidate‑to‑selected similarity, and an
iteration‑specific MMR value. It returns only selected candidate indices. The strongest surviving
indicator is the final ordinal MMR selection order. No BM25 score, cosine similarity, MMR value,
or explicit rank is attached to the returned chunks.

`MmrTopK` may limit the selected set below `ResultsSentToGpt`. The retrieval service then takes
`ResultsSentToGpt`, and the document provider applies a second prefix cap. The second cap is
normally redundant but confirms that the MCP caller does not control the result count.

Exposed by the current MCP result:

- **no explicit BM25 score, query similarity, MMR score, or final chunk rank**.

### 3.3 The two implementations behind `RrfSearch`

`RrfSearch` is one selectable strategy name, but its implementation builds two materially
different forms of hybrid search. The following matrix applies only after `RrfSearch` has been
selected:

| Request condition                                                                                                                                | Search construction | Fusion and score semantics                                                                                                 |
|--------------------------------------------------------------------------------------------------------------------------------------------------|---|----------------------------------------------------------------------------------------------------------------------------|
| Non‑empty subset, no include / exclude, including focus and ES‑native metadata filters                                                           | Top‑level BM25 query plus kNN clause | Elasticsearch combines lexical and kNN scores directly. This is hybrid score combination, not Reciprocal Rank Fusion.      |
| Non‑empty subset with include/exclude, including the Object Manager metadata path **with a query** that supplies matching IDs as an include list | RRF retriever with standard lexical and kNN child retrievers | Reciprocal Rank Fusion. Final score is derived from child ranks rather than raw BM25 and kNN score addition.               |
| Empty subset                                                                                                                                     | RRF retriever, either from configured API‑call JSON or the filtered fluent builder | Reciprocal Rank Fusion. Listed for implementation completeness; reachability through a specific MCP call was not verified. |

The branch conditions are implemented in
[`RrfSearchStrategy`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/RrfSearchStrategy.cs#L33-L89).
Subset, focus, and metadata constraints are constructed as filters, while the Object Manager query
path places matched IDs in the include list
([filter builder](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticQueryFilterBuilder.cs)).

This distinction has a practical consequence: ES‑native and Object Manager metadata routes can
produce different fusion and score semantics for similar user intent when `RrfSearch` is active.

For true RRF, Elasticsearch calculates a final score from child‑retriever ranks:
`sum(1 / (rank_constant + rank))`
([RRF documentation](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/rrf.html)).
The scalar is useful for final ordering but is rank‑derived, query‑local, and not an absolute
relevance measurement. `qna-service` does not expose it or the component ranks.

For the top‑level BM25+kNN construction, Elasticsearch combines query and kNN contributions in
the hit score. That score is likewise query‑local and does not expose a calibrated common
relevance scale.

Exposed by the current MCP result for either `RrfSearch` implementation:

- **no explicit final score, component score / rank, or chunk rank**.

### 3.4 kNN and dense vectors

kNN is the retrieval algorithm over dense vectors. It is present as a component of the current
`RrfSearch` implementation, but no standalone kNN strategy exists in the qna strategy enum.

For cosine vector similarity, Elasticsearch transforms similarity into `_score`; the raw cosine
can be recovered as `(2 * _score) - 1`
([kNN scoring](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/knn-search.html)).
The score is monotonic with similarity, but it is not a probability that a chunk is relevant.
A standalone kNN tool could expose both this numeric signal and its ordinal result rank, but the
current MCP tool cannot request kNN independently.

Exposed by the current MCP result:

- **no standalone kNN result or score**;
- when kNN participates in hybrid retrieval, **its contribution
  and final rank are not exposed**.

The current index mapping always provides a dense‑vector `embedding` field
([index mapping](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Infrastructure/Elasticsearch/ElasticsearchClientWrapper.cs#L105-L145)).
Whether it is populated is controlled independently during indexing, as described in section 4.

### 3.5 Other fusion available in Elasticsearch

Elasticsearch also offers the `linear` retriever. It combines child scores using weights and can
normalize each child result set using `none`, `minmax`, or `l2_norm`
([linear retriever](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever)).
This is not implemented as a current qna strategy or exposed through the MCP retrieval tool. It
is included to show that RRF is not the only possible fusion design and that future backends may
offer other mechanisms.

### 3.6 Relevance indicator summary

The numeric `score` can remain one nullable JSON number when accompanied by an explicit
`scoreType` and `scoreDirection`. Ordinal rank is a separate nullable integer and should not be
encoded in the floating‑point score field.

| Retrieval algorithm or path | Numeric signal available during retrieval | Ordinal order available before grouping | Explicitly exposed by current MCP |
|---|---|---|---|
| BM25 | BM25 `_score`, higher is better within one query | Elasticsearch hit rank | Neither score nor chunk rank |
| BM25 followed by MMR | BM25 score before reranking; query similarity and iteration‑specific MMR value during reranking | Final MMR selection order | Neither numeric signal nor final chunk rank |
| Additive BM25+kNN hybrid | Combined Elasticsearch `_score`; component contributions exist in the query computation | Final hybrid hit rank | Neither combined/component score nor chunk rank |
| True RRF | Rank‑derived RRF score and child ranks | Final fused rank | Neither RRF/component information nor chunk rank |
| Standalone kNN | Cosine‑derived or other configured vector score | kNN hit rank | Not implemented as a standalone current strategy |
| Filter‑only | No relevance score | No relevance rank | No relevance indicator; order depends on the server path |

This table reports algorithm output and current exposure, not one common relevance scale.

### 3.7 Cross‑service vector semantics

Within `qna-service`, the MMR query embedding and candidate‑chunk embeddings use the same model
resolved for that request. `RrfSearch` generates its query vector using `qna-service`'s resolved
model but compares it with document vectors produced earlier by `embedding-service`.

Both services use the key `Embedding.Model.JSON`, but resolve it independently under different
LaunchDarkly contexts
([qna model key](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/Constants/LaunchDarklyOpenAiRequestsKeys.cs),
[embedding‑service resolution](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Infrastructure/LaunchDarkly/LaunchDarklyClientWrapperSingleton.cs#L49-L87)).
**No checked code proves that index‑time and query‑time models must match**. A mismatch would make
the vector comparison invalid or materially degrade it. This is a reasoned risk, not an observed
production incident.

### 3.8 MCP V4 documented direction

**Status: documented direction, absent from the pinned `embedding-service` implementation.**

The MCP V4 engineering plan specifies BM25+MMR retrieval and explicitly excludes kNN from the
planned scope
([MCP V4 engineering plan](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1792245775/Engineering+Plan+MCP+V4+Clair+Search+Q+A+AA+V2)).
The working contract corroborates the BM25+MMR direction through its references to the BM25
candidate pass and pre‑MMR IDs
([MCP V4 working contract](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1780711469/ClaiR+MCP+Tools+AA+V2+Working+Contract)).
If implemented as specified, callers would not be able to request standalone kNN or hybrid
retrieval through that surface.

## 4. Current retrieval control plane: LaunchDarkly

**Status: current behavior at the pinned `qna-service` and `embedding-service` revisions, except where
marked as a point‑in‑time production snapshot or reasoned consequence.**

The current qna MCP V2 / V3 `GetRelevantDocuments` method accepts `query`, `subsetId`, and optional
`focusedDocumentIds`; the aiR Assist provider normally hides and injects trusted scope arguments.
The metadata variant adds optional date / email filters and can omit `query` when at least one
filter is supplied. Neither tool accepts a ranked chunk count, sort order, or continuation token
([qna MCP V3 tool](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.API/Mcp/Tools/V3/DocumentProviderTool.cs)).

Three controls jointly affect retrieval:

1. `air-assist-should-use-text-search-strategy` determines whether chunk embeddings are skipped
   during indexing.
2. `qna-service.retrieval.strategy.configuration` selects `RrfSearch`, `Bm25Search`, or
   `Bm25SearchWithMmr` and provides strategy settings. These settings include candidate counts,
   `results_sent_to_gpt`, RRF parameters, MMR lambda, MMR top‑K, and embedding batch size
   ([retrieval defaults](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyDefaults.cs)).
3. `Embedding.Model.JSON` is one key resolved independently by `qna-service` for query / rerank
   embeddings and by `embedding-service` for indexing embeddings.

### 4.1 Different evaluation contexts

`qna-service` evaluates the retrieval strategy with service, region, cluster, tenant, workspace,
and user attributes when available
([qna LD context](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/LaunchDarkly/LaunchDarklyContextBuilder.cs)).

The indexing activity evaluates `air-assist-should-use-text-search-strategy` through the singleton
parameterless path. Its context contains service, region, and cluster, but not tenant or workspace
([indexing activity](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Application/Temporalio/Activities/ConsumerActivities.cs#L257-L266),
[singleton LD wrapper](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Infrastructure/LaunchDarkly/LaunchDarklyClientWrapperSingleton.cs#L35-L39)).

The model key is resolved by both services under different service and targeting contexts. These
flags can be configured through LaunchDarkly selectors and predicates, but they do not form one
per‑call control surface.

### 4.2 Production snapshot (September 2026)

The linked [Slack thread](https://kcura-pd.slack.com/archives/C05PKQYTZTM/p1788349409928209)
records the production values as the author verified them when the message was posted in early
September 2026. It reported text‑search‑only indexing together with `Bm25SearchWithMmr`. This report
does not re‑verify those values or claim they are current now.

That pair was internally coherent. BM25+MMR generates query and candidate embeddings on demand
and does not require indexed vectors.

### 4.3 Compatibility is not enforced in service code

**No runtime cross‑validation between the indexing switch and retrieval‑strategy flag was found in
the pinned services**. External rollout rules may exist, but they are not enforced by either
application:

- **Verified:** indexed embeddings can be enabled while BM25+MMR is selected. Vectors are stored
  but MMR retrieves text, re‑embeds candidates, and ignores the stored vectors.
- **Reasoned consequence:** indexed embeddings can be disabled while hybrid / RRF is selected. In
  that case **the kNN arm has no indexed document vectors and the intended hybrid behavior is
  expected to collapse toward its lexical contribution**. This was not observed as a production
  incident in this research.

LaunchDarkly remains suitable for rollout, safety gates, environment defaults, and experiments.
**It is a poor sole control for per‑call intent**. Callers matching the same targeting rule inherit
the same algorithm and result budget even when their latency, breadth, context‑window, or
evidence‑coverage requirements differ.

## 5. Relevance lifecycle and current losses

**Status: current `qna-service` implementation unless marked otherwise.**

### 5.1 Elasticsearch response

Elasticsearch `_score` is calculated at query time. It is not part of the index mapping. Ranked
searches return each hit's score and are ordered by `_score` by default
([search API](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/search-your-data.html)).
When results are sorted by another field, scores may not be computed unless `track_scores` is
enabled
([sort documentation](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/sort-search-results.html)).

The current search strategies could read `SearchResponse.Hits[].Score`, but instead map
`searchResults.Documents`
([base search strategy](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/Base/BaseSearchStrategy.cs#L80-L108)).
`MapDocumentsToChunks` preserves iteration order but creates `DocumentChunk` with only chunk ID,
document ID, control number, and content. **Numeric `_score` and an explicit ordinal rank are
not retained by the current mapping**
([mapping](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/Base/BaseSearchStrategy.cs#L151-L166),
[chunk model](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/DocumentChunk.cs#L3-L11)).

### 5.2 Reranking, permissions, and capping

BM25 and hybrid / RRF preserve Elasticsearch hit order at the retrieval‑service boundary.
BM25+MMR replaces BM25 ordering with MMR selection order. The BM25 candidate IDs are retained
separately before MMR, so they are not a final relevance ordering.

Permission filtering removes inaccessible documents while preserving the relative order of
surviving chunks. Query paths then take the strategy‑configured `ResultsSentToGpt` prefix before
grouping
([document provider](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs#L158-L184)).

**The cap is not caller‑controlled**:

- `ResultsSentToGpt` limits ranked retrieved chunks.
- For BM25+MMR, `MmrTopK` can impose a lower effective limit.
- The MMR retrieval method applies `ResultsSentToGpt`, and the provider applies a second normally
  redundant prefix cap.
- Access filtering and short result sets can reduce the actual count.
- A separately attached unranked `firstChunk` can add one text fragment per returned parent
  document beyond the ranked retrieved‑chunk cap.

Consequently, `results_sent_to_gpt = 25` does not guarantee exactly 25 text fragments in the MCP
payload. The V3 tool description says "top 25 documents," but the query path actually caps ranked
chunks before grouping
([V3 MCP tool](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.API/Mcp/Tools/V3/DocumentProviderTool.cs#L35-L55)).

### 5.3 Grouping and remaining order

The provider groups the capped chunk list by parent `DocumentId`. It records the first position
at which each parent appears, orders groups by that position, and sorts chunks within each group
by `chunkId`
([grouping implementation](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs#L318-L342)).

This preserves one coarse relevance indicator:

- document groups remain ordered by the best‑ranked surviving chunk from each document.

It loses the original global chunk order. For example, a ranked stream `[A5, B2, A1]` becomes
groups `A:[1,5]`, `B:[2]`. **Without an explicit rank field, the original sequence cannot generally
be reconstructed**.

**Grouping itself does not require losing rank**. A group could retain `retrievalRank` per chunk or
the response could carry a parallel flat identity list. **The current loss results from discarding
rank and then sorting by chunk ID**.

### 5.4 `retrievedDocumentIds`

`retrievedDocumentIds` does not provide a universal relevance order:

- BM25 and RRF derive distinct document IDs from their pre‑provider chunk list.
- BM25+MMR captures IDs before reranking and top‑K selection.
- The ES‑native metadata path returns the Object Manager access‑result list.
- Filter‑only paths use matching/candidate IDs rather than ranked final chunks.

The field is useful for broader candidate provenance, but not as a substitute for final chunk
rank.

### 5.5 Filter‑only paths

The current V2 / V3 metadata tool permits an omitted query when at least one supported filter is
provided
([V3 metadata tool](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.API/Mcp/Tools/V3/DocumentProviderTool.cs#L223-L329)).
Agent v3 models `query` as optional. Agent v4 inherits the legacy MCP schema when that tool is
used. Agent v3 also falls back to generic keyword retrieval if metadata retrieval returns no
documents
([agent v3 retrieval](https://github.com/relativityone/air-assist-agent/blob/87e422bf05d00c7a0926c256538b1941d19ef515/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/rag_agent.py#L167-L197)).

Two server paths have different order:

- **ES‑native metadata‑only:** Elasticsearch retrieves first chunks sorted by `documentId`,
  permission filtering preserves survivor order, and `ResultsSentToGpt` caps the chunks before
  grouping
  ([ES first chunks](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/ElasticsearchClientWrapper.cs#L280-L329)).
- **Object Manager metadata‑only:** Object Manager identifies matching Relativity document IDs.
  The code takes `MaxCandidates` from an unordered set, fetches first chunks for those IDs from
  Elasticsearch, groups them, and then caps document groups at `ResultsSentToGpt`
  ([metadata paths](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs#L600-L624),
  [first‑chunk retrieval](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs#L728-L802)).

Neither path has relevance ordering. The ES‑native path is deterministic by document ID. **The
Object Manager path does not guarantee deterministic order**.

The planned MCP V4 `GetRelevantDocuments` contract requires a nonblank query, so this filter‑only
behavior is not part of that documented future surface.

### 5.6 `firstChunk`

On legacy query paths, `firstChunk` is usually fetched separately and carries no retrieval rank.
On metadata‑only paths it may be derived from the returned first chunks. A chunk with ID `0` that
was genuinely present in the ranked retrieved collection is an ordinary ranked chunk; it is not
special merely because its ID is zero.

The string‑versus‑object wire representation of the separate `firstChunk` field is outside this
report's scope.

## 6. Response‑shape assessment

**Status: current V2 / V3 implementation, planned MCP V4 contract, and author's assessment are
labelled separately below.**

### 6.1 Current and planned grouped contracts

Current qna MCP V2 / V3 returns typed grouped results. Agent v3 requires
`structured_content["documents"]` and validates each entry as a grouped model
([agent v3 parser](https://github.com/relativityone/air-assist-agent/blob/87e422bf05d00c7a0926c256538b1941d19ef515/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/rag_agent.py#L167-L188)).

Agent v4 accepts either grouped or flat payloads. On the pinned `main`, its grouped‑shape detection
checks all list items, not only the first item
([agent v4 formatter](https://github.com/relativityone/air-assist-agent/blob/87e422bf05d00c7a0926c256538b1941d19ef515/packages/air_assist_core/src/air_assist_core/registry/graphs/v4/chunk_formatter.py)).
The flat branch groups chunks and sorts them by chunk ID. No currently verified `qna-service` MCP V2 / V3
producer requires that fallback.

**The documented MCP V4 working contract** preserves grouped V2 output and **contains no score field**.
It describes `documents[]` as ordered by "first‑appearance BM25 relevance rank" and
`retrievedChunks[]` as ordered by ordinal
([working contract](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1780711469/ClaiR+MCP+Tools+AA+V2+Working+Contract)).
That document also specifies BM25+MMR retrieval, so **its ordering phrase is ambiguous**: it does not
establish whether group order refers to the pre‑MMR BM25 order or post‑MMR selection order. This
requires clarification from the contract owners and is not silently resolved here.

### 6.2 Assessment

**Grouping** is convenient for a consumer that wants document‑order reading context, but it **is a poor
canonical wire shape for retrieval rank**:

- every chunk already carries its parent document ID, so a consumer can group if needed;
- a consumer that needs a flat ranked list must first ungroup;
- ungrouping does not recover the original global order unless each chunk retained rank;
- grouping can remain a consumer helper without determining the shared retrieval contract.

**The proposed contract therefore uses a flat ranked chunk list**. This is a recommendation
of the author's, not an agreed change to `qna-service` or planned MCP V4.

As a secondary aiR Assist cleanup, the agent v4 flat‑payload fallback can be removed after one
cross‑service contract test confirms that every supported producer returns grouped output. This
is intentionally a restrained recommendation, not part of the retrieval contract itself.

## 7. Proposed retrieval and control contract

**Status: author's recommendation for discussion. It is not an agreed `qna-service`,
`embedding-service`, or product contract.**

### 7.1 Scope and principles

This proposal governs chunk‑level relevant‑document retrieval such as `GetRelevantDocuments`.
It does not redefine the separately planned `SearchDocuments` contract. Document search retains
document IDs, exact counts, pagination, filters, and document‑field sorting.

**The retrieval implementation behind MCP should be free to use Elasticsearch or another mature
retrieval system** and should not be coupled to a flat, nested, or parent‑child index layout. The
stable contract describes retrieval concepts. Engine‑specific details can be returned as
diagnostics when they help consumers interpret or reproduce results.

Different profiles may require different backend or index capabilities, such as a compatible
vector space, lexical fields, sparse representations, or stable continuation. Capability
discovery should expose those requirements without making their physical implementation part of
the wire contract.

The contract should follow five principles:

1. The caller can express retrieval intent for each call.
2. The service advertises which profiles and limits are available.
3. The response reports the pipeline that was actually applied.
4. Ranked chunks retain an explicit final rank and are returned best‑first.
5. Non‑ranked retrieval is identified as such and still has deterministic ordering.

The design also follows the open‑closed principle. Consumers rely on stable rank, score, profile,
and continuation semantics, while versioned capability discovery can introduce additional
profile, stage, encoder, and backend identifiers without requiring existing consumers to change.

### 7.2 Capability discovery

The MCP V4 plans already define a capability‑discovery pattern through
`GetSearchCapabilities`. That tool is not present at the pinned implementation baseline. If it is
delivered, retrieval profiles should extend it. Otherwise a new versioned discovery mechanism
should provide the same information
([MCP V4 working contract](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1780711469/ClaiR+MCP+Tools+AA+V2+Working+Contract)).

For every available retrieval profile, discovery should report:

- stable profile name and purpose;
- whether a query is required;
- supported filters and scope constraints;
- whether results are relevance‑ranked;
- score type, range when bounded, and ordering direction when a score is exposed;
- supported fusion and reranking behavior;
- minimum, maximum, and default ranked chunk count;
- whether non‑relevance field sorting is supported;
- whether deterministic ranked output is guaranteed and the conditions under which that guarantee
  holds;
- tie semantics for ranked results;
- required retrieval/index capabilities, such as compatible indexed vectors;
- `vectorSpaces[]`, with one entry per vector‑using retrieval arm, each carrying the arm role and
  an opaque, versioned `vectorSpaceId` (encoder compatibility identifier);
- continuation support and its semantics.

This lets callers select only behavior the active workspace and backend can support. LaunchDarkly
may gate profile availability or defaults, but it should not silently change the semantics of an
accepted request.

A list rather than a single identifier keeps hybrid profiles expressible, for example one dense and
one sparse arm. Each `vectorSpaceId` should be sufficient to establish that query and indexed
vectors use the same model revision, dimensions, normalization, and dense or sparse representation.
Model name, version, dimensions, normalization, and encoder role can remain optional diagnostics
rather than mandatory top‑level metadata.

Capability bounds may explicitly report `unknown` or `unbounded` for total candidate availability
when that concept is meaningful. The minimum cannot be unbounded, and every MCP response page and
`maxChunks` value should remain finite for transport, latency, and context safety. Consumers use
continuation rather than requesting an unbounded response.

### 7.3 Caller‑selected retrieval profile

The request should accept a profile representing retrieval intent rather than forcing every
consumer to inherit one hidden server configuration. Candidate profile names include:

| Profile | Intent | Typical pipeline                                                                 |
|---|---|----------------------------------------------------------------------------------|
| `lexical` | Keyword‑sensitive retrieval | Lexical retrieval such as BM25; no fusion or reranker unless explicitly declared |
| `semantic` | Meaning‑oriented retrieval | kNN or another vector / semantic retriever                                       |
| `hybrid` | Lexical plus semantic coverage | Multiple retrieval arms followed by declared score or rank fusion                |
| `diverse` | Reduce redundancy in the final evidence set | Candidate retrieval followed by a diversity reranker such as MMR                 |
| `filter_only` | Return chunks or context satisfying hard constraints without relevance ranking | Filters plus a deterministic non‑relevance order                                 |
| `auto` | Let the service select an advertised profile | Server selection, with the selected pipeline fully reported                      |

This table is a discussion proposal contingent on the retrieval backend, MCP tool scope, and
capabilities selected by the owning teams. Profiles may be renamed, limited, or extended. They
are capability‑level concepts rather than a required mapping to one Elasticsearch API. An
advanced request may optionally name a physical strategy or parameters when capability discovery
allows it.

`auto` must be explicit or be the documented default when the profile is omitted. It must not
make the response opaque. The applied profile and pipeline are reported exactly as for a
caller‑selected profile.

The service should reject an unavailable profile with a clear capability error. It should not
silently substitute a semantically different algorithm. If product policy permits substitution,
the contract must define it and return both requested and applied profiles.

### 7.4 Caller‑controlled chunk budget

The current tool has no result‑count argument. Its ranked chunk cap comes from
`qna-service.retrieval.strategy.configuration`. This couples every matching caller to one
server‑selected budget despite different latency, context‑window, and evidence‑coverage needs.

The proposed request should add a field such as `maxChunks`:

- it caps ranked chunks in the canonical `chunks[]` response;
- its allowed range and default are advertised through capability discovery;
- the service either validates or clamps it according to one documented rule;
- LaunchDarkly may provide the default when it is omitted, but should not override an accepted
  caller value silently;
- the response reports requested limit, applied limit, actual returned ranked‑chunk count, and
  whether more results are known to be available.

Candidate‑pool size, fusion windows, and reranker depth may remain service‑managed defaults or
be exposed as advanced options. They must appear in applied‑pipeline metadata when they affect
score interpretation or reproducibility.

Unlike the current grouped payload, the proposed flat list has no separately attached
`firstChunk` that can make the effective context exceed `maxChunks`. Any automatically attached
document‑opening chunk (`firstChunk`) or adjacent context chunks must be identified separately and
must not be counted or ranked as retrieval hits.

### 7.5 Continuation and result availability

Receiving exactly `maxChunks` results does not prove that another result exists. Receiving fewer
usually indicates exhaustion, but a timeout, candidate cap, permission filtering, approximate
retrieval, or partial backend behavior can make availability unknown. Score values also do not
prove availability: a last score of `98` after a first score of `100` says nothing by itself about
whether rank `k + 1` exists.

Capability discovery should state whether each profile supports stable top‑down continuation.
When it does, the response should return:

- an opaque continuation token;
- `moreResults` with `true`, `false`, or `unknown` semantics;
- the first and last rank covered by the current page; and
- an explicit exhaustion reason when the result is known to be complete.

The token should bind the query, filters and trusted scope, profile, applied pipeline,
`vectorSpaces[]`, and index/chunk generation. Each page is a contiguous slice of one logical
ranking: with `maxChunks = 20`, the first page holds ranks 1 to 20 and the next page ranks 21 to 40,
without returning earlier chunks again.

The service must not silently use a token against a different corpus or chunk generation or treat
it as a new logical result set. If the bound generation is no longer available or valid, the
service should return a documented stale or invalid continuation condition. The caller can then
issue a new initial request and receive a new logical ranking.

Continuation cannot have identical semantics for every algorithm:

- lexical ranked retrieval can commonly continue using a stable snapshot and search cursor;
- approximate kNN may not define stable deep pagination over the full corpus;
- compound RRF may not support the backend's normal search‑after mechanism; and
- MMR is stateful because each later selection depends on previously selected chunks.

A backend may support those profiles by retaining state or rerunning a larger candidate set, or it
may advertise that continuation is unsupported. If continuation is unavailable, the caller may
request a larger finite `maxChunks`; the response should warn when rerunning fusion or reranking
can change the previously returned prefix
([Elasticsearch pagination](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/paginate-search-results),
[retriever API constraints](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers)).

The contract supports only best‑first retrieval and top‑down continuation. Retrieving the least
relevant chunks or paginating from the bottom is neither required nor advertised.

### 7.6 Applied retrieval metadata

Every response should contain `appliedRetrieval` describing what actually produced the returned
order. At minimum, it should report:

- requested and applied profile;
- candidate retriever or retrievers, such as lexical or kNN;
- fusion type, such as no fusion, additive score combination, RRF, or Elasticsearch `linear`;
- reranker type, such as no reranker or MMR;
- parameters needed to interpret or reproduce the result, including candidate counts, fusion
  windows, MMR lambda, MMR depth, and applied final chunk limit;
- the applied `vectorSpaces[]` when dense or sparse vectors were involved;
- whether ordering is relevance‑based or a non‑relevance sort;
- the sort actually applied;
- score semantics, including score type, direction, range when bounded, and whether scores are
  comparable only within this result set.

Recognizable engine‑specific values are useful diagnostics. For example, reporting Elasticsearch
`linear` distinguishes it from RRF. They do not make Elasticsearch a contract requirement.

These identifiers let operators establish, for every arm, that query and indexed vectors belong to
the same vector space. Optional diagnostics may expose model/version, dimensions,
normalization, representation type, and query/document encoder roles when operationally useful.
They should not expose secrets or backend connection details.

### 7.7 Canonical flat chunk response

The canonical response should be a flat `chunks[]` collection. Each ranked chunk should carry:

- parent document ID;
- mandatory stable chunk ID;
- chunk content;
- one‑based final relevance rank;
- optional final score;
- optional score type and score direction;
- optional component scores / ranks for fusion diagnostics;
- optional reranker diagnostics such as query similarity or MMR score at selection;
- optional human‑readable display/citation identifier, such as the currently available
  `controlNumber`;
- optional chunk‑generation and index‑generation lineage.

Parent document and stable chunk identity are required for deduplication, citations, caching,
continuation, and IR evaluation. `chunkId` is stable within one chunk/index generation; it is not
guaranteed across re‑chunking or re‑indexing, where generation lineage relates old and new chunks. Display identifiers improve human‑readable citations and UI but
do not affect relevance, so they are optional. The current qna chunk model already carries
`controlNumber`
([current chunk model](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/DocumentChunk.cs)).
Generation lineage is a separate optional field
that prevents consumers from mixing ranks, caches, continuations, or relevance judgments across
incompatible re‑chunking or re‑indexing events.

The list is best‑first. Rank `1` is the first and best selected chunk. Rank is the one‑based
position in the logical result set represented by the query and its continuation sequence, after
fusion and reranking; each response page carries a contiguous slice of that ranking. A numeric
score is ordered according to its declared direction.

Determinism is an advertised profile capability, not a consequence of deterministic tie‑breaking
alone. A profile may claim deterministic ranked output only when its entire pipeline produces a
deterministic result under the advertised conditions. For such a profile, the guarantee is end to
end within one chunk/index generation and covers candidate retrieval, fusion, reranking, and final
ordering. When final scores are equal, the profile applies its advertised tie‑break, by default
`documentId ASC, chunkId ASC`. Otherwise capability discovery must declare the weaker guarantee.

For an MMR profile to advertise deterministic output, candidate ordering must itself be
deterministic. The current selector prefers the lowest candidate index when MMR values are equal
within a small tolerance
([MMR selector](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/MaximumMarginalRelevanceSelector.cs#L76-L83)).
That resolves a selector‑level tie only for a fixed candidate list; the profile must also derive or
normalize that list in a deterministic order to claim the end‑to‑end guarantee. Deterministic
ranked output keeps pagination, caching, evaluation, and tests reproducible. The contract does
not pretend that BM25, cosine‑derived, RRF, additive hybrid, and MMR values share a common scale.

Final rank should be present for every relevance‑ranked response even when a meaningful numeric
score is unavailable. This is particularly important for MMR, whose selection score changes as
the selected set grows. A final ordinal rank remains well‑defined, while one scalar score may not
fully describe the process.

Component scores are optional diagnostics, not the canonical cross‑profile relevance measure.
They must not be compared across calls unless their declared semantics explicitly permit it.

### 7.8 Filtering and non‑relevance ordering

Query filters and trusted scope narrow the eligible set before retrieval. They do not themselves
create relevance. A relevance‑ranked profile can apply filters and still return final rank.

Filter‑only retrieval remains an optional proposed capability even though the planned MCP V4
`GetRelevantDocuments` contract requires a query. When `filter_only` is supported:

- relevance rank and score are absent;
- all matching results are treated as equally relevant for this contract;
- the response declares that ordering is not relevance‑based;
- the service uses a stable tie‑break, for example `documentId ASC, chunkId ASC`;
- `appliedRetrieval` reports the deterministic sort.

Future chunk retrieval may optionally accept an explicit non‑relevance sort. If so, the response
reports the applied sort and does not label result position as relevance rank. Neither current
qna MCP V2 / V3 nor planned MCP V4 `GetRelevantDocuments` is claimed to support this parameter.

### 7.9 Grouping and attached context

**Parent grouping should be a consumer‑side transformation over canonical flat chunks**. Consumers
can group by parent document ID and sort by chunk ID when they want document‑order reading
context, without losing the original `rank` field.

The canonical response should not give chunk ID `0` special relevance semantics. If the
document‑opening chunk is retrieved and ranked normally, it appears in `chunks[]` like any other
result. Treat an automatically attached document‑opening chunk (`firstChunk`) or adjacent context
chunks as unranked context, not ranked retrieval results.

Current qna MCP V2 / V3 already carries `firstChunk` in a separate field and removes a retrieved
chunk 0 from `retrievedChunks` when adapting to V3
([V3 grouped adapter](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/GroupedDocumentChunksV3.cs#L20-L49)).
The proposed contract makes the relevance meaning explicit: a separately attached value is
unranked context, while a genuinely retrieved chunk 0 keeps its retrieval rank.

## 8. Follow‑up implementation options

**Status: follow‑up discussion. Agreement on the retrieval contract should precede implementation
and rollout decisions.**

Several implementation paths can satisfy the proposal:

1. **New versioned tools:** introduce a relevance‑aware retrieval tool set while leaving current
   qna MCP V2 / V3 and the separately planned MCP V4 direction untouched. This is the preferred
   low‑risk option because existing consumers keep their current contract.
2. **Future‑version extension:** add the flat response, applied‑pipeline metadata, and request
   controls to a later MCP version after the planned V4 scope.
3. **Compatibility extension:** retain grouped output but add explicit rank / score to every chunk
   or return a parallel ranked chunk identity list. This is less direct for new consumers but can
   preserve an existing wire shape.

The contract is intended for new consumers or updated existing consumers. This exploratory
write‑up does not select the owning service or mandate an in‑place migration.

### 8.1 Validation and rollout

Every implementation path should include contract and integration tests that exercise capability
discovery, requested‑versus‑applied profile and limits, rank/score semantics, continuation,
deterministic filter‑only ordering, and vector‑space compatibility. These tests validate the
normative rules in section 7 rather than restating that contract.

LaunchDarkly can gate rollout, profile availability, defaults, and experiments. It should not
silently change the semantics of an accepted request; any documented default or policy override
must be reflected in `appliedRetrieval`.

### 8.2 Retrieval quality and cost levers

Several index‑side levers could improve retrieval quality or cost once callers can select and
observe retrieval behavior. They are options for the owning teams, not prerequisites of the
contract:

1. **Language analyzers.** The current mapping declares `body` and `title` as text fields
   without an explicit analyzer, so they use Elasticsearch's standard analyzer
   ([index mapping](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Infrastructure/Elasticsearch/ElasticsearchClientWrapper.cs#L105-L145)).
   The `english` analyzer adds stop‑word removal and stemming at both index and query time. That
   may improve lexical recall and reduce index size, at some cost to exact phrase and name
   precision. Changing it requires re‑indexing, and a multilingual corpus needs per‑language
   analyzers or fields
   ([language analyzers](https://www.elastic.co/docs/reference/text-analysis/analysis-lang-analyzer)).
2. **Dense and sparse vectors.** Populated dense or learned‑sparse vectors make `semantic` and
   `hybrid` profiles meaningful. The trade‑off is index storage and indexing work versus
   query‑time inference, as discussed in the
   [Slack thread](https://kcura-pd.slack.com/archives/C05PKQYTZTM/p1788349409928209) and in
   section 10.
3. **Engine‑side inference endpoints.** Encoding both documents and queries through one engine
   inference endpoint, backed by an Elastic model or a PyTorch model imported with Eland, gives
   one model owner. That directly addresses the cross‑service model‑resolution risk in section 3.7
   and simplifies `vectorSpaces[]`
   ([inference API](https://www.elastic.co/docs/explore-analyze/elastic-inference/inference-api),
   [importing models with Eland](https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-import-model)).
   It requires machine‑learning node or managed inference capacity: an earlier cluster
   investigation found the ELSER endpoint exposed but not deployable, because the cluster had no
   ML nodes and the Elastic Inference Service was not connected
   ([client‑side sparse vectors](https://github.com/mierzejk-rel/es-index-explorer/blob/76ed2dc57faa733670c2d4e42461db80b63f2e9a/reports/07-client-side-sparse-vectors.md#L19-L22)).
4. **`semantic_text` and equivalents.** Field types that chunk and encode inside the engine
   remove client‑side embedding code, but they also move chunk boundaries into the engine. That
   affects `chunkId` stability and the generation lineage described in sections 1.4 and 7.7
   ([`semantic_text`](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text)).

Each lever changes which profiles, score types, and vector spaces capability discovery can
advertise. None of them changes the proposed contract.

### 8.3 Agent v4 parser cleanup

Current qna MCP V2 / V3 producers return grouped output, and the planned MCP V4 contract also
specifies grouped output. **Agent v4's flat‑payload fallback therefore appears to be compatibility
debt** rather than a requirement of a verified producer
([agent v4 compatibility parser](https://github.com/relativityone/air-assist-agent/blob/87e422bf05d00c7a0926c256538b1941d19ef515/packages/air_assist_core/src/air_assist_core/registry/graphs/v4/chunk_formatter.py),
[qna V3 typed result](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/GroupedDocumentSearchResultV3.cs)).

As a separate aiR Assist cleanup, remove that fallback only after:

1. one cross‑service contract test covers every supported producer and MCP version; and
2. owners confirm that no supported older or alternate server emits a flat chunk list.

This cleanup does not block the new retrieval contract.

## 9. Recommended decisions and open questions

**Status: author's recommendations requiring owner review and agreement.**

### 9.1 Recommended decisions

- Scope the contract to chunk‑level relevant‑document retrieval, not document search.
- Make a flat `chunks[]` list the canonical response.
- Return relevance‑ranked results best‑first with explicit one‑based final rank.
- Treat every numeric score as typed, directional, and algorithm‑specific.
- Let the caller choose an advertised retrieval profile and ranked chunk budget.
- Return requested and applied profile, limits, ordering, and pipeline metadata.
- Keep filters and trusted scope independent from algorithm selection.
- Keep filter‑only results unranked but deterministically ordered.
- Make grouping a consumer‑side helper.
- Treat an automatically attached document‑opening chunk (`firstChunk`) or adjacent context chunks
  as unranked context, not ranked retrieval results.
- Use LaunchDarkly for availability, defaults, rollout, and experiments rather than as the sole
  opaque per‑call strategy selector.
- Keep the stable contract independent of Elasticsearch and index layout while allowing
  engine‑specific diagnostics.

### 9.2 Open questions

- Should public profile names expose physical algorithms, or only stable intents such as
  `lexical`, `semantic`, `hybrid`, and `diverse`?
- Does the generalist agent choose profiles directly, or do task‑specific skills constrain the
  choice?
- Is `auto` offered, and if so, how is its selection policy versioned and explained?
- Which component scores and ranks are returned by default versus only in diagnostics?
- Is filter‑only chunk retrieval part of the first relevance‑aware version or only an advertised
  optional capability?
- What are the valid `maxChunks` range and validation / clamping semantics?
- Which profiles support continuation, and what state or rerun semantics back their opaque tokens?
- What stable format and ownership rules define `vectorSpaceId` and its `vectorSpaces[]` entries?
- Do owners confirm the proposed semantics that rank is global across continuation pages, ranked
  ties are deterministic by an advertised rule, and `chunkId` is stable only within one
  chunk/index generation?
- Does the new contract coexist indefinitely with legacy tools, or eventually replace them?
- Which service owns capability discovery if planned MCP V4 is not the implementation vehicle?
- How is vector‑space compatibility represented and enforced across indexing and query services?
- In the planned MCP V4 wording, does "first‑appearance BM25 relevance rank" refer to pre‑MMR
  candidate order or post‑MMR selection order?

## 10. Appendix: MMR computation and storage trade‑off

**Status: current implementation finding plus a reasoned performance consequence.**

The current BM25+MMR path retrieves chunk text, embeds the query, batch‑embeds all non‑empty
candidate bodies, and runs MMR in `qna-service`. It does not read stored chunk vectors for MMR and
does not write the calculated vectors back to Elasticsearch
([retrieval service](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/RetrievalService.cs#L82-L163)).

Each tool request performs this work independently. There is no cache, single‑flight mechanism,
or chunk‑identity / content deduplication across requests. When an agent issues concurrent
retrieval calls, **overlapping candidates are embedded independently for each call**. A later hop
that retrieves the same chunk embeds it again. Within one request, `qna-service` only divides the
candidate list into batches and bounds batch concurrency; it does not reuse vectors
([embedding service client wrapper](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/Embedding/EmbeddingService.cs)).

Discarding the vectors is a classic storage‑versus‑computation trade‑off. It avoids persistent
vector storage for MMR but **repeats model inference and is expected to add response latency when
the same chunks recur**. This report does not claim a measured latency delta.

The index mapping can store dense vectors, and normal indexing can populate them when the
text‑search‑only switch is disabled
([index mapping](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Infrastructure/Elasticsearch/ElasticsearchClientWrapper.cs#L105-L145),
[indexing pipeline](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Application/Temporalio/Activities/ConsumerActivities.cs#L257-L266)).
That does not make current MMR reuse those vectors.

Elasticsearch‑side MMR may be worth investigating in a separate performance design, but it is
not an established compatibility claim or recommendation in this report
([ESQL MMR command](https://www.elastic.co/docs/reference/query-languages/esql/commands/mmr)).

## 11. References and source anchors

The following sources are grouped by role. Inline links near claims remain the primary evidence.

### 11.1 Current source code

- [`BaseSearchStrategy.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/Base/BaseSearchStrategy.cs):
  Elasticsearch response mapping and score/rank loss.
- [`RetrievalService.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/RetrievalService.cs):
  strategy dispatch and BM25+MMR pipeline.
- [`MaximumMarginalRelevanceSelector.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/MaximumMarginalRelevanceSelector.cs):
  MMR selection and iteration‑specific score calculation.
- [`DocumentProviderV2.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Services/DocumentProviderV2.cs):
  permission filtering, capping, metadata paths, grouping, and chunk‑ID sorting.
- [`RrfSearchStrategy.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Strategies/RrfSearchStrategy.cs):
  additive hybrid and true RRF query branches.
- [`ElasticQueryFilterBuilder.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/ElasticSearch/Filters/ElasticQueryFilterBuilder.cs):
  subset, focus, and metadata eligibility filters.
- [`RetrievalStrategyType.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyType.cs)
  and
  [`RetrievalStrategyDefaults.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/RetrievalStrategyDefaults.cs):
  selectable strategies and server‑side limits.
- [`LaunchDarklyClientWrapper.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/LaunchDarkly/LaunchDarklyClientWrapper.cs),
  [`LaunchDarklyContextBuilder.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/LaunchDarkly/LaunchDarklyContextBuilder.cs),
  and
  [`LaunchDarklyOpenAiRequestsKeys.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Infrastructure/Constants/LaunchDarklyOpenAiRequestsKeys.cs):
  retrieval/model flag evaluation and targeting context.
- [`ConsumerActivities.cs`](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Application/Temporalio/Activities/ConsumerActivities.cs)
  and
  [`LaunchDarklyClientWrapperSingleton.cs`](https://github.com/relativityone/embedding-service/blob/1e31013745d51969836163caf1156197a27e5ae5/Source/Relativity.Embedding.Infrastructure/LaunchDarkly/LaunchDarklyClientWrapperSingleton.cs):
  index‑time embedding switch and model resolution.
- [`agent v3 rag_agent.py`](https://github.com/relativityone/air-assist-agent/blob/87e422bf05d00c7a0926c256538b1941d19ef515/packages/air_assist_core/src/air_assist_core/registry/graphs/v3/rag_agent.py)
  and
  [`agent v4 chunk_formatter.py`](https://github.com/relativityone/air-assist-agent/blob/87e422bf05d00c7a0926c256538b1941d19ef515/packages/air_assist_core/src/air_assist_core/registry/graphs/v4/chunk_formatter.py):
  grouped response consumption and compatibility parsing.
- [`GroupedDocumentChunksV3.cs`](https://github.com/relativityone/qna-service/blob/15ee362c9e4470db7a774a431f5a42e37e14d577/Source/Relativity.QnA.Application/Models/ElasticSearch/GroupedDocumentChunksV3.cs):
  current structured `firstChunk` adaptation.

### 11.2 Internal contracts, architecture, and research

- [MCP V4 engineering plan](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1792245775/Engineering+Plan+MCP+V4+Clair+Search+Q+A+AA+V2):
  documented future architecture, non‑functional requirements, BM25+MMR scope, and kNN deferral.
- [MCP V4 working contract](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1780711469/ClaiR+MCP+Tools+AA+V2+Working+Contract):
  planned tool shapes, grouping, absence of scores, filters, scope, and sort behavior.
- [ClaiR Search V3 mental model](https://relativity-oda.atlassian.net/wiki/spaces/~712020f52569fec67e4b74bfc314f9d70e4636/pages/1774420235/ClaiR+Search+V3+Tool+Design+Mental+Model):
  document search versus chunk Q&A responsibilities.
- [ClaiR Search and Q&A flow](https://relativity-oda.atlassian.net/wiki/spaces/~712020f52569fec67e4b74bfc314f9d70e4636/pages/1782677688/Clair+Search+Q+A+Flow):
  direct and indirect consumers, artifacts, focus, and citations.
- [Tool, skill, agent, and surface decision framework](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1909031054/Decision+Tree+for+Building+Tools+Skills+Agents+or+New+Surface+Areas+in+Relativity):
  shared tool and task‑specific skill direction.
- [MMR configuration study](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1736540182/2026-07-14+Retrieval+Configuration+Study+MMR+Delta+Document+Count):
  dataset‑dependent MMR effects and result‑count/token trade‑offs.
- [Simple Mode](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1698595003/Simple+Fast+mode+-+Intro)
  and [Deep Research](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1640797199/Deep+Research+-+Intro):
  conflicting latency, breadth, and context objectives.
- [Simple Mode experiment design](https://github.com/mierzejk-rel/es-index-explorer/blob/81905343d0dd3e6bf0708733ed9ba99c4073e696/reports/10-simple-mode-experiment-design.md):
  relevance‑dependent round‑robin research design.
- [Current retrieval analysis](https://github.com/mierzejk-rel/es-index-explorer/blob/3376afc32ab19218e2526571a8f5b094c6ac6b65/reports/03-retrieval-strategies.md):
  secondary consolidated analysis of `qna-service` retrieval paths.
- [Client‑side sparse vectors](https://github.com/mierzejk-rel/es-index-explorer/blob/76ed2dc57faa733670c2d4e42461db80b63f2e9a/reports/07-client-side-sparse-vectors.md):
  ELSER endpoint unavailability and missing ML‑node capacity on the experiment cluster.
- [PR #623 relevance‑order review](https://github.com/relativityone/air-assist-agent/pull/623#issuecomment-5698164026):
  why grouped results without global rank could not support chunk‑level round‑robin.
- [Interactive Memo Drafting integration](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1002176513/Interactive+Memo+Drafting+Agent+-+aiR+Assist+Integration+Work),
  [a4CS MCP](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1742962810/aiR+for+Case+Strategy+MCP),
  [ClaiR memo canvas](https://relativity-oda.atlassian.net/wiki/spaces/PM/pages/1876361382/ClaiR+Drafting+Memo+Canvas),
  and [memo viewer contract](https://relativity-oda.atlassian.net/wiki/spaces/DV/pages/1932198083/a4CS+Memo+Viewer+Contract+for+ClaiR+ADR):
  memo workflow and downstream product boundaries.

### 11.3 Slack discussion (September 2026)

- [Retrieval flags, embedding storage, and model research Slack thread](https://kcura-pd.slack.com/archives/C05PKQYTZTM/p1788349409928209):
  production flag snapshot, embedding/storage trade‑off, and corpus‑dependent model findings.

### 11.4 Elasticsearch documentation

- [Search API and `_score`](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/search-your-data.html)
- [Full‑text relevance and BM25](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/full-text-search.html)
- [Sort and `track_scores`](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/sort-search-results.html)
- [kNN scoring](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/knn-search.html)
- [RRF](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/rrf.html)
- [Linear retriever and normalizers](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever)
- [Rank evaluation, DCG, and NDCG](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/search-rank-eval)
- [ESQL MMR](https://www.elastic.co/docs/reference/query-languages/esql/commands/mmr)
- [Pagination](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/paginate-search-results)
- [Retriever API constraints](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers)
- [Language analyzers](https://www.elastic.co/docs/reference/text-analysis/analysis-lang-analyzer)
- [Inference API](https://www.elastic.co/docs/explore-analyze/elastic-inference/inference-api)
  and [importing models with Eland](https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-import-model)
- [`semantic_text`](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text)


