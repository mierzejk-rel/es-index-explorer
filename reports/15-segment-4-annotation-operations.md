# Segment 4 outcome-blind annotation operations

This guide governs the operational execution of Segment 4. It supplements, but does not
change, the frozen research plan and linguistic codebook recorded in the current analysis
lock.

## One-time Cursor setup

New annotation model calls through the local Cursor Python SDK require a Cursor user API key.
Create one in Cursor Dashboard → Integrations, then keep it outside this repository. The
runner reads only `CURSOR_API_KEY`; it never writes the key to an artifact, manifest, response,
report, or log. A `--resume-only` migration from a verified preflight and complete raw/attempt
pairs neither reads an API key nor initializes an SDK runner.

Set it in your shell profile or for one command:

```bash
export CURSOR_API_KEY="cursor_..."
uv run question-suitability annotate-run
```

The first live run preflights the account-visible models. It stops unless it can resolve both
Claude Opus 5 with high reasoning effort and GPT-5.6 Sol exactly. The non-secret resolved IDs
and selected parameters are frozen in `annotation_preflight_manifest.json` before any call and
copied into the completed `annotation_manifest.json`; no substitute model is used.

## Execution sequence

Run the commands only after the current Segment 3 root has passed:

```bash
uv run question-suitability annotate-emit
uv run question-suitability annotate-run
uv run question-suitability annotate-ingest
```

The canonical post-remediation migration first regenerates `join`, `features`, and
`annotate-emit` in an empty root, then imports only verified annotation source provenance:

```bash
uv run question-suitability annotate-run \
  --migrate-from-root artifacts/question_analysis/simplemode-v1-postreview-p3 \
  --resume-only
uv run question-suitability annotate-ingest
```

`--migrate-from-root` requires `--resume-only`. The migration verifies the historical
preflight against the current prompt, response schemas, SDK version, and emitted batch hashes;
requires exactly 50 raw/attempt pairs; validates each pair before copying it; and does not copy
historical envelopes or ingest outputs. Missing, additional, or conflicting pair files fail
closed instead of invoking an annotation model.

`annotate-emit` deterministically shuffles 577 items using the locked `annotation_shuffle`
stream and produces 25 batches: 24 batches of 24 plus one batch of 1. `annotate-run` invokes
one local one-shot Cursor agent per model and batch, with at most four independent batches in
flight. It retries only SDK failures marked retryable, up to three attempts. Cursor SDK 1.0.28
rejects idempotency keys for local agents, so completed responses are made resumable by
write-once files and hash verification rather than by a server-side idempotency header. Each
valid result is first committed as one atomic response-and-provenance envelope; raw JSONL and
attempt metadata can be reconstructed from that envelope after interruption. A retry after an
indeterminate transport failure can therefore incur a duplicate SDK run; every returned run ID
that reaches the runner is retained in attempt provenance.

The first ten accepted Claude responses in the current Segment 4 execution completed before
atomic envelopes and four-worker execution were introduced. They remain immutable,
hash-verified raw/attempt pairs. The other forty accepted responses also have envelopes. This
historical difference affects interruption recovery only; it does not change prompts, schemas,
model settings, source batches, or normalized annotations.

The historical root remains unchanged. When these pairs are copied into the final remediated
root, resume verifies model and batch identity, the emitted batch hash, response hash, finished
status, strict JSONL schema, and batch-local IDs before deterministically writing the missing
envelope. A valid legacy pair never triggers a model call; an incomplete or conflicting pair
fails closed.

The provenance tree also retains three superseded Claude responses and one rejected Claude
response from the initial long-hash-ID protocol. They are excluded from normalization and
agreement; their retention makes the protocol correction explicit rather than overwriting prior
SDK output.
`annotate-ingest` refuses malformed responses and does not perform prose, Markdown-fence, or
outcome-field recovery.

## Isolation and records

Each Cursor run receives only its current batch, the strict schemas, and annotation
instructions from a fresh temporary working directory. It receives no repository directory,
MCP server, ambient Cursor setting source, MLflow snapshot, outcome table, grade, quality
score, or recommendation artifact.

Model-facing item identifiers are deterministic batch-local keys (`item-001` through
`item-024`). The runner maps them back to the immutable source IDs locally after strict
validation. This avoids transcription errors in long hash-based IDs without exposing any
additional data or recovering malformed annotations by line order.

The workflow stores emitted batches, write-once raw JSONL responses, attempt metadata, a
model/batch manifest, normalized annotations, agreement records, attributable construct-QA
anomalies, and partial report 03 under the analysis root. Existing responses are hash-checked
on resume; a conflicting response is a failure, not an overwrite. Run and agent IDs,
timestamps, status, and token usage are provenance metadata, not outcome data.

## Closed P4 QDMR inventory

This section restates the authoritative project normalization vocabulary frozen in codebook
§6.2. `qdmr_operator_set` is a sorted, duplicate-free subset of exactly these labels:

`SELECT`, `FILTER`, `PROJECT`, `AGGREGATE`, `GROUP`, `SUPERLATIVE`, `COMPARATIVE`, `UNION`,
`INTERSECTION`, `DISCARD`, `SORT`, `BOOLEAN`, `ARITHMETIC`.

The vocabulary is project-specific. It does not assert that Wolfson et al. or every upstream
BREAK implementation emits these exact strings. Because one QDMR step has one operator,
`qdmr_step_count` must be at least the number of unique operators. A lower count is retained
as an immutable raw response and persisted as a nonblocking construct-QA anomaly for later
human validity review; it is never silently repaired.

## Current execution record

The manual API-key checkpoint and live model preflight completed on 2026-09-07. The key remained
external to the repository and is not present in an analysis artifact. Preflight resolved:

- `claude-opus-5`: thinking enabled, high effort, 1M context, fast disabled;
- `gpt-5.6-sol`: medium reasoning, 1M context, fast disabled.

All 25 batches completed for each model. Ingestion validated 1,154 annotations for 577 unique
items and produced `annotations_normalized.parquet`, `annotation_agreement.parquet`,
`annotation_ingest_verification.json`, and partial report 03. The ingestion gate passed with no
outcome columns loaded.

## Outcome blindness

Prompts contain only `item_id`, `item_type`, exact source text, the codebook-derived schema,
and instructions. Emission and ingestion reject outcome-adjacent keys recursively and
case-insensitively. The partial report states only annotation completeness, labels,
missingness, and inter-model reliability. It never treats agreement as validity.

Partial report 03 contains preliminary exact and scale-aware pairwise diagnostics only.
Krippendorff alpha, its interval, and feature-validation statuses belong to the later
`validate-features` dossier. Exhaustivity is nominal in these preliminary diagnostics;
`cognitive_process_level` is the only ordinal categorical P4 label.

The outcome-field contract is versioned independently of the table artifact schema. Version
1 declares the authoritative `trace_pfu_table` schema and the outcome and recommendation field
names frozen today. Annotation responses use case-insensitive **exact field-name matching**
against every trace field and every registered recommendation field. Pre-annotation input
validation uses the same contract but permits shared structural identifiers such as
`rubric_id` and `variant_id`; these identifiers are projected out before batch emission.
Future `recommendation_table_rubric` and `recommendation_table_variant` artifacts cannot be
written until their exact schemas, including future score-band probability fields, are
registered with the same contract.

## P1 remediation verification

The post-audit P1 remediation added the frozen §18.6 test-6 response fixture and executable
coverage for every currently declared outcome-response field, recursive and case-insensitive
refusal, harmless exact-name near misses, and protected artifact schemas. It also added
applicable-QDMR parser coverage for every operator in the 13-element inventory, all four hop
labels, normalization, missingness, malformed JSONL, local-ID mapping, agreement semantics,
and persisted round trips.

All 50 immutable raw responses from the completed run were re-parsed with the remediated exact
validator without invoking either annotation model. The result remained 1,154 normalized rows
for 577 unique items, contained no denied outcome column, and was semantically identical to the
frozen normalized table. The complete 354-test suite passed after remediation; branch coverage
over `es_index_explorer.question_analysis` was 87%. This verification did not rewrite the
historical analysis root.
Because the analysis lock fingerprints implementation files, continuation after remediation
must use a root initialized under the remediated code and migrate the hash-verified immutable
responses; the historical manifest must not be edited in place.

## P2 remediation verification

The P2 remediation made the codebook the authoritative source for the closed operator
vocabulary, withdrew the undefined exploratory `entity_density` name without introducing a
replacement metric, and made QDMR applicability and construct anomalies explicit in Stage 4
verification.

The historical 50 raw responses contain 383 `applicable`, 143
`applicable_after_normalisation`, and zero `not_applicable` question annotations. The zero is
schema-valid observed model behavior, not a parser failure or a claim of construct validity;
faithfulness remains for the frozen human gold/validation process. Claude produced
191 / 72 / 0 and GPT produced 192 / 71 / 0 for applicable / after-normalisation /
not-applicable respectively. Six annotations have
`qdmr_step_count < len(qdmr_operator_set)`. Their IDs, model/batch provenance, source hashes,
raw-response hashes, and observed counts are emitted by remediated ingestion as
`annotation_construct_anomalies.parquet`, without source text or outcome fields. These
anomalies do not change §13.3 sampling probabilities or replace sampled human gold.

Revalidation uses the immutable raw responses without model calls and does not overwrite the
historical root. All 1,154 semantic annotation rows remain identical, no unexpected operator
or outcome field appears, the model-facing prompt hash is unchanged, and the only derived
agreement change is the correction of exhaustivity from ordinal distance to nominal exact
similarity. The complete 361-test suite passed after P2 remediation with 88% branch coverage
over `es_index_explorer.question_analysis`. The final remediated root is intentionally
deferred until all P2/P3 changes are complete so implementation and codebook fingerprints are
migrated once.

## P3 remediation verification

The incomplete duplicate `QUESTION_LABELS` and `EXPECTATION_LABELS` registries were removed;
the Pydantic response schemas and codebook remain authoritative. Normalized
`qdmr_step_count` now uses pandas nullable `Int64` and Arrow nullable `int64`, preserving null
for expectations and inapplicable questions without turning integer counts into floats.

Verification records every member of the closed QDMR-operator and cognitive-process
inventories, including zeroes. In the immutable live responses `ARITHMETIC=0`, `apply=0`, and
`create=1`. These unused inventory levels are corpus coverage, not invalid labels; no label is
forced and the closed vocabularies do not change. Segment 3 and Stage 4 now both compute
`outcome_columns_loaded` from their actual output schemas and fail when a denied field is
present rather than asserting an empty list.

The ten historical Claude envelopes are reconstructible byte-for-byte from verified raw and
attempt pairs. They were reconstructed in memory only during P3 verification; the historical
40-envelope root was not modified. The final migrated root will materialize all 50 envelopes
without invoking an annotation model and will regenerate nullable-integer and nominal
agreement artifacts from the same immutable responses. All 1,154 semantic rows and the
model-facing prompt hash remained unchanged. At that checkpoint, the complete `tests/` tree
contained 377 tests and produced 88% branch coverage when run with
`--cov=es_index_explorer.question_analysis --cov-branch`; that historical count is superseded
by the final closure gate below.

## Final post-remediation closure

The canonical refreshed root is
`artifacts/question_analysis/simplemode-v1-postremediation-final`. It was initialized from
empty storage under the completed remediation code. `join`, `features`, and `annotate-emit`
were rerun rather than copied. No `simplemode-v1-postreview-p4` root was created because no
distinct independently frozen checkpoint existed between `postreview-p3` and this final root.

`annotate-run --migrate-from-root .../simplemode-v1-postreview-p3 --resume-only` completed
with `CURSOR_API_KEY` absent. Migration verified and copied exactly 50 raw JSONL responses,
50 canonical attempt records, and 10 rejected/superseded provenance files. It copied no
historical envelope or ingest output. The final root then reconstructed all 50 envelopes;
the 40 with historical counterparts are byte-identical. All 50 run records and every raw and
attempt file are byte-identical to the historical source, so no new model run or run ID was
introduced.

The migration evidence is persisted as `annotation_migration_verification.json`. Its source
file-set SHA-256 is
`97400bb1eca56e9b7d957748e44bd74cd59e88fa8d4dc8213b4b45d1d6a55584`;
the historical annotation-manifest SHA-256 is
`b4605f0380bc353e97fb1c0b8774ab5e44ad4c5904c669e5b2b71c1eb0239e33`.
The final root manifest SHA-256 is
`ccbe9fc9381868a2b94bb7f8c44a3925d805e5b32e1dd603369c05a1a1fd7f5d`,
and its completed Stage 4 state SHA-256 is
`37620bd914deb561dc6512c683de80f044e82e4844be4695a3effd936d327da2`.

Final `annotate-ingest` passed with:

- 1,154 normalized rows for 577 items, exactly two distinct expected models per item and no
  duplicate `(item_id, model_id)` row;
- semantic identity to the historical raw-derived annotations and the unchanged prompt hash
  `ec9836788ba3198272d0742509e51efd61bec9f177ff0faf37b1bb22a7378bca`;
- six attributable, nonblocking `QDMR_OPERATOR_COUNT_EXCEEDS_STEP_COUNT` rows in
  `annotation_construct_anomalies.parquet`;
- nominal exhaustivity diagnostics (`mean_similarity = exact_agreement_rate =
  0.7908745247148289`) while cognitive-process disagreement remains ordinal;
- pandas nullable `Int64` and Parquet Arrow `int64` for `qdmr_step_count`;
- schema-version-3 verification with computed `outcome_columns_loaded = []`;
- 383 `applicable`, 143 `applicable_after_normalisation`, and zero `not_applicable`
  annotations, with all 13 operator and six cognitive-process inventory counts retained,
  including zeroes.

The canonical full-suite gate passed **392 tests** with **88.49%** branch coverage over
`es_index_explorer.question_analysis`; Ruff, ty, and formatting checks passed for the changed
Stage 4 files. The historical root's recorded artifacts still match every hash in its
`state.json`. S4-07, S4-09, S4-10, S4-12, S4-13, S4-N01, S4-N02, and S4-N03 are therefore
closed. `gold-sample`, `gold-ingest`, `validate-features`, alpha, intervals, HT weighting,
feature statuses, and P4 release gating remain intentionally pending for Segment 5.
