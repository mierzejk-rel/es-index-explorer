# Segment 4 outcome-blind annotation operations

This guide governs the operational execution of Segment 4. It supplements, but does not
change, the frozen research plan and linguistic codebook recorded in the current analysis
lock.

## One-time Cursor setup

The local Cursor Python SDK requires a Cursor user API key. Create one in Cursor Dashboard →
Integrations, then keep it outside this repository. The runner reads only `CURSOR_API_KEY`;
it never writes the key to an artifact, manifest, response, report, or log.

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
model/batch manifest, normalized annotations, agreement records, and partial report 03 under
the analysis root. Existing responses are hash-checked on resume; a conflicting response is a
failure, not an overwrite. Run and agent IDs, timestamps, status, and token usage are
provenance metadata, not outcome data.

## Closed P4 QDMR inventory

`qdmr_operator_set` is a sorted, duplicate-free subset of exactly these BREAK/QDMR operators:

`SELECT`, `FILTER`, `PROJECT`, `AGGREGATE`, `GROUP`, `SUPERLATIVE`, `COMPARATIVE`, `UNION`,
`INTERSECTION`, `DISCARD`, `SORT`, `BOOLEAN`, `ARITHMETIC`.

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
