# es-index-explorer

## Overview
`es-index-explorer` inspects Elasticsearch indices used by aiR Assist and reads
RelativityOne workspace documents so the data can be chunked, embedded, and indexed
into new Elasticsearch indices.

## Setup
- Create a local config: `cp config.example.toml config.toml`
- Fill in secrets in `config.toml`
- Install dependencies: `uv sync`

## Configuration Reference
`config.toml` uses these sections:
- `[cid]` CID v1 token settings (used for Elasticsearch and Relativity CID auth).
- `[token_cache]` CID token cache settings.
- `[retry]` Network retry policy for CID token fetch.
- `[elasticsearch]` Elasticsearch host list and `index_name` (default target index).
- `[relativity]` Relativity tenant/workspace identifiers.
- `[relativity.auth]` Relativity authentication method and credentials.
- `[relativity.fields]` Workspace fields to export — each value is either a field
  **Artifact ID** (integer) or a field **display name** (string). Use IDs when
  multiple fields share the same name.

  ```toml
  extracted_text = 1003668              # Artifact ID
  control_number = "Control Number"   # display name
  ```

- `saved_search_id` (in `[relativity]`) — empty means read all documents. Any integer (including 0) uses that saved search ID.
- `batch_size` (in `[relativity]`) — read page/block size used by both read sources (default 50). Override per run with `--batch-size`.
- `max_text_length` (in `[relativity]`) — `MaxCharactersForLongTextValues` for the QuerySlim source (default 100000).

## Authentication
RelativityOne Object Manager supports three options, configured in
`[relativity.auth]`.

### Basic auth (`method = "basic"`)
Use when you have a Relativity username and password.

1. Base64-encode your credentials:
   `echo -n "username@relativity.com:yourpassword" | base64`
2. Set:
   ```toml
   [relativity.auth]
   method = "basic"
   basic_auth = "<base64 string>"
   ```

### Relativity OAuth2 (`method = "oauth"`)
Use when you have a Relativity OAuth2 client ID and secret for the tenant.

1. Set:
   ```toml
   [relativity.auth]
   method = "oauth"
   oauth_client_id = "<client id>"
   oauth_client_secret = "<client secret>"
   ```
2. The token is requested from
   `https://{relativity.host}/Relativity/Identity/connect/token`
   with `scope=SystemUserInfo`.

### CID v1 tenant session (`method = "cid"`)
Use a CID v1 token scoped to the tenant.

1. Set:
   ```toml
   [relativity.auth]
   method = "cid"
   ```
2. Ensure `[cid]` in `config.toml` has the correct CID client ID/secret for
   Relativity tenant access.
3. Ensure `[relativity].tenant_id` is set so the CID token is scoped correctly.

## Index management
Inspect an index:
```bash
python inspect_index.py 51b1e1f7-f607-4967-a439-cf9babafab2c-1030345-qna-subsetting
```

Set up an index from the declarative definition:
```bash
python setup_index.py as-air-assist-my-workspace-nested --config config.toml
```

Use a different definition file:
```bash
python setup_index.py as-air-assist-my-workspace-nested \
  --config config.toml \
  --index-structure es_index_explorer/index_definitions/air_assist_nested.json
```

Update an existing index in place (additive mappings + dynamic settings only):
```bash
python setup_index.py as-air-assist-my-workspace-nested --config config.toml --update
```

Allow delete-and-recreate for breaking/static changes (destructive):
```bash
python setup_index.py as-air-assist-my-workspace-nested --config config.toml --recreate
```

Skip recreate confirmation prompt:
```bash
python setup_index.py as-air-assist-my-workspace-nested --config config.toml --recreate --yes
```

## Ingest documents

`ingest.py` is the single entrypoint for reading RelativityOne documents and, with
`--index`, indexing them into Elasticsearch. With `--index` each document is chunked
(sentence-first; SaT sentence boundaries with a spaCy clause fallback for over-long
sentences), embedded with `intfloat/multilingual-e5-small` (`passage:` prefix,
L2-normalized), and written as one nested document per RelativityOne document to the
target index. See
[`reports/06-document-indexing-and-semantic-chunking.md`](reports/06-document-indexing-and-semantic-chunking.md).

`--source` selects the Object Manager read mechanism (both behave identically otherwise):
- `queryslim` (default): stateless offset paging; robust for long indexing runs.
- `export`: stateful export run/cursor; efficient for fast bulk and dry-run reads.

### Dry run (read only, the default)

Without `--index`, the tool only reads from RelativityOne to verify connectivity and that
documents are readable. No models are loaded, nothing is chunked or embedded, and no
progress file is written:

```bash
python ingest.py --config config.toml --limit 10
python ingest.py --config config.toml --source export --limit 10
```

### Indexing (resumable)

Prerequisites:
- Install dependencies: `uv sync` (pulls `sentence-transformers`/`torch`, `wtpsplit`, `spacy`).
- Download the spaCy English model used by the clause engine:
  ```bash
  uv run python -m spacy download en_core_web_sm
  ```
- Set `[elasticsearch].index_name` and `[relativity.fields].title = "Unified Title"` in
  `config.toml` (see `config.example.toml`). The `[indexing]` block is optional; defaults
  match report 06 (e.g. `sat_model = "sat-12l-sm"`). The target index must already exist
  (create it with `setup_index.py`); the indexing pipeline never creates or changes the
  index mapping.

Index the workspace (resumes automatically from the progress log):
```bash
python ingest.py --config config.toml --index --output-dir ./logs
```

Retry failures / start fresh:
```bash
python ingest.py --config config.toml --index --output-dir ./logs --retry
python ingest.py --config config.toml --index --output-dir ./logs --fresh
```

Target index: defaults to `[elasticsearch].index_name`; override per run with `--index-name`.
If `--index-name` is given **without** `--index`, you are prompted whether to write (default
No = dry run); in a non-interactive session that combination errors out.
```bash
python ingest.py --config config.toml --index --index-name as-my-other-index --output-dir ./logs
```

Overwrite policy: by default a document whose id already exists is **not** overwritten — it is
reported as a `conflict` error (and is retryable). Pass `--overwrite` to replace existing
documents (recorded as an `overwritten` outcome):
```bash
python ingest.py --config config.toml --index --overwrite --output-dir ./logs
```

Read/block size and limit: both sources use `relativity.batch_size` (default 50); override per
run with `--batch-size`. Use `--limit N` to stop after N documents.

### Progress log

When `--index` is set, progress is appended (one record at a time) to a JSONL file named
`import_{host}_{workspace_id}_{saved_search_id_or_all}.jsonl` in `--output-dir`. Each record is
a final per-document state: a successful index (`ok`, outcome `created`/`overwritten`) or a
failure with its `stage` (`read` or `index`), `error_type`, and `error` message. The Elasticsearch
`caused_by` chain is flattened into the message, so root causes (e.g. an ELSER inference failure)
are visible without re-querying. Successfully indexed ids advance the resume point; `--retry`
re-attempts recorded failures. Completion (100%) is only reported once every document has been
processed end-to-end. During a run the index `refresh_interval` is set to `-1` and restored
afterwards (with a final refresh) for bulk-indexing throughput.

## Testing

The chunker has a dependency-free unit suite (hand-written fakes; no SaT/spaCy/torch):
```bash
uv run pytest tests/unit
```

## Reports
Generated reports are stored in the `reports/` directory.
