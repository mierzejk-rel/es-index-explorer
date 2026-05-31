# es-index-explorer

## Overview
`es-index-explorer` inspects Elasticsearch indices used by aiR Assist and reads
RelativityOne workspace documents so the data can be mapped into new indices.

## Setup
- Create a local config: `cp config.example.toml config.toml`
- Fill in secrets in `config.toml`
- Install dependencies: `uv sync`

## Configuration Reference
`config.toml` uses these sections:
- `[cid]` CID v1 token settings (used for Elasticsearch and Relativity CID auth).
- `[token_cache]` CID token cache settings.
- `[retry]` Network retry policy for CID token fetch.
- `[elasticsearch]` Elasticsearch host list.
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
- `batch_size` (in `[relativity]`) — QuerySlim page size for batch import (default 50).
- `max_text_length` (in `[relativity]`) — `MaxCharactersForLongTextValues` for QuerySlim (default 100000).

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

## Usage
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

Read documents:
```bash
python read_documents.py --config config.toml --limit 10 --save --output-dir reports/
```

## Batch Import (QuerySlim, resumable)

For large workspaces use the batch importer. It supports server-side sort by Artifact ID,
resume after interruptions, and retry of failed documents. Progress is tracked in a JSONL
file keyed by `{host}_{workspace_id}_{saved_search_id_or_all}`.

```bash
python import_documents.py --config config.toml --output-dir ./logs
```

Retry failures:

```bash
python import_documents.py --config config.toml --output-dir ./logs --retry
```

Start from scratch (discard existing state):

```bash
python import_documents.py --config config.toml --output-dir ./logs --fresh
```

Notes:
- Export-based `read_documents.py` is still available for one-shot reads, but it does not support resume.
- `saved_search_id = ""` reads all documents; any integer uses that saved search ID.

## Reports
Generated reports are stored in the `reports/` directory.
