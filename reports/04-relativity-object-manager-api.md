# Relativity Object Manager: QuerySlim vs Export

**Purpose:** Reference for engineers and AI agents choosing between the two OM document
retrieval strategies in `es-index-explorer`, particularly for large workspaces
(hundreds of thousands of documents) with interrupt/resume requirements.

---

## 1. Two Retrieval Strategies

Relativity Object Manager exposes two fundamentally different ways to retrieve
documents from a workspace. Both accept the same condition language for filtering
but differ in sorting, pagination, statefulness, and long-text handling.

### 1.1 QuerySlim

**OM endpoint:** `POST /object/queryslim`

Each call is a stateless request containing fields, condition, sort specification,
start offset, and page length. The server executes the query, sorts, and returns
one page. To retrieve the next page the caller sends a new request with a higher
`start` value.

**Analogy:** `SELECT ... WHERE ... ORDER BY ... OFFSET x LIMIT y` in SQL.

| Capability | Supported |
|---|---|
| Server-side condition/filter | Yes — `Condition` string in the request body |
| Server-side sort | Yes — `Sorts` array (field identifier + direction) |
| Server-side pagination | Yes — `start` (offset) + `length` (page size) |
| Long text streaming | No — fields truncated at `MaxCharactersForLongTextValues`; may return the truncation token `#KCURA99DF2F0FEB88420388879F1282A55760#` |
| Statefulness | Stateless — each page is an independent HTTP POST |
| Resume on crash | Natural — re-issue the request with the last checkpoint offset or condition |

**es-index-explorer code path:**

```
QueryBuilder.build()  →  QueryBuilder.execute()  →  ObjectManagerAPI.query_slim()
```

`sort_by()`, `where()`, and `page()` all feed into `build()`.

### 1.2 Export Session

**OM endpoints:**

1. `POST /object/initializeexport` — creates a server-side cursor, returns a `RunID`.
2. `POST /object/RetrieveNextResultsBlockFromExport` — fetches the next batch using the `RunID`.

The caller initialises once, then fetches batches in a forward-only loop until the
server returns an empty response.

**Analogy:** A server-side cursor — open once, fetch forward in batches.

| Capability | Supported |
|---|---|
| Server-side condition/filter | Yes — `Condition` in the `QueryRequest` at init time |
| Server-side sort | **No** — the export API does not accept `Sorts`; order is internal/undefined |
| Server-side pagination | **No** in the random-access sense — forward-only cursor with `batchSize`, no `start`/`length` |
| Long text streaming | Yes — truncated fields detected via the truncation token and resolved automatically via `StreamLongText` |
| Statefulness | Stateful — the `RunID` is a server-side cursor; if the process dies the cursor is lost |
| Resume on crash | Not possible from the cursor; must re-initialise from scratch |

**es-index-explorer code path:**

```
QueryBuilder.export()  →  ObjectManagerAPI.export_initialize()
                        →  ObjectManagerAPI.export_retrieve_next()  (loop)
                        →  ObjectManagerAPI.stream_long_text()      (per truncated field)
```

Note: `export()` does **not** pass `Sorts` to the init request (see `fluent.py`
lines 168–175 — only `ObjectType`, `Fields`, `Condition`). The `start` parameter
is hardcoded to `0`.

---

## 2. Side-by-Side Comparison

| Feature | QuerySlim | Export Session |
|---|---|---|
| Sort by Artifact ID | Yes | No |
| Filter `'Artifact ID' > N` | Yes | Yes |
| Filter `'Artifact ID' IN [...]` | Yes | Yes |
| Jump to page / resume at offset | Yes (`start` + `length`) | No (forward-only, cursor lost on crash) |
| Long text auto-streaming | No (must detect truncation token and call `StreamLongText` manually) | Yes (built into the export loop) |
| Suitable for interrupt/resume | Yes | No |
| Suitable for "read everything at once" | Yes (with pagination loop) | Yes (simpler code) |

---

## 3. OM Condition Language (Filtering)

Both QuerySlim and Export accept the same condition syntax (AQLC — Applied Query
Language for Conditions). Conditions are passed as a string in the request body.

### 3.1 Operators supported by OM

| Operator | Syntax | Example |
|---|---|---|
| Equals | `==` | `'Artifact ID' == 12345` |
| Not equals | `<>` | `'Artifact ID' <> 12345` |
| Greater than | `>` | `'Artifact ID' > 12345` |
| Greater or equal | `>=` | `'Artifact ID' >= 12345` |
| Less than | `<` | `'Artifact ID' < 12345` |
| Less or equal | `<=` | `'Artifact ID' <= 12345` |
| Between | `BETWEEN ... AND` | `'Artifact ID' BETWEEN 1000 AND 5000` |
| In list | `IN [...]` | `'Artifact ID' IN [100, 200, 300]` |
| In saved search | `IN SAVEDSEARCH` | `'Artifact ID' IN SAVEDSEARCH 12345` |
| Combinators | `AND`, `OR`, parentheses | `('Artifact ID' > 100) AND ('Artifact ID' < 500)` |

### 3.2 What our `conditions.py` supports today

| Method | Operator | Implemented |
|---|---|---|
| `Field.eq(value)` | `==` | Yes |
| `Field.in_(values)` | `IN [...]` | Yes |
| `Field.in_saved_search(id)` | `IN SAVEDSEARCH` | Yes |
| `Field.gt(value)` | `>` | **No — needs adding** |
| `Field.gte(value)` | `>=` | **No — needs adding** |
| `Field.lt(value)` | `<` | **No — needs adding** |
| `Field.lte(value)` | `<=` | **No — needs adding** |
| `Field.ne(value)` | `<>` | **No — needs adding** |
| `Field.between(low, high)` | `BETWEEN ... AND` | **No — needs adding** |
| `Cond.__and__` / `Cond.__or__` | `AND`, `OR` | Yes |

Adding the missing operators is a small change — each is a one-line string template
method on `Field`.

---

## 4. Sorting

Only QuerySlim supports sorting. The `Sorts` array is part of the QuerySlim
request body and accepts one or more sort specifications:

```json
{
  "Sorts": [
    {
      "FieldIdentifier": {"Name": "Artifact ID"},
      "Order": 0,
      "Direction": "Ascending"
    }
  ]
}
```

`Direction` is either `"Ascending"` or `"Descending"`.

In our code, `QueryBuilder.sort_by()` builds this array. It accepts field
Artifact IDs or GUIDs (not field display names directly, though
`RelativityIdentifier` resolves names too).

Export has no equivalent — documents arrive in an undefined internal order.

---

## 5. Long Text Handling

### Export (automatic)

The export generator in `fluent.py` checks each field value for the OM truncation
token (`#KCURA99DF2F0FEB88420388879F1282A55760#`). When detected, it calls
`ObjectManagerAPI.stream_long_text(artifact_id, field_id)` to fetch the full text
via a separate endpoint. This happens transparently inside the `export()` generator.

### QuerySlim (manual)

QuerySlim returns field values directly. When `LongTextBehavior` is `"Tokenized"`
(our default), long text fields exceeding `MaxCharactersForLongTextValues` are
replaced with the truncation token. The caller must detect this and call
`stream_long_text` manually.

The same detection logic used in `export()` can be extracted and reused in a
QuerySlim pagination loop.

---

## 6. Strategy for Large Workspaces with Interrupt/Resume

For workspaces with hundreds of thousands of documents (multi-GB of extracted
text), the recommended approach uses **QuerySlim with sorted pagination**:

### 6.1 Initial full read

```
1. Sort by Artifact ID ascending.
2. Condition: 'Artifact ID' > 0  (or no condition for first run).
3. Page: start=0, length=500 (configurable batch size).
4. For each page:
   a. Process documents.
   b. Detect and resolve truncated long text fields.
   c. Persist last Artifact ID in page to a checkpoint file.
5. Increment start by length, repeat until page returns fewer than length rows.
```

### 6.2 Resume after interrupt

```
1. Read checkpoint: last_artifact_id = 12345.
2. Condition: 'Artifact ID' > 12345
3. Sort by Artifact ID ascending.
4. Page from start=0 (condition skips already-processed documents server-side).
5. Continue as above.
```

Using a condition-based resume (`> last_id`) rather than offset-based (`start=N`)
is more robust: if documents are added or removed between runs, the offset would
be wrong, but the artifact ID condition is always correct.

### 6.3 Retry specific failures

```
1. Read failure log: failed_ids = [100, 200, 300].
2. Condition: 'Artifact ID' IN [100, 200, 300]
3. No sort needed (small set).
4. Single page or small pages.
```

### 6.4 Summary

| Operation | Condition | Sort | Pagination |
|---|---|---|---|
| Full read | None or saved search | Artifact ID ASC | `start`/`length` loop |
| Resume | `'Artifact ID' > {checkpoint}` | Artifact ID ASC | `start`/`length` loop |
| Retry failures | `'Artifact ID' IN [ids]` | None | Single page or small pages |

---

## 7. What `read_documents()` Uses Today

The current `reader.py` uses **Export only** via `builder.export()`. This works
for smaller workspaces or one-shot reads but does not support sorting or resume.

The `QueryBuilder` already has `sort_by()`, `page()`, and `where()` wired for
QuerySlim via `execute()`. Adding a QuerySlim-based reader path requires:

1. Comparison operators on `Field` in `conditions.py` (small).
2. A pagination loop that calls `execute()` repeatedly, handles long text
   truncation, and persists checkpoint state.
3. The existing export path can remain for simpler use cases.

---

## 8. Code Pointers

| Component | File | Key methods |
|---|---|---|
| Low-level QuerySlim | [`object_manager.py`](../es_index_explorer/relativity/object_manager.py) | `query_slim()` |
| Low-level Export | [`object_manager.py`](../es_index_explorer/relativity/object_manager.py) | `export_initialize()`, `export_retrieve_next()` |
| Long text streaming | [`object_manager.py`](../es_index_explorer/relativity/object_manager.py) | `stream_long_text()` |
| Fluent builder | [`fluent.py`](../es_index_explorer/relativity/fluent.py) | `QueryBuilder` |
| QuerySlim execution | [`fluent.py`](../es_index_explorer/relativity/fluent.py) | `execute()` → `query_slim()` |
| Export execution | [`fluent.py`](../es_index_explorer/relativity/fluent.py) | `export()` → `export_initialize()` + loop |
| Conditions | [`conditions.py`](../es_index_explorer/relativity/conditions.py) | `Field`, `Cond`, `field()` |
| Document reader | [`reader.py`](../es_index_explorer/relativity/reader.py) | `read_documents()` (export-based) |
| Truncation token | [`object_manager_models.py`](../es_index_explorer/relativity/object_manager_models.py) | `R1_OBJECT_MANAGER_TRUNCATE_TOKEN` |
