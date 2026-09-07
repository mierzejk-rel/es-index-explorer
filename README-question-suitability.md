# Simple Mode question-suitability analysis

The pipeline implements the analysis contract in
`reports/13-simple-mode-analysis-research-plan.md`. It is isolated from the
existing MLflow snapshot CLI and writes only under:

```text
artifacts/question_analysis/simplemode-v1-postreview-p3/
├── manifest.json
├── state.json
├── tables/
├── draws/
├── annotations/
├── statistics/
├── figures/
├── partial_reports/
└── logs/
```

The `artifacts/question_analysis/simplemode-v1/` directory is the immutable pre-review
Segment 3 run. `artifacts/question_analysis/simplemode-v1-postreview/` is the immutable
P1-remediated run, and `artifacts/question_analysis/simplemode-v1-postreview-p2/` is the
P2-remediated run. All three are retained for comparison and are not resumed or rewritten.
The P3-remediated root shown above is the default for regenerated Segment 3 artifacts and all
later commands.

## Environment

Install the Python groups required by the pipeline:

```bash
uv sync --group analysis --group annotation --group analysis-test
```

- `analysis`: numerical, tabular, plotting, Stanza 1.14.0, spaCy 3.8.14, and
  the pinned `en_core_web_sm` 3.8.0 model.
- `annotation`: Cursor Python SDK only.
- `analysis-test`: pytest, Ruff, and ty for this pipeline.

The official Stanza 1.14.0 resource catalogue is pinned by SHA-256 in
`es_index_explorer/question_analysis/resources/stanza-en-resource-manifest.json`.
Setup downloads only the selected processor/package mapping and its declared
dependencies; those files are checksum verified. The spaCy distribution and
28-file model tree are also verified before loading.

Download them once, then subsequent feature runs are offline:

```bash
UV_ENV_FILE= uv run question-suitability features --download-resources
```

R is not a Python dependency. Its base environment is pinned separately in
`tests/oracles/fwildclusterboot/Dockerfile`; the package lock and reference
fixtures are added with the statistical oracle.

## CLI

```bash
uv run question_suitability.py --help
uv run question_suitability.py status
uv run question-suitability status
uv run question-suitability join
uv run question-suitability features
```

The public commands are:

```text
join → features → annotate-emit → annotate-run → annotate-ingest
     → gold-sample → gold-ingest → validate-features
oracle
fit-layer1 + fit-families → robustness → report
status
```

`oracle` may run at any time before either fit command. Both fits require the
annotation/gold/validation unlock and the recorded oracle pass. `status` is
read-only and works for an uninitialized root.

`join` reads the frozen schema-v3 snapshot and authoritative
rubric TOMLs, then writes the three catalogues, criterion and PFU tables,
discrepancies, structural verification, the frozen F3 rank decision, and
`partial_reports/01-data-contract.md`. Override its default sibling-repository
locations with `--snapshot-dir`, `--rubric-root`, and `--task`.

`features` reads only those catalogues, parses all 263 questions and 314
expectation descriptions with the pinned Stanza/spaCy resources, and writes the
deterministic feature table, token/entity archives, resource verification, and
`partial_reports/02-deterministic-features.md`. Override the model cache with
`--stanza-model-dir`.

Catalogue and feature `use_cases` columns are Parquet logical `list<string>`.
Use `pd.read_parquet(path, dtype_backend="pyarrow")` when Python list cells are
required, or normalize iterable values at the consumer boundary. `source_text`
and `text_sha256` preserve canonical catalogue bytes, including source
whitespace.

Later commands remain unavailable until their implementation segments register
them. The shell never marks a placeholder command complete.

## Segment 4 annotation

Segment 4 uses the local Cursor Python SDK. It needs a one-time Cursor user API
key setup; keep the key outside the repository in `CURSOR_API_KEY`. For the
exact setup, model-access checkpoint, isolation boundary, and command sequence,
see [`reports/15-segment-4-annotation-operations.md`](reports/15-segment-4-annotation-operations.md).

Exit categories are stable:

- `1`: malformed input or persisted state
- `2`: blocking scientific or structural gate
- `3`: incomplete prerequisite or unavailable command implementation
- `4`: numerical non-computability

Every input and completed output is SHA-256 checked. Writes use same-directory
temporary files, `fsync`, and atomic replacement. `manifest.json` records the
analysis-lock tag, tool versions, resource hashes, and named seeds. A running or
failed command can resume; re-running a completed command is an idempotent
no-op.

## Frozen randomness

Master seed `20` derives independent 64-bit streams with SHA-256 over
`"<master_seed>:<stream_name>"`. Python's built-in `hash()` is not used.
