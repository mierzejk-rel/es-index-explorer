# MLflow Snapshot and Rubric Analysis

This document describes the read-only MLflow snapshot exporter and its local
analysis commands in `mlflow_snapshot.py`.

The tool has two separate phases:

1. `export` downloads selected finished MLflow experiment runs into a
   privacy-reduced local snapshot.
2. `analyze`, `analyze-stage-a`, `analyze-stage-b`, and `analyze-rubrics` read
   local snapshot files only. They do not contact MLflow.

`analyze-stage-a` and `analyze-stage-b` are two concrete *stage adapters*
built on one reusable, stage-neutral analysis core. See
[Arm-level stage adapters: a reusable pattern](#arm-level-stage-adapters-a-reusable-pattern)
before adding a Stage C or E adapter.

The enriched rubric workflow requires a **schema v3** snapshot. It retains
rubric question text, run rubric roster identity, and criterion states, but it
continues to exclude agent responses, search queries, document content, XML,
and scorer rationale text.

## Prerequisites

Run commands from the `es-index-explorer` repository root.

Install the MLflow analysis dependency group once:

```bash
UV_ENV_FILE= uv sync --group mlflow
```

`UV_ENV_FILE=` prevents uv from loading a repository `.env` file. It is
optional when that behavior is not needed.

The exporter requires an authenticated Databricks CLI profile. The examples
use `applied-science`:

```bash
databricks auth login --profile applied-science
```

## Commands at a glance

| Command | Contacts MLflow | Required input | Main output |
|---|---:|---|---|
| `export` | Yes | profile | schema v3 local snapshot |
| `analyze` | No | snapshot | generic local summaries |
| `analyze-stage-a` | No | Stage-A-only snapshot | arm-level controlled comparisons and Stage B recommendations |
| `analyze-stage-b` | No | Stage-B-only snapshot | arm-level controlled comparisons and within-trace BM25/dense overlap |
| `analyze-rubrics` | No | schema v3 snapshot, rubric root, task TOML | rubric/question/expectation/use-case analysis |

## Export

### Full S-suite export

This command exports all finished runs under the Simple Mode experiment
folder. Use a new output directory for each independently reproducible
snapshot:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --fresh
```

### Export parameters

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--profile NAME` | Yes | none | Databricks CLI profile used for read-only MLflow access. |
| `--output-dir PATH` | No | `artifacts/mlflow/snapshot-<UTC timestamp>` | Snapshot destination and checkpoint parent. Specify it explicitly whenever interruption/resume is possible. |
| `--all-runs` | No | false | Export every finished run in each selected experiment. Without it, export only the latest finished run per selected experiment. |
| `--trace-fetch-concurrency N` | No | `10` | Concurrent full-trace downloads. Allowed values are `1` through `10`. |
| `--resume` / `--no-resume` | No | `--resume` | Resume the newest checkpoint matching this profile (including one already `completed`), regardless of which paths it was previously exported with, or always create a new checkpoint. |
| `--fresh` | No | false | Ignore matching checkpoints and create a new checkpoint session. Cannot be combined with `--checkpoint-epoch`. |
| `--checkpoint-epoch EPOCH` | No | none | Resume exactly `checkpoint-<EPOCH>` beneath `--output-dir`, even if already `completed`. Cannot be combined with `--fresh`. |
| `--delete-checkpoint-after-publish` | No | false | Delete the entire checkpoint directory once this export finishes. Default is to retain it for a future incremental export. This is the only way to remove cached data. |
| `--skip-fingerprint-validation` | No | false | Opt-in speedup: reuse every already-committed selected run immediately, without checking MLflow for remote metadata/trace changes. Newly discovered runs are still fully downloaded and fingerprinted. See [Skipping fingerprint validation for a faster refresh](#skipping-fingerprint-validation-for-a-faster-refresh). |
| `--experiment-prefix PREFIX` | No | none | Select experiments whose full MLflow name starts with this literal prefix. Repeatable; unioned with every other `--experiment-prefix`/`--experiment-folder` given. |
| `--experiment-folder PATH` | No | `/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/` (only when neither flag is given) | Select experiment names recursively below this MLflow folder. Repeatable; unioned with every other `--experiment-prefix`/`--experiment-folder` given. |
| `--direct-children` | No | false | Limit matching under every given `--experiment-folder` to immediate child experiments, not nested descendants. |

### Selection rules

`--experiment-prefix` and `--experiment-folder` may each be repeated, and both
kinds may be combined in the same command. An experiment name is selected if
it matches *any* of the given prefixes or folders (union/OR semantics). If
neither flag is given at all, the default Simple Mode folder is selected
recursively.

`--direct-children` requires at least one `--experiment-folder` and applies
uniformly to all of them. It is useful when a folder contains nested
experiment groups that should not be included.

Only runs whose MLflow status is `FINISHED` are selected. Failed, running, or
deleted runs are not exported by `export`.

### Changing paths across exports

Because the exporter's local cache is keyed only by `--profile` (not by the
selector), you can freely change, narrow, widen, or add `--experiment-folder`/
`--experiment-prefix` values between invocations against the same
`--output-dir` without losing previously downloaded runs — see
[Adding or updating runs later](#adding-or-updating-runs-later-incremental-re-export).
For example, to add a sibling experiment folder discovered after your first
export:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "experiments/A/" \
  --experiment-folder "experiments/B/" \
  --all-runs \
  --output-dir "artifacts/mlflow/combined-a-and-b"
```

The **published** snapshot always reflects exactly the paths given in the
*most recent* invocation — if you later re-run with only `--experiment-folder
"experiments/A/"`, the published `runs.parquet`/etc. shrinks back down to just
A, even though B's cached shards remain on disk, untouched, ready to be
included again cheaply whenever you add that path back.

### New export versus resume

Use `--fresh` only for a deliberately new export session:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --fresh
```

If the command is interrupted, resume with the same profile and output
directory, but **without** `--fresh` (the selection, `--all-runs`, and
concurrency may differ from the interrupted attempt if you want):

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics"
```

To resume a known checkpoint rather than the newest matching one:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --checkpoint-epoch 1785919731
```

`--fresh` does not delete a completed snapshot already in the output
directory. A successful new export atomically replaces the published snapshot
files at that output path. Use a new `--output-dir` to preserve an older
snapshot. `--fresh` also does not delete any retained checkpoint from a prior
session; it only ignores it and starts a new `checkpoint-<epoch>` alongside it.

### Adding or updating runs later (incremental re-export)

The checkpoint is retained after a successful publish by default, and it is
keyed only by `--profile` — not by the selector. To add newly finished runs
(for example, an experiment arm you finished running later, or a sibling
folder you now also want included), pick up a metadata/status change on an
existing run, or narrow back down to fewer paths, re-run `export` against the
same `--output-dir` and `--profile` with whatever `--experiment-folder`/
`--experiment-prefix` values you currently want:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics"
```

This resumes the retained checkpoint (even though it is marked `completed`,
and even if the selection differs from before) and, for every run matched by
the *current* selection that is already cached, performs a single lightweight
metadata-only fingerprint check instead of a full trace re-download. Only
genuinely new or changed runs are downloaded. Runs cached from a previous,
different selection but not matched this time are left untouched — never
checked, never deleted; the cache only ever grows.

`manifest.json`/parquet files are only re-aggregated and atomically replaced
when the publish is actually needed: something was downloaded or invalidated
this session, there is no valid previous `manifest.json` to compare against
(including the very first export of an empty selection, which still publishes
a valid empty snapshot rather than being skipped), or the current invocation's
selected experiment IDs, run IDs, selector, or `--all-runs` differ from what
was last published — a selector change alone (for example widening or
narrowing `--experiment-folder`/`--experiment-prefix`) triggers a republish
even when it happens to resolve to the exact same run IDs as before.
`--trace-fetch-concurrency` is deliberately not compared: changing only the
download parallelism between invocations does not by itself trigger a
republish, since it does not affect the content of the published data.
Otherwise the command logs that the snapshot is already up to date and leaves
the published files untouched.

Retaining the checkpoint keeps every run's shard payload downloaded so far on
disk, even for paths not included in the most recent invocation, alongside the
published parquet files — disk usage for that output directory only grows
over time as you add more paths. Pass `--delete-checkpoint-after-publish` once
you do not expect to add, refresh, or re-include any more runs for that
`--output-dir`; it deletes the entire checkpoint (every cached shard for every
path ever exported there), not just what is unselected this time:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --delete-checkpoint-after-publish
```

### Skipping fingerprint validation for a faster refresh

The metadata/trace-inventory fingerprint check described above is what makes
incremental re-export safe, but it is also the slowest per-run step — each
already-cached run still pages through its full trace inventory remotely
before it can be reused, which can dominate wall-clock time once dozens of
runs are cached. Pass `--skip-fingerprint-validation` to bypass that check
for every run that already has a committed local shard:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-stage-v3" \
  --skip-fingerprint-validation
```

The exact safety contract:

- Every currently selected run that already has a committed shard is reused
  immediately, with **no** MLflow fingerprint call. Its cached payload is
  trusted as-is, even if its remote metadata, trace inventory, or assessments
  changed since it was originally downloaded.
- A newly discovered run without a committed shard is unaffected: it is still
  fully downloaded and fingerprinted, exactly as without the flag.
- A run that satisfies the current selection (matching paths/prefixes) but
  has since been deleted or is otherwise missing from MLflow is excluded from
  the published snapshot either way — this is unchanged from the default
  behavior and is independent of this flag. Its cached shard, if any, is
  retained untouched, never deleted by this flag.
- `manifest.json` records whether the most recent publish used
  `skip_fingerprint_validation`, so a later switch between skipped and
  validated mode is itself detected as a signature change and triggers a
  local re-aggregation/manifest rewrite (at no MLflow cost), even when no run
  was downloaded or invalidated that session.
- Rerun `export` **without** this flag to restore full validation and refresh
  any run whose remote metadata, trace inventory, or assessments changed
  while validation was skipped. The flag trades freshness assurance for
  speed; it never deletes or corrupts cached data.

### Migration note: checkpoint identity change

The checkpoint's resume-matching identity is now just the Databricks profile
(and its derived tracking URI) — it no longer includes the selector,
`--all-runs`, or `--trace-fetch-concurrency`. A checkpoint created before this
change stored a different identity shape and will not be recognized as a
resume target afterward. The next `export` against that same `--output-dir`
creates a brand-new checkpoint and does one full download for whatever paths
you select; from then on, that new checkpoint is freely resumable and
extensible across any future selector change, exactly as described above. The
old checkpoint directory is not deleted automatically — it is simply no
longer used. Remove it manually (`rm -rf`) if you want the disk space back.

## Export lifecycle, checkpoints, and resume

The exporter uses atomic per-run checkpoints. A checkpoint directory is
created under the output directory:

```text
<output-dir>/
├── checkpoint-<unix epoch>/
│   ├── checkpoint.json
│   ├── experiments.parquet
│   └── runs/
│       └── <run-id>/
│           ├── .committed
│           ├── fingerprint.json
│           └── run_payload.pickle
└── ...
```

### Checkpoint states

| State | Meaning | Resume behavior |
|---|---|---|
| `in_progress` | Run discovery or run download is underway. | Reuses valid committed run shards and downloads missing runs. |
| `interrupted` | The process received `Ctrl+C`. | Same as `in_progress`; committed shards remain valid candidates. |
| `aggregating` | Every selected run has a committed shard and final Parquet publication is underway. | The next invocation verifies/reuses shards and completes publication. |
| `completed` | Final snapshot was published (or re-checked with no changes found). | Checkpoint is retained by default and is a valid resume target: the next matching export reuses every unchanged shard via a metadata-only fingerprint check and only re-publishes if something changed. Pass `--delete-checkpoint-after-publish` to remove it instead. |

### What is committed per run

A run is resumable only when its shard has all three:

1. `run_payload.pickle` — lossless trusted-local Python payload containing
   sanitized rows for that run;
2. `fingerprint.json` — remote identity/fingerprint recorded at download time;
3. `.committed` — written only after the payload and fingerprint have been
   durably written.

Partially downloaded runs stay only in a `.tmp-<run-id>-<pid>` directory.
They have no `.committed` marker and are discarded on the next startup. The
run is downloaded again from the beginning; no incomplete pages or partial
trace set are published.

### Remote checks before reusing a shard

For every selected run, the exporter recalculates a remote fingerprint before
deciding whether to reuse a committed shard, unless `--skip-fingerprint-validation`
is passed, in which case every already-committed selected run is reused
without this check (see
[Skipping fingerprint validation for a faster refresh](#skipping-fingerprint-validation-for-a-faster-refresh)).
The fingerprint covers:

- run ID, experiment ID/name, status, start/end times, and lifecycle stage;
- all run parameters;
- all run metrics;
- trace membership;
- trace ID, status, request timestamp, execution duration;
- safe trace tags/attributes;
- assessment name, value, and sanitized assessment metadata.

If the new fingerprint matches the local `fingerprint.json`, the committed
shard is reused. If it differs, the exporter deletes only that run's shard,
records the run ID in `invalidated_run_ids`, and downloads that one run again.
Other committed runs remain reusable.

This fingerprint check, and any resulting shard invalidation, only happens for
runs matched by the *current* invocation's selector. A run cached from a
previous, different selection that is not matched this time is left
completely alone: not fingerprinted, not touched, not deleted — including
when it was deleted or is no longer `FINISHED` remotely. It simply is not
part of this invocation's published snapshot. Nothing is ever pruned from the
cache automatically; `--delete-checkpoint-after-publish` (removing the entire
checkpoint) is the only deletion mechanism.

### Integrity boundary

The fingerprint does not guarantee detection of an in-place full-span payload
change when MLflow does not update the trace metadata used by the inventory.
Likewise, a changed `rubrics.json` artifact without a corresponding run or
trace metadata change is not independently fingerprinted before a shard is
reused. MLflow run artifacts are normally immutable after a finished run; use
`--fresh` if artifact provenance must be re-downloaded unconditionally.

### Authentication interruption and recovery

Trace API calls use bounded retry. After exhausted authentication retries in
an interactive terminal, the exporter pauses and asks you to run:

```bash
databricks auth login --profile <profile>
```

in another terminal. Press Enter in the exporter after login; it recreates the
MLflow client and resumes the incomplete run. In a non-interactive terminal,
the exporter stops with a command to run before resuming.

## Published snapshot artifacts

After every selected run has committed, the exporter aggregates shards into a
temporary final-staging directory. It publishes each file into `--output-dir`
with atomic replacement, then writes `manifest.json`.

The temporary final-staging directory is always deleted after a successful
publish. The checkpoint directory is retained by default (see
[Adding or updating runs later](#adding-or-updating-runs-later-incremental-re-export))
unless `--delete-checkpoint-after-publish` was passed.

Schema v3 publishes:

| Artifact | Contents |
|---|---|
| `manifest.json` | Schema version, export time, profile, selector, `all_runs`, `trace_fetch_concurrency`, `skip_fingerprint_validation`, selected experiment IDs/count, run IDs/count, privacy policy, checkpoint epoch, and SHA-256/size for every Parquet file. Experiment IDs, run IDs, the selector, `all_runs`, and `skip_fingerprint_validation` are what a later export compares against to decide whether the published snapshot needs to change. |
| `experiments.parquet` | Selected experiment IDs, names, and lifecycle stages. |
| `runs.parquet` | Run status/timing and sanitized experiment configuration, including Simple Mode parameters. |
| `run_metrics.parquet` | All run-level MLflow metric key/value rows. |
| `trace_quality.parquet` | Typed raw assessment values, ordinal grade, detected error modes, trace status, and rubric identity fields. |
| `trace_invocations.parquet` | Root `invoke_*` span identity, actual rubric question, use case, dataset, variant, row ID, rubric path, and parsed rubric/variant indices. |
| `trace_criteria.parquet` | Rubric expectation name, resolved material marker, and PASS/FAIL/UNDETERMINED state. Criterion rationale text is excluded. |
| `run_rubrics.parquet` | Sanitized run-level `rubrics.json` roster: source path/name, per-variant question, dataset, use case, author, artifact hash/status, and roster order. |
| `trace_failures.parquet` | Safe Simple retrieval-plan validation failures. |
| `trace_retrieval.parquet` | Retrieval mode/tool identity, chunk counts, merge/context data, timing attributes, and rank IDs. |
| `span_timings.parquet` | Span timing and successful-attempt metrics without inputs/outputs or error rationale. |

Local export directories are ignored by Git (`artifacts/`).

## Generic local analysis

`analyze` produces generic quality/latency/Pareto outputs and does not contact
MLflow:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py analyze \
  --snapshot "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/"
```

### Analyze parameters

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--snapshot PATH` | Yes | none | Completed schema v3 snapshot directory. |
| `--experiment-prefix PREFIX` | No | none | Analyze local experiment names beginning with this prefix. Repeatable; may be combined with `--experiment-folder`. |
| `--experiment-folder PATH` | No | default Simple Mode folder (only when neither flag is given) | Analyze local experiment names recursively below this path. Repeatable; may be combined with `--experiment-prefix`. |
| `--direct-children` | No | false | Limit matching under every given `--experiment-folder` to immediate child experiments. |

Generic analysis creates or replaces `<snapshot>/analysis/`:

| Artifact | Contents |
|---|---|
| `run_summary.csv` | Run-level quality, latency, context, and retrieval summary. |
| `pareto_candidates.csv` | Non-dominated runs by dataset quality and p50 latency. |
| `grade_distribution.csv` | Ordinal-grade counts by dataset/use case. |
| `plan_validation_failures.csv` | Exported Simple plan-validation failures. |
| `trace_invocations.csv` | Selected invocation identity rows. |
| `trace_criteria.csv` | Selected criterion-state rows. |
| `run_rubrics.csv` | Selected run rubric-roster rows. |
| `summary.json` | Counts, schema version, and analysis caveats. |
| `summary.md` | Human-readable summary and Pareto table. |

## Arm-level stage adapters: a reusable pattern

`analyze-stage-a` and `analyze-stage-b` are **stage adapters**: thin,
stage-specific modules built on one shared, stage-neutral analysis core in
[`es_index_explorer/mlflow_analysis/experiment_arms.py`](es_index_explorer/mlflow_analysis/experiment_arms.py).
The core does not know anything about any specific experiment stage's naming
scheme or parameter set. It implements only the algorithms that are the same
for every stage:

- trace-level and run-level aggregation from the sanitized snapshot
  (`build_trace_level`, `build_run_level`);
- quality/latency summaries, use-case aggregation, operation-timing/retry
  summaries, and grade distributions;
- a generic pairwise-comparison engine (`pairwise_comparisons`) that takes a
  stage's `comparison_type`/`orient_comparison` classifiers as plain
  callables and does the join/grouping/aggregation mechanics itself;
- Pareto-set membership (`pareto_sets`);
- two retrieval-overlap strategies: `between_arm_family_overlap` (compare two
  *different* single-signal arms matched on shared dimensions — Stage A's
  BM25-only vs. dense-only) and `within_trace_family_overlap` (compare two
  signals mixed *within one arm's own trace* — Stage B's heterogeneous
  `current_union` arms);
- shared I/O, hashing/manifest, and Markdown-table helpers.

A stage adapter (`stage_a.py`, `stage_b.py`) supplies only what is genuinely
stage-specific:

1. an arm-identity regex/parser that turns an experiment short name into
   `arm_id`, `dataset_segment`, and a list of extra per-arm dimension columns
   (for example Stage A's `retrieval_family`/`calls`/`fetch`/`context`, or
   Stage B's `tool_multiset`/`calls`/`fetch`/`homogeneous`);
2. `comparison_type`/`orient_comparison` functions that classify an unordered
   arm pair into a single-dimension controlled comparison (or `None` when the
   pair differs on more than one dimension and is therefore not a clean,
   causally interpretable comparison);
3. which overlap strategy applies, if any, and with which family values;
4. expected population counts for the validation gate (arm count x dataset
   count = expected run/experiment count);
5. a Markdown report renderer for that stage's audience.

Nothing about a stage's parameter set, naming, or number of arms is
hardcoded in `experiment_arms.py`. Adding a Stage C or E adapter means writing
a new `stage_c.py`/`stage_e.py` module following this same five-piece recipe,
adding a `analyze-stage-c`/`analyze-stage-e` subcommand in
`mlflow_snapshot.py`, and nothing else. No existing adapter or the shared core
needs to change.

### Stage A arm analysis

`analyze-stage-a` is the original arm-level Stage A analysis, now implemented
as a stage adapter over the shared core. It provides controlled pairwise
comparisons, BM25/dense retrieval overlap between matched single-signal arms,
Pareto sets by arm, operation timing and retry diagnostics, metric
cross-checks, and the Stage B recommendation report.

It is intentionally restricted to the historical 54-run Stage A snapshot. Do
not run it on a combined Stage A/B snapshot; use `analyze` and
`analyze-rubrics` for stage-neutral analysis.

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py analyze-stage-a \
  --snapshot "artifacts/mlflow/simplemode-all-runs" \
  --report "reports/11-stage-a-results-stage-b-redesign.md"
```

#### Stage A parameters

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--snapshot PATH` | Yes | none | Completed 54-run Stage A snapshot. |
| `--report PATH` | No | `reports/11-stage-a-results-stage-b-redesign.md` | Markdown report path to create or replace. |

The command creates `<snapshot>/stage_a_analysis/` with trace/run-level
calculations, pairwise comparisons, overlap, Pareto, quality/latency/use-case
summaries, operation timings, grade distributions, validation, and calculation
manifest files. The report is written to `--report`.

### Stage B arm analysis

`analyze-stage-b` is the Stage B adapter. It validates the exact six-arm
`current_union` matrix and each run's exported runtime configuration before
reporting controlled pairwise comparisons: fetch effect at a matched retrieval
tool multiset, and tool-composition effect — homogeneous versus heterogeneous —
at matched call count and fetch. It calculates within-trace BM25/dense
overlap for heterogeneous arms, Pareto sets, operation timing/retry
diagnostics, metric cross-checks, and an evidence-only Stage B report. It does
not prescribe a Stage C design.

It is intentionally restricted to the Stage B snapshot (6 arms x 3 datasets =
18 runs). Do not run it on a combined Stage A/B snapshot; use `analyze` and
`analyze-rubrics` for stage-neutral analysis.

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py analyze-stage-b \
  --snapshot "artifacts/mlflow/simplemode-stage-b" \
  --report "reports/12-stage-b-results.md"
```

#### Stage B parameters

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--snapshot PATH` | Yes | none | Completed 18-run Stage B snapshot. |
| `--report PATH` | No | `reports/12-stage-b-results.md` | Markdown report path to create or replace. |

The command creates `<snapshot>/stage_b_analysis/` with trace/run-level
calculations, pairwise comparisons, within-trace BM25/dense overlap, Pareto,
quality/latency/use-case summaries, operation timings, grade distributions,
validation, and calculation manifest files. The report is written to
`--report`.

Pairs that differ in call count *and* tool composition at the same time
(for example the c3 `bm25-dense-bm25` arms versus the c4 `bm25-dense-bm25-dense`
arm) are intentionally excluded from `pairwise_comparisons.csv`: they mix two
dimensions and are not a clean, causally interpretable comparison.

The Stage B tool-name segment is an experiment identifier for the configured
tool **multiset**, not a runtime call order. Runtime validation accepts the
exact multiset and the agent canonically orders successful results before
merging. The analysis therefore reports `tool_multiset`, never an emitted-call
or merge-order sequence.

## Rubric analysis

`analyze-rubrics` performs the complete rubric analysis locally against a
schema v3 snapshot. It is stage-neutral: use it for Stage A, B, C, E, or any
future experiment suite included in the snapshot.

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py analyze-rubrics \
  --snapshot "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --rubric-root "/Users/krzysztof.mierzejewski/PycharmProjects/r1-evals-new/src/r1_evals/rubrics/rubric_data" \
  --task "/Users/krzysztof.mierzejewski/PycharmProjects/r1-evals-new/rubrics/tasks/air_assist.toml"
```

### Rubric-analysis parameters

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--snapshot PATH` | Yes | none | Completed schema v3 snapshot. |
| `--rubric-root PATH` | Yes | none | Local `r1-evals-new` `rubric_data` root. |
| `--task PATH` | Yes | none | Air Assist task TOML defining use-case taxonomy and descriptions. |

The command creates or replaces `<snapshot>/rubric_analysis/`.

### Rubric join contract

The analysis uses three sources:

1. Root invocation spans supply actual question and runtime identity.
2. Run `rubrics.json` roster rows supply run-specific rubric path, variant,
   question, dataset, and use-case metadata.
3. Local TOMLs supply canonical question, expectations, multi-label use cases,
   source hash, and task-defined descriptions.

Direct RubricV2 `file_path` identity is preferred. The positional
`invoke_<use_case>_<rubric_index>[_v<variant>]` fallback is recorded as
lower-confidence. A mismatch/ambiguity is reported and excluded from
aggregates.

The analysis excludes legacy `weight` and `category`. Missing V2 `material`
resolves to the V2 runtime default of `true`; it is never inferred from
legacy weight.

### Rubric-analysis artifacts

| Artifact | Contents |
|---|---|
| `rubric_catalogue.csv` | Parsed rubric identity, canonical question, source hash, task/dataset metadata, multi-label use cases, and provenance. |
| `rubric_expectations.csv` | Expectation specification text, material marker, document IDs, indicators, hints, and ordinal position; no legacy weight/category. |
| `trace_rubric_join.csv` | Every invocation's join decision, join method, resolved catalogue ID, and source path. |
| `rubric_join_discrepancies.csv` | Missing artifact, positional fallback, unparsable span identity, path ambiguity, source miss, question/dataset/identity mismatch, and catalogue issues. |
| `trace_rubric_outcomes.csv` | One eligible trace per rubric with RubricV2 score, ordinal grade, error modes, and provenance. |
| `trace_rubric_assessments.csv` | All raw typed MLflow assessment values joined to eligible rubric traces. |
| `trace_expectation_outcomes.csv` | Criterion state joined to its TOML expectation. |
| `rubric_summary.csv` | Aggregate score and ordinal-grade counts by rubric. |
| `question_summary.csv` | Aggregate score and ordinal-grade counts by canonical question. |
| `expectation_summary.csv` | PASS/FAIL/UNDETERMINED counts and rates by expectation. |
| `use_case_summary.csv` | Exploded multi-label use-case summary with technical key, human label, task description, trace count, and mean RubricV2 score. |
| `error_mode_outcomes.csv` | Every detected error mode on eligible traces. |
| `error_mode_summary.csv` | Error-mode count and rate per rubric. |
| `rubric_analysis_coverage.json` | Coverage, exclusions, discrepancy counts, catalogue hash, and complete source snapshot manifest. |

## Export and analysis in one shell command

Export must finish before local analysis begins. Chain commands with `&&`; if
export fails or is interrupted, later commands do not run:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --fresh \
&& UV_ENV_FILE= uv run mlflow_snapshot.py analyze \
  --snapshot "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --experiment-folder "/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/" \
&& UV_ENV_FILE= uv run mlflow_snapshot.py analyze-rubrics \
  --snapshot "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --rubric-root "/Users/krzysztof.mierzejewski/PycharmProjects/r1-evals-new/src/r1_evals/rubrics/rubric_data" \
  --task "/Users/krzysztof.mierzejewski/PycharmProjects/r1-evals-new/rubrics/tasks/air_assist.toml"
```

Do not use `--fresh` in a resumed command. Resume export first; after it
completes, run the local analysis commands.
