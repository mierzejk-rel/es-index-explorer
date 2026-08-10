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
| `--resume` / `--no-resume` | No | `--resume` | Resume the newest incomplete checkpoint with the same CLI identity, or always create a new checkpoint. |
| `--fresh` | No | false | Ignore matching incomplete checkpoints and create a new checkpoint session. Cannot be combined with `--checkpoint-epoch`. |
| `--checkpoint-epoch EPOCH` | No | none | Resume exactly `checkpoint-<EPOCH>` beneath `--output-dir`. Cannot be combined with `--fresh`. |
| `--experiment-prefix PREFIX` | No | none | Select experiments whose full MLflow name starts with this literal prefix. Mutually exclusive with `--experiment-folder`. |
| `--experiment-folder PATH` | No | `/Users/krzysztof.mierzejewski@relativity.com/DSAS-2836/SimpleMode/` | Select experiment names recursively below this MLflow folder. |
| `--direct-children` | No | false | With `--experiment-folder`, select only immediate child experiments, not nested descendants. |

### Selection rules

Exactly one of `--experiment-prefix` and `--experiment-folder` may be
provided. If neither is provided, the default Simple Mode folder is selected
recursively.

`--direct-children` is valid only with `--experiment-folder`. It is useful
when a folder contains nested experiment groups that should not be included.

Only runs whose MLflow status is `FINISHED` are selected. Failed, running, or
deleted runs are not exported by `export`.

### New export versus resume

Use `--fresh` only for a deliberately new export session:

```bash
UV_ENV_FILE= uv run mlflow_snapshot.py export \
  --profile applied-science \
  --all-runs \
  --output-dir "artifacts/mlflow/simplemode-all-runs-v3-rubrics" \
  --fresh
```

If the command is interrupted, resume with the same selection, `--all-runs`,
concurrency, profile, and output directory, but **without** `--fresh`:

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
snapshot.

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
| `completed` | Final snapshot was published. | Checkpoint is deleted after successful publication. Use a new output directory or `--fresh` for another export. |

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
deciding whether to reuse a committed shard. The fingerprint covers:

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

If a formerly selected remote run is no longer selected (for example, it was
deleted or is no longer `FINISHED`), its local run shard is removed from the
checkpoint and omitted from the completed snapshot.

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

The temporary final-staging directory and completed checkpoint directory are
deleted after a successful publish because their contents are now represented
by checksum-recorded final artifacts.

Schema v3 publishes:

| Artifact | Contents |
|---|---|
| `manifest.json` | Schema version, export time, profile, selection, run IDs/counts, privacy policy, checkpoint epoch, and SHA-256/size for every Parquet file. |
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
| `--experiment-prefix PREFIX` | No | none | Analyze local experiment names beginning with this prefix. Mutually exclusive with `--experiment-folder`. |
| `--experiment-folder PATH` | No | default Simple Mode folder | Analyze local experiment names recursively below this path. |
| `--direct-children` | No | false | Limit local folder selection to immediate child experiments. Requires `--experiment-folder`. |

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
