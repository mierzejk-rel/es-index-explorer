# Segment 5 gold-validation operations

This guide governs Segment 5 execution after the verified Stage 4 handoff. The scientific
contract is `reports/13-simple-mode-analysis-research-plan.md`; the user decisions completing
that contract are recorded in `reports/16-segment-5-gold-validation-decisions.md`.

## Root lineage

The immutable lineage is:

```text
simplemode-v1-postreview-p3
  → simplemode-v1-postremediation-final
  → simplemode-v1-segment5
```

The first two roots are historical Stage 4 snapshots and are never modified. Segment 5 uses a
fresh lock because its implementation changes resources fingerprinted by the Stage 4 root.
Before the live gold workflow begins, the new root rematerializes `join`, `features`, and
`annotate-emit`, migrates only hash-verified Stage 4 annotation provenance in resume-only mode,
and regenerates `annotate-ingest`. No annotation model call is required.

## Commands and manual boundary

The implementation-preparation sequence is:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability join
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability features
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability annotate-emit
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability annotate-run \
  --migrate-from-root artifacts/question_analysis/simplemode-v1-postremediation-final \
  --resume-only
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability annotate-ingest
```

Implementation verification stops here. The following commands are live data-gate actions and
run only after the user explicitly starts adjudication:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability gold-sample
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability gold-ingest-initial \
  --adjudication-csv <completed-initial.csv> \
  --provenance-json <initial-human-provenance.json>
# Wait until the trusted initial checkpoint is at least 14 elapsed days old.
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability gold-recode-release
# Optional non-human sidecar; this does not advance the human gate.
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability gold-ingest-provisional \
  --recode-csv <completed-provisional-recode.csv> \
  --provenance-json <provisional-provenance.json> \
  --raw-response <provisional-raw-response>
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability gold-ingest \
  --recode-csv <completed-delayed-recode.csv> \
  --provenance-json <recode-human-provenance.json>
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability validate-features \
  --decisions-dir <committed-feature-decisions>
```

Until those commands complete with qualifying human inputs,
`outcome_modeling_unlocked=false`. Code completion, fixture tests, or provisional LLM
evidence cannot change it.

## Gold sample

Candidate strata are the closed categorical inventory:

`hop_structure`, `exhaustivity_requirement`, `negative_conclusiveness`,
`referring_form_type`, `answer_locality`, and `recall_orientation`.

For a feature, exact model agreement produces the shared level; disagreement produces
`DISPUTED`. Missingness reasons are levels rather than silently dropped rows. Every item is
assigned to exactly one candidate cell: smallest realised candidate frequency first, then
feature name, then level, with `DISPUTED` sorting last within its feature. Each realised
stratum receives a simple random sample without replacement of
`n_h = min(12, N_h)`, and every selected item carries exact inclusion probability
`pi_j = n_h / N_h`.

The adjudication CSV contains local item keys, exact source text, applicable codebook fields,
and both model labels. It contains no outcome, trace, performance, grade, recommendation, or
tier field. `gold_sample.parquet` retains stable IDs, stratum assignments, inclusion
probabilities, and source/model provenance.

## Initial adjudication

The user is the initial human adjudicator. The completed CSV must preserve row and local-key
identity, use only codebook values, distinguish substantive missingness from blank input, and
carry a separate provenance record with:

- annotator kind and role;
- template bundle, completed adjudication, schema, codebook, and decision-ledger hashes;
- started/completed timestamps;
- an outcome-blindness declaration;
- optional non-sensitive notes.

The importer rejects changed source columns, unknown labels, duplicate or missing keys,
incomplete feature cells, outcome fields, and a provenance record that does not identify a
human initial adjudication.

## Delayed blind re-code

The re-code subset is selected independently within each realised gold stratum:

`min(n_h, max(2, ceil(0.20 * n_h)))`.

Selection uses the dedicated cryptographic `gold_recode_sampling` stream. Initial ingest
records a trusted system completion timestamp and persists the selected identities only in
the hidden mapping. `gold-recode-release` is refused until that checkpoint is at least
fourteen elapsed days old. The then-emitted bundle uses new local keys and a separately
shuffled order. It hides model labels, stratum labels, initial labels, initial comments, and
the mapping to initial local keys. It shows only source text and the codebook fields required
for re-coding.

`gold-ingest` accepts only a human re-code started after the trusted release timestamp and
completed no later than the ingest time. It verifies the hidden mapping and all locked
provenance hashes locally and persists human test-retest values only after both bundles pass.

## Provisional frontier-LLM re-code

A future frontier-LLM re-code may exercise the protocol after the same 14-day boundary, but it
is non-human provisional evidence. Its provenance records:

- provider, interface, exact model and version;
- prompt, schema, codebook, template bundle, completed response, and decision-ledger hashes;
- reasoning/temperature and other exposed parameters;
- session identifier, fresh-context declaration, and context-isolation description;
- start/completion timestamps, agent/run identifiers, usage, status, and raw-response hash.

Provisional rows remain physically and semantically separate from `human_recode_*` fields.
They cannot complete canonical `gold-ingest`, cannot satisfy human test-retest requirements,
and cannot appear in `validation_unlock.json` as qualifying evidence. The unresolved need for
independent human expert verification is persisted for later review.

`gold-ingest-provisional` copies the completed CSV, raw response and canonical provenance into
`gold/provisional/`, writes normalized `provisional_*` labels to a separate Parquet table, and
registers a verification artifact with `qualifies_as_human_recode=false` and
`can_unlock_outcome_modeling=false`. It never invokes an SDK or model. Human `gold-ingest`
may run whether the optional sidecar is pending or complete.

## Validation metrics

The formal three-status gate applies to:

`qdmr_step_count`, `hop_structure`, `exhaustivity_requirement`,
`negative_conclusiveness`, `referring_form_type`, `answer_locality`, and
`recall_orientation`.

Each model is evaluated separately against human gold using Horvitz–Thompson confusion totals
or weighted errors:

- binary: balanced accuracy, sensitivity, and specificity; baseline `0.5`;
- nominal multiclass: macro-F1; baseline is the majority-class predictor's realised
  HT-weighted macro-F1;
- numeric `qdmr_step_count`: `1 - MAE_model / MAE_naive`, where the naive prediction is the
  HT-weighted median human count; baseline `0`;
- secondary reliability: Krippendorff alpha at nominal, ordinal, or interval measurement
  level as appropriate.

Every metric and alpha interval uses 2,000 deterministic percentile-bootstrap replicates,
resampling independently with replacement within realised gold strata. Feature/model child
seeds are SHA-256 derived so execution order cannot change results. Both Claude and GPT must
pass each structural rule; the gate summary reports the weaker point result and wider
interval while retaining both dossiers.

Categorical dossiers retain raw and weighted confusion cells and per-class precision/recall
intervals; binary dossiers name sensitivity and specificity. Explicit model missingness on a
substantive human-gold row is an error, not an omitted row. Human-gold missingness is reported
descriptively but remains outside the substantive assigned metric. `qdmr_step_count` receives
weighted absolute-error summaries rather than a binned confusion table. Human test-retest
uses the same scale-matched metric and alpha with the inverse joint gold/re-code inclusion
probability. Observed error patterns and stable item examples are fixed in the dossier before
the material-failure decision.

Krippendorff alpha is calculated from a weighted coincidence matrix. A unit with fewer than
two valid ratings is omitted from both observed coincidences and marginals. Nominal,
frequency-aware ordinal and interval distances are explicit. Inverse inclusion weights form a
project-specific pseudo-population expansion; reference tests pin the equal-weight result and
hand calculations pin unequal-weight behavior.

## Exploratory P4 evidence

`validate-features` also emits separate descriptive dossiers for `presupposition_load`,
`cognitive_process_level`, `qdmr_operator_set`, `qdmr_normalized_question`, `demand_type`,
`specificity`, and structural `qdmr_applicability`. They use scale-matched HT metrics,
stratified intervals, available human test-retest evidence and stable examples. Operator sets
use mean Jaccard and exact-set agreement; normalized-question faithfulness uses exact match
and whitespace/case-normalized token F1. Undefined evidence is recorded rather than coerced.

These dossiers have no three-status field, no feature decision, no family or DSL mapping, and
no unlock contribution. They are written under `statistics/exploratory_p4_evidence/` with a
separate table and report.

The three statuses are `VALIDATED`, `VALIDATED_WITH_LIMITATIONS`, and `NOT_VALIDATED`.
`METRIC_UNDEFINED_DEGENERATE_GOLD` is a separate blocking state, not a fourth validation
status. Dossiers are disclosed in the frozen `annotation_shuffle` order. Each write-once
decision cites its dossier hash and contains a written material-failure assessment; the next
dossier is not disclosed until the prior decision is recorded.

## DSL derivation gate

The design-based surrogate-label extension is established only if all of these obligations
are documented and tested:

1. target estimand and full observed-data structure;
2. surrogate-error and missingness assumptions;
3. incorporation of unequal-probability human-gold sampling;
4. mapping to the frozen two-way clustered estimating equation;
5. nuisance estimation and cross-fitting protocol;
6. influence/score correction and two-way variance derivation;
7. synthetic calibration, bias, coverage, and failure-boundary evidence.

Any unmet obligation yields `DSL_NOT_ESTABLISHED`. Affected P4 families then use human-gold
rows for confirmatory inference; full-population surrogate fits are exploratory only. The
gate is settled without reading outcome results and cannot be overridden because a downstream
estimate looks preferable.

### Current derivation attempt

The current implementation records **`DSL_NOT_ESTABLISHED`**. The retained Egami et al.
result supplies a design-based surrogate-correction foundation for M-estimators/GMM, but the
current project record does not contain a complete influence-function and variance derivation
that jointly covers unequal-probability gold sampling, two annotator surrogates, the frozen
criterion-level estimating equation, and the three-term two-way clustered covariance. Nor is
there synthetic coverage evidence for that combined extension. Claiming the gate passed would
therefore exceed the retained source and the available derivation. The executable gate keeps
all seven obligations individually visible so a later proof can replace this fail-closed
outcome without changing the decision rule.

## Unlock conditions

`validation_unlock.json` authorizes outcome modelling only when all conditions hold:

- qualifying initial human gold has been ingested;
- qualifying delayed human re-code satisfies the 14-day minimum;
- every required feature dossier and decision is complete;
- neither model violates a structural validity rule for an admitted feature;
- no `METRIC_UNDEFINED_DEGENERATE_GOLD` supplementation requirement remains;
- the DSL outcome and any gold-only fallback are recorded;
- every input, decision, and output hash matches the Segment 5 lock.

Provisional LLM evidence is an explicit denial condition. During implementation-only
preparation, `gold-sample`, `gold-ingest-initial`, `gold-recode-release`, `gold-ingest`, and
optional `gold-ingest-provisional`, plus `validate-features`, remain pending and the unlock
remains false.

## Verification

Automated verification comprises unit/integration tests, branch coverage, Ruff, formatting,
ty, schema checks, deterministic hashes, and historical-root immutability. No Grok or other
model-based implementation audit is run. The user owns the later implementation review and
audit.

## Implementation handoff status

The Segment 5 software, schemas, decision ledger, operational documentation, fixture-driven
human/provisional workflow tests, weighted-metric tests, DSL fallback, and explicit unlock
authorization contract are implemented. The complete test tree passes the required
branch-coverage floor; Ruff, formatting, and ty pass for the package and its tests. Exact
current counts are recorded in
`reports/18-stage-5-gold-validation-remediation.md` after each remediation quality gate.

The freshly locked `simplemode-v1-segment5` root has been rematerialized through
`annotate-ingest` from the immutable Stage 4 handoff without an SDK/model call. It contains
1,154 normalized annotation rows for 577 items, 50 raw responses, 50 attempts, 50 envelopes,
and the six retained QDMR construct anomalies. `gold-sample`, `gold-ingest-initial`,
`gold-recode-release`, optional `gold-ingest-provisional`, `gold-ingest`, and
`validate-features` remain pending with zero attempts, and
`outcome_modeling_unlocked=false`. This is the intended implementation-complete,
data-gate-pending handoff.
