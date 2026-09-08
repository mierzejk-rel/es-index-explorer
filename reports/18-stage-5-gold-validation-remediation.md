# Stage 5 gold-validation remediation evidence

Remediation date: 2026-09-08  
Scope: audit findings S5-F1 through S5-F8  
Scientific data-gate status: pending; outcome modelling remains locked

## P0/P1 closure matrix

| Finding | Remediation evidence | Verification | Status |
| --- | --- | --- | --- |
| S5-F1 — feature-long gold ingest | Initial rows use `(local_item_id, feature)` identity; delayed mapping attaches on the same composite key; conditional and joint re-code probabilities are persisted | The real 577-item frame produced 189 sampled items and 1,680 feature rows; all 1,680 round-tripped with unique composite keys in an ephemeral workspace | PASS |
| S5-F2 — incomplete validation dossier | Schema-v2 typed dossiers retain separate model summaries, raw/HT confusion evidence or weighted numeric errors, class intervals, sensitivity/specificity, corpus prevalence, weighted alpha, human test-retest, deterministic failure patterns, weaker/wider summaries, and input hashes | Unit tests validate every feature dossier and execute `run_validate_features` without mocking the dossier builder | PASS |
| S5-F3 — model missingness dropped | Substantive-gold/model-missingness cells remain forecast-only error columns; unknown and raw-null predictions fail closed; human-gold missingness remains in the descriptive matrix but outside the substantive assigned metric | Adversarial tests verify missingness lowers recall/balanced accuracy and appears in raw/HT cells and failure examples | PASS |
| S5-F4 — declared-only delay | Initial human labels receive a trusted workflow checkpoint; the blind re-code is withheld until 14 elapsed days; final ingest requires annotation after release and no future-dated completion; provenance binds completed CSVs and locked schema/codebook/decision hashes | Tests reject release one second early, backdated initial work, and future-dated re-code completion; exact-boundary release passes | PASS |

## P2 closure matrix

| Finding | Remediation evidence | Verification | Status |
| --- | --- | --- | --- |
| S5-F5 — weak/mocked coverage | Added real sequential dossier execution, deterministic replay, detached-unlock and gold-step idempotency checks, hash/outcome/mapping/timing adversarial cases, and a committed canonical-frame regression | Read-only canonical regression reproduces 189 items, 18 strata and 1,680 unique feature rows, then round-trips all rows in an ephemeral initial ingest | PASS |
| S5-F6 — provisional persistence absent | Added optional `gold-ingest-provisional`; normalized values, completed CSV, raw response, provenance, verification and unresolved-human requirement are physically separate from human gold | Integration proves provisional completion leaves human `gold-ingest` pending, cannot start validation or fitting, never creates human fields and never unlocks modelling | PASS |
| S5-F7 — exploratory evidence absent | Added typed non-gating dossiers/table/report for all seven bundled exploratory fields with scale-matched HT metrics, intervals, available human test-retest, operator evidence and normalized-question faithfulness examples | Tests validate all seven dossiers and prove no validation status, family, DSL or unlock contribution | PASS |
| S5-F8 — alpha approximation | Replaced pair-list alpha with an explicit survey-weighted coincidence matrix, valid-unit missingness semantics, and nominal/frequency-aware ordinal/interval distances | Equal-weight reference gives `4/9`, missing-rater reference gives `0`, and hand-calculated HT expansion gives `0.125`; measurement levels and bootstrap determinism are tested | PASS |

## Quality evidence

- Complete unit and question-analysis integration suite: 430 passed.
- Branch coverage: 89% overall; `gold.py` 90%; `validation.py` 92%.
- Ruff, Ruff formatting, ty, IDE diagnostics and `git diff --check`: pass.
- No model-based or Grok post-remediation audit was run.

## Canonical-root evidence

`simplemode-v1-segment5` was rematerialized through `annotate-ingest` from the immutable
`simplemode-v1-postremediation-final` handoff using resume-only annotation migration. It
contains 1,154 normalized rows for 577 items. `gold-sample`, `gold-ingest-initial`,
`gold-recode-release`, optional `gold-ingest-provisional`, human `gold-ingest`, and
`validate-features` are pending with zero attempts and `outcome_modeling_unlocked=false`.

Before/after tree fingerprints for `simplemode-v1-postreview-p3` and
`simplemode-v1-postremediation-final` were identical. No `simplemode-v1-postreview-p4` root
exists. No live gold, provisional or validation artifact was generated in the canonical root.

## Residual scientific scope

No live human adjudication, delayed human re-code, final feature status, DSL proof, P4
release, or Segment 6 outcome modelling is performed or claimed by software remediation.
The HT coincidence extension and stratified percentile intervals remain explicit
project-specific methodological conventions. Genuine independent human verification remains
outstanding.
