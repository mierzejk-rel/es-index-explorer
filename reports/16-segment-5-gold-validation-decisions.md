# Segment 5 gold-validation decision ledger

Decision date: 2026-09-08  
Scope: Segment 5 — Gold Sample, Human Workflow, and Feature-Validation Gate  
Status: frozen before Segment 5 implementation

This ledger records the user decisions that complete the Segment 5 implementation contract.
It is an outcome-blind project record, not a literature review and not an implementation
audit. Each entry distinguishes retained source support from a project-specific methodological
choice or implementation convention.

## 1. Segment title and scope

**Question.** The parent plan defines TODOs 5.1–5.5 as “Gold sample, human workflow and
feature-validation gate”; “Outcome-blind P4 annotation” is Segment 4 and is now complete.
Should the dedicated plan use the parent plan’s Segment 5 scope/title while preserving outcome
blindness throughout?

**Options presented.**

- Use Segment 5 gold/validation scope (recommended).
- Keep “Outcome-blind P4 annotation” as the title, but cover TODOs 5.1–5.5.
- Use a different Segment 5 scope.

**Decision.** “Outcome-blind P4 annotation” was a naming carryover. Use **Segment 5 — Gold
Sample, Human Workflow, and Feature-Validation Gate**.

**Classification.** Implementation-plan correction.  
**Consequence.** Segment 5 implements research-plan §§13.2a–13.4 and CLI commands
`gold-sample`, `gold-ingest`, and `validate-features`; it does not repeat Segment 4 annotation.

## 2. Specification-completion precondition

**Question.** How should the dedicated Segment 5 plan handle unresolved methodological
contracts?

**Options presented.**

- Add a specification-completion precondition before TODO 5.1 (recommended).
- Implement only what is currently specified and fail closed on unresolved parts.
- The user will provide the missing methodological decisions immediately.

**Decision.** Add a **specification-completion precondition** before TODO 5.1.

**Classification.** Project-specific process decision.  
**Consequence.** The research plan is amended before implementation to freeze the validation
inventory, count metric, two-model gate, re-code design, DSL proof obligations, and unlock
semantics.

## 3. Human-editable bundle format

**Question.** Which editable format should `gold-sample` emit for manual adjudication and
delayed re-code?

**Options presented.**

- CSV bundles with codebook/value sheets (recommended).
- Strict JSONL bundles.
- Excel workbook.

**Decision.** Emit **CSV bundles with codebook/value guides**.

**Classification.** Implementation convention.  
**Consequence.** CSV is the editable exchange format; Parquet and canonical JSON remain the
typed, hash-registered analysis artifacts.

## 4. Delayed blind re-code interval and provisional model use

**Question.** What minimum delay should the workflow enforce before accepting a blind re-code?

**Options presented.**

- 14 days (recommended).
- 7 days.
- No fixed duration; require explicit user confirmation.

**Decision.** Enforce a **minimum 14-day delay**.

The current lack of an available human expert permits a frontier LLM to perform a future
re-code only as a **provisional annotation exercise**. It is not equivalent to independent
human re-coding. The protocol records provider/interface, exact model/version, parameters,
prompt and codebook versions, session/context isolation, timestamps, agent/run identifiers,
usage, and source/response hashes. The unresolved need for independent human expert
verification remains explicit.

**Classification.** Project-specific methodological choice.  
**Consequence.** Human re-code acceptance checks elapsed time. Provisional LLM evidence cannot
populate human test-retest fields or authorize outcome modelling.

## 5. Initial adjudicator

**Question.** Who should produce the initial gold adjudication used as the primary validity
criterion?

**Options presented.**

- The user performs the initial human adjudication (recommended).
- A frontier LLM provisionally performs the initial adjudication.
- A separate human expert performs it.

**Decision.** The **user performs the initial human adjudication**.

**Classification.** Project-specific staffing decision within the retained human-gold design.  
**Consequence.** Initial labels may be designated human gold. The single-coder and expertise
limitations remain disclosed; a model-generated initial pass cannot substitute for them.

## 6. Implementation completion versus live data-gate closure

**Question.** May a provisional frontier-LLM delayed re-code unlock outcome modelling before
independent human re-code or expert verification?

**Options presented.**

- No; record provisional results but keep outcome modelling locked (recommended).
- Allow a clearly provisional unlock.
- Complete code but defer running and closing the data gate.

**Decision.** Complete the implementation and tests, but **defer running and closing the live
data gate**. Provisional evidence may exercise the workflow but
`outcome_modeling_unlocked` remains `false` until the required human-gold validation gate is
satisfied.

**Classification.** Project-specific release decision consistent with research-plan §19.  
**Consequence.** Implementation TODO completion and analysis-run unlock are separate states.
The Segment 5 root stops after the verified Stage 4 handoff until the user starts
`gold-sample`.

## 7. Formal feature-validation inventory

**Question.** Which P4 fields should receive the formal three-status validation gate?

**Options presented.**

- Confirmatory P4 predictors only (recommended).
- All current P4 annotation fields.
- A custom inventory.

**Decision.** Gate the **confirmatory P4 predictors only**:

- `qdmr_step_count`;
- `hop_structure`;
- `exhaustivity_requirement`;
- `negative_conclusiveness`;
- `referring_form_type`;
- `answer_locality`;
- `recall_orientation`.

Exploratory P4 fields receive descriptive validity/reliability evidence. The
`qdmr_normalized_question` value remains faithfulness evidence rather than a scored feature.

**Classification.** Project-specific completion of research-plan §§9, 11, and 13.2a.  
**Consequence.** A validation status gates only a family that actually requires that P4
predictor.

## 8. Candidate stratum inventory

**Question.** Which fields should define candidate `(feature, level)` strata before the
rarest-cell strict-partition assignment?

**Options presented.**

- Six confirmatory categorical P4 fields (recommended).
- All categorical P4 fields.
- A custom inventory.

**Decision.** Use these six categorical fields:

- `hop_structure`;
- `exhaustivity_requirement`;
- `negative_conclusiveness`;
- `referring_form_type`;
- `answer_locality`;
- `recall_orientation`.

Explicit missingness reasons remain candidate levels. Numeric `qdmr_step_count` is human-coded
for sampled questions but does not define a categorical stratum.

**Classification.** Project-specific completion of research-plan §13.3.  
**Consequence.** The strict partition and inclusion probabilities are determined before
sampling from this closed inventory.

## 9. Two-model disagreement

**Question.** How should the preliminary label be represented when the two Stage 4 annotators
disagree?

**Options presented.**

- Use `DISPUTED`; never prefer either model (recommended).
- Use a deterministic model priority.
- Require human adjudication of every disagreement.

**Decision.** Use **`DISPUTED`**, never a model priority.

**Classification.** Project-specific methodological choice already motivated by
research-plan §13.3.  
**Consequence.** Agreement yields the shared label; disagreement yields a separate stratum
level while both model labels remain immutable provenance.

## 10. Numeric `qdmr_step_count` validity metric

**Question.** Which validity metric should be frozen for `qdmr_step_count`?

**Options presented.**

- HT-weighted MAE skill versus a constant median (recommended).
- HT-weighted exact agreement.
- Do not validate the numeric feature.

**Decision.** Use **Horvitz–Thompson-weighted MAE skill** against the HT-weighted constant
median, with secondary interval-scale Krippendorff alpha.

The skill is `1 - MAE_model / MAE_naive`; its matched naive baseline is `0`, and larger values
are better.

**Classification.** Project-specific methodological choice.  
**Consequence.** The metric respects numeric distance while retaining the same
interval-versus-baseline gate form as the categorical metrics.

## 11. Delayed re-code subset

**Question.** How should the delayed blind re-code subset be selected from the initial gold
sample?

**Options presented.**

- Deterministic stratified 20%, minimum two per stratum (recommended).
- Re-code the complete gold sample.
- Another deterministic fraction or size.

**Decision.** Select a **deterministic stratified 20%**, at least two items per nonempty
realised stratum, or all items when fewer than two are available, using a dedicated
deterministic seed stream. The 14-day minimum remains mandatory.

**Classification.** Project-specific methodological choice.  
**Consequence.** Human test-retest evidence covers every realised stratum without duplicating
the complete adjudication workload.

## 12. DSL derivation gate

**Question.** How should the implementation handle two-way-cluster-compatible design-based
surrogate correction?

**Options presented.**

- Attempt the derivation against explicit proof obligations; otherwise fail closed
  (recommended).
- Declare the DSL gate failed without attempting the extension.
- Implement the extension as provisionally accepted.

**Decision.** **Attempt the derivation against explicit proof obligations; otherwise fail
closed.**

**Classification.** Project-specific formal gate grounded in the retained Egami et al.
surrogate-inference source but extending beyond its directly supported setting.  
**Consequence.** Unmet estimand, sampling, score, nuisance, influence/variance, or simulation
obligations yield `DSL_NOT_ESTABLISHED` and gold-only confirmatory handling.

## 13. Combining two annotators in the validation gate

**Question.** How should the three-status gate combine the two annotators’ human-gold results?

**Options presented.**

- Require both annotators to pass the structural validity rules (recommended).
- Use a pooled model-versus-human confusion table.
- Select one frozen annotator before reading gold.

**Decision.** **Both annotators must pass.** Compute HT-weighted metrics and intervals
separately for Claude and GPT, retain both complete dossiers, and report the weaker/wider
result in the gate summary.

**Classification.** Project-specific conservative gate decision.  
**Consequence.** No annotator can be selected after viewing gold, and pooled pseudo-replication
of the single human label is avoided.

## 14. Segment 5 root lineage

**Question.** Which analysis-root strategy should Segment 5 use?

**Options presented.**

- Create `simplemode-v1-segment5` from verified Stage 4 lineage (recommended).
- Create `simplemode-v1-postvalidation`.
- Modify the Stage 4 final root in place.

**Decision.** Create **`simplemode-v1-segment5`** from verified Stage 4 lineage.

**Classification.** Reproducibility and versioning convention.  
**Consequence.** `simplemode-v1-postremediation-final` remains the immutable Stage 4 handoff.
The Segment 5 root has fresh implementation/specification fingerprints and receives Stage 4
provenance only through verified resume-only migration.

## 15. Audit ownership

**Instruction.** Do not run Grok 4.6 or any other model-based implementation audit or review.
The user will review the implementation and initiate any later audit.

**Classification.** Process constraint.  
**Consequence.** Automated verification is limited to deterministic tests, lint, formatting,
type checks, schema/hash checks, and artifact reproducibility. A future provisional LLM
re-code is annotation evidence, not an implementation audit.
