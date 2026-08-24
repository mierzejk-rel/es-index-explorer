# Simple Mode question-linguistic codebook

Status: frozen before outcome access  
Scope: 263 question variants and 314 expectation descriptions  
Research specification: `reports/13-simple-mode-analysis-research-plan.md`

This codebook fixes the labels and deterministic rules used by the Simple Mode analysis. It
does not use experimental outcomes. P1-P3 are produced by `features`; P4 is defined here for
the outcome-blind annotation workflow in Segment 4.

## 1. Item grains

- A **question item** is one row of `variant_catalogue.parquet`, keyed by `variant_id`.
- An **expectation item** is one row of `expectation_catalogue.parquet`, keyed by
  `expectation_id`.
- `item_id` is the existing variant or expectation identifier. Text is hashed exactly as stored;
  no lower-casing or whitespace normalization precedes parsing.
- A feature that does not apply to an item type is missing with an explicit reason. It is never
  represented by zero.

## 2. Parser authority

- Stanza 1.14.0 English UD is authoritative for tokenization, morphology, lemmas, dependency
  relations, and every P2 morphosyntactic feature.
- spaCy `en_core_web_sm` 3.8.0 is authoritative only for named-entity spans.
- No parser vote or reconciliation is performed. Model packages and selected files are hashed
  in `parser_resource_manifest.json`.

## 3. P1 structural features

These fields are copied from the Segment 2 catalogues:

- `expectation_count`: number of stable expectations in the rubric.
- `variant_count`: number of `[[input]]` entries in the rubric.
- `use_cases`: normalized, order-preserving list from `meta.use_case`.
- `expectation_document_count`: number of unique exact-string document IDs. It applies only to
  expectation items; question items receive `not_applicable_to_question`.

## 4. P2 parser-derived features

Punctuation tokens (`UPOS=PUNCT`) are excluded from token and dependency-length counts.
Multi-word tokens are represented by their Stanza words, not by the enclosing token range.
Metrics are summed across sentences unless stated otherwise.

### 4.1 Surface and dependency metrics

- `token_count`: count of non-punctuation Stanza words.
- `dependency_tree_depth`: maximum number of dependency arcs from a word to its sentence root;
  a root has depth 0. Cycles or missing heads are parse errors, not depth values.
- `mean_dependency_length`: arithmetic mean of `abs(word_id - head_id)` for non-root,
  non-punctuation words. It is missing with `no_dependencies` when the denominator is zero.

### 4.2 Clause and L2SCA-derived metrics

A **clause head** is:

1. a verbal or auxiliary sentence root; or
2. a verbal or auxiliary dependent with base relation `advcl`, `acl`, `ccomp`, `xcomp`,
   `csubj`, or `parataxis`; or
3. a verbal or auxiliary `conj` dependent with `VerbForm=Fin` or a `Mood` feature.

The base relation is the part before a UD subtype colon.

- `clause_count` (`C`): number of clause heads.
- `subordinate_clause_ratio` (`DC/C`): clause heads whose base relation is `advcl`, `acl`,
  `ccomp`, `xcomp`, or `csubj`, divided by `C`.
- A **complex nominal** is a unique `NOUN`, `PROPN`, or `PRON` head with at least one dependent
  whose base relation is `amod`, `appos`, `compound`, `nmod`, `nummod`, or `acl`.
- `complex_nominals_per_clause` (`CN/C`): complex nominal count divided by `C`.
- `coordination_count`: number of non-punctuation `conj` dependents. This is a deterministic
  exploratory Dimension E feature and is excluded from confirmatory restrictions.

Both ratios are missing with `no_clause` when `C=0`. This occurs legitimately for fragments.
The definitions are UD operationalizations of the L2SCA constructs, not claims that a
dependency tree reproduces constituency-based L2SCA output exactly.

### 4.3 Entity and temporal metrics

- `named_entity_count`: number of non-overlapping spaCy entity spans, regardless of label.
- `temporal_expression_present`: true when at least one spaCy entity has label `DATE` or
  `TIME`; otherwise false.

Email addresses and aliases are not promoted to named entities by regex. Their semantic form
belongs to P4 `referring_form_type`.

## 5. P3 deterministic clause type

`clause_type` applies only to question items. It has four nominal levels and is assigned from
the first non-empty Stanza sentence using this precedence:

1. `open_interrogative`: the sentence contains `PronType=Int`, or begins with a wh-form from
   `who`, `whom`, `whose`, `what`, `which`, `when`, `where`, `why`, or `how`.
2. `closed_interrogative`: the source sentence ends in `?` and rule 1 did not match.
3. `directive_imperative`: the root has `Mood=Imp`, or is `VERB`/`AUX` with no nominal subject,
   clausal subject, or expletive dependent.
4. `declarative_request`: every remaining request.

The reference level is `open_interrogative`; therefore F7 has `k=4` and `q=2*(k-1)=6`.
Expectation items receive `not_applicable_to_expectation`.

Examples:

- “Which documents support the claim?” → `open_interrogative`
- “Did Belford send an email?” → `closed_interrogative`
- “List the supporting documents.” → `directive_imperative`
- “I would like a summary of the response.” → `declarative_request`

For a mixed multi-sentence request, only the first non-empty sentence determines the label.
Fragments ending in `?` are closed interrogatives. Parser failure is not replaced by a
text-only guess.

## 6. P4 question annotation contract

These labels are frozen here but are not generated by Segment 3.

### 6.1 Answerhood

- `exhaustivity_requirement`: `mention_some`, `weakly_exhaustive`, or `mention_all`.
  - `mention_some`: one valid instance or fact answers the request.
  - `weakly_exhaustive`: all salient answers in the stated scope are expected, without a
    requirement to prove that no answer is omitted.
  - `mention_all`: the wording or task requires exhaustive enumeration or a negative conclusion.
- `negative_conclusiveness`: binary. True only when an adequate negative answer must establish
  absence in the stated scope.
- `presupposition_load`: binary exploratory label. True when the wording takes a disputed
  entity, event, relationship, or proposition for granted rather than asking whether it exists.

### 6.2 Retrieval decomposition

- `qdmr_applicability`: `applicable`, `applicable_after_normalisation`, or `not_applicable`.
- `qdmr_step_count`: positive integer when applicable; missing when not applicable.
- `qdmr_operator_set`: sorted unique QDMR operator labels.
- `qdmr_normalized_question`: required only for `applicable_after_normalisation`.
- `hop_structure`: `atomic`, `bridge`, `comparison`, or `intersection`.

Direct interrogatives are normally applicable. A directive is
`applicable_after_normalisation` only when it has a faithful interrogative paraphrase;
otherwise it is `not_applicable`. Inapplicability is missing, never zero.

### 6.3 Reference form

`referring_form_type` has three nominal levels:

- `full_name_form`: conventional personal or organization name, including surname-only and
  title-plus-surname forms.
- `alias_or_handle`: pseudonym, nickname, screen name, or explicit alias.
- `email_address`: a raw email address identifies at least one focal referent.

When several forms occur, use precedence `email_address` > `alias_or_handle` >
`full_name_form`. The reference level is `full_name_form`; F5 has `k=3`, `q=4`.
Generic questions without a focal named referent are missing with `no_focal_referent`.

### 6.4 Task framing

- `recall_orientation`: `precision_oriented` or `recall_oriented`. Known-item or single-fact
  requests are precision-oriented; requests to find, list, review, or summarize all relevant
  material are recall-oriented.
- `cognitive_process_level`: nominal exploratory labels `remember`, `understand`, `apply`,
  `analyze`, `evaluate`, or `create`. No ordinal inference is attached to these labels.

## 7. P4 expectation annotation contract

- `demand_type`: `verbatim_citation`, `entity_identification`, `relational_claim`,
  `temporal_ordering`, `quantification`, or `evaluative_synthesis`.
- `specificity`: binary `specific` or `broad`. A specific expectation names the fact,
  entity, relation, event, or value required; a broad expectation leaves the acceptable
  content materially open.
- `answer_locality`: `single_passage` when one coherent passage can establish the expectation;
  `cross_document_aggregation` only when evidence from multiple documents must be combined.
  An expectation requiring multiple passages from one document is missing with
  `not_classifiable_binary_locality`: it is neither silently folded into the cross-document
  estimand nor treated as single-passage. The reference is `single_passage`; F8 has `k=2`,
  `q=2` among classified expectations.

Document IDs do not determine locality: they are authoring metadata and may list redundant
sources for one fact.

## 8. Measurement scales

- Binary labels use balanced accuracy in the later validation gate.
- Multi-class labels use macro-F1.
- Counts are numeric, not categorical.
- Every categorical label in this closed taxonomy is nominal. The §13.2a ordinal-feature
  branch is therefore inert.

## 9. Missingness and edge cases

- Empty source text or parser failure is blocking for deterministic extraction.
- `no_dependencies`, `no_clause`, `not_applicable_to_question`,
  `not_applicable_to_expectation`, `not_applicable`, and `no_focal_referent` are distinct
  reasons.
- A computed zero is retained only when the construct applies and the observed count is zero.
- Parser/model provenance and source-text SHA-256 accompany every deterministic row.
- Annotation ambiguity is represented by the later annotators' labels and disagreement, not
  by adding unlisted categories.

## 10. Closed exploratory dimensions outside Segment 3 extraction

Dimension F lexical frequency, domain-term density, surprisal, and LingFeat measures remain
exploratory. They are not P1, P2, or P3 in §9 and are not emitted by the Segment 3
`features` command. `entity_density`, `presupposition_load`, `demand_type`, and `specificity`
are likewise exploratory P4 measurements and remain owned by the later outcome-blind
annotation and robustness workflow. No unpinned lexical model or norm is introduced here.

## 11. Hand-authored fixtures

The deterministic test set contains imperative, open and closed interrogative, declarative,
coordination, subordination, complex-nominal, temporal-expression, alias, email-address, and
fragment cases. Expected values follow the rules above and do not use empirical outcomes.
