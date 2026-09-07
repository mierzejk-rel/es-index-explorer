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
  no lower-casing or whitespace normalization precedes parsing. Canonical leading or trailing
  whitespace therefore remains in `source_text` and contributes to `text_sha256`.
- `use_cases` is persisted as Parquet logical `list<string>`. Schema-aware pandas consumers use
  `dtype_backend="pyarrow"` or normalize iterable cell values at their input boundary; a
  default-backend NumPy array is an in-memory representation of the same logical list.
- A feature that does not apply to an item type is missing with an explicit reason. It is never
  represented by zero.

## 2. Parser authority

- Stanza 1.14.0 English UD is authoritative for tokenization, morphology, lemmas, dependency
  relations, and every P2 morphosyntactic feature. The distribution version is exact, and
  setup downloads only the frozen processor/package mapping plus declared dependencies.
- spaCy 3.8.14 with `en_core_web_sm` 3.8.0 is authoritative only for named-entity spans.
- No parser vote or reconciliation is performed. Model packages and selected files are hashed
  in `parser_resource_manifest.json`. Before loading, the installed spaCy model tree is checked
  against the frozen sorted-path/SHA-256/size aggregate in its static resource manifest; a
  same-version modified, missing, or extra model file is rejected.

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
3. a verbal or auxiliary `conj` dependent with `VerbForm=Fin` or a `Mood` feature; or
4. one nonverbal predicate governor per sentence that has at least one `AUX` dependent with
   base relation `cop` and `VerbForm=Fin` or a `Mood` feature, unless the governor is already
   represented by rules 1-3.

The base relation is the part before a UD subtype colon. For rule 4, the copular clause's
effective base relation is the predicate governor's relation. Multiple finite `cop` dependents
of the same governor represent one clause, and a finite copula never duplicates a verbal
predicate already counted by rules 1-3.

- `clause_count` (`C`): number of clause heads.
- `subordinate_clause_ratio` (`DC/C`): clause heads whose base relation is `advcl`, `acl`,
  `ccomp`, `xcomp`, or `csubj`, divided by `C`.
- A **complex nominal** is a unique `NOUN`, `PROPN`, or `PRON` head with at least one dependent
  whose base relation is `amod`, `appos`, `compound`, `nmod`, `nummod`, or `acl`.
- `complex_nominals_per_clause` (`CN/C`): complex nominal count divided by `C`.
- `coordination_count`: number of non-punctuation `conj` dependents. This is a deterministic
  exploratory Dimension E feature and is excluded from confirmatory restrictions.

Both ratios are missing with `no_clause` when `C=0`. Under the finite-copula rule, this state
is reserved for genuine verbless fragments rather than finite copular clauses. The definitions
are project-specific UD operationalizations of the L2SCA constructs, not claims that a
dependency tree reproduces constituency-based L2SCA output exactly.

### 4.3 Entity and temporal metrics

- `named_entity_count`: number of non-overlapping spaCy entity spans, regardless of label.
- `temporal_expression_present`: true when at least one spaCy entity has label `DATE` or
  `TIME`; otherwise false.

Email addresses and aliases are not promoted to named entities by regex. Their semantic form
belongs to P4 `referring_form_type`.

## 5. P3 deterministic clause type

`clause_type` applies only to question items. It has four nominal levels and is assigned from
the first logical sentence using this precedence:

1. `directive_imperative`: the matrix root has `Mood=Imp`, or the narrow parser-repair case
   applies where initial lemma `list` is tagged `NOUN` with base relation `compound` and its
   governor is a subjectless nominal sentence root. Matrix imperative form takes precedence
   over interrogative words inside its complement.
2. `open_interrogative`: a `PronType=Int` token is matrix-level, meaning its dependency path to
   the matrix root does not cross `advcl`, `acl`, `ccomp`, `xcomp`, or `csubj`; coordinated
   matrix interrogatives remain matrix-level. An initial wh-form from `who`, `whom`, `whose`,
   `what`, `which`, `when`, `where`, `why`, or `how` is the deterministic fallback when the
   parser omits `PronType=Int`.
3. `closed_interrogative`: rules 1-2 did not match and either the logical source sentence ends
   in `?`, or it has matrix subject-auxiliary inversion. The inversion rule requires the first
   non-punctuation word to be `AUX`, attached to the matrix predicate as `aux` or `cop` (or
   itself the matrix root), with a nominal subject, clausal subject, or expletive dependent of
   that matrix predicate.
4. `directive_imperative`: rules 1-3 did not match and the matrix root is `VERB`/`AUX` with no
   nominal subject, clausal subject, or expletive dependent.
5. `declarative_request`: every remaining request.

The reference level is `open_interrogative`; therefore F7 has `k=4` and `q=2*(k-1)=6`.
Expectation items receive `not_applicable_to_expectation`.

Examples:

- “Which documents support the claim?” → `open_interrogative`
- “Did Belford send an email?” → `closed_interrogative`
- “List the supporting documents.” → `directive_imperative`
- “I would like a summary of the response.” → `declarative_request`

The first logical sentence is the first non-empty Stanza sentence plus immediately adjacent
Stanza sentence fragments when their boundary falls inside an email address matching
`[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}` in the exact source
text. This repairs Stanza splits such as `eugene.` / `Belford@ellingson.com` for P3 only; it
does not alter source text, sentence/token archives, or P2 features. For a genuine mixed
multi-sentence request, only the first logical sentence determines the label. Fragments ending
in `?` are closed interrogatives. Parser failure is not replaced by an unrestricted text-only
guess.

The four-category inventory is anchored in the retained clause-type literature. The email
boundary repair, finite-copula mapping, matrix dependency-path rule, subject-auxiliary
inversion detector, precedence, and narrow `list` repair are deterministic project-specific
operationalizations; the retained sources do not prescribe these algorithms.

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
- `qdmr_step_count`: positive integer when applicable; missing when not applicable. The wide
  annotation artifact persists this as a nullable integer, never a floating-point count.
- `qdmr_operator_set`: when applicable, a sorted, duplicate-free subset of the closed project
  normalization vocabulary below. Labels are uppercase and no unlisted label is accepted:
  `SELECT`, `FILTER`, `PROJECT`, `AGGREGATE`, `GROUP`, `SUPERLATIVE`, `COMPARATIVE`, `UNION`,
  `INTERSECTION`, `DISCARD`, `SORT`, `BOOLEAN`, `ARITHMETIC`.
- `qdmr_normalized_question`: required only for `applicable_after_normalisation`.
- `hop_structure`: `atomic`, `bridge`, `comparison`, or `intersection`.

The thirteen operator labels are the project’s frozen normalization vocabulary for QDMR
annotation. They are not a claim that every upstream BREAK implementation emits these exact
strings. Wolfson et al. (2020) supports the step-wise decomposition construct; the closed
string vocabulary and normalization behavior are project conventions.

Each QDMR step has one operator, so an applicable annotation has
`qdmr_step_count >= len(qdmr_operator_set)`. A violation is a construct-QA anomaly: retain the
immutable raw annotation, persist the attributable anomaly for human validity review, and do
not silently repair it, convert it to zero, or treat it as substantive missingness.

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
- `cognitive_process_level`: ordinal exploratory labels
  `remember < understand < apply < analyze < evaluate < create`, following the source’s
  assumed cognitive-complexity continuum.

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
- `exhaustivity_requirement` is nominal; its three labels are not an ordered scale.
- `cognitive_process_level` is ordinal and uses quadratically weighted kappa; the §13.2a
  ordinal-feature branch is active for this feature.
- Counts are numeric, not categorical.
- Every other categorical label in this closed taxonomy is binary or nominal.

Closed inventories remain fixed even when a level is unobserved in this corpus. An observed
count of zero is coverage evidence, not permission to remove the level, force an annotation,
or treat the vocabulary as invalid.

## 9. Missingness and edge cases

- Empty source text or parser failure is blocking for deterministic extraction.
- Stanza, spaCy, empty-parse, and deterministic-rule failures are separate blocking extraction
  stages. They are reported by component and `item_id` without source text; no failure is
  replaced by zero, missing, or a substantive label.
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
`features` command. `presupposition_load`, `demand_type`, and `specificity` are exploratory P4
measurements owned by the later outcome-blind annotation and robustness workflow.
`entity_density` is withdrawn: no numerator, denominator, scale, missingness rule, or
literature-supported operational definition was frozen, and it enters no confirmatory family.
The defined `named_entity_count` measure remains available. No unpinned lexical model or norm
is introduced here.

## 11. Hand-authored fixtures

The deterministic test set contains imperative, open and closed interrogative, declarative,
coordination, subordination, complex-nominal, temporal-expression, alias, email-address, and
fragment cases. Expected values follow the rules above and do not use empirical outcomes.
