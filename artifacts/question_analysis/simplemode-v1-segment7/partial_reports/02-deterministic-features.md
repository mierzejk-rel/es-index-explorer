# Partial report 02 — Deterministic features

Feature verification: **PASS**.

## Population and provenance

- Questions: 263.
- Expectation descriptions: 314.
- Total items: 577.
- Stanza: 1.14.0 (8877 archived words).
- spaCy: 3.8.14; `en_core_web_sm` 3.8.0 (NER only).
- Parser manifest SHA-256: `b47153e46a4031938a98f417fdff14c224a973ec02026632bd327030e48011e8`.
- Outcome columns loaded: none.

## Coverage and missingness

- `expectation_document_count`: 263 missing.
- `mean_dependency_length`: 0 missing.
- `subordinate_clause_ratio`: 54 missing.
- `complex_nominals_per_clause`: 54 missing.
- `clause_type`: 314 missing.

## Deterministic distributions

- `expectation_count`: n=577, min=1.0000, median=7.0000, max=21.0000.
- `variant_count`: n=577, min=2.0000, median=4.0000, max=7.0000.
- `expectation_document_count`: n=314, min=0.0000, median=1.0000, max=79.0000.
- `token_count`: n=577, min=2.0000, median=12.0000, max=71.0000.
- `dependency_tree_depth`: n=577, min=1.0000, median=4.0000, max=11.0000.
- `mean_dependency_length`: n=577, min=1.0000, median=2.3000, max=4.9355.
- `clause_count`: n=577, min=0.0000, median=1.0000, max=12.0000.
- `subordinate_clause_ratio`: n=523, min=0.0000, median=0.5000, max=1.0000.
- `complex_nominals_per_clause`: n=523, min=0.0000, median=1.3333, max=9.0000.
- `coordination_count`: n=577, min=0.0000, median=0.0000, max=9.0000.
- `named_entity_count`: n=577, min=0.0000, median=2.0000, max=10.0000.

## Clause type

- `closed_interrogative`: 49.
- `declarative_request`: 1.
- `directive_imperative`: 70.
- `open_interrogative`: 143.

`coordination_count` is retained as deterministic exploratory P2 and is not part of a confirmatory restriction.

No feature-outcome association was computed.
