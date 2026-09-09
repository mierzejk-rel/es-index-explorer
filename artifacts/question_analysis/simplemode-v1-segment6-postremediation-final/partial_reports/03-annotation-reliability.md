# Partial report 03 — Outcome-blind annotation reliability

Annotation ingestion: **PASS**.
- Normalized annotation rows: 1154.
- Unique items: 577.
- Agreement rows: 3644.
- Exact agreement rate: 0.7656421514818881.

## Preliminary feature agreement

- `answer_locality`: n=314, exact=0.876, mean scale-aware similarity=0.876.
- `cognitive_process_level`: n=263, exact=0.696, mean scale-aware similarity=0.910.
- `demand_type`: n=314, exact=0.780, mean scale-aware similarity=0.780.
- `exhaustivity_requirement`: n=263, exact=0.791, mean scale-aware similarity=0.791.
- `hop_structure`: n=263, exact=0.544, mean scale-aware similarity=0.544.
- `negative_conclusiveness`: n=263, exact=0.817, mean scale-aware similarity=0.817.
- `presupposition_load`: n=263, exact=0.551, mean scale-aware similarity=0.551.
- `qdmr_applicability`: n=263, exact=0.996, mean scale-aware similarity=0.996.
- `qdmr_normalized_question`: n=72, exact=0.514, mean scale-aware similarity=0.514.
- `qdmr_operator_set`: n=263, exact=0.544, mean scale-aware similarity=0.827.
- `qdmr_step_count`: n=263, exact=0.555, mean scale-aware similarity=0.766.
- `recall_orientation`: n=263, exact=0.882, mean scale-aware similarity=0.882.
- `referring_form_type`: n=263, exact=0.973, mean scale-aware similarity=0.973.
- `specificity`: n=314, exact=0.955, mean scale-aware similarity=0.955.

## QDMR construct QA

- `qdmr_applicability`: applicable=383, applicable_after_normalisation=143, not_applicable=0.
- Construct anomalies requiring later validity review: 6 ({"QDMR_OPERATOR_COUNT_EXCEEDS_STEP_COUNT": 6}).
- Unobserved closed-inventory QDMR operators: ["ARITHMETIC"].
- Unobserved cognitive-process levels: ["apply"].
- Construct anomalies do not block schema-valid ingestion or alter gold-sampling probabilities.
- Unobserved inventory levels are corpus coverage, not invalid labels.
- A zero applicability-class count is observed annotation behavior, not validity evidence.

- Outcome columns loaded: none.
- Inter-model agreement is reliability evidence, not validity evidence.
