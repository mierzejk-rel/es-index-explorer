# Partial report 01 — Data contract

Structural verification: **PASS**.

## Analysis population

- `emc2_set1`: 21 rubrics, 102 variants, 120 expectations, 2856 traces.
- `emc2_set2`: 20 rubrics, 79 variants, 98 expectations, 2212 traces.
- `mallinckrodt`: 22 rubrics, 82 variants, 96 expectations, 2296 traces.
- Total: 63 rubrics, 263 variants, 314 expectations, 7364 traces.
- Criterion observations: 36624 (expected 36624).
- Designed variants per rubric: 2–7.

## Gates

- Arm-balance failures: 0.
- Ineligible traces: 0.
- Non-blocking join warnings: 7 (see `tables/join_discrepancies.parquet`).
- Material expectations: 314/314 (non-material: 0).
- RubricV2 mismatches: 0.
- Grade mismatches: 0 (0.0000%).
- F3 interaction: confirmatory (rank 5/5, H_F3=6).

## Join-time degenerate-case census

- `P_zero_trace_count`: 1812.
- `F_zero_trace_count`: 2396.
- `U_zero_trace_count`: 7363.
- `all_undetermined_trace_count`: 0.
- `single_expectation_rubric_count`: 10.
- `single_expectation_trace_count`: 1092.
- `missing_error_judgement_trace_count`: 0.
- `error_override_trace_count`: 501.
- `zero_component_smoothing_required_trace_count`: 7363.

All counts are structural diagnostics. No feature–outcome association was computed.
