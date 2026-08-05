# Stage A results and Stage B redesign

Generated from the immutable sanitized snapshot at `/Users/krzysztof.mierzejewski/PycharmProjects/es-index-explorer/artifacts/mlflow/simplemode-all-runs` on
2026-08-05T12:35:28.324892+00:00.

## Executive conclusion

Stage A supports BM25 as the primary Stage B homogeneous `current_union` arm.
Dense-only retrieval is not competitive on Mallinckrodt, while BM25/dense
ranked-result overlap is low enough to justify one heterogeneous BM25-first,
dense-second late-union pilot. Include one BM25 c3 control because matched
Stage A evidence shows a material quality gain in at least one dataset/context.

Recommended Stage B matrix:

1. BM25 `current_union`, c2/f20.
2. Heterogeneous `[bm25, dense]` `current_union`, c2/f20.
3. BM25 `current_union`, c3/f20 control.
4. Do not include dense-only union; defer RRF-union unless its implementation
   cost is negligible.

## Validation and integrity

- Snapshot checksum validation: **True**
- Runs / experiments / arms / datasets: **54 /
  54 / 18 /
  3**
- Unique traces: **4734**
- Exact duplicate assessment rows removed: **102**
- Conflicting duplicate assessments: **0**
- Missing grade / latency / derived rubric key: **0 /
  0 / 0**
- Valid for analysis: **True**

The exporter did not retain `row_id`, `use_case`, `dataset_id`, or
`evalset_variant` trace attributes. Pairing therefore uses the unique root
`invoke_*` span name (`rubric_key`) within each dataset segment. Every trace has
exactly one such key. Use-case names are derived from that key.

## Exact calculations

- `Good+Acceptable rate = (Good count + Acceptable count) / unique trace count`.
- `Good rate = Good count / unique trace count`.
- `Critical grade rate = Critical Error count / unique trace count`.
- `RubricV2 mean = arithmetic mean of the one RubricV2 assessment per trace`.
- End-to-end latency is the `observed_duration_ms / 1000` of the unique
  `UNKNOWN` root span whose name begins with `invoke_`.
- p50/p90/p95 use pandas quantiles with `linear` interpolation.
- Pairwise results inner-join arms on `(dataset_segment, rubric_key)` and publish
  matched sample size, grade wins/ties/losses, pass-rate delta, RubricV2 delta,
  and median paired latency delta.
- Pareto membership is computed independently per dataset. An arm is dominated
  if another arm has quality greater than or equal and p50 latency less than or
  equal, with at least one strict improvement.

## Top observed arms by dataset

| dataset_segment | arm_id | trace_count | good_acceptable_rate | rubric_v2_mean | latency_p50_s | latency_p95_s | avg_total_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| emc2_set1 | S-A-hybrid-c3-rr-f30-g30-rnone | 102 | 0.6863 | 0.7134 | 51.7688 | 117.6255 | 22960.0000 |
| emc2_set1 | S-A-bm25-c3-rr-f20-g20-rnone | 102 | 0.6373 | 0.6393 | 17.2799 | 28.5728 | 19610.0000 |
| emc2_set1 | S-A-hybrid-c2-rr-f30-g25-rnone | 102 | 0.6373 | 0.6594 | 54.6231 | 111.1723 | 18364.0000 |
| emc2_set1 | S-A-hybrid-c1-rr-f30-g30-rnone | 102 | 0.6275 | 0.6709 | 44.2430 | 84.3061 | 18401.0000 |
| emc2_set1 | S-A-hybrid-c3-rr-f30-g20-rnone | 102 | 0.6275 | 0.6411 | 65.4695 | 139.3114 | 19686.0000 |
| emc2_set2 | S-A-bm25-c3-rr-f20-g20-rnone | 79 | 0.6456 | 0.5827 | 16.5661 | 24.8656 | 18967.0000 |
| emc2_set2 | S-A-hybrid-c3-rr-f30-g30-rnone | 79 | 0.6203 | 0.5655 | 43.9857 | 101.3660 | 22004.0000 |
| emc2_set2 | S-A-dense-c3-rr-f20-g20-rnone | 79 | 0.6076 | 0.5910 | 41.9981 | 79.6487 | 19434.0000 |
| emc2_set2 | S-A-hybrid-c2-rr-f30-g25-rnone | 79 | 0.5949 | 0.5875 | 49.7952 | 105.2733 | 17900.0000 |
| emc2_set2 | S-A-dense-c1-rr-f10-g10-rnone | 79 | 0.5696 | 0.4970 | 12.8639 | 18.0359 | 12939.0000 |
| mallinckrodt | S-A-bm25-c3-rr-f20-g20-rnone | 82 | 0.4756 | 0.4759 | 18.7962 | 30.7879 | 26396.0000 |
| mallinckrodt | S-A-bm25-c3-rr-f15-g15-rnone | 82 | 0.4634 | 0.4716 | 18.3408 | 30.0484 | 23450.0000 |
| mallinckrodt | S-A-bm25-c2-rr-f15-g15-rnone | 82 | 0.4512 | 0.4482 | 17.5275 | 28.4500 | 23379.0000 |
| mallinckrodt | S-A-bm25-c1-rr-f20-g20-rnone | 82 | 0.4268 | 0.4084 | 18.7306 | 39.7711 | 21769.0000 |
| mallinckrodt | S-A-bm25-c2-rr-f10-g10-rnone | 82 | 0.4146 | 0.4532 | 15.7720 | 24.7518 | 19693.0000 |

## Pareto sensitivity

The following arms are Pareto-efficient under at least one of Good+Acceptable,
Good-only, or RubricV2 quality, always against p50 latency:

| dataset_segment | arm_id | good_acceptable_rate | good_rate | rubric_v2_mean | latency_p50_s | pareto_pass_p50 | pareto_good_p50 | pareto_rubric_p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| emc2_set1 | S-A-dense-c1-rr-f10-g10-rnone | 0.4118 | 0.2353 | 0.4348 | 15.3313 | True | True | True |
| emc2_set1 | S-A-bm25-c2-rr-f15-g15-rnone | 0.5588 | 0.3725 | 0.5369 | 16.4538 | True | True | True |
| emc2_set1 | S-A-bm25-c3-rr-f20-g20-rnone | 0.6373 | 0.4608 | 0.6393 | 17.2799 | True | True | True |
| emc2_set1 | S-A-dense-c3-rr-f20-g20-rnone | 0.5882 | 0.4608 | 0.6449 | 43.2139 | False | False | True |
| emc2_set1 | S-A-hybrid-c1-rr-f30-g30-rnone | 0.6275 | 0.5000 | 0.6709 | 44.2430 | False | True | True |
| emc2_set1 | S-A-hybrid-c3-rr-f30-g30-rnone | 0.6863 | 0.4706 | 0.7134 | 51.7688 | True | False | True |
| emc2_set2 | S-A-dense-c1-rr-f10-g10-rnone | 0.5696 | 0.2785 | 0.4970 | 12.8639 | True | True | True |
| emc2_set2 | S-A-bm25-c2-rr-f15-g15-rnone | 0.5696 | 0.3038 | 0.5312 | 15.4575 | False | True | True |
| emc2_set2 | S-A-bm25-c3-rr-f15-g15-rnone | 0.5316 | 0.3038 | 0.5322 | 16.4936 | False | False | True |
| emc2_set2 | S-A-bm25-c3-rr-f20-g20-rnone | 0.6456 | 0.3671 | 0.5827 | 16.5661 | True | True | True |
| emc2_set2 | S-A-dense-c3-rr-f15-g15-rnone | 0.5570 | 0.3797 | 0.5448 | 40.0810 | False | True | False |
| emc2_set2 | S-A-dense-c3-rr-f20-g20-rnone | 0.6076 | 0.3924 | 0.5910 | 41.9981 | False | True | True |
| mallinckrodt | S-A-dense-c1-rr-f10-g10-rnone | 0.1951 | 0.1585 | 0.1880 | 7.5596 | True | True | True |
| mallinckrodt | S-A-hybrid-c1-rr-f30-g15-rnone | 0.2195 | 0.1829 | 0.2583 | 9.6263 | True | True | True |
| mallinckrodt | S-A-hybrid-c1-rr-f30-g30-rnone | 0.2805 | 0.2195 | 0.2945 | 14.4264 | True | True | True |
| mallinckrodt | S-A-bm25-c2-rr-f10-g10-rnone | 0.4146 | 0.3171 | 0.4532 | 15.7720 | True | True | True |
| mallinckrodt | S-A-bm25-c2-rr-f15-g15-rnone | 0.4512 | 0.3537 | 0.4482 | 17.5275 | True | True | False |
| mallinckrodt | S-A-bm25-c3-rr-f15-g15-rnone | 0.4634 | 0.3171 | 0.4716 | 18.3408 | True | False | True |
| mallinckrodt | S-A-bm25-c3-rr-f20-g20-rnone | 0.4756 | 0.3049 | 0.4759 | 18.7962 | True | False | True |

## BM25 versus dense retrieval overlap

Chunk identities are unioned across calls within each trace before computing
Jaccard overlap. Counts below are medians over exact matched rubric variations:

| dataset_segment | matched_traces | median_jaccard | median_bm25_unique | median_dense_unique |
| --- | --- | --- | --- | --- |
| emc2_set1 | 612 | 0.1929 | 15.0000 | 12.0000 |
| emc2_set2 | 474 | 0.2060 | 13.0000 | 12.0000 |
| mallinckrodt | 492 | 0.0000 | 20.0000 | 20.0000 |

Low Jaccard plus non-zero unique contributions support testing heterogeneous
late union. This does not prove that unique dense chunks improve answer quality;
Stage B must test that causal hypothesis.

## Matched call-count effects

Largest observed pass-rate changes when call count changes at matched retrieval
family and global context:

| dataset_segment | left_arm | right_arm | matched_trace_count | right_grade_wins | ties | right_grade_losses | right_minus_left_pass_rate | right_minus_left_rubric_v2_mean | right_minus_left_latency_median_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| emc2_set1 | S-A-bm25-c1-rr-f20-g20-rnone | S-A-bm25-c3-rr-f20-g20-rnone | 102 | 37 | 54 | 11 | 0.1471 | 0.1224 | 0.0055 |
| emc2_set1 | S-A-bm25-c1-rr-f10-g10-rnone | S-A-bm25-c2-rr-f10-g10-rnone | 102 | 39 | 44 | 19 | 0.0980 | 0.1147 | 31.1404 |
| emc2_set1 | S-A-hybrid-c1-rr-f30-g15-rnone | S-A-hybrid-c2-rr-f30-g15-rnone | 102 | 29 | 59 | 14 | 0.0980 | 0.0704 | 15.1915 |
| emc2_set2 | S-A-bm25-c1-rr-f20-g20-rnone | S-A-bm25-c3-rr-f20-g20-rnone | 79 | 24 | 47 | 8 | 0.0886 | 0.0764 | 0.1117 |
| mallinckrodt | S-A-hybrid-c1-rr-f30-g15-rnone | S-A-hybrid-c2-rr-f30-g15-rnone | 82 | 18 | 57 | 7 | 0.0854 | 0.0635 | 17.8447 |
| emc2_set2 | S-A-hybrid-c1-rr-f30-g15-rnone | S-A-hybrid-c2-rr-f30-g15-rnone | 79 | 16 | 53 | 10 | 0.0759 | 0.0230 | 18.1883 |
| emc2_set1 | S-A-dense-c1-rr-f10-g10-rnone | S-A-dense-c2-rr-f10-g10-rnone | 102 | 33 | 51 | 18 | 0.0686 | 0.0621 | 23.3212 |
| emc2_set1 | S-A-hybrid-c1-rr-f30-g30-rnone | S-A-hybrid-c3-rr-f30-g30-rnone | 102 | 28 | 53 | 21 | 0.0588 | 0.0425 | 8.6503 |
| emc2_set2 | S-A-dense-c2-rr-f15-g15-rnone | S-A-dense-c3-rr-f15-g15-rnone | 79 | 15 | 50 | 14 | 0.0506 | 0.0550 | 2.6268 |
| emc2_set2 | S-A-hybrid-c1-rr-f30-g30-rnone | S-A-hybrid-c3-rr-f30-g30-rnone | 79 | 25 | 38 | 16 | 0.0506 | 0.0037 | 2.9754 |
| mallinckrodt | S-A-dense-c1-rr-f20-g20-rnone | S-A-dense-c3-rr-f20-g20-rnone | 82 | 10 | 68 | 4 | 0.0488 | 0.0234 | 1.9825 |
| mallinckrodt | S-A-bm25-c1-rr-f20-g20-rnone | S-A-bm25-c3-rr-f20-g20-rnone | 82 | 21 | 47 | 14 | 0.0488 | 0.0674 | -0.1352 |
| mallinckrodt | S-A-bm25-c1-rr-f10-g10-rnone | S-A-bm25-c2-rr-f10-g10-rnone | 82 | 19 | 50 | 13 | 0.0488 | 0.0779 | -3.1440 |
| emc2_set2 | S-A-dense-c1-rr-f20-g20-rnone | S-A-dense-c3-rr-f20-g20-rnone | 79 | 17 | 52 | 10 | 0.0380 | 0.0769 | 6.0486 |
| emc2_set1 | S-A-bm25-c2-rr-f15-g15-rnone | S-A-bm25-c3-rr-f15-g15-rnone | 102 | 23 | 59 | 20 | 0.0294 | 0.0236 | 1.8633 |
| mallinckrodt | S-A-dense-c1-rr-f10-g10-rnone | S-A-dense-c2-rr-f10-g10-rnone | 82 | 8 | 70 | 4 | 0.0244 | 0.0224 | 4.5422 |
| mallinckrodt | S-A-hybrid-c1-rr-f30-g30-rnone | S-A-hybrid-c3-rr-f30-g30-rnone | 82 | 18 | 54 | 10 | 0.0244 | 0.0292 | 3.0761 |
| emc2_set1 | S-A-dense-c1-rr-f20-g20-rnone | S-A-dense-c3-rr-f20-g20-rnone | 102 | 29 | 53 | 20 | 0.0196 | 0.0918 | 16.0211 |

## Operation timing and retries

Each row below summarizes the per-run/arm operation p50s. The complete
dataset-specific values, successful-attempt timings, and retry observation
counts are in `operation_timings.csv`.

| simple_operation | run_arm_observations | median_of_run_p50_ms | median_llm_retry_rate | median_es_retry_rate |
| --- | --- | --- | --- | --- |
| simple.structured_output | 54 | 25210.4223 | 0.0000 | — |
| simple.answer_generation | 54 | 7424.4270 | 0.0000 | — |
| simple.snippet_repair | 32 | 7022.5945 | 0.0000 | — |
| simple.query_plan | 54 | 1827.9060 | 0.0000 | — |
| simple.retrieval_generic | 54 | 705.9370 | — | 0.0000 |
| simple.retrieval_metadata_filter | 30 | 491.2213 | — | 0.0000 |
| simple.query_embedding | 36 | 35.8522 | — | — |

## Stage B recommendation

```json
{
  "c3_control": "include one BM25 c3 control",
  "dense_only_union": "exclude",
  "heterogeneous_pilot": "BM25-first + dense-second current_union c2/f20",
  "primary_homogeneous_arm": "BM25 current_union c2/f20",
  "rationale": {
    "bm25_mallinckrodt_best_pass_rate": 0.47560975609756095,
    "dense_mallinckrodt_best_pass_rate": 0.21951219512195122,
    "most_robust_pareto_arms": {
      "S-A-bm25-c2-rr-f10-g10-rnone": 3,
      "S-A-bm25-c2-rr-f15-g15-rnone": 7,
      "S-A-bm25-c3-rr-f15-g15-rnone": 3,
      "S-A-bm25-c3-rr-f20-g20-rnone": 8,
      "S-A-dense-c1-rr-f10-g10-rnone": 9,
      "S-A-hybrid-c1-rr-f30-g30-rnone": 5
    },
    "overlap_by_dataset": [
      {
        "dataset_segment": "emc2_set1",
        "median_dense_unique": 12.0,
        "median_jaccard": 0.19292803970223327,
        "paired_trace_count": 612
      },
      {
        "dataset_segment": "emc2_set2",
        "median_dense_unique": 12.0,
        "median_jaccard": 0.20604631217838765,
        "paired_trace_count": 474
      },
      {
        "dataset_segment": "mallinckrodt",
        "median_dense_unique": 20.0,
        "median_jaccard": 0.0,
        "paired_trace_count": 492
      }
    ]
  },
  "rrf_union_control": "defer unless implementation cost is negligible"
}
```

The recommendation is falsifiable: all source rows and formulas are published
beside the snapshot. In particular, `pairwise_comparisons.csv` exposes every
paired effect and `retrieval_overlap.csv` exposes every matched Jaccard input.

## Reproducibility and caveats

- Do not pool raw traces across datasets; all headline comparisons are
  dataset-specific.
- Latency is observational. `latency_by_arm_dataset.csv` publishes UTC run
  timestamps and hours so time-of-day confounding can be checked. Runs span
  `2026-07-30T13:35:01.830000+00:00` through `2026-08-04T00:57:03.038000+00:00` and were not
  randomized into common time windows, so small latency differences must not
  be attributed causally to arm parameters.
- The 102 duplicate ordinal-grade rows were identical and removed only after
  equality validation. No conflicting duplicate was found.
- Run-level MLflow metrics are retained as cross-checks, while headline quality
  and latency are recalculated from trace-level rows.
- The snapshot excludes messages, answers, query text, chunk content, XML, and
  scorer rationales by design.

## Audit files

All calculations are under `/Users/krzysztof.mierzejewski/PycharmProjects/es-index-explorer/artifacts/mlflow/simplemode-all-runs/stage_a_analysis`:

- `trace_level.csv`: one row per run/trace, the primary recomputation source.
- `run_level.csv`: per-run raw counts and calculated metrics.
- `quality_by_arm_dataset.csv` and `latency_by_arm_dataset.csv`.
- `pairwise_comparisons.csv`, `retrieval_overlap.csv`, and
  `quality_by_use_case.csv`.
- `pairwise_trace_deltas.csv` contains every exact paired rubric cell;
  `operation_timings.csv` contains operation/retry diagnostics; and
  `grade_distribution.csv` contains every grade numerator and denominator.
- `pareto_by_dataset.csv`, `validation.json`, and
  `calculation_manifest.json`.
