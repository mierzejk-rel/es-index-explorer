# Stage C selection from Stage A and B results

This report selects a small Stage C reasoning experiment from the completed
Simple Mode Stage A and Stage B suite. It does not amend the experiment
design in `10-simple-mode-experiment-design.md`; the subsequent design update
will make the approved selections normative.

## Decision

Run four `reasoning_effort = "low"` arms, each on EMC2 set 1, EMC2 set 2,
and Mallinckrodt:

1. `S-C-bm25-c3-rr-f20-g20-rlow`
   - Stage A reference: `S-A-bm25-c3-rr-f20-g20-rnone`
   - One-mode BM25 round-robin arm with the strongest quality/latency balance.
2. `S-C-bm25-dense-c2-union-f20-rlow`
   - Stage B reference: `S-B-bm25-dense-c2-union-f20-rnone` (B6).
   - Fast two-call heterogeneous current-union arm.
3. `S-C-bm25-dense-bm25-c3-union-f20-rlow`
   - Stage B reference: `S-B-bm25-dense-bm25-c3-union-f20-rnone` (B4).
   - Higher-quality three-call heterogeneous current-union arm.
4. `S-C-bm25-dense-bm25-dense-c4-union-f15-rlow`
   - Stage B reference: `S-B-bm25-dense-bm25-dense-c4-union-f15-rnone` (B5).
   - Accuracy-oriented four-call heterogeneous current-union arm.

This is **12 Stage C experiment runs**: four selected arms times three
datasets. No second Stage A arm is recommended. The selected Stage A arm
already has the best aggregate pass rate and RubricV2 score among Stage A
arms while remaining near the fastest quality frontier.

## Scope and data integrity

Source snapshot:

`/Users/krzysztof.mierzejewski/PycharmProjects/es-index-explorer/artifacts/mlflow/simplemode-stage-v3`

The snapshot is schema v3 and has matching manifest checksums for all Parquet
artifacts. It contains:

- 72 experiments and 72 finished runs;
- 54 Stage A runs (18 arms x 3 datasets);
- 18 Stage B runs (6 arms x 3 datasets);
- all three B5 c4 runs;
- 4,734 Stage A and 1,578 Stage B root `invoke_*` spans, each with an
  ordinal grade; and
- no exported Simple retrieval-plan validation failures.

## Selection method

The selection is descriptive, not a causal claim about retrieval or
reasoning. For each arm, metrics are first calculated separately for each
dataset, then the three dataset-level values are averaged with equal weight:

- primary quality: MLflow `total_pass_rate` (Good plus Acceptable);
- corroborating quality: `RubricV2_avg`;
- latency: `execution_time_p50_s`, with `execution_time_p95_s` as a tail
  guard;
- operational cost: `avg_total_tokens`;
- tie-breaker: the Mallinckrodt quality result, because that corpus was the
  weakest setting for dense and hybrid Stage A retrieval.

Raw traces are not pooled across datasets. Small latency differences are
observational because runs were not randomized into common time windows.

## Stage A selection

| Arm | Mean pass rate | Mean RubricV2 | Median p50 latency (s) | Median p95 latency (s) | Mallinckrodt pass rate | Mallinckrodt RubricV2 |
|---|---:|---:|---:|---:|---:|---:|
| `S-A-bm25-c3-rr-f20-g20-rnone` | 0.5861 | 0.5660 | 17.2795 | 28.5726 | 0.4756 | 0.4759 |
| `S-A-bm25-c2-rr-f15-g15-rnone` | 0.5266 | 0.5055 | 16.4535 | 24.1409 | 0.4512 | 0.4482 |
| `S-A-bm25-c3-rr-f15-g15-rnone` | 0.5278 | 0.5214 | 17.6415 | 26.1607 | 0.4634 | 0.4716 |
| `S-A-hybrid-c3-rr-f30-g30-rnone` | 0.5371 | 0.5342 | 43.9850 | 103.3139 | 0.3049 | 0.3237 |

`S-A-bm25-c3-rr-f20-g20-rnone` is selected. It has the highest Stage A mean
pass rate and mean RubricV2 score. Its median p50 is only 0.83 seconds above
the lower-depth BM25 alternative, while its mean pass rate is 5.95 percentage
points higher and its Mallinckrodt pass rate is 2.44 points higher.

The c3/f15 BM25 alternative uses fewer tokens, but it is slower at p50 and
lower quality on all selection measures. The high-context hybrid arms do not
justify their much larger latency for this reasoning-effort experiment.

## Stage B selection

| Arm | Mean pass rate | Mean RubricV2 | Median p50 latency (s) | Median p95 latency (s) | Mallinckrodt pass rate | Mallinckrodt RubricV2 |
|---|---:|---:|---:|---:|---:|---:|
| B4: `S-B-bm25-dense-bm25-c3-union-f20-rnone` | 0.6217 | 0.6178 | 17.2355 | 32.7559 | 0.4756 | 0.4967 |
| B6: `S-B-bm25-dense-c2-union-f20-rnone` | 0.5836 | 0.5920 | 16.3245 | 28.3064 | 0.4512 | 0.4840 |
| B5: `S-B-bm25-dense-bm25-dense-c4-union-f15-rnone` | 0.6235 | 0.6427 | 21.2625 | 36.9838 | 0.4390 | 0.5351 |
| B3: `S-B-bm25-dense-bm25-c3-union-f15-rnone` | 0.5926 | 0.6033 | 18.4785 | 31.6975 | 0.4390 | 0.4786 |
| B2: `S-B-bm25-bm25-c2-union-f20-rnone` | 0.5660 | 0.5738 | 26.6400 | 78.6514 | 0.4756 | 0.5044 |
| B1: `S-B-bm25-dense-c2-union-f15-rnone` | 0.5711 | 0.5506 | 25.7310 | 64.5538 | 0.4024 | 0.3981 |

Select B4, B5, and B6:

- **B4 is the high-quality Stage B arm.** It is only 0.18 percentage points
  below B5 in mean pass rate while 4.03 seconds faster at median p50. It also
  has a 3.66-point higher Mallinckrodt pass rate. B5's higher mean RubricV2
  makes it a separate accuracy-oriented Stage C arm despite that speed and
  Mallinckrodt trade-off.
- **B6 is the fast Stage B arm.** It is the fastest current-union arm at
  median p50 and maintains a strong 0.5836 mean pass rate. Compared with B4,
  it gives a lower-depth two-call reference with a 0.91-second median-p50
  advantage, while B4 gives the quality-oriented reference.

B2 and B1 are dominated in the intended quality/latency space. B3 is a
plausible lower-cost c3 alternative, but B4 has higher mean quality and lower
median p50.

## Why the selected Stage B arms matter

The selected arms frame the most informative small Stage C comparison:

- B6 asks whether low reasoning improves the fast, two-call heterogeneous
  context enough to justify its additional reasoning cost.
- B4 asks the same question for the stronger three-call mixed context, which
  already improves mean pass rate by 3.81 points over B6 while adding only
  0.91 seconds at median p50.
- B5 asks whether low reasoning is useful for the highest aggregate
  accuracy/RubricV2 c4 breadth context, accepting its higher latency and
  weaker Mallinckrodt pass rate as an explicit trade-off.

The comparison between B4 and B6 is not a clean one-dimension causal test:
it changes requested calls from two to three as well as the retrieval-tool
multiset. It is used here to select complementary reasoning candidates, not
to attribute their existing quality difference solely to the extra call.

## Stage C reasoning treatment

Every LLM call in a Stage C run must use `reasoning_effort = "low"`. This is
already how the v3 graph applies a model configuration:

- the tool-calling request copies all `openai_request` fields except
  `response_format` before creating `llm_create`;
- the structured-output `llm_parse` uses the same `config.openai_request`.

Consequently, one `openai_request.reasoning_effort = "low"` setting applies
to tool planning, answer generation, structured output, and any parse/retry
calls. Do not introduce per-call reasoning overrides in Stage C.

For each selected reference, all non-reasoning settings must remain unchanged:

- model, seed, system prompt, response format, token budget, and tool multiset;
- retrieval mode, requested calls, merge policy, per-call fetch, and global
  context setting;
- metadata-filter policy, one retrieval round, and invocation concurrency 1.

The completion budget remains part of the paired contract: C1 retains 12,000
`max_completion_tokens`; C2/C3/C4 retain 14,000 from B6/B4/B5 respectively.
The budget is not an allocation of reasoning tokens—GPT-5.1 controls actual
reasoning-token use from `reasoning_effort`—but it preserves enough total
completion headroom for the visible response and low-reasoning work.

The intentional treatment change is only `reasoning_effort`: `none` to
`low`. Model version, config filename, and MLflow experiment name naturally
change to identify the new run; they are identifiers, not additional
treatments.

## Evaluation plan

For each Stage C reference pair, report results per dataset before any
three-dataset summary:

1. Good plus Acceptable rate, Good rate, RubricV2 mean, ordinal distribution,
   and critical-error-mode rate.
2. Root `invoke_*` p50/p90/p95 latency, total tokens, prompt tokens,
   completion tokens, and reasoning tokens.
3. Matched rubric-level deltas between `rnone` and `rlow`, joined by dataset
   and rubric identity.
4. Retrieval-plan validity and actual retrieval-call count. They should
   remain unchanged from the `rnone` reference.

Interpret Stage C as a quality/latency/cost trade-off. The expected result is
not necessarily a quality improvement: low reasoning can improve selection,
grounding, or structured-output behavior, but it can also increase latency,
tokens, and variance.

## Registered Stage C design

`10-simple-mode-experiment-design.md` registers the four selected Stage C
arms, their identifiers, the `none` to `low` paired-comparison contract,
completion-budget preservation, and the 12-run execution plan.
