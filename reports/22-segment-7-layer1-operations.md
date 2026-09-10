# Segment 7 Layer 1 operations

## Scope

Segment 7 implements the arm-free empirical-Bayes hierarchy in
`reports/13-simple-mode-analysis-research-plan.md` §5.1 and the decisions in §6.
Section 5.2 belongs to `fit-families`. Segment 7 does not alter the frozen model,
run Segment 8, or treat the propagated hybrid distribution as a posterior.

The current software handoff uses
`artifacts/question_analysis/simplemode-v1-segment7`. It is a working lineage
root, not evidence that outcome modelling is unlocked. `fit-layer1` remains
blocked until `validate-features` and `oracle` are complete and
`outcome_modeling_unlocked=true`.

## Frozen constants

- ALR reference: FAIL; smoothing `eps=0.5` applied to all three components only
  when any count is zero.
- `phi` start clamp: `[0.1, 1000]`.
- Outer optimizer: L-BFGS; objective change `<1e-6`, maximum absolute gradient
  `<1e-4`, maximum 200 iterations; deterministic central differences start at
  `1e-5*max(1,abs(psi_j))`, use a computable one-sided fallback, and halve at
  most 12 times; internal relative-function tolerance `1e-12`.
- Inner Newton: maximum gradient `<1e-8`, objective change `<1e-10`, maximum
  100 iterations; Armijo coefficient `1e-4`, halving factor `0.5`, at most 20
  halvings, and floating-point comparison slack
  `1e-12*max(1,abs(log_density))`.
- Curvature: `K=-H` and
  `lambda_min(K) > 1e-10*max(1,lambda_max(K))`, with successful Cholesky.
- Three starts: `0.5*phi_init`, `phi_init`, `2*phi_init`; objective difference
  at most `1e-6` and maximum component-wise scaled parameter difference at most
  `1e-5`.
- Propagation: base `B_psi=500`; mandatory shared prefixes 250/500/1,000;
  adaptive depths 500/1,000/2,000/4,000; `M_b=40`; `M_cond=20,000`.
- Decisions: `gamma=0.90`, `kappa=0.75`, floors 0.75/0.60/0.50.
- Importance check: 5,000 particles; accept when `ESS/N>=0.10`.
- Elevated numerical-failure disclosure: rate `>5%`.
- Replenishment cap: `10*B_target` global attempts.

## Numerical flow

1. Load eligible raw `(P,F,U,N_r)` rows and require the exact 28-arm balance
   contract for every fitted variant.
2. Compute the method-of-moments start, including the pooled
   `sum_r(V_r-1)` denominator and reference-dataset offset.
3. Fit `psi=(phi,mu_0,d,Sigma_within,Sigma_between)` by nested-Laplace MML
   from all three starts.
4. Generate the global parametric-bootstrap attempt sequence from
   `layer1_bootstrap`; failed outer fits reject globally.
5. For every globally retained `psi*`, fit each rubric's conditional Laplace
   approximation against observed `D`. A rubric-local failure does not reject
   the outer vector for other rubrics.
6. Draw joint per-rubric latent blocks from deterministic `laplace_draws`
   children. All eligible variants in a rubric share one decision depth.
7. Compute persisted event batches, `Pi_prop`, outer-batch MCSE, `Pi_cond`,
   tiers, proportion diagnostics, score bands and stability outputs.

## Draw identity and refinement

`global_outer_attempt_id` is never renumbered. `rubric_retained_index` is local
and is never a synchronization key. Cross-rubric comparisons use only common
`(global_outer_attempt_id,inner_index)` pairs and require at least 250 common
outer IDs.

Each rubric has one `rubric_decision_depth`. Any ambiguous event belonging to
the rubric or one of its variants extends the entire joint rubric block.
Primary events are:

- rubric worst-variant event at each floor;
- each variant's singleton event at each floor;
- rubric `kappa=0.75` proportion event at each floor.

The closed `Pi_prop +/- 2*MCSE` interval triggers refinement when it contains
`gamma`, including endpoint equality. Ambiguity remaining at 4,000 produces
`MONTE_CARLO_INDETERMINATE`.

For one event, each retained outer draw's 40 inner indicators form one batch.
`Pi_prop` is the mean of batch means and
`MCSE=sample_sd(batch_means,ddof=1)/sqrt(B_psi)`.

## Failures

Primary three-start non-equivalence fails `fit-layer1`. Bootstrap
non-equivalence is `INITIALISER_SENSITIVE`, rejects that outer attempt globally,
and is replenished. Conditional mode/curvature failures are rubric-local.

If the global sequence or a required rubric cannot attain `B_target` before
`10*B_target` attempts, `LAYER1_REPLENISHMENT_EXHAUSTED` records scope, target,
attempted and valid counts. The workflow step fails and no partial
recommendation table is emitted.

Importance-resampling inadequacy is not a replenishment failure. It adds an
explicit approximation caveat and does not change a tier.

## Artifacts

- `statistics/layer1_fit.json`
- `statistics/layer1_hyperparameter_diagnostics.json`
- `statistics/layer1_numerical_failures.json`
- `statistics/layer1_importance_diagnostics.parquet`
- `statistics/layer1_prefix_diagnostics.parquet`
- `statistics/layer1_stability.parquet`
- `draws/pi_prop_<rubric_id>.parquet`, one versioned archive per rubric
- `tables/recommendation_table_rubric.parquet`
- `tables/recommendation_table_variant.parquet`
- `partial_reports/05-suitability-estimates.md`

Segment 5 emits the ordered report-04 family:
`04-feature-validation.md`, then
`04-supplement-exploratory-p4-evidence.md`.

## Commands

Before the human gate, only prepare and inspect the working root:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability status --json
```

After genuine unlock:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability fit-layer1
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability status --json
```

`status` reports `fit_layer1_runnable` and stable blockers; the default root
does not imply readiness.

## Verification

Run focused Layer 1 tests, relevant Stage 5/6 workflow regressions, and then:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file pytest tests/ \
  --cov=es_index_explorer.question_analysis --cov-branch \
  --cov-report=term-missing:skip-covered --cov-fail-under=88
UV_NO_ENV_FILE=1 uv run --no-env-file ruff check \
  es_index_explorer/question_analysis tests/unit/question_analysis
UV_NO_ENV_FILE=1 uv run --no-env-file ruff format --check \
  es_index_explorer/question_analysis tests/unit/question_analysis
UV_NO_ENV_FILE=1 uv run --no-env-file ty check \
  es_index_explorer/question_analysis tests/unit/question_analysis
```

No model-based implementation audit is part of Segment 7.
