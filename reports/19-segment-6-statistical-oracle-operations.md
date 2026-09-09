# Segment 6 statistical-oracle operations

This guide documents the reproducible R reference environment and the offline Python
verification used by the Simple Mode question-suitability analysis. The statistical contract
is authoritative in `reports/13-simple-mode-analysis-research-plan.md` §§5.2–5.3, 11.1,
18.2–18.7, and 19.

## Scope

Segment 6 implements numerical primitives and pre-modelling oracles only. It does not fit
Layer 1, fit any confirmatory family, inspect outcome-model results, or change the human-gold
gate. The R comparison validates the linear F6 special case; it does not externally validate
the binomial-GLM or stacked-outcome extensions. Native `fwildclusterboot` validates its raw
scalar linear statistic; a separate base-R derivation validates the production full-matrix
CGM covariance, PSD map, and projected Wald statistic.

## Environment

Host R and RStudio are not required. The oracle image is defined by
`tests/oracles/fwildclusterboot/Dockerfile` and uses:

- `rocker/r-ver:4.4.3` at the Dockerfile's OCI digest;
- the `linux/amd64` platform for a single cross-host reference;
- `fwildclusterboot` 0.14.3 and transitive packages from the committed `renv.lock`;
- a pure base-R covariance reference in `covariance_reference.R`, separate from the Python
  production implementation;
- a read-only fixture input mount and a dedicated writable output mount.

The Dockerfile and `renv.lock` are the infrastructure-as-code boundary. Docker Compose and a
development container are unnecessary for this one-off computation.

## Architecture decision

The canonical oracle is fixed to `linux/amd64`. The Rocker image's OCI index also publishes
`linux/arm64`, which Docker Desktop can run natively on Apple Silicon, but a single canonical
architecture is more important here than the performance of a one-off fixture-generation job.
It prevents compiler, BLAS, and floating-point reduction differences between architectures
from creating non-byte-identical fixture output while appearing to represent the same
calculation.

The statistical result is expected to agree across amd64 and arm64 within the frozen
comparison tolerances, but exact identity is not assumed. Docker is not used by normal
analysis or Python test replay, so the amd64 emulation cost is bounded: the initial image
build on the implementation host took roughly three minutes and fixture generation roughly
ten seconds. These are observed local timings, not a guarantee.

An arm64 build may be used only as a non-canonical compatibility check. Compare it with the
canonical fixture's `p_f` (absolute tolerance `1e-4`), `W_obs` (relative tolerance `1e-6`),
and invalid-statistic count. Do not overwrite the amd64 fixture or change its provenance
unless that comparison passes and the project explicitly re-locks the new reference.

## Manual checkpoint

Before the first image build, start Docker Desktop. Docker Desktop 4.90.0 on macOS supports
the required Linux image and amd64 emulation. No other manual setup is expected.

The first build requires network access to the pinned base image and R package sources. If a
proxy or registry blocks a URL, stop and resolve that access outside the repository. Never
write proxy credentials or registry tokens to tracked files.

## Build and generate the reference

From the repository root:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag simplemode-fwildclusterboot:0.14.3 \
  tests/oracles/fwildclusterboot

UV_NO_ENV_FILE=1 uv run --no-env-file python \
  scripts/generate_fwildclusterboot_fixture.py \
  --analysis-root artifacts/question_analysis/simplemode-v1-segment5 \
  --image simplemode-fwildclusterboot:0.14.3
```

The generator derives the linear input from the immutable Segment 5 handoff, before the
Segment 6 manifest fingerprints the fixture. This avoids a circular lock in which generating
the fixture would invalidate the new root that consumes it. The canonical Segment 6
`oracle` run independently rebuilds the same input from its own tables and requires byte
identity. The generator stages container output outside the fixture directory, validates
every output, and atomically replaces the repository fixture only when all checks pass. The
container cannot write anywhere except its output mount.

Run the generator twice from identical inputs. The input, native output, independent
covariance-reference output, auxiliary-weight, and provenance hashes must be identical.

## Two-layer oracle boundary

`fwildclusterboot` computes the raw linear statistic directly from its add-add-subtract
cluster terms. It does not construct and eigendecompose the full coefficient covariance, so
its `W_obs` must not be described as PSD-projected. The independent base-R reference:

1. forms the unrestricted OLS row scores;
2. independently aggregates rubric, arm, and non-empty intersection cluster sums;
3. applies the realised `G/(G-1)`, `H/(H-1)`, and `GH/(GH-1)` factors;
4. constructs and symmetrises raw `V_3`;
5. applies the frozen `1e-10` eigenvalue rule; and
6. records both raw and projected `W_obs`.

The container rejects generation unless the base-R raw statistic agrees with native
`fwildclusterboot` to relative `1e-6`. The offline gate then runs the production Python
`three_term_cluster_covariance()` and `joint_wald_statistic()` paths against every persisted
R matrix and statistic. This triangulates the external package, the independent mathematical
derivation, and production code.

## Frozen R call

The runner uses resolved-only criterion outcomes, F6 `token_count` plus rubric fixed effects,
rubric and arm clustering, and the token-count null restriction. Its call fixes:

```r
dqrng::dqset.seed(seed_words)
weight_matrix <- fwildclusterboot:::get_weights(
  type = "rademacher",
  full_enumeration = FALSE,
  N_G_bootcluster = arm_count,
  boot_iter = as.integer(contract$bootstrap_replicates),
  sampling = "dqrng"
)

# Reset so boottest consumes the exact persisted schedule.
dqrng::dqset.seed(seed_words)
result <- fwildclusterboot::boottest(
  fit,
  param = contract$restriction_column,
  clustid = c("rubric", "arm"),
  bootcluster = "arm",
  B = as.integer(contract$bootstrap_replicates),
  type = "rademacher",
  impose_null = TRUE,
  engine = "R",
  sampling = "dqrng",
  conf_int = FALSE,
  ssc = fwildclusterboot::boot_ssc(
    adj = FALSE,
    fixef.K = "none",
    cluster.adj = TRUE,
    cluster.df = "conventional"
  )
)
```

The 64-bit `r_oracle` stream seed is split into two signed 32-bit R words without passing
through a floating-point representation. The deliberate private `get_weights()` call persists
the auxiliary schedule before the seed reset; `boottest()` itself does not retain a second
copy of those weights. The native output's schema-v2 typed call metadata records every one of
these arguments, including `conf_int = FALSE`; omission or drift is a validation failure.

## Offline workflow verification

After the reference exists, ordinary verification requires neither Docker nor R:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability \
  --analysis-root artifacts/question_analysis/simplemode-v1-segment6-postaudit-final \
  oracle
```

The command validates fixture provenance and hashes, runs both Python paths, and requires:

- absolute `p_f` difference at most `1e-4`;
- relative raw and projected `W_obs` differences at most `1e-6`, with absolute tolerance
  `1e-12`;
- component-meat, covariance, and eigenvalue agreement at relative `1e-8` and absolute
  `1e-10`;
- exact realised cluster counts and PSD diagnostic flags.

Each numerical comparison records its maximum absolute difference, maximum reference
magnitude, and maximum tolerance ratio
`max(abs(actual - expected) / (atol + rtol * abs(expected)))`. A passing comparison has a
tolerance ratio at most one. This preserves the frozen `rtol`/`atol` gate without reporting a
misleading relative percentage for reference entries inside the absolute-tolerance region.

It writes `statistics/r_oracle_verification.json` and records the `oracle` workflow step only
after every required comparison passes. A mismatch is a blocking gate failure.

The F6 raw covariance is materially indefinite. The canonical fixture therefore records raw
`W_obs` near `0.0243104822`, projected `W_obs` near `0.0216641544`, and a relative projection
shift near `0.1088554`. That shift is expected diagnostic evidence and must remain visible; it
is not resolved by removing the frozen PSD map or substituting the native raw statistic.

## Audit-remediation lineage

`simplemode-v1-segment6` remains the immutable pre-remediation Stage 6 checkpoint; its
schema-v1 oracle artifact is historical evidence and is not rewritten.
`simplemode-v1-segment6-postremediation-final` remains the immutable six-finding remediation
checkpoint. `simplemode-v1-segment6-postaudit-final` is the current Stage 6 checkpoint. It
rematerializes deterministic tables and hash-verified annotation provenance, records the
schema-v4 two-layer verification and schema-v2 native call, and stops while the human-data
gate remains locked. No per-finding temporary root is a canonical checkpoint.

The reference package reports and drops non-finite linear-reference statistics according to
its own documented implementation; the committed run records 223 such draws and Python
reproduces that count exactly. This is part of reproducing the external oracle, not the
production GLM discard policy. The custom sampled WCR engine separately follows §5.3 by
replenishing to exactly 9,999 valid replicates.

The verification artifact names these as distinct contracts:

- `fwildclusterboot_valid_only_strict_no_plus_one` for the native replay;
- `replenish_to_B_then_plus_one` for production sampled WCR; and
- `full_support_bracket_no_plus_one` for production enumeration.

The committed F6 schedule intentionally demonstrates that the first convention is not the
second. The external p-value is never substituted for a production family p-value.

## Full-refit and BH diagnostics

The one-step invariant check applies only while restricted scores and bread are fixed. If the
99% indicator-agreement or 1% Wald-discrepancy criterion fails, each fallback replicate stores
the actual restricted and unrestricted solver results, recomputed bread and covariance, and
full-refit Wald statistic. The p-value is rebuilt from those full-refit Wald values. No
one-step meat is attached to a full-refit replicate, and no fixed-score invariant is claimed
for re-estimated covariance blocks. The synthetic SciPy comparison independently writes the
weighted score equations, Bernoulli bread, CGM cluster sums, PSD map, and Wald calculation;
it does not call production numerical helpers. This remains a synthetic GLM extension check,
not an external-package GLM oracle.

Two-corner BH adjudication emits one deterministic decision per family. Every family in the
lower/upper rejection-set symmetric difference, including an unaffected family whose decision
changes through threshold spillover, carries `AnalysisFlag.BH_INDETERMINATE` exactly.

Bootstrap failure disclosure retains singular and non-finite counts separately and also
persists their combined count, denominator, rate, and strict one-percent trigger. Sampled WCR
uses the requested valid target `B` as the denominator even when replenishment makes
`attempted_replicates > B`; enumeration uses the full attainable support `S_f`. Exactly 1%
does not trigger because the frozen comparison is `rate > 0.01`. Closed exception types,
rather than message parsing, distinguish singular restricted covariance from non-finite
replicates. Any other numerical error propagates and fails the family closed.

## Failure recovery

- Docker daemon unavailable: start Docker Desktop and retry fixture generation.
- Package restore failure: retain the existing fixture; inspect the exact pinned source URL.
- Fixture hash mismatch: do not overwrite the reference until the changed input is explained.
- R/Python mismatch: leave `oracle` incomplete and block all later fitting.
- Numerical non-computability: preserve the categorized failure record; never substitute a
  pseudo-inverse statistic or a different bootstrap package.

## Quality checks

The normal Python suite consumes only committed fixtures. Live Docker regeneration is a
separate one-off integration check. Run the repository quality commands documented in
`README-question-suitability.md`; no model-based audit is part of Segment 6.
