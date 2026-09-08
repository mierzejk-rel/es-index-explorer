# Segment 6 statistical-oracle operations

This guide documents the reproducible R reference environment and the offline Python
verification used by the Simple Mode question-suitability analysis. The statistical contract
is authoritative in `reports/13-simple-mode-analysis-research-plan.md` §§5.2–5.3, 11.1,
18.2–18.7, and 19.

## Scope

Segment 6 implements numerical primitives and pre-modelling oracles only. It does not fit
Layer 1, fit any confirmatory family, inspect outcome-model results, or change the human-gold
gate. The R comparison validates the linear F6 special case; it does not externally validate
the binomial-GLM or stacked-outcome extensions.

## Environment

Host R and RStudio are not required. The oracle image is defined by
`tests/oracles/fwildclusterboot/Dockerfile` and uses:

- `rocker/r-ver:4.4.3` at the Dockerfile's OCI digest;
- the `linux/amd64` platform for a single cross-host reference;
- `fwildclusterboot` 0.14.3 and transitive packages from the committed `renv.lock`;
- a read-only fixture input mount and a dedicated writable output mount.

The Dockerfile and `renv.lock` are the infrastructure-as-code boundary. Docker Compose and a
development container are unnecessary for this one-off computation.

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

Run the generator twice from identical inputs. The input, output, auxiliary-weight, and
provenance hashes must be identical.

## Frozen R call

The runner uses resolved-only criterion outcomes, F6 `token_count` plus rubric fixed effects,
rubric and arm clustering, and the token-count null restriction. Its call fixes:

```r
fwildclusterboot::boottest(
  fit,
  param = "token_count",
  clustid = c("rubric", "arm"),
  bootcluster = "arm",
  B = 9999,
  type = "rademacher",
  impose_null = TRUE,
  engine = "R",
  sampling = "dqrng",
  getauxweights = TRUE,
  ssc = fwildclusterboot::boot_ssc(
    adj = FALSE,
    fixef.K = "none",
    cluster.adj = TRUE,
    cluster.df = "conventional"
  )
)
```

The 64-bit `r_oracle` stream seed is split into two signed 32-bit R words without passing
through a floating-point representation.

## Offline workflow verification

After the reference exists, ordinary verification requires neither Docker nor R:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file question-suitability \
  --analysis-root artifacts/question_analysis/simplemode-v1-segment6 \
  oracle
```

The command validates fixture provenance and hashes, runs the Python linear special case with
the same auxiliary signs, and requires:

- absolute `p_f` difference at most `1e-4`;
- relative `W_obs` difference at most `1e-6`.

It writes `statistics/r_oracle_verification.json` and records the `oracle` workflow step only
after both tolerances pass. A mismatch is a blocking gate failure.

The reference package reports and drops non-finite linear-reference statistics according to
its own documented implementation; the committed run records 223 such draws and Python
reproduces that count exactly. This is part of reproducing the external oracle, not the
production GLM discard policy. The custom sampled WCR engine separately follows §5.3 by
replenishing to exactly 9,999 valid replicates.

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
