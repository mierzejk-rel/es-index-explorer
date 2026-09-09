# `fwildclusterboot` reference environment

This directory keeps the one-off R reference generator separate from Python
dependencies. Host R is never required. The reference platform is fixed to
`linux/amd64`; Docker Desktop provides emulation on Apple Silicon.

The pinned Rocker OCI index also supports `linux/arm64`, but amd64 remains canonical. Fixing
one architecture avoids otherwise harmless compiler, BLAS, and floating-point reduction
differences from changing byte-level fixture output between developer machines. Docker is
needed only for fixture regeneration; ordinary Python tests and the `oracle` command do not
run it. An arm64 build may be used as a compatibility check, but must not replace the
canonical fixture without passing its `p_f`, `W_obs`, and invalid-statistic-count comparisons
and an explicit re-lock.

The environment is reproducible at three levels:

- `Dockerfile` pins Rocker R 4.4.3 by OCI digest;
- `renv.lock` pins `fwildclusterboot` 0.14.3 and all transitive R packages;
- `covariance_reference.R` independently derives the full CGM covariance and frozen PSD map;
- `fixtures/` records the F6 linear input, Rademacher signs, native package output,
  independent covariance output, and provenance hashes.

Build the image from the repository root:

```bash
docker buildx build \
  --platform linux/amd64 \
  --network host \
  --load \
  --tag simplemode-fwildclusterboot:0.14.3 \
  tests/oracles/fwildclusterboot
```

Generate the fixture:

```bash
UV_NO_ENV_FILE=1 uv run --no-env-file python \
  scripts/generate_fwildclusterboot_fixture.py \
  --analysis-root artifacts/question_analysis/simplemode-v1-segment5 \
  --image simplemode-fwildclusterboot:0.14.3
```

The generator gives the container read-only fixture inputs and a temporary
writable output mount. It validates outputs before atomically replacing the
reference and provenance. Run it twice and require identical hashes.

The R runner seeds `dqrng`, pre-draws the auxiliary Rademacher schedule with
`fwildclusterboot:::get_weights`, resets the same seed, and then invokes
`boottest(..., conf_int = FALSE)`. This guarantees that the signs written to
`f6-linear-rademacher-weights.csv` are exactly the schedule consumed by the
native reference calculation; `boottest` does not retain a separate copy.

The normal `question-suitability oracle` command does not invoke Docker. It
replays the committed fixtures in Python. Native `fwildclusterboot` validates
the unprojected scalar statistic; it does not apply the project's full-matrix
PSD map. The base-R reference separately records all three meat components,
raw covariance, eigenvalues, projected covariance, and raw/projected Wald
statistics. The gate calls the production Python CGM/PSD path and records a
pass only when every corresponding object agrees at its frozen tolerance. See
`reports/19-segment-6-statistical-oracle-operations.md` for full operations and
failure recovery.
