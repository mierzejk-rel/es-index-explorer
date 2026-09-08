# `fwildclusterboot` reference environment

This directory keeps the one-off R reference generator separate from Python
dependencies. Host R is never required. The reference platform is fixed to
`linux/amd64`; Docker Desktop provides emulation on Apple Silicon.

The environment is reproducible at three levels:

- `Dockerfile` pins Rocker R 4.4.3 by OCI digest;
- `renv.lock` pins `fwildclusterboot` 0.14.3 and all transitive R packages;
- `fixtures/` records the F6 linear input, Rademacher signs, R output, and
  provenance hashes.

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

The normal `question-suitability oracle` command does not invoke Docker. It
replays the committed fixture in Python and records the pass only when `p_f`
agrees to `1e-4` and `W_obs` agrees to relative `1e-6`. See
`reports/19-segment-6-statistical-oracle-operations.md` for full operations and
failure recovery.
