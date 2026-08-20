# `fwildclusterboot` reference environment

This directory keeps the R reference oracle separate from Python dependencies.
The base image is pinned by OCI index digest and supports both `linux/amd64` and
`linux/arm64`.

Segment 1 establishes only this immutable runtime boundary. The R package lock,
linear-reduction inputs, runner, and expected outputs are added with the
statistical oracle in Segment 6. Host R is never required.

Build the foundation image:

```bash
docker build -t simplemode-fwildclusterboot-base .
```
