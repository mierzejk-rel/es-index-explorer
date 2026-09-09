# Stage 6 post-audit P3 remediation

## Executive verdict

`READY_FOR_NEXT_STAGE`

**YES — S6R-1 through S6R-4 are closed without changing the frozen statistical
methodology.** The original S6-F1 through S6-F6 remediations remain intact. Stage 6
continues to provide independently checked CGM, PSD, and Wald primitives; explicit
bootstrap, enumeration, p-value, and BH conventions; deterministic streams; and
fail-closed numerical behavior. The human-gold gate and outcome fitting remain pending.

## Fresh-finding closure

| Finding | Severity | Final disposition | Evidence |
|---|---:|---|---|
| S6R-1 | P3 | CLOSED | `NumericalAgreement` now reports the maximum ratio to the exact combined tolerance envelope `atol + rtol * abs(reference)`, plus absolute difference and reference magnitude. Near-zero reference entries no longer produce a misleading relative percentage. |
| S6R-2 | P3 | CLOSED | Native R output schema v2 records `conf_int = false`; typed call and SSC models reject missing or changed clustering, `B`, weights, null imposition, engine, sampling, confidence-interval, or correction settings. |
| S6R-3 | P3 | CLOSED | Bootstrap loops catch closed singular-restriction and non-finite-replicate exception types. Message parsing is removed, and unrelated numerical errors propagate instead of being silently counted as discards. |
| S6R-4 | P3 | CLOSED | The SciPy full-refit test independently implements logistic means, weighted estimating equations, bread, cluster sums, CGM inclusion-exclusion, PSD clipping, and Wald calculation without production numerical helpers. |

## Original-remediation regression matrix

| Original finding | Regression status | Evidence |
|---|---|---|
| S6-F1 independent CGM/PSD/Wald | PASS | Regenerated base-R component meats, raw/projected covariance, eigenvalues, and Wald values remain byte-identical; production still passes the independent comparison. |
| S6-F2 p-value conventions | PASS | Native valid-only strict, production replenished `+1`, and full-support bracket conventions are unchanged and separately named. |
| S6-F3 full-refit fallback | PASS | Typed refit results remain real; sampled fallback replenishment and exhaustion are now directly tested. |
| S6-F4 BH corner handling | PASS | Symmetric-difference flags remain exact; equal-p families are deterministic under reversed input order. |
| S6-F5 R operations parity | PASS | Pre-draw, two seed sets, contract-driven `B`, and `conf_int = FALSE` agree across runner, documentation, typed output, and tests. |
| S6-F6 failure disclosure | PASS | Sampled `/B`, enumerated `/S_f`, separate counts, and strict `rate > 0.01` remain unchanged. |

## Oracle independence and numerical-primitives status

| Primitive | Evidence | Status |
|---|---|---|
| Native F6 raw Wald and p-value | External `fwildclusterboot` 0.14.3 | PASS |
| Production CGM meats and raw covariance | Independent base-R derivation | PASS |
| Production PSD covariance and projected Wald | Independent base-R eigendecomposition | PASS |
| Unrestricted logit fit | Statsmodels Binomial GLM | PASS |
| Perturbed restricted/unrestricted refit | SciPy root plus independently written equations and full statistic | PASS |
| Stacked off-diagonal meat | Independent synthetic cluster-sum calculation | PASS |
| Sampled/enumerated p-value arithmetic | Exact boundary calculations | PASS |
| BH lower/upper decisions | Hand-derived corners, spillover, and tie-order tests | PASS |
| ALR, Dirichlet, Laplace, log-Cholesky | Assigned to Segment 7 | CORRECTLY_DEFERRED |

The classifications remain scoped honestly. Native R validates its linear statistic and
p-value. Base R validates the full linear CGM/PSD/Wald path. The independently written
SciPy calculation is a synthetic GLM implementation check, not an external package oracle
for the complete production GLM WCR extension.

## Canonical fixture regeneration

The canonical `linux/amd64` image was rebuilt from the digest-pinned R 4.4.3 base and
committed `renv.lock`. Regeneration from immutable `simplemode-v1-segment5` preserved:

- input SHA-256 `ce72007f916ec4b42d695b3ab4afac5de67444f3ef55dd31205f3d34561c1f2d`;
- contract SHA-256 `36d0bdaf49678a962e5866df30440a36a43ec541788f20ff82adfdf4ce3e8603`;
- covariance-reference SHA-256 `10f0bedc5f605e9b9107c2ac6b0eefce1f069c82aaf44e9353e1b9c8b42af7b3`;
- Rademacher-weight SHA-256 `9bb80124bdf16d20e95ce95f7364975581d7b48935a027ab066ec62d6ee8cc71`;
- native `p_f = 0.8878887070376432`;
- raw `W_obs = 0.024310482190530155`; and
- `invalid_t_count = 223`.

Only the intended native-output schema/call metadata, runner/image identity, and
provenance hashes changed.

## Lineage

- `simplemode-v1-segment5` remains the immutable Stage 5 source.
- `simplemode-v1-segment6` remains the immutable pre-remediation checkpoint.
- `simplemode-v1-segment6-postremediation-final` remains the immutable S6-F1–S6-F6
  checkpoint.
- `simplemode-v1-segment6-postaudit-final` is the current Stage 6 checkpoint and carries
  schema-v4 verification plus schema-v2 native-call evidence.

The current root stops after annotation ingest and the oracle. It does not unlock outcome
modeling or run later statistical stages.

## Verification

The post-fix pre-root gate on 2026-09-09 passed:

- **524 tests**, with no skips;
- **89.50% branch coverage**, above the required 88%;
- Ruff lint;
- Ruff formatting; and
- `ty` type checking.

Targeted Stage 6 execution passed 118 oracle, dependency-contract, bootstrap, covariance,
and CLI tests. Final acceptance additionally requires an offline oracle replay in the new
root, byte-identical deterministic Stage 5/Stage 6 tables, and a repeat of the complete
quality gate after root materialization.

## Residual risk

No P0, P1, P2, or P3 implementation finding remains open. The declared limitations are
unchanged: the complete two-way-clustered stacked GLM bootstrap is an extension beyond the
external linear package; nominal BH does not establish finite-sample FDR control under the
observed dependence; PSD projection can materially change finite-sample covariance; and
Layer 1 numerical primitives remain assigned to Segment 7.
