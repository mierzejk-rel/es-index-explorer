# Stage 6 statistical-primitives remediation

## Executive verdict

`READY_FOR_NEXT_STAGE`

**YES — the Stage 6 numerical primitives and independent statistical oracles are sufficiently
correct, independently validated, deterministic, and contract-compatible for later stages to
consume.** This verdict concerns the Stage 6 numerical layer. The separate human-gold gate
remains locked, and no outcome model was fitted.

All six findings from the pre-remediation audit are closed. No original P0, P1, P2, or P3
finding remains unresolved. ALR, Dirichlet-multinomial, log-Cholesky, and Laplace primitives
remain correctly deferred to Segment 7.

## Finding closure

| Finding | Severity | Final disposition | Evidence |
|---|---:|---|---|
| S6-F1 | P1 | CLOSED | Native `fwildclusterboot` validates the raw linear statistic; independent base R validates full CGM meats, raw covariance, PSD mapping, and projected Wald; production Python matches both paths. |
| S6-F2 | P2 | CLOSED | Native valid-only strict p-value and production replenish-plus-one/full-support-bracket conventions are explicitly distinct in schema-v3 oracle evidence and regression tests. |
| S6-F3 | P2 | CLOSED | Full refits return typed restricted/unrestricted solutions, recomputed bread/covariance, and actual Wald results. Fallback contains no placeholder meat; one-step invariants are separate. |
| S6-F4 | P2 | CLOSED | Deterministic family decisions attach exact `AnalysisFlag.BH_INDETERMINATE` to every lower/upper-corner symmetric-difference family. |
| S6-F5 | P3 | CLOSED | Operations documentation now matches the unchanged runner's sign pre-draw, seed reset, contract-driven `B`, and `conf_int = FALSE`; a static parity test locks the contract. |
| S6-F6 | P3 | CLOSED | Sampled failure disclosure divides by `B`; enumeration divides by `S_f`. Combined count, denominator, rate, and strict one-percent trigger are typed and tested. |

## Specification-to-code status

| Requirement | Implementation | Independent evidence | Status |
|---|---|---|---|
| Linear external oracle | `statistical_oracle.py`, pinned R fixture | `fwildclusterboot` 0.14.3 under R 4.4.3 / linux-amd64 | PASS |
| Three-term CGM covariance | `cluster_covariance.py` | Independent base-R component meats and raw covariance | PASS |
| PSD projection and Wald | `cluster_covariance.py` | Independent base-R eigendecomposition and projected Wald | PASS |
| Bernoulli/linear scores, bread, Newton/KKT | `glm_primitives.py` | Statsmodels, score identities, restriction residuals | PASS |
| Stacked co-primary scores/meat | `stacked_scores.py` | Independent synthetic cluster-sum calculation | PASS |
| Sampled and enumerated WCR p-values | `wild_bootstrap.py` | Exact arithmetic boundary tests | PASS |
| Full-refit validation and fallback | `wild_bootstrap.py` | Independent SciPy root solve on nontrivial signs | PASS |
| BH bracket adjudication | `multiplicity.py` | Hand-derived corner and spillover tests | PASS |
| Deterministic seeds | `seeds.py` | Locked SHA-256 stream values and replay tests | PASS |
| Persisted provenance | oracle fixtures, manifest, verification artifact | SHA-256 tamper tests and deterministic regeneration | PASS |
| ALR / Dirichlet / Laplace / log-Cholesky | Segment 7 | Not a Stage 6 responsibility | CORRECTLY_DEFERRED |

## Oracle independence

| Primitive | Reference class | Result |
|---|---|---|
| Native F6 raw Wald and p-value | Independent external implementation (`fwildclusterboot`) | PASS |
| Production CGM meats/raw covariance | Independent mathematical derivation in base R | PASS |
| Production PSD covariance/projected Wald | Independent mathematical derivation in base R | PASS |
| Unrestricted Bernoulli-logit solve | Independent external implementation (Statsmodels) | PASS |
| Perturbed restricted/unrestricted full refit | Independent SciPy root solve | PASS |
| Stacked off-diagonal meat | Independent synthetic cluster-sum derivation | PASS |
| Sampled/enumerated p-value arithmetic | Independent exact calculation | PASS |
| BH lower/upper corners and flags | Independent hand-derived decisions | PASS |

The F6 raw covariance is materially indefinite. The persisted raw Wald is approximately
`0.0243104822`; the frozen PSD map produces approximately `0.0216641544`, a relative shift of
approximately `0.1088554`. Both values and the projection diagnostics are retained. This
difference is expected evidence, not hidden by substituting the native raw statistic.

## Bootstrap and multiplicity contracts

- Sampled WCR replenishes to exactly `B` valid replicates and uses
  `(1 + E) / (B + 1)`.
- Enumerated WCR uses the full support `S_f = 2^(H_f - 1)`, no `+1`, and the bracket
  `[E_valid/S_f, (E_valid + D)/S_f]`.
- Native R's valid-only strict exceedance mean is labeled as an external replay convention,
  not a production family p-value.
- One-step arm/intersection meat invariance is checked only under fixed restricted scores and
  bread. Full-refit fallback retains actual re-estimated covariance objects.
- Sampled failure disclosure uses `(singular + non_finite) / B`; enumeration uses
  `(singular + non_finite) / S_f`. Exactly one percent does not trigger.
- BH runs at both bracket corners. Every family whose decision changes carries
  `AnalysisFlag.BH_INDETERMINATE`, including threshold-spillover families.

## Determinism and artifacts

- Master seed: `20`.
- Named streams use the first eight big-endian bytes of
  `SHA-256("{master_seed}:{stream_name}")`.
- The R oracle seed is transferred as two signed 32-bit words.
- The auxiliary Rademacher schedule is pre-drawn, persisted, and consumed after resetting the
  same `dqrng` seed.
- Native output, independent covariance reference, weights, source files, image identity, and
  input are covered by provenance hashes.
- `simplemode-v1-segment6` remains the immutable pre-remediation checkpoint.
- `simplemode-v1-segment6-postremediation-final` is the final Stage 6 implementation
  checkpoint. It is prepared only through annotation ingest plus the oracle; the human-data
  and outcome-modeling gates remain pending.

## Verification

The pre-root quality run on 2026-09-09 collected and passed **512 tests** with **89.32%**
branch coverage against an 88% threshold. Ruff lint, Ruff format verification, and `ty`
type checking passed. Focused tests additionally establish:

- byte-stable external and covariance oracle replay;
- raw/projected covariance and Wald agreement;
- explicit native/production p-value convention divergence;
- nontrivial full-refit agreement with SciPy root;
- actual full-refit fallback without placeholder covariance;
- exact BH flag propagation;
- R runner/documentation/call-metadata parity;
- sampled 1/99 trigger versus 1/100 non-trigger;
- enumerated `S_f` denominator and separate failure counts.

The final root is accepted only if the same complete quality gate passes, the two-layer
oracle is recorded as complete, `outcome_modeling_unlocked` remains false, and the
pre-remediation root's file hashes remain unchanged.

## Residual limitations

- The GLM score bootstrap and stacked co-primary system remain declared extensions beyond the
  exact linear theory implemented by the external R package.
- BH is used at nominal `q=0.05`; finite-sample FDR control is not claimed because PRDS is not
  established for the overlapping two-sided family statistics.
- PSD projection can materially change a finite-sample covariance and is always reported.
- These are methodological/numerical limitations, not unresolved implementation findings.
