# Residual compression and Holstein comparison Implementation Plan

> Use executing-plans inline, TDD, and one independent final review.

**Goal/spec:** Execute the user's approved sequence: split global residual
timings; optimize unprotected weighted compression without weakening the error
bound; compare TTN tAMEn to TT tAMEn on the ten-site split Brownian Holstein
model through5fs. Retain the earlier approximately one-hour comparison budget.

**Architecture:** Cumulative per-workspace phase timers, reported as interval
deltas inside diagnostics. Preserve generic/protected truncate! and specialize
only unprotected global residual cuts. A single serial cutting workspace reuses
amplitude, complement, QR/SVD-input and projection buffers; per-node factors
remain valid until their descendants consume them. The same weighted QR,
singular-value tail criterion, cap and sum of discarded bounds are retained.
Use svd! on disposable scratch without identity/complement products.

**Tech Stack:** Julia/LinearAlgebra/LAPACK; existing TTNSolver and TTNDynamics
benchmark infrastructure. No new runtime dependencies or global BLAS changes.

## Constraints and review focus

Preserve existing uncommitted dev/als work; pre-turn snapshot is
`/tmp/ttn-before-residual-compression`; do not commit/reset/migrate it.
No squared-norm residual estimate, no changed tolerance or skipped acceptance.
Test complex transpose versus adjoint, tall/wide/zero/rank-deficient cuts,
shrinking/growing buffer shapes and topology, input/output ownership after
scratch reuse, restrictive caps and cumulative bounds. Verify timer deltas
when workspaces are reused; timers are nonoverlapping wall-time subsets of
global verification (not CPU-time sums across child tasks).

## Tasks

- [x] 1. Add failing tests for diagnostic phase fields/deltas; instrument
  addition, weighted QR, SVD, basis update, final QR. Measure the existing
  residual against the saved HEOM iterate, five warmed trials. Expected:
  nonnegative phase durations, sum bounded by total time, unchanged state.
- [x] 2. Add dense/reference and scratch poisoning/rank/topology tests for a
  private `_residual_truncate!` and workspace; observe missing-function RED.
  Implement weighted unprotected cuts with reusable buffers, removing identity
  products and using svd! on scratch. Preserve the generic/protected path.
  Run targeted tests, then full solver/dynamics suites. Expected: same bounds
  within floating-point tolerance, owned outputs, zero failures.
- [x] 3. Repeat matched certificate/five-interval timings, keep raw before/after
  data and phase measurements. Benchmark source snapshots remain separate.
  Require numerical agreement and unchanged acceptance; report actual gains.
- [x] 4. Reuse existing Holstein model construction and bounded TT–TTN overlap.
  Prepare a dedicated5fs TTN-tAMEn comparison runner; validate axis/model
  agreement on a small case before the ten-site comparison. Run reference,
  refined reference, TT candidate, and TTN candidate within about one hour;
  save progress every interval. Compare common-time state/density errors,
  propagation/setup times and ranks. Never label an unqualified run as an
  equal-accuracy speedup. Preserve all old output directories.
  Bounded comparison complete: TT reached5fs, TTN reached3fs; full5fs comparison
  was not achieved within the approximately one-hour budget. Only qualified
  common-time prefixes are compared; no5fs TTN time is claimed.
- [x] 5. Independent final numerical/code review, necessary fixes, documentation
  and final verification. Explain limitations of the actual measurements.

Pre-flight: timing helpers must be available at runtime to generic truncation;
the residual workspace owns cut scratch, and its existing lease encloses both
addition and compression. Generic truncate! keeps default `timings=nothing`.
The benchmark must disambiguate TTSolver.TAMEnWorkspace from TTNSolver's type.
