# Cross-interval tAMEn warm starts

> **For agentic workers:** Use superpowers:executing-plans inline, with TDD and one independent final review.

**Goal:** Reuse the last accepted space-time solution and auxiliary residual as the next interval's initial guess while preserving certified acceptance and rollback.

**Spec:** User requested continuation after local residual updates; the announced next step is cross-interval warm starts, followed separately by Etter parallel ALS.

**Architecture:** An explicitly owned `TAMEnWarmStart` stores copies of a space-time solution and optional auxiliary residual. It is a guess, never a certificate or cached environment. Direct `tamen_step` calls accept an optional hint and optionally return one. Fixed/adaptive drivers keep only the most recent accepted hint locally, outside the workspace and compact diagnostics. Rejections cannot replace it.

**Design decisions:**
- Continue the existing uncommitted dev/als work in place, no automatic commits.
- Direct API: `warm_start=nothing`, `store_warm_start=false`; info has `warm_start`, `warm_start_used`, `residual_warm_start_used`, `warm_start_status`, and warm preparation/storage timing.
- Driver API: `warm_start::Bool=true`, opt-out for comparison. Driver controls `store_warm_start`; history omits the hint even when storing solutions.
- Check topology/physical dimensions and continuity with the preceding endpoint; incompatible hints fall back to cold initialization with an explicit status. The new initial RHS always comes from the caller.
- Reuse spatial bases and replay the previous interval's normalized temporal increments, scaled by the duration ratio and anchored to the new initial projection. Interpolation is only within the old interval, never polynomial extrapolation. A topology-compatible changed time grid/scheme can reuse solution bases; auxiliary residual reuse requires matching scheme and point count.
- Transfer and normalize the complete time basis, then use the existing protected truncation and ALS. All operator/RHS environments are rebuilt.
- Build hints only after a converged linear solve; adaptive driver commits only after temporal acceptance. Preserve exactly the last accepted state and hint on rejection/failure.
- Constant storage in number of intervals; concurrent calls use independent hints/workspaces. No mutable global cache.

## Review focus

1. A rejected time interval or failed linear solve must not replace accepted initialization state.
2. Changed duration/order/scheme/layout/initial data must not reuse stale matrices or mis-map temporal samples.
3. Input/output/cache aliasing must not affect earlier snapshots, observations or another solve.
4. Trace/invariant protection and the final global residual certificate remain independent of warm guesses.
5. Compact history must not accidentally retain one residual/solution copy per interval.

## Tasks / ledger

- [x] 1. RED/GREEN: owned hint, compatible transfer/rebasing, incompatible fallback, changed grids, direct solver integration; tests versus dense/cold solves and nonmutation.
- [x] 2. RED/GREEN: fixed HEOM and adaptive driver acceptance-only hint reuse, option ownership, compact storage, rejection rollback and opt-out.
- [x] 3. Whole-package tests and a warmed multi-interval eight-mode comparison at matched tolerances; record both speed and accuracy, including any regression.
- [x] 4. Independent review, necessary corrections and final evidence.

Pre-flight: core returns a hint only when requested and linearly converged; drivers may retain it only when the interval is accepted. The hint shares no mutable arrays with diagnostic solutions. Compatibility is geometric/initial-state continuity, not operator identity; every new solve constructs its own RHS and environments.

Task 1 RED: `/tmp/tamen-warm-red.log`, missing store_warm_start API. GREEN: `/tmp/tamen-warm-core.log` 45/45; exact nilpotent increment replay across duration/order/scheme changes adds 16/16 in `/tmp/tamen-warm-predictor.log`.
Task 2 RED: `/tmp/tamen-warm-driver-red.log`, Boolean driver option was passed incorrectly to the core. GREEN: `/tmp/tamen-warm-driver.log`, 508/508 plus existing adaptive43/43; `/tmp/tamen-warm-heom.log`, fused/split20/20.
Pilot: five eight-mode intervals, cold18 vs warm10 sweeps, relative endpoint difference7.52e-9. Timing not final; full five-repeat batch follows regression.

Full regression: `/tmp/tamen-warm-solver-final.log` 1667/1667 including concurrent shared-hint reads and forced-GMRES-failure nonmutation; `/tmp/tamen-warm-dynamics-suite.log` 556/556. Expected Krylov warnings are from intentionally insufficient local work limits.
Final benchmark: `/tmp/tamen-warm-benchmark.log` and `TTNDynamics/benchmarks/tamen_warm_start/data/`; five alternating warmed trials, cold9.236073s vs warm5.505938s, allocations25.8403/17.3411GB, sweeps18/10, checks7/6. Relative endpoint difference7.52e-9, warm trace drift6.11e-15; both satisfy per-interval residual tolerance1e-8. All ten trials and local documentation links verified.

Independent final review `/root/review_tamen_warm_start`: approved, no Critical/Important/Minor findings. Independent scratch tests60/60 cover CN/Cheb source-to-target remapping, changed duration/order, numeric/node ownership, and deliberately mutating observers. Script `/tmp/tamen-warm-independent-review.jl`.
Scope rulings: earlier prototype and general time-estimator accuracy were outside this incremental review; large physical-model/TT comparisons remain unmeasured; Etter parallel ALS remains the next separately implemented algorithm. No scoped defect was deferred. Final `git diff --check` clean in both repositories; all changes remain uncommitted in the existing workspace.
