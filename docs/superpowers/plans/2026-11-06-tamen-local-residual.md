# Local residual TTN implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution and one independent final review.

**Goal:** Remove repeated global residual assembly from intermediate tAMEn sweeps while retaining the current certified acceptance check.

**Architecture:** Approximate `b-Ax` in an auxiliary rank-limited TTN by identity-operator ALS using mixed `z'*A*x` and `z'*b` environments. Reuse this auxiliary TTN across sweeps for basis enrichment. A small local estimate requests a global check; periodic and final checks prevent undetected stagnation. Only a global residual bound can accept a solution.

**Tech stack:** Julia, existing TTN contraction and gauge kernels, standard-library Random for a private deterministic seed.

**Spec:** The user's approved preceding proposal: auxiliary residual TTN, local residual updates, global verification before acceptance, small dense/invariant validation and an eight-mode timing comparison. Cross-interval warm starts and Etter parallel sweeps follow separately.

## Constraints and decisions

- Continue the existing uncommitted `dev/als` implementation in place; preserve unrelated changes, no automatic commits or worktree migration.
- Keep `residual_method=:global` as the old numerical/reference path; default `:local` after validation.
- Defaults: `residual_sweeps=1`, `residual_check_every=4`. Check globally whenever estimated residual <= target, every fourth sweep, and at maxsweeps.
- Unchecked history entries report `Inf` for bounds and explicit `global_residual_checked=false`; no stale bound masquerades as a current certificate.
- Failed global checks supply a compressed residual for subsequent enrichment/restarting the auxiliary approximation.
- No global dense assembly, squared-norm cancellation formula, normalization, or implicit trace correction.
- Local residual state and environments belong to one call. No new shared mutable scratch. Parallel ALS is out of scope.

## Review focus

1. Complex rectangular mixed contractions, arbitrary root/node/leaf ordering.
2. A rank-deficient auxiliary approximation underestimates the true residual: never accept without global verification.
3. Zero residual/zero initial state and product states with invisible multi-site couplings.
4. Diagnostic bounds describe the final iterate even at failed maxsweeps; adaptive driver preserves failure handling.
5. Existing scalar/multiple invariants, CN, time basis and workspace concurrency stay valid.

## Tasks / ledger

- [x] 1. RED/GREEN: dense-reference local residual ALS, nonmutation, bounded auxiliary ranks, deterministic RNG ownership, zero and adversarial states. Create `src/tamen_residual.jl`, `test/tamen_residual.jl`; include and standard-library dependency.
- [x] 2. RED/GREEN: integrate local/global modes and verification gating in `src/tamen.jl`; expose estimates, certificate flags, counts and split timings; demonstrate a false local estimate cannot cause acceptance, and legacy path equivalence.
- [x] 3. Validate dense HEOM split/fused, traces, temporal refinement/adaptation and all package tests. Compare warmed eight-mode local/global timing, allocations, sweeps, certificates and endpoint differences. Save benchmark and docs.
- [x] 4. Independent numerical/API review, address important findings, final verification.

Pre-flight: Task 1 returns an auxiliary TTN and projected norm; Task 2 may use the norm only as a scheduling heuristic, never as a bound. Task 3 measures `residual_seconds` (local + global) and its two new components without double counting.

Task 1 evidence: `/tmp/tamen-local-red.log` missing `_tamen_local_residual`, then `/tmp/tamen-local-unit.log` 14/14; two explicit blind-basis assertions subsequently passed in the full suite.
Task 2 evidence: `/tmp/tamen-local-control-red.log` rejects missing keyword, then `/tmp/tamen-local-green.log` 27/27 controls plus 75/75 prior tAMEn assertions.
Full regression: `/tmp/tamen-local-solver-suite.log` 1094/1094; `/tmp/tamen-local-dynamics-suite.log` 536/536. Expected Krylov warnings come from deliberate insufficient-iteration tests.

Benchmark: `/tmp/tamen-local-benchmark.log` and `TTNDynamics/benchmarks/tamen_local_residual/data/`. Global/local median 0.782371/0.397837 s; allocation 2.36308/1.15967 GB; both four sweeps, checks four/one, endpoint relative difference 2.315e-9, trace error <=1.78e-15.

Single-thread residual/control/workspace regression: `/tmp/tamen-local-single-thread.log`, 64/64.
Final independent review: `/root/review_tamen_local_residual`, no Critical/Important/Minor findings. Its independent run passed 151 assertions, including 20 randomized mixed-rank/complex/reordered-tree cases and failed-final-bound/zero-state checks.
Review scope decisions: cross-interval warm starts and parallel/interleaved basis updates remain explicitly deferred; large physical Holstein and TT-relative performance remain unmeasured; prior uncommitted prototype changes were outside this focused review. No new defect was set aside.
Final verification: documentation links valid, ten benchmark trials satisfy the residual and trace limits, both repositories pass `git diff --check`. Historical threading benchmark pinned to `residual_method=:global` so its stored results stay reproducible.
