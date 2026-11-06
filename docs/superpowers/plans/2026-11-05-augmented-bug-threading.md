# Augmented BUG subtree threading and scaling

The user requested a check of augmented BUG threading and a larger-system
scaling measurement. The existing augmented implementation is serial.

Implementation: add `threaded=true` to augmented BUG and honor the existing
option in fixed/adaptive HEOM and TFD wrappers. Independent sibling subtrees
run concurrently; a parent waits for both children before its Galerkin solve.
Use node-owned array slots and statistics, publish dictionaries sequentially,
and join children on failures while preserving `LocalSolveFailure` for the
adaptive driver. Preparation and final protected truncation remain serial.
Do not change the integration formula or parallel BUG algorithm.

- [x] Add regression tests for serial/threaded state, rank, invariant and
  diagnostic agreement; arbitrary trees, failures, preparation ownership,
  singleton/zero states, and fixed/adaptive model forwarding.
- [x] Observe the missing threaded API failure, then implement subtree tasks
  and deterministic statistics with serial fallback for one worker.
- [x] Run complete solver/model tests on one and multiple workers.
- [x] Measure larger HEOM fused/split and TFD cases at 1/2/4/8 Julia workers,
  BLAS=1, warmups and repeated paired serial/threaded trials. Retain CSVs,
  environment, truncation/invariant diagnostics and scaling figures.
- [x] Document the distinction between augmented subtree concurrency and
  all-node parallel BUG, measured speedup/efficiency and limitations. Obtain
  independent code review and verify final changes.

## Execution notes

- The new test first failed on the missing `threaded` keyword. Focused tests
  then passed 122 solver and 47 model assertions; complete suites passed
  691/691 and 421/421 on both one and four Julia workers.
- Initial test scaffolding assumed exact exponential evolution from reduced
  bases; the existing serial algorithm has time-discretization error. That
  assertion now checks a bounded numerical error, while serial/threaded
  equivalence is checked at roundoff and existing full-rank exact tests remain.
- Independent read-only review found no critical/important issues. The one
  stale HEOM adaptive docstring identified by review was corrected.
- A pilot with M=8/eight size-3 HEOM modes hit maxrank=64 on four steps. The
  main workload uses M=4/eight size-3 modes (104,976 state entries) and four-site
  TFD with size-3 oscillators (also 104,976 entries). Rank-cap misses are logged,
  not hidden or treated as an accuracy result.
- Final measurements use five steps over t=0…0.25, five paired repetitions,
  two warmups per execution mode, BLAS=1 and GC=1. The early-rank-growth window
  has max kept ranks 32 (HEOM) and 44 (TFD), zero truncation misses, matching
  rank histories/densities, and max serial/thread TTN distance 1.46e-15.
  Longer-run rank growth can change these performance characteristics.
- All 120 ordinary timing rows and source hashes were checked. Same-process
  median speedups at 2/4/8 workers are only 1.00–1.07; timing ranges overlap.
  A separate instrumented four-worker process provides 18 phase rows and
  per-node solve timings, with results checked against the ordinary solver.
  One internal core accounts for about 72% of local ODE time in HEOM and 92%
  in TFD, explaining why subtree concurrency offers limited speedup here.
- Benchmarks and figures are reproducible under
  `TTNDynamics/benchmarks/augmented_threads/`; detailed results are in
  `TTNDynamics/docs/augmented_threads.md`. All work remains uncommitted.
