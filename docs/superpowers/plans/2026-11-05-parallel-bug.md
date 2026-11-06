# Parallel BUG implementation plan

**Goal:** Add the first-order parallel TTN BUG method, selectable in HEOM and TFD,
and measure numerical accuracy and actual threading performance.

**Architecture:** Implement Algorithms 1–4 of Ceruti–Kusch–Lubich–Sulz
([arXiv:2412.00858](https://arxiv.org/abs/2412.00858),
[published paper](https://doi.org/10.1137/24M1714502)). Reuse old-state QR
preparation and local Arnoldi actions. Solve every node in the old bases,
then assemble the evolved old/old block and explicit single-new-direction
blocks bottom-up. Truncate with the existing protected truncation.

**API:** `parallel_bug_step(...; threaded=true)`; `method=:augmented` (default)
or `:parallel` in `adaptive_evolve`, `heom_evolve`, and `tfd_evolve`.
`threaded=false` executes the same parallel BUG formula serially. Existing
example configuration `method=:fixed/:adaptive` refers to time-grid selection,
not these solver keywords. No new dependencies or commits are needed.

**Conservation:** Enrich the old state with the invariant's subtree spaces
before preparation. This makes the omitted new/new blocks orthogonal to the
invariant. Keep the optional root correction and expose uncorrected drift.

**Scope:** Autonomous linear SoP generators and existing binary TTNs.
Report per-node `h*eta` omitted-block norms; the paper's optional rank-expansion
retry strategy (section 4.6) is separate from step-doubling and is not included
in this first implementation. The diagnostics are not a global error bound.

## Tasks

- [x] Add independent two-leaf block-formula tests, full-rank/dense checks,
  deep-tree convergence, conservation, scaling/zero states, typed failures,
  and serial/threaded agreement; observe the missing-API failure.
- [x] Implement `src/parallel_bug.jl`, export the API, and share final
  truncation/correction with augmented BUG. Each task owns its Krylov workspace;
  join every task before propagating failures. Validate the complete solver suite.
- [x] Add fixed/adaptive wrapper tests, method dispatch with shared adaptive
  preparation, HEOM fused/split trace checks, and TFD dense-reference convergence.
  Validate both complete suites with one and four Julia threads.
- [x] Add a reproducible warmed benchmark (BLAS=1, augmented/parallel serial/
  parallel threaded, HEOM fused/split and TFD), CSV data and a plot. Document
  measured results, limitations, algorithm paper and consulted author repository.
- [x] Request one independent read-only review, resolve material findings,
  and record final verification.

## Review focus

Complex transpose/conjugation conventions; arbitrary node IDs and leaf order;
rank-deficient initial data and invariants initially absent from the bases;
failed worker tasks must finish before returning and preserve LocalSolveFailure;
adaptive preparation must match the selected method and the changed half-step state.

## Execution record

Implementation proceeds in the current workspace as requested. Pre-existing
code is the baseline; current package git status was clean when work started.

- Observed missing `parallel_bug_step` failure before implementation, then
  missing wrapper keywords before integration. Rectangular assembly dimensions
  and `missing` method validation were caught by tests and corrected.
- `Pkg.test("TTNSolver")`: 553/553 on both one and four Julia workers.
  `Pkg.test("TTNDynamics")`: 374/374 on both configurations. BLAS threads=1.
- Five warmed runs per case/variant on one and four workers; raw results,
  convergence comparisons and inspected SVG/PNG are in
  `TTNDynamics/benchmarks/parallel_bug/`. Same-formula threading speedups on
  four workers range from approximately 1.00 to 1.28; common-step parallel
  errors are larger than augmented errors in these fixtures.
- Independent read-only review checked mathematical assembly against the
  paper, threading, adaptive preparation, benchmark medians and source hashes.
  A single-point docstring ambiguity was corrected; no findings remain.
  Hardware-wide scaling and physical cutoff convergence were not assessed;
  the report explicitly limits its claims to the measured finite models.
- `git diff --check` passed in both packages. README/guide/report local links
  resolve. Changes remain uncommitted for review.
