# Parallel BUG optimization

**Goal:** Reduce work and temporary allocations without changing the first-order
parallel BUG formula, conservation, truncation policy or public API.

**Evidence:** A warmed sampling profile of the M=4, six-mode split HEOM case
attributes roughly half the active samples to local solves, 19% to preparation,
and 17% to the general core RHS used by assembly. Several assembly contractions
multiply blocks whose old-basis rows are explicitly zero; full leaf bases are
reorthogonalized even when their K solve was skipped.

**Design:** Store only new rows of mixed operator contractions. Skip empty
cross blocks; evaluate nonempty ones with reusable rectangular contraction
scratch buffers and accumulate into their corresponding slices. Preserve
old-first orthogonalization and all mathematical blocks. Reuse a full leaf
basis and its old upward operators directly. Compare against a frozen copy of
the pre-optimization implementation in the same Julia process.

## Tasks

- [x] Save the baseline and profile, then add tests for rectangular complex
  actions, empty/zero-coefficient sums, candidate ownership and full-basis skips.
- [x] Implement compact assembly and scratch storage; run the full solver and
  dynamics tests with one and four workers.
- [x] Measure warmed before/after fixed trajectories, stage timings and
  allocations; verify trajectories, ranks, trace/norm and omitted-block
  diagnostics. Add a larger-rank case and a matched-error comparison if useful.
- [x] Publish reproducible CSV/plots and conclusions in docs; obtain one
  independent read-only review and verify any necessary fixes.

## Constraints

Do not infer conservation from post-correction trace alone. Keep serial/threaded
equivalence, arbitrary node order, empty sums, and rank-deficient/tiny states.
No new package dependency, implicit rank retries, or change to default integrator.
The earlier uncommitted parallel BUG implementation belongs to this workspace;
preserve it and its original benchmark report as historical measurement.

## Execution record

The user authorized performance investigation and implementation; proceed
without an additional design-approval round. Work remains uncommitted.

- The new rectangular-kernel test failed on the absent helper, then passed
  against independent complex Cartesian contractions. Complete suites passed:
  TTNSolver 569/569 and TTNDynamics 374/374 on both one and four workers.
  A combined Pkg test process ended with code 143 before a summary; separate
  complete `test/runtests.jl` runs supplied the successful final verification.
- Paired measurement: 120 unique ordinary timing rows and 120 stage-timing
  rows, with five runs per case/execution/version; 30 additional runs at a
  common final-state error target of 1e-4. CSV values, figures and source
  hashes were checked; local documentation links resolve.
- M=4/six-mode HEOM allocation reductions are 35–40%; favorable whole-run
  speedups are 1.2–1.25. Some ODE-dominated cases have no measured speed gain.
  The larger four-site TFD uses rank cap 64, exceeds the truncation budget on
  two of ten steps in both implementations, and has different rank selections
  near truncation scale. Its final state difference is 1.10e-10 and density
  difference 1.52e-13; it is not claimed as a rank-converged calculation.
- At the common error target, the HEOM example favors optimized parallel BUG
  (259 ms versus augmented 404 ms), while TFD requires much smaller parallel
  steps and strongly favors augmented BUG. The default remains augmented.
- Independent review found no material defects. Its allocation-comment
  clarification was applied. Rank-history/omitted-block columns were explicitly
  inspected; the report includes the large capped case's limitations.
- Benchmark-only dynamic module binding access was made world-age safe after
  Julia 1.12 emitted warnings. A `--depwarn=error` smoke run confirmed that
  all stages are timed and its final state matches production to <1e-12.
- No rank-expansion retries were added. Further performance work should focus
  on local ODE/operator action and preparation; compact assembly is now smaller.
