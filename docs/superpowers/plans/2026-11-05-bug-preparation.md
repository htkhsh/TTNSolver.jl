# Reuse BUG preparation

The user authorized optimizing the measured extra BUG cost. Keep the numerical
method, tolerances, public APIs and fixed-step default unchanged. Implement in
the current working tree, preserving existing work; no commit is requested.

Approach: prepare the root-orthogonal input, old subtree operators and recursive
QR complements once for the full step and first half step. These depend only on
the input, operator and protected functional, not the step size. Each evolution
owns its updated bases. The second half step must prepare its changed input.
Keep preparation private and scoped to one adaptive trial so changed inputs,
operators, retries and callbacks cannot leave a persistent stale cache.

- [x] Add tests for reusable preparation, input/result isolation, nonstandard
  trees, protected invariants and equality to three independent BUG steps.
- [x] Split preparation from evolution in `src/bug.jl`; share preparation in
  `src/adaptive.jl`. Keep all setup time charged to the full-step timing.
- [x] Run the TTNSolver and TTNDynamics test suites.
- [x] Measure warmed baseline and optimized HEOM/TFD runs, compare saved
  observables, final states and attempt histories, and document actual gains.

Initial measurements show only small savings from preparation reuse. Extend
the optimization to the repeated local right-hand-side contractions: private
scratch arrays scoped to one local Krylov solve, with fresh output arrays on
every application. Test unequal complex axes against explicit Kronecker
operators and check that neither inputs nor previous outputs are overwritten.
Retain the contraction order, coefficient sum order and solver tolerances.

Review focus: single-leaf trees, zero operators, early local failures,
rank-cap rejection, mutation of a returned state and protected directions.
Baseline source is snapshotted under `/tmp/ttn-bug-before-20261105` for this run.

Completed: TTNSolver 489/489 and TTNDynamics 252/252 tests pass. Five alternating
pairs for each of five fixed/adaptive cases show 1.915–2.542× speedup and
71.4–79.5% less cumulative allocation. Final states and saved densities have
zero measured differences; saved ranks and adaptive attempt histories match.
Independent source reviews found no actionable issues in preparation or
workspace reuse. Results: `TTNDynamics/docs/bug_preparation.md`.
