# BUG contraction optimization

User authorization: optimize the BUG family following the measured Holstein
10-site split bottleneck and the exact-grouping prototype. Work in the existing
development checkout so earlier uncommitted solver work remains available.

Goal: reduce repeated local products and environment contractions without
changing the integrators, public API, tolerances, rank caps or model.

Approach: exact local factor grouping in the shared core kernel; per-node,
per-invocation memoization of equal environment inputs. Global persistent
caches would risk stale states and concurrent writes; approximate operator
compression would introduce an additional accuracy parameter. Neither is
needed for these measured redundancies.

Execution uses writing-plans / executing-plans, tests first, in this session.
The prior diagnostic and the user's instruction establish the intended scope;
routine implementation choices do not require another approval round.

## Tasks

- [x] Save baseline sources and run existing tests (691 passing).
- [x] Add grouped-action tests against independent dense Kronecker products:
  complex rectangular factors, repeated factors, near-but-unequal factors,
  cancellation/zero terms, empty sums, input/output ownership.
- [x] Add exact grouping in `src/bug_workspaces.jl`; use it in both local
  core solves and `src/parallel_workspaces.jl` cross-block contractions.
- [x] Add environment tests for duplicate reuse, complex mixed contractions,
  unchanged caller inputs and fresh results after changed inputs.
- [x] Memoize equal pairs in `src/contractions.jl`; wire preparation and
  updated/mixed environments in `src/bug.jl` and `src/parallel_bug.jl`.
- [x] Reuse large environment buffers. Allocation regression failed at
  10.84 MB before the change, then passed below 2 MB for rank 20 / 12 terms.
- [x] Run solver and dynamics tests: solver 729/729 at 1 and 8 workers;
  dynamics 421/421 at 8 workers.
- [x] Measure both methods from the same saved mature Holstein state, warmed,
  against the saved baseline; record time, allocated bytes and state error.
  Three-trial medians: augmented 36.22 → 7.57 s; parallel 7.80 → 2.47 s.
  Both preserve the output to relative differences below 1.1e-15.
- [x] Run a short complete trajectory comparison if within the one-hour
  computation budget, review changes and document measured limitations.
  0–1 fs: augmented 490.40 → 98.15 s, parallel 291.12 → 99.51 s.
  Final relative state differences 3.93e-11 / 3.62e-8 remain far below the
  reference discrepancies (2.93e-3 / 2.19e-3); rank-cap misses are unchanged.
  CSVs and SVG/PNG figures are in TTNDynamics/benchmarks/holstein10_split.

Review focus: no tolerance-based equality; no coefficient lost or conjugated;
rectangular mixed spaces; caches never cross changed inputs; shared matrices
read-only, action outputs independently owned, worker scratch never shared.

Pre-flight: grouping consumes existing term coefficient and environment arrays;
memoization preserves their term indexing. These interfaces are compatible.

Read-only independent review found no actionable correctness issues in the
grouping, cache lifetime/ownership, rectangular contractions or buffered
complement transformation. Unstructured local-kernel overhead was identified
as a performance risk and measured separately, not assumed to improve.
