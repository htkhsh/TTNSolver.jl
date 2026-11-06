# TT-style acceptance and channel environments: implementation record

**Goal:** Find out why TTN tAMEn was hundreds of times slower than TT tAMEn on the
10-site Holstein HEOM. Then remove the work TT tAMEn does not do, without
weakening any default acceptance criterion.

**Diagnosis (measured before any code change):**
- `_tamen_step` truncated with `atol=0` (maxrank only). Truncating the saved
  3 fs state to relative 1e-4 shrank it from 366k to 54k parameters, with an
  unchanged distance to TT (3.5587e-4 vs 3.5592e-4).
- The global SoP certificate (102 terms) took 96% of the time.
- Local GMRES ran to about `1e-12*norm(b)`.
- A TT-shaped caterpillar tree was slower than the balanced tree (13.3 s vs
  7.6 s to 3 fs) with the same options. The remaining gap is therefore in local
  computation, not the tree shape.

**Constraints:** Existing uncommitted dev/als work stays intact. A pre-change
snapshot was kept outside the repository. Benchmarks record per-case source
hashes, so sources must stay frozen during a batch: one stage-1 batch has
inconsistent hashes for this reason and is quarantined.

## Tasks / ledger

- [x] 1. `truncate!(...; rtol)`: relative budget from the root norm after the
  internal orthogonalization. Tests: equivalence to `atol=rtol*norm`, the larger
  of the two budgets wins, validation leaves input unchanged.
- [x] 2. `tamen_step` options `truncation_rtol`, `certify=false`
  (`:residual_estimate` failures, `certified` field) and `local_tol_factor`.
  Tests: dense references, invariant protection in serial and Etter sweeps, warm
  starts, adaptive intervals, validation, defaults remain certified.
- [x] 3. Holstein 5 fs batch (`batch_tt_acceptance.jl`, configurable output root,
  two-interval warm-up so the warm-start path is compiled before timing).
- [x] 4. Projection error in the time-error estimate: recursive Gram matrices of
  orthogonal components, no squared-norm subtraction, `ttn_distance` fallback for
  non-orthonormal bases. Tests against dense projections and the old formula.
- [x] 5. `_ChannelOperator` (src/channels.jl): channels per edge (identity,
  completed, keyed open terms), coefficient at the lowest-degree support leaf,
  sparse node transitions. An operator-generic layer (`_env_leaf`, `_env_up`,
  `_env_down`, `_core_action`, `_leaf_action`, `_core_terms`, `_leaf_subtract!`)
  routes the serial and Etter sweeps, auxiliary residuals and the time-error
  estimate through it. The SoP path is numerically unchanged. Tests: every edge
  operator `ΣD⊗U` and every local action equal SoP for three-body, scalar and
  zero-coefficient terms on irregular trees. They also check HEOM channel counts,
  agreement of whole solves across threading and sweep modes, and the zero operator.
- [x] 6. Defaults `local_tol_factor=0.01`, `local_operator=:channels` after the
  benchmark. Benchmark specs pass both explicitly, so recorded results keep
  their meaning. TTNSolver 2829 tests, TTNDynamics 595, benchmark verify 16.

## Results

5 fs, medians of three:
- TT tAMEn (BLAS8): 3.222 s.
- TTN with options 1–3 and the time-error change: 15.467 s.
- TTN with channel environments as well: 8.725 s. The state error matches at
  6.66e-4, and the matvec count is identical (4826).

Details: `TTNDynamics/docs/holstein_tt_acceptance.md`.

## Next

Remaining cost at 5 fs: local GMRES (~59%), GC (~30%, 2.46 GB per interval),
environments (~16%). Candidates, in order:
1. Fewer allocations in local actions.
2. Skipping solves at full-rank leaves.
3. A local preconditioner or better initial guesses.
