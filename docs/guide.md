# TTNSolver user guide

TTNSolver provides tree tensor arithmetic and autonomous linear time
integration. The companion TTNDynamics package builds HEOM and TFD models on
these primitives.

The package implements rooted binary trees, ComplexF64 TTN states, product
states, product-sum operators, local contractions, rank truncation and
augmented BUG integration. It requires Julia 1.11 or later and depends on
LinearAlgebra and KrylovKit 0.10.

```julia
using TTNSolver, LinearAlgebra

tree = balanced_tree(3)
state = product_state(tree, [ComplexF64[2im, 3], [1, 0, 2], [1, -im]])
reference = dense(state)                    # size (2, 3, 2)
orthogonalize!(state)                      # preserves amplitude and phase
@assert dense(state) ≈ reference
@assert norm(state) ≈ norm(reference)
```

Run from this directory:

```sh
julia --startup-file=no --project=. examples/basic_ttn.jl
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

## Representation

`BinaryTree(children; root=length(children), leaf_order=nothing)` uses node IDs
to describe ordered left/right children. `(0, 0)` marks a leaf. The default
physical order is ascending leaf ID; explicit `leaf_order` may permute it.
All nodes must form one tree. A single leaf can be the root. Topology and its
cached traversal arrays must be treated as read-only.

`TTNTensor(tree, leaves, cores)` accepts dictionaries keyed by node ID:

- `leaves[id]`: a matrix with shape `(physical_dimension, parent_rank)`.
- `cores[id]`: an array with shape `(parent_rank, left_rank, right_rank)`.
- The root's parent rank is one. The root can be a leaf or an internal node.

Constructors copy numeric arrays and validate dimensions and finite entries.
`copy(state)` copies numeric storage and shares the read-only tree. Input
bases need not be orthonormal. `leafdims` follows physical order;
`tree_ranks` follows node-ID order and includes root rank one.

`dense(state)` produces physical axes in `leaf_order`, so its vectorization
uses Julia column-major indexing. **This differs from TTSolver's Kronecker
ordering.** The [TTNDynamics HEOM adapter](../../TTNDynamics/src/heom.jl)
explicitly maps its fused `vec(ρ)` leaf to separate ket/bra indices. Dense
reconstruction is only for small reference problems and can require
exponential memory.

`dot(x, y)` requires the same ordered tree, physical order and dimensions,
but permits different bond ranks. It conjugates `x`. Inner products, norms
and orthogonalization operate on the tree without dense expansion.

`norm(state)` orthogonalizes an independent copy and applies the standard
array norm to its root. This avoids the loss of small components from
`sqrt(dot(state, state))` when bases nearly cancel, without mutating the state.

`orthogonalize!` absorbs each QR factor into the parent, leaving the amplitude
at the root. It handles zero and rank-deficient states. Overcomplete local
dimensions can shrink, but no tolerance-based rank truncation is performed.
None of these operations normalizes the state or assumes a physical model.

## Product-sum operators

```julia
term = ProductTerm(-im, Dict(1 => ComplexF64[0 1; 1 0]))
generator = SumOfProducts(leafdims(state), [term])
derivative = generator * state
matrix_element = dot(state, generator, state)
```

Factor keys are **physical mode positions** in `leaf_order`, independent of
tree node IDs. Missing factors mean identity; empty sums mean zero. All
coefficients and matrices can be complex and non-Hermitian. `dense(generator)`
acts on `vec(dense(state))`, with the first physical mode varying fastest.

`apply` / `*` are exact block-sum reference actions: ranks may grow by the
number of nonzero terms. They do not truncate or normalize. `dot(x, A, y)`
contracts each product directly and avoids creating that expanded TTN.

## Time evolution and local contractions

`operator_environments(operator, state)` builds owned contraction snapshots.
Rebuild them after changing a state. `local_apply(environments, node, x)`
implements the insertion projection `V' * L * V`; it does not apply an inverse
Gram matrix to a nonorthogonal insertion map.

`augmented_bug_step(state, generator, h; atol, maxrank, conserved, ...)` uses
QR complements and frozen sibling bases for its reduced equations. It augments
old and evolved bases, then integrates cores in expanded child spaces. Full
physical leaf spaces skip their K solve. Inputs remain unchanged; no dense
hierarchy, global generator matrix or rank-expanded derivative is built.
`threaded=true` (default) runs sibling subtrees concurrently when Julia has
multiple worker threads; parents wait for their children. Set `threaded=false`
for sequential execution of the same formula. Preparation and final truncation
are serial. Fixed and adaptive stepping both honor this option; see
[augmented threading](augmented_threads.md).

Local solves use non-Hermitian Arnoldi exponential actions. `krylov_tol` is
relative to the local initial norm, per unit time; temporary scaling restores
the original amplitude. `info.krylov_error_estimate` sums local estimates, not
a global error bound. `atol` controls truncation error per step. Augmented BUG
is first order in time, so assess temporal error by reducing the step size.

Pass `conserved=c` for a nonzero TTN linear functional with `c' * L = 0`.
Its subtree spaces are included during augmentation and protected during
truncation. The caller must supply a genuine invariant. Set
`correct_invariant=false` to disable the root correction; diagnostics retain
the uncorrected value and correction norm. States are not normalized.

`truncate!(state; atol, maxrank, conserved)` is also available independently.
Its default `weighting=:weighted` carries complementary amplitudes downward
with thin QR before selecting each SVD space. This weights cuts by their
contribution to the full state, retaining the same protected directions and
an additive absolute error bound. `weighting=:local` restores conservative
local-core compression. `augmented_bug_step`, `parallel_bug_step`, and
`adaptive_evolve` also default to `truncation=:weighted`; pass
`truncation=:local` to use the previous policy throughout the solve.
See the [Holstein comparison](../../TTNDynamics/docs/weighted_tree.md).
Protected subtree spaces must already be present. The method reports a
conservative `discarded_bound`. A binding `maxrank` may miss the requested
tolerance (`tolerance_met=false`); protected directions cannot be discarded
to satisfy the rank cap. Outputs are root-orthogonal.

`enrich_basis(state, direction)` returns the same state in the union of both
inputs' subtree bases, including internal nodes. It preserves amplitudes and
does not mutate either input or expand a global tensor. For an initial product
state, `enrich_basis(state, generator*state)` can seed directions invisible to
the reduced BUG equations. The exact operator action may temporarily multiply
ranks by its term count. Keep added zero-weight directions through the first
step. This bootstrap does not detect every later missing direction.

### Shared BUG contraction kernels

Both BUG variants combine local product terms when two of the three factors
are exactly equal. Coefficients are absorbed into the remaining factor;
there is no equality tolerance or additional rank approximation. Parallel
BUG uses the same grouping for its rectangular cross blocks. Floating-point
summation order can change, so results need not be bitwise identical.

Upward, complementary and mixed environments reuse contractions with equal
input factors within each node invocation. These environment entries are
read-only and may share storage. Caches never persist across changed states
or operators. Large contraction buffers belong to the invocation; local
Arnoldi action outputs and distinct environment results own their storage.
The implementation does not change time steps, truncation settings or the
conserved subspaces.

## Adaptive integration

Use prescribed steps for routine runs after checking `h` versus `h/2` with
the desired observables. Tensor ranks remain adaptive. `adaptive_evolve` is
available for selecting time steps and for problems with varying time scales.

```julia
solution = adaptive_evolve(state, generator, (0.0, 1.0);
    dt=0.05, saveat=0:0.1:1, atol=1e-7, rtol=1e-5,
    maxrank=32, observe=norm)
```

The driver compares one full step with two half steps and accepts the
two-half-step result. `ttn_distance` computes differences by a TTN block sum
and QR. The estimate includes truncation and root-correction contributions;
acceptance uses `atol + rtol*max(norm(old), norm(new))`. These are local
estimates, not a global accuracy guarantee, and reduced-basis blind spots
still apply. `krylov_tol` is a separate local solver tolerance.

The full step and first half step share preparation of their common input:
root orthogonalization, old operator contractions and recursive complement QR
data. Their updated bases remain independent. The second half step prepares
its changed input independently; no cache survives the attempt. This reuses
work without changing the integration formula or truncation budgets.
Local Krylov actions reuse scratch arrays for mode products and permutations,
while returning a fresh result on every application. Scratch storage belongs
to one local solve; fixed-step evolution benefits from the same kernels.

`truncation_fraction=0.1` allocates that fraction of the error tolerance to
the full step and half of the budget to each half step. Trials are rejected if
a branch misses its truncation budget, an Arnoldi solve fails, or the error
estimate exceeds tolerance. Only `LocalSolveFailure` is retried. `dt` sets the
initial proposal; `dtmin=0` and `dtmax=Inf` by default. Steps land at requested
`saveat` times without interpolation, so output times can affect the mesh.
Endpoints are saved. `observe` receives independent snapshots; for large runs,
use an observable instead of storing whole states.

Results include `times`, `values`, sample `ranks`, `final_state`, `stats`, and
all-attempt `diagnostics`. `maxsteps=100000` limits total attempts and
`maxreject=20` limits consecutive rejections. Limits raise
`AdaptiveIntegrationError`, which contains the last accepted time/state and
attempt history. An insufficient rank cap may require increasing `maxrank`;
reducing `dt` alone need not resolve it.

### Optional timing

Pass `record_timing=true` to collect wall-clock seconds in `solution.timings`:

- `coarse_bug`: the full BUG step used for error estimation, including shared
  preparation and rejected attempts.
- `accepted_update`: the two half steps from accepted attempts.
- `rejected_update`: half steps from rejected attempts, including work before a local failure.
- `difference`: the complete TTN difference construction and QR/norm calculation.
- `candidate_norm`: the candidate state's norm calculation.
- `observations`: snapshot copying, observer callbacks and saved ranks.
- `other`: setup, controller work and remaining driver overhead.
- `total`: total driver time, including setup and timing aggregation.

Each attempt also has `diagnostics[k].timings`, with `coarse_bug`, `fine_bug`,
`difference` and `candidate_norm`. The `fine_bug` time is classified into
accepted or rejected work only after the acceptance decision. Partial work
before a failure is retained in the exception's attempt history. With timing
disabled, both timing fields are `nothing` and no clock reads are performed.

Compilation and GC are included when they occur. Warm up the timed and untimed
paths before repeated comparisons, and measure without concurrent workloads.
Model construction and initial enrichment performed by callers are outside
the driver total. The [measurement report](../../TTNDynamics/docs/adaptive_timing.md)
contains the fixtures, raw data, numerical checks and measured breakdowns.
The [preparation comparison](../../TTNDynamics/docs/bug_preparation.md) measures
the optimization against the previous implementation in the same process.

## Related code and papers

The selectable [parallel BUG method](parallel_bug.md) documents its update,
threading, invariant protection, diagnostic limits, and primary references.

- [TTNDynamics.jl](../../TTNDynamics/README.md) provides HEOM and TFD models.
  See the [HEOM adapter](../../TTNDynamics/src/heom.jl),
  [Holstein–TFD implementation](../../TTNDynamics/src/tfd.jl), and their examples.
- [TTSolver.jl](../../TTSolver/README.md) is the existing TTN solver; its
  `tt_mkron` convention is implemented in
  [TTSolver/src/tt_basic.jl](../../TTSolver/src/tt_basic.jl).
- The augmented BUG step follows the recursive rank-adaptive method of
  G. Ceruti, C. Lubich and D. Sulz, “Rank-adaptive time integration of tree
  tensor networks,” *SIAM Journal on Numerical Analysis* **61**(1), 194–222
  (2023), [doi:10.1137/22M1473790](https://doi.org/10.1137/22M1473790).
- The adaptive controller uses step-doubling around that first-order step. For
  general ODE local-error estimation and step-size control, see E. Hairer,
  S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I:
  Nonstiff Problems*, 2nd revised ed. (1993),
  [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1).
- Implementation decisions are recorded in the
  [foundation design](superpowers/specs/2026-11-04-ttn-foundation-design.md),
  [augmented BUG design](superpowers/specs/2026-11-04-augmented-bug-design.md),
  and [HEOM operator design](superpowers/specs/2026-11-04-ttn-heom-operators-design.md).
