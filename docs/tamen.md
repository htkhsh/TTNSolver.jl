# Space-time TTN AMEn-type solver (experimental)

`tamen_step` solves an autonomous homogeneous linear ODE on one interval;
`tamen_evolve` adapts interval lengths and evaluates independent output times. It follows
the space-time idea of TT tAMEn [1], using a binary tree instead of a chain.
This is a first implementation of **recursive Galerkin ALS with sweep-wise
residual enrichment**, not a verbatim implementation of a published tree AMEn
algorithm. `als_sweep=:etter` adds an Etter-style parallel tree sweep [6], with
an inverse-free parent starting vector. Its SPD convergence results are not
assumed for the non-Hermitian HEOM generator.

```julia
using TTNSolver, LinearAlgebra
x = product_state(balanced_tree(2), [[1,0], [1,0]])
sx = ComplexF64[0 1;1 0]
L = SumOfProducts([2,2], [ProductTerm(-im,Dict(1=>sx,2=>sx))])
y, info = tamen_step(x,L,0.2;time_points=8,rtol=1e-9,maxrank=16)
@show info.converged info.relative_residual_bound info.timings
mid_sample = time_snapshot(info.solution,0.13)
result = tamen_evolve(x,L,(0.,1.);dt=0.5,saveat=0:0.1:1,
    time_points=8,rtol=1e-6,atol=1e-9,solve_rtol=1e-10,
    workspace=TAMEnWorkspace(),threaded=true)
```

## Time discretization and topology

`chebyshev_collocation(nt,h)` uses `nt+1` increasing Chebyshev–Lobatto nodes,
including the known initial point. After eliminating that point,

```text
A X = b,   A = D ⊗ I_space − I_time ⊗ L,   b = d0 ⊗ x0,
D = D_full[2:end,2:end],   d0 = −D_full[2:end,1],   D*ones(nt) = d0.
```

With `time_scheme=:cn` (or `"CN"`), `time_points=nt` is odd and at least 3,
including the initial point, matching the TT interface. The uniform grid uses
`A = G ⊗ I − M ⊗ L`, where row 1 is the initial-condition equation and later
rows are the trapezoidal rule. No generator action on the initial TTN is
expanded to construct this right-hand side.

The last physical axis is time (Julia's first physical axis is fastest).
The new root joins the original spatial tree and one time leaf. All original
node IDs and physical leaf ordering are retained. The time leaf is always
the full identity basis of size `nt`; its bond is exempt from `maxrank`.
`time_slice(info.solution,j)` returns a spatial TTN on the original tree.
Index 1 is the unchanged input; `length(solution.nodes)` is the endpoint.
`time_snapshot(solution,t)` evaluates a relative time in the interval, using
Chebyshev barycentric interpolation or CN piecewise linear interpolation.
It anchors the exact initial value and does not extrapolate. Output interpolation
can temporarily increase spatial ranks by adding the known initial TTN; it does
not truncate or mutate the accepted endpoint. Neither extraction expands the
full spatial tensor.

The binary root adapts the supplied implementation note's three-way-root idea
to the existing binary topology. Local ALS operators remain sums of products.
Experimental `residual_backend=:ttno` compiles a tree operator for global
certification; `:sop` remains the default. See [tree operators](ttno.md) for
the error contract and intermediate-memory limitations.

## Sweeps, enrichment and convergence

Each sweep moves the orthogonality center by exact edge QR and solves local
Galerkin equations with non-Hermitian GMRES [3]. Subtree/operator and RHS
messages are updated on the traversed edge. The time leaf is not optimized.

By default (`residual_method=:local`), an auxiliary TTN approximates `b-A*X`
with ranks bounded by `kickrank`. Identity-operator ALS uses mixed `z'*A*X`
and `z'*b` contractions, without building the full product or a sum of TTNs.
Each local problem is an orthogonal projection, requiring no Krylov solve.
The auxiliary TTN is reused across solution sweeps. Its first seed is privately
randomized with a deterministic seed, independent of the caller's RNG, to expose
couplings invisible in the initial solution bases. `residual_sweeps=1` controls
the number of auxiliary sweeps per solution sweep. Cross-interval warm starts
also retain the auxiliary state when the temporal discretization matches.

The norm of this approximation is **not a residual bound**: a restricted or
poorly oriented auxiliary space can miss the residual completely. It requests
a global check when it reaches the tolerance. Independently, global checks run
every `residual_check_every=4` sweeps and on the last permitted sweep.
Only a successful global check can accept an interval. A failed check supplies
the residual for subsequent enrichment and restarts the auxiliary approximation
from its compressed directions. `residual_method=:global` retains the original
global residual/enrichment path on every sweep for comparison.

Global checks still form `b-A*X` through sequential binary TTN sums, compressing
after each operator term. The sum of discarded-norm bounds is added to the
compressed residual norm. This bound, excluding floating-point roundoff,
controls acceptance. `residual_maxrank` limits this verification TTN; when too
small, its compression bound can prevent convergence even for an accurate iterate.

Residual assembly fuses each signed product-term addition with bottom-up QR.
Each internal node contracts the two nonzero input blocks separately into the
new child bases, then factors their parent-column concatenation. It never builds
the zero-filled direct-sum core or a copied, fully acted TTN. A private compression
kernel consumes this already canonical sum without another copy/initial QR;
its post-cut QR is deferred to the next exact addition, with one final QR before
the residual norm. The public `truncate!` contract remains transactional and
root-orthogonal. Term order, weighted cut budgets, rank caps and accumulated
error bounds are unchanged, up to floating-point and degenerate-basis choices.
No squared-norm subtraction is used near convergence. See the
[certification optimization and measurements](../../TTNDynamics/docs/tamen_certification.md).

The residual supplies up to `kickrank` directions per bond for the next sweep.
Basis union preserves the current iterate; subsequent
solves activate new directions. Temporary ranks may exceed `maxrank`, while
spatial ranks after the sweep obey it. Zero-weight enrichment is not truncated
before the next solve. There is no claim of monotonic residual decrease for
general non-Hermitian problems; stagnation is reported as nonconvergence.

Acceptance requires `residual_bound <= max(atol, rtol*norm(b))`. These tolerances
control the **discretized linear system**, not the time or endpoint error.
Check temporal convergence with `nt` versus `2nt`, or halve interval lengths.
Even an exactly solved low-order collocation system can have significant time
error. The bound excludes floating-point roundoff and does not multiply by
the inverse operator norm to estimate solution error.

`residual_maxrank=nothing` uses twice `maxrank` (saturating at `typemax(Int)`).
Default `strict=true` throws `TAMEnConvergenceError` if `maxsweeps` is exhausted.
Its `diagnostics` retain the last iterate. `strict=false` returns that iterate
with `converged=false` for diagnosis only. Local GMRES failure always throws;
its exception records the failing node, local residual and work counts.
`history` contains `residual_estimate`, `relative_residual_estimate` and
`global_residual_checked`. Unchecked entries set all bound fields to `Inf`,
so a previous iterate's bound is never reported as a current certificate.
Final diagnostics always have a current global bound. `global_checks` counts
verifications and `residual_updates` counts auxiliary local projections.
`timings` separates ALS, residual formation, enrichment and temporal estimation.
`residual_seconds` is the sum of `residual_local_seconds` and
`residual_global_seconds`; do not add the total again to its components.
`timings.residual_global_parts` reports interval deltas for `addition_seconds`,
`weight_qr_seconds`, `svd_seconds`, `basis_update_seconds` and `final_qr_seconds`.
They are nonoverlapping wall-time subsets of global verification, excluding
minor bookkeeping; addition includes waiting for its child tasks. Reusing a
workspace does not accumulate earlier intervals into these diagnostic values.

Unprotected residual cuts use a specialized weighted compression kernel. It
reuses amplitude, complement, QR-input/reflector/R, SVD-input and projection
buffers, and avoids explicit identity/complement products when there are no
protected directions. SVD output and LAPACK's internal work arrays still
allocate. Per-node complementary factors have separate storage. This kernel
keeps the generic cut order, singular-value tail criterion and error bound;
the public `truncate!` and protected compression retain their existing path.
See [compression timings and validation](../../TTNDynamics/docs/tamen_residual_compression.md).
`environment_seconds` is the upward-environment portion
of `als_seconds`, so those two fields must not be added together.

## Linear invariants

For a supplied `conserved=c` satisfying `c' * L = 0`, the solver inserts all
spatial subtree directions of `c`, protects them during weighted truncation,
and finishes with a root Galerkin solve in a space containing `c ⊗ I_time`.
This last solve is necessary: retaining subtree directions alone does not
enforce every time-node constraint. `invariant_values` records all samples,
including the initial value. There is no trace correction or norm normalization.
Use `conserved_observables=[c1,c2,...]` for multiple invariants, mutually
exclusive with the legacy single `conserved` argument. Compression protects the
union of their subtree spaces, even when some observables are linearly dependent;
it does not protect only their sum. `invariant_history` has one row per observable
and one column per time node. Insufficient `maxrank` for that union is an error.

## Adaptive interval control

Use `tamen_evolve(x,L,tspan;dt,...)`. `dt` is the first attempted interval; `dtmin`, `dtmax`, `maxsteps` and
`maxreject` bound retries. `atol/rtol` are temporal tolerances here, while
`solve_atol/solve_rtol` control the space-time linear residual independently.

After a converged spatial solve, project the generator and initial vector into
the final orthonormal spatial space, interpolate onto a finer grid, and evaluate
the integrated temporal defect. Chebyshev doubles the unknown time points; CN
halves its substeps. Add the norm of the initial projection error. No second
spatial ALS propagation or dense spatial generator is needed.

Acceptance requires `time_error_damp * estimated_error <= atol + rtol * scale`,
with scale the larger of the initial norm and fine-grid RMS norm. Default
`time_error_damp=100` follows TT tAMEn's damping convention. Growth is limited
to 2, with safety 0.5 and exponent `1/nt` (Chebyshev) or `1/2` (CN).
`time_error_damp=0` disables temporal rejection but still requires linear
convergence. Failed linear solves and excessive time defects retry from the last
accepted state. Limits throw `TAMEnIntegrationError`, retaining that state and
the attempt history. User/configuration errors propagate without retries.

This reduced-space defect is a local estimate, not a certified global error
bound. Tolerance refinement remains necessary, especially for stiff/non-normal
problems. The driver includes zero-state handling and never normalizes HEOM data.

`saveat` is independent of interval boundaries; changing it does not change the
integration mesh. `observe` receives an independent interpolated TTN, so modifying
it cannot change propagation. History stores compact `.solve` diagnostics unless
`store_solutions=true`; accepted boundaries are in `accepted_times`.

## Cross-interval warm starts

`tamen_evolve(...;warm_start=true)` reuses the last accepted space-time solution
and auxiliary residual. This is the default; `warm_start=false` restores cold
interval initialization while preserving residual reuse within each interval.
Only one accepted hint is retained. A temporally rejected trial or failed linear
solve cannot replace it. Hints are local to the evolution call, not stored in
the workspace or diagnostic history, even with `store_solutions=true`.

For manual interval stepping, explicitly request and pass the owned hint:

```julia
y, d1 = tamen_step(x,L,0.2;store_warm_start=true)
z, d2 = tamen_step(y,L,0.1;warm_start=d1.warm_start,store_warm_start=true)
```

`TAMEnWarmStart` copies its solution, residual and time nodes; its shared tree
topology and all hint fields are read-only. Returned diagnostic solutions have
independent arrays. Concurrent readers can share a hint with separate workspaces.
It contains no operator, environment or acceptance certificate. The new initial
condition always constructs the RHS, and all environments are rebuilt.

The predictor retains the old spatial bases and replays the previous curve's
increments on a normalized interval, scaled by the duration ratio and anchored
to the new initial-state projection. It interpolates inside the previous interval,
never extrapolates its polynomial outside it. Changed point counts or Chebyshev/CN
schemes can reuse solution bases but reset the residual seed. Incompatible tree,
physical dimensions or initial state/previous endpoint cause cold fallback.
Endpoint compatibility is checked within 128 machine epsilons relative to the
larger state norm. A nonfinite duration ratio or predictor also falls back.

`warm_start_used`, `residual_warm_start_used` and `warm_start_status` describe
reuse. `timings.warm_start_seconds` measures transfer and hint storage. Direct
`tamen_step` calls store a new hint only with `store_warm_start=true` and linear
convergence; an adaptive caller must additionally require temporal acceptance.
A warm guess can change sweep counts, ranks and roundoff but never bypasses the
global residual or time-error acceptance checks.

## Workspaces and parallel execution

`TAMEnWorkspace()` owns one current scratch-buffer set per node and caches
unit-interval temporal grids per `(time_points,time_scheme)`. Changing rank or
node shape replaces that node's buffers. New projected operators are rebuilt
from current environments on every local solve. No stale environment survives
an interval or basis update. Global residual additions also reuse node-owned
projection, QR-input, reflector, Q and R buffers, whose capacities grow with the
largest encountered shape. Returned tensors own their arrays; subsequent reuse
cannot change earlier solutions. Discard the workspace to release retained
scratch. `workspace_stats` reports ALS buffer allocations/reuse and time-grid
cache hits, plus `residual_buffer_grows`, `residual_buffer_reuses` and
`residual_tasks` for each interval. These count buffer requests and spawned
addition tasks, not total allocations or LAPACK's internal work arrays.

`als_sweep=:serial` is the existing default: local solves follow a sequential
Gauss–Seidel traversal. With `threaded=true`, upward operator/RHS contractions
and independent subtrees of each global residual addition run concurrently.
The latter divides a bounded thread budget by estimated subtree QR work; each
node owns its scratch and both children join before their parent is processed.
`als_sweep=:etter, threaded=true` also solves sibling
subtrees concurrently; use `threaded=false` to serialize that same algorithm
for reproducible comparisons. Switching sweep algorithms can change convergence
and the required number of sweeps.

The Etter sweep follows Section 4.2, Definition 33 and Algorithm 4 of [6]: solve
the parent, prepare orthonormal child complements using SVD, freeze each child's
operator/RHS environment, solve the children independently, orthogonalize them
toward the parent, and solve the parent again. When preparing the next child,
cached upward messages are rotated with `Vt` only; the actual child receives
`S*Vt`. This distinction preserves the orthonormal complementary frames.
The RHS retains open source-child indices until its mixed contraction.

One adaptation avoids inverse singular values: the pre-branch parent tensor is
projected into the updated child bases as the starting iterate of the final
parent GMRES solve. No `S⁻¹` connector or singular-value floor is used. The final
Galerkin system matches the published construction; its initial Krylov vector
and finite-tolerance iteration path may differ. Zero singular directions are
retained without division. This is an Etter-style implementation, not a claim
of verbatim Algorithm 4 or SPD convergence for space-time HEOM.

Worker-owned node arrays, scratch slots and counters prevent concurrent Dict
or parent-core writes. Each parent joins both branches even on failure, and
only successful joined sweeps publish tensor arrays. Original convergence
exceptions are propagated. The pinned time leaf receives no solve task; the
whole task budget passes to the spatial subtree. The budget is split by subtree
node counts. Rank enrichment, auxiliary residual updates, weighted truncation,
the term-by-term certification loop, its final QR, and the final invariant root
repair remain sequential. Only independent work inside each residual addition
is threaded. See the [residual reuse/threading measurements](../../TTNDynamics/docs/tamen_residual_threads.md).

`info.als_sweep` records the algorithm; `als_parallel_tasks` counts spawned
solution-subtree tasks (not environment tasks). `timings.sweep_seconds` measures
main solution sweeps, including their preparation and initial upward contractions.
It is a subset of `als_seconds`, which also includes protected truncation and
root repair. `environment_seconds` overlaps with these timers; do not add them.
See [reproducible scaling measurements](../../TTNDynamics/docs/tamen_etter.md)
for whole HEOM solves and synthetic single-sweep timings.

Use distinct workspaces for concurrent solves. A lease rejects simultaneous or
reentrant use, and is released on exceptions; `tamen_evolve` holds it for the
entire run, including observers. Start Julia with multiple threads and ordinarily
use BLAS=1 when measuring task parallelism. The library does not change global
BLAS settings. See [Julia's threading guidance](https://docs.julialang.org/en/v1/manual/multi-threading/).
When loading both packages, qualify `TTNSolver.TAMEnWorkspace()` versus
`TTSolver.TAMEnWorkspace(ComplexF64)`; their same-named types are independent.

## Comparison with the existing TT tAMEn interface

| Feature | TT implementation | Current TTN implementation |
|---|---|---|
| Chebyshev / CN | Both | Both; same unknown-count convention for each scheme |
| Temporal interval adaptation | Reduced time residual | Refined-grid integrated defect plus initial projection error |
| Arbitrary output times | `extract_snapshot` | `time_snapshot`, independent `saveat` |
| Sweep limit / enrichment rank | `nswp`, `kickrank` | `maxsweeps`, `kickrank` |
| Local linear solver work | `local_iters` | `krylovdim`, `krylov_maxiter` |
| Multiple invariant vectors | `conserved_observables` | Same keyword, union of protected subtree spaces |
| Reusable scratch | `TAMEnWorkspace` | Same concept, TTN node ownership and concurrent-use guard |
| Cross-interval guesses | Retain solution and residual cores | Owned accepted solution/residual hints, rejection rollback |
| 2-norm projection correction | Optional `correct_2_norm` | Not implemented; no implicit normalization |
| Time-dependent/state-dependent operator and RHS | Callback interfaces | Not yet implemented; autonomous homogeneous SoP only |
| Residual enrichment | Local AMEn residual sweeps | Auxiliary residual ALS between solution sweeps, global verified acceptance |
| Compression modes | `trunc_norm` | Weighted truncation, global residual stopping |
| Solution truncation | Each local SVD at `tol/sqrt(d)` | Off by default; `truncation_rtol` per sweep |
| Acceptance | Local update/residual sizes | Certified global residual; `certify=false` uses the local estimate |
| Local tolerance | `tol/sqrt(d)` | `local_tol_factor*target` (default 0.01) |
| Local operator | MPO environments | Exact sparse channels by default; `local_operator=:sop` per term |

This is functional alignment, not bit-for-bit API or adaptive-mesh compatibility.
Defaults and local solver algorithms differ. Dense-output interpolation explicitly
includes the known initial point; TT's Chebyshev extractor interpolates its
unknown samples. This avoids changing the prescribed initial data at output time.
Remaining work includes operator/RHS callbacks, residual updates interleaved with
individual solution-node updates, a
preconditioner, parallel residual updates, and large Holstein
comparisons at matched full-state accuracy. No speedup claim is made here.

Validation covers polynomial differentiation, dense exponentials and collocation
systems, CN trapezoidal steps, temporal refinement, off-node output, rank growth,
residual bounds, multiple complex invariants, nonstandard trees, failed-solve
rollback and limits, independent output meshes, workspace failure recovery,
serial/threaded agreement and concurrent independent calls. HEOM tests cover
fused/split layouts and mandatory trace protection.

The optional [TT parity check](../examples/tamen/tt_parity.jl) runs both packages
on the same two-component rate equation with five time points, norm correction
off and a single interval. It is not part of the package test dependencies.
On 2026-11-06 the endpoint differences were 2.48e-16 (Chebyshev) and 5.66e-16
(CN). This checks discretization parity, not adaptive-mesh identity or large
low-rank performance; it can be rerun with the command in the script header.

## TT-style acceptance and channel environments

Four options remove work that TT tAMEn does not do. Since 2026-11-06,
`local_tol_factor=0.01` and `local_operator=:channels` are the defaults; they
change cost, not the acceptance criterion. Truncation and uncertified acceptance
remain opt-in. Pass `local_tol_factor=nothing, local_operator=:sop` for the
previous behaviour.

```julia
y, info = tamen_step(x, L, h; rtol=5e-5, maxrank=96, kickrank=4,
    als_sweep=:etter, threaded=true,
    truncation_rtol=1e-5,      # weighted truncation after every sweep
    certify=false,             # accept on the auxiliary residual estimate
    local_tol_factor=0.01,     # local GMRES to 1% of the acceptance target
    local_operator=:channels)  # exact sparse channel environments
info.certified                 # false: no global residual was formed
```

- `truncation_rtol` truncates the space-time solution with the weighted budget
  `truncation_rtol*norm(x)`, keeping protected directions and the time basis.
  Keep it below `rtol` (for example `rtol/3`). The final root solve cannot
  restore discarded non-root directions.
- `certify=false` never builds the global residual and accepts when the
  projected auxiliary residual norm reaches the target. That projection can
  underestimate `norm(b-A*x)`, so choose `rtol` tighter than the accuracy you
  need and validate against a reference. Requires `residual_method=:local`.
  A strict failure throws `TAMEnConvergenceError` with reason
  `:residual_estimate`.
- `local_tol_factor=f` (default 0.01) sets the local GMRES absolute tolerance
  to `f*max(atol,rtol*norm(b))`. `nothing` restores the previous rule, which
  with `krylov_tol=1e-12` resolves local systems to about `1e-12*norm(b)`.
- `local_operator=:channels` (default) builds, once per interval, an exact sparse tree
  operator from the SoP. At each edge a term is either absent (identity
  channel), complete inside the subtree (summed "done" channel), or open.
  Open terms are keyed by their factors on one side, and each coefficient is
  applied once, at the support leaf touched by the fewest terms. Nodes store
  transitions `(α,β,γ,c)`. Upward and downward environments and local
  operators all follow from these transitions. For split HEOM, a bath subtree
  edge carries `2+2*sites` channels and a bath leaf carries 4. This replaces
  roughly one environment per term. Identity channels in QR frames are exact
  identities. The global certificate, when requested, still uses the SoP.

`truncate!(state; rtol)` is the relative budget used by `truncation_rtol`.
The time-error estimate now computes the initial projection error by
recursively adding Gram matrices of mutually orthogonal components. It does
not subtract squared norms, and it falls back to `ttn_distance` if a spatial
basis is not orthonormal.

Tests compare every edge operator `Σ D⊗U` and every local action against
the SoP path. The test operators include three-body, scalar and zero
coefficient terms, on irregular trees. Solutions, time errors and invariant
protection are also compared.

## References and consulted code

1. S. V. Dolgov, *A tensor decomposition algorithm for large ODEs with
   conservation laws*, [arXiv:1403.8085](https://arxiv.org/abs/1403.8085).
   Source for the space-time formulation, alternating solves and invariant spaces.
2. S. V. Dolgov and D. V. Savostyanov, *Alternating minimal energy methods for
   linear systems in higher dimensions*,
   [doi:10.1137/140953289](https://doi.org/10.1137/140953289).
   Background for residual-based rank enrichment; not a convergence guarantee here.
3. [KrylovKit linear-system documentation](https://jutho.github.io/KrylovKit.jl/stable/man/linear/).
   The implementation uses the explicit GMRES algorithm and checks its result.
4. Existing local implementations [TTSolver/src/tamen.jl](../../TTSolver/src/tamen.jl)
   and [time_utils.jl](../../TTSolver/src/time_utils.jl) were consulted. The TTN
   sweep uses this package's contraction and truncation code; it does not convert
   to a TT or call the existing TT solver.
5. User-supplied design note *時空間 TTN-AMEn（tAMEn の木版）実装ノート*.
   Used as design input. The binary root, sum-of-products operator and sweep-wise
   residual enrichment above document the first-stage choices explicitly.
6. S. Etter, *Parallel ALS Algorithm for Solving Linear Systems in the
   Hierarchical Tucker Representation*, SIAM Journal on Scientific Computing
   38(4), A2585–A2609 (2016), [doi:10.1137/15M1038852](https://doi.org/10.1137/15M1038852).
   [Author manuscript](https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2015/2015-25_rev2.pdf),
   Section 4.2, Definition 33 and Algorithm 4; the parent initialization adaptation
   is described above.
