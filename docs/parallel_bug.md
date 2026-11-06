# Parallel BUG

`parallel_bug_step` implements the first-order parallel TTN BUG method of
Ceruti, Kusch, Lubich and Sulz [1], Algorithms 1–4. It supports the same binary
trees, autonomous linear product-sum operators and invariant protection as
`augmented_bug_step`.

```julia
y, info = parallel_bug_step(x, L, 0.01; atol=1e-10, maxrank=64,
                           conserved=c, threaded=true)
solution = adaptive_evolve(x, L, (0., 1.); dt=0.01,
                          method=:parallel, threaded=true, conserved=c)
```

All tolerances and truncation diagnostics retain their existing meanings.
`atol` in a single step is a truncation budget; in `adaptive_evolve` it is part
of a local time-error estimate. `method=:augmented` remains the adaptive
driver's default. Fixed time grids remain the standard model workflow.

## Update

1. Orthogonalize the old TTN, build upward product contractions, and prepare
   the reduced initial coefficients and QR complement environments.
2. Solve every leaf K equation and every internal Galerkin equation in the
   **old** child bases. These equations are independent; each Arnoldi solve
   owns its scratch arrays. Full physical leaf spaces skip their K solve.
3. Assemble from leaves to root. Retain each old subtree basis, append the
   independent directions of its evolved coefficients, and form three blocks:
   the evolved old/old block, `h*F(Y_old)` projected onto new/old directions,
   and the analogous old/new block. Set the new/new block to zero. Mixed
   upward contractions evaluate these blocks locally.
4. Truncate with the existing absolute error budget and protected subspaces;
   optionally correct the linear invariant at the root.

There is no augmented-space ODE solve. Orthogonalization, assembly and
truncation still traverse the tree serially. No singular-value inverse or
dense global tensor/generator is formed. Zero/dependent candidate directions
are omitted instead of padding with zero basis columns.

## Conservation

For `conserved=c`, the old state is first enriched with `c`'s subtree spaces,
without changing its amplitudes. Thus the root old/old Galerkin space contains
`c`, and the explicit new-direction blocks are orthogonal to it. If `c'L=0`,
this preserves `c'Y` to local-solve and roundoff error before the optional
correction. The truncation also retains `c`. This argument does not imply
norm, energy, positivity, or Hermiticity preservation. HEOM trace tests run
with the root correction disabled, for fused and split system layouts.

`maxrank` must be large enough for the protected directions. In split HEOM,
the separate ket/bra bonds require the full system dimension. Preparation can
temporarily use larger ranks than the final cap.

## Threading and diagnostics

Run Julia with `--threads=4` (or set `JULIA_NUM_THREADS=4`) and normally set
`OPENBLAS_NUM_THREADS=1` to avoid nested parallel matrix products.
`threaded=false` executes the same mathematical method serially.
`info.threaded` reports whether the multi-worker path was actually used;
a single Julia worker falls back to serial execution.

All workers finish before assembly or failure propagation. An Arnoldi failure
remains a `LocalSolveFailure`, so the adaptive controller can retry a smaller
time step. Programming errors are not converted into recoverable failures.
Inputs, shared preparation and operator arrays are read-only during worker
execution; results and scratch storage belong to individual tasks.

Besides the common BUG diagnostics, `omitted_block_norms[node]` reports
`h*eta_node`, the Frobenius norm of the new/new derivative block discarded at
that node. Leaves have zero entries. These reduced local diagnostics are not
an estimate of the total global error and are not added to the step-doubling
estimate.

## Limits and validation

This implementation covers the base algorithm, not the optional basis-expansion
retry strategy of [1, section 4.6]. The adaptive driver uses ordinary step
doubling. In particular, neither step doubling nor `h*eta` can detect a
direction invisible to all current reduced equations. Use `enrich_basis`
when initial coupling directions are missing and check convergence in both
time step and rank tolerance. TFD already enriches with `H*initial` by default.

Parallel BUG has a first-order error bound under the paper's approximation
assumptions. Its explicit cross blocks can have larger error or invariant
drift than augmented BUG at the same step size. Compare performance at the
accuracy needed for the application. Small node equations may cost more to
schedule than threading saves; serial assembly and truncation also limit
speedup.

Tests cover a hand-derived two-leaf block update (including its nonzero
omitted block), dense full-space evolution, nonstandard trees, first-order
convergence, zero/tiny states, uncorrected invariants, forty leaves without
global expansion, threaded failures, and fixed/adaptive HEOM and TFD.
Model timings and plots are in the
[comparison report](../../TTNDynamics/docs/parallel_bug.md).

## Compact block assembly

The implementation stores only the new rows of each mixed operator
contraction. Empty new-direction blocks require no contraction; nonempty
blocks use scratch arrays reused across product terms. Full physical leaf
spaces retain their existing orthonormal bases and reduced operators.
This changes the amount of work, not the update formula, time order or
truncation tolerance. A frozen pre-optimization step is used for paired
trajectory and allocation comparisons in the
[optimization report](../../TTNDynamics/docs/parallel_bug_optimization.md).

## References and consulted code

1. G. Ceruti, J. Kusch, C. Lubich and D. Sulz, “A Parallel Basis Update and
   Galerkin Integrator for Tree Tensor Networks,” *SIAM J. Sci. Comput.*
   **48**(1), A27–A48 (2026),
   [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502).
   The publicly accessible [preprint](https://arxiv.org/abs/2412.00858)
   supplies Algorithms 1–4 and the section 4.6 diagnostic definition.
2. The authors' [reproduction repository](https://github.com/JonasKu/Publication-Parallel-BUG-for-TTNs),
   especially [the complex TTN step](https://github.com/JonasKu/Publication-Parallel-BUG-for-TTNs/blob/main/Section5.1/parallel_TTN_integrator/TTN_integrator_complex_parallel_nonglobal.m),
   was consulted for the block construction. This implementation uses the
   existing Julia SoP contractions and Krylov exponential actions; it does not
   vendor the MATLAB implementation or depend on its tensor toolboxes.
