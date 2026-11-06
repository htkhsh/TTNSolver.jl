# TTNSolver.jl

TTNSolver provides tree tensor states, product-sum operators, and
time evolution for autonomous linear systems. Requires Julia 1.11 or later.

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. examples/basic_ttn.jl
```

## Time-evolution algorithms

Experimental **space-time TTN AMEn** uses Chebyshev or Crank–Nicolson time
discretization, recursive ALS/GMRES and residual rank enrichment, following
[Dolgov](https://arxiv.org/abs/1403.8085). `tamen_evolve` adapts interval lengths
using a refined-grid time defect; `tamen_step` solves a single interval.
Multiple linear invariants, reusable workspaces and accepted-interval warm starts
are supported. `als_sweep=:etter, threaded=true` enables sibling ALS solves
following [Etter's parallel tree sweep](https://doi.org/10.1137/15M1038852).
See [options and remaining TT parity gaps](docs/tamen.md).

The first-order recursive augmented BUG integrator updates subtree bases from
leaves to root, evolves in the Galerkin spaces augmented by old and updated
bases, and truncates the resulting ranks. Local
exponential actions use Arnoldi. `threaded=true` (default) updates sibling
subtrees concurrently, with parents waiting for their updated child bases.
See [threading](docs/augmented_threads.md) for execution details.

The first-order **parallel BUG** integrator [3] solves all node equations in
the old bases, then assembles and truncates the updated TTN.
Use `parallel_bug_step(...; threaded=true)` to run node solves concurrently.
See [parallel BUG](docs/parallel_bug.md) for options and limitations.

The standard workflow uses fixed time steps via `augmented_bug_step`, with
`h` and `h/2` convergence checks. Tensor ranks remain adaptive.

Optional `adaptive_evolve(...; method=:augmented)` step-size control supports
either algorithm (`:parallel` selects parallel BUG). It compares one full step with two
half steps and accepts the two-half-step result when the estimated whole-TTN
error meets the tolerance. Otherwise it rejects the attempt and retries with a changed step
size. The estimate accounts for truncation and invariant-correction errors;
it is local and does not guarantee the same global error.

The recursive augmented BUG method follows Ceruti, Lubich and Sulz [1]. The
step-doubling controller is the adaptive error-control layer; its general ODE
background is covered in [2].

See the [user guide](docs/guide.md) for tensor layout, contractions,
truncation controls, solver options, limitations, and related code references.

## References

1. G. Ceruti, C. Lubich and D. Sulz, “Rank-adaptive time integration of tree
   tensor networks,” *SIAM Journal on Numerical Analysis* **61**(1), 194–222
   (2023), [doi:10.1137/22M1473790](https://doi.org/10.1137/22M1473790).
2. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
   Equations I: Nonstiff Problems*, 2nd revised ed. (1993),
   [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1),
   for general local-error and adaptive step-size methods.
3. G. Ceruti, J. Kusch, C. Lubich and D. Sulz, “A Parallel Basis Update and
   Galerkin Integrator for Tree Tensor Networks,” *SIAM Journal on Scientific
   Computing* **48**(1), A27–A48 (2026),
   [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502).
