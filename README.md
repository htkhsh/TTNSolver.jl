# TTNSolver.jl

TTNSolver provides tree tensor states, product-sum operators, and
time evolution for autonomous linear systems. Requires Julia 1.11 or later.

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. examples/basic_ttn.jl
```

## Time-evolution algorithm

The first-order recursive augmented BUG integrator updates subtree bases from
leaves to root, evolves in the Galerkin spaces augmented by old and updated
bases, and truncates the resulting ranks. Local
exponential actions use Arnoldi.

The standard workflow uses fixed time steps via `augmented_bug_step`, with
`h` and `h/2` convergence checks. Tensor ranks remain adaptive.

Optional `adaptive_evolve` step-size control compares one full step with two
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
