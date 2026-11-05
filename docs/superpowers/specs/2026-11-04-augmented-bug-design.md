# Local contractions and augmented BUG

The user explicitly requested proceeding to local contractions and augmented
BUG after the product-sum foundation. Implement in the existing feature
checkout, preserving the TTNSolver / TTNDynamics split and existing examples.

Use the recursive augmented method of Ceruti, Lubich and Sulz,
[Algorithms 4–6](https://arxiv.org/abs/2201.10291), for autonomous linear
SumOfProducts generators. Keep parent-first core indices and ordinary
transpose when reshaping tensor coefficients; use adjoints for projections.
The reference implementation is sequential. Parallel BUG, TTNO compilation,
time-dependent/nonlinear right-hand sides and TFD models remain later work.

## Approach

Termwise upward/downward environments give local operator actions without a
global tensor, a global matrix, or rank-expanded derivative sums. The latter
already exists as an operator check but grows ranks on every RHS evaluation.
Compiling TTNOs is a later optimization. Local ODEs use KrylovKit Arnoldi
exponentiation (explicitly non-Hermitian), checking convergence; dense local
exponentials would have quadratic memory and are only used in tests.

For each child, QR the transposed child unfolding of the restricted old core.
Contract its orthonormal complement with the frozen old sibling environment
and parent reduced operator. Evolve the restricted leaf or recurse into the
subtree. Augment with old bases (including redundant old directions), evolved
bases and optional conserved-functional subtree bases. Project the old core
into the expanded children, then solve its Galerkin equation. Full physical
leaf spaces skip their K solve. Recompute expanded upward environments.

## Truncation and conservation

Root-to-leaf truncation projects the conserved subtree coordinates out before
an SVD of the remaining child unfolding. Retain the protected coordinates
exactly. One optional TTN functional is supported, including functionals with
nontrivial bond ranks; its subtree basis must already be represented.
Distribute the absolute tolerance as atol / number_of_edges, further scaled
by max(1, initial root norm) to bound amplification of local basis errors.
Report the sum of weighted local discarded norms as a conservative error bound. maxrank can
override this tolerance but cannot remove a protected direction; report a
missed tolerance instead of silently claiming success. No norm normalization.

The step includes the conserved functional during augmentation, then measures
its drift after truncation. An optional root-coordinate correction compensates
only integration/roundoff drift and reports its norm and pre-correction drift.
The caller is responsible for supplying a true invariant of the generator.

## Interfaces and checks

- operator_environments(operator, state), local_apply(environment, node, x).
  Environments are owned snapshots; local_apply computes the insertion-map
  projection V' L V, without assuming that V is an isometry.
- truncate!(state; atol, maxrank, conserved): report ranks, discarded bound,
  tolerance_met. Require conserved subtree spaces to be present.
- augmented_bug_step(state, operator, h; atol, maxrank, conserved,
  krylov_tol, krylovdim, krylov_maxiter, correct_invariant): independent state
  and diagnostics. Inputs unmodified, positive finite h, ComplexF64.
- TTNDynamics: heom_evolve(system, initial, times; ...), returning sample
  densities, traces, ranks, step diagnostics and the final TTN.

Tests use independent dense insertion maps for complex local contractions,
full-space exponentials, generated entanglement from a product, small singular
values, reordered/custom trees, zero generators/states, single leaves,
conservation under active truncation and rank caps. An HEOM example compares
actual TTN trajectories to the independent dense ADO equations and saves data
and a plot with errors, trace drift and ranks. No production dense expansion.
