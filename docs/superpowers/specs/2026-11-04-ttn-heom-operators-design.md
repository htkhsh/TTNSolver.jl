# Product-sum operators and the first HEOM model

## Scope and selected approach

The user requested the next agreed increment after the TTN foundation:
product-sum operators and a small dense HEOM comparison. The previously
approved TTNSolver / TTNDynamics split and planned TFD support remain in force.

Use a topology-independent sum of local products first. Exact TTN action via
block sums provides an inspectable correctness baseline. Compiling a compressed
TTNO immediately would introduce operator-compression errors before the basic
HEOM conventions are verified. Lazy derivative sums are another option, but
would introduce a second state representation before BUG's requirements settle.

## TTNSolver operator interface

- `ProductTerm(coefficient, factors)` stores a finite ComplexF64 coefficient
  and a dictionary from positive physical mode positions to square finite
  nonempty ComplexF64 matrices. Missing factors mean identity. Empty factors
  mean a scalar multiple of identity. Constructor inputs are copied.
- `SumOfProducts(dimensions, terms)` owns positive integer local dimensions
  and copied ProductTerms; checks factor positions and sizes. It has no tree.
  Its stored metadata and factor arrays are read-only by convention.
- `leafdims(operator)` returns a copy of dimensions; `dense(operator)` returns
  the matrix acting on `vec(dense(state))`. Therefore mode 1 runs fastest and
  dense Kronecker factors occur in reverse physical order.
- `apply(operator, state)` and `operator * state` produce an independent exact
  TTNTensor on the same tree. Apply factors to leaf matrices, concatenate leaf
  bases and form block-diagonal internal cores. Apply each coefficient exactly
  once at the root. No compression, normalization or dense global expansion.
  Ranks can grow by the number of active terms; this is a reference action,
  not the eventual efficient BUG stepping path.
- Empty sums / zero coefficients act as zero. Single-root-leaf states are
  supported. Input states are never mutated.
- `dot(x, operator, y)` contracts each product through the tree without first
  building the rank-expanded action. It conjugates x but not the operator's
  coefficient. States must share topology/physical order and dimensions;
  operator dimensions must match. No Hermiticity assumption in TTNSolver.
- Runtime dependencies remain LinearAlgebra only; Julia compatibility 1.11+.

## TTNDynamics minimal HEOM layer

Create the companion package now to keep HEOM semantics out of TTNSolver.
Runtime dependencies: TTNSolver and LinearAlgebra. Local development resolves
the unregistered TTNSolver with `Pkg.develop(path="../TTNSolver")`; do not commit
an absolute path or a machine-specific Manifest.

- `HEOMMode(V, rate, forward, backward; scale=nothing)` owns a finite square
  Hermitian coupling, rate with nonnegative real part, two independent finite
  coefficients and a positive real scale. Default scale is
  `sqrt(max(abs(forward), abs(backward), eps(Float64)))`, as in the existing
  ExponentialCorrelation convention. Explicit scales permit matching NoiseExp.
  Backward coefficients are explicit: complex conjugate pole pairs must not
  silently assume backward[k] = conj(forward[k]).
- `HEOMTTNSystem(H, modes, hierarchy_sizes)` owns a finite Hermitian H and
  compatible modes. Each size is the number of states `n=0:size-1`; all are
  positive integers. Zero bath modes are supported. Use independent box
  cutoffs, not a total-tier cutoff. No terminator or residual correction.
- `heom_dimensions(system)` gives `[M^2, hierarchy_sizes...]`. `heom_tree`
  places the system leaf directly below the root and balances the bath subtree.
- `heom_generator(system)` returns a SumOfProducts using the formula below.
- `heom_initial_state(system, rho; tree=heom_tree(system))` creates arbitrary
  finite system matrix rho and bath vacua. No trace/positivity normalization;
  this also supports later response states. Custom trees map their leaf_order
  positions to the same physical dimensions.
- `trace_observable(system; tree=...)` is vec(I) times all bath vacua.
- `root_density_matrix(state, system)` contracts all bath leaves at n=0;
  it must not expand the full hierarchy.
- TFD remains a later model layer and shares the generic operator interface.

## Fixed HEOM convention

Use hbar=1 and mutually consistent time/energy units. For each mode k, let
`a[n,n+1]=sqrt(n+1)`, `adag=a'`, and `N[n,n]=n` (n is zero-based).
With `A^L = I kron A` and `A^R = transpose(A) kron I` for Julia's vec:

```
L = -i (H^L - H^R)
    + sum_k [ -rate_k N_k
              -i scale_k (V_k^L - V_k^R) a_k
              -i/scale_k (forward_k V_k^L - backward_k V_k^R) adag_k ]
```

The stored ADO is `rho_tilde[n] = rho_unscaled[n] /
prod_k(scale_k^n_k * sqrt(n_k!))`. Physical mode 1 contains vec(rho), ket
index fastest. Then hierarchy mode 1 is the fastest hierarchy index.
The existing TTSolver convention instead uses ket/bra/hierarchy factors in
Kronecker order. An explicit index permutation, tested for complex couplings,
connects the two dense matrices. Do not change existing packages.

## Verification and limits

- Test complex non-Hermitian product factors, scalar identity, empty sums,
  factor validation, reordered leaves, custom node numbering and single leaves.
- Exact action and three-argument dot must agree with independent dense products
  on entangled states; verify coefficient placement, input isolation and zero.
- HEOM's reference implementation loops over ADOs and multiplies density
  matrices directly, without using the product-sum builder or Liouville helper.
  Compare a one-pole model and a three-pole model including conjugate rates.
- Check nonzero hierarchy states and box boundaries, `c' * L = 0` on arbitrary
  states, initial/root density and custom tree equivalence. Verify the chosen
  scaling by similarity to the unit-scale operator.
- Supply a runnable dense comparison example. Both packages must pass their
  full suites via Pkg.test. Only Julia 1.12.6 is installed for local verification.
- No BUG, time integration, rank truncation, TTNO compiler, bath fitting,
  parallelism, TFD propagation or registry publication in this increment.
