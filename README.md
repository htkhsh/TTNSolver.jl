# TTNSolver.jl

Tree tensor foundations for Julia, intended to support BUG integration and
HEOM / TFD models through a future companion package, `TTNDynamics.jl`.

The first increment implements rooted binary trees, ComplexF64 TTN states,
product states, dense reconstruction, contracted inner products and norms,
and root-centered QR orthogonalization. It requires Julia 1.11 or later and
depends only on the LinearAlgebra standard library.

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
ordering.** A future HEOM adapter must explicitly map fused `vec(ρ)` leaves
to the existing separate ket/bra indices. Dense reconstruction is only for
small reference problems and can require exponential memory.

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

## Next increments

1. Augmented BUG and independently verified linear-invariant preservation.
2. HEOM and TFD propagation in TTNDynamics, followed by measured optimization.

BUG, time propagation, truncation and TTNO construction are not implemented
yet. The agreed package boundaries and first-increment design are in
[the design document](docs/superpowers/specs/2026-11-04-ttn-foundation-design.md).
