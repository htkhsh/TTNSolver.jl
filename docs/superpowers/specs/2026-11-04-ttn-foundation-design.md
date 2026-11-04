# TTN packages and first implementation

## Agreed direction

The user approved the proposed `TTNSolver.jl` / `TTNDynamics.jl` split on
2026-11-04 and confirmed that TTNDynamics will also support TFD. The first
increment is the TTN foundation described in that proposal. This document
records that agreed scope and the implementation choices below.

The reference is `/home/takahashi/TTN-HEOM × BUG 積分器 実装ノート.md`.
Its future features and unverified conservation claims are research input,
not requirements to implement all at once.

## Approaches

1. **Selected: two new packages.** TTNSolver contains general tree tensor
   algebra and integrators; TTNDynamics contains physical models. This follows
   TTSolver / TTDynamics and allows independent incremental verification.
2. **Alternative: TTNSolver plus the existing TTDynamics.** This shares physics
   builders immediately but requires adapting interfaces tied to TTTensor and
   TTMatrix. Revisit shared model definitions once both backends have working
   implementations; do not extract a third package now.

## Responsibilities and future compatibility

- TTNSolver: topology, tensor states, contractions, orthogonalization,
  truncation, product-sum operators and eventually TTNOs, BUG, generic linear
  invariant handling. It must not depend on HEOMKit or either TT package.
- TTNDynamics: HEOM and TFD models, local spaces, bath data, generator terms,
  initial states, observables, trajectories and examples. It depends on
  TTNSolver. A later increment will create this package.
- HEOM supplies a non-Hermitian generator and the trace functional. TFD
  supplies its Hamiltonian / generator and state observables. Physical leaf
  meanings and conservation policies belong to TTNDynamics. A TTN tensor does
  not assume normalization, a density matrix, a vacuum, or Hermiticity.
- Linear trace conservation and quadratic norm conservation are distinct;
  neither is imposed automatically by basic TTN operations.

## First increment: decisions made for the implementation

- Julia compatibility: `julia = "1.11"`. Runtime dependency: LinearAlgebra only.
- Numeric storage: ComplexF64. Test dependencies: Test and Random.
- `BinaryTree` stores an ordered rooted full binary tree, with `(0, 0)` for
  leaves, explicit physical `leaf_order`, and traversal metadata. One leaf is
  supported, including its role as root. Node numbering need not be postorder.
- `balanced_tree(nleaves)` builds a deterministic tree with leaves numbered
  `1:nleaves`, preserving that physical order. Constructors own their arrays;
  topology is treated as read-only after construction.
- `TTNTensor` stores a tree, leaf matrices `U[n, r]`, and internal cores
  `C[r_parent, r_left, r_right]` separately. The root rank is one. Every
  non-root node's parent rank matches its corresponding parent core dimension.
  Constructors validate dimensions, keys and finite nonempty arrays and own
  their tensor arrays. Input bases need not already be orthonormal.
- `product_state(tree, vectors)` consumes vectors in physical leaf order and
  preserves their amplitudes, including zero vectors.
- `dense(state)` returns an array whose axes follow physical leaf order, with
  Julia column-major `vec` semantics. This debug/reference operation is for
  small systems only. It does not inherit TTSolver's Kronecker vector ordering.
- `leafdims(state)` returns physical dimensions in leaf order;
  `tree_ranks(state)` returns node-to-parent ranks in node-ID order, including
  root rank one.
- `dot(x, y)` contracts environments without dense expansion, conjugating the
  first argument. The two states must have the same ordered topology, physical
  order and dimensions; bond ranks may differ. `norm(x)` is its Euclidean norm,
  computed from a root-orthogonalized copy to avoid cancellation in Gram
  matrices and unnecessary squaring of the amplitude range.
- `orthogonalize!(state)` performs bottom-up thin QR and absorbs each R into
  the parent. Every non-root subtree basis becomes orthonormal; amplitude and
  phase remain at the root. It preserves the represented tensor up to roundoff
  without normalization or tolerance-based truncation. Overcomplete bond
  dimensions may shrink to the local row dimension; rank-deficient inputs and
  zero tensors remain valid. `copy(state)` copies numeric storage independently.
- Scalar type generalization, compact identity leaves and storage optimization
  are later choices. Do not add placeholder APIs for them in this increment.

## Acceptance criteria

1. Tree validation rejects invalid roots/children, shared children, cycles,
   disconnected nodes and invalid leaf orders.
2. A nontrivial three-leaf complex TTN matches an independently summed dense
   tensor. Permuting leaf order permutes physical output axes correctly.
3. Product states, single-leaf states and zero states preserve input amplitudes.
4. Contracted inner products and norms match dense reference calculations,
   including states with different bond ranks.
5. Orthogonalization preserves dense values and norm, produces orthonormal
   non-root unfoldings, and works for rank-deficient/overcomplete inputs and
   arbitrary node numbering. Copying isolates mutations.
6. The package loads in its own project, has a runnable example and passes its
   complete test suite. Existing TTSolver and TTDynamics files are untouched.

## Subsequent increments

1. Product-sum action plus tiny dense HEOM reference. Fix ADO scaling and
   explicitly map fused system leaves to the existing ket/bra convention.
2. Serial augmented BUG, local integration and separately verified linear
   invariant enrichment / truncation. Record trace errors before any correction.
3. HEOM examples and the TFD Hamiltonian/state layer using the same TTN algebra.
4. Optimized TTNO construction, parallelization, parallel BUG and tree changes,
   driven by measured accuracy and runtime.

## Non-goals for this increment

No HEOM or TFD propagation, BUG, truncation, TTNO optimizer, external dependency
installation, GPU/distributed execution, registry publication or modifications
to existing packages. Success here is a verified representation and algebraic
foundation, not a working HEOM integrator.
