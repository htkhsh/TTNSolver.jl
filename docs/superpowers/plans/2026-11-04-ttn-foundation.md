# TTN Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the first approved increment: tree topology and complex TTN state algebra suitable for later HEOM and TFD.

**Architecture:** Create TTNSolver independently of the existing packages. Separate topology, storage and contractions. TTNDynamics and its physical models are subsequent increments.

**Tech Stack:** Julia, LinearAlgebra, Test, Random.

**Spec:** `docs/superpowers/specs/2026-11-04-ttn-foundation-design.md`

## Global Constraints

- Julia compatibility: `julia = "1.11"`. Runtime dependency: LinearAlgebra only.
- Numeric storage: ComplexF64. Test dependencies: Test and Random.
- Root rank one; internal core order `(parent, left, right)`; physical axes in explicit leaf order.
- Preserve amplitudes; no automatic normalization or truncation.
- Work only inside the new TTNSolver package. No remote publication.

## Review Focus

- Complex transpose versus adjoint: explicit complex dense sums and conjugate inner products.
- Single-leaf and zero tensors: preserve amplitudes and define root gauge correctly.
- Arbitrary node IDs and physical order: non-postorder tree and permuted output axes.
- Redundant and rank-deficient bases: QR absorption must preserve represented values.
- Caller-owned input arrays: copies and constructors must isolate subsequent mutation.

## Task 1: Tree topology and package setup

**Files:** `Project.toml`, `.gitignore`, `src/TTNSolver.jl`, `src/trees.jl`, `test/runtests.jl`, `test/trees.jl`.

**Interfaces:** Produces `BinaryTree(children; root=length(children), leaf_order=nothing)`, its `children`, `parent`, `root`, `leaf_order`, `postorder` fields, and `balanced_tree(nleaves::Integer)`.

- [ ] Write tree tests for balanced 1/3/5-leaf trees, explicit reordered leaves, arbitrary root IDs and malformed graphs/orders. Require `ArgumentError` for invalid topology.
- [ ] Run `julia --startup-file=no --project=. test/runtests.jl`; expect failure because topology is not implemented.
- [ ] Implement package setup and validated topology. Store copied vectors and compute traversal from root.
- [ ] Run the same command; expect all tests to pass.
- [ ] Commit the task.

## Task 2: TTN state storage and dense reference

**Files:** `src/states.jl`, `src/TTNSolver.jl`, `test/states.jl`, `test/runtests.jl`.

**Interfaces:** Consumes BinaryTree. Produces `TTNTensor(tree, leaves, cores)`, `product_state(tree, vectors)`, `leafdims(state)`, `tree_ranks(state)`, `dense(state)`, `copy(state)`.

- [ ] Add independent complex dense-sum tests for the tree `[(0,0),(0,0),(0,0),(2,3),(1,4)]`, with leaf dimensions `[2,3,2]` and bond ranks `[2,2,2,3,1]`. Assert `isapprox(dense(x), reference; rtol=1e-12, atol=1e-12)`.
- [ ] Add tests for physical order `[3,1,2]`, product states, one leaf, zero state, copied inputs and invalid arrays/keys/ranks (ArgumentError or DimensionMismatch as appropriate).
- [ ] Run the complete suite; expect missing state API failures.
- [ ] Implement validated storage, amplitude-preserving products and recursive dense expansion. Plain transpose, not adjoint, connects complex core coefficients to subtree bases.
- [ ] Run the complete suite; expect all tests to pass, then commit.

## Task 3: Inner products, root orthogonalization and example

**Files:** `src/algebra.jl`, `src/TTNSolver.jl`, `test/algebra.jl`, `test/runtests.jl`, `README.md`, `examples/basic_ttn.jl`.

**Interfaces:** Consumes all preceding interfaces. Produces `LinearAlgebra.dot(x::TTNTensor,y::TTNTensor)`, `LinearAlgebra.norm(x::TTNTensor)`, `orthogonalize!(x::TTNTensor)` (returns x).

- [ ] Add dense-reference inner-product/norm tests with unequal bond ranks; reject mismatched physical layouts. Test QR reconstruction and non-root Gram matrices against identity at `atol=1e-12` on complex, single-leaf, zero, deficient, overcomplete and non-postorder cases. Assert copies are not mutated.
- [ ] Run the complete suite; expect missing algebra API failures.
- [ ] Implement environment contractions and bottom-up QR with parent absorption, without dense expansion in dot/norm/orthogonalize!.
- [ ] Document physical ordering and limits, and add a deterministic example which asserts reconstruction and norm preservation.
- [ ] Run the complete suite and `julia --startup-file=no --project=. examples/basic_ttn.jl`; expect all tests/assertions to pass, then commit.
- [ ] Run the package test through Pkg, with a writable temporary depot if necessary; inspect results. Request one independent whole-package review and address substantive findings before final verification.
