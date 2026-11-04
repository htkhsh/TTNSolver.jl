# TTN Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

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

- [x] Write tree tests for balanced 1/3/5-leaf trees, explicit reordered leaves, arbitrary root IDs and malformed graphs/orders. Require `ArgumentError` for invalid topology.
- [x] Run `julia --startup-file=no --project=. test/runtests.jl`; expect failure because topology is not implemented.
- [x] Implement package setup and validated topology. Store copied vectors and compute traversal from root.
- [x] Run the same command; expect all tests to pass.
- [x] Commit the task.

## Task 2: TTN state storage and dense reference

**Files:** `src/states.jl`, `src/TTNSolver.jl`, `test/states.jl`, `test/runtests.jl`.

**Interfaces:** Consumes BinaryTree. Produces `TTNTensor(tree, leaves, cores)`, `product_state(tree, vectors)`, `leafdims(state)`, `tree_ranks(state)`, `dense(state)`, `copy(state)`.

- [x] Add independent complex dense-sum tests for the tree `[(0,0),(0,0),(0,0),(2,3),(1,4)]`, with leaf dimensions `[2,3,2]` and bond ranks `[2,2,2,3,1]`. Assert `isapprox(dense(x), reference; rtol=1e-12, atol=1e-12)`.
- [x] Add tests for physical order `[3,1,2]`, product states, one leaf, zero state, copied inputs and invalid arrays/keys/ranks (ArgumentError or DimensionMismatch as appropriate).
- [x] Run the complete suite; expect missing state API failures.
- [x] Implement validated storage, amplitude-preserving products and recursive dense expansion. Plain transpose, not adjoint, connects complex core coefficients to subtree bases.
- [x] Run the complete suite; expect all tests to pass, then commit.

## Task 3: Inner products, root orthogonalization and example

**Files:** `src/algebra.jl`, `src/TTNSolver.jl`, `test/algebra.jl`, `test/runtests.jl`, `README.md`, `examples/basic_ttn.jl`.

**Interfaces:** Consumes all preceding interfaces. Produces `LinearAlgebra.dot(x::TTNTensor,y::TTNTensor)`, `LinearAlgebra.norm(x::TTNTensor)`, `orthogonalize!(x::TTNTensor)` (returns x).

- [x] Add dense-reference inner-product/norm tests with unequal bond ranks; reject mismatched physical layouts. Test QR reconstruction and non-root Gram matrices against identity at `atol=1e-12` on complex, single-leaf, zero, deficient, overcomplete and non-postorder cases. Assert copies are not mutated.
- [x] Run the complete suite; expect missing algebra API failures.
- [x] Implement environment contractions and bottom-up QR with parent absorption, without dense expansion in dot/norm/orthogonalize!.
- [x] Document physical ordering and limits, and add a deterministic example which asserts reconstruction and norm preservation.
- [x] Run the complete suite and `julia --startup-file=no --project=. examples/basic_ttn.jl`; expect all tests/assertions to pass, then commit.
- [x] Run the package test through Pkg, with a writable temporary depot if necessary; inspect results. Request one independent whole-package review and address substantive findings before final verification.

## Completion evidence — 2026-11-04

- Julia 1.12.6: direct suite and `Pkg.test()` both passed 185/185 assertions.
- `examples/basic_ttn.jl` passed, with dense reconstruction error
  `4.002966042486721e-15` and preserved norm.
- Independent review found one substantive issue: computing a norm from
  `sqrt(dot(x,x))` lost small components in nearly cancelling bases and
  overflowed/underflowed on amplitudes `1e200`/`1e-200`.
- Five new numerical assertions reproduced the issue before the fix; QR on
  an independent copy plus the root-array norm made all five pass. Two
  additional assertions verify that querying the norm does not mutate inputs.
- No unresolved review findings. No changes to the existing packages.
- Declared compatibility is Julia 1.11+, but only the installed Julia 1.12.6
  was available for execution in this workspace.
