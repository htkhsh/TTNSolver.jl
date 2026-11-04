# TTN HEOM Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add exact generic product-sum action and verify a minimal TTNDynamics HEOM generator against dense ADO equations.

**Architecture:** Topology-independent operators live in TTNSolver. The dependent HEOM adapter lives in a new sibling TTNDynamics package, with a direct dense reference used only for validation.

**Tech Stack:** Julia, LinearAlgebra, Test, Random; TTNDynamics depends on local TTNSolver.

**Spec:** `docs/superpowers/specs/2026-11-04-ttn-heom-operators-design.md`

## Global Constraints

- Julia compatibility 1.11+; numeric storage ComplexF64.
- TTNSolver runtime dependencies remain LinearAlgebra only.
- TTNDynamics runtime dependencies are TTNSolver and LinearAlgebra.
- No normalization, rank truncation or time evolution; box-cutoff scaled HEOM.
- Physical positions follow leaf_order; dense vec is Julia column-major.
- Existing TTSolver / TTDynamics / HEOMKit files remain unchanged.

## Review Focus

- Coefficients counted once at the root and not conjugated in matrix elements.
- Physical mode position versus node ID; arbitrary tree/leaf numbering.
- Empty and identity products, zero terms, one-leaf tensors and boundary sizes 1.
- Complex backward coefficients, scaling, vacuum trace and box boundaries.
- Nonmutation/ownership and finite validation after ComplexF64 conversion.

## Task 1: Generic product-sum operators

**Files:** `src/operators.jl`, `src/algebra.jl`, `src/TTNSolver.jl`, `test/operators.jl`, `test/runtests.jl`, `README.md`.

**Interfaces:** Produces ProductTerm, SumOfProducts, leafdims(::SumOfProducts), dense(::SumOfProducts), apply(::SumOfProducts,::TTNTensor), *(::SumOfProducts,::TTNTensor), dot(::TTNTensor,::SumOfProducts,::TTNTensor). Reuse existing tree/state APIs and factor out the tree contraction loop if useful.

- [x] Write tests: mixed factors with coefficients 2-im, -0.3im and identity; check dense via reversed kron, exact action and dot at atol/rtol 1e-12 on three_leaf_fixture, unequal ranks, physical order [3,1,2], one-leaf and zero cases. Reject bad positions/dimensions/nonfinite inputs; check array ownership.
- [x] Run `julia --startup-file=no --project=. test/runtests.jl`; expect missing operator APIs.
- [x] Implement owned, validated operator types, dense reference, block-sum action and contracted matrix elements. No full-state dense calls in apply or dot.
- [x] Run the complete suite; expect all tests to pass. Update README with rank-growth limits and commit.

## Task 2: HEOM adapter and independent validation

**Files (sibling ../TTNDynamics):** `Project.toml`, `.gitignore`, `src/TTNDynamics.jl`, `src/heom.jl`, `test/runtests.jl`, `test/heom.jl`, `examples/dense_heom_reference.jl`, `examples/heom_operator_check.jl`, `README.md`.

**Interfaces:** Consumes all operator APIs. Produces HEOMMode(V,rate,forward,backward;scale=nothing), HEOMTTNSystem(H,modes,hierarchy_sizes), heom_dimensions, heom_tree, heom_generator, heom_initial_state(system,rho;tree=...), trace_observable(system;tree=...), root_density_matrix(state,system).

- [x] Write direct ADO-loop dense reference and tests. Use complex Hermitian H/V, one real rate, and a model with one real plus two conjugate rates; sizes [2] and [2,2,2]. Compare full generator/action (atol/rtol 1e-12), trace null vector, density extraction and explicit old ket/bra index permutation. Add scale-similarity, custom-tree, no-bath, size-one and validation/ownership cases.
- [x] Create the package shell and resolve local TTNSolver with offline Pkg.develop; run `julia --startup-file=no --project=. test/runtests.jl`; expect missing HEOM APIs.
- [x] Implement validated model data, product-sum generator, initial/trace states, bath-vacuum contraction and default tree. No full-hierarchy dense call in production.
- [x] Run the complete TTNDynamics suite; expect pass. Add the comparison example and README with scaling, units, cutoff and setup instructions. Run the example; expect operator/action/trace errors below 1e-12. Commit in TTNDynamics.
- [x] Run Pkg.test in both repositories and the existing TTNSolver example; expect pass. Request one independent review across both repositories, fix substantive findings and record evidence.

## Execution record

Implementation proceeds inline as in the previous increment. TTNSolver continues
on its existing feature branch; TTNDynamics gets its own repository and feature
branch. Tests use a temporary writable Julia depot. Task 2 consumes Task 1's
operator API and the existing ComplexF64 TTNTensor; no interface conflicts.

## Completion evidence — 2026-11-04

- TTNSolver: all 237 assertions passed through the direct suite and Pkg.test.
- TTNDynamics: all 76 assertions passed through the direct suite and Pkg.test.
- Both examples passed on Julia 1.12.6. The HEOM comparison produced dense
  operator error 0, action error 8.143537164388336e-17 and trace derivative
  residual 0. These verify generator action, not time-integrator conservation.
- One independent review across both repositories found no unresolved issues.
  Its additional randomized checks passed 103/103 assertions.
- Task 1 code is TTNSolver commit 989488e. Task 2 code is TTNDynamics
  fb549fb..29a6fc3. The companion remains a separate local repository.
- Declared compatibility is Julia 1.11+, but only installed Julia 1.12.6 was
  available for execution. No changes to the existing TTSolver, TTDynamics
  or HEOMKit packages were made.
