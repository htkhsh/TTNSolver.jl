# Augmented BUG Implementation Plan

> Execute with superpowers:executing-plans and meaningful numerical tests.

**Goal:** Advance HEOM states as TTNs with recursive augmented BUG.
**Architecture:** Termwise local environments, Arnoldi local solves, protected
truncation in TTNSolver; model sampling and comparison example in TTNDynamics.
**Tech Stack:** Julia 1.11+, LinearAlgebra, KrylovKit 0.10.
**Spec:** ../specs/2026-11-04-augmented-bug-design.md

## Global constraints

ComplexF64; physical axes follow leaf_order; cores are parent/left/right.
No global dense expansion in solver paths; no two-norm normalization.
Keep existing work and package boundaries. Work in the current feature branch.

## Review focus

Complex transpose versus adjoint; rank-deficient QR; zero and one-leaf states;
rank caps incompatible with conservation; stale environments after augmentation.

## Tasks

- [x] 1. Add test/local_contractions.jl with dense insertion-map checks for all
  nodes and reordered trees. Implement src/contractions.jl and export APIs.
- [x] 2. Add test/truncation.jl for error bounds and conserved direction under
  active compression. Implement src/truncation.jl, validate options and caps.
- [x] 3. Add test/bug.jl comparing full spaces and rank-growing dynamics to
  matrix exponentials. Implement src/bug.jl and add KrylovKit dependency.
- [x] 4. Add TTNDynamics/src/evolution.jl and test/evolution.jl for model
  sampling, conservation and error reduction. Add an executable example and
  standalone plot/data; update both READMEs.
- [x] 5. Run both complete suites, run the comparison example, inspect plot,
  review the change, address findings, record measured limitations.

## Execution record

Authorization: user requested implementation directly within the already
agreed architecture; proceed without another design approval round.
Existing TTNDynamics visualization edits are preserved. No commit/push needed.

Implementation decisions: sequential autonomous linear version; KrylovKit
relative local tolerance by temporary scaling with amplitude restored; first
order temporal error; maxrank misses reported rather than automatic rejection.
Weighted truncation bound uses max(1, initial norm) to account for the external
coefficient map during top-down compression.

Task checks: independent dense insertion maps pass; active truncation protects
both rank-one and rank-two functionals; a 40-leaf step runs without global
storage; the expected Arnoldi nonconvergence raises an error. HEOM step
halving from 0.02 to 0.01 reduces max density error from 1.23e-3 to 6.09e-4.
An independent reviewer checked the numerical implementation.

Review: no critical algorithm issue. Additional random checks passed for
protected truncation bounds and non-product conserved functionals. The review
identified underflow in the root correction for c scaled by 1e-170; reproduced
with a zero generator, then replaced the squared norm by a stable normalized
direction. Regression cases include 1e-170 and 1e170 on one- and three-leaf
trees. Single-sample HEOM calls intentionally do not execute or validate
step-only solver options; this behavior is now explicit in the docstring.

Final verification on Julia 1.12.6: Pkg.test(; allow_reresolve=false) passes
329/329 assertions in TTNSolver and 102/102 in TTNDynamics. The example and
Matplotlib plotting script were rerun after the correction; fine-run trace
drift is 2.67e-15 and all truncation budgets are met. SVG/PNG were inspected;
git diff --check is clean in both packages. Source and artifacts are left
uncommitted in the existing checkouts for the user.
