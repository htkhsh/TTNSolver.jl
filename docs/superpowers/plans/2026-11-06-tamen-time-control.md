# TTN tAMEn time control and parallel preparation

Goal: advance toward the existing TT tAMEn behavior while keeping TTN state,
matrix-free local equations, conservative linear residual checks and HEOM trace
protection. This continues the user's implementation request; no extra approval
or commit is needed. Existing uncommitted prototype belongs to this task history.

Architecture: keep one-interval `tamen_step`, add Chebyshev/CN discretizations,
root-reduced refined-grid time defect, and adaptive `tamen_evolve`. Interpolation
separates output times from interval boundaries. A per-run workspace owns local
buffers; immutable upward environment construction may run on sibling tasks,
while Gauss–Seidel center movement and parent commits remain sequential.

1. [x] RED/GREEN: CN agrees with dense trapezoidal steps; interpolation; refined
   time defect decreases with order/refinement and detects coarse time resolution.
2. [x] RED/GREEN: multiple protected observables, actual scratch reuse, deterministic
   threaded upward environments, separate-workspace concurrent calls and failure release.
3. [x] RED/GREEN: adaptive interval rejection/rollback, limits, independent saveat,
   reduced-space estimator and temporal-vs-linear tolerance separation.
4. [x] RED/GREEN: HEOM adaptive wrapper, fused/split trace checks and owned options.
5. [x] Examples, parity table vs TT tAMEn, references; full regression suites;
   one independent final reviewer and tests for substantive review findings.

Numerical choices: Chebyshev retains the existing nt unknowns plus initial node;
CN matches TT semantics: odd nt >= 3 including the initial node, with an explicit
initial-condition row. The time estimator is an integrated refined-grid defect in
the final spatial space, plus initial projection error; it is not a rigorous global
error bound. `time_error_damp=100` follows TT's safety convention. No HEOM 2-norm
renormalization. Use absolute tolerances for zero/tiny initial data.

Scope of this increment: autonomous homogeneous generators. TT's time-dependent
operator/RHS callbacks, nonlinear updates, 2-norm projection correction, local
AMEn residual sweeps and preconditioning are separate remaining gaps; document
them explicitly rather than accept unsupported options. No speedup claim for
the complete ALS sweep, and no large physics benchmark during implementation.

Verification: Julia 4 threads, BLAS=1, TTNSolver and TTNDynamics complete suites;
CN dense formula, off-node exponential comparison, independent threaded results,
rollback and maxstep/dtmin failure state, all-time multi-invariant conservation.
Review focus: CN initial sample indexing; shared scratch with concurrent solves;
saveat interpolation not changing mesh; relative-only zero scales; tight rank caps
must never accept inaccurate linear solves as a pure time-error success.

Progress: implementation and first full-suite verification complete; final review fixes below.

Progress update: Tasks 1–4 implemented, with missing-API RED runs followed by
GREEN: time schemes47, workspace/invariants21, adaptive driver43, adaptive HEOM20.
Full suites pass TTNSolver1049/1049 and TTNDynamics536/536 (4 threads, BLAS1).
Actual TT-vs-TTN same-problem endpoint differences: Cheb2.48e-16, CN5.66e-16.
Adaptive split example executes both schemes; output CSVs and parity table added.
One independent review dispatched as /root/review_tamen_time_control; no more
implementation agents or long model benchmarks. Existing workspace preserved.

Final review: no major numerical/concurrency findings. A valid subnormal output
time overflowed barycentric reciprocals. Regression RED47 pass/2 errors, scaled
weights fixed it, GREEN49/49. Single-thread workspace tests21/21 also pass.
Source audit additionally found the newly exported workspace name can conflict
with TT benchmark imports; verifying and pinning the TT benchmark binding.

Final verification complete: TTNSolver1051/1051 and TTNDynamics536/536, both
exit0 with four threads. Optional actual TT parity4/4; mixed benchmark import
regression RED ambiguous TAMEnWorkspace then GREEN2/2 after explicit TT binding.
Both git diff --check and documentation link checks pass. No remaining review
findings. Changes left uncommitted in the user's existing workspace.
