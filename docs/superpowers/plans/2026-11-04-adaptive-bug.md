# Adaptive augmented BUG implementation plan

The user approved implementing the previously proposed three changes:
step-doubling error control, rejection/retry, and separation of dt/saveat.
Continue in the existing feature checkouts, retaining all previous edits.

## Design

- TTNSolver provides `ttn_distance(x,y)` using an exact block difference and
  root QR, avoiding cancellation in norm(x)^2+norm(y)^2-2real(dot(x,y)).
- Local Arnoldi nonconvergence/nonfinite output uses a typed
  `LocalSolveFailure`; argument/layout/programming errors are never retried.
- `adaptive_evolve(state, operator, tspan; dt, saveat=nothing, atol=1e-6,
  rtol=1e-4, truncation_fraction=0.1, dtmin=0, dtmax=Inf, maxsteps=100000,
  maxreject=20, observe=copy, ...step_options)` owns the evolving state.
- Compare one full step and two half steps of the first-order method and
  accept the two-half-step state. The comparison uses whole-TTN norms, not
  just the physical density. Add the coarse truncation/correction bounds and
  twice the fine bounds to the raw difference to avoid hiding truncation
  errors in cancellation. This is an error estimate, not a global guarantee.
- Each branch must meet its truncation budget. The full budget is
  truncation_fraction*(atol+rtol*norm(current)); each half gets half of it.
  Compare the final estimate to atol+rtol*max(norm(current),norm(fine)).
  Scale proposed steps with safety 0.9 and exponent 1/2. Record every attempt.
- Reject without modifying the accepted state for failed error tests,
  truncation budgets, or typed local solve failures. Stop with a structured
  `AdaptiveIntegrationError` at dtmin, floating-point time resolution,
  consecutive rejection limit, or total attempt limit; include failure reason
  and attempt history. A permanently inadequate rank cap is not repairable
  simply by shrinking time steps.
- saveat is a finite increasing collection inside the time span; endpoints
  are included automatically. Clamp steps to output times and integrate there,
  without dense interpolation. observe receives independent snapshots. Store
  observed values and sample ranks, plus the final TTN and attempt diagnostics.
- Preserve the positional fixed-step HEOM API. Add
  `heom_evolve(system, initial; tspan, dt, saveat, atol, rtol, ...)` for adaptive
  integration, observing zero-tier density matrices.
- Nonlinear/time-dependent models, dense interpolation, second-order BUG,
  TFD and parallelization remain outside this change. Existing basis-enrichment
  limitations also apply to step-doubling estimates.

## Tasks and checks

- [x] Stable TTN distance and typed local failure; tests against dense small
  systems, near cancellation, tiny scales and 40 leaves.
- [x] Adaptive driver; test exact one-leaf dynamics, output-time alignment,
  rejection rollback, local failure retries, maxrank/dtmin/attempt limits,
  finite options, and input/callback isolation.
- [x] HEOM wrapper and example; independently compare tight/loose runs to
  dense ADO evolution, check traces without correction and preserve fixed API.
  Save adaptive step/error/rejection plots and CSV data.
- [x] Whole-package tests, inspect plots, independent code review and fixes.

## Execution record

Skills: writing-plans and test-driven-development; user authorization already
covers implementing the proposed design. No new approval or commits required.

Measured example: at rtol=1e-3/1e-5 the maximum density errors are 4.03e-4/
4.16e-6, accepted steps 31/44, rejected attempts 3/6; both save 21 samples.
Root correction is disabled and trace drift remains about 1e-14.

Independent review found rescaled local overflow escaping as ArgumentError,
rounded endpoints exceeding dtmax at large absolute times, unequal representable
half lengths requiring a Richardson adjustment, and empty-tuple saveat typing.
Reproduced the failures and added regressions. Fixes check the restored local
array, respect representable time bounds, use alpha=(h1/h2+h2/h1)/2 with coarse
weight alpha and fine weight alpha+1, and construct a typed save vector.

Final verification: TTNSolver `test/runtests.jl` passes 406/406 assertions;
TTNDynamics `Pkg.test(; allow_reresolve=false)` passes 131/131 on Julia 1.12.6.
The final example was rerun successfully and its CSV/plot output inspected.
Git whitespace checks are clean. All reviewed findings are addressed; local
error-estimator limitations and independent Krylov tolerance are documented.
Changes remain uncommitted in the existing working directories.
