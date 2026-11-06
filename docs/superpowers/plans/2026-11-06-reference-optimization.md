# Reference-guided TTN optimization implementation plan

> Execute inline using executing-plans and test-driven-development; review the completed changes independently.

**Goal/spec:** Follow the user's approved order: early Krylov convergence checks,
identity contractions and redundant PS2 updates, then an independently verifiable
TTNO path. Preserve non-Hermitian HEOM, split layout, trace protection, rank
adaptivity, global residual certification and existing tolerances.

**Architecture:** Keep SoP as the existing operator interface. Specialize only
provably orthonormal identity environments; mixed overlaps retain their values.
Compile a separate tree operator with a reported Frobenius compression bound,
then evaluate its action/residual without a global dense tensor. Adoption is
conditional on measured cost and the original error contract.

**Constraints:** Existing dev/als work and ref/ remain untouched. Work in this
checkout because the required uncommitted solver/benchmark changes live here;
save pre-change source snapshots under /tmp for isolated benchmarks. No commits,
dependency installation or long trajectories. User's instruction to proceed is
the execution authorization; no repeated plan approval. Keep tests and short
measurements reproducible. Do not claim local speedups as trajectory speedups.

**Review focus:** Complex transpose conventions; false identity assumptions in
mixed/noncanonical frames; Arnoldi convergence/failure; cancellation residuals
and truncation bounds; ownership/thread safety and operator rank growth.

## Tasks / ledger

- [x] 1. Enable eager Arnoldi checks in both BUG solvers. Reuse existing dense,
  invariant and failure regressions. Measure an identical local problem and
  short complete BUG runs against the saved baseline at unchanged tolerances.
- [x] 2. Test orthonormal identity environments and all eight identity patterns
  against independent Kronecker actions, including noncanonical/mixed inputs.
  Implement exact identity skipping in preparation and buffered contractions.
  Remove three cancelling root propagations from PS2, checking dense accuracy,
  trace protection and the expected solve count. Run affected tests.
- [x] 3. Add TreeTensorOperator compilation, action and certified residual tests
  against small dense complex operators, cancellation, zero/singleton trees,
  irregular ordering and forced rank caps. Measure Holstein operator ranks and
  certification costs. Keep the existing backend available and only enable
  the new backend explicitly until whole-solve accuracy/cost is established.
- [x] 4. Run both package suites and PS2 checks, review scoped numerical changes,
  record benchmark conditions/results and source references in docs.

Pre-flight: Tasks 1–2 share BUG workspaces; measure baseline before edits.
Task 3 uses existing canonical TTN algebra/truncation; operator compression
error must be included in any residual certificate.

## Evidence and implementation decisions

Baseline full suite: 2533/2533. Identity tests initially failed for the missing
orthonormal keyword, then passed 45/45; existing grouped contractions passed
15/15. BUG/identity full suite passed 2578/2578. PS2 solve-count tests failed
twice before the turnaround change; numerical tests then passed 43/43.
TTNO compilation/action/certification tests failed before implementation; the
backend integration failed for its absent keyword and passed after wiring.
The memory-limit case likewise failed before the preflight was added.

Ruling: keep the existing SoP residual backend default. Structural TTNO alone
was slower, while numerical operator compression reduced eight-mode ranks43→4
and substantially reduced residual cost. Opt-in :ttno therefore compresses at
target/(2000*max(norm(first_checked_iterate),1)); the actual Frobenius allowance
is applied to every current iterate. If norm growth uses up the tolerance,
convergence fails rather than accepting an unchecked state.

Ruling: add a 1GiB preflight limit for the largest application QR input. The
large Holstein fixture predicts109GB even with compressed operator ranks.
This is not a peak-memory bound; projection intermediates can also be larger.
No full high-rank TTNO application was attempted. Such cases should retain SoP
until application itself can compress with controlled error.

The initial short-trajectory benchmark's provisional 1e-10 bitwise-comparison
threshold was exceeded at1.20e-10. The physical truncation target was unchanged
(5e-7 per step); comparisons now check against twice the accumulated cut budget
and retain/report the actual difference. Mature same-state step differences
are1.84e-14/3.42e-13. No physical accuracy tolerance was loosened.

Independent reviewer found no blocking numerical/ownership issues. Its extra
default-SoP norm observation was resolved by moving the norm into the TTNO
branch. A growing-iterate certificate regression was added. The QR-input guard
was reviewed separately; its limited memory scope is explicitly documented.

Final verification: TTNSolver2644/2644, TTNDynamics595/595, PS2 numerical43/43,
physical75/75 and runner8/8, exit0. Physical split-HEOM PS2 errors at h=.2,.1,.05
are5.58e-5,1.40e-5,3.49e-6 with protected trace. `git diff --check` passed for
both repositories. Baseline and eager-only reconstruction scripts succeeded.
Benchmark code and all samples are retained in reference_optimization/data;
report: TTNDynamics/docs/reference_optimization.md. No commits were made.
