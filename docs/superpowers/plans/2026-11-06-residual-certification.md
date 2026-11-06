# Residual certification optimization plan

> Execute inline with superpowers:executing-plans, test-first, one independent final review.

**Goal/spec:** User requests optimizing the dominant global residual verification,
consulting TTSolver where useful. Preserve mandatory global bounds, enrichment,
trace protection, fixed/adaptive evolution and threaded/serial ALS support.

**Evidence:** Eight-mode HEOM final residual check allocates1.081GB and takes
0.303–0.382s; 95% is truncate!, with the initial orthogonalization/copy around
half the profile samples. Source TTSolver/src/amen_solve.jl uses reusable local
contractions and projected residuals; those are not a replacement for this
solver's global certificate. No TTSolver changes needed.

**Approaches:** Pairwise squared-norm identities can cancel near convergence and
are rejected. All-term block assembly can inflate ranks/memory and is rejected.
Choose streamed exact canonical addition with leaf operator action, then the
same weighted truncation: build only projected nonzero blocks, QR bottom-up,
and omit the redundant initial QR/copy in compression. No altered tolerance,
operator-term order, or rank cap. Final root norm uses canonical storage.

**Architecture:** New src/tamen_certification.jl owns _tamen_add_action_canonical
and _tamen_residual. Extract existing truncation kernel into private
_truncate_canonical! (owned, root-orthogonal input required); public truncate!
retains validation/copy/rollback and delegates. The exact sum helper creates
independent data, contracts two input blocks separately into new child bases,
and concatenates parent columns only. Root combines signed coefficients.

**Constraints:** Existing uncommitted dev/als work preserved; no commits/pushes.
No public switch or test hook needed. No dense spatial expansion. Bound remains
norm(compressed residual)+sum(discarded bounds), excluding roundoff as before.
Use same weighted cuts and termwise budget. Single-leaf trees, zero terms,
complex coefficients, arbitrary root/ordering and unequal/overcomplete ranks
must work. No cross-call cached tensor/operator values or input mutation.

**Review focus:** Complex transpose conventions in canonical sum; loss of small
residuals by cancellation; canonical precondition and norm shortcut; compression
bound under tight rank caps; input ownership and public truncation rollback.

## Tasks / ledger

- [x] 1. Tests RED then GREEN for canonical acted sums and private compression;
  dense random complex sums, ownership, cancellation, zeros, singleton trees,
  irregular topology, rank cuts and certified tails. Preserve legacy residual
  implementation as test/benchmark reference, not production fallback.
- [x] 2. Wire optimized residual path. Test dense residual bound including
  forced discarded tails, near-cancellation residuals, cap failure, existing
  global/local residual controls and both ALS methods. Run both package suites.
- [x] 3. Benchmark warmed legacy vs optimized standalone certification and full
  HEOM solves (eight modes, split, same rtol/time grid, five repeated trials,
  plus a longer warm-start trajectory). Use isolated reference package process
  for whole-run comparison; no production monkey-patching. Record allocations,
  residual/total times, full-state differences and trace error. Document TTSolver
  source consultation and numerical limitations.
- [x] 4. Independent numerical/ownership review, any necessary fixes and final checks.

Pre-flight: The canonical add returns exact root-orthogonal output for the
private compression kernel; its final QR preserves that condition for direct
root norm. Legacy reference uses unchanged public truncate! semantics. Public
truncation continues to work on a private copy before publishing updates.

Ruling: additionally defer post-cut QR until the next canonical addition, and
perform it once before the final root norm. The exact addition supports arbitrary
input gauges, so repeated post-cut QR is redundant. Public truncate! retains
its canonical output contract; only the private kernel has canonicalize=false.
RED: missing canonicalize keyword in /tmp/residual-defer-red.log. Tests compare
cut bounds/state values and the following exact sum in the deferred gauge.

## Completion evidence

New tests104/104; full suites1828/1828 and595/595. Independent review approved
with no findings,5184/5184 additional assertions (many are exhaustive alias
pairs), including overcomplete complex gauges and tight caps. No scoped defect
was deferred. Existing uncommitted work was preserved.

Five-trial medians: standalone global certificate0.3177→0.2451s and
1.081→0.723GB allocated; first interval0.3808→0.3039s; five-interval accepted-warm
trajectory6.8009→5.2468s and17.264→11.877GB. Same10sweeps/6global checks for the
trajectory, endpoint relative difference2.23e-15, trace error<=9.27e-13.
Reproducible isolated-reference package helper and CSVs live in
TTNDynamics/benchmarks/tamen_certification; report TTNDynamics/docs/tamen_certification.md.

Review scope decisions: benchmark processes/assertions/CSV medians were checked
by the implementer; unrelated preceding changes were not re-reviewed. Neither
universal stability under arbitrary gauge conditioning/overflow nor a roundoff-
inclusive certificate is claimed. Both old and new bounds explicitly exclude
roundoff. These limitations do not weaken the previous acceptance contract.
