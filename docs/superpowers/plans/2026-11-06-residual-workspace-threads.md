# Residual workspace reuse and subtree threading

> Execute inline with executing-plans, TDD and one independent final review.

**Goal/spec:** User explicitly approves both proposed next optimizations: reuse
addition/QR buffers and thread independent residual-construction subtrees.
Preserve the existing global certificate, per-term cuts and solver acceptance.

**Architecture:** A private residual workspace owns per-node growing vectors
for core unfoldings/projections, joined QR matrix, block reflectors, Q and R.
Active reshaped views expose only current dimensions. Nodes own disjoint scratch;
return arrays never alias it. Use mul!/permutedims! and LAPACK geqrt!/gemqrt!
(same block QR as Julia qr!, blocksize36). LAPACK's own small internal work
allocations remain. Child R views are temporary messages, consumed before any
reuse. Workspace capacity grows to high-water demand per node, not a cache of
every historical shape. Public TAMEnWorkspace owns this scratch and reports
growth/reuse/task counters in workspace_stats.

Recursive bottom-up visits run independent children with a bounded thread
budget weighted by estimated QR work. Relatively tiny branches whose rounded
budget is zero receive no separate task (there is no absolute grain-size cutoff);
their work runs serially and the budget passes to the heavier branch. Join both
children on failure. Only the caller creates TTN dictionaries after successful
joins. Workspace leases reject concurrent reuse and release on exceptions.

`threaded=true` applies to residual addition as well as existing environments
and optional Etter ALS. Weighted cuts, term order and final QR remain sequential.
Private _tamen_residual(...;workspace=nothing,threaded=false) supports isolated
calls; an explicit workspace enables reuse across terms/calls. The public solver
always passes its leased workspace. No new public algorithm/acceptance switch.

**Constraints:** Existing uncommitted dev/als state stays in place; no commits,
pushes, global BLAS changes or other package edits. No squared-norm subtraction,
no change to truncation budgets/rank cap, no dense spatial tensor. O(number of
nodes) scratch slots; output ownership despite later reuse and topology changes.

**Review focus:** Complex coefficient/permutation conventions; tall/wide/zero
QR and rank changes; stale/growing-buffer views; node ownership and exception
joining; workspace leases/stats after failures/topology changes; rank-capped
certificates and fixed/adaptive/warm-start propagation under threaded defaults.

## Tasks

- [x] 1. Preserve pre-change canonical reference fixture. RED tests for buffered
  sums/QR, reuse after shrinking/growing ranks, topology changes, ownership and
  exceptions; implement node scratch and serial buffered addition.
- [x] 2. RED tests for threaded equality, actual tasks, failure/join/reuse and
  independent concurrent calls; add bounded subtree traversal, connect public
  workspace/threaded options and counters. Run full solver and dynamics suites.
- [x] 3. Benchmark same-iterate certification: previous code, new kernels with
  neither reuse nor threads, reuse only, threads only and both. Measure1/4/8
  threads where useful, plus previous/current five-interval HEOM trajectories.
  Warm paths, BLAS1, repeated medians, same tolerances/state checks. Record raw
  results and document limits including retained scratch and sequential cuts.
- [x] 4. Independent numerical/concurrency review, required corrections, final
  verification and report without claiming universal scaling.

Pre-flight: define residual-workspace types before TAMEnWorkspace (include
before tamen_certification.jl). Global residual helper uses its own lease nested
inside the existing whole-run lease; raw addition visits use that lease without
reentering. Node metrics merge only after all tasks joined, even on failure.
