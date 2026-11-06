"""A space-time interval did not meet its global residual tolerance.
For global nonconvergence, `diagnostics` contains the last iterate and residual
history; for local GMRES failure it contains the node and work/residual report.
The caller's initial state is unchanged. This is not an accepted time step.
"""
struct TAMEnConvergenceError <: Exception
    reason::Symbol
    diagnostics::NamedTuple
end
function Base.showerror(io::IO,e::TAMEnConvergenceError)
    print(io,"TTN tAMEn failed (",e.reason,")")
    d = e.diagnostics
    if e.reason == :residual_estimate && hasproperty(d,:relative_residual_estimate)
        print(io,": uncertified relative residual estimate=",d.relative_residual_estimate,
              " after ",d.sweeps," sweeps")
    elseif hasproperty(d,:relative_residual_bound)
        print(io,": relative residual bound=",d.relative_residual_bound,
              " after ",d.sweeps," sweeps")
    elseif hasproperty(d,:node)
        print(io,": node=",d.node,", residual=",d.residual)
    end
    print(io,"; inspect exception.diagnostics for details")
end

_full_leaf_basis(U) = size(U,1) == size(U,2)

# Move the orthogonality center across one edge. These exact gauge moves also
# shrink locally overcomplete bonds, but perform no tolerance-based truncation.
function _tamen_down!(x,parent,child,mode)
    C = x.cores[parent]
    M = _complement_matrix(C,mode)
    fac = qr(M)
    rank = min(size(M)...)
    Q,R = Matrix(fac.Q)[:,1:rank],Matrix(fac.R)
    folded = reshape(Q,size(C,1),size(C,5-mode),rank)
    x.cores[parent] = mode == 2 ? permutedims(folded,(1,3,2)) : folded
    if _isleaf(x.tree,child)
        x.leaves[child] = x.leaves[child]*transpose(R)
    else
        x.cores[child] = _parent_product(x.cores[child],R)
    end
    return Q
end

function _tamen_up!(x,parent,child,mode)
    leaf = _isleaf(x.tree,child)
    M = leaf ? x.leaves[child] : _basis_matrix(x.cores[child])
    fac = qr(M)
    rank = min(size(M)...)
    Q,R = Matrix(fac.Q)[:,1:rank],Matrix(fac.R)
    if leaf
        x.leaves[child] = Q
    else
        C = x.cores[child]
        x.cores[child] = reshape(Matrix(transpose(Q)),rank,size(C,2),size(C,3))
    end
    x.cores[parent] = _child_product(x.cores[parent],R,mode)
    return nothing
end

# Eager edge messages: recompute only the complementary message when descending
# and the subtree message when ascending. No message survives a changed basis.
function _tamen_sweep!(x,A,b,time_leaf;tol,krylovdim,krylov_maxiter,workspace,threaded=false,root_only=false)
    tree = x.tree
    modes = Dict(node=>mode for (mode,node) in enumerate(tree.leaf_order))
    environment_start = time_ns()
    up,cross = _tamen_upward(A,x,b;threaded)
    environment_seconds = (time_ns()-environment_start)/1e9
    down = [Matrix{ComplexF64}[] for _ in tree.children]
    rhsdown = Vector{Matrix{ComplexF64}}(undef,length(tree.children))
    down[tree.root] = _root_boundary(A)
    rhsdown[tree.root] = ones(ComplexF64,1,1)
    orthonormal = _assume_orthonormal(A)
    local_solves,matvecs,iterations = 0,0,0
    max_residual = 0.0
    function solve!(node)
        leaf = _isleaf(tree,node)
        initial = leaf ? x.leaves[node] : x.cores[node]
        buffers = _tamen_buffers!(workspace,node,initial)
        if leaf
            rhs = b.leaves[node]*transpose(rhsdown[node])
            action = _leaf_action(A,node,modes[node],down[node],initial;buffers)
        else
            left,right = tree.children[node]
            rhs = _parent_product(_project_children(b.cores[node],cross[left],cross[right]),rhsdown[node])
            action = _core_action(A,node,down[node],up[left],up[right],initial;buffers)
        end
        alg = KrylovKit.GMRES(;tol,krylovdim=min(krylovdim,length(initial)),maxiter=krylov_maxiter)
        result, report = KrylovKit.linsolve(action,rhs,initial,alg)
        local_solves += 1
        matvecs += report.numops
        iterations += report.numiter
        # KrylovKit's GMRES recomputes r=b-A*x explicitly before reporting
        # convergence, so normres is the actual local residual of `result`.
        error = report.normres
        max_residual = max(max_residual,error)
        if report.converged != 1 || !isfinite(error) || !all(isfinite,result)
            throw(TAMEnConvergenceError(:local_gmres,
                (;node,residual=error,local_solves,matvecs,iterations)))
        end
        if leaf
            x.leaves[node] = result
        else
            x.cores[node] = result
        end
    end
    function visit!(node)
        # A leaf basis spanning its whole space needs no solve: the parent is
        # solved again afterwards in the same Galerkin space.
        _isleaf(tree,node) && _full_leaf_basis(x.leaves[node]) || solve!(node)
        (_isleaf(tree,node) || root_only) && return
        left,right = tree.children[node]
        for (child,sibling,mode) in ((left,right,2),(right,left,3))
            child == time_leaf && continue
            Q = _tamen_down!(x,node,child,mode)
            down[child] = _env_down(A,node,mode,Q,down[node],up[sibling];orthonormal)
            B = _parent_product(_child_product(b.cores[node],cross[sibling],5-mode),rhsdown[node])
            rhsdown[child] = Q'*_complement_matrix(B,mode)
            visit!(child)
            _tamen_up!(x,node,child,mode)
            if _isleaf(tree,child)
                up[child] = _env_leaf(A,child,modes[child],x.leaves[child];orthonormal)
                cross[child] = x.leaves[child]'*b.leaves[child]
            else
                a,c = tree.children[child]
                up[child] = _env_up(A,child,_basis_matrix(x.cores[child]),x.cores[child],up[a],up[c];orthonormal)
                cross[child] = _basis_matrix(x.cores[child])'*
                    _basis_matrix(_project_children(b.cores[child],cross[a],cross[c]))
            end
        end
        solve!(node)
    end
    visit!(tree.root)
    # `up`/`cross` remain valid for x's non-root nodes (only root_only reuses them).
    return (;local_solves,matvecs,iterations,max_residual,environment_seconds,up,cross)
end

"""
    tamen_step(initial, generator, duration; time_points=8, rtol=1e-8, atol=0,
               maxrank=64, maxsweeps=12, kickrank=3, conserved=nothing,
               krylov_tol=1e-12, krylovdim=40, krylov_maxiter=100,
               residual_maxrank=nothing, strict=true, time_scheme=:cheb,
               conserved_observables=nothing, estimate_time_error=false,
               workspace=nothing, threaded=false, residual_method=:local,
               residual_sweeps=1, residual_check_every=4, residual_backend=:sop,
               warm_start=nothing, store_warm_start=false, als_sweep=:serial,
               truncation_rtol=0, certify=true, local_tol_factor=0.01,
               local_operator=:channels)

Experimental AMEn-type fixed-interval solve for `dx/dt = generator*x`. Attach
a full time leaf to a new binary root, discretize by Chebyshev collocation,
and use recursive Galerkin ALS/GMRES with auxiliary residual enrichment.
`time_scheme=:cn` uses an odd number of time points including the initial
point, as in TT tAMEn. Chebyshev counts unknowns excluding the initial point.
The generator must be constant over the interval. No dense spatial tensor or
global matrix is constructed; no normalization or trace correction is applied.

`residual_backend=:ttno` selects experimental tree-operator global certification.
It compiles and compresses the space-time SoP once per interval, applies it with subtree QR,
then compresses the explicit residual once. Local ALS still uses SoP. The
default `:sop` uses streamed termwise certification with smaller intermediates.
TTNO construction is counted in global residual time; operator/state bond-rank
products can make this backend memory intensive. No squared-norm cancellation
formula or unchecked acceptance is used.

`residual_maxrank=nothing` defaults to twice `maxrank`, saturating at `typemax(Int)`.
`maxrank` caps spatial bonds after each sweep (temporary enriched ranks can be
larger). The full time basis is exempt. `conserved`, when supplied, must be a
nonzero left null vector of the generator; its spatial subtree directions are
protected. Alternatively, `conserved_observables=[c1,c2,...]` protects their
combined subtree spaces. A final root solve tests all invariants at every
collocation point. `invariant_history` has observables in rows and samples in columns.

`residual_method=:local` approximates `b-A*x` in a `kickrank` auxiliary TTN,
using `residual_sweeps` fixed-rank ALS sweeps. This TTN is reused between solution
sweeps. Its projected norm is only an estimate, never an acceptance bound.
A global check runs when the estimate reaches the target, every
`residual_check_every` sweeps, and on the final sweep. Failed checks supply the
next enrichment direction. `residual_method=:global` retains the original path
with global residual formation on every sweep.

Return `(endpoint, diagnostics)`. `diagnostics.solution` contains the full
space-time TTN, accessible with `time_slice`. Each global residual is assembled
as successive compressed TTN sums. Its reported bound includes accumulated
compression error, excluding floating-point roundoff. Convergence requires
`residual_bound <= max(atol,rtol*norm(b))`. This is a linear-system tolerance,
NOT an estimate of time discretization or endpoint error: refine `time_points`
or the interval size separately. `strict=false` exposes an unconverged iterate
for diagnosis; default `strict=true` throws instead of accepting it. Local
GMRES failure always throws. `estimate_time_error=true` adds a refined-grid
integrated defect; use `tamen_evolve` for automatic interval control.
`als_sweep=:serial` retains sequential Gauss–Seidel local solves; with
`threaded=true`, upward environments and global residual additions run concurrently.
`als_sweep=:etter` freezes orthonormal child complements before independently
solving siblings, then solves the parent in their updated bases. With
`threaded=true`, these subtree solves also run concurrently; `threaded=false`
serializes the same Etter-style algorithm. It uses an inverse-free projected
parent starting vector, not inverse singular-value connector reconstruction.
Auxiliary residual sweeps, enrichment, weighted cuts and final invariant repair
are sequential. Residual additions reuse node-owned projection/QR buffers.
`workspace=TAMEnWorkspace()` reuses per-node scratch and temporal grids. It
must not be shared by concurrent calls. There is no preconditioner yet.
Unchecked history entries have `global_residual_checked=false` and infinite
bound fields; `residual_estimate` is reported separately. With the default
`certify=true`, final diagnostics always contain a current global check and
`certified` equals `converged`. `global_checks` counts them, and
`residual_updates` counts auxiliary local projections. `residual_seconds` is
the sum of `residual_local_seconds` and `residual_global_seconds`.
`timings.residual_global_parts` splits the latter into addition, weight QR,
SVD, basis updates and final QR. These nonoverlapping wall-time subsets exclude
minor bookkeeping; do not add them again to `residual_global_seconds`.
`als_parallel_tasks` counts spawned solution-subtree tasks. `sweep_seconds`
times the main solution sweeps and is a subset of `als_seconds` (which also
includes protected truncation and root repair); `environment_seconds` overlaps.

## TT-style acceptance options

Acceptance defaults remain certified and untruncated. The local tolerance and
local operator defaults below were chosen after the 10-site Holstein study; they
do not change what is accepted. Options matching TT tAMEn (`TTSolver.tamen`):

- `truncation_rtol > 0` truncates the space-time solution after every sweep
  with the weighted budget `truncation_rtol*norm(x)` (protected directions and
  the time basis are kept). Without it, enrichment ranks are only capped by
  `maxrank`. TT tAMEn truncates each local SVD at `tol/sqrt(d)`. Keep
  `truncation_rtol` below `rtol` (for example `rtol/3`): the final root solve
  cannot restore discarded non-root directions, so a larger budget can stall
  the residual above its target.
- `certify=false` never forms the global residual. An interval is accepted
  when the projected auxiliary residual norm `residual_estimate <= target`.
  That norm is a projection of `b-A*x` onto a rank-`kickrank` TTN and can
  underestimate it; choose a tighter `rtol` than the accuracy you need and
  validate against a reference. Requires `residual_method=:local`. Diagnostics
  report `certified=false`, `global_checks=0` and infinite bound fields;
  a strict failure has reason `:residual_estimate`.
- `local_tol_factor=0.01` (default) sets the absolute local GMRES tolerance to
  `local_tol_factor*target`. `local_tol_factor=nothing` restores the previous
  `min(krylov_tol*norm(b),target/100)`, which with `krylov_tol=1e-12` is usually
  far tighter than acceptance needs. Zero initial data still uses `krylov_tol`.

`local_operator=:channels` (default) builds an exact sparse tree operator once per
interval and uses it for ALS environments, local actions, auxiliary residuals
and the time-error estimate. Terms sharing an edge factor are merged into
channels (for HEOM, `2+2*sites` per bath subtree instead of one environment per
term), and identity channels in orthonormal frames are exact identities.
Global certification still uses the SoP term by term. `local_operator=:sop`
restores the previous per-term environments.

Combining `truncation_rtol` with `certify=true` is valid but can be costly:
the truncation error enters the certified residual through `A*δx`, so tight
`rtol` values need more sweeps and larger residual ranks.

`warm_start` accepts a `TAMEnWarmStart` from a preceding accepted interval.
Matching topology and initial/previous-endpoint data reuse its spatial bases
and temporal increments, remapped inside the normalized previous interval.
Changed time order/scheme resets the auxiliary residual; operator environments
are always rebuilt. Incompatible hints fall back to a cold guess, reported by
`warm_start_status`. The new RHS always uses `initial`, never cached data.
`store_warm_start=true` returns an independently owned hint in `info.warm_start`
only after linear convergence; it is otherwise `nothing`. The adaptive caller
must still check temporal acceptance before retaining it. `warm_start_used` and
`residual_warm_start_used` describe reuse; `timings.warm_start_seconds` includes
hint transfer and storage. Hints and returned solutions do not share tensor arrays.
"""
function tamen_step(initial::TTNTensor,generator::SumOfProducts,duration;workspace=nothing,kwargs...)
    return _with_tamen_workspace(workspace) do work
        _tamen_step(initial,generator,duration;workspace=work,kwargs...)
    end
end

function _tamen_step(initial::TTNTensor,generator::SumOfProducts,duration;
                    time_points=8,rtol=1e-8,atol=0.0,maxrank=64,maxsweeps=12,
                    kickrank=3,conserved=nothing,krylov_tol=1e-12,krylovdim=40,
                    krylov_maxiter=100,residual_maxrank=nothing,strict::Bool=true,
                    time_scheme=:cheb,estimate_time_error::Bool=false,
                    conserved_observables=nothing,workspace,threaded::Bool=false,
                    residual_method=:local,residual_sweeps=1,residual_check_every=4,
                    residual_backend=:sop,
                    warm_start=nothing,store_warm_start::Bool=false,als_sweep=:serial,
                    truncation_rtol=0.0,certify::Bool=true,local_tol_factor=0.01,
                    local_operator=:channels)
    _check_operator(generator,initial)
    _check_local_operator(local_operator)
    _check_bug_options(atol,maxrank,krylov_tol,krylovdim,krylov_maxiter)
    rtol isa Real && isfinite(rtol) && rtol >= 0 && isfinite(Float64(rtol)) &&
        (iszero(rtol) || Float64(rtol)>0) ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    (Float64(rtol)>0 || Float64(atol)>0) || throw(ArgumentError("one tolerance must be positive in Float64"))
    if isnothing(residual_maxrank)
        residual_maxrank = maxrank <= div(typemax(Int),2) ? 2Int(maxrank) : typemax(Int)
    end
    residual_method in (:local,:global) || throw(ArgumentError("residual_method must be :local or :global"))
    residual_backend in (:sop,:ttno) || throw(ArgumentError("residual_backend must be :sop or :ttno"))
    als_sweep in (:serial,:etter) || throw(ArgumentError("als_sweep must be :serial or :etter"))
    truncation_rtol isa Real && isfinite(truncation_rtol) && 0 <= truncation_rtol < 1 ||
        throw(ArgumentError("truncation_rtol must be finite and in [0,1)"))
    certify || residual_method == :local ||
        throw(ArgumentError("certify=false requires residual_method=:local"))
    isnothing(local_tol_factor) || (local_tol_factor isa Real && isfinite(local_tol_factor) &&
        0 < local_tol_factor <= 1) ||
        throw(ArgumentError("local_tol_factor must be nothing or a finite number in (0,1]"))
    for (name,value) in ((:maxsweeps,maxsweeps),(:kickrank,kickrank),(:residual_maxrank,residual_maxrank),
                         (:residual_sweeps,residual_sweeps),(:residual_check_every,residual_check_every))
        value isa Integer && value >= 1 || throw(ArgumentError("$name must be a positive integer"))
    end
    work_before = _tamen_work_counts(workspace)
    residual_times_before = _residual_times(workspace.certification)
    grid = _tamen_grid!(workspace,time_points,duration,time_scheme)
    nt = length(grid.d0)
    !isnothing(conserved) && !isnothing(conserved_observables) &&
        throw(ArgumentError("use either conserved or conserved_observables"))
    if !isnothing(conserved_observables)
        conserved_observables isa AbstractVector && all(c->c isa TTNTensor,conserved_observables) ||
            throw(ArgumentError("conserved_observables must be a vector of TTNs"))
    end
    cs = isnothing(conserved_observables) ?
        (isnothing(conserved) ? TTNTensor[] : [_canonical_conserved(initial,conserved)]) :
        [_canonical_conserved(initial,c) for c in conserved_observables]
    time_leaf = length(initial.tree.children)+1
    A = _spacetime_operator(generator,grid)
    certificate_operator = nothing
    b = _time_lift(initial,grid.d0)
    # ALS and auxiliary residuals may use exact channel environments; global
    # certification always applies the original SoP term by term.
    local_A = _local_operator(A,b.tree,local_operator)
    warm_started = time_ns()
    x,auxiliary,warm_start_status = _tamen_initial_guess(initial,grid,warm_start)
    warm_start_seconds = isnothing(warm_start) ? 0.0 : (time_ns()-warm_started)/1e9
    warm_start_used = warm_start_status == :used
    residual_warm_start_used = residual_method==:local && !isnothing(auxiliary)
    residual_method==:global && (auxiliary=nothing)
    protected = [_time_lift(c,ones(nt)) for c in cs]
    bnorm = norm(b)
    target = max(Float64(atol),Float64(rtol)*bnorm)
    isfinite(target) && (iszero(bnorm) || target>0) ||
        throw(ArgumentError("scaled tolerance is zero or nonfinite in Float64"))
    # Zero initial data is exact and avoids a zero absolute GMRES tolerance.
    localtol = iszero(bnorm) ? Float64(krylov_tol) :
        isnothing(local_tol_factor) ? min(krylov_tol*bnorm,target/100) : Float64(local_tol_factor)*target
    localtol>0 && isfinite(localtol) || throw(ArgumentError("local tolerance underflow or overflow"))
    history = NamedTuple[]
    total_solves = total_matvecs = total_iterations = 0
    als_parallel_tasks = 0
    sweep_seconds = 0.0
    als_seconds = residual_seconds = enrichment_seconds = 0.0
    residual_local_seconds = residual_global_seconds = 0.0
    global_checks = residual_updates = 0
    environment_seconds = 0.0
    converged = false
    last_root_environments = nothing
    residual_bound = compression_bound = Inf
    function prepare(x)
        for c in protected
            x = enrich_basis(x,c)
        end
        # truncation_rtol=0 keeps every nonzero direction up to maxrank.
        cut = truncate!(x;atol=0.0,rtol=truncation_rtol,maxrank,conserved=protected,fixed_leaves=[time_leaf])
        _identity_time!(x,time_leaf)
        return x,cut
    end
    x,_ = prepare(x)
    for sweep in 1:maxsweeps
        started = time_ns()
        sweep! = als_sweep==:etter ? _tamen_etter_sweep! : _tamen_sweep!
        stats = sweep!(x,local_A,b,time_leaf;tol=localtol,krylovdim,krylov_maxiter,workspace,threaded)
        sweep_seconds += (time_ns()-started)/1e9
        als_sweep==:etter && (als_parallel_tasks += stats.spawned_tasks)
        x,cut = prepare(x)
        # Reinsert conserved directions after local basis updates and solve in
        # c ⊗ I_time test space. Never truncate again after this root solve.
        rootstats = _tamen_sweep!(x,local_A,b,time_leaf;tol=localtol,krylovdim,krylov_maxiter,workspace,threaded,root_only=true)
        last_root_environments = (;up=rootstats.up,cross=rootstats.cross)
        als_seconds += (time_ns()-started)/1e9
        total_solves += stats.local_solves+rootstats.local_solves
        total_matvecs += stats.matvecs+rootstats.matvecs
        total_iterations += stats.iterations+rootstats.iterations
        environment_seconds += stats.environment_seconds+rootstats.environment_seconds
        # The local norm is only a heuristic. Never retain a previous iterate's
        # certificate in an unchecked history entry or accept on this estimate.
        residual_bound = compression_bound = Inf
        residual_estimate = Inf
        if residual_method == :local
            started = time_ns()
            auxiliary,rd = _tamen_local_residual(local_A,x,b,auxiliary;rank=kickrank,sweeps=residual_sweeps)
            residual_estimate = rd.estimate
            residual_updates += rd.local_updates
            residual_local_seconds += (time_ns()-started)/1e9
        end
        # certify=false never forms the global residual; acceptance then uses
        # the projected auxiliary estimate, as TT tAMEn uses local quantities.
        global_residual_checked = certify && (residual_method == :global ||
            !isfinite(residual_estimate) || residual_estimate <= target ||
            sweep % residual_check_every == 0 || sweep == maxsweeps)
        residual = auxiliary
        if global_residual_checked
            started = time_ns()
            if isnothing(certificate_operator)
                # Reserve a small fraction of the solve tolerance for operator
                # compression. Its actual Frobenius bound is multiplied by the
                # current ||x|| at EVERY check, including later larger iterates.
                certificate_operator = residual_backend==:ttno ?
                    TreeTensorOperator(A,x.tree;atol=target/2000/max(norm(x),1.0)) : A
            end
            residual,residual_bound,compression_bound = _tamen_residual(certificate_operator,x,b,target/20,residual_maxrank;
                workspace=workspace.certification,threaded)
            residual_global_seconds += (time_ns()-started)/1e9
            global_checks += 1
        end
        residual_seconds = residual_local_seconds+residual_global_seconds
        relative = iszero(bnorm) ? residual_bound : residual_bound/bnorm
        push!(history,(;sweep,residual_bound,relative_residual_bound=relative,
            residual_compression_bound=compression_bound,ranks=tree_ranks(x),
            residual_estimate,relative_residual_estimate=iszero(bnorm) ? residual_estimate : residual_estimate/bnorm,
            global_residual_checked,
            discarded_bound=cut.discarded_bound,
            max_local_residual=max(stats.max_residual,rootstats.max_residual)))
        converged = certify ? global_residual_checked && residual_bound <= target :
            residual_estimate <= target
        (converged || sweep == maxsweeps) && break
        started = time_ns()
        truncate!(residual;atol=0.0,maxrank=kickrank)
        auxiliary = residual
        x = enrich_basis(x,residual)
        _identity_time!(x,time_leaf)
        enrichment_seconds += (time_ns()-started)/1e9
    end
    solution = SpaceTimeTTN(x,copy(initial),grid.nodes,time_leaf,grid.scheme)
    invariant_history = [dot(c,time_slice(solution,j)) for c in cs, j in eachindex(grid.nodes)]
    invariant_values = length(cs)==1 ? vec(invariant_history) :
        isempty(cs) ? ComplexF64[] : invariant_history
    error_started = time_ns()
    # The final root-only solve changed only the root, so its spatial-root
    # environments still describe the returned basis.
    time_error = estimate_time_error && converged ?
        _tamen_time_error(solution,generator;threaded,local_operator,
            spacetime_operator=local_A,root_environments=last_root_environments) : nothing
    time_error_seconds = (time_ns()-error_started)/1e9
    warm_started = time_ns()
    next_warm_start = store_warm_start && converged ? TAMEnWarmStart(solution,auxiliary) : nothing
    if !isnothing(next_warm_start)
        warm_start_seconds += (time_ns()-warm_started)/1e9
    end
    info = (;converged,certified=certify && converged,solution,sweeps=length(history),history,
        residual_bound,time_error,
        relative_residual_estimate=isempty(history) ? Inf : last(history).relative_residual_estimate,
        relative_residual_bound=iszero(bnorm) ? residual_bound : residual_bound/bnorm,
        residual_compression_bound=compression_bound,
        local_solves=total_solves,matvecs=total_matvecs,gmres_iterations=total_iterations,
        timings=(;als_seconds,sweep_seconds,residual_seconds,residual_local_seconds,residual_global_seconds,
                  residual_global_parts=_residual_time_delta(workspace.certification,residual_times_before),
                  enrichment_seconds,environment_seconds,time_error_seconds,warm_start_seconds),invariant_values,invariant_history,
        warm_start=next_warm_start,warm_start_used,residual_warm_start_used,warm_start_status,
        residual_method,residual_backend,
        residual_operator_ranks=certificate_operator isa TreeTensorOperator ? tree_ranks(certificate_operator) : Int[],
        residual_operator_discarded_bound=certificate_operator isa TreeTensorOperator ? certificate_operator.discarded_bound : 0.0,
        global_checks,residual_updates,als_sweep,als_parallel_tasks,
        threaded=threaded && Threads.nthreads(:default)>1,
        workspace_stats=_tamen_work_delta(workspace,work_before))
    strict && !converged && throw(TAMEnConvergenceError(certify ? :global_residual : :residual_estimate,info))
    return time_slice(solution,length(grid.nodes)),info
end
