# Ceruti, Kusch, Lubich & Sulz, doi:10.1137/24M1714502, Algorithms 1–4.
# Reduced IVPs use old bases at EVERY node. Only the subsequent linear algebra
# is recursive. Mixed upward contractions compute the explicit cross blocks
# without forming either F(Y) as a TTN or a global tensor.
function _prepare_parallel_bug_step(state,operator,conserved,maxrank)
    # New directions must be orthogonal to the invariant, so its spaces must
    # already be present in the OLD basis used by the Galerkin equations.
    enriched = isnothing(conserved) ? state : enrich_basis(state,conserved)
    return _prepare_bug_step(enriched,operator,conserved,maxrank)
end

"""
    parallel_bug_step(state, operator, h; atol=1e-10, maxrank=typemax(Int),
        conserved=nothing, krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
        correct_invariant=true, threaded=true, truncation=:weighted)

First-order parallel BUG for the autonomous linear equation `Y'=operator*Y`.
All leaf and core ODEs use old bases and can run concurrently. Assemble the
evolved old/old block and `h*F(Y_old)` single-new-direction blocks, then apply
protected truncation. No augmented-space Galerkin ODE or global expansion is
used. `threaded=false` runs exactly the same method serially; `threaded=true`
uses Julia worker threads when available. Each solve owns its workspace.
The default `truncation=:weighted` selects amplitude-weighted compression, as
in `augmented_bug_step`; `:local` selects the conservative local-core policy.

Returns `(new_state, diagnostics)`, without mutating inputs. Solver tolerances,
truncation and invariant correction follow `augmented_bug_step`. Invariant
subtree spaces are additionally inserted into the old basis before solving.
Local Arnoldi failure raises `LocalSolveFailure`, also in threaded execution.

`omitted_block_norms[node]` is `h*eta_node`, the norm of the discarded new/new
derivative block (zero for leaves). It is a diagnostic, not a global error
bound or an automatic rank-expansion retry. The optional rejection strategy
of section 4.6 of the paper is not applied. Check time/rank convergence;
directions invisible to all reduced equations can require `enrich_basis`.
"""
function parallel_bug_step(state::TTNTensor,operator::SumOfProducts,h::Real;
                           atol=1e-10,maxrank=typemax(Int),conserved=nothing,
                           krylov_tol=1e-12,krylovdim=30,krylov_maxiter=100,
                           correct_invariant::Bool=true,threaded::Bool=true,truncation=:weighted)
    isfinite(h) && h > 0 && isfinite(Float64(h)) && Float64(h) > 0 ||
        throw(ArgumentError("h must be positive and finite in Float64"))
    _check_bug_options(atol,maxrank,krylov_tol,krylovdim,krylov_maxiter)
    _check_truncation_weighting(truncation)
    prepared = _prepare_parallel_bug_step(state,operator,conserved,maxrank)
    return _prepared_parallel_bug_step(prepared,operator,h;atol,maxrank,
        krylov_tol,krylovdim,krylov_maxiter,correct_invariant,threaded,truncation)
end

function _parallel_local_solve(prepared,operator,node,h,krylov_tol,krylovdim,krylov_maxiter)
    (;old,initial,boundary,modes,up) = prepared
    x = initial[node]
    leaf = _isleaf(old.tree,node)
    if leaf && node != old.tree.root && size(old.leaves[node],1) == size(old.leaves[node],2)
        return (;value=copy(x),error=0.0,solves=0,skipped=1,dimension=0)
    end
    rhs = if leaf
        _leaf_rhs_workspace(operator,modes[node],boundary[node],x)
    else
        left,right = old.tree.children[node]
        _core_rhs_workspace(operator,boundary[node],up[left],up[right],x)
    end
    scale = norm(x)
    isfinite(scale) || throw(LocalSolveFailure(:nonfinite,Inf))
    iszero(scale) && return (;value=copy(x),error=0.0,solves=1,skipped=0,dimension=length(x))
    y,info = KrylovKit.exponentiate(rhs,Float64(h),x/scale;
        ishermitian=false,issymmetric=false,tol=Float64(krylov_tol),
        krylovdim=min(Int(krylovdim),length(x)),maxiter=Int(krylov_maxiter),verbosity=0)
    info.converged == 1 || throw(LocalSolveFailure(:nonconvergence,Float64(info.normres)))
    value = scale .* y
    all(isfinite,value) || throw(LocalSolveFailure(:nonfinite,Inf))
    return (;value,error=Float64(scale*info.normres),solves=1,skipped=0,dimension=length(x))
end

function _prepared_parallel_bug_step(prepared,operator,h;
                                     atol=1e-10,maxrank=typemax(Int),
                                     krylov_tol=1e-12,krylovdim=30,krylov_maxiter=100,
                                     correct_invariant=true,threaded=true,truncation=:weighted)
    (;old,c,initial,boundary,modes,up) = prepared
    tree,root = old.tree,old.tree.root
    count = length(tree.children)
    solves = Vector{NamedTuple}(undef,count)
    use_threads = threaded && Threads.nthreads(:default) > 1
    if use_threads
        # Catch within each worker to preserve the original exception type.
        # @sync joins ALL workers before inspecting failures, so no task
        # survives a rejected adaptive trial or mutates arrays after return.
        failures = Vector{Any}(nothing,count)
        @sync for node in tree.postorder
            Threads.@spawn try
                solves[node] = _parallel_local_solve(prepared,operator,node,h,
                    krylov_tol,krylovdim,krylov_maxiter)
            catch error
                failures[node] = error
            end
        end
        # Programming errors must not be masked by a recoverable local failure.
        for failure in failures
            !isnothing(failure) && !(failure isa LocalSolveFailure) && throw(failure)
        end
        for failure in failures
            isnothing(failure) || throw(failure)
        end
    else
        for node in tree.postorder
            solves[node] = _parallel_local_solve(prepared,operator,node,h,
                krylov_tol,krylovdim,krylov_maxiter)
        end
    end

    result = copy(old)
    overlap = Vector{Matrix{ComplexF64}}(undef,count)
    mixed = Vector{Vector{Matrix{ComplexF64}}}(undef,count)
    new_mixed = similar(mixed)
    new_columns = Vector{UnitRange{Int}}(undef,count)
    omitted_block_norms = zeros(count)
    for node in tree.postorder
        if _isleaf(tree,node)
            if node == root
                result.leaves[node] = solves[node].value
                continue
            end
            U = old.leaves[node]
            # A full physical leaf cannot acquire a new direction. Its old
            # orthonormal gauge and upward operators already give all we need.
            full_leaf = solves[node].skipped == 1
            Q = full_leaf ? copy(U) : _orth_columns(U,solves[node].value)
            result.leaves[node] = Q
            overlap[node] = full_leaf ? Matrix{ComplexF64}(I,size(U,2),size(U,2)) : Q'*U
            mixed[node] = full_leaf ? up[node] :
                _leaf_operators(operator,modes[node],Q,U)
            oldrank = size(U,2)
        else
            left,right = tree.children[node]
            # Algorithm 4: evolved old/old block, two single-new blocks,
            # and zero in the double-new block. Coordinates allow arbitrary
            # QR gauges while the new basis columns follow the old ones.
            core = _project_children(solves[node].value,overlap[left],overlap[right])
            p,l,r = size(initial[node])
            nl,nr = length(new_columns[left]),length(new_columns[right])
            if nl > 0
                cross = _parallel_core_block(operator,boundary[node],new_mixed[left],up[right],initial[node],(p,nl,r))
                view(core,:,new_columns[left],:) .+= Float64(h) .* _child_product(cross,overlap[right],3)
            end
            if nr > 0
                cross = _parallel_core_block(operator,boundary[node],up[left],new_mixed[right],initial[node],(p,l,nr))
                view(core,:,:,new_columns[right]) .+= Float64(h) .* _child_product(cross,overlap[left],2)
            end
            if nl > 0 && nr > 0
                omitted_block_norms[node] = abs(Float64(h))*norm(
                    _parallel_core_block(operator,boundary[node],new_mixed[left],new_mixed[right],initial[node],(p,nl,nr)))
            end
            if node == root
                result.cores[node] = core
                continue
            end
            old_projected = _project_children(old.cores[node],overlap[left],overlap[right])
            U = _basis_matrix(old_projected)
            Q = _orth_columns(U,_basis_matrix(core))
            result.cores[node] = reshape(Matrix(transpose(Q)),size(Q,2),size(core,2),size(core,3))
            overlap[node] = Q'*U
            mixed[node] = _mixed_subtree_operators(Q,old.cores[node],mixed[left],mixed[right])
            oldrank = size(U,2)
        end
        new_columns[node] = oldrank+1:size(overlap[node],1)
        new_mixed[node] = [action[new_columns[node],:] for action in mixed[node]]
    end
    stats = (;local_solves=sum(s.solves for s in solves),
        skipped_leaf_solves=sum(s.skipped for s in solves),
        max_local_dimension=maximum(s.dimension for s in solves),
        krylov_error_estimate=sum(s.error for s in solves),
        method=:parallel,threaded=use_threads,omitted_block_norms)
    return _finish_bug_step!(result,old,c,stats;atol,maxrank,correct_invariant,truncation)
end
