"""Recoverable local Arnoldi failure, distinct from invalid inputs or code errors."""
struct LocalSolveFailure <: Exception
    reason::Symbol
    error_estimate::Float64
end

Base.showerror(io::IO, error::LocalSolveFailure) =
    print(io, "Arnoldi local solve failed: ", error.reason,
          " (estimated error ", error.error_estimate, ")")

function _check_bug_options(atol, maxrank, krylov_tol, krylovdim, krylov_maxiter)
    _compression_options(atol, maxrank)
    krylov_tol isa Real && isfinite(krylov_tol) && krylov_tol > 0 &&
        isfinite(Float64(krylov_tol)) && Float64(krylov_tol) > 0 ||
        throw(ArgumentError("krylov_tol must be positive and finite in Float64"))
    krylovdim isa Integer && 1 <= krylovdim <= typemax(Int) || throw(ArgumentError("krylovdim must be a positive Int"))
    krylov_maxiter isa Integer && 1 <= krylov_maxiter <= typemax(Int) || throw(ArgumentError("krylov_maxiter must be a positive Int"))
    return nothing
end

# Private, read-only preparation for steps starting from the same state.
# QR complements and their reduced initial data depend on the old state only,
# including at internal nodes. Never reuse this across a changed input/operator.
function _prepare_bug_step(state, operator, conserved, maxrank)
    _check_operator(operator, state)
    old = orthogonalize!(copy(state))
    c = _canonical_conserved(state, conserved)
    if !isnothing(c)
        ranks = tree_ranks(c)
        all(node -> node == c.tree.root || ranks[node] <= maxrank, c.tree.postorder) ||
            throw(ArgumentError("maxrank cannot hold the conserved directions"))
    end
    tree = old.tree
    up = _upward_operators(operator, old;orthonormal=true)
    initial = Vector{Union{Matrix{ComplexF64},Array{ComplexF64,3}}}(undef,length(tree.children))
    boundary = Vector{Vector{Matrix{ComplexF64}}}(undef,length(tree.children))
    root = tree.root
    initial[root] = _isleaf(tree,root) ? old.leaves[root] : old.cores[root]
    boundary[root] = [ones(ComplexF64,1,1) for _ in operator.terms]
    function descend(node)
        _isleaf(tree,node) && return
        left,right = tree.children[node]
        for (child,sibling,mode) in ((left,right,2),(right,left,3))
            complement = _complement_matrix(initial[node],mode)
            fac = qr(complement)
            rank = min(size(complement)...)
            Q,R = Matrix(fac.Q)[:,1:rank],Matrix(fac.R)
            boundary[child] = _complement_operators(Q,boundary[node],up[sibling];orthonormal=true)
            initial[child] = _isleaf(tree,child) ? old.leaves[child] * transpose(R) :
                _parent_product(old.cores[child],R)
            descend(child)
        end
    end
    descend(root)
    modes = Dict(node => mode for (mode,node) in enumerate(tree.leaf_order))
    return (;old,c,initial,boundary,modes,up)
end

"""
    augmented_bug_step(state, operator, h; atol=1e-10, maxrank=typemax(Int),
        conserved=nothing, krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
        correct_invariant=true, threaded=true, truncation=:weighted)

One recursive augmented basis-update-and-Galerkin step for the autonomous
linear equation Y' = operator * Y. Returns `(new_state, diagnostics)` without
mutating either input. Local non-Hermitian exponential actions use Arnoldi;
failure to converge raises an error. No full tensor or generator is expanded.

Old sibling bases and QR complements define each reduced equation. New and old
subtree bases are combined before the parent Galerkin solve. Full physical
leaf spaces skip their K solves. Finally truncate with an absolute tolerance.
With `threaded=true`, sibling subtrees run concurrently on Julia worker
threads; a parent waits for both updated child bases. Preparation and final
truncation remain serial. `threaded=false` uses the same formula sequentially.
Every local solve owns its workspace. Child failures are joined and rethrown
as their original exception type, including `LocalSolveFailure`.

An optional nonzero TTN `conserved` supplies a linear invariant c'Y; the caller
must ensure c'L=0. All its subtree spaces are protected. A root correction
can remove numerical drift; both the uncorrected drift and correction norm
are reported. `atol` controls truncation only, not the BUG time error. This is
the first-order augmented method, with no time-step adaptivity or rejection.
The default `truncation=:weighted` uses full-state amplitude weights during
compression; `:local` selects the conservative local-core SVD policy.
"""
function augmented_bug_step(state::TTNTensor, operator::SumOfProducts, h::Real;
                            atol=1e-10, maxrank=typemax(Int), conserved=nothing,
                            krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
                            correct_invariant::Bool=true, threaded::Bool=true,
                            truncation=:weighted)
    isfinite(h) && h > 0 && isfinite(Float64(h)) && Float64(h) > 0 ||
        throw(ArgumentError("h must be positive and finite in Float64"))
    _check_bug_options(atol, maxrank, krylov_tol, krylovdim, krylov_maxiter)
    _check_truncation_weighting(truncation)
    prepared = _prepare_bug_step(state,operator,conserved,maxrank)
    return _prepared_bug_step(prepared,operator,h;atol,maxrank,
                             krylov_tol,krylovdim,krylov_maxiter,correct_invariant,threaded,truncation)
end

# A task returns its failure instead of wrapping it in TaskFailedException.
# Always join both subtrees, including their descendants, before propagating
# any error. Programming errors take precedence over recoverable failures.
function _bug_child_error(update_child, child)
    try
        update_child(child)
        return nothing
    catch error
        return error
    end
end

function _bug_subtrees!(update_child, left, right, use_threads)
    if !use_threads
        update_child(left)
        update_child(right)
        return nothing
    end
    task = Threads.@spawn _bug_child_error(update_child,left)
    right_failure = _bug_child_error(update_child,right)
    left_failure = fetch(task)
    for failure in (left_failure,right_failure)
        !isnothing(failure) && !(failure isa LocalSolveFailure) && throw(failure)
    end
    for failure in (left_failure,right_failure)
        isnothing(failure) || throw(failure)
    end
    return nothing
end

# Caller validates solver options. Only read prepared arrays; all updated
# bases and coefficients belong to this invocation, including on failure.
function _prepared_bug_step(prepared, operator, h;
                            atol=1e-10, maxrank=typemax(Int),
                            krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
                            correct_invariant=true, threaded=true,truncation=:weighted)
    (;old,c,modes) = prepared
    tree = old.tree
    use_threads = threaded && Threads.nthreads(:default) > 1 && !_isleaf(tree,tree.root)
    # Disjoint vector slots are safe for sibling writes. Do not mutate a shared
    # TTN's dictionaries from tasks, even when their node keys differ.
    updated = similar(prepared.initial)
    # Every child's entry is assigned before its parent's Galerkin solve.
    upnew = Vector{Vector{Matrix{ComplexF64}}}(undef,length(tree.children))
    overlap = Vector{Matrix{ComplexF64}}(undef, length(tree.children))
    conserved_overlap = similar(overlap)
    dimensions = zeros(Int,length(tree.children))
    skipped = zeros(Int,length(tree.children))
    errors = zeros(length(tree.children))

    function integrate(node, rhs, x)
        dimensions[node] = length(x)
        scale = norm(x)
        isfinite(scale) || throw(LocalSolveFailure(:nonfinite, Inf))
        iszero(scale) && return copy(x)
        # KrylovKit uses an absolute tolerance per unit time. Scale the local
        # IVP and restore its amplitude, so small coefficients still update.
        y, info = KrylovKit.exponentiate(rhs, Float64(h), x / scale;
            ishermitian=false, issymmetric=false, tol=Float64(krylov_tol),
            krylovdim=min(Int(krylovdim), length(x)), maxiter=Int(krylov_maxiter),
            eager=true, verbosity=0)
        info.converged == 1 || throw(LocalSolveFailure(:nonconvergence, Float64(info.normres)))
        all(isfinite, y) || throw(LocalSolveFailure(:nonfinite, Inf))
        restored = scale .* y
        all(isfinite, restored) || throw(LocalSolveFailure(:nonfinite, Inf))
        errors[node] = scale * info.normres
        return restored
    end

    function update_child(child)
        initial,boundary = prepared.initial[child],prepared.boundary[child]
        if _isleaf(tree, child)
            U = old.leaves[child]
            if size(U, 1) == size(U, 2)
                skipped[child] = 1
                updated[child] = copy(U)
            else
                evolved = integrate(child,_leaf_rhs_workspace(operator,modes[child],boundary,initial),initial)
                protected = isnothing(c) ? zeros(ComplexF64, size(U, 1), 0) : c.leaves[child]
                updated[child] = _orth_columns(protected, U, evolved)
            end
            Uhat = updated[child]
            overlap[child] = Uhat' * U
            if !isnothing(c)
                conserved_overlap[child] = Uhat' * c.leaves[child]
            end
            upnew[child] = _leaf_operators(operator,modes[child],Uhat;orthonormal=true)
        else
            evolved, _ = evolve_core(child)
            left, right = tree.children[child]
            old_projected = _project_children(old.cores[child], overlap[left], overlap[right])
            protected = isnothing(c) ? zeros(ComplexF64, size(evolved, 2)*size(evolved, 3), 0) :
                _basis_matrix(_project_children(c.cores[child], conserved_overlap[left], conserved_overlap[right]))
            Uhat = _orth_columns(protected, _basis_matrix(old_projected), _basis_matrix(evolved))
            updated[child] = reshape(Matrix(transpose(Uhat)), size(Uhat, 2), size(evolved, 2), size(evolved, 3))
            overlap[child] = Uhat' * _basis_matrix(old_projected)
            if !isnothing(c)
                conserved_overlap[child] = Uhat' * protected
            end
            upnew[child] = _subtree_operators(updated[child],upnew[left],upnew[right];orthonormal=true)
        end
    end

    function evolve_core(node)
        initial,boundary = prepared.initial[node],prepared.boundary[node]
        left, right = tree.children[node]
        _bug_subtrees!(update_child,left,right,use_threads)
        projected = _project_children(initial, overlap[left], overlap[right])
        evolved = integrate(node,_core_rhs_workspace(operator,boundary,upnew[left],upnew[right],projected),projected)
        return evolved, projected
    end

    root = tree.root
    if _isleaf(tree, root)
        root_initial = old.leaves[root]
        updated[root] = integrate(root,_leaf_rhs_workspace(operator,1,prepared.boundary[root],root_initial),root_initial)
    else
        updated[root], _ = evolve_core(root)
    end
    result = copy(old)
    for node in tree.postorder
        if _isleaf(tree,node)
            result.leaves[node] = updated[node]
        else
            result.cores[node] = updated[node]
        end
    end
    stats = (;local_solves=count(>(0),dimensions),skipped_leaf_solves=sum(skipped),
        max_local_dimension=maximum(dimensions),krylov_error_estimate=sum(errors[tree.postorder]),
        method=:augmented,threaded=use_threads)
    return _finish_bug_step!(result,old,c,stats;atol,maxrank,correct_invariant,truncation)
end

# Both BUG variants share protected truncation and the optional root correction.
function _finish_bug_step!(result,old,c,stats;atol,maxrank,correct_invariant,truncation=:weighted)
    tree,root = result.tree,result.tree.root
    augmented_ranks = tree_ranks(result)
    truncation = truncate!(result; atol, maxrank, conserved=c,weighting=truncation)
    invariant_before = isnothing(c) ? nothing : dot(c, old)
    invariant_uncorrected = isnothing(c) ? nothing : dot(c, result)
    correction_norm = 0.0
    if !isnothing(c) && correct_invariant
        a = if _isleaf(tree, root)
            c.leaves[root]
        else
            coordinates = _subtree_overlaps(result, c)
            root_left, root_right = tree.children[root]
            _project_children(c.cores[root], coordinates[root_left], coordinates[root_right])
        end
        functional_norm = norm(a)
        functional_norm > 0 || error("conserved direction was lost from the root space")
        delta = invariant_before - invariant_uncorrected
        root_data = _isleaf(tree, root) ? result.leaves[root] : result.cores[root]
        root_data .+= (delta / functional_norm) .* (a / functional_norm)
        correction_norm = abs(delta) / functional_norm
    end
    invariant_after = isnothing(c) ? nothing : dot(c, result)
    diagnostics = (; augmented_ranks, ranks=tree_ranks(result), truncation,
        stats...,
        invariant_before, invariant_uncorrected, invariant_after, correction_norm)
    return result, diagnostics
end
