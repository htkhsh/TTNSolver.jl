function _compression_options(atol, maxrank)
    atol isa Real && isfinite(atol) && atol >= 0 && isfinite(Float64(atol)) ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    maxrank isa Integer && maxrank >= 1 || throw(ArgumentError("maxrank must be a positive integer"))
end

function _check_truncation_weighting(weighting)
    weighting isa Symbol && weighting in (:local,:weighted) ||
        throw(ArgumentError("truncation weighting must be :local or :weighted"))
end

# Twice-reorthogonalized columns; prioritize protected and old directions by
# passing them first. Each column has its own scale, so tiny physical amplitudes
# do not erase the old normalized basis during augmentation.
function _orth_columns(arrays::AbstractMatrix...)
    n = size(first(arrays), 1)
    all(A->size(A,1)==n,arrays) || throw(DimensionMismatch("basis row counts differ"))
    capacity = min(n,sum(A->size(A,2),arrays))
    Q = Matrix{ComplexF64}(undef,n,capacity)
    v = Vector{ComplexF64}(undef,n)
    projected = similar(v)
    coefficients = Vector{ComplexF64}(undef,capacity)
    count = 0
    for A in arrays, j in axes(A, 2)
        count == n && break
        copyto!(v,view(A,:,j))
        scale = norm(v)
        iszero(scale) && continue
        v ./= scale
        for _ in 1:2
            if count > 0
                active=view(Q,:,1:count)
                weights=view(coefficients,1:count)
                mul!(weights,active',v)
                mul!(projected,active,weights)
                v .-= projected
            end
        end
        s = norm(v)
        s > 64eps(Float64) || continue
        count += 1
        view(Q,:,count) .= v ./ s
    end
    return Q[:,1:count]
end

function _subtree_overlaps(x, c)
    overlaps = Vector{Matrix{ComplexF64}}(undef, length(x.tree.children))
    for node in x.tree.postorder
        overlaps[node] = if _isleaf(x.tree, node)
            x.leaves[node]' * c.leaves[node]
        else
            left, right = x.tree.children[node]
            _basis_matrix(x.cores[node])' *
                _basis_matrix(_project_children(c.cores[node], overlaps[left], overlaps[right]))
        end
    end
    return overlaps
end

function _canonical_conserved(state, conserved)
    isnothing(conserved) && return nothing
    _check_layout(state, conserved)
    c = orthogonalize!(copy(conserved))
    norm(c) > 0 || throw(ArgumentError("conserved functional must be nonzero"))
    return c
end

function _retained_space(M, protected, budget, maxrank)
    n = size(M, 1)
    iszero(budget) && maxrank >= n && return Matrix{ComplexF64}(I, n, n), 0.0
    P = _orth_columns(protected)
    p = size(P, 2)
    p <= maxrank || throw(ArgumentError("maxrank cannot hold the conserved directions"))
    # An explicit complement is small: its dimension is a bond rank, not a
    # product of physical dimensions.
    Z = p == 0 ? Matrix{ComplexF64}(I, n, n) :
        (qr(P).Q * Matrix{ComplexF64}(I, n, n))[:, p+1:n]
    if size(Z, 2) == 0
        return P, 0.0
    end
    fac = svd(Z' * M)
    count = 0
    while count < length(fac.S) && norm(view(fac.S, count+1:length(fac.S))) > budget
        count += 1
    end
    count = min(max(count, p == 0 ? 1 : 0), maxrank-p, length(fac.S))
    W = hcat(P, Z * fac.U[:, 1:count])
    discarded = norm(view(fac.S, count+1:length(fac.S)))
    return W, discarded
end

"""
    truncate!(state; atol=1e-10, rtol=0, maxrank=typemax(Int), conserved=nothing,
              weighting=:weighted, fixed_leaves=Int[])

Root-to-leaf SVD compression, protecting all subtree directions of an optional
TTN linear functional. Its subtree spaces must already be contained in state
(augmented_bug_step arranges this). No normalization is performed.
`conserved` may also be a vector of TTNs, protecting the union of their subtree
directions rather than a sum of their values (which could cancel).

The absolute error budget is `max(atol, rtol*norm(state))`; the norm is read
from the root after the internal orthogonalization, so `rtol` adds no QR pass.
The budget is shared over edges. The default `weighting=:weighted`
propagates complementary amplitudes by thin QR and uses their full-state
weights in each SVD. `:local` uses the original root norm as a conservative
amplification bound for local core errors. Both return `discarded_bound` as a sum
of error bounds; roundoff is not included. A rank
cap can exceed the budget, reported by `tolerance_met=false`, but cannot
discard a protected direction. Leaves/cores are root-orthogonal on return.
Invalid options or incompatible protected spaces leave the input unchanged.
`fixed_leaves` lists leaf node IDs whose full current basis is retained, exempt
from `maxrank`. This supports an uncompressed time basis in space-time solves.
"""
function truncate!(state::TTNTensor; atol=1e-10, rtol=0.0, maxrank=typemax(Int), conserved=nothing,
                   weighting=:weighted, fixed_leaves=Int[])
    _compression_options(atol, maxrank)
    rtol isa Real && isfinite(rtol) && rtol >= 0 && isfinite(Float64(rtol)) ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    _check_truncation_weighting(weighting)
    all(n -> n isa Integer && n in state.tree.leaf_order, fixed_leaves) ||
        throw(ArgumentError("fixed_leaves must contain leaf node IDs"))
    fixed = Set(fixed_leaves)
    cs = conserved isa AbstractVector ? [_canonical_conserved(state,c) for c in conserved] :
         isnothing(conserved) ? TTNTensor[] : [_canonical_conserved(state,conserved)]
    all(c->c isa TTNTensor,cs) || throw(ArgumentError("conserved entries must be TTNs"))
    x = orthogonalize!(copy(state))
    tree = x.tree
    all_overlaps = [_subtree_overlaps(x,c) for c in cs]
    overlaps = isempty(cs) ? nothing : length(cs)==1 ? first(all_overlaps) :
        [hcat((o[node] for o in all_overlaps)...) for node in eachindex(tree.children)]
    if !isempty(cs)
        for node in tree.postorder
            node == tree.root && continue
            for o in all_overlaps
                Z = o[node]
                norm(Z' * Z - I) <= 1e-10 * max(1, size(Z, 2)) ||
                    throw(ArgumentError("conserved subtree space is absent at node $node; augment first"))
            end
            length(cs)>1 && (overlaps[node] = _orth_columns(overlaps[node]))
            (node in fixed || size(overlaps[node],2) <= maxrank) ||
                throw(ArgumentError("maxrank cannot hold the conserved directions"))
        end
    end
    budget = Float64(atol)
    if Float64(rtol) > 0
        amplitude = _isleaf(tree, tree.root) ? x.leaves[tree.root] : x.cores[tree.root]
        budget = max(budget, Float64(rtol)*norm(amplitude))
        isfinite(budget) || throw(ArgumentError("rtol*norm(state) overflows Float64"))
    end
    info = _truncate_canonical!(x;atol=budget,maxrank,weighting,overlaps,fixed)
    merge!(state.leaves, x.leaves)
    merge!(state.cores, x.cores)
    return info
end

# Internal kernel for independently owned, root-orthogonal tensors. Validation,
# protection checks and transactional copying remain in the public wrapper.
# Canonical sum assembly can call this directly without a redundant copy/QR.
function _truncate_canonical!(x;atol,maxrank,weighting=:weighted,
                              overlaps=nothing,fixed=Set{Int}(),canonicalize::Bool=true,timings=nothing)
    tree = x.tree
    weighted = weighting == :weighted
    scale = weighted ? 1.0 : max(1.0, norm(x))
    factors = weighted ? Vector{Matrix{ComplexF64}}(undef,length(tree.children)) : nothing
    weighted && (factors[tree.root] = ones(ComplexF64,1,1))
    edges = length(tree.children)-1
    budget = Float64(atol) / max(1, edges) / scale
    discarded_bound = 0.0
    # Descendants have not yet changed when their parent is processed; their
    # old orthonormal subtree bases therefore make each W an orthogonal cut.
    for node in reverse(tree.postorder)
        _isleaf(tree, node) && continue
        for (child, mode) in zip(tree.children[node], (2, 3))
            C = x.cores[node]
            started=_residual_tick(timings)
            if weighted
                # Complementary bases are orthonormal before cuts in other
                # branches. Later orthogonal projections there can only
                # decrease these tail norms, so cached factors remain bounds.
                amplitude = _parent_product(C,factors[node])
                R = Matrix(qr(_complement_matrix(amplitude,mode)).R)
                M = Matrix(transpose(R))
            else
                M = Matrix(transpose(_complement_matrix(C, mode)))
            end
            _residual_record!(timings,Val(:weight_qr_seconds),started)
            started=_residual_tick(timings)
            P = isnothing(overlaps) ? zeros(ComplexF64, size(M, 1), 0) : overlaps[child]
            W, dropped = child in fixed ? (Matrix{ComplexF64}(I,size(M,1),size(M,1)),0.0) :
                _retained_space(M, P, budget, maxrank)
            discarded_bound += scale * dropped
            _residual_record!(timings,Val(:svd_seconds),started)
            started=_residual_tick(timings)
            # Coefficients transform by W', hence the transposed outside
            # factor transforms by conj(W), not W or W'.
            weighted && (factors[child] = R * conj.(W))
            x.cores[node] = _child_product(C, W', mode)
            if _isleaf(tree, child)
                x.leaves[child] = x.leaves[child] * W
            else
                x.cores[child] = _parent_product(x.cores[child], transpose(W))
            end
            _residual_record!(timings,Val(:basis_update_seconds),started)
        end
    end
    canonicalize && orthogonalize!(x)
    return (; ranks=tree_ranks(x), discarded_bound,
            tolerance_met=discarded_bound <= Float64(atol))
end
