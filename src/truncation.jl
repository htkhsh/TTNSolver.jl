function _compression_options(atol, maxrank)
    atol isa Real && isfinite(atol) && atol >= 0 && isfinite(Float64(atol)) ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    maxrank isa Integer && maxrank >= 1 || throw(ArgumentError("maxrank must be a positive integer"))
end

# Twice-reorthogonalized columns; prioritize protected and old directions by
# passing them first. Each column has its own scale, so tiny physical amplitudes
# do not erase the old normalized basis during augmentation.
function _orth_columns(arrays::AbstractMatrix...)
    n = size(first(arrays), 1)
    Q = zeros(ComplexF64, n, 0)
    for A in arrays, j in axes(A, 2)
        size(Q, 2) == n && return Q
        v = ComplexF64.(A[:, j])
        scale = norm(v)
        iszero(scale) && continue
        v ./= scale
        for _ in 1:2
            v .-= Q * (Q' * v)
        end
        s = norm(v)
        s > 64eps(Float64) || continue
        Q = hcat(Q, v / s)
    end
    return Q
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
    truncate!(state; atol=1e-10, maxrank=typemax(Int), conserved=nothing)

Root-to-leaf SVD compression, protecting all subtree directions of an optional
TTN linear functional. Its subtree spaces must already be contained in state
(augmented_bug_step arranges this). No normalization is performed.

The absolute error budget is shared over edges, with the original root norm
as a conservative amplification bound for local errors. `discarded_bound`
is the sum of these weighted local errors; roundoff is not included. A rank
cap can exceed the budget, reported by `tolerance_met=false`, but cannot
discard a protected direction. Leaves/cores are root-orthogonal on return.
Invalid options or incompatible protected spaces leave the input unchanged.
"""
function truncate!(state::TTNTensor; atol=1e-10, maxrank=typemax(Int), conserved=nothing)
    _compression_options(atol, maxrank)
    c = _canonical_conserved(state, conserved)
    x = orthogonalize!(copy(state))
    tree = x.tree
    overlaps = isnothing(c) ? nothing : _subtree_overlaps(x, c)
    if !isnothing(c)
        for node in tree.postorder
            node == tree.root && continue
            Z = overlaps[node]
            norm(Z' * Z - I) <= 1e-10 * max(1, size(Z, 2)) ||
                throw(ArgumentError("conserved subtree space is absent at node $node; augment first"))
            size(Z, 2) <= maxrank || throw(ArgumentError("maxrank cannot hold the conserved directions"))
        end
    end
    scale = max(1.0, norm(x))
    edges = length(tree.children)-1
    budget = Float64(atol) / max(1, edges) / scale
    discarded_bound = 0.0
    # Descendants have not yet changed when their parent is processed; their
    # old orthonormal subtree bases therefore make each W an orthogonal cut.
    for node in reverse(tree.postorder)
        _isleaf(tree, node) && continue
        for (child, mode) in zip(tree.children[node], (2, 3))
            C = x.cores[node]
            M = Matrix(transpose(_complement_matrix(C, mode)))
            P = isnothing(c) ? zeros(ComplexF64, size(M, 1), 0) : overlaps[child]
            W, dropped = _retained_space(M, P, budget, maxrank)
            discarded_bound += scale * dropped
            x.cores[node] = _child_product(C, W', mode)
            if _isleaf(tree, child)
                x.leaves[child] = x.leaves[child] * W
            else
                x.cores[child] = _parent_product(x.cores[child], transpose(W))
            end
        end
    end
    orthogonalize!(x)
    merge!(state.leaves, x.leaves)
    merge!(state.cores, x.cores)
    return (; ranks=tree_ranks(state), discarded_bound,
            tolerance_met=discarded_bound <= Float64(atol))
end
