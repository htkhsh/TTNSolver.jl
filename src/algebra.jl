# Multiply either child mode by a matrix with shape (new_rank, old_rank).
function _child_product(core::Array{ComplexF64,3}, matrix::AbstractMatrix, mode::Int)
    permutation = mode == 2 ? (2, 1, 3) : (3, 1, 2)
    unfolded = permutedims(core, permutation)
    dimensions = size(unfolded)
    result = matrix * reshape(unfolded, dimensions[1], :)
    folded = reshape(result, size(matrix, 1), dimensions[2], dimensions[3])
    return permutedims(folded, invperm(permutation))
end

function _check_layout(x::TTNTensor, y::TTNTensor)
    tx, ty = x.tree, y.tree
    tx.root == ty.root && tx.children == ty.children && tx.leaf_order == ty.leaf_order ||
        throw(ArgumentError("inner products require the same ordered tree and leaf order"))
    leafdims(x) == leafdims(y) || throw(DimensionMismatch("physical leaf dimensions differ"))
end

"""
    dot(x::TTNTensor, y::TTNTensor)

Contract the inner product, conjugating `x`, without dense expansion. Trees,
physical leaf orders and dimensions must match; bond ranks may differ.
"""
function LinearAlgebra.dot(x::TTNTensor, y::TTNTensor)
    _check_layout(x, y)
    return _tree_dot(x, y, node -> x.leaves[node]' * y.leaves[node])
end

# The caller validates layouts and supplies leaf overlap/operator environments.
function _tree_dot(x::TTNTensor, y::TTNTensor, leaf_environment)
    environments = Vector{Matrix{ComplexF64}}(undef, length(x.tree.children))
    for node in x.tree.postorder
        if _isleaf(x.tree, node)
            environments[node] = leaf_environment(node)
        else
            left, right = x.tree.children[node]
            projected = _child_product(y.cores[node], environments[left], 2)
            projected = _child_product(projected, environments[right], 3)
            xcore = x.cores[node]
            environments[node] = conj.(reshape(xcore, size(xcore, 1), :)) *
                                 transpose(reshape(projected, size(projected, 1), :))
        end
    end
    return environments[x.tree.root][1, 1]
end

"""
    norm(x::TTNTensor)

Compute the Euclidean norm from a root-orthogonalized copy, without dense
expansion or mutation of `x`. QR avoids squaring the conditioning of subtree
bases in `dot(x, x)`; the root uses LinearAlgebra's scaled array norm.
"""
function LinearAlgebra.norm(x::TTNTensor)
    root = x.tree.root
    _isleaf(x.tree, root) && return norm(x.leaves[root])
    canonical = orthogonalize!(copy(x))
    return norm(canonical.cores[root])
end

"""
    ttn_distance(x, y)

Euclidean norm of x-y without global dense expansion or subtraction of
squared norms. An exact block-sum difference is root-orthogonalized by QR.
Bond ranks in this temporary tensor are at most the sum of the input ranks.
Inputs are unchanged and must have the same ordered tree and physical sizes.
"""
function ttn_distance(x::TTNTensor, y::TTNTensor)
    _check_layout(x, y)
    tree = x.tree
    _isleaf(tree, tree.root) && return norm(x.leaves[tree.root] - y.leaves[tree.root])
    leaves = Dict(node => hcat(x.leaves[node], y.leaves[node]) for node in tree.leaf_order)
    cores = Dict{Int,Array{ComplexF64,3}}()
    for node in tree.postorder
        _isleaf(tree, node) && continue
        a, b = x.cores[node], y.cores[node]
        pa, la, ra = size(a)
        pb, lb, rb = size(b)
        isroot = node == tree.root
        core = zeros(ComplexF64, isroot ? 1 : pa+pb, la+lb, ra+rb)
        core[1:pa, 1:la, 1:ra] .= a
        core[isroot ? (1:1) : (pa+1:pa+pb), la+1:la+lb, ra+1:ra+rb] .= isroot ? -b : b
        cores[node] = core
    end
    difference = orthogonalize!(TTNTensor(tree, leaves, cores))
    return norm(difference.cores[tree.root])
end

"""
    orthogonalize!(state)

Move the orthogonality center to the root by bottom-up thin QR. Every
non-root leaf/subtree basis becomes orthonormal; root coefficients retain
the tensor's amplitude and phase. A single-leaf root is left unchanged.

No normalization or tolerance-based truncation is performed. Locally
overcomplete ranks shrink to the unfolding's row dimension. Rank-deficient
and zero inputs are allowed. Returns `state`.
"""
function orthogonalize!(state::TTNTensor)
    tree = state.tree
    for node in tree.postorder
        node == tree.root && continue
        basis = if _isleaf(tree, node)
            state.leaves[node]
        else
            core = state.cores[node]
            Matrix(transpose(reshape(core, size(core, 1), :)))
        end
        factorization = qr(basis)
        rank = min(size(basis)...)
        q = Matrix(factorization.Q)[:, 1:rank]
        r = Matrix(factorization.R)
        if _isleaf(tree, node)
            state.leaves[node] = q
        else
            _, left_rank, right_rank = size(state.cores[node])
            state.cores[node] = reshape(Matrix(transpose(q)), rank, left_rank, right_rank)
        end
        parent = tree.parent[node]
        mode = tree.children[parent][1] == node ? 2 : 3
        state.cores[parent] = _child_product(state.cores[parent], r, mode)
    end
    return state
end
