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
    environments = Vector{Matrix{ComplexF64}}(undef, length(x.tree.children))
    for node in x.tree.postorder
        if _isleaf(x.tree, node)
            environments[node] = x.leaves[node]' * y.leaves[node]
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

"""Compute the Euclidean norm by tree contraction without dense expansion."""
LinearAlgebra.norm(x::TTNTensor) = sqrt(max(0.0, real(dot(x, x))))

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
