# Multiply either child mode by a matrix with shape (new_rank, old_rank).
# Mode-2 products use one GEMM per right-index slice unless the parent
# dimension is small, where one permuted GEMM is faster (measured crossover).
_slice_mode2(p, l) = 3p >= l

# Mode 3 is one GEMM on a (parent*left, right) view, without permutations.
function _child_product(core::Array{ComplexF64,3}, matrix::AbstractMatrix, mode::Int)
    p, l, r = size(core)
    if mode == 3
        return reshape(reshape(core, p*l, r) * transpose(matrix), p, l, size(matrix, 1))
    elseif mode == 2 && _slice_mode2(p, l)
        result = Array{ComplexF64}(undef, p, size(matrix, 1), r)
        Mt = transpose(matrix)
        for k in 1:r
            mul!(view(result, :, :, k), view(core, :, :, k), Mt)
        end
        return result
    end
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
function orthogonalize!(state::TTNTensor; threaded=false)
    tree = state.tree
    leaves,cores = _node_tensors(state)
    factors = Vector{Matrix{ComplexF64}}(undef, length(tree.children))
    position = _postorder_positions(tree)
    # A node takes its children's R factors (in postorder, as a serial sweep
    # applies them), then factors itself. Sibling subtrees are independent.
    _each_subtree(tree; threaded) do node
        if !_isleaf(tree, node)
            for child in sort(collect(tree.children[node]); by=c->position[c])
                mode = tree.children[node][1] == child ? 2 : 3
                cores[node] = _child_product(cores[node], factors[child], mode)
            end
        end
        node == tree.root && return
        basis = if _isleaf(tree, node)
            leaves[node]
        else
            core = cores[node]
            Matrix(transpose(reshape(core, size(core, 1), :)))
        end
        factorization = qr(basis)
        rank = min(size(basis)...)
        q = Matrix(factorization.Q)[:, 1:rank]
        factors[node] = Matrix(factorization.R)
        if _isleaf(tree, node)
            leaves[node] = q
        else
            _, left_rank, right_rank = size(cores[node])
            cores[node] = reshape(Matrix(transpose(q)), rank, left_rank, right_rank)
        end
    end
    _store_node_tensors!(state, leaves, cores)
    return state
end

# Per-node tensors in vectors indexed by node, so tasks never write the
# state's dictionaries concurrently; _store_node_tensors! writes them back.
function _node_tensors(state::TTNTensor)
    n = length(state.tree.children)
    leaves = Vector{Matrix{ComplexF64}}(undef, n)
    cores = Vector{Array{ComplexF64,3}}(undef, n)
    for (node, leaf) in state.leaves
        leaves[node] = leaf
    end
    for (node, core) in state.cores
        cores[node] = core
    end
    return leaves, cores
end
function _store_node_tensors!(state::TTNTensor, leaves, cores)
    for node in keys(state.leaves)
        state.leaves[node] = leaves[node]
    end
    for node in keys(state.cores)
        state.cores[node] = cores[node]
    end
    return state
end

function _postorder_positions(tree)
    position = zeros(Int, length(tree.children))
    for (i, node) in enumerate(tree.postorder)
        position[node] = i
    end
    return position
end

function _subtree_sizes(tree)
    sizes = ones(Int, length(tree.children))
    for node in tree.postorder
        _isleaf(tree, node) || (sizes[node] += sum(sizes[c] for c in tree.children[node]))
    end
    return sizes
end

# Tree pass for work where a node reads or writes only its own slot and its
# children's: `order=:post` runs `f(node)` after both child subtrees,
# `order=:pre` before them. With `threaded`, sibling subtrees run as tasks and
# split the thread budget by size. Each node's work is the same in any order,
# so results do not depend on the thread count.
function _each_subtree(f, tree; threaded=false, order=:post)
    sizes = _subtree_sizes(tree)
    function visit(node, budget)
        order == :pre && f(node)
        if !_isleaf(tree, node)
            left, right = tree.children[node]
            if budget > 1
                lb = clamp(round(Int, budget*sizes[left]/(sizes[left]+sizes[right])), 1, budget-1)
                task = Threads.@spawn visit(left, lb)
                right_error = try visit(right, budget-lb); nothing catch e; e end
                fetch(task)
                isnothing(right_error) || throw(right_error)
            else
                visit(left, 1)
                visit(right, 1)
            end
        end
        order == :post && f(node)
        return nothing
    end
    visit(tree.root, threaded ? Threads.nthreads(:default) : 1)
    return nothing
end
