_isleaf(tree::BinaryTree, node::Int) = tree.children[node] == (0, 0)

"""
    TTNTensor(tree, leaves, cores)

A ComplexF64 tree tensor with physical indices only at leaves. Dictionaries
are keyed by node ID: `leaves[node]` is an `(n, r)` matrix and `cores[node]`
is an `(r_parent, r_left, r_right)` array. The root rank must be one.

The constructor validates and copies numeric storage. Bases need not be
orthonormal. The tree is shared and must be treated as read-only. Do not
change array shapes or dictionary keys directly after construction.
"""
struct TTNTensor
    tree::BinaryTree
    leaves::Dict{Int,Matrix{ComplexF64}}
    cores::Dict{Int,Array{ComplexF64,3}}

    function TTNTensor(tree::BinaryTree, leaves::AbstractDict, cores::AbstractDict)
        Set(keys(leaves)) == Set(tree.leaf_order) ||
            throw(ArgumentError("leaves must have exactly the tree's leaf IDs"))
        inner = findall(node -> !_isleaf(tree, node), eachindex(tree.children))
        Set(keys(cores)) == Set(inner) ||
            throw(ArgumentError("cores must have exactly the tree's internal node IDs"))
        leaf_data = Dict{Int,Matrix{ComplexF64}}()
        core_data = Dict{Int,Array{ComplexF64,3}}()
        for (node, data) in leaves
            _validate_array(data, 2, node)
            leaf_data[node] = Matrix{ComplexF64}(data)
        end
        for (node, data) in cores
            _validate_array(data, 3, node)
            core_data[node] = Array{ComplexF64,3}(data)
        end
        # Conversion to ComplexF64 can itself overflow otherwise finite inputs.
        all(data -> all(isfinite, data), values(leaf_data)) &&
            all(data -> all(isfinite, data), values(core_data)) ||
            throw(ArgumentError("tensor entries must be finite in ComplexF64"))
        ranks = [(_isleaf(tree, node) ? size(leaf_data[node], 2) : size(core_data[node], 1))
                 for node in eachindex(tree.children)]
        ranks[tree.root] == 1 || throw(DimensionMismatch("root rank must be one"))
        for node in inner
            left, right = tree.children[node]
            size(core_data[node]) == (ranks[node], ranks[left], ranks[right]) ||
                throw(DimensionMismatch("core $node does not match its child ranks"))
        end
        return new(tree, leaf_data, core_data)
    end
end

function _validate_array(data, order, node)
    data isa AbstractArray && ndims(data) == order ||
        throw(DimensionMismatch("node $node requires an array of order $order"))
    !isempty(data) && all(isfinite, data) ||
        throw(ArgumentError("node $node requires nonempty finite data"))
end

"""Physical dimensions in `state.tree.leaf_order`."""
leafdims(state::TTNTensor) = [size(state.leaves[node], 1) for node in state.tree.leaf_order]

"""Node-to-parent ranks in node-ID order, including root rank one."""
tree_ranks(state::TTNTensor) =
    [_isleaf(state.tree, node) ? size(state.leaves[node], 2) : size(state.cores[node], 1)
     for node in eachindex(state.tree.children)]

"""Copy numeric storage independently, sharing the read-only topology."""
Base.copy(state::TTNTensor) = TTNTensor(state.tree, state.leaves, state.cores)

"""
    product_state(tree, vectors)

Build a rank-one tensor from local vectors in `tree.leaf_order`. The vectors
are copied and converted to ComplexF64 without normalization. Zero vectors
are allowed.
"""
function product_state(tree::BinaryTree, vectors::AbstractVector)
    length(vectors) == length(tree.leaf_order) ||
        throw(DimensionMismatch("provide one vector per physical leaf"))
    leaves = Dict{Int,Matrix{ComplexF64}}()
    for (node, vector) in zip(tree.leaf_order, vectors)
        vector isa AbstractVector && !isempty(vector) && all(isfinite, vector) ||
            throw(ArgumentError("leaf $node requires a nonempty finite vector"))
        leaves[node] = reshape(ComplexF64.(vector), :, 1)
    end
    cores = Dict(node => ones(ComplexF64, 1, 1, 1)
                 for node in eachindex(tree.children) if !_isleaf(tree, node))
    return TTNTensor(tree, leaves, cores)
end

function _dense_basis(state::TTNTensor, node::Int)
    _isleaf(state.tree, node) && return state.leaves[node], [node]
    left, right = state.tree.children[node]
    left_basis, left_order = _dense_basis(state, left)
    right_basis, right_order = _dense_basis(state, right)
    core = state.cores[node]
    # Column-major physical order: the left subtree index runs fastest.
    # Core coefficients are multiplied directly, without complex conjugation.
    basis = kron(right_basis, left_basis) * transpose(reshape(core, size(core, 1), :))
    return basis, vcat(left_order, right_order)
end

"""
    dense(state)

Expand to a dense array whose axes follow `state.tree.leaf_order`. `vec` then
uses Julia's column-major order, which differs from TTSolver's Kronecker
ordering. Intended for small-system reference calculations only.
"""
function dense(state::TTNTensor)
    basis, traversal_order = _dense_basis(state, state.tree.root)
    dimensions = Tuple(size(state.leaves[node], 1) for node in traversal_order)
    permutation = [only(findall(==(node), traversal_order)) for node in state.tree.leaf_order]
    return permutedims(reshape(basis[:, 1], dimensions), permutation)
end
