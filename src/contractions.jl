# Mode products act on tensor coefficients, so folding uses transpose, not '.
function _parent_product(core::Array{ComplexF64,3}, matrix::AbstractMatrix)
    return reshape(matrix * reshape(core, size(core, 1), :),
                   size(matrix, 1), size(core, 2), size(core, 3))
end

_basis_matrix(core::Array{ComplexF64,3}) = Matrix(transpose(reshape(core, size(core, 1), :)))

function _project_children(core, left, right)
    return _child_product(_child_product(core, left, 2), right, 3)
end

function _subtree_operator(core, left, right)
    return _basis_matrix(core)' * _basis_matrix(_project_children(core, left, right))
end

# The complement of child i has axes (parent boundary, sibling, child rank).
function _complement_matrix(core, mode)
    permutation = mode == 2 ? (1, 3, 2) : (1, 2, 3)
    data = permutedims(core, permutation)
    return reshape(data, :, size(core, mode))
end

function _complement_operator(Q, boundary, sibling)
    b, s = size(boundary, 1), size(sibling, 1)
    data = reshape(Q, b, s, :)
    acted = _child_product(_parent_product(data, boundary), sibling, 2)
    return Q' * reshape(acted, b*s, :)
end

function _upward_operators(operator, state)
    up = [Vector{Matrix{ComplexF64}}(undef, length(operator.terms))
          for _ in state.tree.children]
    modes = Dict(node => mode for (mode, node) in enumerate(state.tree.leaf_order))
    for node in state.tree.postorder
        for (k, term) in enumerate(operator.terms)
            up[node][k] = if _isleaf(state.tree, node)
                U = state.leaves[node]
                U' * _local_action(term, modes[node], U)
            else
                left, right = state.tree.children[node]
                _subtree_operator(state.cores[node], up[left][k], up[right][k])
            end
        end
    end
    return up
end

"""Owned snapshot of termwise subtree and complementary environments.
Treat fields as read-only. Rebuild after changing a state or operator.
"""
struct OperatorEnvironments
    operator::SumOfProducts
    state::TTNTensor
    upward::Vector{Vector{Matrix{ComplexF64}}}
    downward::Vector{Vector{Matrix{ComplexF64}}}
end

"""
    operator_environments(operator, state)

Contract every product upward and downward through the tree. Coefficients
are included only when applying the local sum. Owns copies of the input
state and operator; no global dense tensor or matrix is constructed.
Works for nonorthogonal states as well as root-orthogonal states.
"""
function operator_environments(operator::SumOfProducts, state::TTNTensor)
    _check_operator(operator, state)
    x = copy(state)
    op = SumOfProducts(operator.dimensions, operator.terms)
    up = _upward_operators(op, x)
    down = [Matrix{ComplexF64}[] for _ in x.tree.children]
    down[x.tree.root] = [ones(ComplexF64, 1, 1) for _ in op.terms]
    for node in reverse(x.tree.postorder)
        _isleaf(x.tree, node) && continue
        left, right = x.tree.children[node]
        for (child, sibling, mode) in ((left, right, 2), (right, left, 3))
            Q = _complement_matrix(x.cores[node], mode)
            down[child] = [_complement_operator(Q, down[node][k], up[sibling][k])
                           for k in eachindex(op.terms)]
        end
    end
    return OperatorEnvironments(op, x, up, down)
end

function _leaf_rhs(operator, mode, boundary, x)
    result = zeros(ComplexF64, size(x))
    for (k, term) in enumerate(operator.terms)
        iszero(term.coefficient) && continue
        result .+= term.coefficient .* (_local_action(term, mode, x) * transpose(boundary[k]))
    end
    return result
end

function _core_rhs(operator, boundary, left, right, x)
    result = zeros(ComplexF64, size(x))
    for (k, term) in enumerate(operator.terms)
        iszero(term.coefficient) && continue
        acted = _parent_product(_project_children(x, left[k], right[k]), boundary[k])
        result .+= term.coefficient .* acted
    end
    return result
end

"""
    local_apply(environments, node, x)

Apply V' L V to local leaf/core coefficients x, where V inserts that node's
coefficients into the frozen snapshot. This is a Galerkin projection only
when the insertion map V is orthonormal; no inverse Gram matrix is implied.
"""
function local_apply(env::OperatorEnvironments, node::Integer, x::AbstractArray)
    tree = env.state.tree
    1 <= node <= length(tree.children) || throw(ArgumentError("invalid node ID"))
    old = _isleaf(tree, node) ? env.state.leaves[node] : env.state.cores[node]
    size(x) == size(old) || throw(DimensionMismatch("local coefficient shape differs"))
    data = ComplexF64.(x)
    if _isleaf(tree, node)
        mode = findfirst(==(node), tree.leaf_order)
        return _leaf_rhs(env.operator, mode, env.downward[node], data)
    end
    left, right = tree.children[node]
    return _core_rhs(env.operator, env.downward[node], env.upward[left], env.upward[right], data)
end
