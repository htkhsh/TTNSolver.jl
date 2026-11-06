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

# Several SoP terms have identical restrictions to a subtree/complement.
# Cache only within this call, using exact matrix contents as keys. Returned
# entries may share storage and must remain read-only (as environments do).
# No cache survives a changed basis/operator or is shared between workers.
function _map_unique_pairs(f, left, right)
    cache = Dict{Tuple{Matrix{ComplexF64},Matrix{ComplexF64}},Matrix{ComplexF64}}()
    result = Vector{Matrix{ComplexF64}}(undef, length(left))
    for k in eachindex(left, right)
        result[k] = get!(cache, (left[k], right[k])) do
            f(left[k], right[k])
        end
    end
    return result
end

function _leaf_operators(operator, mode, bra, ket=bra)
    cache = Dict{Union{Nothing,Matrix{ComplexF64}},Matrix{ComplexF64}}()
    result = Vector{Matrix{ComplexF64}}(undef, length(operator.terms))
    for (k,term) in enumerate(operator.terms)
        factor = get(term.factors, mode, nothing)
        result[k] = get!(cache, factor) do
            bra' * (isnothing(factor) ? ket : factor * ket)
        end
    end
    return result
end

function _subtree_operators(core, left, right)
    basis = _basis_matrix(core)
    return _mixed_subtree_operators(basis, core, left, right)
end

function _mixed_subtree_operators(bra, core, left, right)
    isempty(left) && return Matrix{ComplexF64}[]
    p,l,r = size(core)
    a,b = size(first(left),1),size(first(right),1)
    # The input core is fixed throughout this batch. Fold it once, then reuse
    # every large intermediate for the distinct factor pairs at this node.
    # Only the small environment result returned from each call is retained.
    left_in = reshape(permutedims(core,(2,1,3)),l,:)
    left_out = Array{ComplexF64}(undef,a,p,r)
    right_in = Array{ComplexF64}(undef,r,p,a)
    right_out = Array{ComplexF64}(undef,b,p,a)
    projected = Array{ComplexF64}(undef,a,b,p)
    lm,ri,ro = reshape(left_out,a,:),reshape(right_in,r,:),reshape(right_out,b,:)
    basis = reshape(projected,a*b,p)
    return _map_unique_pairs(left, right) do L,R
        mul!(lm,L,left_in)
        permutedims!(right_in,left_out,(3,2,1))
        mul!(ro,R,ri)
        permutedims!(projected,right_out,(3,1,2))
        bra' * basis
    end
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

function _complement_operators(Q, boundary, sibling)
    isempty(boundary) && return Matrix{ComplexF64}[]
    b,s = size(first(boundary),1),size(first(sibling),1)
    # Express the complement as a subtree with Q as both bra and ket. This
    # uses the same buffered contraction for all distinct boundary pairs.
    core = permutedims(reshape(Q,b,s,:),(3,1,2))
    return _mixed_subtree_operators(Q,core,boundary,sibling)
end

function _upward_operators(operator, state)
    up = [Vector{Matrix{ComplexF64}}(undef, length(operator.terms))
          for _ in state.tree.children]
    modes = Dict(node => mode for (mode, node) in enumerate(state.tree.leaf_order))
    for node in state.tree.postorder
        up[node] = if _isleaf(state.tree, node)
            _leaf_operators(operator, modes[node], state.leaves[node])
        else
            left, right = state.tree.children[node]
            _subtree_operators(state.cores[node], up[left], up[right])
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
            down[child] = _complement_operators(Q, down[node], up[sibling])
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

function _core_rhs(operator, boundary, left, right, x, output_size=size(x))
    # Rectangular mixed contractions used during parallel BUG assembly have
    # a different output space; square local Galerkin actions use size(x).
    result = zeros(ComplexF64, output_size)
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
