"""
    ProductTerm(coefficient, factors)

A scalar times a product of local matrices. `factors` maps positive physical
mode positions (not tree node IDs) to square matrices; omitted modes are
identities. Empty factors represent a scalar identity. Inputs are copied and
converted to ComplexF64. Treat stored factor arrays as read-only.
"""
struct ProductTerm
    coefficient::ComplexF64
    factors::Dict{Int,Matrix{ComplexF64}}

    function ProductTerm(coefficient::Number, factors::AbstractDict)
        value = ComplexF64(coefficient)
        isfinite(value) || throw(ArgumentError("coefficient must be finite in ComplexF64"))
        owned = Dict{Int,Matrix{ComplexF64}}()
        for (mode, matrix) in factors
            mode isa Integer && mode >= 1 ||
                throw(ArgumentError("factor positions must be positive integers"))
            matrix isa AbstractMatrix && size(matrix, 1) == size(matrix, 2) ||
                throw(DimensionMismatch("factor $mode must be a square matrix"))
            !isempty(matrix) || throw(ArgumentError("factor $mode must be nonempty"))
            converted = Matrix{ComplexF64}(matrix)
            all(isfinite, converted) || throw(ArgumentError("factor $mode must be finite"))
            owned[mode] = converted
        end
        new(value, owned)
    end
end

"""
    SumOfProducts(dimensions, terms)

A topology-independent operator with fixed physical dimensions. Owns copies
of dimensions, terms and factors. An empty sum is zero. Matrices may be
non-Hermitian. Treat stored arrays and dictionaries as read-only.
"""
struct SumOfProducts
    dimensions::Vector{Int}
    terms::Vector{ProductTerm}

    function SumOfProducts(dimensions::AbstractVector, terms::AbstractVector{<:ProductTerm})
        !isempty(dimensions) && all(n -> n isa Integer && n > 0, dimensions) ||
            throw(ArgumentError("dimensions must be nonempty positive integers"))
        dims = Int.(dimensions)
        owned = ProductTerm[]
        for term in terms
            for (mode, matrix) in term.factors
                mode <= length(dims) || throw(ArgumentError("factor $mode is out of range"))
                size(matrix) == (dims[mode], dims[mode]) ||
                    throw(DimensionMismatch("factor $mode does not match physical dimension"))
            end
            push!(owned, ProductTerm(term.coefficient, term.factors))
        end
        new(dims, owned)
    end
end

"""Return a copy of the operator's physical mode dimensions."""
leafdims(operator::SumOfProducts) = copy(operator.dimensions)

"""
    dense(operator::SumOfProducts)

Build the matrix acting on `vec(dense(state))`, with physical mode 1 fastest.
For small reference problems only: matrix storage is quadratic in the full
tensor dimension.
"""
function dense(operator::SumOfProducts)
    dimension = prod(operator.dimensions)
    result = zeros(ComplexF64, dimension, dimension)
    for term in operator.terms
        iszero(term.coefficient) && continue
        product = ones(ComplexF64, 1, 1)
        for mode in reverse(eachindex(operator.dimensions))
            factor = get(term.factors, mode, nothing)
            if isnothing(factor)
                factor = Matrix{ComplexF64}(I, operator.dimensions[mode], operator.dimensions[mode])
            end
            product = kron(product, factor)
        end
        result .+= term.coefficient .* product
    end
    return result
end

function _check_operator(operator::SumOfProducts, state::TTNTensor)
    operator.dimensions == leafdims(state) ||
        throw(DimensionMismatch("operator and state physical dimensions differ"))
end

_local_action(term::ProductTerm, mode::Int, basis::Matrix{ComplexF64}) =
    haskey(term.factors, mode) ? term.factors[mode] * basis : basis

"""
    apply(operator::SumOfProducts, state::TTNTensor)
    operator * state

Apply each local product and combine terms exactly using block sums, without
dense expansion, compression or normalization. Returns independent numeric
storage on the same tree. Bond ranks can grow by the number of nonzero terms;
this is a correctness baseline, not a compressed propagation algorithm.
"""
function apply(operator::SumOfProducts, state::TTNTensor)
    _check_operator(operator, state)
    terms = filter(term -> !iszero(term.coefficient), operator.terms)
    tree = state.tree
    if isempty(terms)
        vectors = [ones(ComplexF64, n) for n in operator.dimensions]
        fill!(vectors[1], 0)
        return product_state(tree, vectors)
    end
    leaves = Dict{Int,Matrix{ComplexF64}}()
    if _isleaf(tree, tree.root)
        basis = state.leaves[tree.root]
        result = zeros(ComplexF64, size(basis))
        for term in terms
            result .+= term.coefficient .* _local_action(term, 1, basis)
        end
        leaves[tree.root] = result
        return TTNTensor(tree, leaves, Dict())
    end
    for (mode, node) in enumerate(tree.leaf_order)
        leaves[node] = hcat([_local_action(term, mode, state.leaves[node]) for term in terms]...)
    end
    cores = Dict{Int,Array{ComplexF64,3}}()
    count = length(terms)
    for (node, core) in state.cores
        rp, rl, rr = size(core)
        isroot = node == tree.root
        combined = zeros(ComplexF64, isroot ? 1 : count*rp, count*rl, count*rr)
        for (index, term) in enumerate(terms)
            parent = isroot ? (1:1) : ((index-1)*rp+1:index*rp)
            left = (index-1)*rl+1:index*rl
            right = (index-1)*rr+1:index*rr
            combined[parent, left, right] .= (isroot ? term.coefficient : 1) .* core
        end
        cores[node] = combined
    end
    return TTNTensor(tree, leaves, cores)
end

Base.:*(operator::SumOfProducts, state::TTNTensor) = apply(operator, state)

"""
    dot(x::TTNTensor, operator::SumOfProducts, y::TTNTensor)

Compute x' * operator * y by termwise tree contractions. Conjugates x, not
the term coefficients. Does not construct the rank-expanded action or a
dense tensor. x and y must have matching ordered trees and physical layouts.
"""
function LinearAlgebra.dot(x::TTNTensor, operator::SumOfProducts, y::TTNTensor)
    _check_layout(x, y)
    _check_operator(operator, y)
    modes = zeros(Int, length(y.tree.children))
    modes[y.tree.leaf_order] = eachindex(y.tree.leaf_order)
    result = zero(ComplexF64)
    for term in operator.terms
        iszero(term.coefficient) && continue
        leaf_environment = node -> x.leaves[node]' *
            _local_action(term, modes[node], y.leaves[node])
        result += term.coefficient * _tree_dot(x, y, leaf_environment)
    end
    return result
end
