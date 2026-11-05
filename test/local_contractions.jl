# Explicit insertion maps are an independent small-system reference.
function insertion_matrix(state, node)
    data = haskey(state.leaves, node) ? state.leaves[node] : state.cores[node]
    V = zeros(ComplexF64, prod(leafdims(state)), length(data))
    for j in eachindex(data)
        z = copy(state)
        target = haskey(z.leaves, node) ? z.leaves[node] : z.cores[node]
        fill!(target, 0)
        target[j] = 1
        V[:, j] = vec(dense(z))
    end
    return V
end

@testset "Local product environments" begin
    rng = MersenneTwister(923)
    for order in ([1, 2, 3], [3, 1, 2])
        tree, leaves, cores, _ = three_leaf_fixture(; order)
        state = TTNTensor(tree, leaves, cores)
        dims = leafdims(state)
        op = SumOfProducts(dims, [ProductTerm(0.3im, Dict()),
            ProductTerm(-0.2+0.8im, Dict(k => randn(rng, ComplexF64, n, n)
                                       for (k, n) in enumerate(dims)))])
        env = operator_environments(op, state)
        for node in tree.postorder
            data = haskey(leaves, node) ? leaves[node] : cores[node]
            x = randn(rng, ComplexF64, size(data))
            V = insertion_matrix(state, node)
            @test vec(local_apply(env, node, x)) ≈ V' * dense(op) * V * vec(x) rtol=1e-12
        end
        root = tree.root
        expected = local_apply(env, root, cores[root])
        state.leaves[tree.leaf_order[1]] .= 0
        @test local_apply(env, root, cores[root]) == expected
        @test_throws DimensionMismatch local_apply(env, root, ones(ComplexF64, 2, 1, 1))
        @test_throws ArgumentError local_apply(env, 0, cores[root])
    end
    x = product_state(balanced_tree(1), [ComplexF64[2, im]])
    z = SumOfProducts([2], ProductTerm[])
    @test iszero(local_apply(operator_environments(z, x), 1, x.leaves[1]))
    @test_throws DimensionMismatch operator_environments(z, product_state(balanced_tree(1), [ones(3)]))
end
