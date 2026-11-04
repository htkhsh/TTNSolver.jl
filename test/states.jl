# A direct coefficient sum is independent of the matrix contraction used by dense.
function three_leaf_fixture(; ranks=(2, 2, 2, 3), order=[1, 2, 3], seed=412)
    rng = MersenneTwister(seed)
    a, b, c, d = ranks
    tree = BinaryTree([(0, 0), (0, 0), (0, 0), (2, 3), (1, 4)]; leaf_order=order)
    leaves = Dict(1 => randn(rng, ComplexF64, 2, a),
                  2 => randn(rng, ComplexF64, 3, b),
                  3 => randn(rng, ComplexF64, 2, c))
    cores = Dict(4 => randn(rng, ComplexF64, d, b, c),
                 5 => randn(rng, ComplexF64, 1, a, d))
    reference = [sum(leaves[1][i, α] * leaves[2][j, β] * leaves[3][k, γ] *
                     cores[4][δ, β, γ] * cores[5][1, α, δ]
                     for α in 1:a, β in 1:b, γ in 1:c, δ in 1:d)
                 for i in 1:2, j in 1:3, k in 1:2]
    return tree, leaves, cores, reference
end

@testset "TTN state representation" begin
    @testset "Complex coefficients and physical axis order" begin
        tree, leaves, cores, reference = three_leaf_fixture()
        state = TTNTensor(tree, leaves, cores)
        @test dense(state) ≈ reference atol=1e-12 rtol=1e-12
        @test leafdims(state) == [2, 3, 2]
        @test tree_ranks(state) == [2, 2, 2, 3, 1]
        ordered = BinaryTree(tree.children; root=tree.root, leaf_order=[3, 1, 2])
        permuted = TTNTensor(ordered, leaves, cores)
        @test dense(permuted) ≈ permutedims(reference, (3, 1, 2)) atol=1e-12
        @test leafdims(permuted) == [2, 2, 3]

        leaves[1] .= 0
        cores[4] .= 0
        @test dense(state) ≈ reference atol=1e-12
        duplicate = copy(state)
        duplicate.leaves[1] .= 0
        duplicate.cores[5] .= 0
        @test dense(state) ≈ reference atol=1e-12
        @test iszero(dense(duplicate))
    end

    @testset "Unnormalized products, single root and zero states" begin
        vectors = [ComplexF64[2im, 3], ComplexF64[1, -im, 2], ComplexF64[4, -1]]
        expected = [vectors[1][i] * vectors[2][j] * vectors[3][k]
                    for i in 1:2, j in 1:3, k in 1:2]
        tree = BinaryTree([(0, 0), (0, 0), (0, 0), (1, 2), (4, 3)];
                          leaf_order=[3, 1, 2])
        state = product_state(tree, vectors)
        @test dense(state) ≈ expected
        @test tree_ranks(state) == ones(Int, 5)
        @test leafdims(state) == [2, 3, 2]
        vectors[1] .= 0
        @test dense(state) ≈ expected
        @test iszero(dense(product_state(tree, vectors)))

        single = product_state(balanced_tree(1), [ComplexF64[2im, 3]])
        @test dense(single) == ComplexF64[2im, 3]
        @test size(dense(single)) == (2,)
        @test tree_ranks(single) == [1]
        @test eltype(dense(product_state(balanced_tree(1), [[1, 2]]))) == ComplexF64
    end

    @testset "Reject invalid tensor storage" begin
        tree, leaves, cores, _ = three_leaf_fixture()
        @test_throws ArgumentError TTNTensor(tree, Dict(1 => leaves[1]), cores)
        @test_throws ArgumentError TTNTensor(tree, leaves, Dict(4 => cores[4]))
        @test_throws ArgumentError TTNTensor(tree, merge(leaves, Dict(5 => ones(2, 1))), cores)
        @test_throws DimensionMismatch TTNTensor(tree, merge(leaves, Dict(1 => ones(2))), cores)
        @test_throws DimensionMismatch TTNTensor(tree, leaves, merge(cores, Dict(4 => ones(3, 4))))
        @test_throws ArgumentError TTNTensor(tree, merge(leaves, Dict(1 => zeros(0, 2))), cores)
        @test_throws ArgumentError TTNTensor(tree, merge(leaves, Dict(1 => fill(NaN, 2, 2))), cores)
        @test_throws ArgumentError TTNTensor(tree, leaves, merge(cores, Dict(4 => fill(Inf, 3, 2, 2))))
        @test_throws DimensionMismatch TTNTensor(tree, merge(leaves, Dict(1 => ones(2, 3))), cores)
        @test_throws DimensionMismatch TTNTensor(tree, leaves, merge(cores, Dict(5 => ones(2, 2, 3))))
        @test_throws DimensionMismatch TTNTensor(balanced_tree(1), Dict(1 => ones(2, 2)), Dict())
        @test_throws DimensionMismatch product_state(tree, [[1], [2]])
        @test_throws ArgumentError product_state(tree, [[1], [2], Float64[]])
        @test_throws ArgumentError product_state(tree, [[1], [2], [NaN]])
    end
end
