@testset "Conservative TTN truncation" begin
    tree = balanced_tree(2)
    leaves = Dict(n => Matrix{ComplexF64}(I, 3, 3) for n in tree.leaf_order)
    core = reshape(ComplexF64[1 0 0; 0 0.01 0; 0 0 0.0001], 1, 3, 3)
    x = TTNTensor(tree, leaves, Dict(tree.root => core))
    c = product_state(tree, [[0, 0, 1], [0, 0, 1]])
    y = copy(x)
    info = truncate!(y; atol=0.03, conserved=c)
    @test dot(c, y) ≈ dot(c, x) atol=1e-14
    @test maximum(tree_ranks(y)) == 2
    @test norm(dense(y)-dense(x)) <= info.discarded_bound + 1e-14
    @test info.discarded_bound <= 0.03
    @test info.tolerance_met
    test_root_gauge(y)

    # A functional with bond rank two, rather than a single product direction.
    c2 = TTNTensor(tree, Dict(n=>Matrix{ComplexF64}(I, 3, 3)[:,1:2] for n in tree.leaf_order),
                   Dict(tree.root=>reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2)))
    z = copy(x)
    info2 = truncate!(z; atol=0, maxrank=2, conserved=c2)
    @test dot(c2, z) ≈ dot(c2, x) atol=1e-14
    @test norm(dense(z)-dense(x)) <= info2.discarded_bound + 1e-14
    @test !info2.tolerance_met
    @test_throws ArgumentError truncate!(copy(x); conserved=c2, maxrank=1)

    tree, leaves, cores, _ = three_leaf_fixture(; ranks=(2, 3, 2, 6))
    x = TTNTensor(tree, leaves, cores)
    c = product_state(tree, [ComplexF64[1, im], [1, 2, 3], [2, 1]])
    y = copy(x)
    info = truncate!(y; atol=0, maxrank=1, conserved=c)
    @test maximum(tree_ranks(y)) == 1
    @test dot(c, y) ≈ dot(c, x) rtol=1e-12
    @test norm(dense(y)-dense(x)) <= info.discarded_bound + 1e-10
    @test !info.tolerance_met
    test_root_gauge(y)
    z = copy(x)
    info = truncate!(z; atol=0)
    @test dense(z) ≈ dense(x) rtol=1e-12
    @test info.discarded_bound == 0
    @test_throws ArgumentError truncate!(copy(x); atol=-1)
    @test_throws ArgumentError truncate!(copy(x); maxrank=0)
    @test_throws ArgumentError truncate!(copy(x); conserved=x, maxrank=1)
    small = product_state(tree, [[1, 0], [1, 0, 0], [1, 0]])
    @test_throws ArgumentError truncate!(small; conserved=c)
    single = product_state(balanced_tree(1), [ComplexF64[2, im]])
    @test truncate!(single; atol=1).discarded_bound == 0
    @test dense(single) == ComplexF64[2, im]
end
