@testset "Subtree basis enrichment" begin
    tree = BinaryTree([(0,0),(0,0),(0,0),(1,3),(4,2)]; leaf_order=[3,1,2])
    x = product_state(tree,[ComplexF64[1,im],ComplexF64[2,1,0],ComplexF64[1,2im]])
    H = SumOfProducts([2,3,2],[ProductTerm(1,Dict(1=>ComplexF64[1 0;0 -1],
        2=>ComplexF64[0 1 0;1 0 1;0 1 0])),ProductTerm(1,Dict(3=>ComplexF64[0 im;-im 0]))])
    direction = H*x
    original,original_direction = dense(x),dense(direction)
    enriched = enrich_basis(x,direction)
    @test dense(enriched) ≈ original atol=1e-13
    @test dense(x) == original
    @test dense(direction) == original_direction
    @test maximum(tree_ranks(enriched)) > 1
    @test norm(enriched) ≈ norm(x)
    @test dense(enrich_basis(enriched,direction)) ≈ original atol=1e-13
    zero = product_state(tree,[zeros(2),ones(3),ones(2)])
    @test norm(enrich_basis(zero,direction)) == 0
    @test dense(enrich_basis(x,zero)) ≈ original atol=1e-13
    @test dense(orthogonalize!(copy(enriched))) ≈ original atol=1e-13
    @test tree_ranks(orthogonalize!(copy(enriched))) == tree_ranks(enriched)
    one = product_state(balanced_tree(1),[ComplexF64[1,im]])
    @test dense(enrich_basis(one,one)) == dense(one)
    @test_throws ArgumentError enrich_basis(x,product_state(balanced_tree(3),[ones(2),ones(3),ones(2)]))
    @test_throws DimensionMismatch enrich_basis(x,product_state(tree,[ones(2),ones(2),ones(2)]))
end
