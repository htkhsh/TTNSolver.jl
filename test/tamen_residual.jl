@testset "Local auxiliary residual TTN" begin
    rng = MersenneTwister(971)
    tree = BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)];root=1,leaf_order=[5,2,4])
    x = TTNTensor(tree,Dict(n=>randn(rng,ComplexF64,2,2) for n in tree.leaf_order),
        Dict(1=>randn(rng,ComplexF64,1,2,2),3=>randn(rng,ComplexF64,2,2,2)))
    b = product_state(tree,[randn(rng,ComplexF64,2) for _ in 1:3])
    A = SumOfProducts([2,2,2],[ProductTerm(0.3+0.4im,
        Dict(1=>randn(rng,ComplexF64,2,2),3=>randn(rng,ComplexF64,2,2))),
        ProductTerm(-0.7im,Dict(2=>randn(rng,ComplexF64,2,2)))])
    xd,bd = dense(x),dense(b)
    expected = vec(bd)-dense(A)*vec(xd)
    z,d = TTNSolver._tamen_local_residual(A,x,b;rank=4,sweeps=2)
    @test vec(dense(z)) ≈ expected atol=1e-11
    @test d.estimate ≈ norm(expected) atol=1e-11
    @test maximum(tree_ranks(z)) <= 4
    @test dense(x) == xd && dense(b) == bd
    zd = dense(z)
    again,da = TTNSolver._tamen_local_residual(A,x,b,z;rank=4)
    @test vec(dense(again)) ≈ expected atol=1e-11
    @test dense(z) == zd
    low,dl = TTNSolver._tamen_local_residual(A,x,b;rank=1,sweeps=2)
    @test maximum(tree_ranks(low)) == 1
    @test dl.estimate <= norm(expected)+1e-11

    # A product seed using only x's bases misses this entirely. Private seeded
    # auxiliary bases must discover a coupling changing all three leaves.
    product = product_state(tree,[[1,0] for _ in 1:3])
    flip = ComplexF64[0 1;1 0]
    coupled = SumOfProducts([2,2,2],[ProductTerm(1,Dict(j=>flip for j in 1:3))])
    zero = product_state(tree,[[0,0],[1,0],[1,0]])
    direction,dd = TTNSolver._tamen_local_residual(coupled,product,zero;rank=1,sweeps=2)
    @test vec(dense(direction)) ≈ -dense(coupled)*vec(dense(product)) atol=1e-12
    @test dd.estimate ≈ 1 atol=1e-12
    # An intentionally blind reused basis can have zero projected norm while
    # the actual residual is one. This is why local estimates are not bounds.
    blind,db = TTNSolver._tamen_local_residual(coupled,product,zero,product;rank=1)
    @test db.estimate < 1e-12
    @test norm(vec(dense(zero))-dense(coupled)*vec(dense(product))) ≈ 1

    identity = SumOfProducts([2,2,2],[ProductTerm(1,Dict())])
    zerores,dz = TTNSolver._tamen_local_residual(identity,product,product;rank=3)
    @test norm(zerores) < 1e-12
    @test dz.estimate < 1e-12
    # No mutation of the caller's global random stream.
    Random.seed!(94); expected_random = rand()
    Random.seed!(94)
    TTNSolver._tamen_local_residual(A,x,b;rank=2)
    @test rand() == expected_random

    single = product_state(balanced_tree(1),[ComplexF64[1,im]])
    B = SumOfProducts([2],[ProductTerm(0.4im,Dict(1=>flip))])
    zs,ds = TTNSolver._tamen_local_residual(B,single,single;rank=2)
    @test vec(dense(zs)) ≈ vec(dense(single))-dense(B)*vec(dense(single)) atol=1e-12
end
