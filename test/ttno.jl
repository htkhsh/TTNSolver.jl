@testset "Tree operator compilation and action" begin
    rng=MersenneTwister(918)
    trees=(balanced_tree(1),balanced_tree(3),
        BinaryTree([(2,5),(0,0),(0,0),(0,0),(4,3)];root=1,leaf_order=[3,2,4]))
    for tree in trees
        dims=collect(2:1+length(tree.leaf_order))
        terms=[ProductTerm(randn(rng,ComplexF64),Dict(j=>randn(rng,ComplexF64,n,n)))
               for (j,n) in enumerate(dims)]
        push!(terms,ProductTerm(0.2im,Dict()),ProductTerm(-0.3,Dict(
            j=>randn(rng,ComplexF64,n,n) for (j,n) in enumerate(dims))))
        A=SumOfProducts(dims,terms)
        x=product_state(tree,[randn(rng,ComplexF64,n) for n in dims])
        x=TTNSolver._tamen_add(x,product_state(tree,[randn(rng,ComplexF64,n) for n in dims]),0.2im)
        before=dense(x); matrix=dense(A)
        W=TreeTensorOperator(A,tree)
        @test leafdims(W)==dims
        @test dense(W) ≈ matrix atol=2e-12
        @test vec(dense(apply(W,x))) ≈ matrix*vec(before) atol=2e-11
        @test W.discarded_bound==0
        @test dense(x)==before
        @test dense(A)==matrix
        @test_throws ArgumentError apply(W,x;max_intermediate_bytes=1)
        @test dense(x)==before
        # Operator compression has a Frobenius bound; rank caps may exceed atol.
        cut=TreeTensorOperator(A,tree;atol=1e-12,maxrank=1)
        @test norm(dense(cut)-matrix)<=cut.discarded_bound+2e-11
        @test norm(vec(dense(apply(cut,x)))-matrix*vec(before))<=
            cut.discarded_bound*norm(x)+3e-11
        zeroop=TreeTensorOperator(SumOfProducts(dims,ProductTerm[]),tree)
        @test iszero(dense(zeroop))
        @test norm(apply(zeroop,x))<1e-13
        @test_throws DimensionMismatch TreeTensorOperator(A,balanced_tree(length(dims)+1))
    end
end

@testset "tAMEn can certify with a tree operator" begin
    tree=balanced_tree(2)
    x=product_state(tree,[[1.,0],[1.,0]])
    c=product_state(tree,[ones(2),ones(2)])
    G=ComplexF64[-1 0.2;1 -0.2]
    A=SumOfProducts([2,2],[ProductTerm(1,Dict(1=>G)),ProductTerm(0.3,Dict(2=>G))])
    exact=exp(0.1dense(A))*vec(dense(x))
    for sweep in (:serial,:etter)
        y,d=tamen_step(x,A,0.1;time_points=5,rtol=1e-9,maxrank=8,maxsweeps=10,
            conserved=c,residual_backend=:ttno,als_sweep=sweep,threaded=true)
        @test d.converged && d.relative_residual_bound<=1e-9
        @test d.residual_backend==:ttno
        @test vec(dense(y)) ≈ exact atol=2e-9
        @test abs(dot(c,y)-1)<1e-10
        @test !isempty(d.residual_operator_ranks)
        @test isfinite(d.residual_operator_discarded_bound)
    end
    @test_throws ArgumentError tamen_step(x,A,0.1;residual_backend=:invalid)
end

@testset "Tree operator residual certificates" begin
    rng=MersenneTwister(919)
    tree=balanced_tree(3)
    A=SumOfProducts([2,3,2],[ProductTerm(0.3im,Dict(1=>randn(rng,ComplexF64,2,2))),
        ProductTerm(-0.8,Dict(2=>randn(rng,ComplexF64,3,3),3=>randn(rng,ComplexF64,2,2)))])
    x=product_state(tree,[randn(rng,ComplexF64,n) for n in A.dimensions])
    tiny=product_state(tree,[randn(rng,ComplexF64,n) for n in A.dimensions])
    b=TTNSolver._tamen_add(apply(A,x),tiny,1e-9)
    exact=vec(dense(b))-dense(A)*vec(dense(x))
    for operator_cap in (1,20), residual_cap in (1,20)
        W=TreeTensorOperator(A,tree;atol=1e-13,maxrank=operator_cap)
        r,bound,dropped=TTNSolver._tamen_residual(W,x,b,1e-13,residual_cap)
        @test norm(exact-vec(dense(r)))<=dropped+3e-12
        @test norm(exact)<=bound+3e-12
        if operator_cap==20 && residual_cap==20
            @test norm(exact-vec(dense(r)))<3e-12
            @test bound<1e-7
        end
    end
    W=TreeTensorOperator(A,tree;atol=1e-13,maxrank=1)
    larger=copy(x);larger.cores[tree.root] .*= 1e6
    r,bound,dropped=TTNSolver._tamen_residual(W,larger,b,1e-13,20)
    large_exact=vec(dense(b))-dense(A)*vec(dense(larger))
    @test norm(large_exact-vec(dense(r)))<=dropped+1e-5
    @test norm(large_exact)<=bound+1e-5
    # Many leaves rule out a hidden global dense operator or state expansion.
    tree=balanced_tree(32);dims=fill(2,32)
    A=SumOfProducts(dims,[ProductTerm(1,Dict(j=>ComplexF64[1 0;0 -1])) for j in 1:32])
    W=TreeTensorOperator(A,tree)
    x=product_state(tree,[[1,0] for _ in dims])
    @test dot(x,apply(W,x)) ≈ 32 atol=1e-10
    @test maximum(tree_ranks(W))<length(A.terms)
end
