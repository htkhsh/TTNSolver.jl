function certification_state(tree,dims,ranks,rng)
    leaves=Dict(node=>randn(rng,ComplexF64,d,node==tree.root ? 1 : ranks[node]) for (node,d) in zip(tree.leaf_order,dims))
    cores=Dict(node=>randn(rng,ComplexF64,node==tree.root ? 1 : ranks[node],ranks[a],ranks[b])
        for node in tree.postorder if !TTNSolver._isleaf(tree,node) for (a,b) in (tree.children[node],))
    return TTNTensor(tree,leaves,cores)
end

include("fixtures/tamen_residual_reference.jl")

@testset "Global canonical residual certificate" begin
    rng=MersenneTwister(998)
    for tree in (balanced_tree(1),balanced_tree(4),
                 BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)];root=1,leaf_order=[5,2,4]))
        d=length(tree.leaf_order)
        x=certification_state(tree,fill(3,d),fill(2,length(tree.children)),rng)
        b=certification_state(tree,fill(3,d),fill(3,length(tree.children)),rng)
        A=SumOfProducts(fill(3,d),[
            ProductTerm(0.3-0.2im,Dict(1=>randn(rng,ComplexF64,3,3))),
            ProductTerm(-0.1im,Dict(d=>randn(rng,ComplexF64,3,3))),
            ProductTerm(0,Dict()),ProductTerm(0.4,Dict())])
        original_x,original_b=dense(x),dense(b)
        exact=vec(original_b)-dense(A)*vec(original_x)
        for cap in (1,3,20)
            z,bound,dropped=TTNSolver._tamen_residual(A,x,b,1e-10,cap)
            @test norm(vec(dense(z))-exact)<=dropped+1e-11*norm(exact)
            @test norm(exact)<=bound+1e-11*norm(exact)
            @test bound ≈ norm(dense(z))+dropped rtol=1e-12
            @test maximum(tree_ranks(z))<=cap
            if !TTNSolver._isleaf(tree,tree.root)
                old,oldbound,olddrop=reference_tamen_residual(A,x,b,1e-10,cap)
                @test norm(vec(dense(old))-vec(dense(z)))<=olddrop+dropped+1e-10*norm(exact)
                @test abs(oldbound-bound)<=2(olddrop+dropped)+1e-10*norm(exact)
            end
        end
        @test dense(x)==original_x && dense(b)==original_b
        # Empty sums still return an independent canonical residual with its norm.
        z,bound,dropped=TTNSolver._tamen_residual(SumOfProducts(fill(3,d),ProductTerm[]),x,b,0.,20)
        @test dense(z) ≈ original_b rtol=1e-12
        @test bound ≈ norm(original_b) rtol=1e-12
        @test dropped==0
    end
    tree=balanced_tree(4)
    x=orthogonalize!(certification_state(tree,fill(3,4),fill(2,length(tree.children)),rng))
    b=copy(x); b.cores[tree.root].*=1+1e-10
    A=SumOfProducts(fill(3,4),[ProductTerm(0.25,Dict()),ProductTerm(0.75,Dict())])
    z,bound,dropped=TTNSolver._tamen_residual(A,x,b,1e-15*norm(x),30)
    expected=vec(dense(b))-vec(dense(x))
    @test norm(vec(dense(z))-expected)<1e-13*norm(x)
    @test 1e-11*norm(x)<bound<1e-9*norm(x)
    @test norm(expected)<=bound+1e-13*norm(x)
end

@testset "Canonical acted sum without block cores" begin
    rng=MersenneTwister(1964)
    for tree in (balanced_tree(1),balanced_tree(4),
                 BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)];root=1,leaf_order=[5,2,4]))
        d=length(tree.leaf_order)
        rx,ry=fill(2,length(tree.children)),fill(3,length(tree.children))
        rx[tree.root]=ry[tree.root]=1
        x=certification_state(tree,fill(3,d),rx,rng)
        y=certification_state(tree,fill(3,d),ry,rng)
        dx,dy=dense(x),dense(y)
        term=ProductTerm(-0.3+0.2im,Dict(1=>randn(rng,ComplexF64,3,3)))
        op=SumOfProducts(fill(3,d),[term])
        z=TTNSolver._tamen_add_action_canonical(x,y,term,-1)
        @test vec(dense(z)) ≈ vec(dx)-dense(op)*vec(dy) rtol=2e-12
        @test dense(x)==dx && dense(y)==dy
        for node in tree.postorder
            node==tree.root && continue
            Q=TTNSolver._isleaf(tree,node) ? z.leaves[node] : TTNSolver._basis_matrix(z.cores[node])
            @test Q'*Q ≈ I atol=1e-12
        end
        # No output tensor storage may alias either input.
        first(values(z.leaves))[1]+=1
        @test dense(x)==dx && dense(y)==dy
    end
end

@testset "Canonical sum cancellation and weighted bounds" begin
    rng=MersenneTwister(752)
    tree=balanced_tree(4)
    x=orthogonalize!(certification_state(tree,fill(3,4),[n==tree.root ? 1 : 3 for n in eachindex(tree.children)],rng))
    before=dense(x)
    for scale in (1.,1e-100)
        y=copy(x); y.cores[tree.root].*=scale
        z=TTNSolver._tamen_add_action_canonical(y,y,ProductTerm(1-1e-10,Dict()),-1)
        expected=scale*1e-10*before
        @test norm(dense(z)-expected)<1e-13*scale*norm(before)
        @test norm(z)>1e-11*scale*norm(before)
    end
    target=certification_state(tree,fill(3,4),fill(2,length(tree.children)),rng)
    exact=dense(x)+dense(target)
    for cap in (1,3,30)
        z=TTNSolver._tamen_add_action_canonical(x,target,ProductTerm(1,Dict()),1)
        pending=copy(z)
        pending_cut=TTNSolver._truncate_canonical!(pending;atol=1e-8,maxrank=cap,canonicalize=false)
        cut=TTNSolver._truncate_canonical!(z;atol=1e-8,maxrank=cap)
        @test norm(dense(z)-exact)<=cut.discarded_bound+1e-10*norm(exact)
        @test maximum(tree_ranks(z))<=cap
        @test norm(z.cores[tree.root]) ≈ norm(dense(z)) rtol=1e-12
        @test dense(pending) ≈ dense(z) rtol=1e-12
        @test pending_cut.discarded_bound ≈ cut.discarded_bound
        next=TTNSolver._tamen_add_action_canonical(pending,x,ProductTerm(0.2im,Dict()),-1)
        @test dense(next) ≈ dense(pending)-0.2im*dense(x) rtol=1e-12
    end
end
