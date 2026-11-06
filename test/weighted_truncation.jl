using Test, Random, LinearAlgebra, TTNSolver

# The weak root component has weight eps; its leaf-1 tail has weight eps*delta.
# An unweighted internal-core SVD sees delta instead and keeps an extra column.
function weighted_rank_fixture()
    tree=balanced_tree(4)
    leaves=Dict(n=>Matrix{ComplexF64}(I,2,2) for n in tree.leaf_order)
    left,right=tree.children[tree.root]
    A=zeros(ComplexF64,2,2,2); B=copy(A)
    A[1,1,1]=1; A[2,1,2]=sqrt(1-0.01^2); A[2,2,1]=0.01im
    B[1,1,1]=1; B[2,2,2]=1
    root=reshape(ComplexF64[1 0;0 0.001im],1,2,2)
    return TTNTensor(tree,leaves,Dict(left=>A,right=>B,tree.root=>root))
end

@testset "Amplitude-weighted protected truncation" begin
    x=weighted_rank_fixture(); original=dense(x)
    local_state=copy(x); truncate!(local_state;atol=1e-4,weighting=:local)
    @test tree_ranks(local_state)[1]==2
    c=product_state(x.tree,[[1,0] for _ in 1:4])
    y=copy(x)
    info=truncate!(y;atol=1e-4,conserved=c,weighting=:weighted)
    @test tree_ranks(y)[1]==1
    @test norm(dense(y)-original) ≈ 1e-5 atol=1e-12
    @test norm(dense(y)-original)<=info.discarded_bound+1e-12
    @test info.tolerance_met && info.discarded_bound<=1e-4
    @test dot(c,y) ≈ dot(c,x) atol=1e-13
    @test dense(x)==original

    # The default must discard the globally tiny direction as well.
    default_state=copy(x)
    truncate!(default_state;atol=1e-4,conserved=c)
    @test tree_ranks(default_state)[1]==1
    @test norm(dense(default_state)-original) ≈ 1e-5 atol=1e-12

    # Scaling the amplitudes and absolute budget must preserve rank decisions.
    scaled=copy(x); scaled.cores[x.tree.root].*=1e-7im
    truncate!(scaled;atol=1e-11,conserved=c,weighting=:weighted)
    @test tree_ranks(scaled)==tree_ranks(y)
    @test dense(scaled) ≈ 1e-7im*dense(y) atol=1e-18

    # Random complex states detect transpose/conjugation mistakes in the
    # downward complementary factors; errors are checked against dense data.
    rng=MersenneTwister(2003)
    for leaves_count in (3,4,5), cap in (1,2,8), protected in (false,true)
        tree=balanced_tree(leaves_count)
        z=TTNTensor(tree,Dict(n=>randn(rng,ComplexF64,2,2) for n in tree.leaf_order),
            Dict(n=>randn(rng,ComplexF64,n==tree.root ? 1 : 2,2,2)
                 for n in tree.postorder if !TTNSolver._isleaf(tree,n)))
        v=product_state(tree,[randn(rng,ComplexF64,2) for _ in 1:leaves_count])
        protected && (z=enrich_basis(z,v))
        before=dense(z); value=dot(v,z)
        d=truncate!(z;atol=0.1,maxrank=cap,conserved=protected ? v : nothing,weighting=:weighted)
        @test norm(dense(z)-before)<=d.discarded_bound+1e-10*norm(before)
        @test maximum(tree_ranks(z))<=cap
        @test d.tolerance_met==(d.discarded_bound<=0.1)
        protected && @test dot(v,z) ≈ value rtol=1e-10 atol=1e-10
    end

    z=copy(x); z.cores[z.tree.root].*=0
    d=truncate!(z;atol=1e-8,conserved=c,weighting=:weighted)
    @test norm(z)==0 && maximum(tree_ranks(z))==1 && d.discarded_bound==0
    z=copy(x); d=truncate!(z;atol=0,maxrank=1,weighting=:weighted)
    @test !d.tolerance_met
    @test norm(dense(z)-original)<=d.discarded_bound+1e-12
    z=copy(x); d=truncate!(z;atol=0,weighting=:weighted)
    @test dense(z) ≈ original atol=1e-12
    @test d.discarded_bound==0
    single=product_state(balanced_tree(1),[ComplexF64[2,im]])
    @test truncate!(single;weighting=:weighted).discarded_bound==0
    @test dense(single)==ComplexF64[2,im]
    @test_throws ArgumentError truncate!(x;weighting=:invalid)
    @test dense(x)==original
    @test_throws ArgumentError truncate!(x;conserved=x,maxrank=1,weighting=:weighted)
    @test dense(x)==original
end

@testset "BUG weighted truncation dispatch" begin
    x=weighted_rank_fixture()
    # A leaf generator annihilating the vacuum functional.
    op=SumOfProducts(fill(2,4),[ProductTerm(-0.1,Dict(1=>ComplexF64[0 0;0 1]))])
    c=product_state(x.tree,[[1,0] for _ in 1:4])
    exact=exp(0.01*dense(op))*vec(dense(x))
    for step in (augmented_bug_step,parallel_bug_step), threaded in (false,true)
        y,d=step(x,op,0.01;atol=1e-4,conserved=c,truncation=:weighted,
                 correct_invariant=false,threaded)
        @test norm(vec(dense(y))-exact)<=1e-4
        @test dot(c,y) ≈ dot(c,x) atol=1e-12
        @test tree_ranks(y)[1]==1
        @test d.truncation.tolerance_met
        @test_throws ArgumentError step(x,op,0.01;truncation=:invalid)
        default_state,_=step(x,op,0.01;atol=1e-4,conserved=c,
                             correct_invariant=false,threaded)
        local_state,_=step(x,op,0.01;atol=1e-4,conserved=c,truncation=:local,
                           correct_invariant=false,threaded)
        @test tree_ranks(default_state)[1]==1
        @test tree_ranks(local_state)[1]==2
    end
end

@testset "Adaptive weighted default and local opt-out" begin
    x=weighted_rank_fixture()
    op=SumOfProducts(fill(2,4),[ProductTerm(-0.1,Dict(1=>ComplexF64[0 0;0 1]))])
    c=product_state(x.tree,[[1,0] for _ in 1:4])
    exact=exp(0.01*dense(op))*vec(dense(x))
    for method in (:augmented,:parallel)
        options=(;dt=0.01,atol=2e-3,rtol=0.0,conserved=c,
                  correct_invariant=false,threaded=false,method)
        result=adaptive_evolve(x,op,(0.0,0.01);options...)
        @test tree_ranks(result.final_state)[1]==1
        @test norm(vec(dense(result.final_state))-exact)<2e-5
        @test dot(c,result.final_state) ≈ dot(c,x) atol=1e-12
        local_result=adaptive_evolve(x,op,(0.0,0.01);options...,truncation=:local)
        @test tree_ranks(local_result.final_state)[1]==2
        # Parallel BUG still has first-order time error when no tail is cut.
        @test norm(vec(dense(local_result.final_state))-exact)<1e-10
        @test_throws ArgumentError adaptive_evolve(x,op,(0.0,0.0);options...,truncation=:invalid)
    end
end
