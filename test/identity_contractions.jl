@testset "Identity environments require an isometric frame" begin
    rng = MersenneTwister(813)
    op = SumOfProducts([4], [ProductTerm(1,Dict())])
    Q = Matrix(qr(randn(rng,ComplexF64,4,3)).Q)
    # Explicit isometry contract: omitted operators must stay exact identities,
    # rather than turning into nearly-identity dense products at every node.
    up = TTNSolver._leaf_operators(op,1,Q;orthonormal=true)
    @test isone(only(up))
    V = Q * diagm([1.,2.,3.])
    @test only(TTNSolver._leaf_operators(op,1,V)) ≈ V'*V
    @test only(TTNSolver._leaf_operators(op,1,Q,V)) ≈ Q'*V
    @test !isone(only(TTNSolver._leaf_operators(op,1,Q,V)))
    C = permutedims(reshape(Q,2,2,3),(3,1,2))
    identity = [Matrix{ComplexF64}(I,2,2)]
    @test isone(only(TTNSolver._subtree_operators(C,identity,identity;orthonormal=true)))
    @test only(TTNSolver._subtree_operators(2C,identity,identity)) ≈ 4I
    @test isone(only(TTNSolver._complement_operators(Q,identity,identity;orthonormal=true)))
    @test only(TTNSolver._complement_operators(2Q,identity,identity)) ≈ 4I
end

@testset "Identity-specialized complex core and leaf actions" begin
    rng=MersenneTwister(814)
    shape=(2,3,4)
    x=randn(rng,ComplexF64,shape)
    op=SumOfProducts([3,4],[ProductTerm(0.3+0.2im,Dict())])
    for mask in 0:7
        B,L,R=[iszero(mask & (1<<(j-1))) ? randn(rng,ComplexF64,n,n) :
            Matrix{ComplexF64}(I,n,n) for (j,n) in enumerate(shape)]
        expected=(0.3+0.2im)*kron(R,L,B)*vec(x)
        action=TTNSolver._core_rhs_workspace(op,[B],[L],[R],x)
        y=action(x); saved=copy(y)
        @test vec(y) ≈ expected atol=1e-12
        @test action(2x) ≈ 2saved atol=1e-12
        @test y==saved
        @test vec(TTNSolver._parallel_core_block(op,[B],[L],[R],x,shape)) ≈ expected atol=1e-12
    end
    # Exact comparisons only: a small physical diagonal perturbation matters.
    B=Matrix{ComplexF64}(I,2,2); B[1,1]+=1e-8
    L=Matrix{ComplexF64}(I,3,3); R=Matrix{ComplexF64}(I,4,4)
    @test vec(TTNSolver._core_rhs_workspace(op,[B],[L],[R],x)(x)) ≈
        (0.3+0.2im)*kron(R,L,B)*vec(x) atol=1e-13
    for local_identity in (false,true), boundary_identity in (false,true)
        A=randn(rng,ComplexF64,3,3)
        B=boundary_identity ? Matrix{ComplexF64}(I,2,2) : randn(rng,ComplexF64,2,2)
        lop=SumOfProducts([3],[ProductTerm(0.2im,local_identity ? Dict() : Dict(1=>A))])
        U=randn(rng,ComplexF64,3,2)
        action=TTNSolver._leaf_rhs_workspace(lop,1,[B],U)
        @test action(U) ≈ 0.2im*(local_identity ? U : A*U)*transpose(B) atol=1e-13
    end
end
