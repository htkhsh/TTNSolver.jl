@testset "Buffered basis orthogonalization" begin
    rng=MersenneTwister(9101)
    A=randn(rng,ComplexF64,128,32)
    Q=TTNSolver._orth_columns(A)
    @test Q'Q ≈ I atol=1e-12
    @test norm(A-Q*(Q'A))<1e-11
    @test Q[:,1] ≈ A[:,1]/norm(A[:,1]) atol=1e-13
    tiny=1e-100 .* A[:,1:2]
    Z=TTNSolver._orth_columns(zeros(ComplexF64,128,1),tiny,A)
    @test size(Z)==(128,32)
    @test Z[:,1:2] ≈ Q[:,1:2] atol=1e-12
    @test TTNSolver._orth_columns(zeros(3,2))==zeros(ComplexF64,3,0)
    @test size(TTNSolver._orth_columns(A'))==(32,32)
    before=copy(A)
    bytes=@allocated TTNSolver._orth_columns(A)
    @test bytes<200_000 # two basis allocations and vector workspaces, not one per column
    @test A==before
end
