@testset "Local BUG workspaces preserve complex actions" begin
    rng = MersenneTwister(140)
    # Unequal axes and complex, non-Hermitian matrices catch transpose/folding
    # errors. Build an independent Kronecker operator, not a tensor contraction.
    p,l,r = 2,3,4
    coefficients = ComplexF64[0.3im,-0.2+0.7im,0]
    op = SumOfProducts([l,r],[ProductTerm(c,Dict()) for c in coefficients])
    boundary = [randn(rng,ComplexF64,p,p) for _ in coefficients]
    left = [randn(rng,ComplexF64,l,l) for _ in coefficients]
    right = [randn(rng,ComplexF64,r,r) for _ in coefficients]
    x = randn(rng,ComplexF64,p,l,r)
    reference = sum(c*kron(R,L,B) for (c,R,L,B) in zip(coefficients,right,left,boundary))
    action = TTNSolver._core_rhs_workspace(op,boundary,left,right,x)
    input = copy(x)
    y = action(x)
    @test vec(y) ≈ reference*vec(x) rtol=2e-13
    @test x == input
    saved = copy(y)
    @test iszero(action(zero(x)))
    @test y == saved
    @test action(2x) ≈ 2saved rtol=2e-13
    y .= 0
    @test vec(action(x)) ≈ reference*vec(x) rtol=2e-13

    factors = [randn(rng,ComplexF64,l,l),Matrix{ComplexF64}(I,l,l),randn(rng,ComplexF64,l,l)]
    leafop = SumOfProducts([l],[ProductTerm(coefficients[k],k == 2 ? Dict() : Dict(1=>factors[k]))
                               for k in eachindex(coefficients)])
    leaf = randn(rng,ComplexF64,l,p)
    leafref = sum(c*kron(B,A) for (c,B,A) in zip(coefficients,boundary,factors))
    leafaction = TTNSolver._leaf_rhs_workspace(leafop,1,boundary,leaf)
    original = copy(leaf)
    z = leafaction(leaf)
    @test vec(z) ≈ leafref*vec(leaf) rtol=2e-13
    @test leaf == original
    saved = copy(z)
    @test iszero(leafaction(zero(leaf)))
    @test z == saved
    @test leafaction(2leaf) ≈ 2saved rtol=2e-13

    emptyop = SumOfProducts([l],ProductTerm[])
    @test iszero(TTNSolver._leaf_rhs_workspace(emptyop,1,Matrix{ComplexF64}[],leaf)(leaf))
    @test iszero(TTNSolver._core_rhs_workspace(emptyop,Matrix{ComplexF64}[],
        Matrix{ComplexF64}[],Matrix{ComplexF64}[],x)(x))
end
