@testset "Per-invocation environment reuse" begin
    rng = MersenneTwister(862)
    B,L,R = [randn(rng,ComplexF64,n,n) for n in (2,3,4)]
    Lnear = copy(L); Lnear[1,1] += 1e-10
    left,right = [L,copy(L),Lnear], [R,copy(R),R]
    calls = Ref(0)
    products = TTNSolver._map_unique_pairs(left,right) do A,C
        calls[] += 1
        kron(C,A)
    end
    @test calls[] == 2
    @test products[1] == kron(R,L)
    @test products[3] == kron(R,Lnear)
    # Reuse must stay within this invocation, never across changed matrices.
    left[1][1,1] += 0.2
    new = TTNSolver._map_unique_pairs(kron,left,right)
    @test new[1] == kron(left[1],right[1])
    @test new[1] != new[2]

    core = randn(rng,ComplexF64,2,3,4)
    basis = TTNSolver._basis_matrix(core)
    before = deepcopy((core,left,right))
    up = TTNSolver._subtree_operators(core,left,right)
    for k in 1:3
        @test up[k] ≈ basis'*kron(right[k],left[k])*basis rtol=1e-12
    end
    @test (core,left,right) == before
    Q = randn(rng,ComplexF64,8,3)
    boundary,sibling = [B,copy(B),2B],[R,copy(R),R]
    down = TTNSolver._complement_operators(Q,boundary,sibling)
    for k in 1:3
        @test down[k] ≈ Q'*kron(sibling[k],boundary[k])*Q rtol=1e-12
    end
    # Mixed upward environments are rectangular and conjugate only the new
    # bra basis, not the local generator factors or its coefficients.
    ml = [randn(rng,ComplexF64,5,3) for _ in 1:2]
    mr = [randn(rng,ComplexF64,2,4) for _ in 1:2]
    q = randn(rng,ComplexF64,10,6)
    mixed = TTNSolver._mixed_subtree_operators(q,core,ml,mr)
    for k in 1:2
        @test mixed[k] ≈ q'*kron(mr[k],ml[k])*basis rtol=1e-12
    end
    U,V = randn(rng,ComplexF64,3,2),randn(rng,ComplexF64,3,4)
    op = SumOfProducts([3],[ProductTerm(1,Dict()),ProductTerm(2im,Dict()),
        ProductTerm(3,Dict(1=>L)),ProductTerm(4,Dict(1=>copy(L)))])
    leaf = TTNSolver._leaf_operators(op,1,U,V)
    @test leaf[1] ≈ U'*V
    @test leaf[2] == leaf[1]
    @test leaf[3] ≈ U'*L*V
    @test leaf[4] == leaf[3]
    @test isempty(TTNSolver._map_unique_pairs(kron,Matrix{ComplexF64}[],Matrix{ComplexF64}[]))
end

@testset "Environment contractions reuse large scratch arrays" begin
    rng = MersenneTwister(863)
    # Distinct factors prevent memoization from hiding per-term tensor copies.
    # At rank 20 the unbuffered implementation allocates > 9 MB; the workspace
    # needs < 2 MB, including independently owned small environment matrices.
    core = randn(rng,ComplexF64,20,20,20)
    Q = randn(rng,ComplexF64,400,20)
    left = [randn(rng,ComplexF64,20,20) for _ in 1:12]
    right = [randn(rng,ComplexF64,20,20) for _ in 1:12]
    run() = TTNSolver._mixed_subtree_operators(Q,core,left,right)
    result = run()
    @test (@allocated run()) < 2_000_000
    for k in (1,12)
        @test result[k] ≈ Q'*kron(right[k],left[k])*TTNSolver._basis_matrix(core) rtol=1e-12
    end
    saved = copy(result[1])
    result[2] .= 0
    @test result[1] == saved
end
