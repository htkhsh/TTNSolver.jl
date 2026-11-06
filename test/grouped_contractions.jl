@testset "Exact grouped core actions" begin
    rng = MersenneTwister(7124)
    for output in ((3,4,2), (2,3,5))
        input = (3,4,2)
        B,L,R = [randn(rng,ComplexF64,output[j],input[j]) for j in 1:3]
        triples = [(copy(B),randn(rng,ComplexF64,size(L)),copy(R)) for _ in 1:4]
        append!(triples,[(copy(B),copy(L),randn(rng,ComplexF64,size(R))) for _ in 1:3])
        append!(triples,[(randn(rng,ComplexF64,size(B)),copy(L),copy(R)) for _ in 1:2])
        coefficients = randn(rng,ComplexF64,length(triples))
        op = SumOfProducts([2],[ProductTerm(c,Dict()) for c in coefficients])
        boundary,left,right = ([t[j] for t in triples] for j in 1:3)
        x = randn(rng,ComplexF64,input)
        original = deepcopy((boundary,left,right,x))
        reference = sum(c*kron(t[3],t[2],t[1]) for (c,t) in zip(coefficients,triples))
        # A performance contract: repeated factors must eliminate products.
        @test length(TTNSolver._group_core_terms(op,boundary,left,right)) == 3
        action = output == input ? TTNSolver._core_rhs_workspace(op,boundary,left,right,x) :
            y -> TTNSolver._parallel_core_block(op,boundary,left,right,y,output)
        first_result = action(x)
        @test vec(first_result) ≈ reference*vec(x) rtol=1e-12
        saved = copy(first_result)
        y = randn(rng,ComplexF64,input)
        @test vec(action(y)) ≈ reference*vec(y) rtol=1e-12
        @test first_result == saved
        @test (boundary,left,right,x) == original
    end
    B,L,R = [randn(rng,ComplexF64,n,n) for n in (2,3,4)]
    Bnear,Lnear = copy(B),copy(L)
    Bnear[1,1] += 1e-10; Lnear[1,1] += 1e-10
    op = SumOfProducts([2],[ProductTerm(1,Dict()),ProductTerm(-1,Dict())])
    @test length(TTNSolver._group_core_terms(op,[B,Bnear],[L,Lnear],[R,R])) == 2
    x = randn(rng,ComplexF64,2,3,4)
    action = TTNSolver._core_rhs_workspace(op,[B,Bnear],[L,Lnear],[R,R],x)
    @test vec(action(x)) ≈ (kron(R,L,B)-kron(R,Lnear,Bnear))*vec(x) atol=1e-13
    cancel = TTNSolver._core_rhs_workspace(op,[B,B],[L,L],[R,R],x)
    @test iszero(cancel(x))
    emptyop = SumOfProducts([2],ProductTerm[])
    @test isempty(TTNSolver._group_core_terms(emptyop,[],[],[]))
    zeroop = SumOfProducts([2],[ProductTerm(0,Dict())])
    @test isempty(TTNSolver._group_core_terms(zeroop,[B],[L],[R]))
end
