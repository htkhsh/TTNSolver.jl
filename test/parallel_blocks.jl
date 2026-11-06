@testset "Compact parallel BUG contractions" begin
    @test isdefined(TTNSolver,:_parallel_core_block)
    if isdefined(TTNSolver,:_parallel_core_block)
        rng = MersenneTwister(605)
        # Independent Cartesian contraction catches wrong transpose/order and
        # non-square intermediate sizes, including a rectangular parent map.
        for shape in ((2,3,4),(2,0,4),(2,3,0))
            x = randn(rng,ComplexF64,3,2,3)
            p,l,r = shape
            boundary = [randn(rng,ComplexF64,p,3) for _ in 1:3]
            left = [randn(rng,ComplexF64,l,2) for _ in 1:3]
            right = [randn(rng,ComplexF64,r,3) for _ in 1:3]
            coefficients = ComplexF64[0.3+0.7im,-0.2im,0]
            op = SumOfProducts([2],[ProductTerm(a,Dict()) for a in coefficients])
            expected = zeros(ComplexF64,shape)
            for k in 1:3, a in 1:p, b in 1:l, c in 1:r, i in 1:3, j in 1:2, m in 1:3
                expected[a,b,c] += coefficients[k]*boundary[k][a,i]*left[k][b,j]*right[k][c,m]*x[i,j,m]
            end
            inputs = deepcopy((x,boundary,left,right))
            actual = TTNSolver._parallel_core_block(op,boundary,left,right,x,shape)
            @test actual ≈ expected atol=2e-13
            @test (x,boundary,left,right) == inputs
            saved = copy(actual)
            TTNSolver._parallel_core_block(op,boundary,left,right,2x,shape)
            @test actual == saved
        end
        op = SumOfProducts([2],ProductTerm[])
        @test iszero(TTNSolver._parallel_core_block(op,Matrix{ComplexF64}[],
            Matrix{ComplexF64}[],Matrix{ComplexF64}[],ones(ComplexF64,1,2,3),(1,4,5)))
    end
end

@testset "Parallel full leaf spaces and ownership" begin
    # Full complex leaf bases may be reused, but mutating a returned state must
    # not affect either input or the next solve from shared preparation.
    rng = MersenneTwister(78)
    tree = BinaryTree([(2,5),(0,0),(0,0),(0,0),(4,3)];root=1,leaf_order=[3,2,4])
    x = full_state(tree,[2,2,2],rng)
    for node in tree.leaf_order
        x.leaves[node] = Matrix(qr(randn(rng,ComplexF64,2,2)).Q)
    end
    A = ComplexF64[-1 0.4im;0.2+0.1im -0.3]
    op = SumOfProducts([2,2,2],[ProductTerm(0.2im,Dict(1=>A,3=>A)),ProductTerm(-0.3,Dict(2=>A))])
    before = dense(x)
    prepared = TTNSolver._prepare_parallel_bug_step(x,op,nothing,16)
    first_result,info = TTNSolver._prepared_parallel_bug_step(prepared,op,0.05;atol=0)
    @test vec(dense(first_result)) ≈ exp(0.05*dense(op))*vec(before) atol=1e-12
    @test info.skipped_leaf_solves == 3
    @test all(iszero,info.omitted_block_norms)
    first_result.leaves[first(tree.leaf_order)] .= 0
    second_result,_ = TTNSolver._prepared_parallel_bug_step(prepared,op,0.05;atol=0)
    @test vec(dense(second_result)) ≈ exp(0.05*dense(op))*vec(before) atol=1e-12
    @test dense(x) == before
end
