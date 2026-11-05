@testset "Reusable BUG preparation" begin
    # Reusing preparation must not leak updated bases between different steps.
    rng = MersenneTwister(93)
    A = ComplexF64[-0.4 0.2im;0.3 -0.7]
    B = ComplexF64[0 1;1 0]
    for tree in (balanced_tree(1), balanced_tree(3),
                 BinaryTree([(2,5),(0,0),(0,0),(0,0),(4,3)];root=1,leaf_order=[3,2,4]))
        dims = fill(2,length(tree.leaf_order))
        state = length(dims) == 1 ? product_state(tree,[randn(rng,ComplexF64,2)]) :
                full_state(tree,dims,rng)
        op = SumOfProducts(dims,[ProductTerm(1,Dict(1=>A)),
                                ProductTerm(0.2im,Dict(length(dims)=>B))])
        before = dense(state)
        prepared = TTNSolver._prepare_bug_step(state,op,nothing,typemax(Int))
        for h in (0.1,0.05,0.1)
            result,info = TTNSolver._prepared_bug_step(prepared,op,h;atol=0)
            @test vec(dense(result)) ≈ exp(h*dense(op))*vec(before) atol=3e-11
            @test info.truncation.tolerance_met
            # Mutation of a previous output must not corrupt the shared input.
            if isempty(result.cores)
                result.leaves[tree.root] .= 0
            else
                result.cores[tree.root] .= 0
            end
        end
        @test dense(state) == before
    end
end

@testset "Shared adaptive preparation matches independent steps" begin
    # Non-full ranks and a conserved left null vector exercise the recursive
    # reduced equations; equal and unequal halves use their own time lengths.
    tree = balanced_tree(3)
    initial = product_state(tree,[ComplexF64[1,0] for _ in 1:3])
    conserved = product_state(tree,[ones(ComplexF64,2) for _ in 1:3])
    A = ComplexF64[-1 0.3;1 -0.3]
    op = SumOfProducts([2,2,2],[ProductTerm(1,Dict(1=>A,3=>A)),
                               ProductTerm(0.2,Dict(2=>A))])
    options = (;conserved,maxrank=8,correct_invariant=false,
                krylov_tol=1e-12,krylovdim=30,krylov_maxiter=100)
    before = dense(initial)
    for (h1,h2) in ((0.04,0.04),(0.03,0.07))
        budget = 1e-10
        coarse,_ = augmented_bug_step(initial,op,h1+h2;atol=budget,options...)
        middle,_ = augmented_bug_step(initial,op,h1;atol=budget/2,options...)
        fine,info = augmented_bug_step(middle,op,h2;atol=budget/2,options...)
        trial = TTNSolver._adaptive_trial(initial,op,h1,h2,budget,options)
        @test trial.reason == :candidate
        @test dense(trial.candidate) ≈ dense(fine) atol=1e-14
        @test trial.difference ≈ ttn_distance(coarse,fine) atol=1e-14
        @test trial.half2.ranks == info.ranks
        @test dot(conserved,trial.candidate) ≈ dot(conserved,initial) atol=1e-13
    end
    @test dense(initial) == before
end
